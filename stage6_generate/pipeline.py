# stage6_generate/pipeline.py
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from stage4_guard.guard import GuardPipeline
from stage5_retrieve.retriever import ABSTAIN_REPLY, HybridRetriever, RetrievalResult

from .citations import verify_citations
from .errors import ConfigError
from .packing import PackedContext, pack_context
from .prompt import SYSTEM_TEMPLATE, build_user_prompt
from .session import ChatSession

from .citations import ABSTENTION_PHRASES   # add to imports

def _uncited(answer_text: str) -> bool:
    t = (answer_text or "").lower()
    if any(p in t for p in ABSTENTION_PHRASES):
        return False                          # abstaining — citations not expected
    return _MARKER_ONLY.search(answer_text or "") is None

log = logging.getLogger("stage6.pipeline")
PIPELINE_VERSION = "stage6-1.1.0"

_MARKER_ONLY = re.compile(r"\[\d{1,3}\]")
_LEAD_ANSWER = re.compile(r"^\s*(?:answer|assistant)\s*:\s*", re.IGNORECASE)

NUDGE = ("\n\nImportant: your previous reply was not a real answer. Write 2 to 5 "
         "complete sentences in your own words, and place a citation marker like "
         "[1] immediately after each sentence that uses the context. Never reply "
         "with citation markers alone.")


def _is_degenerate(text: str) -> bool:
    """Marker-only or near-empty answers — an observed failure mode of small
    local models. Fewer than 4 substantive words once markers are removed."""
    stripped = _MARKER_ONLY.sub(" ", text or "")
    return bool((text or "").strip()) and len(stripped.split()) < 4


def _strip_answer_prefix(text: str) -> str:
    """Small models sometimes echo the 'Answer:' cue back."""
    return _LEAD_ANSWER.sub("", text or "").strip()


@dataclass
class GenerationConfig:
    max_context_tokens: int = 3500
    max_history_tokens: int = 800
    max_output_tokens: int = 600
    top_n: int = 5

    def validate(self) -> None:
        c = self
        if c.max_context_tokens < 512:
            raise ConfigError("max_context_tokens must be >= 512")
        if c.max_history_tokens < 0:
            raise ConfigError("max_history_tokens must be >= 0")
        if c.max_output_tokens < 64:
            raise ConfigError("max_output_tokens must be >= 64")
        if c.top_n < 1:
            raise ConfigError("top_n must be >= 1")


@dataclass
class Answer:
    question: str
    action: str            # answered | abstained-retrieval | abstained-model |
                           # blocked | chat | refused
    reply: str             # final user-facing text (citation-cleaned)
    answer_text: str = ""
    citations: list[dict] = field(default_factory=list)
    citation_report: dict = field(default_factory=dict)
    generator: str = ""
    usage: dict = field(default_factory=dict)
    context_stats: dict = field(default_factory=dict)
    retrieval_stats: dict = field(default_factory=dict)
    guard: dict = field(default_factory=dict)
    session_id: str | None = None
    timings_ms: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    generation_attempts: int = 1
    pipeline_version: str = PIPELINE_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


class GenerationPipeline:
    """guard -> hybrid retrieval -> abstention -> rank-priority packing ->
    prompt assembly (bounded memory) -> generation (degenerate-safe, one
    retry) -> machine-checked citations -> session update. The LLM is never
    called unless retrieval survived the abstention gate."""

    def __init__(self, retriever: HybridRetriever, generator,
                 guard: GuardPipeline | None = None,
                 session: ChatSession | None = None,
                 config: GenerationConfig | None = None) -> None:
        self.retriever = retriever
        self.generator = generator
        self.guard = guard or GuardPipeline()
        self.session = session
        self.config = config or GenerationConfig()
        self.config.validate()

    def _record(self, question: str, reply: str) -> None:
        if self.session:
            self.session.add("user", question)
            self.session.add("assistant", reply)

    def ask(self, raw_question: str,
            on_delta: Callable[[str], None] | None = None,
            where: dict | None = None) -> Answer:
        timings: dict[str, float] = {}
        warnings: list[str] = []
        sid = self.session.session_id if self.session else None

        t0 = time.perf_counter()
        plan = self.guard.run(raw_question or "")
        timings["guard"] = round((time.perf_counter() - t0) * 1000, 1)
        guard_stats = {"action": plan.action, "intent": plan.intent,
                       "flags": plan.injection_flags}
        if not plan.searchable:
            return Answer(question=raw_question or "", action=plan.action,
                          reply=plan.intent_reply or "", guard=guard_stats,
                          session_id=sid, timings_ms=timings)

        t0 = time.perf_counter()
        result: RetrievalResult = self.retriever.run(
            plan.cleaned_query, top_n=self.config.top_n, where=where)
        timings["retrieve"] = round((time.perf_counter() - t0) * 1000, 1)
        retrieval_stats = {
            "action": result.action, "returned": len(result.chunks),
            "leg_counts": result.leg_counts, "timings_ms": result.timings_ms,
            "top_dense": max((c.dense_score or 0.0 for c in result.chunks),
                             default=None),
            "top_rerank": result.chunks[0].rerank_score if result.chunks else None,
            "abstention_reasons": result.abstention_reasons}
        warnings.extend(result.warnings)
        if result.abstained:
            log.info("abstained at retrieval: %s", result.abstention_reasons)
            reply = result.suggested_reply or ABSTAIN_REPLY
            self._record(plan.cleaned_query, reply)
            return Answer(question=plan.cleaned_query, action="abstained-retrieval",
                          reply=reply, guard=guard_stats,
                          retrieval_stats=retrieval_stats, session_id=sid,
                          timings_ms=timings, warnings=warnings)

        t0 = time.perf_counter()
        packed: PackedContext = pack_context(result.chunks,
                                             self.config.max_context_tokens)
        timings["pack"] = round((time.perf_counter() - t0) * 1000, 1)
        context_stats = {"blocks": len(packed.blocks),
                         "context_tokens": packed.used_tokens,
                         "dropped_ranks": packed.dropped_ranks,
                         "chunk_ids": [b.chunk_id for b in packed.blocks]}
        if packed.dropped_ranks:
            warnings.append(f"context budget dropped ranks {packed.dropped_ranks}")

        history = self.session.render() if self.session else None
        base_prompt = build_user_prompt(plan.cleaned_query, packed, history)
        user_prompt = base_prompt

        attempts = 0
        gen_seconds = 0.0
        usage: dict = {}
        answer_text = ""
        while True:
            attempts += 1
            t0 = time.perf_counter()
            if on_delta is not None:
                if attempts > 1:
                    on_delta("\n\n")          # visual break before the retry
                parts: list[str] = []
                for delta in self.generator.stream(SYSTEM_TEMPLATE, user_prompt,
                                                   self.config.max_output_tokens):
                    parts.append(delta)
                    on_delta(delta)
                answer_text = _strip_answer_prefix("".join(parts))
                usage = dict(getattr(self.generator, "last_usage", {}) or {})
            else:
                raw, usage = self.generator.generate(
                    SYSTEM_TEMPLATE, user_prompt, self.config.max_output_tokens)
                answer_text = _strip_answer_prefix(raw)
            gen_seconds += time.perf_counter() - t0
            timings["generate"] = round(gen_seconds * 1000, 1)
            if attempts >= 2 or not _is_degenerate(answer_text):
                break
            log.warning("degenerate answer (markers-only / near-empty) — "
                        "retrying once with amplified instruction")
            user_prompt = base_prompt + NUDGE
        if attempts > 1:
            warnings.append("first generation attempt was degenerate — retried")

        t0 = time.perf_counter()
        report, clean_text = verify_citations(answer_text, packed)
        timings["verify"] = round((time.perf_counter() - t0) * 1000, 1)
        warnings.extend(report.warnings)

        action = "abstained-model" if report.abstained else "answered"
        self._record(plan.cleaned_query, clean_text)

        log.info("answered in %.1fs: %d citations, %d blocks, %d tok context, "
                 "%d attempt(s)", gen_seconds, len(report.citations),
                 len(packed.blocks), packed.used_tokens, attempts)
        return Answer(question=plan.cleaned_query, action=action, reply=clean_text,
                      answer_text=answer_text,
                      citations=[c.to_dict() for c in report.citations],
                      citation_report=report.to_dict(),
                      generator=self.generator.name, usage=usage,
                      context_stats=context_stats, retrieval_stats=retrieval_stats,
                      guard=guard_stats, session_id=sid, timings_ms=timings,
                      warnings=warnings, generation_attempts=attempts)