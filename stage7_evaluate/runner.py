# stage7_evaluate/runner.py
import logging
from dataclasses import asdict, dataclass, field

from stage4_guard.guard import GuardConfig, GuardPipeline
from stage5_retrieve.retriever import HybridRetriever
from stage6_generate.pipeline import GenerationConfig, GenerationPipeline

from . import metrics
from .errors import ConfigError
from .judge import judge_question

log = logging.getLogger("stage7.runner")
PIPELINE_VERSION = "stage7-1.0.0"
ABSTAIN_ACTIONS = ("abstained-retrieval", "abstained-model")
GUARD_REFUSALS = ("blocked", "refused")


@dataclass
class EvalConfig:
    top_n: int = 5
    claim_word_threshold: float = 0.5 # lexical support per claim
    pass_fact_coverage: float = 0.5 # per-question pass bar
    pass_faithfulness: float = 0.6
    min_evidence: int = 1

    def validate(self) -> None:
        c = self
        if c.top_n < 1:
            raise ConfigError("top_n must be >= 1")
        for k in ("claim_word_threshold", "pass_fact_coverage", "pass_faithfulness"):
            if not (0.0 <= getattr(c, k) <= 1.0):
                raise ConfigError(f"{k} must be within 0..1")
        if c.min_evidence < 1:
            raise ConfigError("min_evidence must be >= 1")


@dataclass
class QuestionResult:
    id: str
    question: str
    expected: str
    action: str = ""
    answer: str = ""
    skipped: str | None = None
    passed: bool | None = None
    metrics: dict = field(default_factory=dict)
    citations: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class EvalRunner:
    """Runs the golden set through the FULL ask() path (guard -> retrieval ->
    abstention -> generation -> citation verification) and scores each
    question. Retrieval is executed once more by the runner itself for
    per-chunk evidence — deterministic, ~100 ms per question."""

    def __init__(self, retriever: HybridRetriever, generator,
                 guard: GuardPipeline | None = None,
                 config: EvalConfig | None = None,
                 judge_generator=None) -> None:
        self.config = config or EvalConfig()
        self.config.validate()
        self.retriever = retriever
        self.guard = guard or GuardPipeline(GuardConfig())
        self.judge = judge_generator
        self.pipeline = GenerationPipeline(
            retriever, generator, guard=self.guard,
            config=GenerationConfig(top_n=self.config.top_n))
        self.titles: list[str] = []
        for rec in retriever.store.iter_records():
            t = rec.get("title")
            if t and t not in self.titles:
                self.titles.append(t)

    def _doc_in_corpus(self, expected_doc: str | None) -> bool:
        if not expected_doc:
            return True
        d = metrics.norm_text(expected_doc)
        return any(d in metrics.norm_text(t) for t in self.titles)

    def run(self, questions, only: str | None = None,
            dry_run: bool = False) -> list[QuestionResult]:
        results: list[QuestionResult] = []
        for gq in questions:
            if only and gq.expected != only:
                continue
            try:
                res = self._eval_question(gq, dry_run)
            except Exception as exc:
                log.exception("eval question %s failed", gq.id)
                res = QuestionResult(gq.id, gq.question, gq.expected,
                                     error=f"{type(exc).__name__}: {exc}",
                                     passed=False)
            results.append(res)
            log.info("%s: expected=%s action=%s passed=%s%s", gq.id, gq.expected,
                     res.action, res.passed,
                     f" SKIPPED ({res.skipped})" if res.skipped else "")
        return results

    def _eval_question(self, gq, dry_run: bool) -> QuestionResult:
        res = QuestionResult(gq.id, gq.question, gq.expected)

        if gq.expected == "answer" and not self._doc_in_corpus(gq.expected_doc):
            res.skipped = (f"expected document {gq.expected_doc!r} not in corpus "
                           f"(indexed: {self.titles})")
            log.warning("SKIP %s: %s", gq.id, res.skipped)
            return res

        plan = self.guard.run(gq.question)
        res.timings_ms.update(plan.timings_ms)

        if gq.expected == "block":
            res.action = plan.action
            res.passed = plan.action in GUARD_REFUSALS
            if not res.passed:
                res.warnings.append(f"guard let it through as '{plan.action}'")
            return res

        if not plan.searchable:
            res.action = plan.action
            # a guard refusal of an abstain-expected query is acceptable
            res.passed = (gq.expected == "abstain" and plan.action == "refused")
            return res

        rr = self.retriever.run(plan.cleaned_query, top_n=self.config.top_n)
        retrieved = [asdict(c) for c in rr.chunks]
        texts = [((c.get("heading_path") or "") + "\n" + c.get("text", "")) for c in retrieved]
        res.action = rr.action
        res.warnings.extend(rr.warnings)

        recall = metrics.context_recall(texts, gq.must_facts)
        recall_missing = [" | ".join(g) for g in gq.must_facts
                          if not metrics.group_present(" ".join(texts), g)]
        if recall_missing:
            res.metrics["recall_missing"] = recall_missing
        relevant = metrics.mark_relevant(retrieved, gq.expected_doc or "",
                                         gq.evidence, self.config.min_evidence)
        res.metrics["context_recall"] = recall[0]
        res.metrics["context_precision"] = metrics.context_precision(retrieved,
                                                                     relevant)
        res.metrics["retrieved"] = len(retrieved)

        if dry_run:
            if gq.expected == "abstain":
                res.passed = rr.abstained
            else:
                res.passed = (rr.action == "retrieve"
                              and recall[0] >= self.config.pass_fact_coverage)
            return res

        answer = self.pipeline.ask(gq.question)
        res.answer = answer.reply[:2000]
        res.action = answer.action
        res.timings_ms.update(answer.timings_ms)
        res.citations = list(answer.citations)
        res.warnings.extend(answer.warnings)

        if gq.expected == "abstain":
            res.passed = answer.action in ABSTAIN_ACTIONS
            if not res.passed:
                f = metrics.faithfulness(answer.reply, texts,
                                         self.config.claim_word_threshold)
                res.metrics["negative_faithfulness"] = f[0]
                res.warnings.append("answered a question that should be refused")
            return res

        cov = metrics.fact_coverage(answer.reply, gq.must_facts)
        relv = metrics.aspect_relevance(answer.reply, gq.aspects)
        faith = metrics.faithfulness(answer.reply, texts,
                                     self.config.claim_word_threshold)
        res.metrics["fact_coverage"] = cov[0]
        res.metrics["answer_relevance"] = relv[0]
        res.metrics["faithfulness"] = faith[0]
        if cov[2]:
            res.metrics["missing_facts"] = [" | ".join(g) for g in cov[2]]
        if relv[1]:
            res.metrics["missing_aspects"] = [" | ".join(g) for g in relv[1]]

        if answer.action == "answered":
            try:
                a_vec = self.retriever.embedder.embed_query(answer.reply)
                r_vec = self.retriever.embedder.embed_query(
                    gq.reference_answer or "")
                res.metrics["semantic_similarity"] = round(
                    metrics.cosine(a_vec, r_vec), 3)
            except Exception as exc:
                res.warnings.append(f"semantic similarity skipped: {exc}")
            cited = bool(answer.citations) and not answer.citation_report.get(
                "uncited")
            res.metrics["cited"] = bool(cited)
            faith_ok = faith[0] is None or faith[0] >= self.config.pass_faithfulness
            res.passed = (cov[0] >= self.config.pass_fact_coverage
                          and faith_ok and cited)
        else:
            res.passed = False
            res.warnings.append(
                "abstained on an answerable question: "
                f"{answer.retrieval_stats.get('abstention_reasons')}")

        if self.judge and answer.action == "answered":
            jd = judge_question(self.judge, gq.question, gq.reference_answer,
                                "\n\n".join(texts), answer.reply)
            if jd:
                for k, v in jd.items():
                    res.metrics[f"judge_{k}"] = v
        return res