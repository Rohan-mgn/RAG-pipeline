import logging
import re
import time
from dataclasses import dataclass, field

from .errors import ConfigError
from .intents import Intent, route_intent
from .patterns import PatternHit, downgrade, scan
from .sanitize import QuerySanitizer

log = logging.getLogger("stage4.guard")

BLOCKED_REPLY = ("I can't process that request — it looks like an attempt to "
                 "override my instructions. Please ask a question about the "
                 "documents instead.")

_INTERROGATIVE = re.compile(
    r"^\s*(?:how|what|why|when|where|which|who|whom|whose|can|could|would|"
    r"should|does|do|did|is|are|was|were|will|shall|may|might)\b", re.IGNORECASE)

_CHAT_KINDS = ("greeting", "thanks", "farewell", "identity", "capability")


@dataclass
class GuardConfig:
    max_chars: int = 2000
    ml_mode: str = "flag"               # off | flag | enforce
    ml_flag_threshold: float = 0.5
    ml_block_threshold: float = 0.9
    pattern_flag_limit: int = 2         # N flag-severity hits -> block
    interrogative_downgrade: bool = True

    def validate(self) -> None:
        if self.ml_mode not in ("off", "flag", "enforce"):
            raise ConfigError(f"ml_mode must be off|flag|enforce, "
                              f"got {self.ml_mode!r}")
        if not (0.0 < self.ml_flag_threshold <= self.ml_block_threshold <= 1.0):
            raise ConfigError("need 0 < ml_flag_threshold <= ml_block_threshold <= 1")


@dataclass
class QueryPlan:
    action: str                    # search | chat | blocked | refused
    cleaned_query: str
    intent: str
    intent_reply: str | None
    injection_blocked: bool = False
    injection_flags: list[str] = field(default_factory=list)
    ml_score: float | None = None
    audit: list[dict] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def searchable(self) -> bool:
        return self.action == "search"


class GuardPipeline:
    """sanitize -> pattern scan (always, microseconds) -> optional ML second
    opinion (only if not already blocked; skipped for chat intents) -> intent
    routing. Every step timed and recorded in the audit trail."""

    def __init__(self, config: GuardConfig | None = None, classifier=None) -> None:
        self.config = config or GuardConfig()
        self.config.validate()
        self.sanitizer = QuerySanitizer(self.config.max_chars)
        self.classifier = classifier       # None -> lexical-only

    def run(self, raw_query: str) -> QueryPlan:
        audit: list[dict] = []
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        san = self.sanitizer.sanitize(raw_query or "")
        timings["sanitize"] = round((time.perf_counter() - t0) * 1000, 2)
        audit.append({"step": "sanitize", "flags": san.flags,
                      "removed_chars": san.removed_chars})
        q = san.cleaned

        if not q:
            return QueryPlan("refused", q, "empty",
                             "Your query was empty after cleaning — "
                             "please rephrase.", audit=audit, timings_ms=timings)

        # -- pattern scan ----------------------------------------------------
        t0 = time.perf_counter()
        hits = scan(q)
        interrogative = bool(_INTERROGATIVE.match(q))
        effective: list[PatternHit] = []
        for h in hits:
            if (h.severity == "block" and interrogative
                    and self.config.interrogative_downgrade):
                h = downgrade(h)           # question ABOUT injection, not one
            effective.append(h)
        timings["patterns"] = round((time.perf_counter() - t0) * 1000, 2)
        audit.append({"step": "patterns", "interrogative": interrogative,
                      "hits": [{"name": h.name, "severity": h.severity,
                                "snippet": h.snippet} for h in effective]})

        blocks = [h for h in effective if h.severity == "block"]
        flags = [h for h in effective if h.severity == "flag"]
        blocked = bool(blocks) or len(flags) >= self.config.pattern_flag_limit

        # -- ML second opinion (skipped when already blocked: saves latency) --
        ml_score: float | None = None
        ml_opinion = "none"
        if not blocked and self.config.ml_mode != "off" and self.classifier:
            t0 = time.perf_counter()
            ml_score = self.classifier.score(q)
            timings["ml"] = round((time.perf_counter() - t0) * 1000, 2)
            if ml_score is not None:
                if ml_score >= self.config.ml_block_threshold:
                    ml_opinion = "injection"
                    if self.config.ml_mode == "enforce":
                        blocked = True
                elif ml_score >= self.config.ml_flag_threshold:
                    ml_opinion = "suspicious"
                else:
                    ml_opinion = "safe"
            audit.append({"step": "ml", "score": ml_score,
                          "opinion": ml_opinion, "mode": self.config.ml_mode})

        injection_flags = [h.name for h in flags]
        if ml_opinion == "suspicious" or (ml_opinion == "injection" and not blocked):
            injection_flags.append(f"ml:{ml_opinion}:{ml_score:.2f}")

        if blocked:
            reasons = [h.name for h in blocks]
            if len(flags) >= self.config.pattern_flag_limit:
                reasons.append(f"flag-limit({len(flags)})")
            if ml_opinion == "injection" and self.config.ml_mode == "enforce":
                reasons.append(f"ml:{ml_score:.2f}")
            audit.append({"step": "decision", "blocked": True, "reasons": reasons})
            log.info("blocked query (%s): %r", ", ".join(reasons), q[:60])
            return QueryPlan("blocked", q, "injection-attempt", BLOCKED_REPLY,
                             injection_blocked=True,
                             injection_flags=injection_flags, ml_score=ml_score,
                             audit=audit, timings_ms=timings)

        # -- intent routing ---------------------------------------------------
        t0 = time.perf_counter()
        intent: Intent = route_intent(q)
        timings["intent"] = round((time.perf_counter() - t0) * 1000, 2)
        audit.append({"step": "intent", "kind": intent.kind})

        if intent.kind in _CHAT_KINDS:
            return QueryPlan("chat", q, intent.kind, intent.reply,
                             injection_flags=injection_flags, ml_score=ml_score,
                             audit=audit, timings_ms=timings)
        if intent.kind == "low-coherence":
            return QueryPlan("refused", q, intent.kind, intent.reply,
                             injection_flags=injection_flags, ml_score=ml_score,
                             audit=audit, timings_ms=timings)
        return QueryPlan("search", intent.remainder or q, "search", None,
                         injection_flags=injection_flags, ml_score=ml_score,
                         audit=audit, timings_ms=timings)