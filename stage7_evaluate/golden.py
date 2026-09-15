# stage7_evaluate/golden.py
import json
from dataclasses import dataclass, field
from pathlib import Path

from stage1_extract.artifacts import sha256_text

from .errors import GoldenSetError

GOLDEN_SCHEMA = "golden-1.0"


@dataclass
class GoldenQuestion:
    id: str
    question: str
    expected: str # answer | abstain | block
    expected_doc: str | None = None # substring matched against corpus titles
    evidence: list[str] = field(default_factory=list) # chunk-relevance keywords
    must_facts: list[list[str]] = field(default_factory=list) # synonym groups
    aspects: list[list[str]] = field(default_factory=list) # relevance groups
    reference_answer: str | None = None
    expected_pages: list[list[int]] = field(default_factory=list) # informational


def load_golden(path) -> list[GoldenQuestion]:
    """Loads and validates. Answer-questions must carry expected_doc,
    evidence, must_facts, aspects and a reference answer — the ground truth
    the metrics consume. Fails loudly on any gap."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GoldenSetError(f"cannot read golden set {p}: {exc}") from exc
    if data.get("schema_version") != GOLDEN_SCHEMA:
        raise GoldenSetError(f"{p}: schema {data.get('schema_version')!r} != "
                             f"{GOLDEN_SCHEMA!r}")
    questions: list[GoldenQuestion] = []
    ids: set[str] = set()
    for i, q in enumerate(data.get("questions", [])):
        for k in ("id", "question", "expected"):
            if not q.get(k):
                raise GoldenSetError(f"question[{i}] missing '{k}'")
        if q["expected"] not in ("answer", "abstain", "block"):
            raise GoldenSetError(f"{q['id']}: expected must be answer|abstain|block")
        if q["id"] in ids:
            raise GoldenSetError(f"duplicate question id {q['id']!r}")
        ids.add(q["id"])
        if q["expected"] == "answer":
            for k in ("expected_doc", "reference_answer"):
                if not q.get(k):
                    raise GoldenSetError(f"{q['id']}: answer-question requires {k}")
            for k in ("evidence", "must_facts", "aspects"):
                if not q.get(k):
                    raise GoldenSetError(f"{q['id']}: answer-question requires {k}")
        questions.append(GoldenQuestion(
            id=q["id"], question=q["question"], expected=q["expected"],
            expected_doc=q.get("expected_doc"),
            evidence=list(q.get("evidence", [])),
            must_facts=[list(g) for g in q.get("must_facts", [])],
            aspects=[list(g) for g in q.get("aspects", [])],
            reference_answer=q.get("reference_answer"),
            expected_pages=[list(pg) for pg in q.get("expected_pages", [])]))
    if not questions:
        raise GoldenSetError(f"{p}: golden set is empty")
    return questions


def golden_fingerprint(path) -> str:
    return sha256_text(Path(path).read_text(encoding="utf-8"))