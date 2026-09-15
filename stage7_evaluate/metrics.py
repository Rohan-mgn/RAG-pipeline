# stage7_evaluate/metrics.py
"""Deterministic, corpus-grounded metric implementations.

Honest labels — these are local, reproducible approximations of the Ragas
definitions: claim support is lexical (word containment), not semantic NLI;
correctness/relevance blend fact coverage with embedding similarity.
--judge local adds the LLM-judged layer on top of these numbers."""
import re
from typing import Iterable

from stage2_chunk.sentences import RegexSentenceSplitter
from stage5_retrieve.bm25 import tokenize

_SPLITTER = RegexSentenceSplitter()
_MARKER = re.compile(r"\[\d{1,3}\]")


def norm_text(s: str) -> str:
    return " ".join((s or "").casefold().split())


def strip_markers(s: str) -> str:
    return _MARKER.sub(" ", s or "")


def group_present(text: str, group: Iterable[str]) -> bool:
    t = norm_text(text)
    return any(norm_text(p) in t for p in group)


# --- retrieval metrics ------------------------------------------------------

def context_recall(retrieved_texts: list[str],
                   must_facts: list[list[str]]) -> tuple[float, int, int]:
    """Fraction of ground-truth fact groups whose supporting text was
    actually retrieved. Returns (score, hits, total)."""
    if not must_facts:
        return 1.0, 0, 0
    corpus = norm_text(" ".join(retrieved_texts))
    hit = sum(1 for g in must_facts if any(norm_text(p) in corpus for p in g))
    return hit / len(must_facts), hit, len(must_facts)


def mark_relevant(chunks: list[dict], expected_doc: str, evidence: list[str],
                  min_evidence: int = 1) -> list[bool]:
    """A retrieved chunk is relevant iff it comes from the expected document
    AND contains >= min_evidence evidence keywords. Conservative: misses
    paraphrase, never credits the wrong document."""
    doc = norm_text(expected_doc)
    ev = [norm_text(e) for e in evidence]
    out = []
    for c in chunks:
        ok_doc = doc in norm_text(c.get("title") or "")
        text = norm_text(c.get("text") or "")
        hits = sum(1 for e in ev if e in text)
        out.append(ok_doc and hits >= min_evidence)
    return out


def context_precision(chunks: list[dict], relevant: list[bool]) -> float:
    """Ragas context precision: mean of precision@rank over the positions of
    relevant chunks. Rewards relevant chunks ranked at the top."""
    positions = [i for i, r in enumerate(relevant) if r]
    if not positions:
        return 0.0
    total, cum = 0.0, 0
    for i, r in enumerate(relevant):
        if r:
            cum += 1
            total += cum / (i + 1)
    return total / len(positions)


# --- generation metrics -----------------------------------------------------

def split_claims(answer: str) -> list[str]:
    text = strip_markers(answer).strip()
    return [s for s in _SPLITTER(text) if len(s.split()) >= 3]


def claim_supported(claim: str, context_texts: list[str],
                    threshold: float = 0.5) -> bool:
    words = set(tokenize(claim))
    if not words:
        return True
    ctx = set(tokenize(" ".join(context_texts)))
    return len(words & ctx) / len(words) >= threshold


def faithfulness(answer: str, context_texts: list[str],
                 threshold: float = 0.5) -> tuple[float | None, int, int]:
    """Fraction of answer claims lexically supported by the context.
    (None, 0, 0) for empty/abstention answers — not applicable, not zero."""
    claims = split_claims(answer)
    if not claims:
        return None, 0, 0
    sup = sum(1 for c in claims if claim_supported(c, context_texts, threshold))
    return sup / len(claims), sup, len(claims)


def fact_coverage(answer: str,
                  fact_groups: list[list[str]]) -> tuple[float, int, list]:
    if not fact_groups:
        return 1.0, 0, []
    missing = [list(g) for g in fact_groups if not group_present(answer, g)]
    hit = len(fact_groups) - len(missing)
    return hit / len(fact_groups), hit, missing


def aspect_relevance(answer: str,
                     aspect_groups: list[list[str]]) -> tuple[float, list]:
    if not aspect_groups:
        return 1.0, []
    missing = [list(g) for g in aspect_groups if not group_present(answer, g)]
    return (len(aspect_groups) - len(missing)) / len(aspect_groups), missing


def cosine(a, b) -> float:
    import numpy as np
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(va @ vb) / (na * nb)