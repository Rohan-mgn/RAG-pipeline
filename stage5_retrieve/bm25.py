import logging
import math
import re
from collections import Counter, defaultdict

log = logging.getLogger("stage5.bm25")

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset("""
a an and are as at be but by for from has have he her hers his if in into is it
its of on or she that the their theirs them then there these they this those to
was were what when where which who whom why will with you your how can could
should would does do did about
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower())
            if t not in _STOPWORDS]


class BM25Index:
    """Okapi BM25 over the indexed chunk records. Pure stdlib; rebuilt from
    the vector store at process start, so the lexical corpus can never
    diverge from the vector corpus."""

    def __init__(self, records, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.records: list[dict] = []
        self.by_id: dict[str, dict] = {}
        self._len: list[int] = []
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        for rec in records:
            i = len(self.records)
            terms = tokenize(rec.get("text", ""))
            self.records.append(rec)
            self.by_id[rec["chunk_id"]] = rec
            self._len.append(len(terms))
            for t, f in Counter(terms).items():
                self._postings[t][i] = f
        self.N = len(self.records)
        self.avgdl = (sum(self._len) / self.N) if self.N else 0.0
        self._idf = {t: math.log(1.0 + (self.N - len(d) + 0.5) / (len(d) + 0.5))
                     for t, d in self._postings.items()}

    def search(self, query: str, k: int = 20,
               where: dict | None = None) -> list[tuple[str, float]]:
        if not self.N or k <= 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        for t in tokenize(query):
            docs = self._postings.get(t)
            if not docs:
                continue
            idf = self._idf[t]
            for i, f in docs.items():
                if where is not None:
                    rec = self.records[i]
                    if any(rec.get(fld) != val for fld, val in where.items()):
                        continue
                dl = self._len[i] or 1
                scores[i] += idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9)))
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        return [(self.records[i]["chunk_id"], s) for i, s in ranked]