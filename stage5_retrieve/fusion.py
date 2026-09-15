from collections import defaultdict


def rrf_fuse(rankings, k: int = 60) -> dict[str, float]:
    """Reciprocal-rank fusion. rankings: ordered id lists, best first.
    Returns id -> fused score. Cheap, parameter-light, no trained weights."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] += 1.0 / (k + rank)
    return dict(scores)