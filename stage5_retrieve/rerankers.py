import logging

log = logging.getLogger("stage5.rerankers")

DEFAULT_RERANK_MODEL = "ms-marco-TinyBERT-L-2-v2"


class NoopReranker:
    """Keeps fused order; scores stay None so abstention leans on the dense
    floor. Active whenever flashrank is unavailable — loud in logs."""
    name = "none"

    def rerank(self, query, candidates):
        return [(c, None) for c in candidates]


class FlashRankReranker:
    """Cross-encoder reranking via FlashRank (ONNX, CPU-friendly). TinyBERT-L-2
    is ~4 MB and reranks a dozen candidates in tens of milliseconds."""
    name = f"flashrank:{DEFAULT_RERANK_MODEL}"

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL,
                 cache_dir: str = "models/flashrank", max_chars: int = 2000) -> None:
        try:
            import truststore # make requests verify against the
            truststore.inject_into_ssl() # OS cert store — the same source
        except ImportError: # docling's downloads already
            pass # succeed with on this machine
        from flashrank import Ranker, RerankRequest
        self._RerankRequest = RerankRequest
        self._ranker = Ranker(model_name=model_name, cache_dir=cache_dir)
        self.max_chars = max_chars

    def rerank(self, query, candidates):
        passages = []
        for c in candidates:
            text = c.get("text", "")
            if c.get("heading_path"):
                text = f"{c['heading_path']}\n{text}"
            passages.append({"id": c["chunk_id"], "text": text[:self.max_chars]})
        results = self._ranker.rerank(
            self._RerankRequest(query=query, passages=passages)) or []
        by_id = {str(r.get("id")): float(r.get("score", 0.0)) for r in results}
        scored = [(c, by_id.get(c["chunk_id"])) for c in candidates]
        scored.sort(key=lambda x: (x[1] is None, -(x[1] or 0.0)))
        return scored


def make_reranker(mode: str = "auto", model_name: str = DEFAULT_RERANK_MODEL,
                  cache_dir: str = "models/flashrank"):
    if mode == "none":
        return NoopReranker()
    try:
        r = FlashRankReranker(model_name, cache_dir)
        log.info("reranker: %s", r.name)
        return r
    except Exception as exc:
        if mode == "flashrank":
            raise
        log.warning("flashrank unavailable (%s) — rerank leg disabled "
                    "(pip install flashrank)", exc)
        return NoopReranker()