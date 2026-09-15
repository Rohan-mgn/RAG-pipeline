import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from stage3_embed.embedders import Embedder, HashEmbedder
from stage3_embed.errors import EmbedderMismatchError
from stage3_embed.pipeline import MANIFEST_FILE
from stage3_embed.store import NumpyVectorStore

from .bm25 import BM25Index
from .errors import ConfigError
from .fusion import rrf_fuse
from .rerankers import NoopReranker, make_reranker

log = logging.getLogger("stage5.retriever")
PIPELINE_VERSION = "stage5-1.0.0"
ABSTAIN_REPLY = ("I don't have enough information in the indexed documents "
                 "to answer that question confidently.")


@dataclass
class AbstentionPolicy:
    dense_floor: float = 0.65     # calibrated: on-topic ~0.74+, misses ~0.55
    rerank_floor: float = 0.30
    margin_floor: float = 0.05    # top1-top2 rerank gap considered "clear"
    margin_window: float = 0.20   # only check margin when top score is weakish

    def validate(self) -> None:
        for name in ("dense_floor", "rerank_floor"):
            if not (0.0 <= getattr(self, name) <= 1.0):
                raise ConfigError(f"{name} must be within 0..1")
        if self.margin_floor < 0 or self.margin_window < 0:
            raise ConfigError("margins must be >= 0")


@dataclass
class RetrieverConfig:
    k_dense: int = 20
    k_sparse: int = 20
    fused_k: int = 12
    top_n: int = 5
    rrf_k: int = 60
    use_dense: bool = True
    use_sparse: bool = True
    use_rerank: bool = True
    abstain: bool = True
    overfetch: int = 4           # multiplier when range filters post-filter

    def validate(self) -> None:
        c = self
        if min(c.k_dense, c.k_sparse, c.fused_k, c.top_n, c.rrf_k, c.overfetch) < 1:
            raise ConfigError("k values and overfetch must be >= 1")
        if c.fused_k < c.top_n:
            raise ConfigError("fused_k must be >= top_n")


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    heading_path: str
    kind: str
    text: str
    page_start: int
    page_end: int
    token_estimate: int
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    fused_rank: int | None = None
    rerank_score: float | None = None
    final_rank: int = 0

    @property
    def citation(self) -> str:
        pages = (f"p.{self.page_start}" if self.page_start == self.page_end
                 else f"pp.{self.page_start}-{self.page_end}")
        section = f", {self.heading_path}" if self.heading_path else ""
        return f"{self.title or 'unknown source'}, {pages}{section}"


@dataclass
class RetrievalResult:
    query: str
    action: str                              # retrieve | abstain
    chunks: list[RetrievedChunk]
    abstained: bool = False
    abstention_reasons: list[str] = field(default_factory=list)
    suggested_reply: str | None = None
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    leg_counts: dict[str, int] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    pipeline_version: str = PIPELINE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _evaluate_confidence(chunks, policy, rerank_active, dense_active=True):
    """Explainable abstention: every reason is a named, numeric fact.
    Cross-encoder is the strongest signal (below floor -> refuse). Dense
    floor governs only when the reranker can't vouch for the hit."""
    if not chunks:
        return ["no candidates retrieved"]
    reasons: list[str] = []
    top_rerank = chunks[0].rerank_score if rerank_active else None
    if top_rerank is not None and top_rerank < policy.rerank_floor:
        reasons.append(f"top rerank score {top_rerank:.3f} < "
                       f"floor {policy.rerank_floor:.2f}")
    weak_rerank = top_rerank is None or top_rerank < policy.rerank_floor
    if dense_active:
        dense = [c.dense_score for c in chunks if c.dense_score is not None]
        top_dense = max(dense) if dense else 0.0
        if top_dense < policy.dense_floor and weak_rerank:
            reasons.append(f"top dense score {top_dense:.3f} < "
                           f"floor {policy.dense_floor:.2f}")
    return reasons


class HybridRetriever:
    def __init__(self, store: NumpyVectorStore, embedder: Embedder,
                 reranker=None, policy: AbstentionPolicy | None = None,
                 config: RetrieverConfig | None = None) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker if reranker is not None else make_reranker("auto")
        self.policy = policy or AbstentionPolicy()
        self.config = config or RetrieverConfig()
        self.policy.validate()
        self.config.validate()
        t0 = time.perf_counter()
        self.bm25 = BM25Index(store.iter_records())
        self.bm25_build_ms = round((time.perf_counter() - t0) * 1000, 1)
        log.info("bm25 index built over %d chunks in %.0f ms",
                 self.bm25.N, self.bm25_build_ms)

    @classmethod
    def load(cls, index_dir, embedder: Embedder | None = None, reranker=None,
             policy: AbstentionPolicy | None = None,
             config: RetrieverConfig | None = None) -> "HybridRetriever":
        out = Path(index_dir)
        store = NumpyVectorStore(out)
        manifest: dict = {}
        mpath = out / MANIFEST_FILE
        if mpath.exists():
            try:
                manifest = json.loads(mpath.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("corrupt manifest at %s — proceeding without "
                            "staleness check", mpath)
        if embedder is None:
            embedder = _resolve_embedder(manifest.get("embedder_signature"))
        sig = manifest.get("embedder_signature")
        if sig and embedder.signature != sig:
            raise EmbedderMismatchError(
                f"index at {out} was built with {sig!r}; this embedder is "
                f"{embedder.signature!r}. Use the matching embedder or rebuild.")
        declared = (manifest.get("totals") or {}).get("chunks")
        if declared is not None and declared != store.count():
            log.warning("STALE INDEX: manifest declares %d vectors, store holds "
                        "%d — re-run stage3 to reconcile", declared, store.count())
        return cls(store, embedder, reranker=reranker, policy=policy, config=config)

    def run(self, query: str, top_n: int | None = None,
            where: dict | None = None,
            page_range: tuple[int, int] | None = None) -> RetrievalResult:
        cfg = self.config
        top_n = top_n or cfg.top_n
        timings: dict[str, float] = {"bm25_build_ms": self.bm25_build_ms}
        warnings: list[str] = []
        dense_scores: dict[str, float] = {}
        sparse_scores: dict[str, float] = {}

        def page_ok(rec: dict) -> bool:
            if not page_range:
                return True
            lo, hi = page_range
            return rec.get("page_end", 0) >= lo and rec.get("page_start", 0) <= hi

        if cfg.use_dense:
            t0 = time.perf_counter()
            qvec = self.embedder.embed_query(query)
            k = cfg.k_dense * (cfg.overfetch if page_range else 1)
            hits = self.store.search(qvec, k=k, where=where)
            dense_scores = {h["chunk_id"]: h.score for h in hits
                            if page_ok(h.record)}
            timings["dense_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        else:
            warnings.append("dense leg disabled")

        if cfg.use_sparse:
            t0 = time.perf_counter()
            k = cfg.k_sparse * (cfg.overfetch if page_range else 1)
            sparse_scores = {cid: s for cid, s in
                             self.bm25.search(query, k=k, where=where)
                             if page_ok(self.bm25.by_id.get(cid, {}))}
            timings["sparse_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        else:
            warnings.append("sparse leg disabled")

        t0 = time.perf_counter()
        rankings = []
        if dense_scores:
            rankings.append(sorted(dense_scores, key=dense_scores.get, reverse=True))
        if sparse_scores:
            rankings.append(sorted(sparse_scores, key=sparse_scores.get, reverse=True))
        fused = rrf_fuse(rankings, cfg.rrf_k)
        top_ids = [cid for cid, _ in sorted(fused.items(),
                                            key=lambda x: (-x[1], x[0]))][:cfg.fused_k]
        timings["fuse_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        chunks: list[RetrievedChunk] = []
        for rank, cid in enumerate(top_ids, 1):
            rec = self.bm25.by_id.get(cid)
            if rec is None:
                warnings.append(f"fused candidate {cid[:12]} missing from lexical "
                                "corpus — skipped (stale vector?)")
                continue
            chunks.append(RetrievedChunk(
                chunk_id=rec["chunk_id"], doc_id=rec.get("doc_id", ""),
                title=rec.get("title", ""), heading_path=rec.get("heading_path", ""),
                kind=rec.get("kind", "text"), text=rec.get("text", ""),
                page_start=int(rec.get("page_start", 0)),
                page_end=int(rec.get("page_end", 0)),
                token_estimate=int(rec.get("token_estimate", 0)),
                dense_score=dense_scores.get(cid),
                sparse_score=sparse_scores.get(cid),
                fused_score=fused.get(cid), fused_rank=rank))

        rerank_active = False
        if cfg.use_rerank and chunks:
            t0 = time.perf_counter()
            pairs = self.reranker.rerank(
                query, [self.bm25.by_id[c.chunk_id] for c in chunks])
            by_id = {rec["chunk_id"]: score for rec, score in pairs}
            for c in chunks:
                c.rerank_score = by_id.get(c.chunk_id)
            chunks.sort(key=lambda c: (c.rerank_score is None,
                                       -(c.rerank_score or 0.0)))
            rerank_active = any(c.rerank_score is not None for c in chunks)
            timings["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        elif not cfg.use_rerank:
            warnings.append("rerank leg disabled")

        chunks = chunks[:top_n]
        for i, c in enumerate(chunks, 1):
            c.final_rank = i

        reasons: list[str] = []
        if cfg.abstain:
            reasons = _evaluate_confidence(chunks, self.policy, rerank_active,
                                           dense_active=cfg.use_dense)
            if rerank_active and len(chunks) > 1:
                s = [c.rerank_score for c in chunks if c.rerank_score is not None]
                if (len(s) > 1
                        and s[0] < self.policy.rerank_floor + self.policy.margin_window
                        and (s[0] - s[1]) < self.policy.margin_floor):
                    warnings.append(f"no clear winner: rerank margin "
                                    f"{s[0] - s[1]:.3f}")

        abstained = bool(reasons)
        return RetrievalResult(
            query=query, action="abstain" if abstained else "retrieve",
            chunks=chunks, abstained=abstained, abstention_reasons=reasons,
            suggested_reply=ABSTAIN_REPLY if abstained else None,
            warnings=warnings, timings_ms=timings,
            leg_counts={"dense": len(dense_scores), "sparse": len(sparse_scores),
                        "fused": len(fused), "reranked": int(rerank_active),
                        "returned": len(chunks)},
            policy=asdict(self.policy))


def _resolve_embedder(signature: str | None) -> Embedder:
    if signature and signature.startswith("hash-test:"):
        try:
            dim = int(signature.split(":", 1)[1])
        except ValueError as exc:
            raise EmbedderMismatchError(
                f"manifest embedder_signature {signature!r} is malformed — "
                "expected 'hash-test:<dim>'") from exc
        if not (1 <= dim <= 8192):
            raise EmbedderMismatchError(
                f"manifest embedder_signature {signature!r}: dim out of range")
        return HashEmbedder(dim=dim)
    from stage3_embed.embedders import FastEmbedEmbedder
    return FastEmbedEmbedder()