import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import StoreError

log = logging.getLogger("stage3.store")


@dataclass
class SearchHit:
    score: float          # cosine similarity
    record: dict

    def __getitem__(self, key):
        return self.record[key]


class NumpyVectorStore:
    """Exact (brute-force) cosine search persisted in ONE atomic .npz file.

    Right-sized for <= ~100k vectors: the full matrix-vector product is a few
    milliseconds on CPU and recall@k is exactly 1.0 by construction — no ANN
    approximation, no server, no index-corruption modes. The public surface
    (upsert/delete/search/count) mirrors a vector-DB client so swapping in
    Qdrant/Milvus later is one adapter class, not a pipeline rewrite."""

    INDEX_FILE = "index.npz"

    def __init__(self, out_dir) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / self.INDEX_FILE
        self._ids: list[str] = []
        self._meta: list[dict] = []
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with np.load(self.path, allow_pickle=False) as z:
                self._vectors = np.asarray(z["vectors"], dtype=np.float32)
                self._ids = [str(x) for x in z["ids"].tolist()]
                self._meta = json.loads(z["meta_json"].item())
        except Exception as exc:
            raise StoreError(f"cannot load index {self.path}: {exc}") from exc
        if len(self._ids) != len(self._meta) or len(self._ids) != len(self._vectors):
            raise StoreError(f"index {self.path}: misaligned ids/meta/vectors")

    def save(self) -> None:
        """Single-file atomic save — a crash mid-save can never leave a torn
        index, because os.replace is atomic on the same filesystem."""
        dt = f"<U{max((len(i) for i in self._ids), default=1)}"
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, vectors=self._vectors,
                                ids=np.array(self._ids, dtype=dt),
                                meta_json=np.array(json.dumps(self._meta,
                                                              ensure_ascii=False)))
        os.replace(tmp, self.path)

    def wipe(self) -> None:
        self._ids, self._meta = [], []
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        if self.path.exists():
            self.path.unlink()

    # -- writes ------------------------------------------------------------
    def _normalize(self, vectors):
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        bad = norms.ravel() == 0.0
        if bad.any():
            log.warning("%d zero-norm vector(s) — kept as zero (matches nothing); "
                        "degenerate input text?", int(bad.sum()))
        norms[norms == 0.0] = 1.0
        return arr / norms

    def upsert(self, records: list[dict], vectors) -> int:
        if not records:
            return 0
        if len(records) != len(vectors):
            raise StoreError(f"upsert: {len(records)} records vs {len(vectors)} vectors")
        ids = [r["chunk_id"] for r in records]
        if len(set(ids)) != len(ids):
            raise StoreError("upsert: duplicate chunk_id within batch")
        vecs = self._normalize(vectors)
        drop = set(ids)
        if self._ids:
            keep = [i for i, x in enumerate(self._ids) if x not in drop]
            self._vectors = self._vectors[keep] if keep else self._vectors[:0]
            self._ids = [self._ids[i] for i in keep]
            self._meta = [self._meta[i] for i in keep]
        if self._vectors.shape[0] == 0:
            self._vectors = vecs
        else:
            if vecs.shape[1] != self._vectors.shape[1]:
                raise StoreError(f"dimension mismatch: index={self._vectors.shape[1]} "
                                 f"incoming={vecs.shape[1]} — mixed embedding models?")
            self._vectors = np.vstack([self._vectors, vecs])
        self._ids.extend(ids)
        self._meta.extend(records)
        return len(ids)

    def delete(self, doc_id: str | None = None, ids: list[str] | None = None) -> int:
        if not self._ids:
            return 0
        victims = set(ids or ())
        keep = []
        for i, (vid, meta) in enumerate(zip(self._ids, self._meta)):
            if doc_id is not None and meta.get("doc_id") == doc_id:
                continue
            if vid in victims:
                continue
            keep.append(i)
        removed = len(self._ids) - len(keep)
        if removed:
            self._vectors = self._vectors[keep] if keep else self._vectors[:0]
            self._ids = [self._ids[i] for i in keep]
            self._meta = [self._meta[i] for i in keep]
        return removed

    # -- reads -------------------------------------------------------------
    def count(self, doc_id: str | None = None) -> int:
        if doc_id is None:
            return len(self._ids)
        return sum(1 for m in self._meta if m.get("doc_id") == doc_id)

    def iter_records(self, doc_id: str | None = None):
        for m in self._meta:
            if doc_id is None or m.get("doc_id") == doc_id:
                yield m

    def search(self, vector, k: int = 10,
               where: dict | None = None) -> list[SearchHit]:
        """Exact cosine search with optional equality pre-filter — e.g.
        where={"doc_id": ...}, the hook Stage 5 uses for scoped retrieval."""
        if not self._ids:
            return []
        q = self._normalize(vector)[0]
        sims = self._vectors @ q
        hits: list[SearchHit] = []
        for idx in np.argsort(-sims):
            meta = self._meta[int(idx)]
            if where and any(meta.get(f) != v for f, v in where.items()):
                continue
            hits.append(SearchHit(score=float(sims[int(idx)]), record=meta))
            if len(hits) >= k:
                break
        return hits