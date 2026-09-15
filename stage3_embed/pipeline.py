import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from stage1_extract.artifacts import sha256_text, utc_now_iso
from stage2_chunk.artifacts import ChunkingArtifact
from stage2_chunk.tokens import default_token_estimator

from .embedders import Embedder, RetryPolicy, embed_batched, make_embedder
from .errors import EmbedderMismatchError, ManifestError, Stage3Error
from .metadata import build_chunk_record
from .store import NumpyVectorStore

log = logging.getLogger("stage3.pipeline")

PIPELINE_VERSION = "stage3-1.0.0"
MANIFEST_SCHEMA = "index-1.0"
MANIFEST_FILE = "manifest.json"


@dataclass
class DocResult:
    doc_id: str
    status: str                 # indexed | duplicate | pruned | failed
    chunks: int = 0
    tokens: int = 0
    error: str | None = None


@dataclass
class RunReport:
    results: list[DocResult] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return sum(r.status == "failed" for r in self.results)


def _fresh_manifest() -> dict[str, Any]:
    return {"schema_version": MANIFEST_SCHEMA, "embedder_signature": None,
            "pipeline_version": PIPELINE_VERSION, "dim": None,
            "docs": {}, "totals": {"chunks": 0, "tokens": 0, "cost": 0.0}}


class EmbeddingPipeline:
    """Stage 3 orchestration. Ordering guarantee:
    embed FIRST -> delete old doc vectors -> upsert -> verify count -> persist.
    A crash anywhere leaves the worst case as duplicate IDs (idempotent),
    never a deleted document with nothing replacing it."""

    def __init__(self, chunk_dir, out_dir, embedder: Embedder | None = None,
                 batch_size: int = 32,
                 estimator: Callable[[str], int] = default_token_estimator,
                 retry: RetryPolicy | None = None, force: bool = False,
                 rebuild: bool = False, prune: bool = False,
                 cost_per_1k_tokens: float = 0.0) -> None:
        if batch_size < 1:
            raise Stage3Error("batch_size must be >= 1")
        self.chunk_dir = Path(chunk_dir)
        self.out_dir = Path(out_dir)
        self.embedder = embedder or make_embedder("auto")
        self.batch_size = batch_size
        self.estimator = estimator
        self.retry = retry or RetryPolicy()
        self.force = force
        self.prune = prune
        self.cost_per_1k = cost_per_1k_tokens

        self.store = NumpyVectorStore(self.out_dir)
        self.manifest_path = self.out_dir / MANIFEST_FILE
        self.manifest = self._load_manifest()

        if rebuild:
            log.warning("rebuild: wiping index and manifest at %s", self.out_dir)
            self.store.wipe()
            self.manifest = _fresh_manifest()

        existing_sig = self.manifest.get("embedder_signature")
        if existing_sig and existing_sig != self.embedder.signature:
            raise EmbedderMismatchError(
                f"index at {self.out_dir} was built with '{existing_sig}' but this "
                f"run uses '{self.embedder.signature}'. Vectors from different "
                "embedding models are not comparable — re-embed everything with "
                "--rebuild, or point --out at a new directory.")

    # -- manifest ----------------------------------------------------------
    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return _fresh_manifest()
        try:
            m = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestError(f"corrupt manifest {self.manifest_path}: {exc}") from exc
        if m.get("schema_version") != MANIFEST_SCHEMA:
            raise ManifestError(
                f"{self.manifest_path}: schema {m.get('schema_version')!r} != "
                f"{MANIFEST_SCHEMA!r} — rebuild with --rebuild")
        m.setdefault("docs", {})
        m.setdefault("totals", {"chunks": 0, "tokens": 0, "cost": 0.0})
        return m

    def _save_manifest(self) -> None:
        tmp = self.manifest_path.with_name(self.manifest_path.name + ".tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, self.manifest_path)

    def _refresh_totals(self) -> None:
        docs = self.manifest["docs"]
        self.manifest["totals"] = {
            "chunks": sum(e["chunks"] for e in docs.values()),
            "tokens": sum(e.get("tokens_actual", 0) for e in docs.values()),
            "cost_usd": round(sum(e.get("cost_usd", 0.0) for e in docs.values()), 6),
        }

    # -- run ---------------------------------------------------------------
    def run(self) -> RunReport:
        files = sorted(self.chunk_dir.glob("*.json"))
        if not files:
            log.warning("no chunking artifacts in %s", self.chunk_dir)
        report = RunReport()
        known: set[str] = set()
        for f in files:
            try:
                result = self._process(f)
                report.results.append(result)
                known.add(result.doc_id)
            except Stage3Error as exc:
                log.error("failed: %s — %s", f.name, exc)
                report.results.append(DocResult(f.stem, "failed", error=str(exc)))
            except Exception:
                log.exception("unexpected failure: %s", f)
                report.results.append(DocResult(f.stem, "failed",
                                                error="unexpected (see log)"))
        if self.prune:
            report.results.extend(self._prune(known))
        else:
            stale = [d for d in self.manifest["docs"] if d not in known]
            if stale:
                log.warning("%d doc(s) in the index have no chunking artifact "
                            "(stale vectors). Re-run with --prune to remove: %s",
                            len(stale), ", ".join(d[:12] for d in stale))
        log.info("stage3 summary: %d indexed, %d duplicate, %d pruned, %d failed "
                 "(%d docs) | corpus: %d vectors, %d docs",
                 sum(r.status == "indexed" for r in report.results),
                 sum(r.status == "duplicate" for r in report.results),
                 sum(r.status == "pruned" for r in report.results),
                 report.failed, len(files),
                 self.store.count(), len(self.manifest["docs"]))
        return report

    def _process(self, path: Path) -> DocResult:
        art = ChunkingArtifact.load(path)
        entry = self.manifest["docs"].get(art.doc_id)
        source_hash = sha256_text("".join(c.content_hash for c in art.chunks))

        if (entry and not self.force
                and entry.get("source_chunk_hash") == source_hash
                and entry.get("embedder_signature") == self.embedder.signature):
            log.info("duplicate skip: %s (%d chunks current, %s)",
                     art.doc_id[:12], entry["chunks"],
                     (art.source_title or "")[:40])
            return DocResult(art.doc_id, "duplicate", chunks=entry["chunks"])

        texts = [c.to_embedding_text() for c in art.chunks]
        forecast = sum(c.token_estimate for c in art.chunks)
        log.info("indexing %s (%s): %d chunks, ~%d tokens forecast",
                 art.doc_id[:12], (art.source_title or art.source)[:50],
                 len(texts), forecast)

        t0 = time.perf_counter()
        vectors = embed_batched(self.embedder, texts, batch_size=self.batch_size,
                                policy=self.retry,
                                on_progress=self._progress(art.doc_id))
        t_embed = time.perf_counter() - t0

        actual = sum(self.estimator(t) for t in texts)
        if forecast and actual > forecast * 1.5:
            log.warning("token divergence on %s: actual %d vs forecast %d",
                        art.doc_id[:12], actual, forecast)

        index_meta = {"embedder_signature": self.embedder.signature,
                      "ingested_at": utc_now_iso(),
                      "pipeline_version": PIPELINE_VERSION}
        records = [build_chunk_record(c, art, index_meta) for c in art.chunks]

        t0 = time.perf_counter()
        removed = self.store.delete(doc_id=art.doc_id)   # shrink-case safety
        self.store.upsert(records, vectors)
        t_write = time.perf_counter() - t0

        live = self.store.count(doc_id=art.doc_id)       # verified write
        verified = live == len(records)
        if not verified:
            log.error("VERIFIED WRITE FAILED: %s — %d/%d vectors present",
                      art.doc_id[:12], live, len(records))
        if removed:
            log.info("replaced %d stale vectors for %s", removed, art.doc_id[:12])

        self.manifest["docs"][art.doc_id] = {
            "chunks": len(records),
            "source_chunk_hash": source_hash,
            "input_content_hash": art.input_content_hash,
            "config_fingerprint": art.config_fingerprint,
            "embedder_signature": self.embedder.signature,
            "indexed_at": utc_now_iso(),
            "tokens_forecast": forecast,
            "tokens_actual": actual,
            "cost_usd": round(self.cost_per_1k * actual / 1000.0, 6),
            "verified": verified,
            "timings_s": {"embed": round(t_embed, 1),
                          "write": round(t_write, 2),
                          "total": round(t_embed + t_write, 1)},
        }
        self.manifest["embedder_signature"] = self.embedder.signature
        self.manifest["dim"] = self.embedder.dim
        self._refresh_totals()

        self.store.save()          # persist after every doc: a crash loses at
        self._save_manifest()      # most the doc in flight, never the corpus
        log.info("indexed %s: %d vectors, %.0fs embed, verified=%s",
                 art.doc_id[:12], len(records), t_embed, verified)
        return DocResult(art.doc_id, "indexed", chunks=len(records), tokens=actual)

    def _progress(self, doc_id: str, every: int = 5):
        state = {"batches": 0, "t0": time.perf_counter()}

        def cb(done: int, total: int) -> None:
            state["batches"] += 1
            if state["batches"] % every == 0 or done == total:
                elapsed = max(time.perf_counter() - state["t0"], 1e-6)
                rate = done / elapsed
                eta = (total - done) / max(rate, 1e-6)
                log.info("  %s: %d/%d chunks (%.0f/s, eta %.0fs)",
                         doc_id[:12], done, total, rate, eta)
        return cb

    def _prune(self, known: set[str]) -> list[DocResult]:
        results = []
        for doc_id in list(self.manifest["docs"]):
            if doc_id in known:
                continue
            removed = self.store.delete(doc_id=doc_id)
            del self.manifest["docs"][doc_id]
            log.info("pruned %s: %d vectors removed (source artifact gone)",
                     doc_id[:12], removed)
            results.append(DocResult(doc_id, "pruned", chunks=removed))
        if results:
            self._refresh_totals()
            self.store.save()
            self._save_manifest()
        return results