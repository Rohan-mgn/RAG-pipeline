import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("stage3.metadata")


def sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Vector-store metadata contract: scalar values only, no None (several
    stores raise on null). Lists of scalars are joined; anything else is
    dropped with a warning — audible, never fatal, never silent."""
    clean: dict[str, Any] = {}
    for key, val in meta.items():
        if val is None:
            continue
        if isinstance(val, (str, int, float, bool)):
            clean[key] = val
        elif isinstance(val, (list, tuple)):
            if not val:
                continue
            if all(isinstance(x, (str, int, float, bool)) for x in val):
                clean[key] = " | ".join(str(x) for x in val)
            else:
                log.warning("metadata %r: non-scalar list dropped", key)
        else:
            log.warning("metadata %r: unsupported %s dropped",
                        key, type(val).__name__)
    return clean


def build_chunk_record(chunk, chunk_artifact, index_meta: dict[str, Any]) -> dict[str, Any]:
    """Flat, scalar-only, cross-store-safe record. The embedder signature
    rides on every vector, so a mixed-model index is detectable even if the
    manifest is lost — belt and suspenders on model pinning."""
    title = chunk_artifact.source_title or Path(chunk_artifact.source).name
    return sanitize_metadata({
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "seq": chunk.seq,
        "kind": chunk.kind,
        "heading_path": " > ".join(chunk.heading_path),
        "text": chunk.text,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "char_count": chunk.char_count,
        "token_estimate": chunk.token_estimate,
        "content_hash": chunk.content_hash,
        "title": title,
        "source": chunk_artifact.source,
        "source_pipeline_version": chunk_artifact.pipeline_version,
        "source_warnings": len(chunk_artifact.warnings),
        "embedder": index_meta["embedder_signature"],
        "ingested_at": index_meta["ingested_at"],
        "pipeline_version": index_meta["pipeline_version"],
    })