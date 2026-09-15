import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from stage1_extract.artifacts import sha256_text, utc_now_iso

from .errors import IncompatibleArtifactError

CHUNK_SCHEMA_VERSION = "chunking-1.0"


@dataclass
class Chunk:
    chunk_id: str            # sha256(f"{doc_id}::{seq}") — deterministic
    seq: int
    doc_id: str
    kind: str                # text | table
    heading_path: list[str]  # e.g. ["Chapter 5", "5.2 Evaluation"]
    text: str
    page_start: int
    page_end: int
    char_count: int
    token_estimate: int
    content_hash: str        # sha256(text) — detects drift on re-run

    def to_embedding_text(self) -> str:
        """One-call access for Stage 3: heading-prefixed chunk text."""
        prefix = " > ".join(self.heading_path)
        return f"{prefix}\n\n{self.text}" if prefix else self.text


@dataclass
class ChunkingArtifact:
    schema_version: str
    doc_id: str
    source_title: str | None
    source: str
    input_content_hash: str      # Stage 1 content_hash — change => auto re-chunk
    pipeline_version: str
    config: dict[str, Any]       # full chunking config snapshot
    config_fingerprint: str      # config change => auto re-chunk
    chunking_timestamp: str
    chunks: list[Chunk]
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, path)    # atomic, idempotent overwrite

    @classmethod
    def load(cls, path: Path) -> "ChunkingArtifact":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != CHUNK_SCHEMA_VERSION:
            raise IncompatibleArtifactError(
                f"{path}: schema {data.get('schema_version')!r} != "
                f"{CHUNK_SCHEMA_VERSION!r}; re-run Stage 2. Failing loudly.")
        fields = ("chunk_id", "seq", "doc_id", "kind", "heading_path", "text",
                  "page_start", "page_end", "char_count", "token_estimate",
                  "content_hash")
        chunks = [Chunk(**{k: c[k] for k in fields}) for c in data["chunks"]]
        return cls(schema_version=data["schema_version"], doc_id=data["doc_id"],
                   source_title=data["source_title"], source=data["source"],
                   input_content_hash=data["input_content_hash"],
                   pipeline_version=data["pipeline_version"], config=data["config"],
                   config_fingerprint=data["config_fingerprint"],
                   chunking_timestamp=data["chunking_timestamp"], chunks=chunks,
                   stats=data.get("stats", {}), warnings=data.get("warnings", []),
                   audit=data.get("audit", {}))