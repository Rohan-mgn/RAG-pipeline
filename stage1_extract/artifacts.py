import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import IncompatibleArtifactError

SCHEMA_VERSION = "extraction-1.1"
PIPELINE_VERSION = "stage1-1.0.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ParsedPage:
    page_number: int            # 1-based, matches the printed page, never renumbered
    width: float | None
    height: float | None
    rotation: int
    text: str                   # cleaned body text; headings prefixed '#'
    status: str                 # ok | needs-ocr | skipped-empty | low-quality
    score: float                # quality score 0..1
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class ExtractionArtifact:
    schema_version: str
    doc_id: str                 # sha256 of source bytes -> dedup key, content-addressed
    content_hash: str           # sha256 of cleaned text -> detects content drift
    source: str
    title: str | None           # None allowed here; stripped at vector-write time (S3)
    author: str | None
    created_at: str | None      # ISO 8601 or None
    modified_at: str | None
    ingestion_timestamp: str    # ISO 8601 UTC
    pipeline_version: str
    parser: str
    pages: list[ParsedPage]
    audit: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, path)          # atomic + idempotent overwrite; no orphans

    @classmethod
    def load(cls, path: Path) -> "ExtractionArtifact":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            raise IncompatibleArtifactError(
                f"{path}: schema {data.get('schema_version')!r} != {SCHEMA_VERSION!r}; "
                "re-run Stage 1 or migrate the artifact. Failing loudly, not silently.")
        pages = [ParsedPage(**{k: p.get(k) for k in
                     ("page_number", "width", "height", "rotation", "text",
                      "status", "score", "flags", "warnings")})
                 for p in data["pages"]]
        return cls(schema_version=data["schema_version"], doc_id=data["doc_id"],
                   content_hash=data["content_hash"], source=data["source"],
                   title=data["title"], author=data["author"],
                   created_at=data["created_at"], modified_at=data["modified_at"],
                   ingestion_timestamp=data["ingestion_timestamp"],
                   pipeline_version=data["pipeline_version"], parser=data["parser"],
                   pages=pages, audit=data.get("audit", {}), stats=data.get("stats", {}))