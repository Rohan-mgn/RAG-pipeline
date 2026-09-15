# stage2_chunk/pipeline.py
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from stage1_extract.artifacts import ExtractionArtifact, sha256_text

from .artifacts import CHUNK_SCHEMA_VERSION
from .chunker import ChunkerConfig, chunk_document
from .errors import ArtifactError, Stage2Error
from .sentences import make_splitter
from .tokens import default_token_estimator

log = logging.getLogger("stage2.pipeline")
PIPELINE_VERSION = "stage2-1.0.5"


@dataclass
class DocResult:
    doc_id: str
    status: str # ingested | duplicate | failed
    chunks: int = 0
    error: str | None = None


@dataclass
class RunReport:
    results: list[DocResult] = field(default_factory=list)

    @property
    def ingested(self) -> int:
        return sum(r.status == "ingested" for r in self.results)

    @property
    def duplicates(self) -> int:
        return sum(r.status == "duplicate" for r in self.results)

    @property
    def failed(self) -> int:
        return sum(r.status == "failed" for r in self.results)


def config_fingerprint(cfg: ChunkerConfig, splitter_name: str) -> str:
    payload = {"target_tokens": cfg.target_tokens, "max_tokens": cfg.max_tokens,
               "min_tokens": cfg.min_tokens, "overlap_sentences": cfg.overlap_sentences,
               "detect_headings": cfg.detect_headings, "splitter": splitter_name,
               "pipeline_version": PIPELINE_VERSION}
    return sha256_text(json.dumps(payload, sort_keys=True))


class ChunkingPipeline:
    def __init__(self, in_dir, out_dir, config: ChunkerConfig | None = None,
                 splitter: str = "auto", estimator=default_token_estimator,
                 force: bool = False) -> None:
        self.in_dir = Path(in_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or ChunkerConfig()
        self.config.validate()
        self.splitter, self.splitter_name = make_splitter(splitter)
        self.estimator = estimator
        self.force = force
        self.fingerprint = config_fingerprint(self.config, self.splitter_name)

    def run(self) -> RunReport:
        files = sorted(self.in_dir.glob("*.json"))
        if not files:
            log.warning("no extraction artifacts in %s", self.in_dir)
        report = RunReport()
        for f in files:
            try:
                report.results.append(self._process(f))
            except Stage2Error as exc:
                log.error("failed: %s — %s", f.name, exc)
                report.results.append(DocResult(f.stem, "failed", error=str(exc)))
            except Exception:
                log.exception("unexpected failure: %s", f)
                report.results.append(DocResult(f.stem, "failed",
                                                error="unexpected (see log)"))
        log.info("stage2 summary: %d chunked, %d duplicate, %d failed (%d docs)",
                 report.ingested, report.duplicates, report.failed, len(files))
        return report

    def _process(self, path: Path) -> DocResult:
        try:
            art = ExtractionArtifact.load(path)
        except Exception as exc:
            raise ArtifactError(f"{path.name}: {exc}") from exc

        out_path = self.out_dir / f"{art.doc_id}.json"
        if out_path.exists() and not self.force:
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
            if (existing.get("schema_version") == CHUNK_SCHEMA_VERSION
                    and existing.get("input_content_hash") == art.content_hash
                    and existing.get("config_fingerprint") == self.fingerprint):
                log.info("duplicate skip: %s (%d chunks already current)",
                         art.doc_id[:12], len(existing.get("chunks", [])))
                return DocResult(art.doc_id, "duplicate",
                                 chunks=len(existing.get("chunks", [])))

        artifact = chunk_document(art, self.config, self.splitter, self.estimator,
                                  pipeline_version=PIPELINE_VERSION,
                                  config_fingerprint=self.fingerprint)
        artifact.save(out_path)
        log.info("chunked %s (%s): %d chunks, avg %d tok, %d headings, "
                 "%d stitched paras, %d warnings",
                 art.doc_id[:12], (art.title or art.source)[:50],
                 artifact.stats["chunks"], artifact.stats["tokens_avg"],
                 artifact.stats["headings"], artifact.stats["stitched_paragraphs"],
                 len(artifact.warnings))
        return DocResult(art.doc_id, "ingested", chunks=artifact.stats["chunks"])