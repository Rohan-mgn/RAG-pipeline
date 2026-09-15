import logging
from dataclasses import dataclass, field
from pathlib import Path

from .artifacts import (SCHEMA_VERSION, PIPELINE_VERSION, ExtractionArtifact,
                        ParsedPage, sha256_file, sha256_text, utc_now_iso)
from .errors import ParseError, QualityGateError, Stage1Error
from .parsers import RawDoc, make_parser, read_pdf_metadata
from .pdfdates import normalize_pdf_date
from .textproc import process_page, strip_running_elements

log = logging.getLogger("stage1.ingest")

MIN_PAGE_SCORE = 0.35     # below this a page is 'low-quality'
MAX_BAD_RATIO = 0.60      # more than this fraction of usable pages bad -> refuse artifact


@dataclass
class FileResult:
    source: str
    doc_id: str | None = None
    status: str = ""                  # ingested | duplicate | failed
    error: str | None = None
    pages: int = 0
    content_hash: str | None = None


@dataclass
class IngestReport:
    results: list[FileResult] = field(default_factory=list)

    @property
    def ingested(self) -> int:
        return sum(r.status == "ingested" for r in self.results)

    @property
    def duplicates(self) -> int:
        return sum(r.status == "duplicate" for r in self.results)

    @property
    def failed(self) -> int:
        return sum(r.status == "failed" for r in self.results)


class Ingester:
    def __init__(self, out_dir: str | Path, parser=None,
                 min_page_score: float = MIN_PAGE_SCORE,
                 max_bad_ratio: float = MAX_BAD_RATIO, force: bool = False) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.parser = parser or make_parser()
        self.min_page_score = min_page_score
        self.max_bad_ratio = max_bad_ratio
        self.force = force

    def ingest_directory(self, root: str | Path,
                         patterns=("*.pdf",)) -> IngestReport:
        root = Path(root)
        files = sorted({p for pat in patterns for p in root.rglob(pat)})
        if not files:
            log.warning("no files matching %s under %s", list(patterns), root)
        report = IngestReport()
        for f in files:
            try:
                report.results.append(self.ingest_file(f))
            except Stage1Error as exc:            # classified failure
                log.error("failed: %s — %s", f.name, exc)
                report.results.append(FileResult(source=str(f), status="failed",
                                                 error=str(exc)))
            except Exception:                     # unexpected: full traceback
                log.exception("unexpected failure: %s", f)
                report.results.append(FileResult(source=str(f), status="failed",
                                                 error="unexpected (see log)"))
        log.info("stage1 summary: %d ingested, %d duplicate, %d failed (%d files)",
                 report.ingested, report.duplicates, report.failed, len(files))
        return report

    def ingest_file(self, path: str | Path) -> FileResult:
        path = Path(path)
        doc_id = sha256_file(path)
        artifact_path = self.out_dir / f"{doc_id}.json"

        if artifact_path.exists() and not self.force:
            log.info("duplicate skip: %s (doc_id=%s)", path.name, doc_id[:12])
            return FileResult(source=str(path), doc_id=doc_id, status="duplicate")

        try:
            raw: RawDoc = self.parser.parse(path)
        except Exception as exc:
            raise ParseError(f"{path.name}: {type(exc).__name__}: {exc}") from exc

        meta = read_pdf_metadata(path)
        pages: list[ParsedPage] = []
        for rp in raw.pages:                       # per-page failure isolation
            try:
                pp = process_page(rp.text)
            except Exception:
                pp = process_page("")             # never let one page kill the doc
                pp.warnings.append("page processing error — quarantined as empty")
            warnings = list(pp.warnings)
            if rp.needs_ocr:
                status = "needs-ocr"
                warnings.append("sparse text + image content — OCR unavailable/disabled")
            elif not pp.clean.strip():
                status = "skipped-empty"           # page number kept -> citations stay true
            elif pp.score < self.min_page_score:
                status = "low-quality"
            else:
                status = "ok"
            pages.append(ParsedPage(page_number=rp.page_number, width=rp.width,
                                    height=rp.height, rotation=rp.rotation,
                                    text=pp.clean, status=status, score=pp.score,
                                    flags=pp.flags, warnings=warnings))

        pages.sort(key=lambda p: p.page_number)
        deleted = strip_running_elements(pages)
        for p in pages:
            if p.status == "ok" and not p.text.strip():
                p.status = "skipped-empty"
                p.warnings.append("emptied by boilerplate removal")

        body = [p for p in pages if p.status != "skipped-empty"]
        bad = [p for p in body if p.status in ("low-quality", "needs-ocr")]
        if not body or len(bad) / max(len(body), 1) > self.max_bad_ratio:
            raise QualityGateError(
                f"{path.name}: {len(bad)}/{len(body)} usable pages below quality floor"
                + ("" if body else " — no extractable text (scanned, no OCR?)"))

        full_text = "\n".join(p.text for p in pages)
        art = ExtractionArtifact(
            schema_version=SCHEMA_VERSION, doc_id=doc_id,
            content_hash=sha256_text(full_text), source=str(path),
            title=raw.title or meta.get("title"), author=raw.author or meta.get("author"),
            created_at=normalize_pdf_date(raw.created_raw or meta.get("created_raw")),
            modified_at=normalize_pdf_date(raw.modified_raw or meta.get("modified_raw")),
            ingestion_timestamp=utc_now_iso(), pipeline_version=PIPELINE_VERSION,
            parser=self.parser.name, pages=pages,
            audit={"deleted_lines": deleted,
                   "config": {"parser": self.parser.name,
                              "min_page_score": self.min_page_score,
                              "max_bad_ratio": self.max_bad_ratio}},
            stats={"pages": len(pages),
                   "empty": sum(p.status == "skipped-empty" for p in pages),
                   "needs_ocr": sum(p.status == "needs-ocr" for p in pages),
                   "low_quality": sum(p.status == "low-quality" for p in pages),
                   "chars": len(full_text),
                   "boilerplate_lines_removed": len(deleted)})
        art.save(artifact_path)
        log.info("ingested: %s — %d pages, %d chars, parser=%s, doc_id=%s",
                 path.name, len(pages), len(full_text), self.parser.name, doc_id[:12])
        return FileResult(source=str(path), doc_id=doc_id, status="ingested",
                          pages=len(pages), content_hash=art.content_hash)