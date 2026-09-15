import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("stage1.parsers")


@dataclass
class RawPage:
    page_number: int
    width: float | None
    height: float | None
    rotation: int
    text: str
    needs_ocr: bool = False


@dataclass
class RawDoc:
    pages: list[RawPage]
    title: str | None = None
    author: str | None = None
    created_raw: str | None = None
    modified_raw: str | None = None


def read_pdf_metadata(path: Path) -> dict:
    """Metadata via pypdf — independent of the layout parser, pure-Python dep,
    and failure is non-fatal (returns {} rather than killing ingestion)."""
    try:
        from pypdf import PdfReader
        md = PdfReader(str(path)).metadata or {}

        def s(key: str) -> str | None:
            v = md.get(key)
            v = str(v).strip() if v is not None else ""
            return v or None

        return {"title": s("/Title"), "author": s("/Author"),
                "created_raw": s("/CreationDate"), "modified_raw": s("/ModDate")}
    except Exception:
        return {}


def _try_tesseract(opts, lang: str) -> bool:
    """Enable Tesseract OCR if docling exposes the option class and the binary
    exists. Option-class names vary across docling minors — probe, never crash."""
    from docling.datamodel import pipeline_options as po
    for name in ("TesseractClOcrOptions", "TesseractOcrOptions"):
        cls = getattr(po, name, None)
        if cls is None:
            continue
        for lang_arg in ([lang], lang):
            try:
                opts.ocr_options = cls(lang=lang_arg)
                opts.do_ocr = True
                return True
            except Exception:
                continue
    return False


class DoclingParser:
    """Primary parser: layout-aware reading order, two-column pages, tables,
    headings. Attribute access is defensive so a docling minor-version change
    degrades gracefully instead of crashing mid-batch."""
    name = "docling"

    def __init__(self, do_ocr: bool = True, ocr_lang: str = "eng") -> None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_table_structure = True
        _mode = next((getattr(TableFormerMode, m) for m in ("LIGHT", "FAST", "ACCURATE")
                      if hasattr(TableFormerMode, m)), None)
        if _mode is not None:
            opts.table_structure_options.mode = _mode  # CPU-friendly
        opts.do_ocr = False
        if do_ocr and shutil.which("tesseract"):
            opts.do_ocr = _try_tesseract(opts, ocr_lang)
        self._ocr_enabled = opts.do_ocr
        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})

    @property
    def ocr_enabled(self) -> bool:
        return self._ocr_enabled

    def parse(self, path: Path) -> RawDoc:
        result = self._converter.convert(str(path))
        doc = result.document

        sizes: dict[int, tuple] = {}
        for no, pg in (getattr(doc, "pages", None) or {}).items():
            size = getattr(pg, "size", None)
            sizes[no] = (getattr(size, "width", None), getattr(size, "height", None),
                         int(getattr(pg, "angle", 0) or 0))

        chunks: dict[int, list[str]] = {}
        pictures: dict[int, int] = {}
        for entry in doc.iterate_items():
            item, level = entry if isinstance(entry, tuple) else (entry, 0)
            prov = getattr(item, "prov", None) or []
            page_no = int(getattr(prov[0], "page_no", 1)) if prov else 1
            label = str(getattr(item, "label", "")).lower()

            if label == "picture":
                pictures[page_no] = pictures.get(page_no, 0) + 1
                continue
            if label.endswith("table") and hasattr(item, "export_to_markdown"):
                chunks.setdefault(page_no, []).append(
                    f"\n[TABLE]\n{item.export_to_markdown()}\n[/TABLE]\n")
                continue
            text = str(getattr(item, "text", "") or "").strip()
            if not text:
                continue
            if label == "title":
                piece = f"\n\n# {text}\n\n"
            elif "section_header" in label:
                depth = min(max(int(level or 1), 1), 5) + 1
                piece = f"\n\n{'#' * depth} {text}\n\n"
            elif "list_item" in label:
                piece = f"- {text}\n"
            else:
                piece = text + "\n\n"
            chunks.setdefault(page_no, []).append(piece)

        last = max(sizes or chunks or {0: None})
        pages = []
        for no in range(1, last + 1):
            w, h, rot = sizes.get(no, (None, None, 0))
            text = "".join(chunks.get(no, [])).strip()
            sparse = len(text) < 40 and pictures.get(no, 0) > 0
            pages.append(RawPage(no, w, h, rot, text, needs_ocr=sparse))
        return RawDoc(pages=pages, **read_pdf_metadata(path))


class PyMuPDFParser:
    """Fallback parser: no ML models, no torch — runs on any interpreter that
    has wheels. Reading order approximated by sorted blocks; true two-column
    fidelity is what Docling is for. Sparse text + images -> needs_ocr flag."""
    name = "pymupdf"

    def parse(self, path: Path) -> RawDoc:
        import pymupdf
        with pymupdf.open(str(path)) as doc:
            md = doc.metadata or {}
            pages = []
            for i, page in enumerate(doc, start=1):
                blocks = page.get_text("blocks", sort=True)   # coordinate-ordered
                text = "\n".join(str(b[4]).strip() for b in blocks
                                 if int(b[6]) == 0 and str(b[4]).strip())
                sparse = (len(text.strip()) < 40
                          and len(page.get_images(full=True)) > 0)
                pages.append(RawPage(i, page.rect.width, page.rect.height,
                                     int(page.rotation), text, needs_ocr=sparse))
        return RawDoc(pages=pages,
                      title=md.get("title") or None, author=md.get("author") or None,
                      created_raw=md.get("creationDate") or None,
                      modified_raw=md.get("modDate") or None)


def make_parser(prefer: str = "auto", do_ocr: bool = True):
    if prefer in ("auto", "docling"):
        try:
            p = DoclingParser(do_ocr=do_ocr)
            log.info("parser: docling (ocr=%s)", p.ocr_enabled)
            return p
        except Exception as exc:               # import failure, wheel gap, model load
            if prefer == "docling":
                raise
            log.warning("docling unavailable (%s) — using PyMuPDF fallback", exc)
    log.info("parser: pymupdf (fallback)")
    return PyMuPDFParser()