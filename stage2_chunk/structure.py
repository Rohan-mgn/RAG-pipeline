# stage2_chunk/structure.py
import re
from dataclasses import dataclass, field, replace

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_CHAPTER_RE = re.compile(r"^(?:chapter|appendix|part)\s+(?:\d+|[IVXLCDM]+|[A-Z])\b",
                         re.IGNORECASE)
_NUMBERED_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3})*)\s+(\S.*)$")
_BULLET = re.compile(r"^[-*•]\s+")

# A '#' line carrying code tokens is likelier a Colab-cell comment or code
# than a document heading. Structural signal — no book-specific strings.
_MD_CODE_TOKENS = re.compile(r"[()=;@`]|import |def |return |print\(|\bself\b|http| # ")


def _md_heading_ok(text: str) -> bool:
    """Plausibility guard for '#' heading lines: short prose, free of code
    tokens. Applied UNIVERSALLY (all parsers) — production evidence showed
    even docling labels notebook/code cells as titles."""
    s = text.strip()
    return 0 < len(s) <= 90 and not _MD_CODE_TOKENS.search(s)


@dataclass
class Block:
    kind: str                                  # heading | para | list_item | table
    text: str
    page_start: int
    page_end: int
    level: int | None = None                   # headings only
    marks: list[tuple[int, int]] = field(default_factory=list)  # (char offset, page)


def plain_heading_level(line: str) -> int | None:
    """Heuristic for plain-text pages: 'Chapter 5', '3.2 Why XAI'.
    Conservative — requires a numbering pattern, short line, no sentence-ending
    punctuation, capitalized title start. Body text never matches."""
    s = line.strip()
    if not (0 < len(s) <= 90) or s.endswith((".", ",", ";")):
        return None
    if _CHAPTER_RE.match(s):
        return 1
    m = _NUMBERED_RE.match(s)
    if m and not m.group(2)[0].islower():      # '2020 was a year' -> body text
        return 1 + m.group(1).count(".")
    return None


def page_blocks(text: str, page_number: int, detect_headings: bool = True,
                trust_markdown: bool = True) -> list[Block]:
    # TOC suppression: a page carrying many chapter/numbered entries is a
    # table of contents, not a heading sequence. Entries stay body text.
    if detect_headings:
        lines_all = [l for l in text.split("\n") if l.strip()]
        chapterish = sum(1 for l in lines_all if _CHAPTER_RE.match(l.strip()))
        numberedish = sum(1 for l in lines_all if _NUMBERED_RE.match(l.strip()))
        if chapterish >= 3 or numberedish >= 6:
            detect_headings = False

    blocks: list[Block] = []
    para: list[str] = []
    table: list[str] = []
    in_table = False

    def flush_para() -> None:
        if para:
            joined = " ".join(s.strip() for s in para).strip()
            if joined:
                blocks.append(Block("para", joined, page_number, page_number,
                                    marks=[(0, page_number)]))
            para.clear()

    def flush_table() -> None:
        if table:
            blocks.append(Block("table", "\n".join(table), page_number, page_number,
                                marks=[(0, page_number)]))
            table.clear()

    for raw in text.split("\n"):
        s = raw.strip()
        if in_table:
            if s == "[/TABLE]":
                in_table = False
                flush_table()
            elif s:
                table.append(s)
            continue
        if s == "[TABLE]":
            flush_para()
            in_table = True
            continue
        if not s:
            flush_para()
            continue
        m = _MD_HEADING.match(s)
        if m and (trust_markdown or _md_heading_ok(m.group(2))):
            flush_para()
            blocks.append(Block("heading", " ".join(m.group(2).split()),
                                page_number, page_number, level=len(m.group(1)),
                                marks=[(0, page_number)]))
            continue
        if detect_headings:
            lvl = plain_heading_level(s)
            if lvl is not None:
                flush_para()
                blocks.append(Block("heading", " ".join(s.split()),
                                    page_number, page_number, level=lvl,
                                    marks=[(0, page_number)]))
                continue
        if _BULLET.match(s):
            flush_para()
            blocks.append(Block("list_item", _BULLET.sub("", s),
                                page_number, page_number, marks=[(0, page_number)]))
            continue
        para.append(s)

    flush_para()
    if in_table:                # unclosed [TABLE] at page end — never silently drop
        flush_table()
    return blocks


def _ends_sentence(text: str) -> bool:
    t = text.rstrip()
    return bool(t) and t[-1] in ".!?…\"'”)]"


def stitch_blocks(blocks: list[Block]) -> tuple[list[Block], dict]:
    """Rejoin paragraphs split by page boundaries. Only when: previous para
    lacks terminal punctuation, next page starts lowercase/digit, pages are
    consecutive. Tracks char-offset page marks so every sentence keeps its
    true page for citation."""
    result: list[Block] = []
    stats = {"stitched": 0, "dehyphenated": 0}
    buf: Block | None = None
    for b in blocks:
        if b.kind != "para":
            if buf:
                result.append(buf)
                buf = None
            result.append(b)
            continue
        if (buf is not None
                and b.page_start == buf.page_end + 1
                and not _ends_sentence(buf.text)
                and (b.text[:1].islower() or b.text[:1].isdigit())):
            if buf.text.endswith("-"):
                buf.text = buf.text[:-1]
                stats["dehyphenated"] += 1
            else:
                buf.text += " "
            buf.marks.append((len(buf.text), b.page_start))
            buf.text += b.text
            buf.page_end = b.page_end
            stats["stitched"] += 1
        else:
            if buf:
                result.append(buf)
            buf = replace(b, marks=list(b.marks))
    if buf:
        result.append(buf)
    return result, stats


def page_at(marks: list[tuple[int, int]], idx: int) -> int:
    page = marks[0][1] if marks else 1
    for off, pg in marks:
        if off <= idx:
            page = pg
        else:
            break
    return page


def extract_blocks(art, detect_headings: bool = True) -> tuple[list[Block], dict]:
    """Universal heading plausibility guard: NO parser's '#' lines are trusted
    blindly — everything passes _md_heading_ok. (trust_markdown param of
    page_blocks is retained for API compatibility but defaults to False here.)"""
    blocks: list[Block] = []
    headings = []
    for p in art.pages:
        if p.status not in ("ok", "low-quality", "needs-ocr"):
            continue
        for b in page_blocks(p.text, p.page_number, detect_headings,
                             trust_markdown=False):
            blocks.append(b)
            if b.kind == "heading":
                headings.append({"page": b.page_start, "level": b.level, "text": b.text})
    return blocks, {"headings": headings}