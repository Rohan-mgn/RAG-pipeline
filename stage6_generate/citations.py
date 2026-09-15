import re
from dataclasses import dataclass, field

from .packing import PackedContext

_MARKER = re.compile(r"\[(\d{1,3})\]")

ABSTENTION_PHRASES = ("i don't have enough information",
                      "i do not have enough information",
                      "insufficient information in the indexed documents")


@dataclass
class Citation:
    marker: int
    chunk_id: str
    title: str
    section: str
    pages: str

    def to_dict(self) -> dict:
        return {"marker": self.marker, "chunk_id": self.chunk_id,
                "title": self.title, "section": self.section, "pages": self.pages}


@dataclass
class CitationReport:
    citations: list[Citation] = field(default_factory=list)
    invalid_markers: list[int] = field(default_factory=list)
    uncited: bool = False
    abstained: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"citations": [c.to_dict() for c in self.citations],
                "invalid_markers": self.invalid_markers,
                "uncited": self.uncited, "abstained": self.abstained,
                "warnings": self.warnings}


def verify_citations(answer_text: str,
                     packed: PackedContext) -> tuple[CitationReport, str]:
    """The machine-checked citation contract: markers parsed, out-of-range
    markers stripped from displayed text with an audible warning, valid
    markers resolved to chunk IDs + pages + sections, and uncited
    non-abstaining answers flagged as grounding risks.
    Returns (report, cleaned_text)."""
    n = len(packed.blocks)
    text = answer_text or ""
    markers = [int(m) for m in _MARKER.findall(text)]
    valid = sorted({m for m in markers if 1 <= m <= n})
    invalid = sorted({m for m in markers if not (1 <= m <= n)})

    warnings: list[str] = []
    for m in invalid:
        text = re.sub(rf"\[{m}\]", "", text)
    if invalid:
        warnings.append(f"stripped out-of-range citation markers {invalid} "
                        f"(only [1]-[{n}] exist)")

    abstained = any(p in text.lower() for p in ABSTENTION_PHRASES)
    cmap = packed.citation_map
    citations: list[Citation] = []
    for m in valid:
        b = cmap[m]
        pages = (f"p.{b.page_start}" if b.page_start == b.page_end
                 else f"pp.{b.page_start}-{b.page_end}")
        citations.append(Citation(marker=m, chunk_id=b.chunk_id, title=b.title,
                                  section=b.section, pages=pages))

    uncited = bool(text.strip()) and not valid and not abstained
    if uncited:
        warnings.append("answer carries no valid citations — grounding risk")
    if abstained and valid:
        warnings.append("answer abstains but also cites context — prompt drift?")

    report = CitationReport(citations=citations, invalid_markers=invalid,
                            uncited=uncited, abstained=abstained, warnings=warnings)
    return report, text.strip()