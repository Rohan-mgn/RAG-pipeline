import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from .artifacts import ParsedPage

CID_TOKEN = re.compile(r"\(cid:\d+\)")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_DEHYPHEN = re.compile(r"([A-Za-z])-\n([a-z])")
_MOJIBAKE = ("\ufffd", "â€", "Ã¢", "Â\xa0")
_ZONE = 2          # header/footer territory = top-2 and bottom-2 lines only


@dataclass
class ProcessedPage:
    clean: str
    warnings: list[str]
    score: float
    flags: list[str]

def clean_text(raw: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    text = raw
    if CID_TOKEN.search(text):
        warnings.append(f"stripped {len(CID_TOKEN.findall(text))} (cid:xx) tokens")
        text = CID_TOKEN.sub(" ", text)
    text = unicodedata.normalize("NFKC", text) # NBSP, fi-ligatures, etc.
    text = text.replace("\t", " ")
    text = CONTROL_CHARS.sub(" ", text)
    text, n = _DEHYPHEN.subn(r"\1\2", text) # FIX: no trailing newline —
    if n: # 'explic-\nitly' -> 'explicitly'
        warnings.append(f"rejoined {n} hyphenated line breaks")
    text = re.sub(r"[ ]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip("\n"), warnings


def _score_raw(raw: str) -> tuple[float, list[str]]:
    """Explainable 0..1 score. Every penalty records a named flag."""
    stripped = raw.strip()
    if not stripped:
        return 1.0, ["empty"]
    n = max(len(raw), 1)
    penalty, flags = 0.0, []
    cid = len(CID_TOKEN.findall(raw))
    if cid:
        penalty += min(1.0, 4.0 * cid * 7 / n)
        flags.append(f"cid-garbage x{cid}")
    bad = raw.count("\ufffd")
    if bad:
        penalty += min(0.5, 0.02 * bad)
        flags.append(f"replacement-chars x{bad}")
    for marker in _MOJIBAKE:
        if marker in raw:
            penalty += 0.15
            flags.append(f"mojibake:{marker!r}")
            break
    ctrl = len(CONTROL_CHARS.findall(raw))
    if ctrl:
        penalty += min(0.3, 0.01 * ctrl)
        flags.append(f"control-chars x{ctrl}")
    alnum = sum(c.isalnum() for c in raw) / n
    if alnum < 0.35:
        penalty += 0.4
        flags.append(f"low-alnum-ratio {alnum:.2f}")
    lines = [l for l in stripped.splitlines() if l.strip()]
    if len(lines) > 8:
        tiny = sum(1 for l in lines if len(l.strip()) <= 2) / len(lines)
        if tiny > 0.5:
            penalty += 0.3
            flags.append(f"one-char-line-ratio {tiny:.2f}")
    return round(max(0.0, 1.0 - penalty), 3), flags


def process_page(raw: str) -> ProcessedPage:
    score, flags = _score_raw(raw)      # score the RAW text (pre-clean) — that's
    clean, warnings = clean_text(raw)   # where the encoding-garbage signal lives
    return ProcessedPage(clean, warnings, score, flags)


def _sig(line: str) -> str:
    """Signature for cross-page matching: digits -> '#', so 'Page 3' == 'Page 7'."""
    s = re.sub(r"\d+", "#", line.strip().lower())
    return re.sub(r"\s+", " ", s)


def _is_page_number(line: str) -> bool:
    s = line.strip()
    return (len(s) <= 12 and any(c.isdigit() for c in s)
            and re.fullmatch(r"[\d\s.\-–—]+", s) is not None)


def strip_running_elements(pages: list[ParsedPage]) -> list[dict]:
    """Frequency-based header/footer removal.
    Rules that protect the body: candidates are ONLY lines in the top/bottom
    _ZONE lines of a page; a signature must repeat on >= 40% of pages (min 3);
    bare page numbers must be pure digit-ish tokens <= 12 chars. Everything
    deleted is returned with page + reason — the audit trail."""
    min_hits = max(3, math.ceil(len(pages) * 0.4))
    counts: Counter[str] = Counter()
    for p in pages:
        lines = [l for l in p.text.splitlines() if l.strip()]
        if not lines:
            continue
        for sig in {_sig(l) for l in (lines[:_ZONE] + lines[-_ZONE:])}:
            counts[sig] += 1
    repeated = {s for s, c in counts.items() if c >= min_hits and s not in ("", "#")}

    audit: list[dict] = []
    for p in pages:
        lines = p.text.splitlines()
        if len(lines) <= 2 * _ZONE:      # too short to strip safely — body-only page
            continue
        keep: list[str] = []
        for i, line in enumerate(lines):
            in_zone = i < _ZONE or i >= len(lines) - _ZONE
            s = line.strip()
            if in_zone and s and (_sig(line) in repeated or _is_page_number(line)):
                audit.append({"page": p.page_number, "line": s,
                              "reason": "running-header/footer"
                                        if _sig(line) in repeated else "page-number"})
                continue
            keep.append(line)
        p.text = "\n".join(keep).strip("\n")
    return audit