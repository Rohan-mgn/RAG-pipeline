import re
import unicodedata
from dataclasses import dataclass, field

_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")


@dataclass
class SanitizeResult:
    original: str
    cleaned: str
    flags: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def removed_chars(self) -> int:
        return len(self.original) - len(self.cleaned)


class QuerySanitizer:
    """Unicode hardening, evasion-resistant by design: NFKC folds fullwidth
    lookalikes (ｉｇｎｏｒｅ -> ignore) so pattern matching can't be evaded by
    lookalike glyphs; zero-width and control characters are stripped;
    whitespace collapsed; pathological length capped — audibly, with flags."""

    def __init__(self, max_chars: int = 2000) -> None:
        self.max_chars = max_chars

    def sanitize(self, text: str) -> SanitizeResult:
        original = text or ""
        s = original
        flags: list[str] = []
        if _ZERO_WIDTH.search(s):
            flags.append("zero-width-chars-stripped")
        s = _ZERO_WIDTH.sub("", s)
        if _CONTROL.search(s):
            flags.append("control-chars-stripped")
        s = _CONTROL.sub(" ", s)
        nfkc = unicodedata.normalize("NFKC", s)
        if nfkc != s:
            flags.append("nfkc-normalized")
        s = nfkc
        s = _WS.sub(" ", s).strip()
        truncated = False
        if len(s) > self.max_chars:
            cut = s[: self.max_chars]
            if " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            s = cut
            truncated = True
            flags.append(f"truncated-at-{self.max_chars}-chars")
        return SanitizeResult(original=original, cleaned=s, flags=flags,
                              truncated=truncated)