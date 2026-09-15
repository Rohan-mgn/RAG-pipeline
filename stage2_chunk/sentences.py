import logging
import re

from .errors import ConfigError

log = logging.getLogger("stage2.sentences")

# Split after . ! ? … (optionally before a closing quote/bracket) when followed
# by whitespace + something that starts a new sentence.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?…\"”)])\s+(?=[A-Z0-9\"“(\[])")
_MASK = "\u2038"   # caret-insertion char: never appears in real document text

_ABBREVIATIONS = ("e.g", "i.e", "etc", "vs", "fig", "figs", "eq", "eqs", "no",
                  "nos", "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st",
                  "approx", "al", "ch", "chpt", "sec", "pp", "p", "vol", "cf",
                  "ca", "viz", "dept", "univ", "inc", "ltd", "co", "ed", "eds",
                  "min", "max", "avg")
_ABBREV_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.",
    re.IGNORECASE)
_INITIAL_RE = re.compile(r"\b[A-Z]\.")     # "J. Smith", "U.S."


class RegexSentenceSplitter:
    """Stdlib splitter: mask abbreviation/initial dots, split on sentence
    boundaries, unmask. No models, no network, deterministic."""
    name = "regex"

    def __call__(self, text: str) -> list[str]:
        masked = _ABBREV_RE.sub(lambda m: m.group(0)[:-1] + _MASK, text)
        masked = _INITIAL_RE.sub(lambda m: m.group(0)[:-1] + _MASK, masked)
        parts = _SENT_BOUNDARY.split(masked)
        return [p.replace(_MASK, ".").strip() for p in parts if p.strip()]


class SpacySentenceSplitter:
    """Injected upgrade: better boundary detection, same interface."""
    name = "spacy"

    def __init__(self, model: str = "en_core_web_sm") -> None:
        import spacy
        self.nlp = spacy.load(model, exclude=["ner", "lemmatizer", "textcat"])

    def __call__(self, text: str) -> list[str]:
        return [s.text.strip() for s in self.nlp(text).sents if s.text.strip()]


def make_splitter(mode: str = "auto"):
    """Returns (splitter, name). 'auto' degrades to regex; explicit 'spacy'
    fails loudly instead of silently downgrading."""
    if mode in ("auto", "spacy"):
        try:
            sp = SpacySentenceSplitter()
            log.info("sentence splitter: spacy/en_core_web_sm")
            return sp, sp.name
        except Exception as exc:
            if mode == "spacy":
                raise ConfigError(f"spacy splitter requested but unavailable: {exc}") from exc
            log.warning("spacy unavailable (%s) — using regex splitter", exc)
    log.info("sentence splitter: regex (stdlib)")
    rx = RegexSentenceSplitter()
    return rx, rx.name