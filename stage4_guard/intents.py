import re
from dataclasses import dataclass

REPLIES = {
    "greeting": "Hello. I answer questions using only the documents in this "
                "index — ask me anything about their content.",
    "thanks": "You're welcome.",
    "farewell": "Goodbye.",
    "identity": "I'm a document retrieval assistant. I search the indexed "
                "documents and answer strictly from what I find, with citations.",
    "capability": "I answer questions grounded in the indexed documents, with "
                  "page-level citations. Ask about the concepts, methods, or "
                  "examples they cover.",
    "low-coherence": "That doesn't read like a question I can search for — "
                     "please rephrase.",
}

_GREETINGS = {"hi", "hello", "hey", "yo", "howdy", "sup", "greetings",
              "hi there", "hello there", "good morning", "good afternoon",
              "good evening"}
_THANKS = {"thanks", "thank you", "thx", "ty", "thanks a lot",
           "much appreciated", "appreciate it", "great thanks"}
_FAREWELL = {"bye", "goodbye", "see you", "see ya", "later", "cya", "good night"}
_IDENTITY = {"who are you", "what are you", "are you a bot", "are you an ai",
             "are you human", "are you chatgpt", "whats your name",
             "what is your name", "who made you", "what model are you"}
_CAPABILITY = {"help", "what can you do", "what do you do", "what can i ask",
               "what can i ask you", "what are you for", "how does this work",
               "what do you know", "what topics can you answer"}

_PUNCT = re.compile(r"[^\w\s]")
_LEAD_GREETING = re.compile(
    r"^\s*(?:hi|hello|hey|yo|greetings|good (?:morning|afternoon|evening))"
    r"\s*[,!.]?\s+", re.IGNORECASE)


def _norm(s: str) -> str:
    return _PUNCT.sub("", s.lower()).strip()


@dataclass
class Intent:
    kind: str              # search | greeting | thanks | farewell | identity |
                           # capability | low-coherence
    reply: str | None = None
    remainder: str | None = None      # query to search if kind == search


def strip_leading_greeting(text: str) -> str | None:
    """'hi, what is XAI?' -> 'what is XAI?'. Only strips when a real question
    remains; a bare 'hi' is a full greeting, handled by the sets above."""
    m = _LEAD_GREETING.match(text)
    if not m:
        return None
    rest = text[m.end():].strip()
    if len(rest) >= 8 or len(rest.split()) >= 2:
        return rest
    return None


def _looks_gibberish(text: str) -> bool:
    """Vowel-less token density + character-diversity floor. Protects acronyms:
    'what is LSTM' has real words around the acronym, so it passes."""
    tokens = text.split()
    if len(tokens) < 2:
        return False
    sus = [t for t in tokens
           if len(t) >= 4 and not any(c in "aeiouAEIOU" for c in t)]
    real = [t for t in tokens if t not in sus]
    if len(sus) >= 2 and len(real) <= 1:
        return True
    if len(set(text)) <= max(2, len(text) // 8):     # "aaaaaaaaaa..."
        return True
    return False


def route_intent(text: str) -> Intent:
    n = _norm(text)
    for kind, table in (("greeting", _GREETINGS), ("thanks", _THANKS),
                        ("farewell", _FAREWELL), ("identity", _IDENTITY),
                        ("capability", _CAPABILITY)):
        if n in table:
            return Intent(kind, REPLIES[kind])
    stripped = strip_leading_greeting(text)
    if stripped:
        return Intent("search", None, remainder=stripped)
    if _looks_gibberish(text):
        return Intent("low-coherence", REPLIES["low-coherence"])
    return Intent("search", None, remainder=text)