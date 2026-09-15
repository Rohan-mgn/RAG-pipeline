import re
from dataclasses import dataclass, replace


@dataclass
class PatternHit:
    name: str
    severity: str          # "block" | "flag"
    snippet: str
    description: str


@dataclass
class InjectionPattern:
    name: str
    regex: re.Pattern
    severity: str
    description: str


def _p(name: str, pattern: str, severity: str, description: str) -> InjectionPattern:
    return InjectionPattern(name, re.compile(pattern, re.IGNORECASE),
                            severity, description)


# Precision-first: every pattern must be rare in legitimate questions about
# machine-learning documents. Weak signals are "flag" (annotate, proceed);
# strong signals are "block". Nothing here is corpus-specific.
PATTERNS: list[InjectionPattern] = [
    _p("instruction-override",
       r"\b(?:ignore|disregard|forget|override|bypass)\s+"
       r"(?:all\s+|any\s+|the\s+|your\s+)?"
       r"(?:previous|prior|above|earlier|preceding|past)\s+"
       r"(?:instructions?|prompts?|rules?|context|directives?|messages?|guardrails?)",
       "block", "attempt to void prior instructions"),
    _p("instruction-override-blanket",
       r"\b(?:ignore|disregard|forget|override)\s+(?:all|any|your)\s+"
       r"(?:instructions?|prompts?|rules?|directives?|guardrails?)",
       "block", "blanket instruction voiding"),
    _p("role-hijack",
       r"\b(?:you are now|from now on you(?:'re| are)|"
       r"act as if you(?:'re| are)|pretend (?:that )?you(?:'re| are))\b",
       "flag", "identity reassignment attempt"),
    _p("mode-switch",
       r"\b(?:developer|admin|god|dan|jailbroken|do\s?anything\s?now)\s+mode\b|"
       r"\b(?:enter|switch to|activate|enable)\s+(?:developer|admin|god|dan)\s+mode\b",
       "block", "known jailbreak mode activation"),
    _p("system-prompt-probe",
       r"\b(?:reveal|show|print|display|repeat|output|expose|leak|give me)\b"
       r"[^.?!]{0,60}?\b(?:system prompt|initial prompt|original instructions?|"
       r"hidden instructions?|secret prompt|your instructions)\b",
       "block", "system prompt exfiltration probe"),
    _p("chat-template-forgery",
       r"<\|?\s*(?:im_start|im_end|endoftext|system|assistant|user)\s*\|?>|"
       r"\[\s*(?:INST|SYSTEM)\s*\]|###\s*(?:SYSTEM|ASSISTANT|USER)\b",
       "block", "forged chat-control tokens"),
    _p("exfiltration",
       r"\b(?:send|post|upload|email|transmit|exfiltrate|forward)\b[^.?!]{0,50}?"
       r"\b(?:api\s?keys?|credentials?|passwords?|secrets?|"
       r"environment variables?|\.env)\b",
       "flag", "credential exfiltration instruction"),
    _p("repetition-exfil",
       r"\b(?:repeat|verbatim|word[- ]for[- ]word)\b[^.?!]{0,40}"
       r"\b(?:above|everything|all the text)\b",
       "flag", "verbatim context dump request"),
    _p("encoding-evasion",
       r"(?:decode|base64|rot13|hex[- ]decode)\s*[(=:]|\b[A-Za-z0-9+/]{90,}={0,2}\b",
       "flag", "encoded-payload smuggling"),
    _p("html-smuggle",
       r"<(?:iframe|script|img[^>]{0,120}onerror|svg[^>]{0,120}onload)",
       "flag", "HTML/script injection into rendered output"),
]


def scan(text: str, patterns: list[InjectionPattern] = PATTERNS) -> list[PatternHit]:
    hits: list[PatternHit] = []
    for p in patterns:
        m = p.regex.search(text)
        if m:
            hits.append(PatternHit(p.name, p.severity, m.group(0)[:60], p.description))
    return hits


def downgrade(hit: PatternHit) -> PatternHit:
    """block -> flag (used for interrogative contexts: questions ABOUT
    injection are legitimate; commands PERFORMING it are not)."""
    return replace(hit, severity="flag")