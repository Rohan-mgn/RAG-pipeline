# stage7_evaluate/judge.py
import json
import logging
import re

log = logging.getLogger("stage7.judge")

JUDGE_SYSTEM = """You are a strict, skeptical evaluation judge for a retrieval-augmented question answering system.

You will be given a QUESTION, a REFERENCE ANSWER (ground truth), the retrieved CONTEXT, and the system's ANSWER.

Score three metrics, each from 0.0 to 1.0:
- faithfulness: how much of the ANSWER is supported by the CONTEXT alone. Unsupported or invented claims lower the score.
- relevance: how well the ANSWER addresses the QUESTION that was asked.
- correctness: how closely the ANSWER matches the REFERENCE ANSWER in substance.

Be harsh: partial support means a partial score, not full credit.

Reply with ONLY a JSON object, no other text:
{"faithfulness": 0.0, "relevance": 0.0, "correctness": 0.0}"""


def judge_question(generator, question: str, reference: str | None,
                   context_text: str, answer: str) -> dict | None:
    """LLM-judged scores, or None on any failure — never raises. Caveat: a
    model judging its own family of outputs has self-preference bias; treat
    as a cross-check over the deterministic metrics, not ground truth."""
    user = (f"QUESTION: {question}\n\nREFERENCE ANSWER: {reference or '(none)'}\n\n"
            f"CONTEXT:\n{(context_text or '')[:6000]}\n\nANSWER: {answer}")
    try:
        text, _ = generator.generate(JUDGE_SYSTEM, user, max_tokens=120)
    except Exception as exc:
        log.warning("judge call failed: %s", exc)
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        log.warning("judge reply not JSON: %r", (text or "")[:120])
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k in ("faithfulness", "relevance", "correctness"):
        v = data.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0:
            out[k] = float(v)
    return out or None