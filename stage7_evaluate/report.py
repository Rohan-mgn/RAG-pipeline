# stage7_evaluate/report.py
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from stage1_extract.artifacts import utc_now_iso

from .runner import PIPELINE_VERSION, QuestionResult

log = logging.getLogger("stage7.report")
REPORT_SCHEMA = "eval-1.0"


def aggregate(results: list[QuestionResult]) -> dict:
    def avg(rs, key):
        v = [r.metrics.get(key) for r in rs
             if isinstance(r.metrics.get(key), (int, float))
             and not isinstance(r.metrics.get(key), bool)]
        return round(sum(v) / len(v), 3) if v else None

    answerable = [r for r in results if r.expected == "answer" and not r.skipped]
    answered = [r for r in answerable if r.action == "answered"]
    refusals = [r for r in results
                if r.expected in ("abstain", "block") and not r.skipped]
    neg = [r.metrics.get("negative_faithfulness") for r in refusals
           if isinstance(r.metrics.get("negative_faithfulness"), float)]

    return {
        "context_recall": avg(answerable, "context_recall"),
        "context_precision": avg(answerable, "context_precision"),
        "faithfulness": avg(answered, "faithfulness"),
        "answer_correctness": avg(answered, "fact_coverage"),
        "answer_relevance": avg(answered, "answer_relevance"),
        "semantic_similarity": avg(answered, "semantic_similarity"),
        "citation_rate": (round(sum(1 for r in answered if r.metrics.get("cited"))
                                / len(answered), 3) if answered else None),
        "refusal_accuracy": (round(sum(1 for r in refusals if r.passed)
                                   / len(refusals), 3) if refusals else None),
        "negative_control_faithfulness": (round(sum(neg) / len(neg), 3)
                                          if neg else None),
        "judge_faithfulness": avg(answered, "judge_faithfulness"),
        "judge_answer_correctness": avg(answered, "judge_correctness"),
        "judge_answer_relevance": avg(answered, "judge_relevance"),
        "questions_total": len(results),
        "answerable": len(answerable),
        "answered": len(answered),
        "abstained_answerable": len(answerable) - len(answered),
        "refusal_cases": len(refusals),
        "skipped": sum(1 for r in results if r.skipped),
        "failed": sum(1 for r in results if r.error),
    }


def build_report(results, agg: dict, config, golden_fp: str, corpus: dict,
                 dry_run: bool) -> dict:
    return {
        "schema_version": REPORT_SCHEMA,
        "run_timestamp": utc_now_iso(),
        "pipeline_version": PIPELINE_VERSION,
        "dry_run": dry_run,
        "config": asdict(config),
        "golden_fingerprint": golden_fp,
        "corpus": corpus,
        "aggregates": agg,
        "questions": [r.to_dict() for r in results],
    }


METRIC_KEYS = ("context_recall", "context_precision", "faithfulness",
               "answer_correctness", "answer_relevance", "semantic_similarity",
               "citation_rate", "refusal_accuracy", "negative_control_faithfulness",
               "judge_faithfulness", "judge_answer_correctness",
               "judge_answer_relevance")
COUNT_KEYS = ("questions_total", "answerable", "answered",
              "abstained_answerable", "refusal_cases", "skipped", "failed")


def save_report(report: dict, out_dir="artifacts/eval/runs",
                update_latest: bool = True) -> Path:
    """Timestamped file always. latest.json only for full-scope runs — a
    dry-run or --only-filtered report is not a comparable baseline and must
    never hijack latest.json (the exact failure that fired the bogus gate)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = out / f"eval-{ts}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    os.replace(tmp, path)
    if update_latest:
        (out / "latest.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def diff_reports(new: dict, old: dict, delta: float = 0.05) -> dict:
    """Regression gate. Only METRIC_KEYS score drops gate the exit code.
    Count fields are informational, compared only when both runs share scope
    (same mode, question count, golden set); a scope mismatch prints a loud
    note instead of fabricating regressions. pass->fail flips always gate."""
    na, oa = new.get("aggregates", {}), old.get("aggregates", {})
    scope_note = None
    if (new.get("dry_run") != old.get("dry_run")
            or na.get("questions_total") != oa.get("questions_total")
            or new.get("golden_fingerprint") != old.get("golden_fingerprint")
            or (new.get("corpus") or {}).get("vectors") != (old.get("corpus") or {}).get("vectors")):
        scope_note = ("baseline scope differs (dry-run flag, question count, "
                      "golden set, or corpus size) — count diffs suppressed; "
                      "score diffs bundle confounds, read them with the dry-run "
                       "attribution steps")

    regressions, improvements = [], []
    for k in METRIC_KEYS:
        v, o = na.get(k), oa.get(k)
        if not isinstance(v, (int, float)) or not isinstance(o, (int, float)):
            continue
        d = v - o
        if d < -delta:
            regressions.append({"metric": k, "old": o, "new": v, "delta": round(d, 3)})
        elif d > delta:
            improvements.append({"metric": k, "old": o, "new": v, "delta": round(d, 3)})

    count_changes = []
    if scope_note is None:
        for k in COUNT_KEYS:
            v, o = na.get(k), oa.get(k)
            if isinstance(v, (int, float)) and isinstance(o, (int, float)) and v != o:
                count_changes.append({"metric": k, "old": o, "new": v})

    old_pass = {q.get("id"): q.get("passed") for q in old.get("questions", [])}
    flips = [{"id": q.get("id"), "was": old_pass.get(q.get("id")),
              "now": q.get("passed")}
             for q in new.get("questions", [])
             if q.get("id") in old_pass and old_pass[q.get("id")] is True
             and q.get("passed") is not True]
    return {"regressions": regressions, "improvements": improvements,
            "count_changes": count_changes, "scope_note": scope_note,
            "flips": flips}