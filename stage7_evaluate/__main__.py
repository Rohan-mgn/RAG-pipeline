# stage7_evaluate/__main__.py
import argparse
import json
import logging
import sys
from pathlib import Path

from stage4_guard.guard import GuardConfig, GuardPipeline
from stage5_retrieve.rerankers import make_reranker
from stage5_retrieve.retriever import HybridRetriever
from stage6_generate.generators import (DEFAULT_MODEL, DEFAULT_OLLAMA_URL,
                                        FakeGenerator, OllamaGenerator)

from .errors import GoldenSetError
from .golden import golden_fingerprint, load_golden
from .report import aggregate, build_report, diff_reports, save_report
from .runner import EvalConfig, EvalRunner

log = logging.getLogger("stage7.cli")

_SUMMARY_ORDER = ["context_recall", "context_precision", "faithfulness",
                  "answer_correctness", "answer_relevance",
                  "semantic_similarity", "citation_rate", "refusal_accuracy",
                  "negative_control_faithfulness", "judge_faithfulness",
                  "judge_answer_correctness", "judge_answer_relevance"]


def _summary_lines(agg: dict) -> list[str]:
    lines = []
    for k in _SUMMARY_ORDER:
        v = agg.get(k)
        if v is not None:
            lines.append(f" {k:<30} {v}")
    lines.append(f" {'answered / answerable':<30} {agg.get('answered')}/"
                 f"{agg.get('answerable')} "
                 f"(abstained-on-answerable: {agg.get('abstained_answerable')}, "
                 f"skipped: {agg.get('skipped')}, errors: {agg.get('failed')})")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="stage7_evaluate",
        description="Stage 7 — evaluate the RAG pipeline on the five metrics "
                    "against a golden question set")
    ap.add_argument("--golden", default="eval/golden.json")
    ap.add_argument("--index", default="artifacts/index")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--url", default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--only", choices=("answer", "abstain", "block"))
    ap.add_argument("--dry-run", action="store_true",
                    help="retrieval metrics only — no generation, no Ollama")
    ap.add_argument("--judge", choices=("none", "local"), default="none")
    ap.add_argument("--compare", metavar="REPORT",
                    help="diff against a previous eval report; exit 1 on "
                         "regression (loaded BEFORE this run overwrites "
                         "latest.json)")
    ap.add_argument("--delta", type=float, default=0.05,
                    help="regression threshold on aggregate drops")
    ap.add_argument("--out-dir", default="artifacts/eval/runs")
    ap.add_argument("--list", action="store_true", help="list golden set")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    # load baseline BEFORE running: latest.json is overwritten by this run
    baseline = None
    if args.compare:
        try:
            baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.error("cannot read baseline %s: %s", args.compare, exc)
            return 1

    try:
        questions = load_golden(args.golden)
    except GoldenSetError as exc:
        log.error("golden set: %s", exc)
        return 1

    if args.list:
        for gq in questions:
            print(f"{gq.expected:<7} {gq.id:<26} {gq.question[:70]}")
        return 0

    retriever = HybridRetriever.load(args.index, reranker=make_reranker("auto")) 
    if retriever.store.count() == 0:
        log.error("index at %s is EMPTY (0 vectors) - refusing to evaluate "
                  "against nothing. Rebuild the corpus first (stage 1 -> 2 -> 3). "
                  " Anempty-corpus run 'passes' its refusal controls vacuously "
                  "and would pin a meaningless baseline.", args.index)
        return 1

    generator = (FakeGenerator() if args.dry_run
                 else OllamaGenerator(model=args.model, url=args.url))
    judge_gen = (OllamaGenerator(model=args.model, url=args.url, temperature=0.0)
                 if (args.judge == "local" and not args.dry_run) else None)

    config = EvalConfig(top_n=args.top)
    runner = EvalRunner(retriever, generator, guard=GuardPipeline(GuardConfig()),
                        config=config, judge_generator=judge_gen)
    results = runner.run(questions, only=args.only, dry_run=args.dry_run)

    agg = aggregate(results)
    corpus = {"vectors": retriever.store.count(), "titles": runner.titles,
              "embedder": retriever.embedder.signature}
    report = build_report(results, agg, config, golden_fingerprint(args.golden),
                          corpus, args.dry_run)
    path = save_report(report, args.out_dir)

    print(f"\n=== Stage 7 evaluation{' (dry run)' if args.dry_run else ''} "
          f"=== corpus: {corpus['vectors']} vectors ===")
    print("\n".join(_summary_lines(agg)))
    print()
    for r in results:
        status = ("SKIP" if r.skipped else
                  "PASS" if r.passed else
                  "FAIL" if r.passed is not None else "?")
        key = ""
        for mk, label in (("context_recall", "recall"),
                          ("fact_coverage", "cov"), ("faithfulness", "faith")):
            v = r.metrics.get(mk)
            if isinstance(v, float):
                key += f" {label}={v:.2f}"
        why = r.skipped or r.error or ""
        print(f" [{status}] {r.id:<26} {r.expected}->{r.action or '-':<20}"
              f"{key} {why[:70]}")
    print(f"\nreport: {path}")

    if baseline is not None:
        d = diff_reports(report, baseline, args.delta)
        if d.get("scope_note"):
            print(f"\nWARNING: {d['scope_note']}")
        if d["regressions"]:
            print("\nREGRESSIONS vs baseline:")
            for reg in d["regressions"]:
                print(f" {reg['metric']}: {reg['old']} -> {reg['new']} "
                      f"({reg['delta']:+.3f})")
        if d["improvements"]:
            print("improvements:",
                  ", ".join(f"{i['metric']} {i['delta']:+.3f}"
                            for i in d["improvements"]))
        for ch in d.get("count_changes", []):
            print(f" count: {ch['metric']} = {ch['new']} (was {ch['old']})")
        if d["flips"]:
            print("question flips (pass -> fail):",
                  ", ".join(f["id"] for f in d["flips"]))
        if d["regressions"] or d["flips"]:
            return 1
        print("\nno regressions vs baseline")
    return 0

if __name__ == "__main__":
    sys.exit(main())