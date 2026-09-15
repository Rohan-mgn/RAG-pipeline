# stage4_guard/__main__.py
import argparse
import json
import logging
import sys
from dataclasses import asdict

from .errors import ConfigError
from .guard import GuardConfig, GuardPipeline
from .mlclassifier import try_load_classifier


def _render(plan) -> str:
    lines = [f"action:     {plan.action.upper()}",
             f"query:      {plan.cleaned_query!r}"]
    if plan.intent != "search":
        lines.append(f"intent:     {plan.intent}")
    if plan.injection_flags:
        lines.append(f"flags:      {', '.join(plan.injection_flags)}")
    if plan.ml_score is not None:
        lines.append(f"ml score:   {plan.ml_score:.3f}")
    if plan.intent_reply:
        lines.append(f"reply:      {plan.intent_reply}")
    lines.append(f"timings_ms: {plan.timings_ms}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="stage4_guard",
        description="Stage 4 — query sanitization, injection guard, intent routing")
    ap.add_argument("--query", help="single query to evaluate")
    ap.add_argument("--batch", help="file with one query per line")
    ap.add_argument("--load-ml", action="store_true",
                    help="load the ONNX injection classifier (one-time "
                         "~120-200 MB download; adds ~0.1-0.5 s per query)")
    ap.add_argument("--ml", choices=("off", "flag", "enforce"), default=None,
                    help="override ml_mode (default: flag)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    if not args.query and not args.batch:
        ap.error("provide --query or --batch")

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    try:
        cfg = GuardConfig()
        if args.ml:
            cfg.ml_mode = args.ml
        cfg.validate()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    classifier = try_load_classifier() if args.load_ml else None
    guard = GuardPipeline(cfg, classifier=classifier)

    if args.query:
        plan = guard.run(args.query)
        print(json.dumps(asdict(plan), indent=2, ensure_ascii=False)
              if args.json else _render(plan))
        return 0

    lines = [l.strip() for l in open(args.batch, encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    print(f"{'action':<9} {'ml':>5}  query")
    for line in lines:
        plan = guard.run(line)
        ml = f"{plan.ml_score:.2f}" if plan.ml_score is not None else "-"
        print(f"{plan.action:<9} {ml:>5}  {line[:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())