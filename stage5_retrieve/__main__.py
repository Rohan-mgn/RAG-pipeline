# stage5_retrieve/__main__.py
import argparse
import json
import logging
import sys
from dataclasses import asdict

from stage4_guard.guard import GuardConfig, GuardPipeline
from stage4_guard.mlclassifier import try_load_classifier

from .errors import ConfigError
from .rerankers import DEFAULT_RERANK_MODEL, make_reranker
from .retriever import (AbstentionPolicy, HybridRetriever, RetrieverConfig,
                        RetrievalResult)


def render(result: RetrievalResult) -> str:
    lines = [f"query:   {result.query!r}",
             f"action:  {result.action.upper()}"]
    if result.abstained:
        lines.append(f"reply:   {result.suggested_reply}")
        lines.extend(f"  reason: {r}" for r in result.abstention_reasons)
    for c in result.chunks:
        rr = f"{c.rerank_score:.3f}" if c.rerank_score is not None else "  -  "
        dd = f"{c.dense_score:.3f}" if c.dense_score is not None else "  -  "
        ss = f"{c.sparse_score:.2f}" if c.sparse_score is not None else "  -  "
        lines.append(f"{c.final_rank}. [rerank {rr} | dense {dd} | bm25 {ss}] "
                     f"{c.citation}")
        if c.heading_path:
            lines.append(f"     {c.heading_path}")
        lines.append(f"     {c.text[:220].replace(chr(10), ' ')}")
    if result.warnings:
        lines.append(f"warnings: {'; '.join(result.warnings)}")
    lines.append(f"legs:    {result.leg_counts}")
    lines.append(f"timings: {result.timings_ms}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="stage5_retrieve",
        description="Stage 5 — hybrid retrieval (dense + BM25 + RRF + rerank) "
                    "with calibrated abstention")
    ap.add_argument("query", nargs="+", help="the user question")
    ap.add_argument("--index", default="artifacts/index")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--k-dense", type=int, default=20)
    ap.add_argument("--k-sparse", type=int, default=20)
    ap.add_argument("--fused-k", type=int, default=12)
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--no-sparse", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-abstain", action="store_true")
    ap.add_argument("--rerank-mode", choices=("auto", "flashrank", "none"),
                    default="auto")
    ap.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    ap.add_argument("--dense-floor", type=float, default=0.65)
    ap.add_argument("--rerank-floor", type=float, default=0.30)
    ap.add_argument("--doc-id", help="restrict retrieval to one document")
    ap.add_argument("--title", help="restrict retrieval by document title")
    ap.add_argument("--page-min", type=int)
    ap.add_argument("--page-max", type=int)
    ap.add_argument("--load-ml", action="store_true",
                    help="enable the ONNX injection classifier in the guard")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    query = " ".join(args.query)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    guard = GuardPipeline(GuardConfig(),
                          classifier=try_load_classifier() if args.load_ml else None)
    plan = guard.run(query)
    if not plan.searchable:
        print(f"action: {plan.action.upper()} ({plan.intent})")
        print(f"reply:  {plan.intent_reply}")
        return 0

    where = {}
    if args.doc_id:
        where["doc_id"] = args.doc_id
    if args.title:
        where["title"] = args.title
    page_range = None
    if args.page_min is not None or args.page_max is not None:
        page_range = (args.page_min or 0, args.page_max or 10**9)

    try:
        retriever = HybridRetriever.load(
            args.index,
            reranker=make_reranker(args.rerank_mode, model_name=args.rerank_model),
            policy=AbstentionPolicy(dense_floor=args.dense_floor,
                                    rerank_floor=args.rerank_floor),
            config=RetrieverConfig(
                k_dense=args.k_dense, k_sparse=args.k_sparse,
                fused_k=args.fused_k, top_n=args.top,
                use_dense=not args.no_dense, use_sparse=not args.no_sparse,
                use_rerank=not args.no_rerank, abstain=not args.no_abstain))
        result = retriever.run(plan.cleaned_query, where=where or None,
                               page_range=page_range)
    except (ConfigError, Exception) as exc:
        logging.getLogger("stage5.cli").error("%s", exc)
        return 1

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
          if args.json else render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())