# stage6_generate/__main__.py
import argparse
import json
import logging
import sys

from stage4_guard.guard import GuardConfig, GuardPipeline
from stage5_retrieve.rerankers import make_reranker
from stage5_retrieve.retriever import HybridRetriever

from .generators import DEFAULT_MODEL, DEFAULT_OLLAMA_URL, OllamaGenerator
from .pipeline import Answer, GenerationConfig, GenerationPipeline
from .session import ChatSession

log = logging.getLogger("stage6.cli")


def render_sources(answer: Answer) -> str:
    if not answer.citations:
        return ""
    lines = ["", "Sources:"]
    for c in answer.citations:
        sec = f", {c['section']}" if c["section"] else ""
        lines.append(f"  [{c['marker']}] {c['title']}, {c['pages']}{sec}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="stage6_generate",
        description="Stage 6 — grounded generation with verified citations")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--index", default="artifacts/index")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--url", default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--context-tokens", type=int, default=3500)
    ap.add_argument("--history-tokens", type=int, default=800)
    ap.add_argument("--max-output", type=int, default=600)
    ap.add_argument("--session",
                    help="session id for persistent memory (default: ephemeral)")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    question = " ".join(args.question)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    session = ChatSession(session_id=args.session,
                          max_history_tokens=args.history_tokens,
                          store_dir="artifacts/sessions" if args.session else None)
    try:
        retriever = HybridRetriever.load(args.index, reranker=make_reranker("auto"))
        generator = OllamaGenerator(model=args.model, url=args.url,
                                    temperature=args.temperature)
        pipeline = GenerationPipeline(
            retriever, generator, guard=GuardPipeline(GuardConfig()),
            session=session,
            config=GenerationConfig(max_context_tokens=args.context_tokens,
                                    max_history_tokens=args.history_tokens,
                                    max_output_tokens=args.max_output,
                                    top_n=args.top))
        if args.json or args.no_stream:
            answer = pipeline.ask(question)
        else:
            print()
            answer = pipeline.ask(question,
                                  on_delta=lambda d: print(d, end="", flush=True))
            print()
    except Exception as exc:
        log.error("%s", exc)
        return 1

    if args.json:
        print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
        return 0
    if not args.no_stream:              # streamed text already printed above
        tail = render_sources(answer)
    else:
        print(answer.reply)
        tail = render_sources(answer)
    if tail:
        print(tail)
    for w in answer.warnings:
        print(f"warning: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())