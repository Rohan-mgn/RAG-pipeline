# stage3_embed/__main__.py
import argparse
import json
import logging
import sys
from pathlib import Path

from .embedders import HashEmbedder, make_embedder
from .errors import EmbedderMismatchError, Stage3Error
from .pipeline import MANIFEST_FILE, EmbeddingPipeline
from .store import NumpyVectorStore

log = logging.getLogger("stage3.cli")


def _embedder_for(sig: str):
    if sig and sig.startswith("hash-test:"):
        return HashEmbedder(dim=int(sig.split(":", 1)[1]))
    return make_embedder("fastembed")


def run_search(out_dir: str, query: str, k: int) -> int:
    out = Path(out_dir)
    store = NumpyVectorStore(out)
    if store.count() == 0:
        print(f"index at {out} is empty — run ingestion first")
        return 1
    manifest = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    sig = manifest.get("embedder_signature")
    if sig and sig.startswith("hash-test"):
        print("!! non-semantic (hash-test) index — results are meaningless "
              "except as a wiring smoke test\n")
    embedder = _embedder_for(sig)
    if sig and embedder.signature != sig:
        raise EmbedderMismatchError(
            f"index built with {sig}, query embedder is {embedder.signature}")
    hits = store.search(embedder.embed_query(query), k=k)
    print(f"query: {query!r}  (k={k}, embedder={sig}, corpus={store.count()})\n")
    for rank, h in enumerate(hits, 1):
        m = h.record
        head = m.get("heading_path") or "(no heading)"
        print(f"{rank}. [{h.score:.3f}] {m.get('title', '?')} "
              f"— p.{m.get('page_start')}-{m.get('page_end')} ({m.get('kind')})")
        print(f"     {head}")
        print(f"     {(m.get('text') or '')[:180].replace(chr(10), ' ')}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="stage3_embed",
        description="Stage 3 — embed chunks and build the exact-search vector index")
    ap.add_argument("indir", nargs="?", default="artifacts/chunking",
                    help="Stage 2 output (default: artifacts/chunking)")
    ap.add_argument("--out", default="artifacts/index")
    ap.add_argument("--embedder", choices=("auto", "fastembed", "hash"),
                    default="auto", help="'hash' is NON-semantic — explicit only")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cost-per-1k", type=float, default=0.0,
                    help="USD per 1k tokens — 0 for local models (default)")
    ap.add_argument("--force", action="store_true", help="re-embed current docs")
    ap.add_argument("--rebuild", action="store_true",
                    help="wipe index and re-embed EVERYTHING (required when "
                         "switching embedding models)")
    ap.add_argument("--prune", action="store_true",
                    help="delete vectors whose chunking artifact no longer exists")
    ap.add_argument("--search", metavar="QUERY",
                    help="search the built index and exit (no ingestion)")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    try:
        if args.search:
            return run_search(args.out, args.search, args.k)
        pipeline = EmbeddingPipeline(
            args.indir, args.out, embedder=make_embedder(args.embedder),
            batch_size=args.batch_size, force=args.force, rebuild=args.rebuild,
            prune=args.prune, cost_per_1k_tokens=args.cost_per_1k)
        report = pipeline.run()
    except Stage3Error as exc:
        log.error("%s", exc)
        return 1
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())