# stage2_chunk/__main__.py
import argparse
import logging
import sys

from .chunker import ChunkerConfig
from .pipeline import ChunkingPipeline


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="stage2_chunk",
        description="Stage 2 — chunk extraction artifacts into citation-grade chunks")
    ap.add_argument("indir", help="artifacts/extraction (Stage 1 output)")
    ap.add_argument("--out", default="artifacts/chunking")
    ap.add_argument("--target", type=int, default=160)
    ap.add_argument("--max", type=int, default=256)
    ap.add_argument("--min", type=int, default=30)
    ap.add_argument("--overlap", type=int, default=1)
    ap.add_argument("--splitter", choices=("auto", "regex", "spacy"), default="auto")
    ap.add_argument("--no-heading-detect", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-chunk even if artifacts are current")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    cfg = ChunkerConfig(target_tokens=args.target, max_tokens=args.max,
                        min_tokens=args.min, overlap_sentences=args.overlap,
                        detect_headings=not args.no_heading_detect)
    report = ChunkingPipeline(args.indir, args.out, config=cfg,
                              splitter=args.splitter, force=args.force).run()
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())