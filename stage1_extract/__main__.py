# stage1_extract/__main__.py
import argparse
import logging
import sys

from .ingest import Ingester
from .parsers import make_parser


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="stage1_extract",
        description="Stage 1 — parse PDFs into versioned extraction artifacts")
    ap.add_argument("docs", help="directory of PDFs (searched recursively)")
    ap.add_argument("--out", default="artifacts/extraction")
    ap.add_argument("--parser", choices=("auto", "docling", "pymupdf"), default="auto")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-parse even if an artifact for this doc_id exists")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    ingester = Ingester(args.out, parser=make_parser(args.parser,
                                                     do_ocr=not args.no_ocr),
                        force=args.force)
    report = ingester.ingest_directory(args.docs)
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())