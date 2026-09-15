import re
from dataclasses import asdict, dataclass

from stage1_extract.artifacts import sha256_text, utc_now_iso

from .artifacts import CHUNK_SCHEMA_VERSION, Chunk, ChunkingArtifact
from .errors import ChunkingError, ConfigError, PathologicalInputError
from .structure import Block, extract_blocks, page_at, stitch_blocks
from .tokens import TokenEstimator, default_token_estimator

MAX_CHUNKS = 50_000


@dataclass
class ChunkerConfig:
    target_tokens: int = 160      # aim: flush when the next sentence would exceed
    max_tokens: int = 256         # hard ceiling — tested, never exceeded
    min_tokens: int = 30          # trailing chunks below this merge backward
    overlap_sentences: int = 1    # context carry-over across chunk boundaries
    detect_headings: bool = True  # plain-text heading heuristic (non-docling pages)

    def validate(self) -> None:
        c = self
        if not (0 < c.min_tokens < c.target_tokens < c.max_tokens):
            raise ConfigError(f"need 0 < min < target < max tokens, got "
                              f"{c.min_tokens}/{c.target_tokens}/{c.max_tokens}")
        if not (0 <= c.overlap_sentences <= 5):
            raise ConfigError("overlap_sentences must be within 0..5")


@dataclass
class Unit:
    """One sentence (or table row): the atomic unit of chunk assembly."""
    text: str
    page: int
    tokens: int


def _split_oversized(text: str, max_tokens: int, estimator: TokenEstimator,
                     warnings: list[str]) -> list[str]:
    """Word-boundary split for units bigger than max_tokens; character split
    as the last resort for monster words. Always audible — a warning is recorded."""
    pieces: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for w in text.split():
        t = estimator(w + " ")
        if t > max_tokens:                       # single monster word
            per_char = estimator(w) / max(len(w), 1)
            plen = max(1, int(max_tokens / per_char))
            for i in range(0, len(w), plen):
                piece = w[i:i + plen]
                while estimator(piece) > max_tokens and len(piece) > 1:
                    piece = piece[:-1]
                pieces.append(piece)
            continue
        if cur and cur_tok + t > max_tokens:
            pieces.append(" ".join(cur))
            cur, cur_tok = [], 0
        cur.append(w)
        cur_tok += t
    if cur:
        pieces.append(" ".join(cur))
    warnings.append(f"oversized unit split on word/char boundaries: {text[:60]}…")
    return pieces


def make_units(block: Block, splitter, estimator: TokenEstimator,
               config: ChunkerConfig, warnings: list[str]) -> list[Unit]:
    sents = splitter(block.text)
    if not sents and block.text.strip():
        sents = [block.text.strip()]
    prefix = "- " if block.kind == "list_item" else ""
    units: list[Unit] = []
    pos, first = 0, True
    for s in sents:
        idx = block.text.find(s, pos)
        if idx < 0:
            idx = pos
        page = page_at(block.marks, idx)
        text = prefix + s if first else s
        t = estimator(text)
        if t > config.max_tokens:
            for piece in _split_oversized(text, config.max_tokens, estimator, warnings):
                units.append(Unit(piece, page, estimator(piece)))
        else:
            units.append(Unit(text, page, t))
        first = False
        pos = idx + len(s)
    return units


def table_units(block: Block, config: ChunkerConfig, estimator: TokenEstimator,
                warnings: list[str]) -> list[Unit]:
    units: list[Unit] = []
    for row in (r.strip() for r in block.text.split("\n")):
        if not row:
            continue
        t = estimator(row)
        if t > config.max_tokens:
            for piece in _split_oversized(row, config.max_tokens, estimator, warnings):
                units.append(Unit(piece, block.page_start, estimator(piece)))
        else:
            units.append(Unit(row, block.page_start, t))
    return units


class _Assembler:
    """Greedy sentence packing. Invariants: (1) a heading always flushes —
    chapters never blend; (2) no chunk exceeds max_tokens; (3) overlap only
    seeds within the same heading path; (4) tables never mix with prose."""

    def __init__(self, doc_id: str, config: ChunkerConfig,
                 estimator: TokenEstimator, warnings: list[str]) -> None:
        self.doc_id = doc_id
        self.cfg = config
        self.est = estimator
        self.warnings = warnings
        self.pending: list[dict] = []
        self.current: list[Unit] = []
        self.stack: list[tuple[int, str]] = []       # (level, heading text)
        self.last_flush: dict | None = None

    def _tok(self, units: list[Unit]) -> int:
        return sum(u.tokens for u in units)

    def _path(self) -> list[str]:
        return [t for _, t in self.stack]

    def heading(self, block: Block) -> None:
        self._flush()
        self.last_flush = None          # no overlap across section boundaries
        while self.stack and self.stack[-1][0] >= block.level:
            self.stack.pop()
        self.stack.append((block.level, block.text))

    def sentences(self, units: list[Unit]) -> None:
        for u in units:
            if self.current and self._tok(self.current) + u.tokens > self.cfg.target_tokens:
                self._flush()
            if self.current and self._tok(self.current) + u.tokens > self.cfg.max_tokens:
                self._flush()
            if not self.current:
                seed = self._seed()
                if seed and self._tok(seed) + u.tokens <= self.cfg.max_tokens:
                    self.current = list(seed)
            self.current.append(u)

    def _seed(self) -> list[Unit]:
        if (self.cfg.overlap_sentences <= 0 or self.last_flush is None
                or self.last_flush.get("path") != self._path()):
            return []
        tail = self.last_flush["units"][-self.cfg.overlap_sentences:]
        budget = max(1, self.cfg.target_tokens // 4)
        out, spent = [], 0
        for u in reversed(tail):
            if spent + u.tokens > budget:
                break
            out.insert(0, u)
            spent += u.tokens
        return out

    def table(self, units: list[Unit]) -> None:
        self._flush()
        self.last_flush = None
        cur: list[Unit] = []
        for u in units:
            if cur and self._tok(cur) + u.tokens > self.cfg.max_tokens:
                self.pending.append({"kind": "table", "units": cur,
                                     "path": self._path(), "joiner": "\n"})
                cur = []
            cur.append(u)
        if cur:
            self.pending.append({"kind": "table", "units": cur,
                                 "path": self._path(), "joiner": "\n"})

    def _flush(self) -> None:
        if not self.current:
            return
        entry = {"kind": "text", "units": self.current,
                 "path": self._path(), "joiner": " "}
        self.pending.append(entry)
        self.last_flush = entry
        self.current = []

    def finalize(self) -> list[Chunk]:
        self._flush()
        out: list[dict] = []
        for e in self.pending:
            # merge tiny trailing chunks backward (same section, fits in max)
            if (out and e["kind"] == "text" and out[-1]["kind"] == "text"
                    and out[-1]["path"] == e["path"]
                    and self._tok(e["units"]) < self.cfg.min_tokens
                    and self._tok(out[-1]["units"]) + self._tok(e["units"])
                        <= self.cfg.max_tokens):
                prev, cur = out[-1]["units"], e["units"]
                # drop the longest prefix of cur that is already the suffix
                # of prev (overlap-seeded units)
                k = 0
                for m in range(1, min(len(prev), len(cur)) + 1):
                    if [u.text for u in prev[-m:]] == [u.text for u in cur[:m]]:
                        k = m
                prev.extend(cur[k:])
            else:
                out.append(e)
        chunks: list[Chunk] = []
        for seq, e in enumerate(out):
            text = e["joiner"].join(u.text for u in e["units"]).strip()
            pages = [u.page for u in e["units"]]
            chunks.append(Chunk(
                chunk_id=sha256_text(f"{self.doc_id}::{seq}"), seq=seq,
                doc_id=self.doc_id, kind=e["kind"], heading_path=e["path"],
                text=text, page_start=min(pages), page_end=max(pages),
                char_count=len(text), token_estimate=self._tok(e["units"]),
                content_hash=sha256_text(text)))
        if len(chunks) > MAX_CHUNKS:
            raise PathologicalInputError(
                f"{len(chunks)} chunks exceeds cap {MAX_CHUNKS} — refusing to write")
        return chunks


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def verify_lossless(all_units: list[Unit], chunks: list[Chunk]) -> None:
    """Hard gate: every sentence that entered assembly appears in some chunk.
    Runs on every document, every run — not just in tests."""
    corpus = _norm(" ".join(c.text for c in chunks))
    missing = [u.text for u in all_units if _norm(u.text) not in corpus]
    if missing:
        raise ChunkingError(
            f"silent loss: {len(missing)}/{len(all_units)} units missing; "
            f"first: {missing[0][:80]!r}")


def verify_integrity(doc_id: str, chunks: list[Chunk]) -> None:
    for i, c in enumerate(chunks):
        if c.seq != i or c.chunk_id != sha256_text(f"{doc_id}::{i}"):
            raise ChunkingError(f"integrity: chunk {i} id/seq mismatch")
        if c.page_start > c.page_end:
            raise ChunkingError(
                f"integrity: chunk {i} page range {c.page_start}-{c.page_end}")
    if len({c.chunk_id for c in chunks}) != len(chunks):
        raise ChunkingError("integrity: duplicate chunk ids")


def chunk_document(art, config: ChunkerConfig, splitter,
                   estimator: TokenEstimator = default_token_estimator,
                   pipeline_version: str = "dev",
                   config_fingerprint: str = "") -> ChunkingArtifact:
    config.validate()
    warnings: list[str] = []
    blocks, struct_audit = extract_blocks(art, config.detect_headings)
    blocks, stitch_stats = stitch_blocks(blocks)
    asm = _Assembler(art.doc_id, config, estimator, warnings)
    all_units: list[Unit] = []
    for b in blocks:
        if b.kind == "heading":
            asm.heading(b)
        elif b.kind == "table":
            units = table_units(b, config, estimator, warnings)
            all_units.extend(units)
            asm.table(units)
        else:
            units = make_units(b, splitter, estimator, config, warnings)
            all_units.extend(units)
            asm.sentences(units)
    chunks = asm.finalize()
    if not chunks:
        raise ChunkingError(
            f"{art.doc_id[:12]}: no chunks produced from {len(blocks)} blocks")
    verify_lossless(all_units, chunks)
    verify_integrity(art.doc_id, chunks)

    toks = [c.token_estimate for c in chunks]
    stats = {"chunks": len(chunks),
             "text_chunks": sum(c.kind == "text" for c in chunks),
             "table_chunks": sum(c.kind == "table" for c in chunks),
             "tokens_total": sum(toks),
             "tokens_avg": round(sum(toks) / len(toks)),
             "tokens_min": min(toks), "tokens_max": max(toks),
             "pages": len(art.pages),
             "headings": len(struct_audit["headings"]),
             "stitched_paragraphs": stitch_stats["stitched"],
             "dehyphenated_joins": stitch_stats["dehyphenated"],
             "oversized_units_split": sum(w.startswith("oversized") for w in warnings)}
    audit = {"headings": struct_audit["headings"][:500],
             "heading_detection": "markdown+heuristic" if config.detect_headings
                                  else "markdown-only",
             "splitter": splitter.name}
    return ChunkingArtifact(
        schema_version=CHUNK_SCHEMA_VERSION, doc_id=art.doc_id,
        source_title=art.title, source=art.source,
        input_content_hash=art.content_hash, pipeline_version=pipeline_version,
        config={**asdict(config), "splitter": splitter.name},
        config_fingerprint=config_fingerprint,
        chunking_timestamp=utc_now_iso(), chunks=chunks,
        stats=stats, warnings=warnings, audit=audit)