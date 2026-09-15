from dataclasses import dataclass

from stage2_chunk.tokens import TokenEstimator, default_token_estimator


@dataclass
class ContextBlock:
    index: int            # 1-based; this IS the citation marker number
    chunk_id: str
    doc_id: str
    title: str
    section: str
    kind: str
    page_start: int
    page_end: int
    text: str
    tokens: int

    def render(self) -> str:
        pages = (f"p.{self.page_start}" if self.page_start == self.page_end
                 else f"pp.{self.page_start}-{self.page_end}")
        head = f"[{self.index}] {self.title}"
        if self.section:
            head += f" — {self.section}"
        head += f" ({pages})"
        return f"{head}\n{self.text}"


@dataclass
class PackedContext:
    blocks: list[ContextBlock]
    dropped_ranks: list[int]
    budget: int
    used_tokens: int

    @property
    def citation_map(self) -> dict[int, ContextBlock]:
        return {b.index: b for b in self.blocks}

    def render(self) -> str:
        return "\n\n".join(b.render() for b in self.blocks)


def pack_context(chunks, budget: int,
                 estimator: TokenEstimator = default_token_estimator) -> PackedContext:
    """Rank-priority token packing.

    Guarantees:
    - rank 1 (the reranker's best chunk) is ALWAYS block [1], even if it alone
      exceeds the budget (Stage 2's 256-token cap makes that unreachable in
      practice, but the guarantee is structural, not statistical);
    - no block is ever silently truncated — a chunk that doesn't fit is
      skipped whole, and its rank is recorded in dropped_ranks;
    - used_tokens is estimator-based, so the caller can enforce the num_ctx
      bound downstream."""
    blocks: list[ContextBlock] = []
    dropped: list[int] = []
    used = 0
    for rank, c in enumerate(chunks, start=1):
        section = c.heading_path or ""
        body = f"{c.title} — {section}\n{c.text}" if section else f"{c.title}\n{c.text}"
        toks = estimator(body)
        if blocks and used + toks > budget:
            dropped.append(rank)
            continue
        blocks.append(ContextBlock(index=len(blocks) + 1, chunk_id=c.chunk_id,
                                   doc_id=c.doc_id, title=c.title, section=section,
                                   kind=c.kind, page_start=c.page_start,
                                   page_end=c.page_end, text=c.text, tokens=toks))
        used += toks
    if chunks:
        assert blocks and blocks[0].chunk_id == chunks[0].chunk_id, \
            "packing invariant violated: rank-1 chunk must be block [1]"
    return PackedContext(blocks=blocks, dropped_ranks=dropped,
                         budget=budget, used_tokens=used)