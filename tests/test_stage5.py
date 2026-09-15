# tests/test_stage5.py
import json
import logging

import pytest

from stage1_extract.artifacts import sha256_text, utc_now_iso
from stage2_chunk.artifacts import Chunk, ChunkingArtifact
from stage3_embed.embedders import HashEmbedder
from stage3_embed.errors import EmbedderMismatchError
from stage3_embed.pipeline import EmbeddingPipeline

from stage5_retrieve.bm25 import BM25Index, tokenize
from stage5_retrieve.fusion import rrf_fuse
from stage5_retrieve.rerankers import NoopReranker
from stage5_retrieve.retriever import HybridRetriever, RetrieverConfig


def write_chunks(tmp_path, doc_id, texts, title):
    chunks = [Chunk(chunk_id=sha256_text(f"{doc_id}::{i}"), seq=i, doc_id=doc_id,
                    kind="text", heading_path=[], text=t,
                    page_start=i + 1, page_end=i + 1, char_count=len(t),
                    token_estimate=max(1, len(t) // 4),
                    content_hash=sha256_text(t))
              for i, t in enumerate(texts)]
    art = ChunkingArtifact(schema_version="chunking-1.0", doc_id=doc_id,
                           source_title=title, source=f"docs/{doc_id}.pdf",
                           input_content_hash=sha256_text(doc_id),
                           pipeline_version="t", config={}, config_fingerprint="cf",
                           chunking_timestamp=utc_now_iso(), chunks=chunks,
                           stats={}, warnings=[])
    d = tmp_path / "chunks"
    d.mkdir(exist_ok=True)
    art.save(d / f"{doc_id}.json")


def build(tmp_path, docs, dim=64, reranker=None, **kw):
    for doc_id, (title, texts) in docs.items():
        write_chunks(tmp_path, doc_id, texts, title)
    EmbeddingPipeline(tmp_path / "chunks", tmp_path / "index",
                      embedder=HashEmbedder(dim=dim), batch_size=4).run()
    return HybridRetriever.load(tmp_path / "index",
                                reranker=reranker or NoopReranker(), **kw)


def test_tokenize_stops_and_case():
    assert tokenize("The Shapley VALUES are what") == ["shapley", "values"]


def test_bm25_ranks_marker_and_ignores_stopwords():
    recs = [{"chunk_id": "a", "doc_id": "d", "text": "golden retriever training guide"},
            {"chunk_id": "b", "doc_id": "d", "text": "neural network optimization tips"},
            {"chunk_id": "c", "doc_id": "d", "text": "the the the the"}]
    idx = BM25Index(recs)
    hits = idx.search("golden retriever training", k=3)
    assert hits and hits[0][0] == "a"
    assert all(cid != "c" for cid, _ in hits)


def test_bm25_where_filter():
    recs = [{"chunk_id": "a", "doc_id": "d1", "text": "quantum tunneling effect"},
            {"chunk_id": "b", "doc_id": "d2", "text": "quantum tunneling effect"}]
    idx = BM25Index(recs)
    assert [cid for cid, _ in idx.search("quantum", k=5,
                                         where={"doc_id": "d2"})] == ["b"]


def test_rrf_prefers_dual_leg_hits():
    fused = rrf_fuse([["x", "a", "b"], ["y", "a", "c"]])
    assert fused["a"] > fused["x"] and fused["a"] > fused["y"]


def test_hybrid_end_to_end_hit(tmp_path):
    r = build(tmp_path, {"d" * 64: ("Test Book", [
        "gradient descent optimization basics",
        "reinforcement learning policy gradient",
        "the attention mechanism explained"])})
    result = r.run("gradient descent optimization basics")
    assert not result.abstained and result.chunks
    top = result.chunks[0]
    assert top.dense_score is not None and top.dense_score > 0.9
    assert top.title == "Test Book"
    assert result.leg_counts["dense"] > 0 and result.leg_counts["sparse"] > 0


def test_abstention_on_miss(tmp_path):
    r = build(tmp_path, {"d" * 64: ("Book", [
        "gradient descent optimization basics",
        "reinforcement learning policy gradient"])})
    result = r.run("kumquat harvest calendar lunar")
    assert result.abstained
    assert result.abstention_reasons
    assert result.suggested_reply


def test_rerank_reorders(tmp_path):
    class MarkerReranker:
        name = "marker"
        def rerank(self, query, candidates):
            return [(c, 0.9 if "gold" in c.get("text", "") else 0.05)
                    for c in candidates]
    r = build(tmp_path, {"d" * 64: ("Book", [
        "atlas filler passage about maps",
        "atlas gold standard passage"])},
              reranker=MarkerReranker())
    result = r.run("atlas")
    assert result.chunks
    assert "gold" in result.chunks[0].text
    assert result.chunks[0].rerank_score == 0.9
    assert result.chunks[0].final_rank == 1


def test_doc_scope_filter(tmp_path):
    r = build(tmp_path, {
        "a" * 64: ("Book A", ["shared topic alpha details"]),
        "b" * 64: ("Book B", ["shared topic alpha details"])})
    result = r.run("shared topic alpha details", where={"doc_id": "a" * 64})
    assert result.chunks and all(c.doc_id == "a" * 64 for c in result.chunks)


def test_page_range_filter(tmp_path):
    texts = [f"topic marker sentence number {i} about testing" for i in range(8)]
    r = build(tmp_path, {"d" * 64: ("Book", texts)})
    result = r.run("topic marker sentence", page_range=(3, 4))
    assert result.chunks
    assert all(c.page_start in (3, 4) for c in result.chunks)


def test_dense_disabled_still_retrieves(tmp_path):
    r = build(tmp_path, {"d" * 64: ("Book", ["lexically findable marker passage"])},
              config=RetrieverConfig(use_dense=False))
    result = r.run("lexically findable marker")
    assert result.chunks and result.chunks[0].dense_score is None
    assert result.leg_counts["dense"] == 0
    assert not result.abstained          # dense floor must not fire when leg off


def test_sparse_disabled_still_retrieves(tmp_path):
    r = build(tmp_path, {"d" * 64: ("Book", ["gradient descent optimization basics"])},
              config=RetrieverConfig(use_sparse=False))
    result = r.run("gradient descent optimization basics")
    assert result.chunks and result.chunks[0].sparse_score is None
    assert not result.abstained          # dense exact-match carries it


def test_embedder_mismatch_on_load(tmp_path):
    build(tmp_path, {"d" * 64: ("Book", ["one text chunk here"])})
    with pytest.raises(EmbedderMismatchError):
        HybridRetriever.load(tmp_path / "index", embedder=HashEmbedder(dim=32),
                             reranker=NoopReranker())


def test_staleness_warning(tmp_path, caplog):
    build(tmp_path, {"d" * 64: ("Book", ["one text chunk here"])})
    mp = tmp_path / "index" / "manifest.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    m["totals"]["chunks"] = 99
    mp.write_text(json.dumps(m))
    with caplog.at_level(logging.WARNING, logger="stage5.retriever"):
        HybridRetriever.load(tmp_path / "index", reranker=NoopReranker())
    assert any("STALE" in rec.message for rec in caplog.records)


def test_noop_rerank_preserves_fused_order(tmp_path):
    r = build(tmp_path, {"d" * 64: ("Book", ["alpha beta", "gamma delta"])})
    result = r.run("alpha beta")
    assert all(c.rerank_score is None for c in result.chunks)
    ranks = [c.fused_rank for c in result.chunks]
    assert ranks == sorted(ranks)