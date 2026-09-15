# tests/test_stage3.py
import pytest

from stage1_extract.artifacts import sha256_text, utc_now_iso
from stage2_chunk.artifacts import Chunk, ChunkingArtifact
from stage3_embed.embedders import (DOCUMENT_PREFIX, HashEmbedder, RetryPolicy,
                                    apply_task_prefix, embed_batched)
from stage3_embed.errors import (EmbedderMismatchError, StoreError,
                                 TransientEmbeddingError)
from stage3_embed.metadata import sanitize_metadata
from stage3_embed.pipeline import EmbeddingPipeline
from stage3_embed.store import NumpyVectorStore


def make_chunk_artifact(tmp_path, doc_id, texts, title="Test Book",
                        heading_path=("Chapter 1", "1.1 Detail")):
    chunks = [Chunk(chunk_id=sha256_text(f"{doc_id}::{i}"), seq=i, doc_id=doc_id,
                    kind="text", heading_path=list(heading_path),
                    text=t, page_start=i + 1, page_end=i + 1, char_count=len(t),
                    token_estimate=max(1, len(t) // 4), content_hash=sha256_text(t))
              for i, t in enumerate(texts)]
    art = ChunkingArtifact(schema_version="chunking-1.0", doc_id=doc_id,
                           source_title=title, source=f"docs/{doc_id}.pdf",
                           input_content_hash=sha256_text(doc_id),
                           pipeline_version="stage2-1.0.0", config={},
                           config_fingerprint="cf",
                           chunking_timestamp=utc_now_iso(), chunks=chunks,
                           stats={}, warnings=[])
    d = tmp_path / "chunks"
    d.mkdir(exist_ok=True)
    art.save(d / f"{doc_id}.json")
    return art


def run_pipe(tmp_path, embedder=None, retry=None, **kw):
    return EmbeddingPipeline(tmp_path / "chunks", tmp_path / "index",
                             embedder=embedder or HashEmbedder(dim=32),
                             batch_size=4, retry=retry, **kw)


def test_sanitize_metadata():
    m = sanitize_metadata({"a": None, "b": "x", "c": 1, "d": 2.5, "e": True,
                           "f": {"nested": 1}, "g": ["Chapter 1", "1.1"], "h": []})
    assert m == {"b": "x", "c": 1, "d": 2.5, "e": True, "g": "Chapter 1 | 1.1"}
    assert all(isinstance(v, (str, int, float, bool)) for v in m.values())


def test_task_prefix_idempotent():
    assert apply_task_prefix("search_query: hi", DOCUMENT_PREFIX) == "search_document: hi"
    assert apply_task_prefix("plain", DOCUMENT_PREFIX) == "search_document: plain"


def test_hash_embedder_deterministic():
    e = HashEmbedder(dim=16)
    assert e.embed_query("same text") == e.embed_query("same text")
    assert e.embed_query("same text") != e.embed_query("different text")
    assert len(e.embed_query("x")) == 16


def test_retry_transient_then_success():
    class Flaky(HashEmbedder):
        def __init__(self):
            super().__init__(dim=8)
            self.calls = 0
        def embed_documents(self, texts):
            self.calls += 1
            if self.calls <= 2:
                raise TransientEmbeddingError("flaky")
            return super().embed_documents(texts)
    e = Flaky()
    out = embed_batched(e, ["a", "b"], batch_size=2,
                        policy=RetryPolicy(attempts=4, base_delay=0.0, max_delay=0.0))
    assert e.calls == 3 and len(out) == 2


def test_fail_fast_non_retryable():
    class Broken(HashEmbedder):
        def __init__(self):
            super().__init__(dim=8)
            self.calls = 0
        def embed_documents(self, texts):
            self.calls += 1
            raise ValueError("bad request")
    e = Broken()
    with pytest.raises(ValueError):
        embed_batched(e, ["a"], batch_size=1,
                      policy=RetryPolicy(attempts=4, base_delay=0.0, max_delay=0.0))
    assert e.calls == 1     # a guaranteed-400 is never retried


def test_deterministic_skip(tmp_path):
    make_chunk_artifact(tmp_path, "a" * 64, [f"alpha sentence {i}" for i in range(6)])
    r1 = run_pipe(tmp_path).run()
    r2 = run_pipe(tmp_path).run()
    assert [r.status for r in r1.results] == ["indexed"]
    assert [r.status for r in r2.results] == ["duplicate"]
    assert NumpyVectorStore(tmp_path / "index").count() == 6


def test_replace_semantics_shrink_case(tmp_path):
    doc = "a" * 64
    make_chunk_artifact(tmp_path, doc, [f"alpha unique sentence {i}" for i in range(50)])
    run_pipe(tmp_path).run()
    make_chunk_artifact(tmp_path, doc, [f"alpha unique sentence {i}" for i in range(10)])
    report = run_pipe(tmp_path).run()
    assert [r.status for r in report.results] == ["indexed"]
    store = NumpyVectorStore(tmp_path / "index")
    assert store.count(doc) == 10                       # no stale orphans survive
    assert sorted(m["text"].rsplit(" ", 1)[-1]
                  for m in store.iter_records(doc)) == [str(i) for i in range(10)]


def test_embed_failure_preserves_old_index(tmp_path):
    doc = "e" * 64
    make_chunk_artifact(tmp_path, doc, ["original content here"])
    run_pipe(tmp_path).run()
    make_chunk_artifact(tmp_path, doc, ["changed content here"])

    class Exploding(HashEmbedder):
        def embed_documents(self, texts):
            raise TransientEmbeddingError("backend down")

    report = run_pipe(tmp_path, embedder=Exploding(dim=32),
                      retry=RetryPolicy(attempts=2, base_delay=0.0,
                                        max_delay=0.0)).run()
    assert report.results[0].status == "failed"
    store = NumpyVectorStore(tmp_path / "index")
    assert store.count(doc) == 1                        # old vectors untouched:
    assert "original content here" in [m["text"]          # embed-first ordering
                                       for m in store.iter_records(doc)]


def test_embedder_pinning(tmp_path):
    make_chunk_artifact(tmp_path, "p" * 64, ["pinning test content"])
    run_pipe(tmp_path).run()
    with pytest.raises(EmbedderMismatchError):
        run_pipe(tmp_path, embedder=HashEmbedder(dim=64))   # different model/dim


def test_rebuild_wipes(tmp_path):
    make_chunk_artifact(tmp_path, "a" * 64, ["alpha one", "alpha two"])
    make_chunk_artifact(tmp_path, "b" * 64, ["beta one"])
    run_pipe(tmp_path).run()
    report = run_pipe(tmp_path, rebuild=True).run()
    assert [r.status for r in report.results] == ["indexed", "indexed"]
    assert NumpyVectorStore(tmp_path / "index").count() == 3


def test_prune_removes_missing_docs(tmp_path):
    make_chunk_artifact(tmp_path, "a" * 64, ["alpha one", "alpha two"])
    make_chunk_artifact(tmp_path, "b" * 64, ["beta one"])
    run_pipe(tmp_path).run()
    (tmp_path / "chunks" / ("b" * 64 + ".json")).unlink()
    report = run_pipe(tmp_path, prune=True).run()
    statuses = {r.doc_id: r.status for r in report.results}
    assert statuses["b" * 64] == "pruned"
    store = NumpyVectorStore(tmp_path / "index")
    assert store.count() == 2 and store.count("b" * 64) == 0


def test_search_filter_by_doc(tmp_path):
    make_chunk_artifact(tmp_path, "a" * 64, ["alpha content one", "alpha content two"])
    make_chunk_artifact(tmp_path, "b" * 64, ["beta content one"])
    run_pipe(tmp_path).run()
    store = NumpyVectorStore(tmp_path / "index")
    emb = HashEmbedder(dim=32)
    hits = store.search(emb.embed_query("beta content one"), k=5,
                        where={"doc_id": "a" * 64})
    assert hits and all(h["doc_id"] == "a" * 64 for h in hits)


def test_exact_self_match(tmp_path):
    t = "the shapley value assigns credit fairly among features"
    make_chunk_artifact(tmp_path, "c" * 64, [t, "unrelated filler text about taxes"],
                        heading_path=())
    run_pipe(tmp_path).run()
    store = NumpyVectorStore(tmp_path / "index")
    hits = store.search(HashEmbedder(dim=32).embed_query(t), k=2)
    assert hits[0]["text"] == t and hits[0].score > 0.99


def test_upsert_guards(tmp_path):
    store = NumpyVectorStore(tmp_path)
    rec = [{"chunk_id": "a", "doc_id": "d", "text": "t"}]
    with pytest.raises(StoreError):                    # count mismatch
        store.upsert(rec, [[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(StoreError):                    # duplicate ids in batch
        store.upsert([rec[0], dict(rec[0])], [[0.1, 0.2], [0.1, 0.2]])
    store.upsert(rec, [[0.1, 0.2]])
    with pytest.raises(StoreError):                    # mixed dimensions
        store.upsert([{"chunk_id": "b", "doc_id": "d", "text": "u"}],
                     [[0.1, 0.2, 0.3]])
    assert store.count() == 1