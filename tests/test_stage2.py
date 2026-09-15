# tests/test_stage2.py
import pytest

from stage1_extract.artifacts import (SCHEMA_VERSION, ExtractionArtifact,
                                      ParsedPage, sha256_text, utc_now_iso)
from stage2_chunk.artifacts import ChunkingArtifact
from stage2_chunk.chunker import ChunkerConfig
from stage2_chunk.errors import ConfigError
from stage2_chunk.pipeline import ChunkingPipeline
from stage2_chunk.sentences import RegexSentenceSplitter
from stage2_chunk.structure import plain_heading_level


def make_extraction(tmp_path, pages, doc_id="d0" * 32, title="Test Doc"):
    pp = [ParsedPage(i + 1, None, None, 0, t,
                     "ok" if t.strip() else "skipped-empty", 0.9, [], [])
          for i, t in enumerate(pages)]
    art = ExtractionArtifact(
        schema_version=SCHEMA_VERSION, doc_id=doc_id,
        content_hash=sha256_text("\n".join(pages)), source="t.pdf", title=title,
        author=None, created_at=None, modified_at=None,
        ingestion_timestamp=utc_now_iso(), pipeline_version="stage1-1.0.0",
        parser="fake", pages=pp)
    d = tmp_path / "ex"
    d.mkdir(exist_ok=True)
    art.save(d / f"{doc_id}.json")
    return art


def run(tmp_path, config=None):
    return ChunkingPipeline(tmp_path / "ex", tmp_path / "out", config=config).run()


def load_out(tmp_path):
    return ChunkingArtifact.load(next((tmp_path / "out").glob("*.json")))


def test_regex_splitter_abbreviations():
    split = RegexSentenceSplitter()
    text = ("We use tools e.g. PyTorch and frameworks like Dr. Smith's library. "
            "Fig. 3 shows results. Accuracy reached 99.9 percent. It worked.")
    sents = split(text)
    assert len(sents) == 4
    assert sents[0].endswith("library.")          # no split after e.g. / Dr.
    assert sents[1] == "Fig. 3 shows results."    # no split after Fig.
    assert sents[2] == "Accuracy reached 99.9 percent."  # no split at 99.9


def test_plain_heading_level():
    assert plain_heading_level("Chapter 5") == 1
    assert plain_heading_level("Appendix B") == 1
    assert plain_heading_level("3.2 Why XAI Matters") == 2
    assert plain_heading_level("3.2.1 Deep dive") == 3
    assert plain_heading_level("2020 was a year of change") is None
    assert plain_heading_level("This is a normal sentence.") is None


def test_deterministic_ids_and_idempotent_skip(tmp_path):
    make_extraction(tmp_path, [
        " ".join(f"Body sentence {i} talks about model training steps."
                 for i in range(30))])
    r1 = run(tmp_path)
    r2 = run(tmp_path)
    assert r1.results[0].status == "ingested"
    assert r2.results[0].status == "duplicate"
    art = load_out(tmp_path)
    assert art.chunks[0].chunk_id == sha256_text(f"{art.doc_id}::0")
    assert [c.seq for c in art.chunks] == list(range(len(art.chunks)))


def test_heading_path_and_section_isolation(tmp_path):
    alpha = " ".join(f"Alpha section sentence {i} covers feature engineering basics."
                     for i in range(20))
    beta = " ".join(f"Beta section sentence {i} covers model evaluation metrics."
                    for i in range(20))
    make_extraction(tmp_path,
                    [f"# Chapter One\n\n{alpha}\n\n## 1.1 Subsection\n\n{beta}"])
    run(tmp_path)
    chunks = load_out(tmp_path).chunks
    a = [c for c in chunks if "Alpha" in c.text]
    b = [c for c in chunks if "Beta" in c.text]
    assert a and b
    assert all(c.heading_path == ["Chapter One"] for c in a)
    assert all(c.heading_path == ["Chapter One", "1.1 Subsection"] for c in b)
    assert not any("Alpha" in c.text and "Beta" in c.text for c in chunks)


def test_size_bound_hard_max(tmp_path):
    body = " ".join(f"Padding sentence {i} with plenty of additional filler words."
                    for i in range(40))
    make_extraction(tmp_path, [body])
    cfg = ChunkerConfig(target_tokens=50, max_tokens=64, min_tokens=10)
    run(tmp_path, config=cfg)
    chunks = load_out(tmp_path).chunks
    assert chunks
    assert all(c.token_estimate <= 64 for c in chunks)   # the hard guarantee


def test_no_silent_loss(tmp_path):
    sents = [f"Unique marker {i} sentence about retrieval quality." for i in range(25)]
    make_extraction(tmp_path, [" ".join(sents)])
    run(tmp_path)
    corpus = " ".join(c.text for c in load_out(tmp_path).chunks)
    assert all(s in corpus for s in sents)


def test_cross_page_stitching(tmp_path):
    make_extraction(tmp_path, [
        "The training procedure continues to",
        "converge after 100 epochs. Then evaluation begins in earnest."])
    run(tmp_path)
    chunks = load_out(tmp_path).chunks
    c = next(c for c in chunks if "continues to converge" in c.text)
    assert c.page_start == 1 and c.page_end == 2


def test_oversized_sentence_split(tmp_path):
    giant = " ".join(["word"] * 400)     # ~2000 chars, one giant "sentence"
    make_extraction(tmp_path, [f"Start here. {giant}. End there."])
    run(tmp_path)
    art = load_out(tmp_path)
    assert all(c.token_estimate <= 256 for c in art.chunks)
    corpus = " ".join(c.text for c in art.chunks)
    assert "Start here." in corpus and "End there." in corpus
    assert any(w.startswith("oversized") for w in art.warnings)


def test_config_validation():
    with pytest.raises(ConfigError):
        ChunkerConfig(target_tokens=100, max_tokens=50, min_tokens=10).validate()


def test_table_kept_atomic(tmp_path):
    page = ("Intro paragraph before the table. It explains the numbers.\n\n"
            "[TABLE]\n| Model | F1 |\n| BERT | 0.91 |\n[/TABLE]\n\n"
            "Outro paragraph after the table.")
    make_extraction(tmp_path, [page])
    run(tmp_path)
    chunks = load_out(tmp_path).chunks
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 1
    assert tables[0].text == "| Model | F1 |\n| BERT | 0.91 |"
    assert not any("| BERT" in c.text for c in chunks if c.kind == "text")


def test_tiny_trailing_chunk_merged(tmp_path):
    body = " ".join(f"Sentence number {i} about the training pipeline runs."
                    for i in range(6))
    make_extraction(tmp_path, [f"{body}\n\nTiny tail."])
    cfg = ChunkerConfig(target_tokens=70, max_tokens=140, min_tokens=45)
    run(tmp_path, config=cfg)
    chunks = load_out(tmp_path).chunks
    assert len(chunks) == 1                     # tail merged, not left as a fragment
    assert "Tiny tail." in chunks[0].text
    assert "Sentence number 5" in chunks[0].text


def test_rechunk_on_content_change(tmp_path):
    art = make_extraction(tmp_path,
                          ["Version one body text about embeddings. " * 5])
    assert run(tmp_path).results[0].status == "ingested"
    make_extraction(tmp_path,
                    ["Version two completely different text about search. " * 5],
                    doc_id=art.doc_id)          # same doc_id, new content
    assert run(tmp_path).results[0].status == "ingested"   # not 'duplicate'
    assert "Version two" in load_out(tmp_path).chunks[0].text


# --- heading-pollution regression tests (Patch B, universal guard) ---------

def test_notebook_code_comment_not_heading():
    from stage2_chunk.structure import page_blocks
    blocks = page_blocks("# @title Setup from google.colab import auth\nx = 1", 1,
                         detect_headings=True, trust_markdown=False)
    assert all(b.kind != "heading" for b in blocks)


def test_short_markdown_heading_kept_without_trust():
    from stage2_chunk.structure import page_blocks
    blocks = page_blocks("# Building Custom ML Models on Vertex AI", 1,
                         detect_headings=True, trust_markdown=False)
    assert blocks[0].kind == "heading"
    assert blocks[0].text == "Building Custom ML Models on Vertex AI"


def test_toc_page_suppressed():
    from stage2_chunk.structure import page_blocks
    page = "\n".join(f"Chapter {i} Some Topic {i*20}" for i in range(1, 7))
    assert not any(b.kind == "heading" for b in page_blocks(page, 1))


def test_docling_junk_titles_also_demoted():
    from stage2_chunk.structure import extract_blocks
    pp = [ParsedPage(1, None, None, 0,
                     "# @title Setup from google.colab import auth",
                     "ok", 0.9, [], [])]
    art = ExtractionArtifact(
        schema_version=SCHEMA_VERSION, doc_id="z" * 64,
        content_hash=sha256_text("z"), source="z.pdf", title="Z", author=None,
        created_at=None, modified_at=None, ingestion_timestamp=utc_now_iso(),
        pipeline_version="t", parser="docling", pages=pp)
    blocks, meta = extract_blocks(art)
    assert meta["headings"] == []      # docling junk demoted — guard is universal