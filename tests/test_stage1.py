# tests/test_stage1.py
import json
from pathlib import Path

import pytest

from stage1_extract.artifacts import ExtractionArtifact
from stage1_extract.errors import IncompatibleArtifactError, QualityGateError
from stage1_extract.ingest import Ingester
from stage1_extract.parsers import RawDoc, RawPage
from stage1_extract.pdfdates import normalize_pdf_date


class FakeParser:
    name = "fake"
    def __init__(self):
        self.docs = {}
    def parse(self, path):
        return self.docs[Path(path).name]


def make_doc(page_texts, **meta):
    return RawDoc(pages=[RawPage(i + 1, 612.0, 792.0, 0, t)
                         for i, t in enumerate(page_texts)], **meta)


@pytest.fixture
def parser():
    return FakeParser()


def test_pdf_date_normalization():
    assert normalize_pdf_date("D:20230211100457+01'00'") == "2023-02-11T10:04:57+01:00"
    assert normalize_pdf_date("D:20230211100457Z") == "2023-02-11T10:04:57+00:00"
    assert normalize_pdf_date("D:2023") == "2023-01-01T00:00:00+00:00"
    assert normalize_pdf_date("not a date") is None
    assert normalize_pdf_date(None) is None
    assert normalize_pdf_date("D:20231340") is None      # month 13: never a crash


def test_dedup_and_idempotency(tmp_path, parser):
    f = tmp_path / "a.pdf"; f.write_bytes(b"same-bytes")
    parser.docs["a.pdf"] = make_doc(["alpha content", "beta content"])
    ing = Ingester(tmp_path / "art", parser=parser)
    assert ing.ingest_file(f).status == "ingested"
    assert ing.ingest_file(f).status == "duplicate"
    assert len(list((tmp_path / "art").glob("*.json"))) == 1


def test_force_reparse_deterministic(tmp_path, parser):
    f = tmp_path / "d.pdf"; f.write_bytes(b"d")
    parser.docs["d.pdf"] = make_doc(["stable content page"])
    ing = Ingester(tmp_path / "art", parser=parser, force=True)
    r1, r2 = ing.ingest_file(f), ing.ingest_file(f)
    assert r1.status == r2.status == "ingested"
    assert r1.content_hash == r2.content_hash
    assert len(list((tmp_path / "art").glob("*.json"))) == 1

def test_cid_and_hyphen_cleanup(tmp_path, parser):
    f = tmp_path / "x.pdf"; f.write_bytes(b"x")
    page = ("The classifier was evaluated on a balanced dataset drawn from "
            "production traffic over several months. The result was expli-\n"
            "citly stated (cid:12)(cid:15) in the report, and an independent "
            "audit later confirmed the metrics across repeated runs and "
            "different random seeds used throughout the study.")
    parser.docs["x.pdf"] = make_doc([page])
    ing = Ingester(tmp_path / "art", parser=parser)
    r = ing.ingest_file(f)
    art = ExtractionArtifact.load(tmp_path / "art" / f"{r.doc_id}.json")
    assert "(cid:" not in art.pages[0].text
    assert "explicitly" in art.pages[0].text # verifies Edit 1 — a newline
    assert any("cid" in fl for fl in art.pages[0].flags) # mid-word fails here


def test_quality_gate_refuses_garbage(tmp_path, parser):
    f = tmp_path / "bad.pdf"; f.write_bytes(b"bad")
    parser.docs["bad.pdf"] = make_doc(["(cid:1)(cid:2)(cid:3) junk " * 10] * 5)
    ing = Ingester(tmp_path / "art", parser=parser)
    with pytest.raises(QualityGateError):
        ing.ingest_file(f)
    assert not list((tmp_path / "art").glob("*.json"))   # gate refuses to write


def test_boilerplate_zone_restricted(tmp_path, parser):
    pages = [f"Great Book\n\nChapter {i} body content with 42 numbers.\n\n{i}"
             for i in range(1, 6)]
    f = tmp_path / "b.pdf"; f.write_bytes(b"b")
    parser.docs["b.pdf"] = make_doc(pages, title="Great Book")
    ing = Ingester(tmp_path / "art", parser=parser)
    r = ing.ingest_file(f)
    art = ExtractionArtifact.load(tmp_path / "art" / f"{r.doc_id}.json")
    joined = "\n".join(p.text for p in art.pages)
    assert "Great Book" not in joined                 # running header stripped
    assert "42" in joined                             # body numbers survive
    assert art.title == "Great Book"                  # ...but live on as metadata
    reasons = {d["reason"] for d in art.audit["deleted_lines"]}
    assert reasons == {"running-header/footer", "page-number"}


def test_empty_pages_keep_alignment(tmp_path, parser):
    f = tmp_path / "c.pdf"; f.write_bytes(b"c")
    parser.docs["c.pdf"] = make_doc(["one", "", "three"])
    ing = Ingester(tmp_path / "art", parser=parser)
    r = ing.ingest_file(f)
    art = ExtractionArtifact.load(tmp_path / "art" / f"{r.doc_id}.json")
    assert [p.page_number for p in art.pages] == [1, 2, 3]   # never renumbered
    assert art.pages[1].status == "skipped-empty"


def test_schema_gate(tmp_path, parser):
    f = tmp_path / "s.pdf"; f.write_bytes(b"s")
    parser.docs["s.pdf"] = make_doc(["schema gate text"])
    ing = Ingester(tmp_path / "art", parser=parser)
    r = ing.ingest_file(f)
    p = tmp_path / "art" / f"{r.doc_id}.json"
    data = json.loads(p.read_text())
    data["schema_version"] = "extraction-0.9"
    p.write_text(json.dumps(data))
    with pytest.raises(IncompatibleArtifactError):
        ExtractionArtifact.load(p)


def test_directory_failure_isolation(tmp_path, parser):
    (tmp_path / "docs").mkdir()
    good = tmp_path / "docs" / "good.pdf"; good.write_bytes(b"g")
    broken = tmp_path / "docs" / "broken.pdf"; broken.write_bytes(b"b")
    parser.docs["good.pdf"] = make_doc(["fine text here"])   # broken.pdf: parser raises
    ing = Ingester(tmp_path / "art", parser=parser)
    rep = ing.ingest_directory(tmp_path / "docs")
    by_name = {Path(r.source).name: r.status for r in rep.results}
    assert by_name == {"good.pdf": "ingested", "broken.pdf": "failed"}