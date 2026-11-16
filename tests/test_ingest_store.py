from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ragcore.chunkers.sentences import QAPairChunker, SentenceChunker, chunk_id, split_sentences
from ragcore.loaders.files import FileLoader
from ragcore.loaders.records import RecordLoader, conversation_documents
from ragcore.pipeline.rag import Rag
from ragcore.stores.sqlite import SQLiteStore, fts_query
from ragcore.types import Document
from tests.conftest import DOCS

SAMPLE_PDF = Path(__file__).resolve().parents[1] / "eval_data" / "corpus" / "mobility-programme.pdf"


def test_chunks_follow_headings_and_keep_them():
    chunks = SentenceChunker(size=200, overlap=40).chunk(DOCS[0])
    assert [c.metadata["section"] for c in chunks] == ["Leave policy > Annual leave",
                                                        "Leave policy > Requesting leave"]
    assert chunks[1].text.startswith("Leave policy > Requesting leave\n")
    assert "HR-17" in chunks[1].text and "HR-17" not in chunks[0].text
    assert chunks[0].metadata["team"] == "hr" and chunks[0].metadata["source"] == "leave.md"


def test_chunker_never_cuts_mid_sentence_and_overlaps():
    sentences = [f"Sentence number {i} talks about topic {i}." for i in range(12)]
    document = Document(id="d", text=" ".join(sentences))
    chunks = SentenceChunker(size=160, overlap=60).chunk(document)
    assert len(chunks) > 2
    for chunk in chunks:
        assert len(chunk.text) <= 160
        assert chunk.text.rstrip().endswith(".")
        assert all(line in sentences for line in chunk.text.split("\n"))
    first_lines, second_lines = chunks[0].text.split("\n"), chunks[1].text.split("\n")
    assert first_lines[-1] == second_lines[0]          # the overlap carries the last sentence forward


def test_hash_lines_inside_code_blocks_are_not_headings():
    text = ("# Guide\n\n## Setup\n\n```bash\n# install deps\nmake bootstrap\n```\n\nThen run make doctor.\n\n"
            "~~~\n## not a heading either\n~~~\n\n  ## Tests ##\n\nRun make test.")
    chunks = SentenceChunker(size=1000, overlap=0).chunk(Document(id="g.md", text=text))
    assert [c.metadata["section"] for c in chunks] == ["Guide > Setup", "Guide > Tests"]
    assert "# install deps\nmake bootstrap\n```" in chunks[0].text           # the code block stays whole
    assert "## not a heading either" in chunks[0].text


def test_long_text_without_full_stops_is_cut_between_lines_with_overlap():
    rows = [f"Category {i} BC-{1000 + i} Owner {i}" for i in range(60)]     # a table as pypdf extracts it
    chunks = SentenceChunker(size=200, overlap=60).chunk(Document(id="t.pdf", text="\n".join(rows)))
    assert len(chunks) > 5
    for chunk in chunks:
        assert len(chunk.text) <= 200
        assert all(line in rows for line in chunk.text.split("\n"))      # no row is ever cut
    assert chunks[1].text.split("\n")[0] in chunks[0].text.split("\n")       # neighbours overlap by whole rows


def test_a_single_long_line_is_cut_between_words():
    words = [f"word{i}" for i in range(300)]
    chunks = SentenceChunker(size=120, overlap=0).chunk(Document(id="w.txt", text=" ".join(words)))
    pieces = [token for chunk in chunks for token in chunk.text.split()]
    assert all(len(chunk.text) <= 120 for chunk in chunks) and pieces == words


def test_numbered_headings_are_not_sentence_ends():
    assert split_sentences("Paid within 5 days.\n9. Budget codes Use the code. The cap is 40. Above it, ask.") == [
        "Paid within 5 days.", "9. Budget codes Use the code.", "The cap is 40.", "Above it, ask."]


def test_chunk_ids_are_stable():
    assert chunk_id("a.md", 3) == chunk_id("a.md", 3) != chunk_id("a.md", 4)


def test_qa_chunker_splits_pairs():
    document = Document(id="faq", text="Q: How many days?\nA: 25 days.\n\nQ: Which form?\nA: HR-17.")
    chunks = QAPairChunker().chunk(document)
    assert [c.text for c in chunks] == ["Q: How many days?\nA: 25 days.", "Q: Which form?\nA: HR-17."]


def test_ingest_is_idempotent_and_replaces_changed_documents(rag, store):
    total = store.count()
    report = rag.ingest(DOCS)
    assert (report.ingested, report.skipped, store.count()) == (0, 3, total)

    changed = Document(id="leave.md", source="leave.md", metadata={"team": "hr"},
                       text="# Leave policy\n\nEmployees receive 30 days of annual leave.")
    report = rag.ingest([changed])
    assert report.ingested == 1
    texts = [h.chunk.text for h in store.search_lexical("annual leave", 10)]
    assert any("30 days" in t for t in texts) and not any("25 days" in t for t in texts)
    assert not store.search_lexical("HR-17", 10)       # the removed section's chunk is gone from the index


def test_prune_removes_documents_missing_from_a_full_run(rag, store):
    report = rag.ingest(DOCS[:2], prune=True)
    assert report.removed == 1 and store.document_ids() == ["expenses.md", "leave.md"]
    assert not store.search_lexical("passwords", 10)


def test_lexical_search_finds_exact_codes(store, rag):
    hits = store.search_lexical("What does EXP-204 mean?", 5)
    assert hits and "EXP-204" in hits[0].chunk.text
    assert fts_query('say "hello" EXP-204') == '"say" OR "hello" OR "EXP-204"'
    assert store.search_lexical('"; DROP TABLE chunks; --', 5) == []
    assert store.count() > 0


def test_filters_apply_to_both_searches(store, rag, embedder):
    vector = embedder.embed_query("receipt missing code")
    assert {h.chunk.doc_id for h in store.search_vector(vector, 10, {"team": "hr"})} == {"leave.md"}
    assert {h.chunk.doc_id for h in store.search_vector(vector, 10, {"team": ["hr", "it"]})} == {"leave.md",
                                                                                               "security.md"}
    assert store.search_lexical("receipt", 10, {"team": "hr"}) == []
    assert store.search_lexical("receipt", 10, {"doc_id": "expenses.md"})
    with pytest.raises(ValueError):
        store.search_lexical("receipt", 10, {"team') OR 1=1 --": "x"})


def test_vector_search_orders_by_similarity(store, rag, embedder):
    hits = store.search_vector(embedder.embed_query("passwords characters authentication"), 3)
    assert hits[0].chunk.doc_id == "security.md"
    assert hits[0].scores["vector"] >= hits[-1].scores["vector"]


def test_store_refuses_a_different_embedding_size(tmp_path):
    path = tmp_path / "s.db"
    SQLiteStore(path, dim=8).close()
    with pytest.raises(RuntimeError, match="8-dimension"):
        SQLiteStore(path, dim=16)


def test_file_and_record_loaders(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nhello")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("world")
    (tmp_path / "skip.bin").write_text("x")
    documents = list(FileLoader(tmp_path, metadata={"brain": "demo"}).load())
    assert [d.id for d in documents] == ["a.md", "sub/b.txt"]
    assert documents[0].metadata == {"kind": "markdown", "brain": "demo"}

    records = [{"id": 1, "text": "hi", "client": "Sofia"}, {"id": 2, "text": ""}]
    loaded = list(RecordLoader(records, metadata_fields=["client"]).load())
    assert [(d.id, d.metadata) for d in loaded] == [("1", {"client": "Sofia"})]

    messages = [{"conversation_id": "c1", "sender": "me" if i % 2 else "sofia", "text": f"m{i}"} for i in range(10)]
    windows = list(conversation_documents(messages, window=4, overlap=1))
    assert [d.id for d in windows] == ["c1:0", "c1:3", "c1:6"]
    assert windows[1].text.splitlines()[0] == "me: m3" and windows[-1].text.endswith("m9")


def test_an_unreadable_file_is_reported_and_does_not_stop_the_load(tmp_path):
    (tmp_path / "good.md").write_text("# Good\n\nhello")
    (tmp_path / "broken.pdf").write_bytes(b"this is not a pdf")
    loader = FileLoader(tmp_path)
    assert [d.id for d in loader.load()] == ["good.md"]
    assert list(loader.failed) == ["broken.pdf"]


def test_prune_keeps_what_was_stored_for_a_file_that_now_fails(tmp_path, store, embedder):
    rag = Rag(store=store, embedder=embedder, chunker=SentenceChunker())
    (tmp_path / "notes.md").write_text("# Notes\n\nThe meal limit is 40 euros.")
    shutil.copy(SAMPLE_PDF, tmp_path / "mobility.pdf")
    assert rag.ingest(FileLoader(tmp_path)).ingested == 2

    (tmp_path / "mobility.pdf").write_bytes(b"%PDF-1.4 truncated by a failed download")
    report = rag.ingest(FileLoader(tmp_path), prune=True)
    assert list(report.failed) == ["mobility.pdf"] and report.removed == 0
    assert store.document_ids() == ["mobility.pdf", "notes.md"]
    assert store.search_lexical("BikeLane", 5)          # the old text is still searchable

    (tmp_path / "mobility.pdf").unlink()                 # a file that is really gone is still pruned
    assert rag.ingest(FileLoader(tmp_path), prune=True).removed == 1


def test_a_pdf_without_text_is_reported_as_empty(tmp_path, store, embedder):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(tmp_path / "scan.pdf", "wb") as handle:
        writer.write(handle)
    report = Rag(store=store, embedder=embedder, chunker=SentenceChunker()).ingest(FileLoader(tmp_path))
    assert report.empty == ["scan.pdf"] and report.ingested == 0 and not report.failed


def test_loader_skips_hidden_folders_and_reads_a_byte_order_mark(tmp_path):
    (tmp_path / "bom.md").write_text("﻿# Title\n\n## Part\n\nSome text.", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "readme.md").write_text("# Not mine")
    documents = list(FileLoader(tmp_path).load())
    assert [d.id for d in documents] == ["bom.md"]
    assert SentenceChunker().chunk(documents[0])[0].metadata["section"] == "Title > Part"
