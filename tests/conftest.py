from __future__ import annotations

import pytest

from ragcore.chunkers.sentences import SentenceChunker
from ragcore.embedders.fake import FakeEmbedder
from ragcore.generators.generators import FakeGenerator
from ragcore.pipeline.rag import Rag
from ragcore.stores.sqlite import SQLiteStore
from ragcore.types import Document

DOCS = [
    Document(id="leave.md", source="leave.md", metadata={"team": "hr"}, text=(
        "# Leave policy\n\n## Annual leave\n\nEmployees receive 25 days of annual leave per year. "
        "Unused days carry over up to a cap of 5 days.\n\n## Requesting leave\n\n"
        "Submit form HR-17 at least two weeks before the first day of absence.")),
    Document(id="expenses.md", source="expenses.md", metadata={"team": "finance"}, text=(
        "# Expenses\n\n## Limits\n\nMeals are reimbursed up to 40 euros per day. Taxis need a receipt.\n\n"
        "## Rejections\n\nCode EXP-204 means the receipt is missing. Code EXP-310 means the claim is late.")),
    Document(id="security.md", source="security.md", metadata={"team": "it"}, text=(
        "# Security\n\n## Passwords\n\nPasswords must have at least 14 characters and multi-factor "
        "authentication is mandatory.\n\n## Incidents\n\nReport a lost laptop within 2 hours to the security desk.")),
]


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def store(embedder):
    return SQLiteStore(":memory:", dim=embedder.dim, embedder_name=embedder.name)


@pytest.fixture
def rag(store, embedder):
    pipeline = Rag(store=store, embedder=embedder, chunker=SentenceChunker(size=200, overlap=40),
                   generator=FakeGenerator())
    pipeline.ingest(DOCS)
    return pipeline
