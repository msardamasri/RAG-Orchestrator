"""Profiles: one TOML file names every part of a pipeline, and ``build_rag`` assembles it.

A brain picks its parts by name. To add a part the library does not ship, register a factory::

    from ragcore.config import register
    register("store", "my_store", lambda **options: MyStore(**options))
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ragcore.pipeline.conversation import ConversationSettings
from ragcore.pipeline.rag import Rag
from ragcore.pipeline.retrieve import RetrievalSettings

Factory = Callable[..., Any]


class RagConfig(BaseModel):
    """Each part is ``{"kind": "<registered name>", ...options passed to its factory}``."""

    store: dict[str, Any] = Field(default_factory=lambda: {"kind": "sqlite", "path": "ragcore.db"})
    embedder: dict[str, Any] = Field(
        default_factory=lambda: {"kind": "local", "model": "intfloat/multilingual-e5-small"})
    chunker: dict[str, Any] = Field(default_factory=lambda: {"kind": "sentences", "size": 1000, "overlap": 200})
    reranker: dict[str, Any] = Field(
        default_factory=lambda: {"kind": "cross_encoder", "model": "BAAI/bge-reranker-base"})
    generator: dict[str, Any] | None = None
    retrieval: dict[str, Any] = Field(default_factory=dict)
    conversation: dict[str, Any] = Field(default_factory=dict)
    max_tokens: int = 512
    temperature: float = 0.1


def load_profile(path: str | Path) -> RagConfig:
    with open(path, "rb") as handle:
        return RagConfig.model_validate(tomllib.load(handle))


# -- factories (imports are lazy so an unused extra is never required) ------------------------------
def _sqlite_store(**options):
    from ragcore.stores.sqlite import SQLiteStore
    return SQLiteStore(**options)


def _qdrant_store(embedder_name: str = "", **options):
    from ragcore.stores.qdrant import QdrantStore
    return QdrantStore(**options)


def _local_embedder(**options):
    from ragcore.embedders.local import LocalEmbedder
    return LocalEmbedder(**options)


def _openai_embedder(**options):
    from ragcore.embedders.openai import OpenAIEmbedder
    return OpenAIEmbedder(**options)


def _fake_embedder(**options):
    from ragcore.embedders.fake import FakeEmbedder
    return FakeEmbedder(**options)


def _sentence_chunker(**options):
    from ragcore.chunkers.sentences import SentenceChunker
    return SentenceChunker(**options)


def _qa_chunker(**options):
    from ragcore.chunkers.sentences import QAPairChunker
    return QAPairChunker(**options)


def _whole_chunker(**options):
    from ragcore.chunkers.sentences import WholeDocumentChunker
    return WholeDocumentChunker(**options)


def _cross_encoder(**options):
    from ragcore.rerankers.rerankers import CrossEncoderReranker
    return CrossEncoderReranker(**options)


def _openai_compat(**options):
    from ragcore.generators.generators import OpenAICompatGenerator
    return OpenAICompatGenerator(**options)


def _fake_generator(**options):
    from ragcore.generators.generators import FakeGenerator
    return FakeGenerator(**options)


REGISTRY: dict[str, dict[str, Factory]] = {
    "store": {"sqlite": _sqlite_store, "qdrant": _qdrant_store},
    "embedder": {"local": _local_embedder, "openai": _openai_embedder, "fake": _fake_embedder},
    "chunker": {"sentences": _sentence_chunker, "qa_pairs": _qa_chunker, "whole": _whole_chunker},
    "reranker": {"cross_encoder": _cross_encoder, "none": lambda **_: None},
    "generator": {"openai_compat": _openai_compat, "fake": _fake_generator, "none": lambda **_: None},
}


def register(part: str, kind: str, factory: Factory) -> None:
    REGISTRY[part][kind] = factory


def _build(part: str, spec: dict[str, Any], **extra):
    options = dict(spec)
    kind = options.pop("kind", None)
    if kind not in REGISTRY[part]:
        raise ValueError(f"unknown {part} kind {kind!r}; available: {sorted(REGISTRY[part])}")
    return REGISTRY[part][kind](**options, **extra)


def build_rag(config: RagConfig | None = None, **overrides) -> Rag:
    """Assemble a pipeline from a config. Pass ready-made parts as overrides (``store=...``, ``embedder=...``,
    ``chunker=...``, ``reranker=...``, ``generator=...``) to bypass the registry for that part."""
    config = config or RagConfig()
    embedder = overrides.get("embedder") or _build("embedder", config.embedder)
    store = overrides.get("store") or _build("store", config.store, dim=embedder.dim, embedder_name=embedder.name)
    chunker = overrides.get("chunker") or _build("chunker", config.chunker)
    reranker = overrides["reranker"] if "reranker" in overrides else _build("reranker", config.reranker)
    if "generator" in overrides:
        generator = overrides["generator"]
    else:
        generator = _build("generator", config.generator) if config.generator else None
    return Rag(store=store, embedder=embedder, chunker=chunker, generator=generator, reranker=reranker,
               retrieval=RetrievalSettings(**config.retrieval),
               conversation=ConversationSettings(**config.conversation),
               fingerprint=json.dumps(config.chunker, sort_keys=True),
               max_tokens=config.max_tokens, temperature=config.temperature)
