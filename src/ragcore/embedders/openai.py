"""OpenAI embeddings (extra: ``openai``). The adapter the original project used, kept as an option."""

from __future__ import annotations

import os
from collections.abc import Sequence

DIMENSIONS = {"text-embedding-3-large": 3072, "text-embedding-3-small": 1536, "text-embedding-ada-002": 1536}


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-large", api_key_env: str = "OPENAI_API_KEY",
                 base_url: str | None = None, dim: int | None = None, batch_size: int = 128):
        from openai import OpenAI

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set")
        self.name = f"openai:{model}"
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.batch_size = batch_size
        self.dim = dim or DIMENSIONS.get(model) or len(self.embed_query("dimension probe"))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            response = self.client.embeddings.create(input=list(texts[start:start + self.batch_size]),
                                                     model=self.model)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding
