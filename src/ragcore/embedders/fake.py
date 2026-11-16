"""A deterministic embedder with no model: hashed bag of words. For tests and offline smoke runs only.

Texts that share words land close together, so retrieval behaves sensibly in tests, but it has no notion of
meaning: use a real embedder for anything you measure.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

WORD = re.compile(r"[a-záéíóúñü0-9]+")


class FakeEmbedder:
    name = "fake"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in WORD.findall(text.lower()):
            stem = word[:-1] if len(word) > 3 and word.endswith("s") else word
            digest = hashlib.blake2b(stem.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            vector[index] += 1.0 if digest[4] % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
