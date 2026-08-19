from __future__ import annotations

import hashlib
import math
import re


class TestEmbedder:
    """Deterministic test double; production supports Google cloud only."""

    __test__ = False

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension
        self.document_batch_sizes: list[int] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batch_sizes.append(len(texts))
        return [self._one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one(text)

    def close(self) -> None:
        return None

    def _one(self, text: str) -> list[float]:
        normalized = re.sub(r"\s+", " ", text.casefold()).strip()
        tokens = re.findall(r"[\w-]+|[\u3400-\u9fff]", normalized)
        features = tokens + [
            normalized[index : index + 3]
            for index in range(max(0, len(normalized) - 2))
        ]
        vector = [0.0] * self.dimension
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "big")
            vector[number % self.dimension] += -1.0 if number & 1 else 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
