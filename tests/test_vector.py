from __future__ import annotations

import json
import math

import httpx
import pytest

from meno.config import Settings
from meno.vector import EmbeddingError, GoogleEmbedder


def test_google_embedder_uses_batch_documents_and_query_task_types():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-goog-api-key"] == "test-google-key-value"
        payload = json.loads(request.content)
        seen.append(payload)
        if request.url.path.endswith(":batchEmbedContents"):
            assert all(item["taskType"] == "RETRIEVAL_DOCUMENT" for item in payload["requests"])
            return httpx.Response(
                200,
                json={
                    "embeddings": [
                        {"values": [float(index + 1)] * 128}
                        for index, _item in enumerate(payload["requests"])
                    ]
                },
            )
        assert payload["taskType"] == "RETRIEVAL_QUERY"
        return httpx.Response(200, json={"embedding": {"values": [2.0] * 128}})

    settings = Settings(
        google_api_key="test-google-key-value",
        embedding_dimension=128,
        google_batch_size=2,
    )
    embedder = GoogleEmbedder(settings, transport=httpx.MockTransport(handler))
    documents = embedder.embed_documents(["one", "two", "three"])
    query = embedder.embed_query("question")
    embedder.close()

    assert len(documents) == 3
    assert len(seen) == 3  # two document batches and one query
    assert math.isclose(sum(value * value for value in query), 1.0)
    assert all(len(vector) == 128 for vector in documents)


def test_google_embedder_retries_429_without_exposing_secret(monkeypatch):
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": {"message": "quota"}})
        return httpx.Response(200, json={"embedding": {"values": [1.0] * 128}})

    settings = Settings(
        google_api_key="test-google-key-value",
        embedding_dimension=128,
        google_max_retries=1,
    )
    embedder = GoogleEmbedder(settings, transport=httpx.MockTransport(handler))
    monkeypatch.setattr(embedder, "_backoff", lambda *_args, **_kwargs: None)
    assert len(embedder.embed_query("question")) == 128
    assert attempts == 2
    embedder.close()


def test_google_embedder_rejects_wrong_dimension():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embedding": {"values": [1.0] * 16}})

    settings = Settings(
        google_api_key="test-google-key-value",
        embedding_dimension=128,
    )
    embedder = GoogleEmbedder(settings, transport=httpx.MockTransport(handler))
    with pytest.raises(EmbeddingError, match="dimension mismatch"):
        embedder.embed_query("question")
    embedder.close()
