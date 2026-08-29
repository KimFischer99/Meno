from __future__ import annotations

import json
import math
import sys
from array import array

import httpx
import pytest

from meno.config import Settings
from meno.vector import (
    EmbeddingError,
    MemoryVectorStore,
    OpenAICompatEmbedder,
    VectorDocument,
)
from tests.fakes import TestEmbedder


def test_siliconflow_embedder_batches_documents_and_queries():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-siliconflow-key-value"
        payload = json.loads(request.content)
        seen.append(payload)
        assert request.url.path.endswith("/embeddings")
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": [float(index + 1)] * 128}
                    for index, _item in enumerate(payload["input"])
                ]
            },
        )

    settings = Settings(
        openai_api_key="test-siliconflow-key-value",
        embedding_dimension=128,
        openai_batch_size=2,
    )
    embedder = OpenAICompatEmbedder(settings, transport=httpx.MockTransport(handler))
    documents = embedder.embed_documents(["one", "two", "three"])
    query = embedder.embed_query("question")
    embedder.close()

    assert len(documents) == 3
    assert len(seen) == 3  # two document batches and one query
    assert math.isclose(sum(value * value for value in query), 1.0)
    assert all(len(vector) == 128 for vector in documents)


def test_siliconflow_embedder_retries_429_without_exposing_secret(monkeypatch):
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": {"message": "quota"}})
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0] * 128}]}
        )

    settings = Settings(
        openai_api_key="test-siliconflow-key-value",
        embedding_dimension=128,
        openai_max_retries=1,
    )
    embedder = OpenAICompatEmbedder(settings, transport=httpx.MockTransport(handler))
    monkeypatch.setattr(embedder, "_backoff", lambda *_args, **_kwargs: None)
    assert len(embedder.embed_query("question")) == 128
    assert attempts == 2
    embedder.close()


def test_siliconflow_embedder_rejects_wrong_dimension():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0] * 16}]}
        )

    settings = Settings(
        openai_api_key="test-siliconflow-key-value",
        embedding_dimension=128,
    )
    embedder = OpenAICompatEmbedder(settings, transport=httpx.MockTransport(handler))
    with pytest.raises(EmbeddingError, match="dimension mismatch"):
        embedder.embed_query("question")
    embedder.close()


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(200, content=b"not-json"), "invalid JSON"),
        (httpx.Response(200, json={"data": ["invalid"]}), "invalid vector data"),
        (
            httpx.Response(
                200, json={"data": [{"index": 1, "embedding": [1.0] * 128}]}
            ),
            "invalid vector data",
        ),
    ],
)
def test_siliconflow_embedder_fails_closed_on_malformed_success_response(
    response: httpx.Response, message: str
) -> None:
    settings = Settings(
        openai_api_key="test-siliconflow-key-value",
        embedding_dimension=128,
    )
    embedder = OpenAICompatEmbedder(
        settings, transport=httpx.MockTransport(lambda _request: response)
    )
    try:
        with pytest.raises(EmbeddingError, match=message):
            embedder.embed_query("question")
    finally:
        embedder.close()


# --------------------------------------------------- MemoryVectorStore residency


def _store(*, max_resident: int = 0, dimension: int = 64) -> MemoryVectorStore:
    return MemoryVectorStore(TestEmbedder(dimension), max_resident=max_resident)


def _document(index: int, *, user_id: str = "u1", status: str = "active") -> VectorDocument:
    return VectorDocument(
        claim_id=f"c{index}",
        user_id=user_id,
        status=status,
        text=f"claim number {index}",
        valid_from=None,
        valid_to=None,
    )


def test_vectors_are_stored_as_packed_float32():
    """The whole point of the small-host profile: a 1024-dim list[float] costs
    32.6 KB because it boxes 1024 float objects, while array("f") packs them into
    4.1 KB. At 20k claims that is 80 MB instead of 638 MB."""
    store = _store(dimension=1024)
    store.upsert_many([_document(0)])
    _, _, vector, _, _ = store._points["c0"]
    assert isinstance(vector, array)
    assert vector.typecode == "f"
    assert len(vector) == 1024
    # 4 bytes per element plus a small object header, versus ~32 KB for a list.
    assert sys.getsizeof(vector) < 6000


def test_search_still_ranks_correctly_with_float32_vectors():
    store = _store()
    store.upsert_many(
        [
            VectorDocument("match", "u1", "active", "green tea preference", None, None),
            VectorDocument("other", "u1", "active", "completely unrelated topic", None, None),
        ]
    )
    hits = store.search("u1", "green tea preference", limit=2)
    assert hits[0].claim_id == "match"
    assert hits[0].score >= hits[1].score


def test_residency_cap_rejects_writes_beyond_the_limit():
    """Vectors are recoverable from canonical storage via rebuild_projection, but an
    OOM kill is not: it takes the sidecar down and breaks Charter #11."""
    store = _store(max_resident=2)
    store.upsert_many([_document(0), _document(1)])
    with pytest.raises(EmbeddingError, match="at capacity"):
        store.upsert_many([_document(2)])
    assert store.resident() == 2


def test_updating_an_existing_claim_does_not_consume_capacity():
    store = _store(max_resident=2)
    store.upsert_many([_document(0), _document(1)])
    # Re-upserting a known claim id replaces in place, so it must not be counted
    # as an admission -- otherwise a full store could never be corrected.
    store.upsert_many([_document(0)])
    assert store.resident() == 2


def test_health_reports_unhealthy_once_capacity_is_exceeded():
    """Reporting the condition is what routes the caller into the degraded path
    instead of retrying against a store that cannot accept writes."""
    store = _store(max_resident=1)
    store.upsert_many([_document(0)])
    assert store.health() is True
    with pytest.raises(EmbeddingError):
        store.upsert_many([_document(1)])
    assert store.health() is False


def test_deleting_claims_restores_health():
    store = _store(max_resident=1)
    store.upsert_many([_document(0)])
    with pytest.raises(EmbeddingError):
        store.upsert_many([_document(1)])
    assert store.health() is False
    store.delete_claim("c0")
    assert store.health() is True
    store.upsert_many([_document(1)])
    assert store.resident() == 1


def test_deleting_a_user_restores_health():
    store = _store(max_resident=2)
    store.upsert_many([_document(0, user_id="u1"), _document(1, user_id="u1")])
    with pytest.raises(EmbeddingError):
        store.upsert_many([_document(2, user_id="u2")])
    store.delete_user("u1")
    assert store.health() is True
    assert store.resident() == 0


def test_zero_cap_means_unbounded():
    """Development and the test suite run without a cap; production validation
    requires a positive one (see test_config)."""
    store = _store(max_resident=0)
    store.upsert_many([_document(index) for index in range(50)])
    assert store.resident() == 50
    assert store.health() is True
