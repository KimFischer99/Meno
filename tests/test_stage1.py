from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from threading import Lock
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import select, update

from meno.api import build_service
from meno.cli import main as cli_main
from meno.config import Settings
from meno.db import AuditEvent, Outbox, ProjectionOutbox
from meno.schemas import Facet, IngestRequest, RetrieveRequest
from meno.service import _aware
from meno.vector import (
    EmbeddingError,
    MemoryVectorStore,
    OpenAICompatEmbedder,
    QdrantVectorStore,
)
from tests.fakes import TestEmbedder


def make_service(tmp_path, embedder=None, **overrides):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'meno.sqlite3'}",
        vector_mode="memory",
        embedding_dimension=256,
        **overrides,
    )
    embedder = embedder or TestEmbedder(settings.embedding_dimension)
    return build_service(settings, embedder=embedder)


def ingest_event(service, event_id: str, text: str, *, user_id: str = "user-a", **kwargs):
    return service.ingest(
        IngestRequest(
            user_id=user_id,
            event_id=event_id,
            source={"type": "hermes_turn"},
            content={"role": "user", "text": text},
            consent_scope=["personalization", "task_planning"],
            **kwargs,
        ),
        f"key-{event_id}",
    )


class PoisonEmbedder(TestEmbedder):
    """Fails document embedding whenever a text contains the poison marker."""

    __test__ = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if any("poison" in text for text in texts):
            raise EmbeddingError("provider down")
        return super().embed_documents(texts)


class SlowCountingEmbedder(TestEmbedder):
    """Makes an ignored SQLite row lock reliably visible to concurrent workers."""

    __test__ = False

    def __init__(self, dimension: int):
        super().__init__(dimension)
        self.document_calls = 0
        self._calls_lock = Lock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._calls_lock:
            self.document_calls += 1
        time.sleep(0.05)
        return super().embed_documents(texts)


def test_outbox_failure_is_incremental_and_backs_off(tmp_path):
    service = make_service(
        tmp_path,
        embedder=PoisonEmbedder(256),
        outbox_commit_batch_size=2,
        outbox_retry_base_seconds=60.0,
    )
    try:
        ingest_event(service, "good-event", "I prefer tea over coffee")
        ingest_event(service, "bad-event", "poison pill")

        # Canonical rows commit first. The durable projection queue isolates the
        # poison row while retaining the healthy vector mutation.
        assert service.process_outbox() == 2

        with service.session_factory() as session:
            good = session.scalar(select(Outbox).where(Outbox.event_id == "good-event"))
            bad = session.scalar(select(Outbox).where(Outbox.event_id == "bad-event"))
            projections = session.scalars(
                select(ProjectionOutbox).order_by(ProjectionOutbox.id)
            ).all()
        assert good.status == "processed"
        assert bad.status == "processed"
        assert bad.attempts == 0
        assert [row.status for row in projections] == ["processed", "pending"]
        assert projections[1].attempts == 1
        assert projections[1].error == "EmbeddingError"
        assert _aware(projections[1].next_attempt_at) > datetime.now(UTC)

        # Exponential backoff: the failed row is not due, so nothing is picked up.
        assert service.process_outbox() == 0

        # The good event's claim was not rolled back with the failing chunk.
        response = service.retrieve(
            RetrieveRequest(
                user_id="user-a",
                purpose="response_personalization",
                context={"query": "tea"},
                constraints={"min_confidence": 0.5},
            )
        )
        assert any("tea" in str(facet.value) for facet in response.facets)
    finally:
        service.close()


def test_sqlite_outbox_workers_are_serialized_within_process(tmp_path):
    embedder = SlowCountingEmbedder(256)
    service = make_service(tmp_path, embedder=embedder)
    try:
        ingest_event(service, "one-event", "I prefer tea over coffee")
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _index: service.process_outbox(), range(2)))

        assert sorted(results) == [0, 1]
        assert embedder.document_calls == 1
    finally:
        service.close()


def test_requeue_failed_outbox_round_trip(tmp_path):
    service = make_service(tmp_path, embedder=PoisonEmbedder(256))
    try:
        ingest_event(service, "doomed-event", "poison pill")
        assert service.process_outbox() == 1
        with service.session_factory.begin() as session:
            session.execute(
                update(ProjectionOutbox)
                .where(ProjectionOutbox.user_id == "user-a")
                .values(status="failed", attempts=20)
            )
        failed_status = service.drain_status("user-a")
        assert failed_status["failed_projection"] == 1
        assert failed_status["drained"] is False

        assert service.requeue_failed_projection() == 1
        with service.session_factory() as session:
            row = session.scalar(
                select(ProjectionOutbox).where(ProjectionOutbox.user_id == "user-a")
            )
        assert row.status == "pending"
        assert row.attempts == 0
        assert row.error is None
        assert row.next_attempt_at is None

        # After requeue (and a healthy provider) the row processes normally.
        service.vector_store.embedder = TestEmbedder(256)
        assert service.process_outbox() == 0
        assert service.drain_status("user-a")["drained"] is True
    finally:
        service.close()


def test_drain_status_counts_pending_and_failed(tmp_path):
    service = make_service(tmp_path)
    try:
        assert service.drain_status("user-a") == {
            "user_id": "user-a",
            "pending_outbox": 0,
            "failed_outbox": 0,
            "pending_projection": 0,
            "failed_projection": 0,
            "drained": True,
            "policy_version": service.settings.policy_version,
        }
        ingest_event(service, "pending-event", "I prefer tea")
        status = service.drain_status("user-a")
        assert status["pending_outbox"] == 1
        assert status["drained"] is False
        service.process_outbox()
        status = service.drain_status("user-a")
        assert status["pending_outbox"] == 0
        assert status["drained"] is True
        # Other users' outbox rows do not leak into the count.
        assert service.drain_status("user-b")["pending_outbox"] == 0
    finally:
        service.close()


def test_drain_endpoint(client):
    response = client.get("/v1/users/user-a/drain")
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "user-a"
    assert payload["pending_outbox"] == 0
    assert payload["failed_outbox"] == 0
    assert payload["drained"] is True


def test_circuit_breaker_opens_and_recovers():
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(429, json={"error": {"message": "quota"}})

    settings = Settings(
        openai_api_key="test-siliconflow-key-value",
        embedding_dimension=128,
        openai_max_retries=0,
        openai_circuit_breaker_threshold=2,
        openai_circuit_breaker_seconds=0.05,
    )
    embedder = OpenAICompatEmbedder(settings, transport=httpx.MockTransport(handler))
    with pytest.raises(EmbeddingError, match="remained unavailable"):
        embedder.embed_query("one")
    with pytest.raises(EmbeddingError, match="remained unavailable"):
        embedder.embed_query("two")
    assert attempts == 2

    # Circuit is open: calls fail fast without hitting the provider.
    with pytest.raises(EmbeddingError, match="circuit breaker is open"):
        embedder.embed_query("three")
    assert attempts == 2

    # After cooldown a half-open trial is allowed; failure re-opens immediately.
    time.sleep(0.07)
    with pytest.raises(EmbeddingError, match="remained unavailable"):
        embedder.embed_query("four")
    assert attempts == 3
    with pytest.raises(EmbeddingError, match="circuit breaker is open"):
        embedder.embed_query("five")
    assert attempts == 3
    embedder.close()


def test_circuit_breaker_closes_after_successful_trial():
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            return httpx.Response(429, json={"error": {"message": "quota"}})
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0] * 128}]}
        )

    settings = Settings(
        openai_api_key="test-siliconflow-key-value",
        embedding_dimension=128,
        openai_max_retries=0,
        openai_circuit_breaker_threshold=2,
        openai_circuit_breaker_seconds=0.05,
    )
    embedder = OpenAICompatEmbedder(settings, transport=httpx.MockTransport(handler))
    for _ in range(2):
        with pytest.raises(EmbeddingError):
            embedder.embed_query("boom")
    time.sleep(0.07)
    assert len(embedder.embed_query("recovered")) == 128
    assert len(embedder.embed_query("still recovered")) == 128
    embedder.close()


def test_retrieve_audit_is_buffered_until_flush(tmp_path):
    service = make_service(tmp_path)
    try:
        ingest_event(service, "audit-event", "I prefer Python for data science")
        service.process_outbox()
        response = service.retrieve(
            RetrieveRequest(
                user_id="user-a",
                purpose="response_personalization",
                context={"query": "data science language"},
                constraints={"min_confidence": 0.5},
            )
        )
        assert response.facets

        # Read path performs no transactional audit writes anymore.
        with service.session_factory() as session:
            rows = session.scalars(
                select(AuditEvent).where(
                    AuditEvent.event_name == "meno.retrieve.facet_selected"
                )
            ).all()
        assert rows == []

        flushed = service.flush_audit_buffer()
        assert flushed == len(response.facets)
        with service.session_factory() as session:
            rows = session.scalars(
                select(AuditEvent)
                .where(AuditEvent.event_name == "meno.retrieve.facet_selected")
                .order_by(AuditEvent.created_at)
            ).all()
            previous = session.scalar(
                select(AuditEvent)
                .where(AuditEvent.event_name != "meno.retrieve.facet_selected")
                .order_by(AuditEvent.created_at.desc())
                .limit(1)
            )
        assert len(rows) == len(response.facets)
        assert {row.claim_id for row in rows} == {
            facet.claim_id for facet in response.facets
        }
        # Hash chain stays intact across the async boundary.
        assert rows[0].prev_hash == previous.current_hash
        for earlier, later in pairwise(rows):
            assert later.prev_hash == earlier.current_hash
    finally:
        service.close()


def test_audit_buffer_drops_oldest_when_full(tmp_path):
    service = make_service(tmp_path, audit_buffer_max=2)
    try:
        for index in range(3):
            service._buffer_audit(
                event_name=f"meno.test.{index}",
                trace_id="t",
                user_id="u",
                action="test",
                purpose=None,
                decision={},
                revision=0,
                event_ids=[],
            )
        assert len(service._audit_buffer) == 2
        assert service._audit_buffer[0]["event_name"] == "meno.test.1"
    finally:
        service.close()


def test_render_escapes_attributes_and_preserves_closing_tag(tmp_path):
    service = make_service(tmp_path)
    try:
        facet = Facet(
            claim_id="claim-a",
            kind="preference",
            value='tea < coffee & "water" ' * 100,
            relevance=1.0,
            confidence=0.9,
            evidence_ids=["event-a"],
            why_selected=[],
        )
        rendered = service._render('user\"<&', 3, [facet], max_tokens=64)

        assert 'user_id="user&quot;&lt;&amp;"' in rendered
        assert len(rendered) <= 64 * 4
        assert rendered.endswith("</user_context>")
        assert "tea &lt; coffee &amp; &quot;water&quot;" in rendered
    finally:
        service.close()


def test_worker_flushes_buffered_retrieve_audit(client, ingest):
    ingest("I prefer Python for data science")
    service = client.app.state.meno
    response = client.post(
        "/v1/retrieve",
        json={
            "user_id": "user-a",
            "purpose": "response_personalization",
            "context": {"query": "data science language"},
            "constraints": {"min_confidence": 0.5},
        },
    )
    assert response.status_code == 200
    assert response.json()["facets"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with service.session_factory() as session:
            count = len(
                session.scalars(
                    select(AuditEvent).where(
                        AuditEvent.event_name == "meno.retrieve.facet_selected"
                    )
                ).all()
            )
        if count >= len(response.json()["facets"]):
            return
        time.sleep(0.05)
    raise AssertionError("background worker did not flush retrieve audits")


def test_memory_store_filters_temporal_window():
    store = MemoryVectorStore(TestEmbedder(128))
    now = datetime.now(UTC)
    store.upsert(
        "future-claim", "user-a", "active", "future preference",
        valid_from=now + timedelta(days=1),
    )
    store.upsert(
        "expired-claim", "user-a", "active", "expired preference",
        valid_from=now - timedelta(days=10),
        valid_to=now - timedelta(days=1),
    )
    store.upsert(
        "current-claim", "user-a", "active", "current preference",
        valid_from=now - timedelta(days=1),
    )
    hits = store.search("user-a", "preference", 10, as_of=now)
    assert {hit.claim_id for hit in hits} == {"current-claim"}
    # Without as_of the store keeps its previous unfiltered behavior.
    hits = store.search("user-a", "preference", 10)
    assert {hit.claim_id for hit in hits} == {
        "future-claim",
        "expired-claim",
        "current-claim",
    }


def test_retrieve_pushes_as_of_into_vector_search(tmp_path):
    service = make_service(tmp_path)
    try:
        calls = []
        original = service.vector_store.search

        def spy(user_id, query, limit, as_of=None):
            calls.append(as_of)
            return original(user_id, query, limit, as_of=as_of)

        service.vector_store.search = spy
        as_of = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
        service.retrieve(
            RetrieveRequest(
                user_id="user-a",
                purpose="response_personalization",
                context={"query": "anything", "as_of": as_of},
            )
        )
        assert calls == [as_of]
    finally:
        service.close()


class FakeQdrantClient:
    def __init__(self, *args, **kwargs) -> None:
        self.queries: list[dict] = []
        self.upserts: list[dict] = []

    def collection_exists(self, _collection: str) -> bool:
        return True

    def get_collection(self, _collection: str):
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=128))
            )
        )

    def query_points(self, **kwargs):
        self.queries.append(kwargs)
        return SimpleNamespace(points=[])

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def close(self) -> None:
        return None


def test_qdrant_pushes_temporal_payload_and_filter(monkeypatch):
    monkeypatch.setattr("meno.vector.QdrantClient", FakeQdrantClient)
    settings = Settings(
        vector_mode="qdrant",
        embedding_dimension=128,
        qdrant_collection="meno_test_temporal",
    )
    store = QdrantVectorStore(settings, TestEmbedder(128))
    valid_from = datetime(2026, 1, 1, tzinfo=UTC)
    valid_to = datetime(2026, 2, 1, tzinfo=UTC)
    store.upsert(
        "claim-1", "user-a", "active", "I prefer tea",
        valid_from=valid_from, valid_to=valid_to,
    )
    payload = store.client.upserts[0]["points"][0].payload
    assert payload["valid_from_ts"] == valid_from.timestamp()
    assert payload["valid_to_ts"] == valid_to.timestamp()

    as_of = datetime(2026, 1, 15, tzinfo=UTC)
    store.search("user-a", "tea", 5, as_of=as_of)
    query_filter = store.client.queries[0]["query_filter"]
    conditions = query_filter.must
    from_ts = next(
        condition for condition in conditions
        if getattr(condition, "key", None) == "valid_from_ts"
    )
    assert from_ts.range.lte == as_of.timestamp()
    nested = next(condition for condition in conditions if hasattr(condition, "should"))
    null_branch = next(
        branch for branch in nested.should if branch.__class__.__name__ == "IsNullCondition"
    )
    range_branch = next(
        branch for branch in nested.should if getattr(branch, "key", None) == "valid_to_ts"
    )
    assert null_branch.is_null.key == "valid_to_ts"
    assert range_branch.range.gt == as_of.timestamp()

    # No as_of: filter stays exactly the user/status pair.
    store.search("user-a", "tea", 5)
    plain = store.client.queries[1]["query_filter"]
    assert {condition.key for condition in plain.must} == {"user_id", "status"}


def test_cli_requeue_failed(tmp_path, monkeypatch, capsys):
    database = tmp_path / "cli.sqlite3"
    monkeypatch.setenv("MENO_DATABASE_URL", f"sqlite:///{database}")
    monkeypatch.setenv("MENO_OPENAI_API_KEY", "test-siliconflow-key-value")
    settings = Settings(database_url=f"sqlite:///{database}", embedding_dimension=256)
    service = build_service(settings, embedder=TestEmbedder(256))
    ingest_event(service, "failed-event", "I prefer tea")
    with service.session_factory.begin() as session:
        session.execute(
            update(Outbox).where(Outbox.event_id == "failed-event").values(status="failed")
        )
    service.close()

    monkeypatch.setattr(sys, "argv", ["meno", "requeue-failed"])
    cli_main()
    assert capsys.readouterr().out.strip() == "1"

    service = build_service(settings, embedder=TestEmbedder(256))
    assert service.drain_status("user-a")["pending_outbox"] == 1
    service.close()
