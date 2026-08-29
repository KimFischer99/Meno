"""Write-path integration tests for deterministic reflection.

The unit tests in `test_reflection.py` cover the derivation rules. These cover
what only the real write path can show: that a pattern claim lands with complete
evidence, that it is idempotent and supersedes correctly, that the flag is truly
off by default, and that consent and sensitivity are not widened on the way in.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from meno.api import create_app
from meno.config import Settings
from meno.db import AuditEvent, Claim, ClaimEvidence
from tests.fakes import TestEmbedder

BASE = datetime(2026, 1, 1, tzinfo=UTC)
# Three statements on one dimension, spread far enough apart to clear the span
# requirement. Values are phrased so the extractor's preference patterns fire.
STATEMENTS = (
    "I prefer green tea in the morning.",
    "I prefer green tea when working.",
    "I prefer green tea over soda.",
)


def _make_client(tmp_path, *, reflection: bool, name: str = "meno.sqlite3"):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / name}",
        vector_mode="memory",
        embedding_dimension=256,
        worker_poll_seconds=0.01,
        reflection_enabled=reflection,
    )
    app = create_app(settings, embedder=TestEmbedder(settings.embedding_dimension))
    return TestClient(app)


def _ingest(
    client,
    text,
    *,
    user_id="user-a",
    occurred_at=BASE,
    scopes=None,
    key=None,
):
    response = client.post(
        "/v1/ingest",
        headers={"Idempotency-Key": key or f"k-{user_id}-{occurred_at.isoformat()}"},
        json={
            "user_id": user_id,
            "occurred_at": occurred_at.isoformat(),
            "source": {"type": "hermes_turn", "profile": "test", "session_id": "s1"},
            "content": {"role": "user", "text": text},
            "consent_scope": scopes or ["personalization", "task_planning"],
        },
    )
    assert response.status_code == 202, response.text
    client.app.state.meno.process_outbox()


def _ingest_series(client, *, scopes_per_event=None, user_id="user-a"):
    for index, text in enumerate(STATEMENTS):
        _ingest(
            client,
            text,
            user_id=user_id,
            occurred_at=BASE + timedelta(days=15 * index),
            scopes=scopes_per_event[index] if scopes_per_event else None,
        )


def _patterns(client, user_id="user-a"):
    service = client.app.state.meno
    with service.session_factory() as session:
        return session.query(Claim).filter(
            Claim.user_id == user_id, Claim.kind == "pattern"
        ).all()


@pytest.fixture
def reflect_client(tmp_path):
    with _make_client(tmp_path, reflection=True) as client:
        yield client


def test_repeated_preference_produces_a_pattern_claim(reflect_client):
    _ingest_series(reflect_client)
    patterns = _patterns(reflect_client)
    assert len(patterns) == 1
    pattern = patterns[0]
    assert pattern.kind == "pattern"
    assert pattern.semantic_channel == "pattern.repeated_preference"
    assert pattern.source_type == "reflection"
    assert pattern.origin_role == "user"
    assert pattern.status == "active"
    assert "repeatedly" in pattern.value


def test_pattern_carries_evidence_from_every_source_event(reflect_client):
    """Charter #5 requires lineage on every selected facet; a pattern with
    partial evidence would claim support it cannot show."""
    _ingest_series(reflect_client)
    pattern = _patterns(reflect_client)[0]
    service = reflect_client.app.state.meno
    with service.session_factory() as session:
        rows = session.query(ClaimEvidence).filter(
            ClaimEvidence.claim_id == pattern.id
        ).all()
    assert len(rows) >= 3
    assert {row.relation for row in rows} == {"reflection_source"}


def test_flag_off_produces_no_patterns(tmp_path):
    with _make_client(tmp_path, reflection=False) as client:
        _ingest_series(client)
        assert _patterns(client) == []


def test_reprocessing_the_same_events_is_idempotent(reflect_client):
    """The derivation key excludes wall-clock time and processing order, so a
    second pass must not create a second pattern (Charter #12)."""
    _ingest_series(reflect_client)
    before = [claim.id for claim in _patterns(reflect_client)]
    reflect_client.app.state.meno.process_outbox()
    after = [claim.id for claim in _patterns(reflect_client)]
    assert before == after


def test_a_new_supporting_event_supersedes_the_previous_pattern(reflect_client):
    """The support set is part of the derivation key, so a grown set is a new
    claim; the old summary must not stay active alongside it."""
    _ingest_series(reflect_client)
    first = _patterns(reflect_client)[0]
    _ingest(
        reflect_client,
        "I prefer green tea after lunch.",
        occurred_at=BASE + timedelta(days=60),
    )
    patterns = _patterns(reflect_client)
    active = [claim for claim in patterns if claim.status == "active"]
    assert len(active) == 1
    assert active[0].id != first.id
    superseded = next(claim for claim in patterns if claim.id == first.id)
    assert superseded.status == "superseded"
    assert superseded.superseded_reason == "reflection_refreshed"
    assert superseded.superseded_by_id == active[0].id
    assert active[0].supersedes_id == first.id


def test_allowed_purposes_is_narrowed_to_the_shared_scope(reflect_client):
    """Retrieval admits a claim if either the scope or the purpose matches, so a
    union here would serve the pattern under a purpose one source never allowed."""
    _ingest_series(
        reflect_client,
        scopes_per_event=[
            ["personalization", "task_planning"],
            ["personalization"],
            ["personalization", "task_planning"],
        ],
    )
    pattern = _patterns(reflect_client)[0]
    assert pattern.allowed_purposes == ["personalization"]


def test_pattern_is_reachable_through_retrieval(reflect_client):
    """`pattern` is already in STATE_LAYER_KINDS, so once produced it must
    actually be selectable -- otherwise the layer exists only in the database."""
    _ingest_series(reflect_client)
    response = reflect_client.post(
        "/v1/retrieve",
        json={
            "user_id": "user-a",
            "purpose": "response_personalization",
            "context": {"query": "green tea preference", "task_type": "conversation_recall"},
            "constraints": {"max_facets": 12, "min_confidence": 0.3},
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    kinds = {facet["kind"] for facet in payload["facets"]}
    assert "pattern" in kinds, kinds
    pattern_facet = next(facet for facet in payload["facets"] if facet["kind"] == "pattern")
    assert pattern_facet["evidence_ids"], "a selected pattern must expose its lineage"


def test_reflection_emits_an_audit_row_with_its_sources(reflect_client):
    """Charter #5: a derived claim must be explainable from the audit log alone."""
    _ingest_series(reflect_client)
    service = reflect_client.app.state.meno
    with service.session_factory() as session:
        rows = session.query(AuditEvent).filter(
            AuditEvent.event_name == "meno.claim.reflection"
        ).all()
    assert rows, "reflection must leave an audit trail"
    decision = rows[-1].decision
    assert decision["reason"] == "reflection_derived"
    assert len(decision["source_claim_ids"]) >= 3
    assert decision["evidence_count"] >= 3
    assert rows[-1].source_event_ids


def test_users_do_not_share_patterns(reflect_client):
    """Charter #1 is zero cross-user leakage; reflection reads per user."""
    _ingest_series(reflect_client, user_id="user-a")
    _ingest(reflect_client, "I prefer black coffee always.", user_id="user-b")
    assert len(_patterns(reflect_client, "user-a")) == 1
    assert _patterns(reflect_client, "user-b") == []
