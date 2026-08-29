from __future__ import annotations

import ast
import os
import sys
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, text, update

from meno.api import build_service
from meno.cli import main as cli_main
from meno.config import Settings
from meno.db import AuditEvent, Claim, ClaimEvidence, Event, Outbox
from meno.extractor import extract_claims
from meno.schemas import FeedbackRequest, IngestRequest
from meno.service import _semantic_key
from tests.fakes import TestEmbedder

V2 = "meno-extractor-2.0.0"
V3 = "meno-extractor-3.0.0"


def make_service(tmp_path, embedder=None, **overrides):
    overrides.setdefault("extractor_version", V2)
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


def claims_for(service, user_id="user-a"):
    from sqlalchemy.orm import selectinload

    with service.session_factory() as session:
        return session.scalars(
            select(Claim)
            .options(selectinload(Claim.evidence))
            .where(Claim.user_id == user_id)
            .order_by(Claim.created_at)
        ).all()


def make_event(event_id: str, content: str, role: str = "user") -> Event:
    return Event(
        id=event_id,
        user_id="user-a",
        occurred_at=datetime.now(UTC),
        source_type="hermes_turn",
        role=role,
        content=content,
        content_hash="sha256:x",
        consent_scope=["personalization"],
    )


# --- D1: schema migration -------------------------------------------------


def test_migrate_on_fresh_db_is_idempotent(tmp_path):
    service = make_service(tmp_path)
    try:
        ingest_event(service, "e1", "I prefer tea")
        service.process_outbox()
        first = service.migrate()
        assert first["columns_added"] == []
        assert first["backfilled"] == 0
        assert first["deduplicated"] == 0
        assert first["index_created"] is True
        second = service.migrate()
        assert second == first
        claims = claims_for(service)
        assert claims and all(claim.semantic_key for claim in claims)
        assert claims[0].routing_slot == "beverage"
        assert claims[0].routing_basis == "deterministic_slot"
    finally:
        service.close()


def test_routing_migration_revert_is_idempotent(tmp_path):
    service = make_service(tmp_path)
    try:
        ingest_event(service, "e1", "I prefer tea")
        service.process_outbox()
        claim = claims_for(service)[0]
        semantic_key = claim.semantic_key
        first = service.revert_routing_migration()
        assert first == {"cleared": 1}
        cleared = claims_for(service)[0]
        assert cleared.semantic_key == semantic_key
        assert cleared.routing_slot is None
        assert cleared.routing_basis is None
        assert cleared.router_version is None
        assert service.revert_routing_migration() == {"cleared": 0}
        migrated = service.migrate()
        assert migrated["routing_backfilled"] == 1
        restored = claims_for(service)[0]
        assert restored.routing_slot == "beverage"
        assert restored.routing_basis == "deterministic_slot"
    finally:
        service.close()


def test_migrate_legacy_schema_backfills_dedupes_and_indexes(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    settings = Settings(
        database_url=f"sqlite:///{database}", vector_mode="memory", embedding_dimension=256
    )
    # Build a pre-migration meno_claims table (without the three new columns).
    from sqlalchemy import create_engine

    engine = create_engine(settings.database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE meno_claims (
                    id VARCHAR(64) PRIMARY KEY,
                    derivation_key VARCHAR(128) UNIQUE,
                    user_id VARCHAR(256) NOT NULL,
                    kind VARCHAR(32) NOT NULL,
                    origin_role VARCHAR(16) NOT NULL DEFAULT 'user',
                    semantic_channel VARCHAR(256) NOT NULL,
                    value TEXT NOT NULL,
                    status VARCHAR(32) NOT NULL DEFAULT 'active',
                    confidence FLOAT NOT NULL DEFAULT 0.7,
                    half_life_days FLOAT,
                    sensitive BOOLEAN NOT NULL DEFAULT 0,
                    allowed_purposes JSON NOT NULL,
                    source_type VARCHAR(64) NOT NULL,
                    valid_from DATETIME NOT NULL,
                    valid_to DATETIME,
                    supersedes_id VARCHAR(64),
                    extractor_version VARCHAR(128) NOT NULL,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                )
                """
            )
        )
        for claim_id, derivation_key, value, valid_from in (
            ("c-old-1", "dk-1", "tea", "2026-01-01 00:00:00+00:00"),
            ("c-old-2", "dk-2", "coffee", "2026-01-02 00:00:00+00:00"),
        ):
            connection.execute(
                text(
                    """
                    INSERT INTO meno_claims (
                        id, derivation_key, user_id, kind, origin_role,
                        semantic_channel, value, status, confidence, half_life_days,
                        sensitive, allowed_purposes, source_type, valid_from,
                        extractor_version, created_at, updated_at
                    ) VALUES (
                        :id, :dk, 'user-a', 'preference', 'user',
                        'preference.explicit', :value, 'active', 0.9, 180,
                        0, '["personalization"]', 'hermes_turn', :valid_from,
                        'meno-extractor-1.0.0', :valid_from, :valid_from
                    )
                    """
                ),
                {
                    "id": claim_id,
                    "dk": derivation_key,
                    "value": value,
                    "valid_from": valid_from,
                },
            )
    engine.dispose()

    service = build_service(settings, embedder=TestEmbedder(256))
    try:
        report = service.migrate()
        assert set(report["columns_added"]) == {
            "semantic_key",
            "superseded_by_id",
            "superseded_reason",
            "routing_slot",
            "routing_basis",
            "router_version",
            "stance",
        }
        assert report["backfilled"] == 2
        assert report["routing_backfilled"] == 2
        assert report["deduplicated"] == 1
        assert report["index_created"] is True

        claims = claims_for(service)
        assert all(claim.semantic_key for claim in claims)
        active = [claim for claim in claims if claim.status == "active"]
        superseded = [claim for claim in claims if claim.status == "superseded"]
        # The most recent claim (c-old-2) is kept; the older one is superseded.
        assert [claim.id for claim in active] == ["c-old-2"]
        assert [claim.id for claim in superseded] == ["c-old-1"]
        assert superseded[0].superseded_by_id == "c-old-2"
        assert superseded[0].superseded_reason == "version_upgrade"

        with service.engine.begin() as connection:
            indexes = connection.execute(
                text("PRAGMA index_list(meno_claims)")
            ).all()
        assert any(row[1] == "uq_claim_active_semantic" for row in indexes)

        # Re-running is a no-op.
        again = service.migrate()
        assert again["columns_added"] == []
        assert again["backfilled"] == 0
        assert again["routing_backfilled"] == 0
        assert again["deduplicated"] == 0
    finally:
        service.close()


def test_partial_unique_index_rejects_second_active_claim(tmp_path):
    service = make_service(tmp_path)
    try:
        ingest_event(service, "e1", "I prefer tea")
        service.process_outbox()
        old = claims_for(service)[0]
        with (
            pytest.raises(Exception, match="uq_claim_active_semantic|UNIQUE"),
            service.session_factory.begin() as session,
        ):
            session.add(
                Claim(
                    id="rogue-claim",
                    derivation_key="sha256:rogue",
                    user_id=old.user_id,
                    kind=old.kind,
                    origin_role="user",
                    semantic_channel=old.semantic_channel,
                    value="tea",
                    status="active",
                    confidence=0.9,
                    half_life_days=180,
                    sensitive=False,
                    allowed_purposes=["personalization"],
                    source_type="hermes_turn",
                    valid_from=datetime.now(UTC),
                    extractor_version=V2,
                    semantic_key=old.semantic_key,
                )
            )
        # The old claim is untouched after the rejected insert.
        assert claims_for(service)[0].status == "active"
    finally:
        service.close()


# --- D2: worker version filter + reprocess ---------------------------------


def test_worker_ignores_rows_from_other_versions(tmp_path):
    service = make_service(tmp_path)
    try:
        ingest_event(service, "e1", "I prefer tea")
        with service.session_factory.begin() as session:
            session.execute(
                update(Outbox)
                .where(Outbox.event_id == "e1")
                .values(processor_version="meno-extractor-1.0.0")
            )
        assert service.process_outbox() == 0
        with service.session_factory() as session:
            row = session.scalar(select(Outbox).where(Outbox.event_id == "e1"))
        assert row.status == "pending"
        assert claims_for(service) == []
    finally:
        service.close()


def test_reprocess_enqueues_cancels_and_is_idempotent(tmp_path):
    service = make_service(tmp_path)
    try:
        ingest_event(service, "e1", "I prefer tea")
        ingest_event(service, "e2", "I prefer coffee")
        service.process_outbox()

        report = service.reprocess(V3)
        assert report == {
            "extractor_version": V3,
            "targeted": 2,
            "inserted": 2,
            "cancelled": 0,
        }
        # Idempotent re-run.
        assert service.reprocess(V3)["inserted"] == 0

        # A stale pending row from another version gets cancelled.
        with service.session_factory.begin() as session:
            session.add(
                Outbox(
                    id="stale-row",
                    event_id="e1",
                    processor_version="meno-extractor-1.0.0",
                )
            )
        report = service.reprocess(V3)
        assert report["inserted"] == 0
        assert report["cancelled"] == 1
        with service.session_factory() as session:
            stale = session.get(Outbox, "stale-row")
            rows = session.scalars(
                select(Outbox).where(Outbox.event_id == "e1", Outbox.processor_version == V3)
            ).all()
        assert stale.status == "cancelled"
        assert len(rows) == 1 and rows[0].status == "pending"

        # User scoping limits the target set.
        ingest_event(service, "e3", "I prefer jazz", user_id="user-b")
        report = service.reprocess(V3, user_id="user-b")
        assert report["targeted"] == 1
    finally:
        service.close()


def test_reprocess_reports_real_inserted_count_on_postgresql():
    pg_url = os.environ.get("MENO_TEST_PG_URL")
    if not pg_url:
        pytest.skip("MENO_TEST_PG_URL not set")
    # Unique user/event scope keeps the test isolated on a shared database.
    suffix = uuid.uuid4().hex[:12]
    user_id = f"pg-user-{suffix}"
    settings = Settings(
        database_url=pg_url,
        vector_mode="memory",
        embedding_dimension=256,
    )
    service = build_service(settings, embedder=TestEmbedder(256))
    try:
        ingest_event(service, f"e1-{suffix}", "I prefer tea", user_id=user_id)
        ingest_event(service, f"e2-{suffix}", "I prefer coffee", user_id=user_id)
        service.process_outbox()

        # Multi-row INSERT ... ON CONFLICT DO NOTHING reports rowcount -1 on
        # PostgreSQL; the report must show the real inserted count instead.
        report = service.reprocess(V3, user_id=user_id)
        assert report == {
            "extractor_version": V3,
            "targeted": 2,
            "inserted": 2,
            "cancelled": 0,
        }
        # Idempotent re-run inserts nothing.
        assert service.reprocess(V3, user_id=user_id)["inserted"] == 0
    finally:
        service.close()


def test_replay_under_new_version_supersedes_old_claim(tmp_path):
    service = make_service(tmp_path)
    try:
        ingest_event(service, "e1", "I prefer tea")
        service.process_outbox()
        old = claims_for(service)[0]
        assert old.extractor_version == V2

        service.reprocess(V3)
        replayed = make_service(tmp_path, extractor_version=V3)
        try:
            assert replayed.process_outbox() == 1
        finally:
            replayed.close()

        claims = claims_for(service)
        active = [claim for claim in claims if claim.status == "active"]
        superseded = [claim for claim in claims if claim.status == "superseded"]
        assert len(active) == 1
        assert active[0].extractor_version == V3
        assert active[0].id != old.id
        assert [claim.id for claim in superseded] == [old.id]
        assert superseded[0].superseded_by_id == active[0].id
        assert superseded[0].superseded_reason == "version_upgrade"
        assert superseded[0].valid_to is not None
        # The invariant holds: one active claim per (user, semantic_key).
        assert active[0].semantic_key == superseded[0].semantic_key
    finally:
        service.close()


# --- Key-level coordination -------------------------------------------------


def test_same_value_restatement_dedupes_into_one_claim(tmp_path):
    service = make_service(tmp_path)
    try:
        ingest_event(service, "e1", "I prefer tea")
        ingest_event(service, "e2", "I prefer tea")
        service.process_outbox()
        claims = claims_for(service)
        assert len(claims) == 1
        claim = claims[0]
        assert claim.status == "active"
        assert {item.event_id for item in claim.evidence} == {"e1", "e2"}
    finally:
        service.close()


def test_slot_evolution_merges_into_one_active_claim(tmp_path):
    service = make_service(tmp_path)
    try:
        ingest_event(service, "e1", "I prefer tea")
        service.process_outbox()
        ingest_event(service, "e2", "I prefer green tea")
        service.process_outbox()
        claims = claims_for(service)
        active = [claim for claim in claims if claim.status == "active"]
        superseded = [claim for claim in claims if claim.status == "superseded"]
        assert len(active) == 1
        assert active[0].value == "green tea"
        assert len(superseded) == 1
        assert superseded[0].value == "tea"
        assert superseded[0].superseded_reason == "contradiction"
        assert superseded[0].superseded_by_id == active[0].id
    finally:
        service.close()


def test_negation_supersedes_earlier_affirmation(tmp_path):
    service = make_service(tmp_path)
    try:
        ingest_event(service, "e1", "I prefer tea")
        service.process_outbox()
        ingest_event(service, "e2", "I no longer like tea")
        service.process_outbox()
        claims = claims_for(service)
        active = [claim for claim in claims if claim.status == "active"]
        assert len(active) == 1
        # The reversal is carried by stance, not by leftover sentence text, so the
        # supersede chain reads as one dimension changing direction.
        assert active[0].stance == "negative"
        assert active[0].value == "tea"
        superseded = [claim for claim in claims if claim.status == "superseded"]
        assert superseded[0].stance == "positive"
        assert superseded[0].superseded_reason == "contradiction"
    finally:
        service.close()


def test_event_level_cleanup_withdraws_stale_claim(tmp_path):
    service = make_service(tmp_path)
    try:
        ingest_event(service, "e1", "I prefer tea")
        service.process_outbox()
        old = claims_for(service)[0]
        # Simulate a key-scheme change: the old claim's key is not reproduced
        # by the new version, so replay must withdraw it.
        with service.session_factory.begin() as session:
            session.execute(
                update(Claim)
                .where(Claim.id == old.id)
                .values(semantic_key=_semantic_key("user-a", "preference", "preference.explicit", "legacy-key"))
            )

        service.reprocess(V3)
        replayed = make_service(tmp_path, extractor_version=V3)
        try:
            replayed.process_outbox()
        finally:
            replayed.close()

        claims = {claim.id: claim for claim in claims_for(service)}
        assert claims[old.id].status == "superseded"
        assert claims[old.id].superseded_reason == "extractor_withdrawn"
        active = [claim for claim in claims.values() if claim.status == "active"]
        assert len(active) == 1 and active[0].extractor_version == V3
    finally:
        service.close()


def test_multi_evidence_claim_survives_partial_replay(tmp_path):
    service = make_service(tmp_path)
    try:
        ingest_event(service, "e1", "I prefer tea")
        ingest_event(service, "e2", "I prefer tea")
        service.process_outbox()
        claim = claims_for(service)[0]
        assert {item.event_id for item in claim.evidence} == {"e1", "e2"}
        # Move the claim to a legacy key so replay does not reproduce it.
        with service.session_factory.begin() as session:
            session.execute(
                update(Claim)
                .where(Claim.id == claim.id)
                .values(semantic_key=_semantic_key("user-a", "preference", "preference.explicit", "legacy-key"))
            )
            session.add(Outbox(id="replay-e1", event_id="e1", processor_version=V3))

        replayed = make_service(tmp_path, extractor_version=V3)
        try:
            replayed.process_outbox()
            # e2 has not been replayed under V3 yet: the claim still stands.
            with service.session_factory() as session:
                current = session.get(Claim, claim.id)
                assert current.status == "active"

            with service.session_factory.begin() as session:
                session.add(Outbox(id="replay-e2", event_id="e2", processor_version=V3))
            replayed.process_outbox()
            with service.session_factory() as session:
                current = session.get(Claim, claim.id)
                assert current.status == "superseded"
                assert current.superseded_reason == "extractor_withdrawn"
        finally:
            replayed.close()
    finally:
        service.close()


# --- D3: explicit_feedback protection ---------------------------------------


def test_explicit_feedback_claim_is_never_auto_superseded(tmp_path):
    service = make_service(tmp_path)
    try:
        ingest_event(service, "e1", "I prefer tea")
        service.process_outbox()
        old = claims_for(service)[0]
        corrected = service.feedback(
            FeedbackRequest(
                user_id="user-a",
                claim_id=old.id,
                action="correct",
                correction="I prefer coffee",
            )
        )
        replacement_id = corrected["claim_id"]

        service.reprocess(V3)
        replayed = make_service(tmp_path, extractor_version=V3)
        try:
            replayed.process_outbox()
        finally:
            replayed.close()

        with service.session_factory() as session:
            replacement = session.get(Claim, replacement_id)
            assert replacement.status == "active"
            # No machine claim was created for the protected key.
            machine = session.scalars(
                select(Claim).where(
                    Claim.user_id == "user-a",
                    Claim.source_type != "explicit_feedback",
                    Claim.status == "active",
                )
            ).all()
            assert machine == []
            audits = session.scalars(
                select(AuditEvent).where(
                    AuditEvent.event_name == "meno.claim.coordination"
                )
            ).all()
        assert audits
        assert audits[0].claim_id == replacement_id
        assert audits[0].decision["reason"] == "explicit_feedback_protected"
    finally:
        service.close()


# --- D4: health --------------------------------------------------------------


def test_health_reports_extractor_version(tmp_path):
    service = make_service(tmp_path)
    try:
        assert service.health()["extractor_version"] == V2
    finally:
        service.close()


# --- D5: feedback correct idempotency ----------------------------------------


def test_feedback_correct_is_idempotent(tmp_path):
    service = make_service(tmp_path)
    try:
        ingest_event(service, "e1", "I prefer tea")
        service.process_outbox()
        old = claims_for(service)[0]
        request = FeedbackRequest(
            user_id="user-a", claim_id=old.id, action="correct", correction="I prefer coffee"
        )
        first = service.feedback(request)
        second = service.feedback(request)
        assert second["idempotent_replay"] is True
        assert second["claim_id"] == first["claim_id"]

        claims = claims_for(service)
        active = [claim for claim in claims if claim.status == "active"]
        assert len(active) == 1
        assert active[0].id == first["claim_id"]
        assert active[0].source_type == "explicit_feedback"
        superseded = [claim for claim in claims if claim.status == "superseded"]
        assert superseded[0].superseded_reason == "feedback_correct"
        assert superseded[0].superseded_by_id == first["claim_id"]
    finally:
        service.close()


def test_feedback_correct_on_rejected_claim_fails(tmp_path):
    service = make_service(tmp_path)
    try:
        ingest_event(service, "e1", "I prefer tea")
        service.process_outbox()
        old = claims_for(service)[0]
        service.feedback(FeedbackRequest(user_id="user-a", claim_id=old.id, action="reject"))
        with pytest.raises(ValueError, match="cannot correct"):
            service.feedback(
                FeedbackRequest(
                    user_id="user-a",
                    claim_id=old.id,
                    action="correct",
                    correction="I prefer coffee",
                )
            )
    finally:
        service.close()


def test_feedback_correct_supersedes_conflicting_machine_claim(tmp_path):
    service = make_service(tmp_path)
    try:
        ingest_event(service, "e1", "I prefer tea")
        ingest_event(service, "e2", "I prefer jazz")
        service.process_outbox()
        claims = {claim.value: claim for claim in claims_for(service)}
        tea = claims["tea"]
        jazz = claims["jazz"]
        corrected = service.feedback(
            FeedbackRequest(
                user_id="user-a",
                claim_id=jazz.id,
                action="correct",
                correction="I prefer coffee",
            )
        )
        with service.session_factory() as session:
            tea_now = session.get(Claim, tea.id)
            assert tea_now.status == "superseded"
            assert tea_now.superseded_reason == "feedback_correct"
            assert tea_now.superseded_by_id == corrected["claim_id"]
            replacement = session.get(Claim, corrected["claim_id"])
            assert replacement.status == "active"
            assert replacement.semantic_key == tea.semantic_key
    finally:
        service.close()


# --- Item 6: extractor rewrite ------------------------------------------------


def test_extractor_yields_multiple_preferences_per_event():
    candidates = extract_claims(
        make_event("e1", "I prefer tea. I like Python for data work.")
    )
    assert len(candidates) == 2
    values = {candidate.value for candidate in candidates}
    assert "tea" in values
    assert any("Python" in value for value in values)
    assert all(candidate.slot for candidate in candidates)


def test_extractor_drops_low_substance_content():
    assert extract_claims(make_event("e1", "hello")) == []
    assert extract_claims(make_event("e2", "ok", role="assistant")) == []


def test_extractor_truncates_long_episodic_content():
    content = "word " * 200
    candidates = extract_claims(make_event("e1", content))
    assert len(candidates) == 1
    assert candidates[0].kind == "episodic"
    assert len(candidates[0].value) <= 281
    assert candidates[0].value.endswith("…")


def test_extractor_negation_records_negative_stance_on_the_same_object():
    """Both directions share the object so a reversal collides on the slot key.

    The old contract kept the whole sentence for negations, which made the two
    directions structurally incomparable and hid reversals from the claim chain.
    """
    affirmed = extract_claims(make_event("e0", "I prefer tea"))
    negated = extract_claims(make_event("e1", "I no longer like tea"))

    assert len(negated) == 1
    assert negated[0].stance == "negative"
    assert negated[0].slot == "beverage"
    # Same normalized object, opposite stance.
    assert negated[0].value == affirmed[0].value
    assert affirmed[0].stance == "positive"


def test_extractor_preference_without_slot_falls_back_to_value_key():
    candidates = extract_claims(make_event("e1", "I prefer xyzzy-quux"))
    assert len(candidates) == 1
    assert candidates[0].slot is None


# --- Item 7: evidence reinforcement -------------------------------------------


def make_claim(evidence_count: int, *, age_days: float = 0.0, half_life_days: float = 100.0):
    claim = Claim(
        id="claim-under-test",
        user_id="user-a",
        kind="preference",
        semantic_channel="preference.explicit",
        value="tea",
        status="active",
        confidence=0.8,
        half_life_days=half_life_days,
        sensitive=False,
        allowed_purposes=["personalization"],
        source_type="hermes_turn",
        valid_from=datetime.now(UTC) - timedelta(days=age_days),
        extractor_version=V2,
        semantic_key="sha256:key",
    )
    for index in range(evidence_count):
        claim.evidence.append(ClaimEvidence(event_id=f"event-{index}"))
    return claim


def test_single_evidence_decays_with_age(tmp_path):
    service = make_service(tmp_path)
    try:
        claim = make_claim(1, age_days=100.0)
        effective = service._effective_confidence(claim, as_of=datetime.now(UTC))
        assert effective == pytest.approx(0.4, abs=1e-3)
    finally:
        service.close()


def test_multiple_evidence_lifts_confidence(tmp_path):
    service = make_service(tmp_path)
    try:
        now = datetime.now(UTC)
        single = service._effective_confidence(make_claim(1, age_days=100.0), as_of=now)
        triple = service._effective_confidence(make_claim(3, age_days=100.0), as_of=now)
        assert triple > single
        # Reinforcement also lifts the base above the calibrated confidence.
        fresh = service._effective_confidence(make_claim(4, age_days=0.0), as_of=now)
        assert fresh > 0.8
    finally:
        service.close()


def test_mixed_reinforcement_and_decay(tmp_path):
    service = make_service(tmp_path)
    try:
        now = datetime.now(UTC)
        # Reinforced but old: still decays, yet beats a single-evidence claim.
        reinforced_old = service._effective_confidence(
            make_claim(3, age_days=200.0), as_of=now
        )
        single_old = service._effective_confidence(make_claim(1, age_days=200.0), as_of=now)
        reinforced_fresh = service._effective_confidence(
            make_claim(3, age_days=0.0), as_of=now
        )
        assert single_old < reinforced_old < reinforced_fresh
    finally:
        service.close()


def test_score_reflects_evidence_count(tmp_path):
    service = make_service(tmp_path)
    try:
        now = datetime.now(UTC)
        scores = [
            service._score(0.5, 0.5, make_claim(count), now)
            for count in (1, 2, 4)
        ]
        assert scores[0] < scores[1] < scores[2]
    finally:
        service.close()


# --- CLI -----------------------------------------------------------------------


def test_cli_migrate_and_reprocess(tmp_path, monkeypatch, capsys):
    database = tmp_path / "cli.sqlite3"
    monkeypatch.setenv("MENO_DATABASE_URL", f"sqlite:///{database}")
    monkeypatch.setenv("MENO_OPENAI_API_KEY", "test-siliconflow-key-value")
    settings = Settings(database_url=f"sqlite:///{database}", embedding_dimension=256)
    service = build_service(settings, embedder=TestEmbedder(256))
    ingest_event(service, "e1", "I prefer tea")
    service.process_outbox()
    service.close()

    monkeypatch.setattr(sys, "argv", ["meno", "migrate"])
    cli_main()
    migrate_report = ast.literal_eval(capsys.readouterr().out.strip())
    assert migrate_report["index_created"] is True

    monkeypatch.setattr(sys, "argv", ["meno", "reprocess", "--extractor-version", V3])
    cli_main()
    reprocess_report = ast.literal_eval(capsys.readouterr().out.strip())
    assert reprocess_report == {
        "extractor_version": V3,
        "targeted": 1,
        "inserted": 1,
        "cancelled": 0,
    }
