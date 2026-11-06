from dataclasses import replace
from datetime import UTC, datetime, timedelta
from threading import Event, Thread

from sqlalchemy import func, select

from meno.db import AuditEvent, Claim, Outbox
from meno.db import Event as EventRow
from meno.hermes_plugin import DurableSpool, MenoMemoryProvider


def audit_entry(index):
    return {
        "event_name": "meno.test",
        "trace_id": f"trace-{index}",
        "user_id": "test-user",
        "action": "test",
        "purpose": None,
        "decision": {"allowed": True},
        "revision": index,
        "event_ids": [],
    }


def test_audit_spill_failure_retains_events(app, monkeypatch):
    service = app.state.meno
    service.settings = replace(service.settings, audit_buffer_max=1)
    monkeypatch.setattr(service, "_spill_audit_event", lambda entry: False)
    service._buffer_audit(**audit_entry(1))
    service._buffer_audit(**audit_entry(2))
    assert service.flush_audit_buffer() == 2
    with service.session_factory() as session:
        assert list(session.scalars(select(AuditEvent.trace_id))) == ["trace-1", "trace-2"]
    service.close()


def test_concurrent_audit_flush_keeps_new_overflow(app, monkeypatch):
    service = app.state.meno
    service.settings = replace(service.settings, audit_buffer_max=1)
    service._buffer_audit(**audit_entry(1))
    service._buffer_audit(**audit_entry(2))
    flushing, release = Event(), Event()
    clear = service._clear_audit_spill

    def paused_clear():
        flushing.set()
        assert release.wait(3)
        clear()

    monkeypatch.setattr(service, "_clear_audit_spill", paused_clear)
    flush = Thread(target=service.flush_audit_buffer)
    flush.start()
    assert flushing.wait(3)
    producer = Thread(target=lambda: [service._buffer_audit(**audit_entry(i)) for i in (3, 4)])
    producer.start()
    release.set()
    flush.join(3)
    producer.join(3)
    assert not flush.is_alive() and not producer.is_alive()
    assert service.flush_audit_buffer() == 2
    with service.session_factory() as session:
        assert list(session.scalars(select(AuditEvent.trace_id))) == [
            "trace-1",
            "trace-2",
            "trace-3",
            "trace-4",
        ]
    service.close()


def test_consent_key_replays_and_conflicts_return_400(client):
    body = {
        "user_id": "test-user",
        "source": "hermes_turn",
        "purpose": "response_personalization",
        "allowed_operations": ["retrieve"],
        "status": "active",
    }
    headers = {"Idempotency-Key": "test-consent"}
    first = client.post("/v1/consents", headers=headers, json=body)
    replay = client.post("/v1/consents", headers=headers, json=body)
    assert first.status_code == replay.status_code == 200
    assert replay.json()["consent_id"] == first.json()["consent_id"]
    assert replay.json()["idempotent_replay"] is True
    conflict = client.post("/v1/consents", headers=headers, json={**body, "status": "revoked"})
    assert conflict.status_code == 400


def test_queued_hermes_turn_cannot_restore_deleted_memory(client, tmp_path):
    provider = MenoMemoryProvider()
    provider._user_id = "queued-user"
    provider._spool = DurableSpool(tmp_path / "provider-spool.sqlite3")
    provider.sync_turn("I prefer terse answers", "")
    _, payload = provider._spool.peek()
    assert payload["occurred_at"]
    deletion = client.post(
        "/v1/deletions",
        headers={"Idempotency-Key": "forget-queued-user"},
        json={"user_id": "queued-user", "scope": "all"},
    )
    assert deletion.status_code == 200
    replay = client.post(
        "/v1/ingest",
        headers={"Idempotency-Key": payload["event_id"]},
        json=payload,
    )
    assert replay.status_code == 202
    assert replay.json()["reason"] == "before_deletion_watermark"
    assert replay.json()["accepted"] is False


def test_batch_replay_cannot_restore_deleted_memory(client):
    def batch_event(event_id, text, occurred_at):
        return {
            "user_id": "batch-deleted-user",
            "event_id": event_id,
            "occurred_at": occurred_at.isoformat(),
            "source": {"type": "hermes_turn", "session_id": "batch-session"},
            "content": {"role": "user", "text": text},
            "consent_scope": ["personalization"],
        }

    def post_batch(key, events):
        response = client.post(
            "/v1/ingest/batch", headers={"Idempotency-Key": key}, json={"events": events}
        )
        assert response.status_code == 202, response.text
        return response.json()

    def retrieve_facets():
        response = client.post(
            "/v1/retrieve",
            json={
                "user_id": "batch-deleted-user",
                "purpose": "response_personalization",
                "context": {"query": "project codename", "task_type": "test"},
                "constraints": {"min_confidence": 0.5},
            },
        )
        return response.json()["facets"]

    def row_counts(service):
        with service.session_factory() as session:
            return tuple(
                session.scalar(select(func.count()).select_from(model))
                for model in (EventRow, Outbox, Claim)
            )

    service = client.app.state.meno
    past = datetime.now(UTC) - timedelta(days=1)
    original = [
        batch_event("batch-del-1", "My project codename is cedar", past),
        batch_event("batch-del-2", "I prefer Python for data work", past),
    ]
    assert post_batch("batch-del-first", original)["accepted"] == 2
    service.process_outbox()
    assert retrieve_facets()

    deletion = client.post(
        "/v1/deletions",
        headers={"Idempotency-Key": "forget-batch-deleted-user"},
        json={"user_id": "batch-deleted-user", "scope": "all"},
    )
    assert deletion.status_code == 200
    assert retrieve_facets() == []
    counts_after_delete = row_counts(service)

    replay = post_batch("batch-del-replay", original)
    assert replay["accepted"] == 0
    assert [item["event_id"] for item in replay["events"]] == ["batch-del-1", "batch-del-2"]
    for item in replay["events"]:
        assert item["accepted"] is False
        assert item["skipped"] is True
        assert item["reason"] == "before_deletion_watermark"
    service.process_outbox()
    assert row_counts(service) == counts_after_delete
    assert retrieve_facets() == []

    # Events newer than the watermark in the same batch are still accepted.
    future = datetime.now(UTC) + timedelta(days=1)
    mixed = post_batch(
        "batch-del-mixed",
        [
            original[0],
            batch_event("batch-del-3", "My project codename is willow", future),
        ],
    )
    assert mixed["accepted"] == 1
    skipped, accepted = mixed["events"]
    assert skipped["skipped"] is True and skipped["reason"] == "before_deletion_watermark"
    assert accepted["event_id"] == "batch-del-3"
    assert accepted.get("skipped") is None
    assert accepted["idempotent_replay"] is False
    with service.session_factory() as session:
        assert session.get(EventRow, "batch-del-3") is not None
        assert session.get(EventRow, "batch-del-1") is None
