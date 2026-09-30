from dataclasses import replace
from threading import Event, Thread

from sqlalchemy import select

from meno.db import AuditEvent
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
