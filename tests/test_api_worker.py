from __future__ import annotations

import asyncio
import logging
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from meno.db import AuditEvent, Claim
from tests.test_api import retrieve

DEADLINE_SECONDS = 10.0


def wait_until(predicate, what: str):
    deadline = time.monotonic() + DEADLINE_SECONDS
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def post_ingest(client, text: str):
    response = client.post(
        "/v1/ingest",
        headers={"Idempotency-Key": f"key-{hash(text)}"},
        json={
            "user_id": "user-a",
            "source": {"type": "hermes_turn", "profile": "meno-test", "session_id": "s1"},
            "content": {"role": "user", "text": text},
            "consent_scope": ["personalization", "task_planning"],
        },
    )
    assert response.status_code == 202, response.text


def retrieved_facets(client, query: str):
    return retrieve(client, "user-a", query).json()["facets"]


def claim_count(service) -> int:
    with service.session_factory() as session:
        return session.scalar(select(func.count()).select_from(Claim))


def test_worker_survives_transient_outbox_failure(app, monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger="meno.api")
    with TestClient(app) as client:
        service = client.app.state.meno
        real = service.process_outbox
        state = {"failures_left": 3, "calls": 0}

        def flaky(*args, **kwargs):
            state["calls"] += 1
            if state["failures_left"] > 0:
                state["failures_left"] -= 1
                raise ConnectionError("database connection dropped")
            return real(*args, **kwargs)

        monkeypatch.setattr(service, "process_outbox", flaky)
        post_ingest(client, "以后默认先给结论，再给细节。")

        facets = wait_until(
            lambda: retrieved_facets(client, "我希望回答采用什么结构？"),
            "claim to be projected after outbox recovery",
        )
        assert any("先给结论" in facet["value"] for facet in facets)
        assert state["failures_left"] == 0
        assert state["calls"] > 3
        # Let a few more rounds run: reprocessing must not duplicate the claim.
        calls = state["calls"]
        wait_until(lambda: state["calls"] >= calls + 3, "extra worker rounds")
        assert claim_count(service) == len(facets) == 1
    assert "meno outbox worker round failed" in caplog.text


def test_worker_survives_audit_flush_failure(app, monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger="meno.api")
    with TestClient(app) as client:
        service = client.app.state.meno
        real = service.flush_audit_buffer
        state = {"failures_left": 1, "calls": 0}

        def flaky(*args, **kwargs):
            state["calls"] += 1
            if state["failures_left"] > 0:
                state["failures_left"] -= 1
                raise ConnectionError("audit flush failed")
            return real(*args, **kwargs)

        monkeypatch.setattr(service, "flush_audit_buffer", flaky)
        wait_until(lambda: state["failures_left"] == 0, "injected flush failure")

        # The worker must keep going: the outbox event still becomes a claim ...
        post_ingest(client, "以后默认先给结论，再给细节。")
        facets = wait_until(
            lambda: retrieved_facets(client, "我希望回答采用什么结构？"),
            "claim to be projected after flush failure",
        )
        assert facets

        # ... and the audit entries buffered by retrieval reach the database.
        def retrieve_audits() -> int:
            with service.session_factory() as session:
                return session.scalar(
                    select(func.count())
                    .select_from(AuditEvent)
                    .where(AuditEvent.action == "retrieve")
                )

        wait_until(retrieve_audits, "buffered retrieve audit to be flushed")
    assert "meno outbox worker round failed" in caplog.text


def test_shutdown_closes_service_when_final_flush_fails(app, monkeypatch):
    closed = []
    with pytest.raises(RuntimeError, match="final flush failed"), TestClient(app) as client:
        service = client.app.state.meno
        real_flush = service.flush_audit_buffer
        real_close = service.close

        def flush(*args, **kwargs):
            # The worker flushes from a thread; only the final flush from
            # lifespan shutdown runs on the event loop itself.
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return real_flush(*args, **kwargs)
            raise RuntimeError("final flush failed")

        def close():
            workers = [
                task for task in asyncio.all_tasks() if task.get_name() == "meno-outbox-worker"
            ]
            closed.append(all(task.done() for task in workers))
            real_close()

        monkeypatch.setattr(service, "flush_audit_buffer", flush)
        monkeypatch.setattr(service, "close", close)
    # close() ran exactly once, after the background worker had finished.
    assert closed == [True]
