from __future__ import annotations

import json
import stat
import time
from contextlib import closing
from contextvars import ContextVar
from threading import Event

import httpx
import pytest

import meno.hermes_plugin as plugin
from meno.hermes_plugin import DurableSpool, MenoMemoryProvider


def test_durable_spool_round_trip_and_permissions(tmp_path):
    path = tmp_path / "profile" / "meno" / "spool.sqlite3"
    spool = DurableSpool(path)
    payload = {"event_id": "event-1", "content": {"role": "user", "text": "hello"}}
    assert spool.enqueue(payload)
    assert spool.enqueue(payload)
    assert spool.count() == 1
    event_id, restored = spool.peek()
    assert event_id == "event-1"
    assert restored == payload
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    spool.complete(event_id)
    assert spool.count() == 0


def test_provider_uses_profile_settings_and_preserves_worker_context(tmp_path, monkeypatch):
    for key in ("MENO_API_URL", "MENO_API_TOKEN", "MENO_USER_ID", "MENO_PROVIDER_TIMEOUT_SECONDS"):
        monkeypatch.setenv(key, "other-profile")
    settings = {
        "MENO_API_URL": "http://127.0.0.1:9876/",
        "MENO_API_TOKEN": "profile-token",
        "MENO_USER_ID": "profile-user",
        "MENO_PROVIDER_TIMEOUT_SECONDS": "0.25",
    }
    scope = ContextVar("profile_settings")
    scope.set(settings)
    monkeypatch.setattr(plugin, "get_secret", lambda name, default=None: scope.get().get(name, default))
    sent = Event()
    observed = []

    def ingest(request):
        observed.append((request, scope.get()))
        sent.set()
        return httpx.Response(202, json={"accepted": True})

    client_class = httpx.Client
    monkeypatch.setattr(
        plugin.httpx, "Client",
        lambda **kwargs: client_class(**kwargs, transport=httpx.MockTransport(ingest)),
    )
    provider = MenoMemoryProvider()
    assert provider.is_available()
    assert provider._timeout == 0.25
    provider.initialize("session-a", hermes_home=tmp_path, user_id="runtime-user")
    scope.set({"MENO_API_TOKEN": "later-profile-token"})
    try:
        provider.sync_turn("I prefer concise answers", "")
        assert sent.wait(2)
    finally:
        provider.shutdown()
    request, worker_settings = observed[0]
    assert worker_settings == settings
    assert str(request.url) == "http://127.0.0.1:9876/v1/ingest"
    assert request.headers["Authorization"] == "Bearer profile-token"
    payload = json.loads(request.content)
    assert payload["user_id"] == "profile-user"
    assert payload["source"]["session_id"] == "session-a"
    assert provider._spool.path == tmp_path / "meno/provider-spool.sqlite3"
    assert provider._spool.count() == 0


def test_profile_without_token_does_not_inherit_process_token(monkeypatch):
    monkeypatch.setenv("MENO_API_TOKEN", "other-profile-token")
    monkeypatch.setattr(plugin, "get_secret", lambda name, default=None: default)
    provider = MenoMemoryProvider()
    assert not provider.is_available()
    assert "MENO_API_TOKEN" in provider.unavailable_reason()


def test_unscoped_profile_error_is_not_hidden(monkeypatch):
    monkeypatch.setenv("MENO_API_TOKEN", "other-profile-token")

    def unscoped(name, default=None):
        raise RuntimeError("No active profile secret scope")

    monkeypatch.setattr(plugin, "get_secret", unscoped)
    with pytest.raises(RuntimeError, match="No active profile secret scope"):
        MenoMemoryProvider()


def _start_provider(tmp_path, monkeypatch, handler):
    monkeypatch.setattr(plugin, "get_secret", lambda name, default=None: {
        "MENO_API_URL": "http://127.0.0.1:9876/",
        "MENO_API_TOKEN": "token",
        "MENO_USER_ID": "user",
    }.get(name, default))
    client_class = httpx.Client
    monkeypatch.setattr(
        plugin.httpx, "Client",
        lambda **kwargs: client_class(**kwargs, transport=httpx.MockTransport(handler)),
    )
    provider = MenoMemoryProvider()
    provider.initialize("session-a", hermes_home=tmp_path)
    return provider


def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_spool_dead_letter_moves_row_and_truncates_reason(tmp_path):
    spool = DurableSpool(tmp_path / "spool.sqlite3")
    spool.enqueue({"event_id": "bad", "content": {"text": "x"}})
    spool.enqueue({"event_id": "good", "content": {"text": "y"}})
    spool.dead_letter("bad", 422, "r" * 10_000)
    assert spool.count() == 1
    assert spool.dead_letter_count() == 1
    assert spool.peek()[0] == "good"
    with closing(spool._connect()) as connection:
        status, reason = connection.execute("SELECT status_code, reason FROM dead_letter").fetchone()
    assert status == 422
    assert len(reason) == plugin.DEAD_LETTER_REASON_CHARS


@pytest.mark.parametrize("status", [400, 413, 422])
def test_rejected_event_is_dead_lettered_and_does_not_block_queue(tmp_path, monkeypatch, status):
    delivered = []

    def ingest(request):
        payload = json.loads(request.content)
        if payload["content"]["text"] == "bad":
            return httpx.Response(status, json={"detail": "rejected"})
        delivered.append(payload["content"]["text"])
        return httpx.Response(202, json={"accepted": True})

    provider = _start_provider(tmp_path, monkeypatch, ingest)
    try:
        provider.sync_turn("bad", "good")
        assert _wait_for(lambda: delivered == ["good"])
        assert _wait_for(lambda: provider._spool.count() == 0)
    finally:
        provider.shutdown()
    assert provider._spool.dead_letter_count() == 1


def test_spool_keeps_accepting_events_behind_rejected_event(tmp_path, monkeypatch):
    delivered = []

    def ingest(request):
        text = json.loads(request.content)["content"]["text"]
        if text == "bad":
            return httpx.Response(422, json={"detail": "too long"})
        delivered.append(text)
        return httpx.Response(202, json={"accepted": True})

    provider = _start_provider(tmp_path, monkeypatch, ingest)
    provider._spool.max_pending = 1
    try:
        provider.sync_turn("bad", "")
        for index in range(5):
            assert _wait_for(lambda: provider._spool.count() == 0)
            provider.sync_turn(f"turn-{index}", "")
        assert _wait_for(lambda: len(delivered) == 5)
    finally:
        provider.shutdown()
    assert provider._spool.dead_letter_count() == 1


@pytest.mark.parametrize("failure", [500, 429, 408, "connect"])
def test_transient_failures_are_retried_and_row_stays_spooled(tmp_path, monkeypatch, failure):
    attempts = []

    def ingest(request):
        attempts.append(1)
        if len(attempts) < 2:
            if failure == "connect":
                raise httpx.ConnectError("down", request=request)
            return httpx.Response(failure, json={"detail": "later"})
        return httpx.Response(202, json={"accepted": True})

    provider = _start_provider(tmp_path, monkeypatch, ingest)
    try:
        provider.sync_turn("hello", "")
        assert _wait_for(lambda: provider._spool.count() == 0, timeout=5.0)
        assert len(attempts) == 2
    finally:
        provider.shutdown()
    assert provider._spool.dead_letter_count() == 0
