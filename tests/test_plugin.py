from __future__ import annotations

import json
import stat
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
