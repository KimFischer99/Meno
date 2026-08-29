from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from meno.api import create_app
from meno.config import Settings
from tests.fakes import TestEmbedder


@pytest.fixture
def app(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'meno.sqlite3'}",
        vector_mode="memory",
        embedding_dimension=256,
        worker_poll_seconds=0.01,
    )
    return create_app(settings, embedder=TestEmbedder(settings.embedding_dimension))


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def ingest(client):
    def _ingest(
        text: str,
        *,
        user_id: str = "user-a",
        role: str = "user",
        source: str = "hermes_turn",
        scopes: list[str] | None = None,
    ):
        response = client.post(
            "/v1/ingest",
            headers={"Idempotency-Key": f"key-{user_id}-{hash(text)}"},
            json={
                "user_id": user_id,
                "source": {"type": source, "profile": "meno-test", "session_id": "s1"},
                "content": {"role": role, "text": text},
                "consent_scope": scopes or ["personalization", "task_planning"],
            },
        )
        assert response.status_code == 202, response.text
        client.app.state.meno.process_outbox()
        return response.json()

    return _ingest
