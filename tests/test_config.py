from __future__ import annotations

import pytest

from meno.config import Settings


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"database_url": "sqlite:///meno.db", "vector_mode": "qdrant"}, "PostgreSQL"),
        (
            {
                "database_url": "postgresql+psycopg://meno:x@localhost/meno",
                "vector_mode": "memory",
            },
            "Qdrant",
        ),
    ],
)
def test_production_rejects_degraded_backends(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Settings(
            environment="production",
            google_api_key="test-google-key-with-more-than-20-characters",
            api_token="test-token-with-more-than-32-characters",
            **kwargs,
        ).validate()


def test_non_google_embedding_provider_is_rejected():
    with pytest.raises(ValueError, match="must be google"):
        Settings(embedding_provider="fastembed").validate()


def test_production_requires_api_token():
    with pytest.raises(ValueError, match="MENO_API_TOKEN"):
        Settings(
            environment="production",
            database_url="postgresql+psycopg://meno:x@localhost/meno",
            vector_mode="qdrant",
            google_api_key="test-google-key-with-more-than-20-characters",
        ).validate()


def test_production_requires_google_api_key():
    with pytest.raises(ValueError, match="Google embedding API key"):
        Settings(
            environment="production",
            database_url="postgresql+psycopg://meno:x@localhost/meno",
            vector_mode="qdrant",
            api_token="test-token-with-more-than-32-characters",
        ).validate()


def test_credentials_file_supplies_model_and_secret(tmp_path, monkeypatch):
    credentials = tmp_path / "google.txt"
    credentials.write_text("Model: gemini-embedding-001\nKey: secret-value-long-enough-for-tests\n")
    credentials.chmod(0o600)
    monkeypatch.setenv("MENO_GOOGLE_CREDENTIALS_FILE", str(credentials))
    settings = Settings.from_env()
    assert settings.embedding_model == "gemini-embedding-001"
    assert settings.google_api_key == "secret-value-long-enough-for-tests"
    assert "secret-value" not in repr(settings)


def test_production_rejects_world_readable_credentials(tmp_path):
    credentials = tmp_path / "google.txt"
    credentials.write_text("Key: secret-value-long-enough-for-tests\n")
    credentials.chmod(0o644)
    with pytest.raises(ValueError, match="group/world"):
        Settings(
            environment="production",
            database_url="postgresql+psycopg://meno:x@localhost/meno",
            vector_mode="qdrant",
            google_credentials_file=str(credentials),
            google_api_key="secret-value-long-enough-for-tests",
            api_token="test-token-with-more-than-32-characters",
        ).validate()
