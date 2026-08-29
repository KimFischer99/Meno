from __future__ import annotations

import pytest

from meno.config import Settings


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        # PostgreSQL still implies Qdrant: a multi-process deployment cannot share
        # an in-process vector store.
        (
            {
                "database_url": "postgresql+psycopg://meno:x@localhost/meno",
                "vector_mode": "memory",
            },
            "Qdrant",
        ),
        ({"database_url": "mysql://meno@localhost/meno", "vector_mode": "qdrant"}, "SQLite"),
    ],
)
def test_production_rejects_degraded_backends(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Settings(
            environment="production",
            openai_api_key="test-siliconflow-key-with-more-than-20-characters",
            api_token="test-token-with-more-than-32-characters",
            **kwargs,
        ).validate()


def _sqlite_production(**overrides):
    settings = {
        "environment": "production",
        "database_url": "sqlite:////var/lib/meno/meno.db",
        "vector_mode": "memory",
        "openai_api_key": "test-siliconflow-key-with-more-than-20-characters",
        "api_token": "test-token-with-more-than-32-characters",
        **overrides,
    }
    return Settings(**settings)


def test_production_accepts_sqlite_with_an_absolute_path():
    """The small-host profile (see deploy/): SQLite canonical plus in-process
    vectors. Permitted because the SPEC No-Go forbids the *vector store* becoming
    the source of truth, not SQLite being it -- the #12 replay lane runs on SQLite."""
    _sqlite_production().validate()
    _sqlite_production(database_url="sqlite+pysqlite:////var/lib/meno/meno.db").validate()


@pytest.mark.parametrize(
    ("database_url", "message"),
    [
        # SQLAlchemy SQLite URLs need four slashes for an absolute path; three
        # leaves it relative to the working directory.
        ("sqlite:///meno.db", "absolute path"),
        ("sqlite:///./data/meno.db", "absolute path"),
        ("sqlite:///:memory:", "in-memory"),
        ("sqlite:///", "in-memory"),
        # Canonical state must survive a reboot to stay replayable.
        ("sqlite:////tmp/meno.db", "temporary directory"),
        ("sqlite:////var/tmp/meno.db", "temporary directory"),
        ("sqlite:////dev/shm/meno.db", "temporary directory"),
    ],
)
def test_production_sqlite_must_be_durable_and_absolute(database_url, message):
    with pytest.raises(ValueError, match=message):
        _sqlite_production(database_url=database_url).validate()


def test_production_memory_vectors_require_a_residency_cap():
    """Unbounded in-process vectors on a small host end in an OOM kill, which takes
    the sidecar down and breaks Charter #11 (a sidecar failure must not block Hermes)."""
    with pytest.raises(ValueError, match="MENO_VECTOR_MAX_RESIDENT"):
        _sqlite_production(vector_max_resident=0).validate()


def test_negative_residency_cap_is_rejected():
    with pytest.raises(ValueError, match="MENO_VECTOR_MAX_RESIDENT"):
        Settings(vector_max_resident=-1).validate()


@pytest.mark.parametrize("token", ["", "short-token"])
def test_sqlite_production_still_requires_the_api_token(token):
    """Load-bearing: the API middleware skips authentication entirely when the
    token is empty, so relaxing the storage requirement must not relax this."""
    with pytest.raises(ValueError, match="MENO_API_TOKEN"):
        _sqlite_production(api_token=token).validate()


def test_sqlite_production_still_requires_the_embedding_key():
    with pytest.raises(ValueError, match="SiliconFlow embedding API key"):
        _sqlite_production(openai_api_key="").validate()


def test_non_siliconflow_embedding_provider_is_rejected():
    with pytest.raises(ValueError, match="must be siliconflow"):
        Settings(embedding_provider="fastembed").validate()


def test_production_requires_api_token():
    with pytest.raises(ValueError, match="MENO_API_TOKEN"):
        Settings(
            environment="production",
            database_url="postgresql+psycopg://meno:x@localhost/meno",
            vector_mode="qdrant",
            openai_api_key="test-siliconflow-key-with-more-than-20-characters",
        ).validate()


def test_production_requires_siliconflow_api_key():
    with pytest.raises(ValueError, match="SiliconFlow embedding API key"):
        Settings(
            environment="production",
            database_url="postgresql+psycopg://meno:x@localhost/meno",
            vector_mode="qdrant",
            api_token="test-token-with-more-than-32-characters",
        ).validate()


def test_environment_supplies_siliconflow_model_and_secret(monkeypatch):
    monkeypatch.setenv("MENO_EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("MENO_OPENAI_API_KEY", "secret-value-long-enough-for-tests")
    settings = Settings.from_env()
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.openai_api_key == "secret-value-long-enough-for-tests"
    assert "secret-value" not in repr(settings)


def test_production_routing_requires_config_hash():
    with pytest.raises(ValueError, match="pinned config SHA-256"):
        Settings(
            environment="production",
            database_url="postgresql+psycopg://meno:x@localhost/meno",
            vector_mode="qdrant",
            openai_api_key="secret-value-long-enough-for-tests",
            api_token="test-token-with-more-than-32-characters",
            semantic_routing_enabled=True,
        ).validate()


def test_semantic_routing_flag_defaults_off_and_parses(monkeypatch):
    monkeypatch.delenv("MENO_SEMANTIC_ROUTING_ENABLED", raising=False)
    assert Settings.from_env().semantic_routing_enabled is False
    monkeypatch.setenv("MENO_SEMANTIC_ROUTING_ENABLED", "true")
    assert Settings.from_env().semantic_routing_enabled is True
    monkeypatch.setenv("MENO_SEMANTIC_ROUTING_ENABLED", "invalid")
    with pytest.raises(ValueError, match="must be a boolean"):
        Settings.from_env()


def test_context_activation_defaults_off_and_parses(monkeypatch):
    monkeypatch.delenv("MENO_CONTEXT_ACTIVATION_ENABLED", raising=False)
    assert Settings.from_env().context_activation_enabled is False
    monkeypatch.setenv("MENO_CONTEXT_ACTIVATION_ENABLED", "true")
    assert Settings.from_env().context_activation_enabled is True


def test_user_token_materialization_defaults_off_and_parses(monkeypatch):
    monkeypatch.delenv("MENO_USER_TOKEN_MATERIALIZATION_ENABLED", raising=False)
    assert Settings.from_env().user_token_materialization_enabled is False
    monkeypatch.setenv("MENO_USER_TOKEN_MATERIALIZATION_ENABLED", "true")
    assert Settings.from_env().user_token_materialization_enabled is True


def test_preference_distribution_requires_user_token_materialization(monkeypatch):
    monkeypatch.setenv("MENO_PREFERENCE_DISTRIBUTION_ENABLED", "true")
    with pytest.raises(ValueError, match="require.*MENO_USER_TOKEN"):
        Settings.from_env()

    monkeypatch.setenv("MENO_USER_TOKEN_MATERIALIZATION_ENABLED", "true")
    assert Settings.from_env().preference_distribution_enabled is True


def test_preference_distribution_v2_defaults_off_and_requires_v1_flag(monkeypatch):
    monkeypatch.delenv("MENO_PREFERENCE_DISTRIBUTION_V2_ENABLED", raising=False)
    assert Settings.from_env().preference_distribution_v2_enabled is False

    monkeypatch.setenv("MENO_PREFERENCE_DISTRIBUTION_V2_ENABLED", "true")
    with pytest.raises(ValueError, match="requires MENO_PREFERENCE_DISTRIBUTION"):
        Settings.from_env()

    monkeypatch.setenv("MENO_USER_TOKEN_MATERIALIZATION_ENABLED", "true")
    monkeypatch.setenv("MENO_PREFERENCE_DISTRIBUTION_ENABLED", "true")
    assert Settings.from_env().preference_distribution_v2_enabled is True


def test_clarification_requires_distribution_and_activation(monkeypatch):
    monkeypatch.setenv("MENO_CLARIFICATION_OPPORTUNITIES_ENABLED", "true")
    with pytest.raises(ValueError, match="MENO_PREFERENCE_DISTRIBUTION_ENABLED"):
        Settings.from_env()

    monkeypatch.setenv("MENO_USER_TOKEN_MATERIALIZATION_ENABLED", "true")
    monkeypatch.setenv("MENO_PREFERENCE_DISTRIBUTION_ENABLED", "true")
    with pytest.raises(ValueError, match="MENO_CONTEXT_ACTIVATION_ENABLED"):
        Settings.from_env()

    monkeypatch.setenv("MENO_CONTEXT_ACTIVATION_ENABLED", "true")
    assert Settings.from_env().clarification_opportunities_enabled is True


def test_preference_history_defaults_off_and_requires_semantic_routing(monkeypatch):
    monkeypatch.delenv("MENO_PREFERENCE_HISTORY_RETRIEVAL_ENABLED", raising=False)
    assert Settings.from_env().preference_history_retrieval_enabled is False

    monkeypatch.setenv("MENO_PREFERENCE_HISTORY_RETRIEVAL_ENABLED", "true")
    with pytest.raises(ValueError, match="requires MENO_SEMANTIC_ROUTING_ENABLED"):
        Settings.from_env()

    monkeypatch.setenv("MENO_SEMANTIC_ROUTING_ENABLED", "true")
    settings = Settings.from_env()
    assert settings.preference_history_retrieval_enabled is True
    assert settings.preference_history_max_facets == 2
    assert settings.preference_history_max_events == 4


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"preference_history_max_facets": 0}, "MAX_FACETS"),
        ({"preference_history_max_facets": 9}, "MAX_FACETS"),
        ({"preference_history_max_events": 0}, "MAX_EVENTS"),
        ({"preference_history_max_events": 17}, "MAX_EVENTS"),
    ],
)
def test_preference_history_bounds_are_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Settings(**kwargs).validate()
