from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name, "true" if default else "false").casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class Settings:
    environment: str = "development"
    database_url: str = "sqlite:///./meno.db"
    vector_mode: str = "memory"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "meno_claims_bge_m3_1024_v1"
    embedding_provider: str = "siliconflow"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    embedding_projection_version: str = "siliconflow-bge-m3-1024-v1"
    openai_api_key: str = field(default="", repr=False)
    openai_base_url: str = "https://api.siliconflow.cn/v1"
    openai_timeout_seconds: float = 15.0
    openai_batch_size: int = 32
    openai_max_retries: int = 4
    openai_circuit_breaker_threshold: int = 3
    openai_circuit_breaker_seconds: float = 30.0
    vector_upsert_batch_size: int = 128
    # Cap on vectors held in memory by MemoryVectorStore. At 1024 dims each costs
    # ~4.1 KB as array("f"), so 20k is ~80 MB -- sized for a 2 GB single host.
    # 0 disables the cap (development and tests).
    vector_max_resident: int = 20_000
    outbox_commit_batch_size: int = 32
    outbox_retry_base_seconds: float = 1.0
    outbox_retry_max_seconds: float = 300.0
    audit_buffer_max: int = 10_000
    # Overflow file for audit events that no longer fit in the in-memory buffer.
    # Empty means "next to the SQLite database file". Audit events are never dropped.
    audit_spill_path: str = ""
    policy_version: str = "meno-policy-2.0.0"
    extractor_version: str = "meno-extractor-2.1.0"
    semantic_routing_enabled: bool = False
    context_activation_enabled: bool = False
    user_token_materialization_enabled: bool = False
    user_token_snapshot_base_interval: int = 100
    preference_distribution_enabled: bool = False
    preference_distribution_v2_enabled: bool = False
    clarification_opportunities_enabled: bool = False
    semantic_router_config_file: str = (
        "benchmarks/fixtures/semantic-router-phaseb-frozen-config.json"
    )
    semantic_router_config_sha256: str = ""
    preference_history_retrieval_enabled: bool = False
    preference_history_max_facets: int = 2
    preference_history_max_events: int = 4
    layered_retrieval_enabled: bool = False
    layered_state_facet_reserve: float = 0.5
    layered_episodic_length_neutralized: bool = True
    layered_render_skip_oversized: bool = True
    evidence_selection_enabled: bool = False
    evidence_selection_redundancy_penalty: float = 0.5
    # Deterministic reflection: derive `pattern` claims from repeated preferences.
    # See src/meno/reflection.py; default off like every other state-layer flag.
    reflection_enabled: bool = False
    worker_poll_seconds: float = 0.2
    api_host: str = "127.0.0.1"
    api_port: int = 8765
    api_token: str = field(default="", repr=False)

    @classmethod
    def from_env(cls) -> Settings:
        explicit_model = _env("MENO_EMBEDDING_MODEL", "")
        settings = cls(
            environment=_env("MENO_ENV", "development"),
            database_url=_env("MENO_DATABASE_URL", "sqlite:///./meno.db"),
            vector_mode=_env("MENO_VECTOR_MODE", "memory"),
            qdrant_url=_env("MENO_QDRANT_URL", "http://127.0.0.1:6333"),
            qdrant_collection=_env("MENO_QDRANT_COLLECTION", "meno_claims_bge_m3_1024_v1"),
            embedding_provider=_env("MENO_EMBEDDING_PROVIDER", "siliconflow"),
            embedding_model=explicit_model or "BAAI/bge-m3",
            embedding_dimension=int(_env("MENO_EMBEDDING_DIMENSION", "1024")),
            embedding_projection_version=_env(
                "MENO_EMBEDDING_PROJECTION_VERSION",
                "siliconflow-bge-m3-1024-v1",
            ),
            openai_api_key=_env("MENO_OPENAI_API_KEY", ""),
            openai_base_url=_env("MENO_OPENAI_BASE_URL", "https://api.siliconflow.cn/v1").rstrip(
                "/"
            ),
            openai_timeout_seconds=float(_env("MENO_OPENAI_TIMEOUT_SECONDS", "15")),
            openai_batch_size=int(_env("MENO_OPENAI_BATCH_SIZE", "32")),
            openai_max_retries=int(_env("MENO_OPENAI_MAX_RETRIES", "4")),
            openai_circuit_breaker_threshold=int(
                _env("MENO_OPENAI_CIRCUIT_BREAKER_THRESHOLD", "3")
            ),
            openai_circuit_breaker_seconds=float(_env("MENO_OPENAI_CIRCUIT_BREAKER_SECONDS", "30")),
            vector_upsert_batch_size=int(_env("MENO_VECTOR_UPSERT_BATCH_SIZE", "128")),
            vector_max_resident=int(_env("MENO_VECTOR_MAX_RESIDENT", "20000")),
            outbox_commit_batch_size=int(_env("MENO_OUTBOX_COMMIT_BATCH_SIZE", "32")),
            outbox_retry_base_seconds=float(_env("MENO_OUTBOX_RETRY_BASE_SECONDS", "1")),
            outbox_retry_max_seconds=float(_env("MENO_OUTBOX_RETRY_MAX_SECONDS", "300")),
            audit_buffer_max=int(_env("MENO_AUDIT_BUFFER_MAX", "10000")),
            audit_spill_path=_env("MENO_AUDIT_SPILL_PATH", ""),
            policy_version=_env("MENO_POLICY_VERSION", "meno-policy-2.0.0"),
            extractor_version=_env("MENO_EXTRACTOR_VERSION", "meno-extractor-2.1.0"),
            semantic_routing_enabled=_env_bool("MENO_SEMANTIC_ROUTING_ENABLED", False),
            context_activation_enabled=_env_bool("MENO_CONTEXT_ACTIVATION_ENABLED", False),
            user_token_materialization_enabled=_env_bool(
                "MENO_USER_TOKEN_MATERIALIZATION_ENABLED", False
            ),
            user_token_snapshot_base_interval=int(_env("MENO_SNAPSHOT_BASE_INTERVAL", "100")),
            preference_distribution_enabled=_env_bool(
                "MENO_PREFERENCE_DISTRIBUTION_ENABLED", False
            ),
            preference_distribution_v2_enabled=_env_bool(
                "MENO_PREFERENCE_DISTRIBUTION_V2_ENABLED", False
            ),
            clarification_opportunities_enabled=_env_bool(
                "MENO_CLARIFICATION_OPPORTUNITIES_ENABLED", False
            ),
            semantic_router_config_file=_env(
                "MENO_SEMANTIC_ROUTER_CONFIG_FILE",
                "benchmarks/fixtures/semantic-router-phaseb-frozen-config.json",
            ),
            semantic_router_config_sha256=_env("MENO_SEMANTIC_ROUTER_CONFIG_SHA256", ""),
            preference_history_retrieval_enabled=_env_bool(
                "MENO_PREFERENCE_HISTORY_RETRIEVAL_ENABLED", False
            ),
            preference_history_max_facets=int(_env("MENO_PREFERENCE_HISTORY_MAX_FACETS", "2")),
            preference_history_max_events=int(_env("MENO_PREFERENCE_HISTORY_MAX_EVENTS", "4")),
            layered_retrieval_enabled=_env_bool("MENO_LAYERED_RETRIEVAL_ENABLED", False),
            layered_state_facet_reserve=float(_env("MENO_LAYERED_STATE_FACET_RESERVE", "0.5")),
            layered_episodic_length_neutralized=_env_bool(
                "MENO_LAYERED_EPISODIC_LENGTH_NEUTRALIZED", True
            ),
            layered_render_skip_oversized=_env_bool(
                "MENO_LAYERED_RENDER_SKIP_OVERSIZED", True
            ),
            evidence_selection_enabled=_env_bool("MENO_EVIDENCE_SELECTION_ENABLED", False),
            evidence_selection_redundancy_penalty=float(
                _env("MENO_EVIDENCE_SELECTION_REDUNDANCY_PENALTY", "0.5")
            ),
            reflection_enabled=_env_bool("MENO_REFLECTION_ENABLED", False),
            worker_poll_seconds=float(_env("MENO_WORKER_POLL_SECONDS", "0.2")),
            api_host=_env("MENO_API_HOST", "127.0.0.1"),
            api_port=int(_env("MENO_API_PORT", "8765")),
            api_token=_env("MENO_API_TOKEN", ""),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not 10 <= self.user_token_snapshot_base_interval <= 1000:
            raise ValueError("MENO_SNAPSHOT_BASE_INTERVAL must be between 10 and 1000")
        if self.embedding_provider != "siliconflow":
            raise ValueError("MENO_EMBEDDING_PROVIDER must be siliconflow")
        if not re.fullmatch(r"[A-Za-z0-9./_-]+", self.embedding_model):
            raise ValueError("MENO_EMBEDDING_MODEL is invalid")
        if not 128 <= self.embedding_dimension <= 3072:
            raise ValueError("embedding dimension must be between 128 and 3072")
        if not self.embedding_projection_version:
            raise ValueError("MENO_EMBEDDING_PROJECTION_VERSION is required")
        if not 1 <= self.vector_upsert_batch_size <= 256:
            raise ValueError("MENO_VECTOR_UPSERT_BATCH_SIZE must be between 1 and 256")
        if not 1 <= self.openai_batch_size <= 128:
            raise ValueError("MENO_OPENAI_BATCH_SIZE must be between 1 and 128")
        if self.openai_timeout_seconds <= 0:
            raise ValueError("MENO_OPENAI_TIMEOUT_SECONDS must be positive")
        if not 0 <= self.openai_max_retries <= 8:
            raise ValueError("MENO_OPENAI_MAX_RETRIES must be between 0 and 8")
        if not 1 <= self.openai_circuit_breaker_threshold <= 32:
            raise ValueError("MENO_OPENAI_CIRCUIT_BREAKER_THRESHOLD must be between 1 and 32")
        if self.openai_circuit_breaker_seconds <= 0:
            raise ValueError("MENO_OPENAI_CIRCUIT_BREAKER_SECONDS must be positive")
        if not 1 <= self.outbox_commit_batch_size <= 256:
            raise ValueError("MENO_OUTBOX_COMMIT_BATCH_SIZE must be between 1 and 256")
        if self.outbox_retry_base_seconds <= 0:
            raise ValueError("MENO_OUTBOX_RETRY_BASE_SECONDS must be positive")
        if self.outbox_retry_max_seconds < self.outbox_retry_base_seconds:
            raise ValueError(
                "MENO_OUTBOX_RETRY_MAX_SECONDS must be >= MENO_OUTBOX_RETRY_BASE_SECONDS"
            )
        if self.audit_buffer_max < 1:
            raise ValueError("MENO_AUDIT_BUFFER_MAX must be positive")
        if self.semantic_routing_enabled and not self.semantic_router_config_file:
            raise ValueError("MENO_SEMANTIC_ROUTER_CONFIG_FILE is required")
        if self.semantic_router_config_sha256 and not re.fullmatch(
            r"[0-9a-f]{64}", self.semantic_router_config_sha256
        ):
            raise ValueError("MENO_SEMANTIC_ROUTER_CONFIG_SHA256 must be lowercase SHA-256")
        if self.preference_history_retrieval_enabled and not self.semantic_routing_enabled:
            raise ValueError("preference history retrieval requires MENO_SEMANTIC_ROUTING_ENABLED")
        if self.preference_distribution_enabled and not self.user_token_materialization_enabled:
            raise ValueError(
                "preference distributions require MENO_USER_TOKEN_MATERIALIZATION_ENABLED"
            )
        if self.preference_distribution_v2_enabled and not self.preference_distribution_enabled:
            raise ValueError(
                "preference distribution v2 requires MENO_PREFERENCE_DISTRIBUTION_ENABLED"
            )
        if self.clarification_opportunities_enabled and not self.preference_distribution_enabled:
            raise ValueError(
                "clarification opportunities require MENO_PREFERENCE_DISTRIBUTION_ENABLED"
            )
        if self.clarification_opportunities_enabled and not self.context_activation_enabled:
            raise ValueError("clarification opportunities require MENO_CONTEXT_ACTIVATION_ENABLED")
        if not 1 <= self.preference_history_max_facets <= 8:
            raise ValueError("MENO_PREFERENCE_HISTORY_MAX_FACETS must be between 1 and 8")
        if not 1 <= self.preference_history_max_events <= 16:
            raise ValueError("MENO_PREFERENCE_HISTORY_MAX_EVENTS must be between 1 and 16")
        if not 0 <= self.layered_state_facet_reserve <= 1:
            raise ValueError("MENO_LAYERED_STATE_FACET_RESERVE must be between 0 and 1")
        if not 0 <= self.evidence_selection_redundancy_penalty <= 1:
            raise ValueError(
                "MENO_EVIDENCE_SELECTION_REDUNDANCY_PENALTY must be between 0 and 1"
            )
        if self.vector_mode not in {"memory", "qdrant"}:
            raise ValueError("MENO_VECTOR_MODE must be memory or qdrant")
        if self.vector_max_resident < 0:
            raise ValueError("MENO_VECTOR_MAX_RESIDENT must not be negative")
        if self.environment == "production":
            # Two supported production profiles:
            #   postgresql + qdrant  -- horizontal headroom, multi-process
            #   sqlite + memory      -- single small host (see deploy/)
            # SQLite as canonical does not breach the SPEC No-Go: what that
            # forbids is the vector store becoming the source of truth. The #12
            # replay lane already runs on SQLite and passes.
            postgres = self.database_url.startswith(
                ("postgresql://", "postgresql+psycopg://")
            )
            sqlite = self.database_url.startswith(("sqlite:", "sqlite+"))
            if not (postgres or sqlite):
                raise ValueError("production requires PostgreSQL or SQLite")
            if sqlite:
                self._validate_production_sqlite()
            if postgres and self.vector_mode != "qdrant":
                raise ValueError("production PostgreSQL requires Qdrant")
            if self.vector_mode == "memory" and self.vector_max_resident <= 0:
                # Unbounded in-memory vectors on a small host end in an OOM kill,
                # which takes the sidecar down and breaks Charter #11.
                raise ValueError(
                    "production with vector_mode=memory requires a positive "
                    "MENO_VECTOR_MAX_RESIDENT"
                )
            if len(self.openai_api_key) < 20:
                raise ValueError("production requires a SiliconFlow embedding API key")
            # Load-bearing: the API middleware skips authentication entirely when
            # api_token is empty, so an unset token publishes every user's memory.
            if len(self.api_token) < 32:
                raise ValueError("production requires MENO_API_TOKEN with at least 32 characters")
            if self.semantic_routing_enabled and not self.semantic_router_config_sha256:
                raise ValueError("production semantic routing requires a pinned config SHA-256")

    def _validate_production_sqlite(self) -> None:
        """A production SQLite file must survive a reboot and be private.

        Canonical state has to be replayable (SPEC :1393), so a path that a
        restart or a tmp sweep can erase is not a canonical store.

        SQLAlchemy's SQLite URLs put the path after three slashes, and an absolute
        path therefore has four: `sqlite:///rel.db` is relative to the working
        directory, `sqlite:////var/lib/meno/meno.db` is absolute.
        """
        _, _, remainder = self.database_url.partition("sqlite")
        # Strip an optional driver suffix such as sqlite+pysqlite:
        if remainder.startswith("+"):
            _, _, remainder = remainder.partition(":")
        elif remainder.startswith(":"):
            remainder = remainder[1:]
        if not remainder.startswith("///"):
            raise ValueError("production SQLite URL is malformed")
        path = remainder[3:]
        if path in {":memory:", ""} or path.startswith(":memory:"):
            raise ValueError("production SQLite must not be an in-memory database")
        if not path.startswith("/"):
            raise ValueError(
                "production SQLite requires an absolute path (four slashes), e.g. "
                "sqlite:////var/lib/meno/meno.db"
            )
        if path.startswith(("/tmp/", "/var/tmp/", "/dev/shm/")):
            raise ValueError(
                "production SQLite must not live under a temporary directory; "
                "canonical state must survive a reboot"
            )
