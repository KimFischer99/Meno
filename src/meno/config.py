from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _read_google_credentials(path_value: str) -> dict[str, str]:
    if not path_value:
        return {}
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise ValueError(f"Google credentials file does not exist: {path}")
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        separator = ":" if ":" in line else ("=" if "=" in line else "")
        if not separator:
            if "key" in result:
                raise ValueError("Google credentials file contains an unlabelled extra value")
            result["key"] = line
            continue
        label, value = line.split(separator, 1)
        normalized = label.strip().casefold().replace("_", " ")
        if normalized in {"key", "api key", "google api key", "gemini api key"}:
            result["key"] = value.strip()
        elif normalized in {"model", "embedding model", "google embedding model"}:
            result["model"] = value.strip()
    return result


@dataclass(frozen=True)
class Settings:
    environment: str = "development"
    database_url: str = "sqlite:///./meno.db"
    vector_mode: str = "memory"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "meno_claims_gemini_768_v1"
    embedding_provider: str = "google"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimension: int = 768
    embedding_projection_version: str = "gemini-embedding-001-768-v1"
    google_credentials_file: str = ""
    google_api_key: str = field(default="", repr=False)
    google_api_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    google_timeout_seconds: float = 15.0
    google_batch_size: int = 32
    google_max_retries: int = 4
    vector_upsert_batch_size: int = 128
    policy_version: str = "meno-policy-1.0.0"
    extractor_version: str = "meno-extractor-1.0.0"
    worker_poll_seconds: float = 0.2
    api_host: str = "127.0.0.1"
    api_port: int = 8765
    api_token: str = field(default="", repr=False)

    @classmethod
    def from_env(cls) -> Settings:
        credentials_file = _env("MENO_GOOGLE_CREDENTIALS_FILE", "")
        credentials = _read_google_credentials(credentials_file)
        explicit_model = _env("MENO_EMBEDDING_MODEL", "")
        settings = cls(
            environment=_env("MENO_ENV", "development"),
            database_url=_env("MENO_DATABASE_URL", "sqlite:///./meno.db"),
            vector_mode=_env("MENO_VECTOR_MODE", "memory"),
            qdrant_url=_env("MENO_QDRANT_URL", "http://127.0.0.1:6333"),
            qdrant_collection=_env(
                "MENO_QDRANT_COLLECTION", "meno_claims_gemini_768_v1"
            ),
            embedding_provider=_env("MENO_EMBEDDING_PROVIDER", "google"),
            embedding_model=explicit_model
            or credentials.get("model", "gemini-embedding-001"),
            embedding_dimension=int(_env("MENO_EMBEDDING_DIMENSION", "768")),
            embedding_projection_version=_env(
                "MENO_EMBEDDING_PROJECTION_VERSION",
                "gemini-embedding-001-768-v1",
            ),
            google_credentials_file=credentials_file,
            google_api_key=_env("MENO_GOOGLE_API_KEY", "") or credentials.get("key", ""),
            google_api_base_url=_env(
                "MENO_GOOGLE_API_BASE_URL",
                "https://generativelanguage.googleapis.com/v1beta",
            ).rstrip("/"),
            google_timeout_seconds=float(
                _env("MENO_GOOGLE_TIMEOUT_SECONDS", "15")
            ),
            google_batch_size=int(_env("MENO_GOOGLE_BATCH_SIZE", "32")),
            google_max_retries=int(_env("MENO_GOOGLE_MAX_RETRIES", "4")),
            vector_upsert_batch_size=int(
                _env("MENO_VECTOR_UPSERT_BATCH_SIZE", "128")
            ),
            policy_version=_env("MENO_POLICY_VERSION", "meno-policy-1.0.0"),
            extractor_version=_env(
                "MENO_EXTRACTOR_VERSION", "meno-extractor-1.0.0"
            ),
            worker_poll_seconds=float(_env("MENO_WORKER_POLL_SECONDS", "0.2")),
            api_host=_env("MENO_API_HOST", "127.0.0.1"),
            api_port=int(_env("MENO_API_PORT", "8765")),
            api_token=_env("MENO_API_TOKEN", ""),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.embedding_provider != "google":
            raise ValueError("MENO_EMBEDDING_PROVIDER must be google")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.embedding_model):
            raise ValueError("MENO_EMBEDDING_MODEL is invalid")
        if not 128 <= self.embedding_dimension <= 3072:
            raise ValueError("embedding dimension must be between 128 and 3072")
        if not self.embedding_projection_version:
            raise ValueError("MENO_EMBEDDING_PROJECTION_VERSION is required")
        if not 1 <= self.google_batch_size <= 128:
            raise ValueError("MENO_GOOGLE_BATCH_SIZE must be between 1 and 128")
        if not 1 <= self.vector_upsert_batch_size <= 256:
            raise ValueError("MENO_VECTOR_UPSERT_BATCH_SIZE must be between 1 and 256")
        if self.google_timeout_seconds <= 0:
            raise ValueError("MENO_GOOGLE_TIMEOUT_SECONDS must be positive")
        if not 0 <= self.google_max_retries <= 8:
            raise ValueError("MENO_GOOGLE_MAX_RETRIES must be between 0 and 8")
        if self.google_credentials_file:
            path = Path(self.google_credentials_file).expanduser()
            if not path.is_file():
                raise ValueError(f"Google credentials file does not exist: {path}")
            if self.environment == "production":
                mode = stat.S_IMODE(path.stat().st_mode)
                if mode & 0o077:
                    raise ValueError("Google credentials file must not be group/world accessible")
        if self.vector_mode not in {"memory", "qdrant"}:
            raise ValueError("MENO_VECTOR_MODE must be memory or qdrant")
        if self.environment == "production":
            if not self.database_url.startswith(("postgresql://", "postgresql+psycopg://")):
                raise ValueError("production requires PostgreSQL")
            if self.vector_mode != "qdrant":
                raise ValueError("production requires Qdrant")
            if len(self.google_api_key) < 20:
                raise ValueError("production requires a Google embedding API key")
            if len(self.api_token) < 32:
                raise ValueError("production requires MENO_API_TOKEN with at least 32 characters")
