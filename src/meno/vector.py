from __future__ import annotations

import logging
import math
import random
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx
from qdrant_client import QdrantClient, models

from .config import Settings


class Embedder(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def close(self) -> None: ...


log = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class GoogleEmbedder:
    """Google Gemini embeddings with distinct document/query task types."""

    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None) -> None:
        if not settings.google_api_key:
            raise ValueError(
                "Google embedding credentials are required; set "
                "MENO_GOOGLE_CREDENTIALS_FILE"
            )
        self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension
        self.batch_size = settings.google_batch_size
        self.max_retries = settings.google_max_retries
        self._circuit_threshold = settings.google_circuit_breaker_threshold
        self._circuit_cooldown = settings.google_circuit_breaker_seconds
        self._circuit_lock = threading.Lock()
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._client = httpx.Client(
            base_url=settings.google_api_base_url,
            headers={"x-goog-api-key": settings.google_api_key},
            timeout=httpx.Timeout(settings.google_timeout_seconds),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            transport=transport,
            trust_env=False,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), self.batch_size):
            vectors.extend(
                self._embed_batch(
                    texts[offset : offset + self.batch_size],
                    task_type="RETRIEVAL_DOCUMENT",
                )
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        payload = {
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_QUERY",
            "outputDimensionality": self.dimension,
        }
        response = self._post(f"/models/{self.model}:embedContent", payload)
        vector = response.get("embedding", {}).get("values", [])
        return self._validate_and_normalize(vector)

    def close(self) -> None:
        self._client.close()

    def _embed_batch(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        if not texts:
            return []
        model_path = f"models/{self.model}"
        payload = {
            "requests": [
                {
                    "model": model_path,
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                    "outputDimensionality": self.dimension,
                }
                for text in texts
            ]
        }
        response = self._post(f"/{model_path}:batchEmbedContents", payload)
        embeddings = response.get("embeddings", [])
        if len(embeddings) != len(texts):
            raise EmbeddingError(
                "Google embedding API returned an unexpected number of vectors"
            )
        return [
            self._validate_and_normalize(item.get("values", [])) for item in embeddings
        ]

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._circuit_check()
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(path, json=payload)
            except httpx.HTTPError as exc:
                if attempt >= self.max_retries:
                    self._record_failure()
                    raise EmbeddingError("Google embedding API transport failure") from exc
                self._backoff(attempt)
                continue
            if response.is_success:
                body = response.json()
                if not isinstance(body, dict):
                    raise EmbeddingError("Google embedding API returned an invalid response")
                self._record_success()
                return body
            if response.status_code not in {408, 429, 500, 502, 503, 504}:
                raise EmbeddingError(
                    f"Google embedding API rejected the request ({response.status_code})"
                )
            if attempt >= self.max_retries:
                self._record_failure()
                raise EmbeddingError(
                    f"Google embedding API remained unavailable ({response.status_code})"
                )
            retry_after = response.headers.get("retry-after", "")
            self._backoff(attempt, retry_after)
        raise AssertionError("unreachable")

    def _circuit_check(self) -> None:
        with self._circuit_lock:
            open_until = self._circuit_open_until
        if open_until and time.monotonic() < open_until:
            raise EmbeddingError("Google embedding API circuit breaker is open")

    def _record_success(self) -> None:
        with self._circuit_lock:
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0

    def _record_failure(self) -> None:
        with self._circuit_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._circuit_threshold:
                self._circuit_open_until = time.monotonic() + self._circuit_cooldown
                log.warning(
                    "Google embedding circuit breaker open for %.1fs after %d failures",
                    self._circuit_cooldown,
                    self._consecutive_failures,
                )

    def _backoff(self, attempt: int, retry_after: str = "") -> None:
        try:
            delay = min(10.0, max(0.0, float(retry_after)))
        except ValueError:
            delay = 0.0
        if not delay:
            delay = min(10.0, 0.25 * (2**attempt)) + random.uniform(0, 0.1)
        time.sleep(delay)

    def _validate_and_normalize(self, values: Any) -> list[float]:
        if not isinstance(values, list) or len(values) != self.dimension:
            raise EmbeddingError(
                f"Google embedding dimension mismatch; expected {self.dimension}"
            )
        vector = [float(value) for value in values]
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm == 0:
            raise EmbeddingError("Google embedding API returned an invalid vector")
        return [value / norm for value in vector]


@dataclass(frozen=True)
class VectorHit:
    claim_id: str
    score: float


@dataclass(frozen=True)
class VectorDocument:
    claim_id: str
    user_id: str
    status: str
    text: str
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class VectorStore(Protocol):
    def upsert(
        self,
        claim_id: str,
        user_id: str,
        status: str,
        text: str,
        *,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
    ) -> None: ...

    def upsert_many(self, documents: list[VectorDocument]) -> None: ...

    def search(
        self, user_id: str, query: str, limit: int, as_of: datetime | None = None
    ) -> list[VectorHit]: ...

    def delete_claim(self, claim_id: str) -> None: ...

    def delete_user(self, user_id: str) -> None: ...

    def health(self) -> bool: ...

    def close(self) -> None: ...


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _epoch(value: datetime | None) -> float | None:
    return _aware(value).timestamp() if value is not None else None


def _temporally_valid(
    valid_from: datetime | None, valid_to: datetime | None, as_of: datetime | None
) -> bool:
    if as_of is None:
        return True
    moment = _aware(as_of)
    if valid_from is not None and _aware(valid_from) > moment:
        return False
    return valid_to is None or _aware(valid_to) > moment


class MemoryVectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._points: dict[
            str, tuple[str, str, list[float], datetime | None, datetime | None]
        ] = {}
        self._lock = threading.Lock()

    def upsert(
        self,
        claim_id: str,
        user_id: str,
        status: str,
        text: str,
        *,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
    ) -> None:
        self.upsert_many(
            [VectorDocument(claim_id, user_id, status, text, valid_from, valid_to)]
        )

    def upsert_many(self, documents: list[VectorDocument]) -> None:
        vectors = self.embedder.embed_documents([document.text for document in documents])
        with self._lock:
            for document, vector in zip(documents, vectors, strict=True):
                self._points[document.claim_id] = (
                    document.user_id,
                    document.status,
                    vector,
                    document.valid_from,
                    document.valid_to,
                )

    def search(
        self, user_id: str, query: str, limit: int, as_of: datetime | None = None
    ) -> list[VectorHit]:
        query_vector = self.embedder.embed_query(query)
        with self._lock:
            candidates = [
                VectorHit(claim_id, _cosine(query_vector, vector))
                for claim_id, (owner, status, vector, valid_from, valid_to) in self._points.items()
                if owner == user_id
                and status == "active"
                and _temporally_valid(valid_from, valid_to, as_of)
            ]
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:limit]

    def delete_claim(self, claim_id: str) -> None:
        with self._lock:
            self._points.pop(claim_id, None)

    def delete_user(self, user_id: str) -> None:
        with self._lock:
            doomed = [key for key, value in self._points.items() if value[0] == user_id]
            for key in doomed:
                del self._points[key]

    def health(self) -> bool:
        return True

    def close(self) -> None:
        self.embedder.close()


class QdrantVectorStore:
    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        self.embedder = embedder
        self.collection = settings.qdrant_collection
        self.projection_version = settings.embedding_projection_version
        self.upsert_batch_size = settings.vector_upsert_batch_size
        self.client = QdrantClient(url=settings.qdrant_url, timeout=10)
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=embedder.dimension, distance=models.Distance.COSINE
                ),
            )
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="user_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="status",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="valid_from_ts",
                field_schema=models.PayloadSchemaType.FLOAT,
            )
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="valid_to_ts",
                field_schema=models.PayloadSchemaType.FLOAT,
            )
        else:
            collection = self.client.get_collection(self.collection)
            vectors = collection.config.params.vectors
            actual_dimension = getattr(vectors, "size", None)
            if actual_dimension != embedder.dimension:
                raise ValueError(
                    f"Qdrant collection {self.collection} has dimension "
                    f"{actual_dimension}, expected {embedder.dimension}"
                )

    def upsert(
        self,
        claim_id: str,
        user_id: str,
        status: str,
        text: str,
        *,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
    ) -> None:
        self.upsert_many(
            [VectorDocument(claim_id, user_id, status, text, valid_from, valid_to)]
        )

    def upsert_many(self, documents: list[VectorDocument]) -> None:
        if not documents:
            return
        vectors = self.embedder.embed_documents([document.text for document in documents])
        points = [
            models.PointStruct(
                id=document.claim_id,
                vector=vector,
                payload={
                    "user_id": document.user_id,
                    "status": document.status,
                    "projection_version": self.projection_version,
                    "valid_from_ts": _epoch(document.valid_from),
                    "valid_to_ts": _epoch(document.valid_to),
                },
            )
            for document, vector in zip(documents, vectors, strict=True)
        ]
        for offset in range(0, len(points), self.upsert_batch_size):
            self.client.upsert(
                collection_name=self.collection,
                wait=True,
                points=points[offset : offset + self.upsert_batch_size],
            )

    def search(
        self, user_id: str, query: str, limit: int, as_of: datetime | None = None
    ) -> list[VectorHit]:
        must: list[models.Condition] = [
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
            models.FieldCondition(key="status", match=models.MatchValue(value="active")),
        ]
        if as_of is not None:
            as_of_ts = _epoch(as_of)
            must.append(
                models.FieldCondition(
                    key="valid_from_ts", range=models.Range(lte=as_of_ts)
                )
            )
            must.append(
                models.Filter(
                    should=[
                        models.IsNullCondition(
                            is_null=models.PayloadField(key="valid_to_ts")
                        ),
                        models.FieldCondition(
                            key="valid_to_ts", range=models.Range(gt=as_of_ts)
                        ),
                    ]
                )
            )
        response = self.client.query_points(
            collection_name=self.collection,
            query=self.embedder.embed_query(query),
            query_filter=models.Filter(must=must),
            limit=limit,
            with_payload=False,
        )
        return [VectorHit(str(point.id), float(point.score)) for point in response.points]

    def delete_claim(self, claim_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(points=[claim_id]),
            wait=True,
        )

    def delete_user(self, user_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id", match=models.MatchValue(value=user_id)
                        )
                    ]
                )
            ),
            wait=True,
        )

    def health(self) -> bool:
        try:
            self.client.get_collection(self.collection)
            return True
        except Exception:  # noqa: BLE001 - readiness must report all client failures
            return False

    def close(self) -> None:
        self.client.close()
        self.embedder.close()


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider != "google":
        raise ValueError("only Google cloud embeddings are supported")
    return GoogleEmbedder(settings)


def make_vector_store(settings: Settings, embedder: Embedder) -> VectorStore:
    if settings.vector_mode == "qdrant":
        return QdrantVectorStore(settings, embedder)
    return MemoryVectorStore(embedder)


def _cosine(left: list[float], right: list[float]) -> float:
    return max(0.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True))))
