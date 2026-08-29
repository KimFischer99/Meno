from __future__ import annotations

import asyncio
import contextlib
import hmac
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from .config import Settings
from .db import make_session_factory
from .schemas import (
    ConsentRequest,
    DeletionRequest,
    FeedbackRequest,
    IngestBatchRequest,
    IngestRequest,
    PredictRequest,
    RetrieveRequest,
    RetrieveResponse,
)
from .semantic_policy import FrozenSemanticRoutingPolicy
from .service import MenoService
from .vector import Embedder, VectorStore, make_embedder, make_vector_store


def build_service(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
) -> MenoService:
    session_factory, engine = make_session_factory(settings.database_url)
    resolved_embedder = embedder
    if vector_store is None:
        resolved_embedder = resolved_embedder or make_embedder(settings)
        if getattr(resolved_embedder, "dimension", None) != settings.embedding_dimension:
            raise ValueError("embedding provider dimension does not match service settings")
        vector_store = make_vector_store(settings, resolved_embedder)
    elif resolved_embedder is None:
        resolved_embedder = getattr(vector_store, "embedder", None)
    if resolved_embedder is not None and (
        getattr(resolved_embedder, "dimension", None) != settings.embedding_dimension
    ):
        raise ValueError("embedding provider dimension does not match service settings")
    semantic_policy = None
    if settings.semantic_routing_enabled:
        if resolved_embedder is None:
            raise ValueError("semantic routing requires an embedding provider")
        semantic_policy = FrozenSemanticRoutingPolicy.from_settings(
            settings, resolved_embedder
        )
    return MenoService(
        settings,
        session_factory,
        vector_store,
        engine=engine,
        semantic_policy=semantic_policy,
    )


def create_app(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    vector_store: VectorStore | None = None,
) -> FastAPI:
    resolved = settings or Settings.from_env()
    service = build_service(resolved, embedder=embedder, vector_store=vector_store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.meno = service
        stop = asyncio.Event()

        async def worker() -> None:
            while not stop.is_set():
                await asyncio.to_thread(service.process_outbox)
                await asyncio.to_thread(service.flush_audit_buffer)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=resolved.worker_poll_seconds)
                except TimeoutError:
                    continue

        task = asyncio.create_task(worker(), name="meno-outbox-worker")
        yield
        stop.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        service.flush_audit_buffer()
        service.close()

    app = FastAPI(
        title="Meno Personal Agent Memory",
        version="2.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if request.url.path.startswith("/v1/") and resolved.api_token:
            expected = f"Bearer {resolved.api_token}"
            supplied = request.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied, expected):
                return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        return await call_next(request)

    def require_key(value: str | None) -> str:
        if not value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Idempotency-Key header is required",
            )
        return value

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready(response: Response):
        result = service.health()
        if result["status"] != "ok":
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return result

    @app.post("/v1/ingest", status_code=status.HTTP_202_ACCEPTED)
    def ingest(
        request: IngestRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        try:
            return service.ingest(request, require_key(idempotency_key))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/ingest/batch", status_code=status.HTTP_202_ACCEPTED)
    def ingest_batch(
        request: IngestBatchRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        try:
            return service.ingest_many(request.events, require_key(idempotency_key))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/retrieve", response_model=RetrieveResponse)
    def retrieve(request: RetrieveRequest) -> RetrieveResponse:
        return service.retrieve(request)

    @app.post("/v1/predict")
    def predict(request: PredictRequest):
        return service.predict(request)

    @app.post("/v1/feedback")
    def feedback(
        request: FeedbackRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        require_key(idempotency_key)
        try:
            return service.feedback(request)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/consents")
    def consent(
        request: ConsentRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        require_key(idempotency_key)
        return service.set_consent(request)

    @app.post("/v1/deletions")
    def deletion(
        request: DeletionRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        require_key(idempotency_key)
        try:
            return service.delete(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/audit/{claim_id}")
    def audit(claim_id: str):
        try:
            return service.audit_claim(claim_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/revisions/{user_id}")
    def revisions(user_id: str):
        return service.revisions(user_id)

    @app.get("/v1/users/{user_id}/token")
    def user_token(user_id: str, revision: int | None = None):
        try:
            return service.user_token(user_id, revision)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/users/{user_id}/token-diff")
    def user_token_diff(user_id: str, from_revision: int, to_revision: int):
        try:
            return service.diff_user_tokens(user_id, from_revision, to_revision)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/users/{user_id}/drain")
    def drain(user_id: str):
        return service.drain_status(user_id)

    app.state.meno = service
    return app
