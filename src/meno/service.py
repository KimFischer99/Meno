from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import uuid
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, selectinload

from .config import Settings
from .db import (
    AuditEvent,
    Claim,
    ClaimEdge,
    ClaimEvidence,
    Consent,
    DeletionJob,
    Event,
    Feedback,
    Outbox,
    UserRevision,
)
from .extractor import extract_claims
from .schemas import (
    ConsentRequest,
    DeletionRequest,
    Facet,
    FeedbackRequest,
    IngestRequest,
    PredictRequest,
    RetrieveRequest,
    RetrieveResponse,
)
from .vector import VectorDocument, VectorStore

PURPOSE_SCOPE = {
    "response_personalization": "personalization",
    "task_planning": "task_planning",
    "proactive_suggestion": "proactive_suggestion",
}

log = logging.getLogger(__name__)


def _id() -> str:
    return str(uuid.uuid4())


def _claim_id(derivation_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"meno:claim:{derivation_key}"))


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


class MenoService:
    def __init__(self, settings: Settings, session_factory, vector_store: VectorStore, engine=None) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.vector_store = vector_store
        self.engine = engine
        self._audit_buffer: deque[dict[str, Any]] = deque()
        self._audit_lock = threading.Lock()

    def close(self) -> None:
        self.vector_store.close()
        if self.engine is not None:
            self.engine.dispose()

    def ingest(self, request: IngestRequest, idempotency_key: str) -> dict[str, Any]:
        event_id = request.event_id or _id()
        trace_id = str(request.metadata.get("trace_id") or _id())
        with self.session_factory.begin() as session:
            existing = session.get(Event, event_id)
            if existing:
                if existing.user_id != request.user_id or existing.content_hash != _sha(request.content.text):
                    raise ValueError("event_id already exists with different content")
                revision = self._revision(session, request.user_id)
                return {
                    "trace_id": trace_id,
                    "event_id": event_id,
                    "state_revision": revision,
                    "policy_version": self.settings.policy_version,
                    "accepted": True,
                    "idempotent_replay": True,
                }
            content_hash = _sha(request.content.text)
            provided_hash = request.metadata.get("content_hash")
            if provided_hash and provided_hash != content_hash:
                raise ValueError("content_hash mismatch")
            event = Event(
                id=event_id,
                user_id=request.user_id,
                occurred_at=request.occurred_at,
                source_type=request.source.type,
                source_profile=request.source.profile,
                session_id=request.source.session_id,
                role=request.content.role,
                content=request.content.text,
                content_hash=content_hash,
                consent_scope=request.consent_scope,
                event_metadata={**request.metadata, "idempotency_key_hash": _sha(idempotency_key)},
            )
            session.add(event)
            session.add(
                Outbox(
                    id=_id(),
                    event_id=event_id,
                    processor_version=self.settings.extractor_version,
                )
            )
            revision = self._bump_revision(session, request.user_id)
            self._audit(
                session,
                event_name="meno.ingest.accepted",
                trace_id=trace_id,
                user_id=request.user_id,
                action="ingest",
                purpose=None,
                decision={"allowed": True, "role": request.content.role},
                revision=revision,
                event_ids=[event_id],
            )
        return {
            "trace_id": trace_id,
            "event_id": event_id,
            "state_revision": revision,
            "policy_version": self.settings.policy_version,
            "accepted": True,
            "idempotent_replay": False,
        }

    def ingest_many(
        self, requests: list[IngestRequest], idempotency_key: str
    ) -> dict[str, Any]:
        if not requests:
            raise ValueError("batch must contain at least one event")
        if len(requests) > 256:
            raise ValueError("batch may contain at most 256 events")
        event_ids = [request.event_id for request in requests]
        if any(event_id is None for event_id in event_ids):
            raise ValueError("every batch event requires event_id")
        resolved_ids = [str(event_id) for event_id in event_ids]
        if len(resolved_ids) != len(set(resolved_ids)):
            raise ValueError("batch event_id values must be unique")

        results: list[dict[str, Any]] = []
        with self.session_factory.begin() as session:
            existing = {
                event.id: event
                for event in session.scalars(
                    select(Event).where(Event.id.in_(resolved_ids))
                ).all()
            }
            for index, (request, event_id) in enumerate(
                zip(requests, resolved_ids, strict=True)
            ):
                trace_id = str(request.metadata.get("trace_id") or _id())
                content_hash = _sha(request.content.text)
                previous = existing.get(event_id)
                if previous is not None:
                    if previous.user_id != request.user_id or previous.content_hash != content_hash:
                        raise ValueError("event_id already exists with different content")
                    results.append(
                        {
                            "event_id": event_id,
                            "state_revision": self._revision(session, request.user_id),
                            "idempotent_replay": True,
                        }
                    )
                    continue
                provided_hash = request.metadata.get("content_hash")
                if provided_hash and provided_hash != content_hash:
                    raise ValueError("content_hash mismatch")
                session.add(
                    Event(
                        id=event_id,
                        user_id=request.user_id,
                        occurred_at=request.occurred_at,
                        source_type=request.source.type,
                        source_profile=request.source.profile,
                        session_id=request.source.session_id,
                        role=request.content.role,
                        content=request.content.text,
                        content_hash=content_hash,
                        consent_scope=request.consent_scope,
                        event_metadata={
                            **request.metadata,
                            "idempotency_key_hash": _sha(f"{idempotency_key}:{index}"),
                        },
                    )
                )
                session.add(
                    Outbox(
                        id=_id(),
                        event_id=event_id,
                        processor_version=self.settings.extractor_version,
                    )
                )
                revision = self._bump_revision(session, request.user_id)
                self._audit(
                    session,
                    event_name="meno.ingest.accepted",
                    trace_id=trace_id,
                    user_id=request.user_id,
                    action="ingest",
                    purpose=None,
                    decision={"allowed": True, "role": request.content.role, "batch": True},
                    revision=revision,
                    event_ids=[event_id],
                )
                results.append(
                    {
                        "event_id": event_id,
                        "state_revision": revision,
                        "idempotent_replay": False,
                    }
                )
        return {
            "accepted": len(results),
            "policy_version": self.settings.policy_version,
            "events": results,
        }

    def process_outbox(self, limit: int = 100) -> int:
        processed = 0
        while processed < limit:
            chunk_size = min(self.settings.outbox_commit_batch_size, limit - processed)
            count = self._process_outbox_chunk(chunk_size)
            if count <= 0:
                break
            processed += count
        return processed

    def _process_outbox_chunk(self, limit: int) -> int:
        now = _now()
        with self.session_factory() as session:
            rows = session.scalars(
                select(Outbox)
                .where(
                    Outbox.status == "pending",
                    (Outbox.next_attempt_at.is_(None)) | (Outbox.next_attempt_at <= now),
                )
                .order_by(Outbox.created_at)
                .limit(limit)
                .with_for_update(skip_locked=True)
            ).all()
            if not rows:
                return 0
            row_ids = [row.id for row in rows]
            try:
                events = {
                    event.id: event
                    for event in session.scalars(
                        select(Event).where(Event.id.in_([row.event_id for row in rows]))
                    ).all()
                }
                planned: list[tuple[Outbox, Event, str, Any]] = []
                for row in rows:
                    event = events.get(row.event_id)
                    if event is None:
                        continue
                    for candidate_index, candidate in enumerate(extract_claims(event)):
                        derivation_key = _sha(
                            f"{event.id}:{row.processor_version}:{candidate_index}"
                        )
                        planned.append((row, event, derivation_key, candidate))

                derivation_keys = [item[2] for item in planned]
                existing = set(
                    session.scalars(
                        select(Claim.derivation_key).where(
                            Claim.derivation_key.in_(derivation_keys)
                        )
                    ).all()
                )
                documents: list[VectorDocument] = []
                for _row, event, derivation_key, candidate in planned:
                    if derivation_key in existing:
                        continue
                    claim = Claim(
                        id=_claim_id(derivation_key),
                        derivation_key=derivation_key,
                        user_id=event.user_id,
                        kind=candidate.kind,
                        origin_role=event.role,
                        semantic_channel=candidate.semantic_channel,
                        value=candidate.value,
                        status="active",
                        confidence=candidate.confidence,
                        half_life_days=candidate.half_life_days,
                        sensitive=candidate.sensitive,
                        allowed_purposes=event.consent_scope,
                        source_type=event.source_type,
                        valid_from=event.occurred_at,
                        extractor_version=self.settings.extractor_version,
                    )
                    session.add(claim)
                    session.add(
                        ClaimEvidence(
                            claim_id=claim.id,
                            event_id=event.id,
                            relation="explicit_statement",
                        )
                    )
                    if not claim.sensitive:
                        documents.append(
                            VectorDocument(
                                claim_id=claim.id,
                                user_id=claim.user_id,
                                status=claim.status,
                                text=claim.value,
                                valid_from=claim.valid_from,
                                valid_to=claim.valid_to,
                            )
                        )

                session.flush()
                self.vector_store.upsert_many(documents)
                for row in rows:
                    event = events.get(row.event_id)
                    if event is not None:
                        revision = self._bump_revision(session, event.user_id)
                        self._audit(
                            session,
                            event_name="meno.outbox.processed",
                            trace_id=_id(),
                            user_id=event.user_id,
                            action="derive",
                            purpose=None,
                            decision={
                                "allowed": True,
                                "embedding_projection_version": (
                                    self.settings.embedding_projection_version
                                ),
                            },
                            revision=revision,
                            event_ids=[event.id],
                        )
                    row.status = "processed"
                    row.processed_at = _now()
                    row.error = None
                    row.next_attempt_at = None
                session.commit()
                return len(rows)
            except Exception as exc:  # noqa: BLE001 - outbox retains provider failures
                session.rollback()
                self._mark_outbox_failure(row_ids, exc)
                return 0

    def _mark_outbox_failure(self, row_ids: list[str], exc: Exception) -> None:
        with self.session_factory.begin() as failure_session:
            failed_rows = failure_session.scalars(
                select(Outbox)
                .where(Outbox.id.in_(row_ids))
                .with_for_update(skip_locked=True)
            ).all()
            now = _now()
            for failed in failed_rows:
                failed.attempts += 1
                failed.error = str(exc)[:4000]
                if failed.attempts >= 20:
                    failed.status = "failed"
                    failed.next_attempt_at = None
                else:
                    delay = min(
                        self.settings.outbox_retry_max_seconds,
                        self.settings.outbox_retry_base_seconds * (2 ** (failed.attempts - 1)),
                    )
                    failed.next_attempt_at = now + timedelta(seconds=delay)

    def requeue_failed_outbox(self, limit: int | None = None) -> int:
        with self.session_factory.begin() as session:
            statement = (
                select(Outbox).where(Outbox.status == "failed").order_by(Outbox.created_at)
            )
            if limit is not None:
                statement = statement.limit(limit)
            rows = session.scalars(statement).all()
            for row in rows:
                row.status = "pending"
                row.attempts = 0
                row.error = None
                row.next_attempt_at = None
            return len(rows)

    def drain_status(self, user_id: str) -> dict[str, Any]:
        with self.session_factory() as session:
            counts = {
                status: session.scalar(
                    select(func.count(Outbox.id))
                    .join(Event, Outbox.event_id == Event.id)
                    .where(Event.user_id == user_id, Outbox.status == status)
                )
                or 0
                for status in ("pending", "failed")
            }
        return {
            "user_id": user_id,
            "pending_outbox": counts["pending"],
            "failed_outbox": counts["failed"],
            "drained": counts["pending"] == 0,
            "policy_version": self.settings.policy_version,
        }

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        trace_id = _id()
        degraded = False
        as_of = _aware(request.context.as_of) if request.context.as_of else _now()
        try:
            hits = self.vector_store.search(
                request.user_id,
                request.context.query,
                limit=max(32, request.constraints.max_facets * 4),
                as_of=as_of,
            )
        except Exception:  # noqa: BLE001 - any vector failure activates canonical fallback
            hits = []
            degraded = True

        with self.session_factory.begin() as session:
            revision = self._revision(session, request.user_id)
            if not hits:
                claims = session.scalars(
                    select(Claim)
                    .options(selectinload(Claim.evidence))
                    .where(Claim.user_id == request.user_id, Claim.status == "active")
                    .order_by(Claim.updated_at.desc())
                    .limit(max(32, request.constraints.max_facets * 4))
                ).all()
                hits = [type("Hit", (), {"claim_id": claim.id, "score": 0.2}) for claim in claims]
                degraded = degraded or bool(claims)

            claim_ids = [hit.claim_id for hit in hits]
            claims_by_id = {
                claim.id: claim
                for claim in session.scalars(
                    select(Claim)
                    .options(selectinload(Claim.evidence))
                    .where(Claim.id.in_(claim_ids))
                ).all()
            }
            scope = PURPOSE_SCOPE[request.purpose]
            source_types = {claim.source_type for claim in claims_by_id.values()}
            revoked_sources = set(
                session.scalars(
                    select(Consent.source).where(
                        Consent.user_id == request.user_id,
                        Consent.source.in_(source_types),
                        Consent.purpose.in_([request.purpose, scope]),
                        Consent.status == "revoked",
                    )
                ).all()
            )
            facets: list[Facet] = []
            for hit in hits:
                claim = claims_by_id.get(hit.claim_id)
                if claim is None:
                    continue
                allowed, reasons, effective_confidence = self._admit(
                    claim, request, revoked_sources
                )
                if not allowed:
                    continue
                evidence_ids = [item.event_id for item in claim.evidence]
                score = self._score(hit.score, effective_confidence, claim, as_of)
                facets.append(
                    Facet(
                        claim_id=claim.id,
                        kind=claim.kind,
                        value=claim.value,
                        relevance=score,
                        confidence=effective_confidence,
                        evidence_ids=evidence_ids,
                        why_selected=reasons,
                    )
                )
            facets.sort(key=lambda item: item.relevance, reverse=True)
            facets = facets[: request.constraints.max_facets]
            rendered = self._render(request.user_id, revision, facets, request.constraints.max_rendered_tokens)
        for facet in facets:
            self._buffer_audit(
                event_name="meno.retrieve.facet_selected",
                trace_id=trace_id,
                user_id=request.user_id,
                claim_id=facet.claim_id,
                action="retrieve",
                purpose=request.purpose,
                decision={
                    "allowed": True,
                    "relevance": facet.relevance,
                    "effective_confidence": facet.confidence,
                    "policy_version": self.settings.policy_version,
                },
                revision=revision,
                event_ids=facet.evidence_ids,
            )
        return RetrieveResponse(
            trace_id=trace_id,
            user_id=request.user_id,
            state_revision=revision,
            token_revision_id=f"user-token:{request.user_id}:{revision}",
            facets=facets,
            rendered_context=rendered,
            degraded=degraded,
            policy_version=self.settings.policy_version,
        )

    def feedback(self, request: FeedbackRequest) -> dict[str, Any]:
        if request.action == "correct" and not request.correction:
            raise ValueError("correction is required for correct action")
        trace_id = _id()
        new_claim_id: str | None = None
        old_claim_id = request.claim_id
        with self.session_factory.begin() as session:
            claim = session.get(Claim, request.claim_id)
            if claim is None or claim.user_id != request.user_id:
                raise LookupError("claim not found")
            session.add(
                Feedback(
                    id=_id(),
                    user_id=request.user_id,
                    claim_id=claim.id,
                    action=request.action,
                    correction=request.correction,
                )
            )
            if request.action == "confirm":
                claim.confidence = max(claim.confidence, 0.98)
                claim.updated_at = _now()
                self.vector_store.upsert(
                    claim.id,
                    claim.user_id,
                    claim.status,
                    claim.value,
                    valid_from=claim.valid_from,
                    valid_to=claim.valid_to,
                )
            elif request.action == "reject":
                claim.status = "rejected"
                claim.valid_to = _now()
                claim.updated_at = _now()
                self.vector_store.delete_claim(claim.id)
            else:
                claim.status = "superseded"
                claim.valid_to = _now()
                claim.updated_at = _now()
                self.vector_store.delete_claim(claim.id)
                feedback_event = Event(
                    id=_id(),
                    user_id=claim.user_id,
                    occurred_at=_now(),
                    source_type="explicit_feedback",
                    source_profile=None,
                    session_id=None,
                    role="user",
                    content=request.correction or "",
                    content_hash=_sha(request.correction or ""),
                    consent_scope=claim.allowed_purposes,
                    event_metadata={"feedback_action": "correct", "claim_id": claim.id},
                )
                session.add(feedback_event)
                replacement = Claim(
                    id=_id(),
                    derivation_key=_sha(f"feedback:{feedback_event.id}"),
                    user_id=claim.user_id,
                    kind=claim.kind,
                    origin_role="user",
                    semantic_channel=claim.semantic_channel,
                    value=request.correction or "",
                    status="active",
                    confidence=0.99,
                    half_life_days=claim.half_life_days,
                    sensitive=claim.sensitive,
                    allowed_purposes=claim.allowed_purposes,
                    source_type="explicit_feedback",
                    valid_from=_now(),
                    supersedes_id=claim.id,
                    extractor_version="explicit-feedback-v1",
                )
                session.add(replacement)
                session.flush()
                new_claim_id = replacement.id
                session.add(
                    ClaimEvidence(
                        claim_id=replacement.id,
                        event_id=feedback_event.id,
                        relation="explicit_correction",
                    )
                )
                session.add(
                    ClaimEdge(
                        id=_id(),
                        user_id=claim.user_id,
                        source_claim_id=replacement.id,
                        target_claim_id=claim.id,
                        relation_type="supersedes",
                        confidence=1.0,
                    )
                )
                self.vector_store.upsert(
                    replacement.id,
                    replacement.user_id,
                    replacement.status,
                    replacement.value,
                    valid_from=replacement.valid_from,
                    valid_to=replacement.valid_to,
                )
            revision = self._bump_revision(session, request.user_id)
            self._audit(
                session,
                event_name="meno.feedback.applied",
                trace_id=trace_id,
                user_id=request.user_id,
                claim_id=new_claim_id or old_claim_id,
                action="feedback",
                purpose=None,
                decision={"allowed": True, "feedback_action": request.action},
                revision=revision,
                event_ids=[],
            )
        return {
            "trace_id": trace_id,
            "state_revision": revision,
            "claim_id": new_claim_id or old_claim_id,
            "superseded_claim_id": old_claim_id if new_claim_id else None,
            "policy_version": self.settings.policy_version,
        }

    def set_consent(self, request: ConsentRequest) -> dict[str, Any]:
        trace_id = _id()
        with self.session_factory.begin() as session:
            consent = Consent(
                id=_id(),
                user_id=request.user_id,
                source=request.source,
                purpose=request.purpose,
                allowed_operations=request.allowed_operations,
                data_categories=request.data_categories,
                sensitive_data=request.sensitive_data,
                status=request.status,
                expires_at=request.expires_at,
            )
            session.add(consent)
            revision = self._bump_revision(session, request.user_id)
            self._audit(
                session,
                event_name="meno.consent.changed",
                trace_id=trace_id,
                user_id=request.user_id,
                action="consent",
                purpose=request.purpose,
                decision={"allowed": request.status == "active", "source": request.source},
                revision=revision,
                event_ids=[],
            )
        return {
            "trace_id": trace_id,
            "consent_id": consent.id,
            "state_revision": revision,
            "policy_version": self.settings.policy_version,
        }

    def delete(self, request: DeletionRequest) -> dict[str, Any]:
        if request.scope == "source" and not request.source:
            raise ValueError("source is required")
        if request.scope == "claim" and not request.claim_id:
            raise ValueError("claim_id is required")
        job_id = _id()
        subject_hash = _sha(request.user_id)
        trace_id = _id()
        with self.session_factory.begin() as session:
            job = DeletionJob(id=job_id, subject_hash=subject_hash, scope=request.scope)
            session.add(job)
            if request.scope == "claim":
                claim = session.get(Claim, request.claim_id)
                if claim and claim.user_id == request.user_id:
                    self.vector_store.delete_claim(claim.id)
                    session.delete(claim)
            elif request.scope == "source":
                claims = session.scalars(
                    select(Claim).where(
                        Claim.user_id == request.user_id,
                        Claim.source_type == request.source,
                    )
                ).all()
                for claim in claims:
                    self.vector_store.delete_claim(claim.id)
                    session.delete(claim)
                session.execute(
                    delete(Event).where(
                        Event.user_id == request.user_id,
                        Event.source_type == request.source,
                    )
                )
            else:
                self.vector_store.delete_user(request.user_id)
                session.execute(delete(Feedback).where(Feedback.user_id == request.user_id))
                session.execute(delete(Consent).where(Consent.user_id == request.user_id))
                session.execute(delete(Claim).where(Claim.user_id == request.user_id))
                session.execute(delete(Event).where(Event.user_id == request.user_id))
            revision = self._bump_revision(session, request.user_id)
            receipt = _sha(f"{job_id}:{subject_hash}:{request.scope}:{revision}")
            job.status = "completed"
            job.completed_at = _now()
            job.receipt_hash = receipt
            self._audit(
                session,
                event_name="privacy.deletion.completed",
                trace_id=trace_id,
                user_id=request.user_id,
                action="delete",
                purpose=None,
                decision={"allowed": True, "scope": request.scope, "receipt": receipt},
                revision=revision,
                event_ids=[],
            )
        return {
            "trace_id": trace_id,
            "deletion_job_id": job_id,
            "state_revision": revision,
            "receipt_hash": receipt,
            "status": "completed",
        }

    def audit_claim(self, claim_id: str) -> dict[str, Any]:
        with self.session_factory() as session:
            claim = session.get(Claim, claim_id)
            if claim is None:
                raise LookupError("claim not found")
            events = [session.get(Event, evidence.event_id) for evidence in claim.evidence]
            return {
                "claim_id": claim.id,
                "statement": claim.value,
                "kind": claim.kind,
                "origin_role": claim.origin_role,
                "status": claim.status,
                "confidence": {
                    "calibrated": claim.confidence,
                    "effective": self._effective_confidence(claim),
                },
                "validity": {
                    "valid_from": claim.valid_from,
                    "valid_to": claim.valid_to,
                    "supersedes": claim.supersedes_id,
                },
                "evidence": [
                    {
                        "event_id": event.id,
                        "source": event.source_type,
                        "timestamp": event.occurred_at,
                        "content_hash": event.content_hash,
                    }
                    for event in events
                    if event is not None
                ],
                "transformations": [
                    {"processor": "claim-extractor", "version": claim.extractor_version}
                ],
                "policy_version": self.settings.policy_version,
            }

    def revisions(self, user_id: str) -> dict[str, Any]:
        with self.session_factory() as session:
            row = session.get(UserRevision, user_id)
            counts = {
                status: len(
                    session.scalars(
                        select(Claim).where(Claim.user_id == user_id, Claim.status == status)
                    ).all()
                )
                for status in ("active", "superseded", "rejected")
            }
            return {
                "user_id": user_id,
                "state_revision": row.revision if row else 0,
                "updated_at": row.updated_at if row else None,
                "claim_counts": counts,
                "policy_version": self.settings.policy_version,
            }

    def predict(self, request: PredictRequest) -> dict[str, Any]:
        ranked = []
        for candidate in request.candidates:
            recall = self.retrieve(
                RetrieveRequest(
                    user_id=request.user_id,
                    purpose="task_planning",
                    context={"query": f"{request.context.query} {candidate.description}"},
                    constraints={"max_facets": 4, "min_confidence": 0.55},
                )
            )
            score = sum(facet.relevance * facet.confidence for facet in recall.facets)
            ranked.append(
                {
                    "candidate_id": candidate.id,
                    "score": score,
                    "supporting_claim_ids": [facet.claim_id for facet in recall.facets],
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return {
            "trace_id": _id(),
            "user_id": request.user_id,
            "ranked_candidates": ranked,
            "advisory_only": True,
            "policy_version": self.settings.policy_version,
        }

    def health(self) -> dict[str, Any]:
        with self.session_factory() as session:
            session.execute(select(1))
        vector_ok = self.vector_store.health()
        return {
            "status": "ok" if vector_ok else "degraded",
            "database": "ok",
            "vector": "ok" if vector_ok else "unavailable",
            "environment": self.settings.environment,
            "policy_version": self.settings.policy_version,
            "embedding_provider": self.settings.embedding_provider,
            "embedding_model": self.settings.embedding_model,
            "embedding_projection_version": self.settings.embedding_projection_version,
        }

    def rebuild_projection(self, batch_size: int | None = None) -> int:
        resolved_batch_size = batch_size or self.settings.vector_upsert_batch_size
        if not 1 <= resolved_batch_size <= 256:
            raise ValueError("projection rebuild batch size must be between 1 and 256")
        rebuilt = 0
        after_id = ""
        while True:
            with self.session_factory() as session:
                claims = session.scalars(
                    select(Claim)
                    .where(
                        Claim.id > after_id,
                        Claim.status == "active",
                        Claim.sensitive.is_(False),
                    )
                    .order_by(Claim.id)
                    .limit(resolved_batch_size)
                ).all()
            if not claims:
                return rebuilt
            self.vector_store.upsert_many(
                [
                    VectorDocument(
                        claim_id=claim.id,
                        user_id=claim.user_id,
                        status=claim.status,
                        text=claim.value,
                        valid_from=claim.valid_from,
                        valid_to=claim.valid_to,
                    )
                    for claim in claims
                ]
            )
            rebuilt += len(claims)
            after_id = claims[-1].id

    def _admit(
        self,
        claim: Claim,
        request: RetrieveRequest,
        revoked_sources: set[str],
    ) -> tuple[bool, list[str], float]:
        if claim.user_id != request.user_id or claim.status != "active":
            return False, [], 0.0
        as_of = _aware(request.context.as_of) if request.context.as_of else _now()
        if _aware(claim.valid_from) > as_of:
            return False, [], 0.0
        if claim.valid_to and _aware(claim.valid_to) <= as_of:
            return False, [], 0.0
        scope = PURPOSE_SCOPE[request.purpose]
        if scope not in claim.allowed_purposes and request.purpose not in claim.allowed_purposes:
            return False, [], 0.0
        if claim.sensitive and not request.constraints.allow_sensitive:
            return False, [], 0.0
        if claim.origin_role == "assistant" and request.context.task_type != "conversation_recall":
            return False, [], 0.0
        effective = self._effective_confidence(claim, as_of)
        if effective < request.constraints.min_confidence:
            return False, [], effective
        if claim.source_type in revoked_sources:
            return False, [], effective
        if not claim.evidence and claim.source_type != "explicit_feedback":
            return False, [], effective
        return True, ["semantic match", "active claim", "evidence present", "purpose allowed"], effective

    def _effective_confidence(self, claim: Claim, as_of: datetime | None = None) -> float:
        if not claim.half_life_days:
            return claim.confidence
        effective_at = as_of or _now()
        age_days = max(0.0, (effective_at - _aware(claim.valid_from)).total_seconds() / 86400)
        return max(0.0, min(1.0, claim.confidence * math.exp(-math.log(2) * age_days / claim.half_life_days)))

    def _score(
        self, semantic: float, confidence: float, claim: Claim, as_of: datetime
    ) -> float:
        age_days = max(0.0, (as_of - _aware(claim.valid_from)).total_seconds() / 86400)
        recency = math.exp(-age_days / 180)
        evidence = min(1.0, 0.5 + 0.1 * len(claim.evidence))
        risk = 0.2 if claim.sensitive else 0.0
        score = 0.55 * semantic + 0.2 * confidence + 0.1 * recency + 0.15 * evidence - risk
        return max(0.0, min(1.0, score))

    def _render(self, user_id: str, revision: int, facets: list[Facet], max_tokens: int) -> str:
        lines = [f'<user_context user_id="{user_id}" revision="{revision}">']
        for facet in facets:
            safe_value = str(facet.value).replace("<", "&lt;").replace(">", "&gt;")
            lines.append(
                f'- [{facet.kind}] {safe_value} '
                f'(confidence={facet.confidence:.3f}, evidence={",".join(facet.evidence_ids)})'
            )
        lines.append("</user_context>")
        rendered = "\n".join(lines)
        return rendered[: max_tokens * 4]

    def _revision(self, session: Session, user_id: str) -> int:
        row = session.get(UserRevision, user_id)
        return row.revision if row else 0

    def _bump_revision(self, session: Session, user_id: str) -> int:
        dialect = session.get_bind().dialect.name
        values = {"user_id": user_id, "revision": 1, "updated_at": _now()}
        if dialect in {"postgresql", "sqlite"}:
            insert = postgresql_insert if dialect == "postgresql" else sqlite_insert
            statement = (
                insert(UserRevision)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=[UserRevision.user_id],
                    set_={
                        "revision": UserRevision.revision + 1,
                        "updated_at": values["updated_at"],
                    },
                )
                .returning(UserRevision.revision)
            )
            return int(session.execute(statement).scalar_one())

        row = session.scalar(
            select(UserRevision).where(UserRevision.user_id == user_id).with_for_update()
        )
        if row is None:
            row = UserRevision(**values)
            session.add(row)
        else:
            row.revision += 1
            row.updated_at = values["updated_at"]
        session.flush()
        return row.revision

    def _buffer_audit(self, **entry: Any) -> None:
        with self._audit_lock:
            if len(self._audit_buffer) >= self.settings.audit_buffer_max:
                self._audit_buffer.popleft()
                log.warning("audit buffer full; dropping oldest buffered audit event")
            self._audit_buffer.append(entry)

    def flush_audit_buffer(self) -> int:
        with self._audit_lock:
            batch = list(self._audit_buffer)
            self._audit_buffer.clear()
        if not batch:
            return 0
        try:
            with self.session_factory.begin() as session:
                for entry in batch:
                    self._audit(session, **entry)
        except Exception:  # buffered audits must survive transient DB failures
            log.exception("audit flush failed; rebuffering %d events", len(batch))
            with self._audit_lock:
                self._audit_buffer.extendleft(reversed(batch))
            return 0
        return len(batch)

    def _audit(
        self,
        session: Session,
        *,
        event_name: str,
        trace_id: str,
        user_id: str,
        action: str,
        purpose: str | None,
        decision: dict[str, Any],
        revision: int,
        event_ids: list[str],
        claim_id: str | None = None,
    ) -> None:
        if session.get_bind().dialect.name == "postgresql":
            session.execute(
                text("SELECT pg_advisory_xact_lock(:lock_id)"),
                {"lock_id": 1_296_386_663},
            )
        previous = session.scalar(select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(1))
        prev_hash = previous.current_hash if previous else None
        payload = {
            "event_name": event_name,
            "trace_id": trace_id,
            "user_hash": _sha(user_id),
            "claim_id": claim_id,
            "action": action,
            "purpose": purpose,
            "decision": decision,
            "state_revision": revision,
            "source_event_ids": event_ids,
            "prev_hash": prev_hash,
        }
        current_hash = _sha(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        session.add(
            AuditEvent(
                id=_id(),
                event_name=event_name,
                trace_id=trace_id,
                user_hash=payload["user_hash"],
                claim_id=claim_id,
                action=action,
                purpose=purpose,
                decision=decision,
                state_revision=revision,
                source_event_ids=event_ids,
                prev_hash=prev_hash,
                current_hash=current_hash,
            )
        )
        session.flush()
