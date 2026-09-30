from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from .config import Settings
from .context_activation import ContextActivationPolicy
from .db import (
    AuditEvent,
    Claim,
    ClaimEdge,
    ClaimEvidence,
    Consent,
    DeletionJob,
    Event,
    Feedback,
    IdempotencyRecord,
    Outbox,
    PreferenceDistribution,
    ProjectionOutbox,
    UserRevision,
    UserTokenSnapshot,
    UserTokenSnapshotDelta,
)
from .extractor import ClaimCandidate, extract_claims, preference_slot
from .reflection import PatternCandidate, derive_patterns, pattern_derivation_basis
from .schemas import (
    ClarificationOpportunity,
    ConsentRequest,
    DeletionRequest,
    Facet,
    FeedbackRequest,
    IngestRequest,
    PredictRequest,
    RetrieveRequest,
    RetrieveResponse,
)
from .semantic_policy import FrozenSemanticRoutingPolicy
from .semantic_router import (
    CandidateEnvelope,
    DimensionScore,
    ReferenceEnvelope,
    RouterDecision,
)
from .snapshot_delta import apply_delta, canonical_json, content_hash, make_delta
from .vector import VectorDocument, VectorStore

PURPOSE_SCOPE = {
    "response_personalization": "personalization",
    "task_planning": "task_planning",
    "proactive_suggestion": "proactive_suggestion",
}
_MAX_USER_TOKEN_DELTA_SPAN = 1000

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


_NORMALIZE_EDGE_PUNCT = '。.!！?？,，;；:：\'"\'""" '


def _audit_spill_path(settings: Settings) -> Path:
    """Overflow file for audit events that no longer fit in the memory buffer.

    Audit events are never dropped: overflow is appended here and merged back
    into the database on the next successful flush.
    """
    if settings.audit_spill_path:
        return Path(settings.audit_spill_path)
    url = settings.database_url
    if url.startswith("sqlite"):
        raw = url.split("///", 1)[-1] or "meno.db"
        return Path(raw).parent / "meno-audit-spill.jsonl"
    return Path("/var/lib/meno/meno-audit-spill.jsonl")


def _normalize_value(value: str) -> str:
    """Deterministic value normalization for semantic keys (stdlib only)."""
    folded = re.sub(r"\s+", " ", value.casefold()).strip()
    return folded.strip(_NORMALIZE_EDGE_PUNCT)


# Stable state layers per Meno_SPEC.md: canonical representation must distinguish
# abstraction levels instead of treating every memory as the same chunk. These kinds
# carry durable user state; "episodic" and "state" are the transient conversational
# layers that must not crowd them out of the injection budget.
STATE_LAYER_KINDS = frozenset({"preference", "trait", "pattern", "constraint", "fact"})

# Canonical state-layer values are short normalized strings; episodic values keep the
# raw utterance. Discount similarity above this length so length alone cannot win.
_EPISODIC_NEUTRAL_LENGTH = 120
_EPISODIC_MIN_DISCOUNT = 0.6


def _episodic_length_discount(value: str) -> float:
    """Scale an episodic similarity score down toward canonical-length parity."""
    length = len(value)
    if length <= _EPISODIC_NEUTRAL_LENGTH:
        return 1.0
    discount = (_EPISODIC_NEUTRAL_LENGTH / length) ** 0.5
    return max(_EPISODIC_MIN_DISCOUNT, discount)


# Short tokens and function words carry no evidence signal, so they would make
# every candidate look equally redundant during set-level selection.
_CONTENT_STOPWORDS = frozenset(
    (
        "about", "above", "after", "again", "against", "because", "been", "before",
        "being", "below", "between", "both", "could", "does", "doing", "during",
        "each", "from", "further", "have", "having", "here", "into", "itself",
        "more", "most", "only", "other", "over", "same", "should", "some", "such",
        "than", "that", "their", "them", "then", "there", "these", "they", "this",
        "those", "through", "under", "until", "very", "were", "what", "when",
        "where", "which", "while", "will", "with", "would", "your", "user",
        "assistant",
    )
)


def _content_tokens(text: str) -> set[str]:
    """Content-bearing tokens used for query coverage and redundancy comparison."""
    return {
        token
        for token in re.findall(r"[a-z0-9]{4,}", text.lower())
        if token not in _CONTENT_STOPWORDS
    }


# Puts the query-coverage term on the same scale as a relevance score, which sits
# in [0, 1] and typically lands near 0.5-0.9 for admitted facets.
_COVERAGE_WEIGHT = 1.0


def _semantic_key(
    user_id: str, kind: str, semantic_channel: str, value: str, slot: str | None = None
) -> str:
    """Key of the "(user, semantic)" invariant: at most one active claim each.

    Slot-keyed candidates (extractor v2 preferences) collapse onto one key per
    preference dimension; everything else falls back to the normalized value.
    """
    basis = slot if slot else _normalize_value(value)
    return _sha(f"{user_id}:{kind}:{semantic_channel}:{basis}")


# Evidence reinforcement (stage-2 item 7): each extra evidence event both lifts
# the confidence base and stretches the effective half-life, so re-confirmed
# preferences decay slower instead of purely aging out.
EVIDENCE_CONFIDENCE_STEP = 0.04
EVIDENCE_HALF_LIFE_FACTOR = 0.5

# Claim coordination protocol columns (design doc section 3.1). DDL is valid
# for both SQLite (constant default, no nullability change) and PostgreSQL.
_MIGRATION_COLUMNS = (
    ("semantic_key", "VARCHAR(71) NOT NULL DEFAULT ''"),
    ("superseded_by_id", "VARCHAR(64) NULL REFERENCES meno_claims(id)"),
    ("superseded_reason", "VARCHAR(32) NULL"),
    ("routing_slot", "VARCHAR(64) NULL"),
    ("routing_basis", "VARCHAR(32) NULL"),
    ("router_version", "VARCHAR(71) NULL"),
    ("stance", "VARCHAR(16) NULL"),
)


@dataclass(frozen=True)
class _RoutingResolution:
    semantic_key: str
    routing_slot: str | None
    routing_basis: str
    router_version: str | None
    decision: RouterDecision | None = None
    skip: bool = False


class MenoService:
    def __init__(
        self,
        settings: Settings,
        session_factory,
        vector_store: VectorStore,
        engine=None,
        semantic_policy: FrozenSemanticRoutingPolicy | None = None,
    ) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.vector_store = vector_store
        self.engine = engine
        self.semantic_policy = semantic_policy
        self.context_activation_policy = ContextActivationPolicy()
        self._audit_buffer: deque[dict[str, Any]] = deque()
        self._audit_lock = threading.Lock()
        self._audit_spill_path = _audit_spill_path(settings)
        # The audit hash chain reads the newest row's current_hash as prev_hash.
        # PostgreSQL serializes that read-modify-write across processes with an
        # advisory lock held to transaction end, but SQLite has no equivalent:
        # a second transaction reading before the first commits would link to
        # the same head and fork the chain. Every transaction that appends audit
        # rows holds this lock until it commits, so a head read always observes
        # committed rows. Re-entrant because _audit also takes it (and because
        # the outbox chunk recurses on poison rows).
        self._audit_chain_lock = threading.RLock()
        # SQLite ignores FOR UPDATE SKIP LOCKED. Serialize workers within one
        # service process so the dev/test backend cannot materialize an outbox
        # row twice; PostgreSQL still provides cross-process row locking.
        self._outbox_lock = threading.Lock()
        self._projection_lock = threading.Lock()

    def close(self) -> None:
        self.vector_store.close()
        if self.engine is not None:
            self.engine.dispose()

    def ingest(self, request: IngestRequest, idempotency_key: str) -> dict[str, Any]:
        event_id = request.event_id or _id()
        trace_id = str(request.metadata.get("trace_id") or _id())
        with self._audit_chain_lock, self.session_factory.begin() as session:
            existing = session.get(Event, event_id)
            if existing:
                if existing.user_id != request.user_id or existing.content_hash != _sha(
                    request.content.text
                ):
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
            watermark = self._deletion_watermark(session, request.user_id)
            if watermark is not None and _aware(request.occurred_at) <= watermark:
                # The user asked to be forgotten after this event happened; a queued
                # client replay must not resurrect it.
                return {
                    "trace_id": trace_id,
                    "event_id": event_id,
                    "state_revision": self._revision(session, request.user_id),
                    "policy_version": self.settings.policy_version,
                    "accepted": False,
                    "skipped": True,
                    "reason": "before_deletion_watermark",
                }
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
            try:
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
            except IntegrityError as exc:
                # A concurrent request inserting the same event_id may already have
                # committed between our idempotency check above and this flush.
                # Re-open a fresh transaction and reconcile: if the row now exists
                # with identical content, treat this as an idempotent replay instead
                # of surfacing a 500 that silently drops the event.
                log.warning(
                    "ingest idempotency race caught for event_id=%s; reconciling",
                    event_id,
                )
                session.rollback()
                with self.session_factory() as check:
                    winner = check.get(Event, event_id)
                    if winner is not None:
                        if (
                            winner.user_id != request.user_id
                            or winner.content_hash != _sha(request.content.text)
                        ):
                            raise ValueError(
                                "event_id already exists with different content"
                            ) from exc
                        replay_revision = self._revision(check, request.user_id)
                        return {
                            "trace_id": trace_id,
                            "event_id": event_id,
                            "state_revision": replay_revision,
                            "policy_version": self.settings.policy_version,
                            "accepted": True,
                            "idempotent_replay": True,
                        }
                raise
        return {
            "trace_id": trace_id,
            "event_id": event_id,
            "state_revision": revision,
            "policy_version": self.settings.policy_version,
            "accepted": True,
            "idempotent_replay": False,
        }

    def ingest_many(self, requests: list[IngestRequest], idempotency_key: str) -> dict[str, Any]:
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
        with self._audit_chain_lock, self.session_factory.begin() as session:
            existing = {
                event.id: event
                for event in session.scalars(select(Event).where(Event.id.in_(resolved_ids))).all()
            }
            for index, (request, event_id) in enumerate(zip(requests, resolved_ids, strict=True)):
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
        with self._outbox_lock:
            processed = 0
            while processed < limit:
                chunk_size = min(self.settings.outbox_commit_batch_size, limit - processed)
                count = self._process_outbox_chunk(chunk_size)
                if count <= 0:
                    break
                processed += count
        self.process_projection_outbox(limit=max(limit * 4, 100))
        return processed

    def _process_outbox_chunk(self, limit: int, only_row_ids: list[str] | None = None) -> int:
        now = _now()
        with self._audit_chain_lock, self.session_factory() as session:
            # D2: only rows stamped with the running extractor version are
            # picked up, so processing code, row version and claim stamp agree.
            statement = select(Outbox).where(
                Outbox.status == "pending",
                Outbox.processor_version == self.settings.extractor_version,
                (Outbox.next_attempt_at.is_(None)) | (Outbox.next_attempt_at <= now),
            )
            if only_row_ids is not None:
                statement = statement.where(Outbox.id.in_(only_row_ids))
            rows = session.scalars(
                statement.order_by(Outbox.created_at).limit(limit).with_for_update(skip_locked=True)
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
                planned: list[
                    tuple[Outbox, Event, str, ClaimCandidate, str, CandidateEnvelope]
                ] = []
                for row in rows:
                    event = events.get(row.event_id)
                    if event is None:
                        continue
                    for candidate_index, candidate in enumerate(extract_claims(event)):
                        derivation_key = _sha(
                            f"{event.id}:{row.processor_version}:{candidate_index}"
                        )
                        semantic_key = _semantic_key(
                            event.user_id,
                            candidate.kind,
                            candidate.semantic_channel,
                            candidate.value,
                            candidate.slot,
                        )
                        planned.append(
                            (
                                row,
                                event,
                                derivation_key,
                                candidate,
                                semantic_key,
                                CandidateEnvelope(
                                    user_id=event.user_id,
                                    kind=candidate.kind,
                                    semantic_channel=candidate.semantic_channel,
                                    value=candidate.value,
                                    sensitive=bool(candidate.sensitive),
                                    injection_detected=False,
                                    deterministic_slot=candidate.slot,
                                ),
                            )
                        )

                derivation_keys = [item[2] for item in planned]
                existing = set(
                    session.scalars(
                        select(Claim.derivation_key).where(
                            Claim.derivation_key.in_(derivation_keys)
                        )
                    ).all()
                )
                active_by_key: dict[tuple[str, str], Claim] = {}
                planned_users = {item[1].user_id for item in planned}
                if planned_users:
                    active_by_key = {
                        (claim.user_id, claim.semantic_key): claim
                        for claim in session.scalars(
                            select(Claim)
                            .options(selectinload(Claim.evidence))
                            .where(
                                Claim.status == "active",
                                Claim.user_id.in_(planned_users),
                            )
                            .with_for_update()
                        ).all()
                    }

                score_by_index: dict[int, DimensionScore | None] = {}
                if self.settings.semantic_routing_enabled and self.semantic_policy is not None:
                    routed_indices = [
                        index
                        for index, item in enumerate(planned)
                        if item[2] not in existing and self._should_route(item[3])
                    ]
                    if routed_indices:
                        try:
                            routed_scores = self.semantic_policy.score_many(
                                [planned[index][5] for index in routed_indices]
                            )
                            if len(routed_scores) != len(routed_indices):
                                raise ValueError(
                                    "semantic policy returned an unexpected score count"
                                )
                        except Exception as exc:  # noqa: BLE001 - privacy fail-closed
                            log.warning(
                                "semantic routing scoring failed closed (%s)",
                                type(exc).__name__,
                            )
                            routed_scores = [None] * len(routed_indices)
                        score_by_index = dict(zip(routed_indices, routed_scores, strict=True))

                produced_keys: dict[str, set[str]] = {}
                protected_skips: dict[str, list[Claim]] = {}
                routing_audits: dict[str, list[tuple[str | None, dict[str, Any]]]] = {}
                for index, (
                    row,
                    event,
                    derivation_key,
                    candidate,
                    semantic_key,
                    envelope,
                ) in enumerate(planned):
                    if derivation_key in existing:
                        continue
                    resolution = self._resolve_routing(
                        event,
                        candidate,
                        semantic_key,
                        envelope,
                        score_by_index.get(index),
                        active_by_key,
                    )
                    if resolution.decision is not None:
                        routing_audits.setdefault(event.id, []).append(
                            (
                                resolution.decision.matched_claim_id,
                                self._routing_audit_record(
                                    event,
                                    sum(prior[1].id == event.id for prior in planned[:index]),
                                    candidate,
                                    semantic_key,
                                    resolution,
                                ),
                            )
                        )
                    if resolution.skip:
                        protected = produced_keys.setdefault(event.id, set())
                        protected.update(
                            claim.semantic_key
                            for claim in active_by_key.values()
                            if claim.user_id == event.user_id
                            and claim.kind == candidate.kind
                            and claim.semantic_channel == candidate.semantic_channel
                        )
                        continue
                    produced_keys.setdefault(event.id, set()).add(resolution.semantic_key)
                    coordinated = self._coordinate_candidate(
                        session,
                        row,
                        event,
                        derivation_key,
                        candidate,
                        resolution.semantic_key,
                        resolution.routing_slot,
                        resolution.routing_basis,
                        resolution.router_version,
                        active_by_key,
                        protected_skips,
                    )
                    if resolution.decision is not None and coordinated is not None:
                        routing_audits[event.id][-1] = (
                            coordinated.id,
                            routing_audits[event.id][-1][1],
                        )

                session.flush()
                # Event-level cleanup: withdraw stale-version claims derived
                # from the reprocessed events once every evidence event has
                # been processed under the current version.
                chunk_event_ids = {row.event_id for row in rows}
                for row in rows:
                    event = events.get(row.event_id)
                    if event is None:
                        continue
                    self._withdraw_stale_claims(
                        session,
                        event,
                        row.processor_version,
                        produced_keys.get(event.id, set()),
                        chunk_event_ids,
                        active_by_key,
                    )

                session.flush()
                # Reflection runs after stale withdrawal so it summarizes the
                # final active set, and before revision materialization so any
                # derived pattern lands in the same snapshot as its sources.
                reflection_audits: dict[str, list[dict[str, Any]]] = {}
                if self.settings.reflection_enabled:
                    reflection_audits = self._derive_reflections(
                        session,
                        {event.user_id for event in events.values()},
                        self.settings.extractor_version,
                    )
                    session.flush()
                materialize_revisions: dict[str, int] = {}
                for row in rows:
                    event = events.get(row.event_id)
                    if event is not None:
                        revision = self._bump_revision(session, event.user_id)
                        materialize_revisions[event.user_id] = revision
                        self._audit(
                            session,
                            event_name="meno.outbox.processed",
                            trace_id=_id(),
                            user_id=event.user_id,
                            action="derive",
                            purpose=None,
                            decision={
                                "allowed": True,
                                "extractor_version": row.processor_version,
                                "embedding_projection_version": (
                                    self.settings.embedding_projection_version
                                ),
                            },
                            revision=revision,
                            event_ids=[event.id],
                        )
                        for claim_id, routing_decision in routing_audits.get(event.id, []):
                            self._audit(
                                session,
                                event_name="meno.claim.semantic_routing",
                                trace_id=_id(),
                                user_id=event.user_id,
                                claim_id=claim_id,
                                action="semantic_routing_applied",
                                purpose=None,
                                decision=routing_decision,
                                revision=revision,
                                event_ids=[event.id],
                            )
                        for skipped in protected_skips.get(event.id, []):
                            self._audit(
                                session,
                                event_name="meno.claim.coordination",
                                trace_id=_id(),
                                user_id=event.user_id,
                                claim_id=skipped.id,
                                action="derive",
                                purpose=None,
                                decision={
                                    "allowed": False,
                                    "reason": "explicit_feedback_protected",
                                    "semantic_key": skipped.semantic_key,
                                    "extractor_version": row.processor_version,
                                },
                                revision=revision,
                                event_ids=[event.id],
                            )
                    row.status = "processed"
                    row.processed_at = _now()
                    row.error = None
                    row.next_attempt_at = None
                # Reflection is per-user rather than per-event, so its audit rows
                # are emitted once the user's revision for this chunk is known.
                for user_id, records in reflection_audits.items():
                    revision = materialize_revisions.get(user_id)
                    for record in records:
                        self._audit(
                            session,
                            event_name="meno.claim.reflection",
                            trace_id=_id(),
                            user_id=user_id,
                            claim_id=record["claim_id"],
                            action="derive",
                            purpose=None,
                            decision=record["decision"],
                            revision=revision,
                            event_ids=record["source_event_ids"],
                        )
                for user_id, revision in materialize_revisions.items():
                    self._materialize_user_state(session, user_id, revision)
                session.commit()
                return len(rows)
            except Exception as exc:  # noqa: BLE001 - outbox retains provider failures
                session.rollback()
                if len(row_ids) > 1:
                    # A provider batch can fail because of one poison document.
                    # Retry the selected rows separately so healthy events commit
                    # and only the actual poison rows consume their retry budget.
                    return sum(self._process_outbox_chunk(1, [row_id]) for row_id in row_ids)
                self._mark_outbox_failure(row_ids, exc)
                return 0

    def _should_route(self, candidate: ClaimCandidate) -> bool:
        return bool(
            self.settings.semantic_routing_enabled
            and self.semantic_policy is not None
            and candidate.kind == "preference"
            and candidate.semantic_channel == "preference.explicit"
            and candidate.slot is None
            and not candidate.sensitive
        )

    def _resolve_routing(
        self,
        event: Event,
        candidate: ClaimCandidate,
        canonical_key: str,
        envelope: CandidateEnvelope,
        score: DimensionScore | None,
        active_by_key: dict[tuple[str, str], Claim],
    ) -> _RoutingResolution:
        if candidate.slot is not None:
            return _RoutingResolution(
                canonical_key,
                candidate.slot,
                "deterministic_slot",
                None,
            )
        if not self._should_route(candidate):
            return _RoutingResolution(canonical_key, None, "none", None)
        assert self.semantic_policy is not None
        references = [
            self._reference_from_claim(claim)
            for claim in active_by_key.values()
            if claim.user_id == event.user_id
            and claim.kind == candidate.kind
            and claim.semantic_channel == candidate.semantic_channel
        ]
        try:
            decision = self.semantic_policy.decide(envelope, references, score)
        except Exception as exc:  # noqa: BLE001 - privacy fail-closed
            log.warning("semantic routing decision failed closed (%s)", type(exc).__name__)
            router = self.semantic_policy.router
            decision = RouterDecision(
                action="new_key",
                proposed_semantic_key=None,
                matched_claim_id=None,
                dimension=None,
                score=None,
                margin=None,
                strategy_version=router.strategy_version,
                router_version=router.router_version,
                reason="provider_unavailable",
                protected_reference=False,
            )
        if decision.action == "reuse_key":
            matched = self._claim_by_id(active_by_key, decision.matched_claim_id)
            routing_slot = matched.routing_slot if matched is not None else decision.dimension
            return _RoutingResolution(
                decision.proposed_semantic_key or canonical_key,
                routing_slot or decision.dimension,
                "semantic_router",
                decision.router_version,
                decision,
            )
        if decision.action == "reject":
            return _RoutingResolution(
                canonical_key,
                None,
                "none",
                decision.router_version,
                decision,
                skip=True,
            )
        accepted_dimension = (
            decision.dimension if decision.reason == "no_matching_reference" else None
        )
        resolved_key = (
            _semantic_key(
                event.user_id,
                candidate.kind,
                candidate.semantic_channel,
                candidate.value,
                accepted_dimension,
            )
            if accepted_dimension is not None
            else canonical_key
        )
        return _RoutingResolution(
            resolved_key,
            accepted_dimension,
            "semantic_router" if accepted_dimension is not None else "none",
            decision.router_version,
            decision,
        )

    @staticmethod
    def _claim_by_id(
        active_by_key: dict[tuple[str, str], Claim], claim_id: str | None
    ) -> Claim | None:
        if claim_id is None:
            return None
        return next(
            (claim for claim in active_by_key.values() if claim.id == claim_id),
            None,
        )

    @staticmethod
    def _reference_from_claim(claim: Claim) -> ReferenceEnvelope:
        return ReferenceEnvelope(
            claim_id=claim.id,
            user_id=claim.user_id,
            semantic_key=claim.semantic_key,
            kind=claim.kind,
            semantic_channel=claim.semantic_channel,
            value=claim.value,
            sensitive=bool(claim.sensitive),
            status=claim.status,
            source_type=claim.source_type,
            deterministic_slot=claim.routing_slot,
        )

    def _routing_audit_record(
        self,
        event: Event,
        candidate_index: int,
        candidate: ClaimCandidate,
        canonical_key: str,
        resolution: _RoutingResolution,
    ) -> dict[str, Any]:
        assert resolution.decision is not None
        decision = resolution.decision
        return {
            "allowed": decision.action != "reject",
            "event_id": event.id,
            "candidate_index": candidate_index,
            "kind": candidate.kind,
            "semantic_channel": candidate.semantic_channel,
            "has_deterministic_slot": candidate.slot is not None,
            "action": decision.action,
            "dimension": decision.dimension,
            "score": decision.score,
            "margin": decision.margin,
            "reason": decision.reason,
            "protected_reference": decision.protected_reference,
            "matched_claim_id": decision.matched_claim_id,
            "strategy_version": decision.strategy_version,
            "router_version": decision.router_version,
            "routing_basis": resolution.routing_basis,
            "semantic_key": resolution.semantic_key,
            "proposed_key_matches_canonical": (resolution.semantic_key == canonical_key),
            "config_sha256": (
                self.semantic_policy.config_sha256 if self.semantic_policy is not None else None
            ),
            "prototype_sha256": (
                self.semantic_policy.prototype_sha256 if self.semantic_policy is not None else None
            ),
        }

    def _coordinate_candidate(
        self,
        session: Session,
        row: Outbox,
        event: Event,
        derivation_key: str,
        candidate: ClaimCandidate,
        semantic_key: str,
        routing_slot: str | None,
        routing_basis: str,
        router_version: str | None,
        active_by_key: dict[tuple[str, str], Claim],
        protected_skips: dict[str, list[Claim]],
    ) -> Claim | None:
        """Key-level coordination (design doc section 3.2), in-transaction."""
        existing_claim = active_by_key.get((event.user_id, semantic_key))
        if existing_claim is not None and existing_claim.source_type == "explicit_feedback":
            # D3: human corrections always win over machine extraction.
            protected_skips.setdefault(event.id, []).append(existing_claim)
            return existing_claim
        if existing_claim is not None:
            same_value = _normalize_value(existing_claim.value) == _normalize_value(candidate.value)
            # A stance flip on the same object is a contradiction, not a repeat:
            # "I prefer tea" -> "I no longer like tea" must supersede so the
            # reversal is readable from the claim chain.
            same_stance = existing_claim.stance == candidate.stance
            if (
                existing_claim.extractor_version == row.processor_version
                and same_value
                and same_stance
            ):
                # Same version, key and value: reinforce, do not duplicate.
                if all(item.event_id != event.id for item in existing_claim.evidence):
                    existing_claim.evidence.append(
                        ClaimEvidence(event_id=event.id, relation="explicit_statement")
                    )
                    existing_claim.updated_at = _now()
                return existing_claim
            reason = (
                "version_upgrade"
                if existing_claim.extractor_version != row.processor_version
                else "contradiction"
            )
            # Deactivate and flush before inserting the successor: the partial
            # unique index is checked immediately on both dialects.
            self._supersede_claim(session, existing_claim, reason=reason)
            session.flush()
            claim = self._build_claim(
                session,
                row,
                event,
                derivation_key,
                candidate,
                semantic_key,
                routing_slot,
                routing_basis,
                router_version,
            )
            session.flush()
            existing_claim.superseded_by_id = claim.id
            claim.supersedes_id = existing_claim.id
            session.add(
                ClaimEdge(
                    id=_id(),
                    user_id=event.user_id,
                    source_claim_id=claim.id,
                    target_claim_id=existing_claim.id,
                    relation_type="supersedes",
                    confidence=1.0,
                )
            )
            active_by_key[(event.user_id, semantic_key)] = claim
            return claim
        claim = self._build_claim(
            session,
            row,
            event,
            derivation_key,
            candidate,
            semantic_key,
            routing_slot,
            routing_basis,
            router_version,
        )
        active_by_key[(event.user_id, semantic_key)] = claim
        return claim

    def _build_claim(
        self,
        session: Session,
        row: Outbox,
        event: Event,
        derivation_key: str,
        candidate: ClaimCandidate,
        semantic_key: str,
        routing_slot: str | None,
        routing_basis: str,
        router_version: str | None,
    ) -> Claim:
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
            # R1: stamp the row's version, not the running process's default.
            extractor_version=row.processor_version,
            semantic_key=semantic_key,
            stance=candidate.stance,
            routing_slot=routing_slot,
            routing_basis=routing_basis,
            router_version=router_version,
        )
        claim.evidence.append(ClaimEvidence(event_id=event.id, relation="explicit_statement"))
        session.add(claim)
        if not claim.sensitive:
            self._enqueue_projection_upsert(session, claim)
        return claim

    def _supersede_claim(
        self,
        session: Session,
        claim: Claim,
        *,
        reason: str,
        superseded_by_id: str | None = None,
    ) -> None:
        now = _now()
        claim.status = "superseded"
        claim.valid_to = now
        claim.updated_at = now
        claim.superseded_by_id = superseded_by_id
        claim.superseded_reason = reason
        self._enqueue_projection_delete_claim(session, claim.user_id, claim.id)

    def _derive_reflections(
        self, session: Session, user_ids: set[str], extractor_version: str
    ) -> dict[str, list[dict[str, Any]]]:
        """Derive `pattern` claims from each user's repeated preferences.

        Reflection is the second abstraction layer `Meno_SPEC.md` :119 asks for:
        `pattern` claims summarize preferences that recur across windows, and the
        retrieval side already reserves budget for them via ``STATE_LAYER_KINDS``.
        The synthesis itself lives in ``reflection.derive_patterns`` and is pure;
        this method only supplies state and persists the result.

        Returns per-user audit records so the caller can emit them once it knows
        the chunk's revision.
        """
        audits: dict[str, list[dict[str, Any]]] = {}
        for user_id in sorted(user_ids):
            # Superseded preferences are included on purpose: a paraphrased
            # restatement supersedes rather than merges (see _coordinate_candidate),
            # so the repetition a pattern summarizes lives in the chain, not in the
            # single active claim. `derive_patterns` uses stance, not supersede
            # reason, to tell a rephrasing from a change of mind.
            claims = session.scalars(
                select(Claim)
                .options(selectinload(Claim.evidence))
                .where(
                    Claim.user_id == user_id,
                    Claim.kind == "preference",
                    Claim.status.in_(("active", "superseded")),
                )
            ).all()
            if not claims:
                continue
            # Dimensions the user has explicitly corrected are off limits: an
            # inference must never overwrite an explicit statement (SPEC No-Go).
            # `_withdraw_stale_claims` protects explicit feedback on the write
            # path; this is the same rule for the reflection route.
            protected = frozenset(
                claim.semantic_key
                for claim in session.scalars(
                    select(Claim).where(
                        Claim.user_id == user_id,
                        Claim.source_type == "explicit_feedback",
                    )
                ).all()
            )
            evidence_ids = {
                claim.id: [item.event_id for item in claim.evidence] for claim in claims
            }
            needed = {event_id for ids in evidence_ids.values() for event_id in ids}
            events = {
                event.id: event
                for event in session.scalars(
                    select(Event).where(Event.id.in_(needed))
                ).all()
            } if needed else {}

            for candidate in derive_patterns(
                claims, events, evidence_ids, protected_semantic_keys=protected
            ):
                semantic_key = _semantic_key(
                    candidate.user_id,
                    "pattern",
                    candidate.semantic_channel,
                    candidate.value,
                    candidate.routing_slot,
                )
                derivation_key = _sha(
                    pattern_derivation_basis(
                        candidate.user_id,
                        semantic_key,
                        candidate.source_event_ids,
                        extractor_version,
                    )
                )
                if session.scalar(
                    select(Claim.id).where(Claim.derivation_key == derivation_key)
                ):
                    continue
                claim = self._build_pattern_claim(
                    session, candidate, semantic_key, derivation_key, extractor_version
                )
                audits.setdefault(candidate.user_id, []).append(
                    {
                        "claim_id": claim.id,
                        "source_event_ids": list(candidate.source_event_ids),
                        "decision": {
                            "allowed": True,
                            "reason": "reflection_derived",
                            "semantic_channel": candidate.semantic_channel,
                            "semantic_key": semantic_key,
                            "source_claim_ids": list(candidate.source_claim_ids),
                            "evidence_count": candidate.evidence_count,
                            "stance": candidate.stance,
                            "sensitive": candidate.sensitive,
                            "allowed_purposes": list(candidate.allowed_purposes),
                            "extractor_version": extractor_version,
                        },
                    }
                )
        return audits

    def _build_pattern_claim(
        self,
        session: Session,
        candidate: PatternCandidate,
        semantic_key: str,
        derivation_key: str,
        extractor_version: str,
    ) -> Claim:
        """Persist a pattern claim with evidence from every source event.

        Separate from ``_build_claim`` because that one derives `valid_from`,
        `allowed_purposes`, and `origin_role` from a single event. A pattern spans
        several, so those three are resolved by ``derive_patterns`` instead --
        `allowed_purposes` as an intersection rather than a union, which is what
        keeps the pattern from serving information under a purpose its narrowest
        source never consented to.

        Any existing active claim on the same semantic key is superseded, so a
        changed support set replaces the previous summary instead of coexisting
        with it.
        """
        existing = session.scalars(
            select(Claim).where(
                Claim.user_id == candidate.user_id,
                Claim.semantic_key == semantic_key,
                Claim.status == "active",
            )
        ).all()
        # Deactivate and flush before inserting the successor: the partial unique
        # index on (user_id, semantic_key) is checked immediately on both dialects,
        # so inserting first raises even though the old row is about to be retired.
        for prior in existing:
            self._supersede_claim(session, prior, reason="reflection_refreshed")
        if existing:
            session.flush()
        claim = Claim(
            id=_claim_id(derivation_key),
            derivation_key=derivation_key,
            user_id=candidate.user_id,
            kind="pattern",
            # Derived from user statements, never from assistant turns: only
            # user-role claims feed `derive_patterns`.
            origin_role="user",
            semantic_channel=candidate.semantic_channel,
            value=candidate.value,
            status="active",
            confidence=candidate.confidence,
            half_life_days=candidate.half_life_days,
            sensitive=candidate.sensitive,
            allowed_purposes=list(candidate.allowed_purposes),
            source_type="reflection",
            valid_from=candidate.valid_from,
            extractor_version=extractor_version,
            semantic_key=semantic_key,
            stance=candidate.stance,
            routing_slot=candidate.routing_slot,
            routing_basis="reflection",
            router_version=None,
        )
        for event_id in candidate.source_event_ids:
            claim.evidence.append(
                ClaimEvidence(event_id=event_id, relation="reflection_source")
            )
        session.add(claim)
        session.flush()
        for prior in existing:
            prior.superseded_by_id = claim.id
            claim.supersedes_id = prior.id
            session.add(
                ClaimEdge(
                    id=_id(),
                    user_id=candidate.user_id,
                    source_claim_id=claim.id,
                    target_claim_id=prior.id,
                    relation_type="supersedes",
                    confidence=1.0,
                )
            )
        if not claim.sensitive:
            self._enqueue_projection_upsert(session, claim)
        return claim

    def _withdraw_stale_claims(
        self,
        session: Session,
        event: Event,
        processor_version: str,
        produced: set[str],
        chunk_event_ids: set[str],
        active_by_key: dict[tuple[str, str], Claim],
    ) -> None:
        stale = session.scalars(
            select(Claim)
            .join(ClaimEvidence, ClaimEvidence.claim_id == Claim.id)
            .options(selectinload(Claim.evidence))
            .where(
                ClaimEvidence.event_id == event.id,
                Claim.status == "active",
                Claim.extractor_version != processor_version,
            )
        ).all()
        for claim in stale:
            if claim.source_type == "explicit_feedback":
                continue  # D3: never auto-withdraw human corrections
            if claim.semantic_key in produced:
                continue
            evidence_event_ids = {item.event_id for item in claim.evidence}
            pending = evidence_event_ids - chunk_event_ids
            if pending:
                processed = set(
                    session.scalars(
                        select(Outbox.event_id).where(
                            Outbox.event_id.in_(pending),
                            Outbox.processor_version == processor_version,
                            Outbox.status == "processed",
                        )
                    ).all()
                )
                if not evidence_event_ids <= (processed | chunk_event_ids):
                    # Some evidence events have not been reprocessed under this
                    # version yet; the claim still stands (design doc R5).
                    continue
            self._supersede_claim(session, claim, reason="extractor_withdrawn")
            active_by_key.pop((claim.user_id, claim.semantic_key), None)

    @staticmethod
    def _enqueue_projection_upsert(session: Session, claim: Claim) -> None:
        session.add(
            ProjectionOutbox(
                operation="upsert",
                user_id=claim.user_id,
                claim_id=claim.id,
                payload={
                    "status": claim.status,
                    "text": claim.value,
                    "valid_from": claim.valid_from.isoformat(),
                    "valid_to": claim.valid_to.isoformat() if claim.valid_to else None,
                },
            )
        )

    @staticmethod
    def _enqueue_projection_delete_claim(session: Session, user_id: str, claim_id: str) -> None:
        session.add(
            ProjectionOutbox(
                operation="delete_claim",
                user_id=user_id,
                claim_id=claim_id,
                payload={},
            )
        )

    @staticmethod
    def _enqueue_projection_delete_user(session: Session, user_id: str) -> None:
        session.add(
            ProjectionOutbox(
                operation="delete_user",
                user_id=user_id,
                claim_id=None,
                payload={},
            )
        )

    def process_projection_outbox(self, limit: int = 400) -> int:
        """Apply derived-index mutations durably and idempotently in order."""

        if not 1 <= limit <= 10_000:
            raise ValueError("projection outbox limit must be between 1 and 10000")
        with self._projection_lock, self.session_factory.begin() as session:
            if session.get_bind().dialect.name == "postgresql":
                session.execute(text("SELECT pg_advisory_xact_lock(72460320260823)"))
            now = _now()
            rows = session.scalars(
                select(ProjectionOutbox)
                .where(
                    ProjectionOutbox.status == "pending",
                    (ProjectionOutbox.next_attempt_at.is_(None))
                    | (ProjectionOutbox.next_attempt_at <= now),
                )
                .order_by(ProjectionOutbox.id)
                .limit(limit)
                .with_for_update(skip_locked=True)
            ).all()
            processed = 0
            index = 0
            while index < len(rows):
                row = rows[index]
                if row.operation == "upsert":
                    end = index + 1
                    while end < len(rows) and rows[end].operation == "upsert":
                        end += 1
                    group = rows[index:end]
                    documents = [self._projection_document(item) for item in group]
                    try:
                        self.vector_store.upsert_many(documents)
                    except Exception:  # noqa: BLE001 - isolate poison projection rows
                        for item, document in zip(group, documents, strict=True):
                            try:
                                self.vector_store.upsert_many([document])
                            except Exception as exc:  # noqa: BLE001 - durable retry
                                self._mark_projection_failure(item, exc, now)
                                return processed
                            self._mark_projection_processed(item)
                            processed += 1
                    else:
                        for item in group:
                            self._mark_projection_processed(item)
                        processed += len(group)
                    index = end
                    continue
                try:
                    if row.operation == "delete_claim":
                        self.vector_store.delete_claim(row.claim_id or "")
                    elif row.operation == "delete_user":
                        self.vector_store.delete_user(row.user_id)
                    else:
                        raise ValueError("unsupported projection operation")
                except Exception as exc:  # noqa: BLE001 - durable retry boundary
                    self._mark_projection_failure(row, exc, now)
                    return processed
                self._mark_projection_processed(row)
                processed += 1
                index += 1
            return processed

    @staticmethod
    def _projection_document(row: ProjectionOutbox) -> VectorDocument:
        payload = row.payload
        return VectorDocument(
            claim_id=row.claim_id or "",
            user_id=row.user_id,
            status=payload["status"],
            text=payload["text"],
            valid_from=datetime.fromisoformat(payload["valid_from"]),
            valid_to=(
                datetime.fromisoformat(payload["valid_to"]) if payload.get("valid_to") else None
            ),
        )

    @staticmethod
    def _mark_projection_processed(row: ProjectionOutbox) -> None:
        row.status = "processed"
        row.processed_at = _now()
        row.error = None
        row.next_attempt_at = None

    def _mark_projection_failure(
        self, row: ProjectionOutbox, exc: Exception, now: datetime
    ) -> None:
        row.attempts += 1
        row.error = type(exc).__name__
        if row.attempts >= 20:
            row.status = "failed"
            row.next_attempt_at = None
            return
        delay = min(
            self.settings.outbox_retry_max_seconds,
            self.settings.outbox_retry_base_seconds * (2 ** (row.attempts - 1)),
        )
        row.next_attempt_at = now + timedelta(seconds=delay)

    def _mark_outbox_failure(self, row_ids: list[str], exc: Exception) -> None:
        with self.session_factory.begin() as failure_session:
            failed_rows = failure_session.scalars(
                select(Outbox).where(Outbox.id.in_(row_ids)).with_for_update(skip_locked=True)
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
        with self._audit_chain_lock, self.session_factory.begin() as session:
            statement = select(Outbox).where(Outbox.status == "failed").order_by(Outbox.created_at)
            if limit is not None:
                statement = statement.limit(limit)
            rows = session.scalars(statement).all()
            for row in rows:
                row.status = "pending"
                row.attempts = 0
                row.error = None
                row.next_attempt_at = None
            return len(rows)

    def requeue_failed_projection(self, limit: int | None = None) -> int:
        with self._audit_chain_lock, self.session_factory.begin() as session:
            statement = (
                select(ProjectionOutbox)
                .where(ProjectionOutbox.status == "failed")
                .order_by(ProjectionOutbox.id)
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
            outbox_counts = {
                status: session.scalar(
                    select(func.count(Outbox.id))
                    .join(Event, Outbox.event_id == Event.id)
                    .where(Event.user_id == user_id, Outbox.status == status)
                )
                or 0
                for status in ("pending", "failed")
            }
            projection_counts = {
                status: session.scalar(
                    select(func.count(ProjectionOutbox.id)).where(
                        ProjectionOutbox.user_id == user_id,
                        ProjectionOutbox.status == status,
                    )
                )
                or 0
                for status in ("pending", "failed")
            }
        return {
            "user_id": user_id,
            "pending_outbox": outbox_counts["pending"],
            "failed_outbox": outbox_counts["failed"],
            "pending_projection": projection_counts["pending"],
            "failed_projection": projection_counts["failed"],
            "drained": not any(outbox_counts.values()) and not any(projection_counts.values()),
            "policy_version": self.settings.policy_version,
        }

    def reprocess(
        self,
        extractor_version: str,
        user_id: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Enqueue replay of existing events under a new extractor version.

        Idempotent: (event_id, processor_version) is unique and re-runs insert
        nothing; other-version pending rows for the target events are cancelled
        so a stale worker can never mislabel them (design doc section 2.1).
        """
        if not extractor_version:
            raise ValueError("extractor_version is required")
        with self._audit_chain_lock, self.session_factory.begin() as session:
            statement = select(Event.id).order_by(Event.created_at)
            if user_id is not None:
                statement = statement.where(Event.user_id == user_id)
            if limit is not None:
                statement = statement.limit(limit)
            event_ids = session.scalars(statement).all()
            if not event_ids:
                return {
                    "extractor_version": extractor_version,
                    "targeted": 0,
                    "inserted": 0,
                    "cancelled": 0,
                }
            values = [
                {"id": _id(), "event_id": event_id, "processor_version": extractor_version}
                for event_id in event_ids
            ]
            dialect = session.get_bind().dialect.name
            if dialect == "postgresql":
                # rowcount is -1 for a multi-row VALUES insert on PostgreSQL, so
                # count the RETURNING rows instead: ON CONFLICT DO NOTHING only
                # returns rows that were actually inserted.
                result = session.execute(
                    postgresql_insert(Outbox)
                    .values(values)
                    .on_conflict_do_nothing(index_elements=["event_id", "processor_version"])
                    .returning(Outbox.event_id)
                )
                inserted = len(result.all())
            elif dialect == "sqlite":
                result = session.execute(
                    sqlite_insert(Outbox)
                    .values(values)
                    .on_conflict_do_nothing(index_elements=["event_id", "processor_version"])
                )
                inserted = result.rowcount
            else:
                present = set(
                    session.scalars(
                        select(Outbox.event_id).where(
                            Outbox.event_id.in_(event_ids),
                            Outbox.processor_version == extractor_version,
                        )
                    ).all()
                )
                inserted = 0
                for value in values:
                    if value["event_id"] not in present:
                        session.add(Outbox(**value))
                        inserted += 1
            cancelled = session.execute(
                update(Outbox)
                .where(
                    Outbox.event_id.in_(event_ids),
                    Outbox.processor_version != extractor_version,
                    Outbox.status == "pending",
                )
                .values(status="cancelled")
            ).rowcount
        return {
            "extractor_version": extractor_version,
            "targeted": len(event_ids),
            "inserted": int(inserted),
            "cancelled": int(cancelled),
        }

    def migrate(self, batch_size: int = 500) -> dict[str, Any]:
        """Idempotent schema migration for the claim coordination protocol.

        Steps (design doc section 4.1): add the coordination columns, backfill
        semantic_key, dedupe existing active claims per (user, semantic_key),
        then create the partial unique index. Every step is re-runnable.
        """
        if not 1 <= batch_size <= 10_000:
            raise ValueError("migration batch size must be between 1 and 10000")
        bind = self.engine if self.engine is not None else self.session_factory.kw["bind"]
        dialect = bind.dialect.name
        report: dict[str, Any] = {
            "columns_added": [],
            "backfilled": 0,
            "routing_backfilled": 0,
            "deduplicated": 0,
            "index_created": False,
        }
        with bind.begin() as connection:
            existing = self._existing_columns(connection, dialect)
            for name, ddl in _MIGRATION_COLUMNS:
                if name in existing:
                    continue
                connection.execute(text(f"ALTER TABLE meno_claims ADD COLUMN {name} {ddl}"))
                report["columns_added"].append(name)
            connection.execute(text("DROP INDEX IF EXISTS uq_claim_active_semantic"))

        with self._audit_chain_lock, self.session_factory.begin() as session:
            active_claims = session.scalars(
                select(Claim).where(Claim.status == "active").with_for_update()
            ).all()
            for claim in active_claims:
                if claim.routing_basis == "semantic_router":
                    continue
                slot = preference_slot(claim.value)
                target_key = _semantic_key(
                    claim.user_id,
                    claim.kind,
                    claim.semantic_channel,
                    claim.value,
                    slot,
                )
                if claim.semantic_key != target_key:
                    claim.semantic_key = target_key
                    report["backfilled"] += 1
                target_basis = "deterministic_slot" if slot else "none"
                if (
                    claim.routing_slot != slot
                    or claim.routing_basis != target_basis
                    or claim.router_version is not None
                ):
                    claim.routing_slot = slot
                    claim.routing_basis = target_basis
                    claim.router_version = None
                    report["routing_backfilled"] += 1

        while True:
            with self._audit_chain_lock, self.session_factory.begin() as session:
                claims = session.scalars(
                    select(Claim).where(Claim.semantic_key == "").limit(batch_size)
                ).all()
                if not claims:
                    break
                for claim in claims:
                    claim.semantic_key = _semantic_key(
                        claim.user_id,
                        claim.kind,
                        claim.semantic_channel,
                        claim.value,
                        claim.routing_slot,
                    )
                report["backfilled"] += len(claims)

        report["deduplicated"] = self._dedupe_active_claims()

        with bind.begin() as connection:
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_claim_active_semantic "
                    "ON meno_claims (user_id, semantic_key) WHERE status = 'active'"
                )
            )
        report["index_created"] = True
        return report

    def revert_routing_migration(self) -> dict[str, int]:
        """Clear Phase B routing provenance without dropping schema or audit rows."""

        bind = self.engine if self.engine is not None else self.session_factory.kw["bind"]
        with bind.begin() as connection:
            existing = self._existing_columns(connection, bind.dialect.name)
        required = {"routing_slot", "routing_basis", "router_version"}
        if not required <= existing:
            return {"cleared": 0}
        with self._audit_chain_lock, self.session_factory.begin() as session:
            cleared = (
                session.scalar(
                    select(func.count(Claim.id)).where(
                        (Claim.routing_slot.is_not(None))
                        | (Claim.routing_basis.is_not(None))
                        | (Claim.router_version.is_not(None))
                    )
                )
                or 0
            )
            session.execute(
                update(Claim).values(
                    routing_slot=None,
                    routing_basis=None,
                    router_version=None,
                )
            )
        return {"cleared": int(cleared)}

    @staticmethod
    def _existing_columns(connection, dialect: str) -> set[str]:
        if dialect == "sqlite":
            rows = connection.execute(text("PRAGMA table_info(meno_claims)")).all()
            return {row[1] for row in rows}
        rows = connection.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'meno_claims'"
            )
        ).all()
        return {row[0] for row in rows}

    def _dedupe_active_claims(self) -> int:
        deduplicated = 0
        with self._audit_chain_lock, self.session_factory.begin() as session:
            duplicates = session.execute(
                select(Claim.user_id, Claim.semantic_key)
                .where(Claim.status == "active", Claim.semantic_key != "")
                .group_by(Claim.user_id, Claim.semantic_key)
                .having(func.count() > 1)
            ).all()
            for user_id, semantic_key in duplicates:
                claims = session.scalars(
                    select(Claim)
                    .where(
                        Claim.user_id == user_id,
                        Claim.semantic_key == semantic_key,
                        Claim.status == "active",
                    )
                    .order_by(Claim.valid_from.desc(), Claim.id.desc())
                ).all()
                keeper, *stale = claims
                for claim in stale:
                    self._supersede_claim(
                        session,
                        claim,
                        reason="version_upgrade",
                        superseded_by_id=keeper.id,
                    )
                    session.add(
                        ClaimEdge(
                            id=_id(),
                            user_id=user_id,
                            source_claim_id=keeper.id,
                            target_claim_id=claim.id,
                            relation_type="supersedes",
                            confidence=1.0,
                        )
                    )
                    deduplicated += 1
        return deduplicated

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

        with self._audit_chain_lock, self.session_factory.begin() as session:
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

            if self.settings.context_activation_enabled and request.constraints.allow_sensitive:
                known_claim_ids = {hit.claim_id for hit in hits}
                sensitive_claims = session.scalars(
                    select(Claim)
                    .where(
                        Claim.user_id == request.user_id,
                        Claim.status == "active",
                        Claim.sensitive.is_(True),
                    )
                    .order_by(Claim.updated_at.desc())
                    .limit(max(32, request.constraints.max_facets * 4))
                ).all()
                for claim in sensitive_claims:
                    if claim.id in known_claim_ids:
                        continue
                    decision = self.context_activation_policy.decide(
                        query=request.context.query,
                        task_type=request.context.task_type,
                        kind=claim.kind,
                        value=claim.value,
                        routing_slot=claim.routing_slot,
                        sensitive=True,
                        semantic_score=None,
                    )
                    if decision.allowed:
                        hits.append(
                            type(
                                "Hit",
                                (),
                                {
                                    "claim_id": claim.id,
                                    "score": decision.lexical_score,
                                },
                            )
                        )

            claim_ids = [hit.claim_id for hit in hits]
            claims_by_id = {
                claim.id: claim
                for claim in session.scalars(
                    select(Claim)
                    .options(selectinload(Claim.evidence))
                    .where(Claim.id.in_(claim_ids))
                ).all()
            }
            histories_by_claim = self._load_preference_histories(
                session, list(claims_by_id.values()), request
            )
            scope = PURPOSE_SCOPE[request.purpose]
            source_types = {claim.source_type for claim in claims_by_id.values()}
            source_types.update(
                historical.source_type
                for history in histories_by_claim.values()
                for historical in history
            )
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
            clarification_opportunities = self._relevant_clarification_opportunities(
                session, request
            )
            clarification_claim_ids = {
                opportunity.claim_id for opportunity in clarification_opportunities
            }
            facets: list[Facet] = []
            enriched_history_facets = 0
            for hit in hits:
                claim = claims_by_id.get(hit.claim_id)
                if claim is None:
                    continue
                allowed, reasons, effective_confidence = self._admit(
                    claim, request, revoked_sources
                )
                if not allowed:
                    continue
                if claim.id in clarification_claim_ids:
                    continue
                if self.settings.context_activation_enabled:
                    activation = self.context_activation_policy.decide(
                        query=request.context.query,
                        task_type=request.context.task_type,
                        kind=claim.kind,
                        value=claim.value,
                        routing_slot=claim.routing_slot,
                        sensitive=bool(claim.sensitive),
                        semantic_score=hit.score,
                    )
                    if not activation.allowed:
                        continue
                    reasons = [*reasons, *activation.reasons]
                evidence_ids = [item.event_id for item in claim.evidence]
                value: Any = claim.value
                history = histories_by_claim.get(claim.id, [])
                if (
                    history
                    and enriched_history_facets < self.settings.preference_history_max_facets
                ):
                    admitted_history = [
                        historical
                        for historical in history
                        if self._admit_historical_claim(historical, request, revoked_sources)
                    ]
                    distinct_history: list[Claim] = []
                    previous_value: str | None = None
                    for historical in admitted_history:
                        normalized = _normalize_value(historical.value)
                        if normalized in {previous_value, _normalize_value(claim.value)}:
                            continue
                        distinct_history.append(historical)
                        previous_value = normalized
                    if distinct_history:
                        value = {
                            "current": claim.value,
                            "history_oldest_to_newest": [
                                historical.value for historical in distinct_history
                            ],
                        }
                        evidence_ids = list(
                            dict.fromkeys(
                                evidence_ids
                                + [
                                    item.event_id
                                    for historical in distinct_history
                                    for item in historical.evidence
                                ]
                            )
                        )
                        reasons = [*reasons, "bounded supersedes history"]
                        enriched_history_facets += 1
                score = self._score(hit.score, effective_confidence, claim, as_of)
                facets.append(
                    Facet(
                        claim_id=claim.id,
                        kind=claim.kind,
                        value=value,
                        stance=claim.stance,
                        relevance=score,
                        confidence=effective_confidence,
                        evidence_ids=evidence_ids,
                        why_selected=reasons,
                    )
                )
            facets.sort(key=lambda item: item.relevance, reverse=True)
            facets = self._apply_facet_budget(
                facets, request.constraints.max_facets, request.context.query
            )
            rendered = self._render(
                request.user_id, revision, facets, request.constraints.max_rendered_tokens
            )
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
                    "history_count": (
                        len(facet.value.get("history_oldest_to_newest", []))
                        if isinstance(facet.value, dict)
                        else 0
                    ),
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
            clarification_opportunities=clarification_opportunities,
            degraded=degraded,
            policy_version=self.settings.policy_version,
        )

    def feedback(
        self, request: FeedbackRequest, *, idempotency_key: str = ""
    ) -> dict[str, Any]:
        if request.action == "correct" and not request.correction:
            raise ValueError("correction is required for correct action")
        trace_id = _id()
        fingerprint = _sha(request.model_dump_json())
        new_claim_id: str | None = None
        old_claim_id = request.claim_id
        with self._audit_chain_lock, self.session_factory.begin() as session:
            replay = self._idempotent_replay(
                session, endpoint="feedback", key=idempotency_key, fingerprint=fingerprint
            )
            if replay is not None:
                return replay
            claim = session.get(Claim, request.claim_id)
            if claim is None or claim.user_id != request.user_id:
                raise LookupError("claim not found")
            if request.action == "correct" and claim.status != "active":
                # D5 (review #B-8): correcting an already-corrected claim
                # replays idempotently instead of creating a duplicate.
                if claim.status == "superseded" and claim.superseded_by_id is not None:
                    prior = session.get(Claim, claim.superseded_by_id)
                    if prior is not None and prior.source_type == "explicit_feedback":
                        return {
                            "trace_id": trace_id,
                            "state_revision": self._revision(session, request.user_id),
                            "claim_id": prior.id,
                            "superseded_claim_id": claim.id,
                            "idempotent_replay": True,
                            "policy_version": self.settings.policy_version,
                        }
                raise ValueError(f"cannot correct a claim in status {claim.status!r}")
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
                if not claim.sensitive:
                    self._enqueue_projection_upsert(session, claim)
            elif request.action == "reject":
                claim.status = "rejected"
                claim.valid_to = _now()
                claim.updated_at = _now()
                self._enqueue_projection_delete_claim(session, claim.user_id, claim.id)
            else:
                correction = request.correction or ""
                replacement_key = _semantic_key(
                    claim.user_id,
                    claim.kind,
                    claim.semantic_channel,
                    correction,
                    preference_slot(correction),
                )
                correction_slot = preference_slot(correction)
                claim.status = "superseded"
                claim.valid_to = _now()
                claim.updated_at = _now()
                claim.superseded_reason = "feedback_correct"
                self._enqueue_projection_delete_claim(session, claim.user_id, claim.id)
                session.flush()
                # A machine claim holding the same key yields to the human
                # correction (D3 applies in both directions).
                conflicts = session.scalars(
                    select(Claim).where(
                        Claim.user_id == claim.user_id,
                        Claim.semantic_key == replacement_key,
                        Claim.status == "active",
                        Claim.id != claim.id,
                    )
                ).all()
                for conflict in conflicts:
                    self._supersede_claim(session, conflict, reason="feedback_correct")
                if conflicts:
                    session.flush()
                feedback_event = Event(
                    id=_id(),
                    user_id=claim.user_id,
                    occurred_at=_now(),
                    source_type="explicit_feedback",
                    source_profile=None,
                    session_id=None,
                    role="user",
                    content=correction,
                    content_hash=_sha(correction),
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
                    value=correction,
                    status="active",
                    confidence=0.99,
                    half_life_days=claim.half_life_days,
                    sensitive=claim.sensitive,
                    allowed_purposes=claim.allowed_purposes,
                    source_type="explicit_feedback",
                    valid_from=_now(),
                    supersedes_id=claim.id,
                    extractor_version="explicit-feedback-v1",
                    semantic_key=replacement_key,
                    routing_slot=correction_slot,
                    routing_basis=("deterministic_slot" if correction_slot is not None else "none"),
                    router_version=None,
                )
                session.add(replacement)
                session.flush()
                new_claim_id = replacement.id
                claim.superseded_by_id = replacement.id
                for conflict in conflicts:
                    conflict.superseded_by_id = replacement.id
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
                if not replacement.sensitive:
                    self._enqueue_projection_upsert(session, replacement)
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
            self._materialize_user_state(session, request.user_id, revision)
            result = {
                "trace_id": trace_id,
                "state_revision": revision,
                "claim_id": new_claim_id or old_claim_id,
                "superseded_claim_id": old_claim_id if new_claim_id else None,
                "policy_version": self.settings.policy_version,
            }
            self._record_idempotency(
                session,
                endpoint="feedback",
                key=idempotency_key,
                fingerprint=fingerprint,
                response=result,
            )
        self.process_projection_outbox()
        return result

    def set_consent(
        self, request: ConsentRequest, *, idempotency_key: str = ""
    ) -> dict[str, Any]:
        trace_id = _id()
        fingerprint = _sha(request.model_dump_json())
        with self._audit_chain_lock, self.session_factory.begin() as session:
            replay = self._idempotent_replay(
                session, endpoint="consent", key=idempotency_key, fingerprint=fingerprint
            )
            if replay is not None:
                return replay
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
            self._materialize_user_state(session, request.user_id, revision)
            result = {
                "trace_id": trace_id,
                "consent_id": consent.id,
                "state_revision": revision,
                "policy_version": self.settings.policy_version,
            }
            self._record_idempotency(
                session,
                endpoint="consent",
                key=idempotency_key,
                fingerprint=fingerprint,
                response=result,
            )
        return result

    def delete(
        self, request: DeletionRequest, *, idempotency_key: str = ""
    ) -> dict[str, Any]:
        if request.scope == "source" and not request.source:
            raise ValueError("source is required")
        if request.scope == "claim" and not request.claim_id:
            raise ValueError("claim_id is required")
        job_id = _id()
        subject_hash = _sha(request.user_id)
        trace_id = _id()
        fingerprint = _sha(request.model_dump_json())
        with self._audit_chain_lock, self.session_factory.begin() as session:
            replay = self._idempotent_replay(
                session, endpoint="deletion", key=idempotency_key, fingerprint=fingerprint
            )
            if replay is not None:
                return replay
            job = DeletionJob(id=job_id, subject_hash=subject_hash, scope=request.scope)
            session.add(job)
            if request.scope == "claim":
                claim = session.get(Claim, request.claim_id)
                if claim and claim.user_id == request.user_id:
                    self._enqueue_projection_delete_claim(session, request.user_id, claim.id)
                    session.delete(claim)
            elif request.scope == "source":
                claims = session.scalars(
                    select(Claim).where(
                        Claim.user_id == request.user_id,
                        Claim.source_type == request.source,
                    )
                ).all()
                for claim in claims:
                    self._enqueue_projection_delete_claim(session, request.user_id, claim.id)
                    session.delete(claim)
                session.execute(
                    delete(Event).where(
                        Event.user_id == request.user_id,
                        Event.source_type == request.source,
                    )
                )
            else:
                self._enqueue_projection_delete_user(session, request.user_id)
                session.execute(
                    delete(UserTokenSnapshotDelta).where(
                        UserTokenSnapshotDelta.user_id == request.user_id
                    )
                )
                session.execute(
                    delete(UserTokenSnapshot).where(UserTokenSnapshot.user_id == request.user_id)
                )
                session.execute(
                    delete(PreferenceDistribution).where(
                        PreferenceDistribution.user_id == request.user_id
                    )
                )
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
            if request.scope != "all":
                self._materialize_user_state(session, request.user_id, revision)
            result = {
                "trace_id": trace_id,
                "deletion_job_id": job_id,
                "state_revision": revision,
                "receipt_hash": receipt,
                "status": "completed",
            }
            self._record_idempotency(
                session,
                endpoint="deletion",
                key=idempotency_key,
                fingerprint=fingerprint,
                response=result,
            )
        self.process_projection_outbox()
        return result

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
                    "superseded_by": claim.superseded_by_id,
                    "superseded_reason": claim.superseded_reason,
                },
                "semantic_key": claim.semantic_key,
                "routing": {
                    "slot": claim.routing_slot,
                    "basis": claim.routing_basis,
                    "router_version": claim.router_version,
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

    def user_token(self, user_id: str, revision: int | None = None) -> dict[str, Any]:
        with self.session_factory() as session:
            snapshot = self._user_snapshot_record(session, user_id, revision=revision)
            if snapshot is None:
                raise LookupError("user token snapshot not found")
            payload = self._read_user_snapshot_payload(session, snapshot)
            return {
                "snapshot_id": snapshot.id,
                "user_id": snapshot.user_id,
                "state_revision": snapshot.state_revision,
                "schema_version": snapshot.schema_version,
                "policy_version": snapshot.policy_version,
                "extractor_version": snapshot.extractor_version,
                "content_hash": snapshot.content_hash,
                "payload": payload,
            }

    @staticmethod
    def _user_snapshot_record(
        session: Session,
        user_id: str,
        *,
        revision: int | None = None,
        before_revision: int | None = None,
    ) -> UserTokenSnapshot | UserTokenSnapshotDelta | None:
        base_query = select(UserTokenSnapshot).where(UserTokenSnapshot.user_id == user_id)
        delta_query = select(UserTokenSnapshotDelta).where(
            UserTokenSnapshotDelta.user_id == user_id
        )
        if revision is not None:
            base_query = base_query.where(UserTokenSnapshot.state_revision == revision)
            delta_query = delta_query.where(UserTokenSnapshotDelta.state_revision == revision)
        elif before_revision is not None:
            base_query = base_query.where(
                UserTokenSnapshot.state_revision < before_revision
            ).order_by(UserTokenSnapshot.state_revision.desc())
            delta_query = delta_query.where(
                UserTokenSnapshotDelta.state_revision < before_revision
            ).order_by(UserTokenSnapshotDelta.state_revision.desc())
        else:
            base_query = base_query.order_by(UserTokenSnapshot.state_revision.desc())
            delta_query = delta_query.order_by(UserTokenSnapshotDelta.state_revision.desc())
        base = session.scalar(base_query.limit(1))
        delta = session.scalar(delta_query.limit(1))
        if base is not None and delta is not None and base.state_revision == delta.state_revision:
            raise ValueError("user token revision has both base and delta records")
        records = [record for record in (base, delta) if record is not None]
        return max(records, key=lambda record: record.state_revision) if records else None

    def _read_user_snapshot_payload(
        self, session: Session, target: UserTokenSnapshot | UserTokenSnapshotDelta
    ) -> dict[str, Any]:
        if isinstance(target, UserTokenSnapshot):
            payload = target.payload
            if (
                not isinstance(payload, dict)
                or payload.get("user_id") != target.user_id
                or payload.get("state_revision") != target.state_revision
                or content_hash(payload) != target.content_hash
            ):
                raise ValueError("user token base hash or metadata mismatch")
            return payload

        base_revision = target.base_revision
        if (
            base_revision >= target.state_revision
            or target.state_revision - base_revision > _MAX_USER_TOKEN_DELTA_SPAN
        ):
            raise ValueError("user token delta has an invalid base revision")
        base = session.scalar(
            select(UserTokenSnapshot).where(
                UserTokenSnapshot.user_id == target.user_id,
                UserTokenSnapshot.state_revision == base_revision,
            )
        )
        if base is None or not isinstance(base.payload, dict):
            raise ValueError("user token delta base is missing")
        if (
            base.payload.get("user_id") != base.user_id
            or base.payload.get("state_revision") != base.state_revision
            or content_hash(base.payload) != base.content_hash
        ):
            raise ValueError("user token base hash or metadata mismatch")
        competing_base = session.scalar(
            select(UserTokenSnapshotDelta.id).where(
                UserTokenSnapshotDelta.user_id == target.user_id,
                UserTokenSnapshotDelta.state_revision == base_revision,
            )
        )
        if competing_base is not None:
            raise ValueError("user token base revision also has a delta")
        intervening_bases = session.scalars(
            select(UserTokenSnapshot.state_revision).where(
                UserTokenSnapshot.user_id == target.user_id,
                UserTokenSnapshot.state_revision > base_revision,
                UserTokenSnapshot.state_revision <= target.state_revision,
            )
        ).all()
        if intervening_bases:
            raise ValueError("user token delta chain crosses another base")
        deltas = session.scalars(
            select(UserTokenSnapshotDelta)
            .where(
                UserTokenSnapshotDelta.user_id == target.user_id,
                UserTokenSnapshotDelta.state_revision > base_revision,
                UserTokenSnapshotDelta.state_revision <= target.state_revision,
            )
            .order_by(UserTokenSnapshotDelta.state_revision)
        ).yield_per(32)

        payload = base.payload
        previous_revision = base_revision
        last_delta_id = None
        for delta_count, delta in enumerate(deltas, 1):
            if delta_count > _MAX_USER_TOKEN_DELTA_SPAN:
                raise ValueError("user token delta chain exceeds the safety limit")
            if (
                delta.base_revision != base_revision
                or delta.previous_revision != previous_revision
                or delta.state_revision <= previous_revision
                or (delta.schema_version, delta.policy_version, delta.extractor_version)
                != (base.schema_version, base.policy_version, base.extractor_version)
            ):
                raise ValueError("user token delta chain is broken")
            payload = apply_delta(payload, delta.delta, delta.user_id, delta.state_revision)
            previous_revision = delta.state_revision
            last_delta_id = delta.id
        if (
            previous_revision != target.state_revision
            or last_delta_id != target.id
            or content_hash(payload) != target.content_hash
        ):
            raise ValueError("user token delta target is unreachable or has a hash mismatch")
        return payload

    def diff_user_tokens(
        self, user_id: str, from_revision: int, to_revision: int
    ) -> dict[str, Any]:
        before = self.user_token(user_id, from_revision)
        after = self.user_token(user_id, to_revision)
        before_state = {item["semantic_key"]: item for item in before["payload"]["active_state"]}
        after_state = {item["semantic_key"]: item for item in after["payload"]["active_state"]}
        added_keys = sorted(after_state.keys() - before_state.keys())
        removed_keys = sorted(before_state.keys() - after_state.keys())
        changed_keys = sorted(
            key
            for key in before_state.keys() & after_state.keys()
            if before_state[key] != after_state[key]
        )
        return {
            "user_id": user_id,
            "from_revision": from_revision,
            "to_revision": to_revision,
            "added": [after_state[key] for key in added_keys],
            "removed": [before_state[key] for key in removed_keys],
            "changed": [
                {"before": before_state[key], "after": after_state[key]} for key in changed_keys
            ],
            "consent_changed": (
                before["payload"]["consent_state"] != after["payload"]["consent_state"]
            ),
            "consent_before": before["payload"]["consent_state"],
            "consent_after": after["payload"]["consent_state"],
            "preference_distributions_changed": (
                before["payload"].get("preference_distributions", [])
                != after["payload"].get("preference_distributions", [])
            ),
            "preference_distributions_before": before["payload"].get(
                "preference_distributions", []
            ),
            "preference_distributions_after": after["payload"].get("preference_distributions", []),
            "policy_changed": before["policy_version"] != after["policy_version"],
            "extractor_changed": (before["extractor_version"] != after["extractor_version"]),
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
            "extractor_version": self.settings.extractor_version,
            "embedding_provider": self.settings.embedding_provider,
            "embedding_model": self.settings.embedding_model,
            "embedding_projection_version": self.settings.embedding_projection_version,
            "semantic_routing_enabled": self.semantic_policy is not None,
            "preference_history_retrieval_enabled": (
                self.settings.preference_history_retrieval_enabled
            ),
            "preference_history_max_facets": (self.settings.preference_history_max_facets),
            "preference_history_max_events": (self.settings.preference_history_max_events),
            "semantic_router_config_sha256": (
                self.semantic_policy.config_sha256 if self.semantic_policy is not None else None
            ),
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

    def _load_preference_histories(
        self,
        session: Session,
        active_claims: list[Claim],
        request: RetrieveRequest,
    ) -> dict[str, list[Claim]]:
        if (
            not self.settings.preference_history_retrieval_enabled
            or request.context.as_of is not None
        ):
            return {}
        chains: dict[str, list[Claim]] = {
            claim.id: []
            for claim in active_claims
            if claim.kind == "preference" and claim.supersedes_id is not None
        }
        frontier = {
            claim.id: claim.supersedes_id
            for claim in active_claims
            if claim.id in chains and claim.supersedes_id is not None
        }
        seen: dict[str, set[str]] = {claim_id: set() for claim_id in chains}
        for _ in range(self.settings.preference_history_max_events):
            requested_ids = {claim_id for claim_id in frontier.values() if claim_id}
            if not requested_ids:
                break
            loaded = {
                claim.id: claim
                for claim in session.scalars(
                    select(Claim)
                    .options(selectinload(Claim.evidence))
                    .where(Claim.id.in_(requested_ids))
                ).all()
            }
            next_frontier: dict[str, str | None] = {}
            for active_id, historical_id in frontier.items():
                if historical_id is None or historical_id in seen[active_id]:
                    continue
                seen[active_id].add(historical_id)
                historical = loaded.get(historical_id)
                if (
                    historical is None
                    or historical.status != "superseded"
                    or historical.user_id != request.user_id
                ):
                    continue
                chains[active_id].append(historical)
                next_frontier[active_id] = historical.supersedes_id
            frontier = next_frontier
        return {
            claim_id: list(reversed(history)) for claim_id, history in chains.items() if history
        }

    def _admit_historical_claim(
        self,
        claim: Claim,
        request: RetrieveRequest,
        revoked_sources: set[str],
    ) -> bool:
        if claim.user_id != request.user_id or claim.status != "superseded":
            return False
        scope = PURPOSE_SCOPE[request.purpose]
        if scope not in claim.allowed_purposes and request.purpose not in claim.allowed_purposes:
            return False
        if claim.sensitive and not request.constraints.allow_sensitive:
            return False
        if claim.origin_role == "assistant" and request.context.task_type != "conversation_recall":
            return False
        effective_at = _aware(claim.valid_to) if claim.valid_to else _now()
        if self._effective_confidence(claim, effective_at) < request.constraints.min_confidence:
            return False
        if claim.source_type in revoked_sources:
            return False
        return bool(claim.evidence) or claim.source_type == "explicit_feedback"

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
        return (
            True,
            ["semantic match", "active claim", "evidence present", "purpose allowed"],
            effective,
        )

    def _effective_confidence(self, claim: Claim, as_of: datetime | None = None) -> float:
        if not claim.half_life_days:
            return claim.confidence
        # Evidence reinforcement: re-confirmed preferences (same semantic key,
        # multiple evidence events) get a higher base and a longer effective
        # half-life instead of purely decaying with age.
        evidence_count = max(1, len(claim.evidence))
        reinforcement = math.log2(evidence_count)
        base = min(0.99, claim.confidence + EVIDENCE_CONFIDENCE_STEP * reinforcement)
        half_life = claim.half_life_days * (1.0 + EVIDENCE_HALF_LIFE_FACTOR * reinforcement)
        effective_at = as_of or _now()
        age_days = max(0.0, (effective_at - _aware(claim.valid_from)).total_seconds() / 86400)
        return max(0.0, min(1.0, base * math.exp(-math.log(2) * age_days / half_life)))

    def _score(self, semantic: float, confidence: float, claim: Claim, as_of: datetime) -> float:
        age_days = max(0.0, (as_of - _aware(claim.valid_from)).total_seconds() / 86400)
        recency = math.exp(-age_days / 180)
        evidence = min(1.0, 0.5 + 0.1 * len(claim.evidence))
        risk = 0.2 if claim.sensitive else 0.0
        effective_semantic = semantic
        if (
            self.settings.layered_retrieval_enabled
            and self.settings.layered_episodic_length_neutralized
            and claim.kind == "episodic"
        ):
            # Episodic claims keep the raw verbatim utterance while state-layer claims
            # hold a short canonical string, so a kind-blind cosine score rewards
            # episodic values for length alone. Discount the similarity component by
            # how far the value exceeds the canonical-length band.
            effective_semantic = semantic * _episodic_length_discount(claim.value)
        score = (
            0.55 * effective_semantic + 0.2 * confidence + 0.1 * recency + 0.15 * evidence - risk
        )
        return max(0.0, min(1.0, score))

    def _select_evidence(
        self, facets: list[Facet], max_facets: int, query: str
    ) -> list[Facet]:
        """Greedy set-level selection: cover the query, avoid redundant evidence.

        Pointwise relevance ranks each claim against the query independently, so the
        budget fills with near-duplicates that repeat one aspect while the claims
        that decide the answer sit just outside the cut. Measured on the frozen
        PersonaMem corpus, decisive tokens appear in ~90% of full history but only
        ~25% of the injected top-12, and an oracle that picks the best 12 from the
        same stored state reaches 0.88 where the live path reaches 0.42.

        Each step picks the facet with the highest marginal gain: its own relevance,
        plus the query terms it newly covers, minus overlap with what is already
        selected. Ties fall back to relevance order, so behavior stays deterministic.

        The coverage term is weighted to be commensurate with relevance. Raw
        newly-covered-term count divided by query length is far smaller than a
        typical relevance score, so without the weight a cluster of near-duplicate
        high-relevance claims still wins the whole budget.
        """
        if len(facets) <= max_facets:
            return facets
        query_terms = _content_tokens(query)
        penalty = self.settings.evidence_selection_redundancy_penalty
        remaining = list(facets)
        # Precompute once: token sets are reused across every greedy step.
        tokens = {id(facet): _content_tokens(str(facet.value)) for facet in facets}
        selected: list[Facet] = []
        covered: set[str] = set()
        selected_tokens: set[str] = set()
        while remaining and len(selected) < max_facets:
            best: Facet | None = None
            best_gain = float("-inf")
            for facet in remaining:
                facet_tokens = tokens[id(facet)]
                if not facet_tokens:
                    gain = facet.relevance
                else:
                    new_terms = len((facet_tokens & query_terms) - covered)
                    query_gain = (
                        _COVERAGE_WEIGHT * new_terms / len(query_terms) if query_terms else 0.0
                    )
                    overlap = len(facet_tokens & selected_tokens) / len(facet_tokens)
                    gain = facet.relevance + query_gain - penalty * overlap
                if gain > best_gain:
                    best_gain = gain
                    best = facet
            if best is None:
                break
            selected.append(best)
            best_tokens = tokens[id(best)]
            covered |= best_tokens & query_terms
            selected_tokens |= best_tokens
            remaining.remove(best)
        selected.sort(key=lambda item: item.relevance, reverse=True)
        return selected

    def _apply_facet_budget(
        self, facets: list[Facet], max_facets: int, query: str = ""
    ) -> list[Facet]:
        """Truncate to the injection budget, reserving slots for stable state layers.

        Default behavior is a flat relevance cut. With layered retrieval enabled, the
        state layers named in ``STATE_LAYER_KINDS`` are guaranteed a share of the
        budget so that long verbatim episodic values cannot occupy every slot; both
        groups stay in relevance order and any unused reserve returns to the pool.
        With evidence selection enabled, each group is filled by marginal gain
        against the query rather than by pointwise relevance alone.
        """
        if not self.settings.layered_retrieval_enabled:
            if self.settings.evidence_selection_enabled:
                return self._select_evidence(facets, max_facets, query)
            return facets[:max_facets]
        if len(facets) <= max_facets:
            return facets
        reserve = int(max_facets * self.settings.layered_state_facet_reserve)
        if reserve <= 0:
            if self.settings.evidence_selection_enabled:
                return self._select_evidence(facets, max_facets, query)
            return facets[:max_facets]
        state = [facet for facet in facets if facet.kind in STATE_LAYER_KINDS]
        other = [facet for facet in facets if facet.kind not in STATE_LAYER_KINDS]
        if self.settings.evidence_selection_enabled:
            selected_state = self._select_evidence(state, reserve, query)
            selected_other = self._select_evidence(
                other, max_facets - len(selected_state), query
            )
        else:
            selected_state = state[:reserve]
            selected_other = other[: max_facets - len(selected_state)]
        # Any reserve the state layers did not use falls back to the remaining facets.
        if len(selected_state) + len(selected_other) < max_facets:
            chosen = {id(facet) for facet in (*selected_state, *selected_other)}
            for facet in facets:
                if len(selected_state) + len(selected_other) >= max_facets:
                    break
                if id(facet) not in chosen:
                    selected_other.append(facet)
        selected = [*selected_state, *selected_other]
        selected.sort(key=lambda item: item.relevance, reverse=True)
        return selected

    @staticmethod
    def _later_facet_fits(facets: list[Facet], start: int, line_budget: int) -> bool:
        """Whether any facet from ``start`` onward would fit whole in ``line_budget``.

        Used to decide if skipping an oversized value is safe: if nothing behind it
        fits, skipping would waste the remaining budget instead of filling it.
        """
        for facet in facets[start:]:
            if isinstance(facet.value, dict) and "current" in facet.value:
                continue
            prefix = f"- [{escape(facet.kind, quote=True)}] "
            evidence = ",".join(escape(item, quote=True) for item in facet.evidence_ids)
            suffix = f" (confidence={facet.confidence:.3f}, evidence={evidence})"
            value = escape(str(facet.value), quote=True)
            if len(prefix) + len(value) + len(suffix) <= line_budget:
                return True
        return False

    def _render(self, user_id: str, revision: int, facets: list[Facet], max_tokens: int) -> str:
        opening = f'<user_context user_id="{escape(user_id, quote=True)}" revision="{revision}">'
        closing = "</user_context>"
        lines = [opening]
        char_budget = max_tokens * 4
        skip_oversized = (
            self.settings.layered_retrieval_enabled
            and self.settings.layered_render_skip_oversized
        )
        for position, facet in enumerate(facets):
            # Direction is a suffix on the kind tag rather than words in the value:
            # an English marker like "prefers" collides with option and query wording
            # downstream, shifting lexical scores for reasons unrelated to user state.
            tag = escape(facet.kind, quote=True)
            if facet.stance == "negative":
                tag = f"{tag}:rejected"
            elif facet.stance == "positive":
                tag = f"{tag}:held"
            if (
                isinstance(facet.value, dict)
                and "current" in facet.value
                and "history_oldest_to_newest" in facet.value
            ):
                prefix = f"- [{tag} timeline] "
                raw_value = (
                    f"current={facet.value['current']}; "
                    "history_oldest_to_newest="
                    + json.dumps(facet.value["history_oldest_to_newest"], ensure_ascii=False)
                )
            else:
                prefix = f"- [{tag}] "
                raw_value = str(facet.value)
            evidence = ",".join(escape(item, quote=True) for item in facet.evidence_ids)
            suffix = f" (confidence={facet.confidence:.3f}, evidence={evidence})"
            current_length = len("\n".join(lines))
            line_budget = char_budget - current_length - len(closing) - 2
            if line_budget <= len(prefix) + len(suffix):
                continue

            safe_value = escape(raw_value, quote=True)
            if len(prefix) + len(safe_value) + len(suffix) > line_budget:
                # Skipping an oversized value only helps if something behind it still
                # fits whole; otherwise skipping would forfeit the remaining budget
                # entirely, so fall through and truncate to use it.
                if skip_oversized and self._later_facet_fits(
                    facets, position + 1, line_budget
                ):
                    continue
                value_budget = line_budget - len(prefix) - len(suffix)
                if value_budget <= 1:
                    continue
                low, high = 0, len(raw_value)
                while low < high:
                    middle = (low + high + 1) // 2
                    if len(escape(raw_value[:middle], quote=True)) <= value_budget - 1:
                        low = middle
                    else:
                        high = middle - 1
                safe_value = escape(raw_value[:low], quote=True) + "…"
            lines.append(prefix + safe_value + suffix)
            if len(safe_value) < len(escape(raw_value, quote=True)):
                break

        lines.append(closing)
        return "\n".join(lines)

    def _revision(self, session: Session, user_id: str) -> int:
        row = session.get(UserRevision, user_id)
        return row.revision if row else 0

    def _materialize_user_state(self, session: Session, user_id: str, revision: int) -> None:
        self._materialize_preference_distributions(session, user_id, revision)
        self._materialize_user_token(session, user_id, revision)

    def _materialize_preference_distributions(
        self, session: Session, user_id: str, revision: int
    ) -> None:
        if not self.settings.preference_distribution_enabled:
            return
        session.flush()
        claims = list(
            session.scalars(
                select(Claim)
                .options(selectinload(Claim.evidence))
                .where(Claim.user_id == user_id, Claim.kind == "preference")
                .order_by(Claim.semantic_key, Claim.valid_from, Claim.id)
            ).all()
        )
        grouped: dict[str, list[Claim]] = {}
        for claim in claims:
            grouped.setdefault(claim.semantic_key, []).append(claim)
        confirmed_claim_ids = set(
            session.scalars(
                select(Feedback.claim_id).where(
                    Feedback.user_id == user_id,
                    Feedback.action == "confirm",
                )
            ).all()
        )
        existing = {
            distribution.semantic_key: distribution
            for distribution in session.scalars(
                select(PreferenceDistribution).where(PreferenceDistribution.user_id == user_id)
            ).all()
        }
        retained_keys: set[str] = set()
        for semantic_key, history in grouped.items():
            active = [claim for claim in history if claim.status == "active"]
            if len(active) != 1:
                continue
            current = active[0]
            labels = {
                _normalize_value(claim.value): claim.value
                for claim in history
                if claim.status != "rejected"
            }
            if not labels:
                continue
            alpha = {label: 0.5 for label in labels}
            total_support = 0
            support_event_ids: set[str] = set()
            for claim in history:
                if claim.status == "rejected":
                    continue
                label = _normalize_value(claim.value)
                support = max(1, len(claim.evidence))
                total_support += support
                support_event_ids.update(item.event_id for item in claim.evidence)
                alpha[label] += 0.25 * support
            current_label = _normalize_value(current.value)
            alpha[current_label] += total_support + (
                2.0 if current.source_type == "explicit_feedback" else 1.0
            )
            total_alpha = sum(alpha.values())
            probabilities = {labels[label]: alpha[label] / total_alpha for label in sorted(alpha)}
            strategy_version = (
                "preference-dirichlet-beta-v2"
                if self.settings.preference_distribution_v2_enabled
                else "preference-dirichlet-v1"
            )
            distribution_type = (
                "dirichlet_beta"
                if self.settings.preference_distribution_v2_enabled
                else "dirichlet"
            )
            parameters = {
                "labels": [labels[label] for label in sorted(labels)],
                "alpha": {labels[label]: alpha[label] for label in sorted(alpha)},
                "probabilities": probabilities,
                "mode": current.value,
                "mode_probability": probabilities[current.value],
                "support_event_ids": sorted(support_event_ids),
            }
            if self.settings.preference_distribution_v2_enabled:
                active_support = max(1, len(current.evidence))
                explicitly_supported = (
                    current.source_type == "explicit_feedback" or current.id in confirmed_claim_ids
                )
                explicit_feedback_bonus = 2.0 if explicitly_supported else 0.0
                belief_alpha = active_support + explicit_feedback_bonus
                belief_beta = 1.0
                belief_probability = belief_alpha / (belief_alpha + belief_beta)
                parameters["belief"] = {
                    "distribution_type": "beta",
                    "alpha": belief_alpha,
                    "beta": belief_beta,
                    "probability": belief_probability,
                    "unknown_mass": 1.0 - belief_probability,
                    "evidence_basis": "active_support_with_unit_unknown_prior",
                }
                parameters["actionable_probability"] = (
                    probabilities[current.value] * belief_probability
                )
            canonical = json.dumps(
                {
                    "semantic_key": semantic_key,
                    "distribution_type": distribution_type,
                    "strategy_version": strategy_version,
                    "calibration_status": "uncalibrated",
                    "parameters": parameters,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            content_hash = _sha(canonical)
            distribution = existing.get(semantic_key)
            if distribution is None:
                distribution = PreferenceDistribution(
                    id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"meno:preference-distribution:{user_id}:{semantic_key}",
                        )
                    ),
                    user_id=user_id,
                    semantic_key=semantic_key,
                    distribution_type=distribution_type,
                    strategy_version=strategy_version,
                    calibration_status="uncalibrated",
                    parameters=parameters,
                    state_revision=revision,
                    content_hash=content_hash,
                )
                session.add(distribution)
            else:
                distribution.distribution_type = distribution_type
                distribution.strategy_version = strategy_version
                distribution.calibration_status = "uncalibrated"
                distribution.parameters = parameters
                distribution.state_revision = revision
                distribution.content_hash = content_hash
                distribution.updated_at = _now()
            retained_keys.add(semantic_key)
        for semantic_key, distribution in existing.items():
            if semantic_key not in retained_keys:
                session.delete(distribution)
        session.flush()

    def _materialize_user_token(self, session: Session, user_id: str, revision: int) -> None:
        if not self.settings.user_token_materialization_enabled:
            return
        session.flush()
        claims = list(
            session.scalars(
                select(Claim)
                .options(selectinload(Claim.evidence))
                .where(Claim.user_id == user_id, Claim.status == "active")
                .order_by(Claim.semantic_key, Claim.id)
            ).all()
        )
        consents = list(
            session.scalars(
                select(Consent)
                .where(Consent.user_id == user_id)
                .order_by(Consent.source, Consent.purpose, Consent.granted_at)
            ).all()
        )
        preference_distributions = list(
            session.scalars(
                select(PreferenceDistribution)
                .where(PreferenceDistribution.user_id == user_id)
                .order_by(PreferenceDistribution.semantic_key)
            ).all()
        )
        claims_by_semantic_key = {claim.semantic_key: claim for claim in claims}
        active_state = [
            {
                "claim_id": claim.id,
                "kind": claim.kind,
                "semantic_channel": claim.semantic_channel,
                "semantic_key": claim.semantic_key,
                "routing_slot": claim.routing_slot,
                "value": claim.value,
                "confidence": {
                    "calibrated": claim.confidence,
                    "half_life_days": claim.half_life_days,
                    "support_count": len(claim.evidence),
                },
                "sensitive": bool(claim.sensitive),
                "allowed_purposes": sorted(claim.allowed_purposes),
                "source_type": claim.source_type,
                "valid_from": _aware(claim.valid_from).isoformat(),
                "valid_to": (_aware(claim.valid_to).isoformat() if claim.valid_to else None),
                "evidence_ids": sorted(item.event_id for item in claim.evidence),
            }
            for claim in claims
        ]
        consent_state = sorted(
            (
                {
                    "source": consent.source,
                    "purpose": consent.purpose,
                    "status": consent.status,
                    "allowed_operations": sorted(consent.allowed_operations),
                    "data_categories": sorted(consent.data_categories),
                    "sensitive_data": bool(consent.sensitive_data),
                    "expires_at": (
                        _aware(consent.expires_at).isoformat() if consent.expires_at else None
                    ),
                }
                for consent in consents
            ),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
        clarification_opportunities = self._build_clarification_opportunities(
            user_id, claims_by_semantic_key, preference_distributions
        )
        payload = {
            "user_id": user_id,
            "state_revision": revision,
            "active_state": active_state,
            "consent_state": consent_state,
            "preference_distributions": [
                {
                    "semantic_key": distribution.semantic_key,
                    "distribution_type": distribution.distribution_type,
                    "strategy_version": distribution.strategy_version,
                    "calibration_status": distribution.calibration_status,
                    "parameters": distribution.parameters,
                    "content_hash": distribution.content_hash,
                }
                for distribution in preference_distributions
            ],
            "clarification_opportunities": clarification_opportunities,
            "uncertainty": {
                "low_confidence_claim_ids": sorted(
                    claim.id for claim in claims if claim.confidence < 0.55
                ),
                "unsupported_claim_ids": sorted(claim.id for claim in claims if not claim.evidence),
                "ambiguous_claim_ids": sorted(
                    opportunity["claim_id"] for opportunity in clarification_opportunities
                ),
            },
        }
        snapshot_hash = content_hash(payload)
        snapshot_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                (
                    f"meno:user-token:{user_id}:{revision}:"
                    f"{self.settings.policy_version}:{self.settings.extractor_version}"
                ),
            )
        )
        schema_version = "1.0.0"
        existing = self._user_snapshot_record(session, user_id, revision=revision)
        if existing is not None:
            existing_payload = self._read_user_snapshot_payload(session, existing)
            if (
                existing.id != snapshot_id
                or existing.content_hash != snapshot_hash
                or existing.schema_version != schema_version
                or existing.policy_version != self.settings.policy_version
                or existing.extractor_version != self.settings.extractor_version
                or canonical_json(existing_payload) != canonical_json(payload)
            ):
                raise ValueError("user token revision is not deterministic")
            return

        interval = self.settings.user_token_snapshot_base_interval
        if interval <= 0:
            raise ValueError("user token snapshot base interval must be positive")
        previous = self._user_snapshot_record(
            session, user_id, before_revision=revision
        )
        is_base = previous is None
        base_revision = revision
        delta_payload = None
        if previous is not None:
            if revision <= previous.state_revision:
                raise ValueError("user token revisions must increase")
            previous_base_revision = (
                previous.state_revision
                if isinstance(previous, UserTokenSnapshot)
                else previous.base_revision
            )
            base_revision = previous_base_revision
            same_versions = (
                previous.schema_version == schema_version
                and previous.policy_version == self.settings.policy_version
                and previous.extractor_version == self.settings.extractor_version
            )
            is_base = not same_versions or revision - base_revision >= interval
            if not is_base:
                previous_payload = self._read_user_snapshot_payload(session, previous)
                delta_payload = make_delta(previous_payload, payload)
                if delta_payload is None:
                    log.warning(
                        "user token snapshot: delta not representable at rev=%s, storing full base",
                        revision,
                    )
                is_base = delta_payload is None

        if is_base:
            session.add(
                UserTokenSnapshot(
                    id=snapshot_id,
                    user_id=user_id,
                    state_revision=revision,
                    schema_version=schema_version,
                    policy_version=self.settings.policy_version,
                    extractor_version=self.settings.extractor_version,
                    payload=payload,
                    content_hash=snapshot_hash,
                )
            )
        else:
            assert previous is not None and delta_payload is not None
            session.add(
                UserTokenSnapshotDelta(
                    id=snapshot_id,
                    user_id=user_id,
                    state_revision=revision,
                    base_revision=base_revision,
                    previous_revision=previous.state_revision,
                    delta=delta_payload,
                    content_hash=snapshot_hash,
                    schema_version=schema_version,
                    policy_version=self.settings.policy_version,
                    extractor_version=self.settings.extractor_version,
                )
            )
        if is_base:
            log.info(
                "user token snapshot: base rev=%s claims=%s",
                revision,
                len(payload.get("active_state") or []),
            )
        else:
            log.info(
                "user token snapshot: delta rev=%s base_rev=%s",
                revision,
                base_revision,
            )
        session.flush()

    @staticmethod
    def _is_ambiguous_preference(value: str) -> bool:
        folded = value.casefold()
        return any(
            marker in folded
            for marker in (
                "maybe",
                "might",
                "perhaps",
                "not sure",
                "sometimes",
                "可能",
                "也许",
                "不确定",
                "有时候",
            )
        )

    def _build_clarification_opportunities(
        self,
        user_id: str,
        claims_by_semantic_key: dict[str, Claim],
        distributions: list[PreferenceDistribution],
    ) -> list[dict[str, Any]]:
        if not self.settings.clarification_opportunities_enabled:
            return []
        opportunities: list[dict[str, Any]] = []
        for distribution in distributions:
            claim = claims_by_semantic_key.get(distribution.semantic_key)
            if claim is None or claim.source_type == "explicit_feedback":
                continue
            parameters = distribution.parameters
            reason: str | None = None
            if self._is_ambiguous_preference(claim.value) and claim.confidence < 0.98:
                reason = "ambiguous_expression"
            elif claim.confidence < 0.75:
                reason = "low_confidence"
            elif (
                len(parameters.get("labels", [])) > 1
                and float(parameters.get("mode_probability", 1.0)) < 0.65
            ):
                reason = "competing_evidence"
            if reason is None:
                continue
            slot = claim.routing_slot or claim.semantic_channel
            opportunity_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    (
                        f"meno:clarification:{user_id}:{claim.semantic_key}:"
                        f"{reason}:{distribution.content_hash}"
                    ),
                )
            )
            opportunities.append(
                {
                    "opportunity_id": opportunity_id,
                    "claim_id": claim.id,
                    "semantic_key": claim.semantic_key,
                    "routing_slot": claim.routing_slot,
                    "value": claim.value,
                    "reason": reason,
                    "question": (
                        f"You expressed an uncertain preference about {slot}. "
                        f'Should I remember "{claim.value}" as your current default?'
                    ),
                    "evidence_ids": sorted(item.event_id for item in claim.evidence),
                }
            )
        return opportunities

    def _relevant_clarification_opportunities(
        self, session: Session, request: RetrieveRequest
    ) -> list[ClarificationOpportunity]:
        if not self.settings.clarification_opportunities_enabled:
            return []
        snapshot = self._user_snapshot_record(session, request.user_id)
        if snapshot is None:
            return []
        payload = self._read_user_snapshot_payload(session, snapshot)
        result: list[ClarificationOpportunity] = []
        context = f"{request.context.task_type or ''} {request.context.query}"
        query_slot = preference_slot(context.replace("_", " "))
        for item in payload.get("clarification_opportunities", []):
            if item.get("routing_slot") and item["routing_slot"] != query_slot:
                continue
            result.append(ClarificationOpportunity.model_validate(item))
        return result

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

    def _audit_spill_file(self) -> Path:
        return self._audit_spill_path

    def _spill_audit_event(self, entry: dict[str, Any]) -> bool:
        """Persist one overflowed audit event. Audit events are never dropped."""
        path = self._audit_spill_file()
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            log.exception("audit spill write failed; event stays in the memory buffer")
            return False
        return True

    def _read_spilled_audit_events(self) -> list[dict[str, Any]] | None:
        path = self._audit_spill_file()
        if not path.exists():
            return []
        entries: list[dict[str, Any]] = []
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        log.error("audit spill is not valid JSON; retained for recovery")
                        return None
        except OSError:
            log.exception("audit spill read failed")
            return None
        return entries

    def _clear_audit_spill(self) -> None:
        try:
            self._audit_spill_file().unlink(missing_ok=True)
        except OSError:
            log.warning("audit spill cleanup failed for %s", self._audit_spill_file())

    def _deletion_watermark(self, session: Session, user_id: str) -> datetime | None:
        """Cutoff for client replays: newest completed full-user deletion."""
        value = session.scalar(
            select(DeletionJob.completed_at)
            .where(
                DeletionJob.subject_hash == _sha(user_id),
                DeletionJob.scope == "all",
                DeletionJob.status == "completed",
                DeletionJob.completed_at.is_not(None),
            )
            .order_by(DeletionJob.completed_at.desc())
            .limit(1)
        )
        return _aware(value) if value is not None else None

    def _idempotent_replay(
        self,
        session: Session,
        *,
        endpoint: str,
        key: str,
        fingerprint: str,
    ) -> dict[str, Any] | None:
        if not key:
            return None
        row = session.scalar(
            select(IdempotencyRecord).where(
                IdempotencyRecord.endpoint == endpoint,
                IdempotencyRecord.key_hash == _sha(key),
            )
        )
        if row is None:
            return None
        if row.request_fingerprint != fingerprint:
            raise ValueError("Idempotency-Key reused with a different payload")
        payload = json.loads(row.response_json)
        payload["idempotent_replay"] = True
        return payload

    def _record_idempotency(
        self,
        session: Session,
        *,
        endpoint: str,
        key: str,
        fingerprint: str,
        response: dict[str, Any],
    ) -> None:
        if not key:
            return
        session.add(
            IdempotencyRecord(
                id=_id(),
                endpoint=endpoint,
                key_hash=_sha(key),
                request_fingerprint=fingerprint,
                response_json=json.dumps(response, ensure_ascii=False, default=str),
            )
        )

    def _buffer_audit(self, **entry: Any) -> None:
        with self._audit_lock:
            if len(self._audit_buffer) >= self.settings.audit_buffer_max:
                # Integrity first: the oldest buffered event goes to disk, never away.
                spilled = self._audit_buffer.popleft()
                if not self._spill_audit_event(spilled):
                    self._audit_buffer.appendleft(spilled)
            self._audit_buffer.append(entry)
            if len(self._audit_buffer) > self.settings.audit_buffer_max:
                log.warning("audit buffer full; retained overflow in memory after spill failure")

    def flush_audit_buffer(self) -> int:
        # ponytail: one process lock also covers spill I/O; use a durable queue
        # if multiple processes ever need to share an overflow file.
        with self._audit_lock:
            spilled = self._read_spilled_audit_events()
            if spilled is None:
                return 0
            buffered = list(self._audit_buffer)
            batch = spilled + buffered  # spilled events are older; keep append order
            if not batch:
                return 0
            self._audit_buffer.clear()
            try:
                with self._audit_chain_lock, self.session_factory.begin() as session:
                    for entry in batch:
                        self._audit(session, **entry)
            except Exception:  # buffered audits must survive transient DB failures
                log.exception("audit flush failed; rebuffering %d events", len(batch))
                # Spilled entries stay on disk; restore only the in-memory entries.
                self._audit_buffer.extendleft(reversed(buffered))
                return 0
            self._clear_audit_spill()
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
        # Guard for callers that did not enter through an audit-writing
        # transaction wrapper (see __init__): the head read and the insert must
        # not interleave with another thread's.
        with self._audit_chain_lock:
            previous = session.scalar(
                select(AuditEvent)
                .order_by(AuditEvent.created_at.desc(), AuditEvent.id.desc())
                .limit(1)
            )
            prev_hash = previous.current_hash if previous else None
            created_at = _now()
            if previous is not None and created_at <= _aware(previous.created_at):
                created_at = _aware(previous.created_at) + timedelta(microseconds=1)
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
                    created_at=created_at,
                )
            )
            session.flush()
