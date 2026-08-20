from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Event(Base):
    __tablename__ = "meno_events"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(256), index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    source_type: Mapped[str] = mapped_column(String(64), index=True)
    source_profile: Mapped[str | None] = mapped_column(String(128))
    session_id: Mapped[str | None] = mapped_column(String(256), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(71))
    consent_scope: Mapped[list[str]] = mapped_column(JSON)
    event_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Claim(Base):
    __tablename__ = "meno_claims"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    derivation_key: Mapped[str | None] = mapped_column(String(128), unique=True, index=True)
    user_id: Mapped[str] = mapped_column(String(256), index=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    origin_role: Mapped[str] = mapped_column(String(16), default="user", index=True)
    semantic_channel: Mapped[str] = mapped_column(String(256), index=True)
    value: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), index=True, default="active")
    confidence: Mapped[float] = mapped_column(Float, default=0.7)
    half_life_days: Mapped[float | None] = mapped_column(Float)
    sensitive: Mapped[bool] = mapped_column(Boolean, default=False)
    allowed_purposes: Mapped[list[str]] = mapped_column(JSON)
    source_type: Mapped[str] = mapped_column(String(64), index=True)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    supersedes_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("meno_claims.id"))
    extractor_version: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    evidence: Mapped[list[ClaimEvidence]] = relationship(
        cascade="all, delete-orphan", back_populates="claim"
    )


class ClaimEvidence(Base):
    __tablename__ = "meno_claim_evidence"
    __table_args__ = (UniqueConstraint("claim_id", "event_id", name="uq_claim_event"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("meno_claims.id", ondelete="CASCADE"), index=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("meno_events.id", ondelete="CASCADE"), index=True)
    relation: Mapped[str] = mapped_column(String(64), default="supports")
    claim: Mapped[Claim] = relationship(back_populates="evidence")


class Consent(Base):
    __tablename__ = "meno_consents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(256), index=True)
    source: Mapped[str] = mapped_column(String(64), index=True)
    purpose: Mapped[str] = mapped_column(String(64), index=True)
    allowed_operations: Mapped[list[str]] = mapped_column(JSON)
    data_categories: Mapped[list[str]] = mapped_column(JSON)
    sensitive_data: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(16), default="active")
    granted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class UserRevision(Base):
    __tablename__ = "meno_user_revisions"

    user_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Outbox(Base):
    __tablename__ = "meno_outbox"
    __table_args__ = (
        UniqueConstraint("event_id", "processor_version", name="uq_outbox_processor"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("meno_events.id", ondelete="CASCADE"), index=True)
    processor_version: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditEvent(Base):
    __tablename__ = "meno_audit_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_name: Mapped[str] = mapped_column(String(128), index=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    user_hash: Mapped[str] = mapped_column(String(71), index=True)
    claim_id: Mapped[str | None] = mapped_column(String(64), index=True)
    action: Mapped[str] = mapped_column(String(64))
    purpose: Mapped[str | None] = mapped_column(String(64))
    decision: Mapped[dict[str, Any]] = mapped_column(JSON)
    state_revision: Mapped[int] = mapped_column(Integer)
    source_event_ids: Mapped[list[str]] = mapped_column(JSON)
    prev_hash: Mapped[str | None] = mapped_column(String(71))
    current_hash: Mapped[str] = mapped_column(String(71), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Feedback(Base):
    __tablename__ = "meno_feedback"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(256), index=True)
    claim_id: Mapped[str] = mapped_column(String(64), index=True)
    action: Mapped[str] = mapped_column(String(16))
    correction: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class DeletionJob(Base):
    __tablename__ = "meno_deletion_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    subject_hash: Mapped[str] = mapped_column(String(71), index=True)
    scope: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="pending")
    receipt_hash: Mapped[str | None] = mapped_column(String(71))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Entity(Base):
    __tablename__ = "meno_entities"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(256), index=True)
    canonical_name: Mapped[str] = mapped_column(String(512), index=True)
    entity_type: Mapped[str] = mapped_column(String(64), default="unknown")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ClaimEntity(Base):
    __tablename__ = "meno_claim_entities"
    __table_args__ = (UniqueConstraint("claim_id", "entity_id", name="uq_claim_entity"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("meno_claims.id", ondelete="CASCADE"), index=True)
    entity_id: Mapped[str] = mapped_column(ForeignKey("meno_entities.id", ondelete="CASCADE"), index=True)
    relation: Mapped[str] = mapped_column(String(64), default="mentions")


class ClaimEdge(Base):
    __tablename__ = "meno_claim_edges"
    __table_args__ = (
        UniqueConstraint("source_claim_id", "target_claim_id", "relation_type", name="uq_claim_edge"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(256), index=True)
    source_claim_id: Mapped[str] = mapped_column(ForeignKey("meno_claims.id", ondelete="CASCADE"), index=True)
    target_claim_id: Mapped[str] = mapped_column(ForeignKey("meno_claims.id", ondelete="CASCADE"), index=True)
    relation_type: Mapped[str] = mapped_column(String(64), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    evidence_event_id: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16), default="active")
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


def make_session_factory(database_url: str):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)
    if database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False), engine
