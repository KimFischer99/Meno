from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EventSource(StrictModel):
    type: str = Field(min_length=1, max_length=64)
    profile: str | None = Field(default=None, max_length=128)
    session_id: str | None = Field(default=None, max_length=256)


class EventContent(StrictModel):
    role: Literal["user", "assistant", "tool", "system"]
    text: str = Field(min_length=1, max_length=200_000)


class IngestRequest(StrictModel):
    user_id: str = Field(min_length=1, max_length=256)
    event_id: str | None = Field(default=None, max_length=128)
    occurred_at: datetime = Field(default_factory=utcnow)
    source: EventSource
    content: EventContent
    consent_scope: list[str] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestBatchRequest(StrictModel):
    events: list[IngestRequest] = Field(min_length=1, max_length=256)


class RetrieveContext(StrictModel):
    query: str = Field(min_length=1, max_length=16_000)
    as_of: datetime | None = None
    task_type: str | None = None
    platform: str | None = None
    workspace: str | None = None


class RetrieveConstraints(StrictModel):
    max_facets: int = Field(default=8, ge=1, le=32)
    max_rendered_tokens: int = Field(default=800, ge=64, le=4096)
    min_confidence: float = Field(default=0.55, ge=0, le=1)
    allow_sensitive: bool = False


class RetrieveRequest(StrictModel):
    user_id: str = Field(min_length=1, max_length=256)
    session_id: str | None = None
    purpose: Literal[
        "response_personalization", "task_planning", "proactive_suggestion"
    ]
    context: RetrieveContext
    constraints: RetrieveConstraints = Field(default_factory=RetrieveConstraints)


class Facet(StrictModel):
    claim_id: str
    kind: Literal["fact", "episodic", "pattern", "trait", "preference", "state", "community"]
    value: Any
    relevance: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str]
    why_selected: list[str]


class RetrieveResponse(StrictModel):
    trace_id: str
    user_id: str
    state_revision: int
    token_revision_id: str
    facets: list[Facet]
    rendered_context: str
    degraded: bool
    policy_version: str


class FeedbackRequest(StrictModel):
    user_id: str = Field(min_length=1, max_length=256)
    claim_id: str
    action: Literal["confirm", "reject", "correct"]
    correction: str | None = Field(default=None, max_length=16_000)


class ConsentRequest(StrictModel):
    user_id: str = Field(min_length=1, max_length=256)
    source: str = Field(min_length=1, max_length=64)
    purpose: str = Field(min_length=1, max_length=64)
    allowed_operations: list[str] = Field(min_length=1)
    data_categories: list[str] = Field(default_factory=list)
    sensitive_data: bool = False
    status: Literal["active", "revoked"] = "active"
    expires_at: datetime | None = None


class DeletionRequest(StrictModel):
    user_id: str = Field(min_length=1, max_length=256)
    scope: Literal["all", "source", "claim"] = "all"
    source: str | None = None
    claim_id: str | None = None


class PredictCandidate(StrictModel):
    id: str
    description: str = Field(min_length=1, max_length=4000)


class PredictRequest(StrictModel):
    user_id: str
    context: RetrieveContext
    candidates: list[PredictCandidate] = Field(min_length=1, max_length=32)
