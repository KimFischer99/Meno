from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select, update

from benchmarks.verify_invariants import audit_chain_summary
from meno.api import build_service
from meno.config import Settings
from meno.db import AuditEvent, Claim, ClaimEvidence, ProjectionOutbox
from meno.schemas import ConsentRequest, FeedbackRequest, IngestRequest, RetrieveRequest
from meno.semantic_policy import FrozenSemanticRoutingPolicy
from meno.semantic_router import (
    ENSEMBLE_STRATEGY_VERSION,
    CandidateEnvelope,
    DimensionScore,
    PolicyRouter,
    ReferenceEnvelope,
    RouterDecision,
)
from meno.service import _semantic_key
from tests.fakes import TestEmbedder


class StaticPolicy:
    config_sha256 = "c" * 64
    prototype_sha256 = "d" * 64

    def __init__(self, *, unavailable: bool = False) -> None:
        self.unavailable = unavailable
        self.score_calls: list[list[CandidateEnvelope]] = []
        self.router = PolicyRouter(strategy_version=ENSEMBLE_STRATEGY_VERSION)

    def score_many(
        self, candidates: list[CandidateEnvelope]
    ) -> list[DimensionScore | None]:
        self.score_calls.append(list(candidates))
        if self.unavailable:
            return [None] * len(candidates)
        return [
            DimensionScore(
                dimension="answer_style",
                score=0.8,
                second_dimension="beverage",
                second_score=0.5,
                margin=0.3,
                strategy_version=ENSEMBLE_STRATEGY_VERSION,
            )
            for _candidate in candidates
        ]

    def decide(
        self,
        candidate: CandidateEnvelope,
        references: list[ReferenceEnvelope],
        score: DimensionScore | None,
    ) -> RouterDecision:
        return self.router.decide(candidate, references, score, 0.475, 0.005)


def make_service(
    tmp_path: Path,
    policy: StaticPolicy | None = None,
    *,
    history: bool = False,
):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'phase-b.sqlite3'}",
        vector_mode="memory",
        embedding_dimension=256,
        extractor_version="meno-extractor-2.1.0",
    )
    service = build_service(settings, embedder=TestEmbedder(256))
    if policy is not None:
        service.settings = replace(
            settings,
            semantic_routing_enabled=True,
            preference_history_retrieval_enabled=history,
        )
        service.semantic_policy = policy  # type: ignore[assignment]
    return service


def ingest(
    service,
    event_id: str,
    text: str,
    *,
    user_id: str = "user-a",
    source: str = "hermes_turn",
) -> None:
    service.ingest(
        IngestRequest(
            event_id=event_id,
            user_id=user_id,
            source={"type": source},
            content={"role": "user", "text": text},
            consent_scope=["personalization"],
        ),
        f"key-{event_id}",
    )


def active_claims(service, user_id: str = "user-a") -> list[Claim]:
    with service.session_factory() as session:
        return session.scalars(
            select(Claim)
            .where(Claim.user_id == user_id, Claim.status == "active")
            .order_by(Claim.created_at, Claim.id)
        ).all()


def routing_audits(service) -> list[AuditEvent]:
    with service.session_factory() as session:
        return session.scalars(
            select(AuditEvent).where(
                AuditEvent.event_name == "meno.claim.semantic_routing"
            )
        ).all()


def audit_rows(service) -> list[dict[str, object]]:
    with service.session_factory() as session:
        rows = session.scalars(
            select(AuditEvent).order_by(AuditEvent.created_at, AuditEvent.id)
        ).all()
        return [
            {
                "event_name": row.event_name,
                "trace_id": row.trace_id,
                "user_hash": row.user_hash,
                "claim_id": row.claim_id,
                "action": row.action,
                "purpose": row.purpose,
                "decision": row.decision,
                "state_revision": row.state_revision,
                "source_event_ids": row.source_event_ids,
                "prev_hash": row.prev_hash,
                "current_hash": row.current_hash,
            }
            for row in rows
        ]


def test_default_off_preserves_canonical_value_keys(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    try:
        ingest(service, "e1", "I prefer point-first prose")
        ingest(service, "e2", "I prefer leading with the takeaway")
        assert service.process_outbox() == 2
        claims = active_claims(service)
        assert len(claims) == 2
        assert all(claim.routing_basis == "none" for claim in claims)
        assert routing_audits(service) == []
    finally:
        service.close()


def test_deterministic_slot_bypasses_semantic_policy(tmp_path: Path) -> None:
    policy = StaticPolicy()
    service = make_service(tmp_path, policy)
    try:
        ingest(service, "e1", "I prefer tea")
        assert service.process_outbox() == 1
        claim = active_claims(service)[0]
        assert claim.routing_slot == "beverage"
        assert claim.routing_basis == "deterministic_slot"
        assert claim.router_version is None
        assert policy.score_calls == []
        assert routing_audits(service) == []
    finally:
        service.close()


def test_semantic_new_then_reuse_persists_provenance_and_audit(tmp_path: Path) -> None:
    policy = StaticPolicy()
    service = make_service(tmp_path, policy)
    try:
        ingest(service, "e1", "I prefer point-first prose")
        assert service.process_outbox() == 1
        first = active_claims(service)[0]
        first_key = first.semantic_key
        assert first_key == _semantic_key(
            "user-a",
            "preference",
            "preference.explicit",
            "point-first prose",
            "answer_style",
        )
        assert first.routing_slot == "answer_style"
        assert first.routing_basis == "semantic_router"
        assert first.router_version is not None

        ingest(service, "e2", "I prefer leading with the takeaway")
        assert service.process_outbox() == 1
        active = active_claims(service)
        assert len(active) == 1
        assert active[0].semantic_key == first_key
        assert active[0].routing_slot == "answer_style"
        assert active[0].routing_basis == "semantic_router"
        audits = routing_audits(service)
        assert [audit.decision["action"] for audit in audits] == [
            "new_key",
            "reuse_key",
        ]
        assert all("point-first" not in str(audit.decision) for audit in audits)
        assert audit_chain_summary(audit_rows(service))["passed"] is True
    finally:
        service.close()


def test_preference_history_enriches_current_facet_without_reprojecting(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path, StaticPolicy(), history=True)
    try:
        ingest(service, "e1", "I prefer point-first prose")
        ingest(service, "e2", "I prefer leading with the takeaway")
        assert service.process_outbox() == 2
        store = service.vector_store
        assert len(store._points) == 1  # type: ignore[attr-defined]
        with service.session_factory() as session:
            projections = session.scalars(select(ProjectionOutbox)).all()
            assert len(projections) == 3
            assert {projection.status for projection in projections} == {"processed"}

        response = service.retrieve(
            RetrieveRequest(
                user_id="user-a",
                purpose="response_personalization",
                context={"query": "How have my response preferences changed?"},
            )
        )
        timeline = next(facet for facet in response.facets if isinstance(facet.value, dict))
        assert timeline.value == {
            "current": "leading with the takeaway",
            "history_oldest_to_newest": ["point-first prose"],
        }
        assert timeline.evidence_ids == ["e2", "e1"]
        assert "bounded supersedes history" in timeline.why_selected
        assert "[preference:held timeline]" in response.rendered_context
        assert "history_oldest_to_newest" in response.rendered_context

        assert service.flush_audit_buffer() == len(response.facets)
        with service.session_factory() as session:
            audit = session.scalars(
                select(AuditEvent)
                .where(AuditEvent.event_name == "meno.retrieve.facet_selected")
                .order_by(AuditEvent.created_at.desc(), AuditEvent.id.desc())
            ).first()
            assert audit is not None
            assert audit.decision["history_count"] == 1
            assert "point-first prose" not in str(audit.decision)
    finally:
        service.close()


def test_preference_history_default_off_and_sensitive_history_stays_hidden(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path, StaticPolicy())
    try:
        ingest(service, "e1", "I prefer point-first prose")
        ingest(service, "e2", "I prefer leading with the takeaway")
        assert service.process_outbox() == 2
        response = service.retrieve(
            RetrieveRequest(
                user_id="user-a",
                purpose="response_personalization",
                context={"query": "response preferences"},
            )
        )
        assert all(not isinstance(facet.value, dict) for facet in response.facets)

        service.settings = replace(
            service.settings, preference_history_retrieval_enabled=True
        )
        with service.session_factory.begin() as session:
            historical = session.scalar(
                select(Claim).where(Claim.status == "superseded")
            )
            assert historical is not None
            historical.sensitive = True
        response = service.retrieve(
            RetrieveRequest(
                user_id="user-a",
                purpose="response_personalization",
                context={"query": "response preferences"},
            )
        )
        assert all(not isinstance(facet.value, dict) for facet in response.facets)
        assert "point-first prose" not in response.rendered_context
    finally:
        service.close()


def test_preference_history_is_not_injected_for_explicit_as_of_queries(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path, StaticPolicy(), history=True)
    try:
        ingest(service, "e1", "I prefer point-first prose")
        ingest(service, "e2", "I prefer leading with the takeaway")
        assert service.process_outbox() == 2

        response = service.retrieve(
            RetrieveRequest(
                user_id="user-a",
                purpose="response_personalization",
                context={
                    "query": "response preferences",
                    "as_of": datetime.now(UTC),
                },
            )
        )
        assert all(not isinstance(facet.value, dict) for facet in response.facets)
        assert "point-first prose" not in response.rendered_context
    finally:
        service.close()


def test_revoked_historical_source_is_not_injected(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path, StaticPolicy(), history=True)
    try:
        ingest(
            service,
            "e1",
            "I prefer point-first prose",
            source="obsidian",
        )
        ingest(service, "e2", "I prefer leading with the takeaway")
        assert service.process_outbox() == 2
        service.set_consent(
            ConsentRequest(
                user_id="user-a",
                source="obsidian",
                purpose="response_personalization",
                allowed_operations=["ingest"],
                status="revoked",
            )
        )

        response = service.retrieve(
            RetrieveRequest(
                user_id="user-a",
                purpose="response_personalization",
                context={"query": "response preferences"},
            )
        )
        assert response.facets
        assert all(not isinstance(facet.value, dict) for facet in response.facets)
        assert "leading with the takeaway" in response.rendered_context
        assert "point-first prose" not in response.rendered_context
    finally:
        service.close()


def test_semantic_new_key_merges_with_later_deterministic_slot(tmp_path: Path) -> None:
    policy = StaticPolicy()
    service = make_service(tmp_path, policy)
    try:
        ingest(service, "e1", "I prefer point-first prose")
        assert service.process_outbox() == 1
        semantic_key = active_claims(service)[0].semantic_key
        ingest(service, "e2", "I prefer concise answers")
        assert service.process_outbox() == 1
        active = active_claims(service)
        assert len(active) == 1
        assert active[0].semantic_key == semantic_key
        assert active[0].routing_basis == "deterministic_slot"
    finally:
        service.close()


def test_migration_preserves_semantic_router_key_and_provenance(tmp_path: Path) -> None:
    service = make_service(tmp_path, StaticPolicy())
    try:
        ingest(service, "e1", "I prefer point-first prose")
        assert service.process_outbox() == 1
        before = active_claims(service)[0]
        before_key = before.semantic_key
        report = service.migrate()
        after = active_claims(service)[0]
        assert report["routing_backfilled"] == 0
        assert after.semantic_key == before_key
        assert after.routing_slot == "answer_style"
        assert after.routing_basis == "semantic_router"
    finally:
        service.close()


def test_semantic_routing_is_scoped_per_user(tmp_path: Path) -> None:
    policy = StaticPolicy()
    service = make_service(tmp_path, policy)
    try:
        ingest(service, "e1", "I prefer point-first prose", user_id="user-a")
        ingest(service, "e2", "I prefer leading with the takeaway", user_id="user-b")
        assert service.process_outbox() == 2
        user_a = active_claims(service, "user-a")
        user_b = active_claims(service, "user-b")
        assert len(user_a) == len(user_b) == 1
        assert user_a[0].semantic_key != user_b[0].semantic_key
        assert all(
            claim.routing_slot == "answer_style" for claim in [*user_a, *user_b]
        )
    finally:
        service.close()


def test_sensitive_candidate_never_reaches_semantic_provider(tmp_path: Path) -> None:
    policy = StaticPolicy()
    service = make_service(tmp_path, policy)
    try:
        ingest(service, "e1", "I prefer not to discuss my diagnosis")
        assert service.process_outbox() == 1
        claim = active_claims(service)[0]
        assert claim.sensitive is True
        assert claim.routing_basis == "none"
        assert policy.score_calls == []
        assert routing_audits(service) == []
    finally:
        service.close()


def test_reprocessing_same_event_does_not_duplicate_routing_audit(tmp_path: Path) -> None:
    policy = StaticPolicy()
    service = make_service(tmp_path, policy)
    try:
        ingest(service, "e1", "I prefer point-first prose")
        assert service.process_outbox() == 1
        ingest(service, "e1", "I prefer point-first prose")
        assert service.process_outbox() == 0
        assert len(active_claims(service)) == 1
        assert len(routing_audits(service)) == 1
    finally:
        service.close()


def test_provider_unavailable_fails_closed_without_outbox_retry(tmp_path: Path) -> None:
    policy = StaticPolicy(unavailable=True)
    service = make_service(tmp_path, policy)
    try:
        ingest(service, "e1", "I prefer point-first prose")
        assert service.process_outbox() == 1
        claim = active_claims(service)[0]
        assert claim.routing_slot is None
        assert claim.routing_basis == "none"
        audit = routing_audits(service)[0]
        assert audit.decision["reason"] == "provider_unavailable"
        with service.session_factory() as session:
            from meno.db import Outbox

            row = session.scalar(select(Outbox).where(Outbox.event_id == "e1"))
            assert row.status == "processed"
            assert row.attempts == 0
    finally:
        service.close()


@pytest.mark.parametrize("failure", ["raise", "wrong_length", "decide"])
def test_semantic_policy_contract_failures_do_not_retry_outbox(
    tmp_path: Path, failure: str
) -> None:
    class BrokenPolicy(StaticPolicy):
        def score_many(self, candidates):
            if failure == "raise":
                raise ValueError("malformed provider response")
            if failure == "wrong_length":
                return []
            return super().score_many(candidates)

        def decide(self, candidate, references, score):
            if failure == "decide":
                raise RuntimeError("invalid strategy state")
            return super().decide(candidate, references, score)

    service = make_service(tmp_path, BrokenPolicy())
    try:
        ingest(service, "e1", "I prefer point-first prose")
        assert service.process_outbox() == 1
        assert len(active_claims(service)) == 1
        audit = routing_audits(service)[0]
        assert audit.decision["reason"] == "provider_unavailable"
        with service.session_factory() as session:
            from meno.db import Outbox

            row = session.scalar(select(Outbox).where(Outbox.event_id == "e1"))
            assert row.status == "processed"
            assert row.attempts == 0
    finally:
        service.close()


def test_injected_policy_cannot_bypass_disabled_kill_switch(tmp_path: Path) -> None:
    policy = StaticPolicy()
    service = make_service(tmp_path)
    service.semantic_policy = policy  # type: ignore[assignment]
    try:
        ingest(service, "e1", "I prefer point-first prose")
        assert service.process_outbox() == 1
        assert active_claims(service)[0].routing_basis == "none"
        assert policy.score_calls == []
        assert routing_audits(service) == []
    finally:
        service.close()


def test_ambiguous_semantic_references_reject_without_new_claim(tmp_path: Path) -> None:
    policy = StaticPolicy()
    service = make_service(tmp_path, policy)
    try:
        with service.session_factory.begin() as session:
            for index in range(2):
                session.add(
                    Claim(
                        id=f"reference-{index}",
                        derivation_key=f"reference-dk-{index}",
                        user_id="user-a",
                        kind="preference",
                        origin_role="user",
                        semantic_channel="preference.explicit",
                        value=f"historic wording {index}",
                        status="active",
                        confidence=0.9,
                        half_life_days=180,
                        sensitive=False,
                        allowed_purposes=["personalization"],
                        source_type="hermes_turn",
                        valid_from=datetime.now(UTC),
                        extractor_version="meno-extractor-2.0.0",
                        semantic_key=f"sha256:reference-{index}",
                        routing_slot="answer_style",
                        routing_basis="semantic_router",
                        router_version="router-v1",
                    )
                )
        ingest(service, "e1", "I prefer point-first prose")
        with service.session_factory.begin() as session:
            for index in range(2):
                session.add(
                    ClaimEvidence(claim_id=f"reference-{index}", event_id="e1")
                )
        assert service.process_outbox() == 1
        assert len(active_claims(service)) == 2
        audit = routing_audits(service)[0]
        assert audit.decision["action"] == "reject"
        assert audit.decision["reason"] == "ambiguous_semantic_key"
    finally:
        service.close()


def test_feedback_correction_without_slot_clears_semantic_provenance(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path, StaticPolicy())
    try:
        ingest(service, "e1", "I prefer point-first prose")
        assert service.process_outbox() == 1
        routed = active_claims(service)[0]
        result = service.feedback(
            FeedbackRequest(
                user_id="user-a",
                claim_id=routed.id,
                action="correct",
                correction="I prefer gently phrased guidance",
            )
        )
        with service.session_factory() as session:
            replacement = session.get(Claim, result["claim_id"])
            assert replacement.routing_slot is None
            assert replacement.routing_basis == "none"
            assert replacement.router_version is None
    finally:
        service.close()


def test_feedback_projection_failure_is_durable_and_retryable(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    try:
        ingest(service, "e1", "I prefer tea")
        assert service.process_outbox() == 1
        claim = active_claims(service)[0]
        original_upsert_many = service.vector_store.upsert_many

        def fail_projection(_documents):
            raise RuntimeError("projection unavailable")

        service.vector_store.upsert_many = fail_projection  # type: ignore[method-assign]
        service.feedback(
            FeedbackRequest(user_id="user-a", claim_id=claim.id, action="confirm")
        )
        with service.session_factory() as session:
            refreshed = session.get(Claim, claim.id)
            pending = session.scalar(
                select(ProjectionOutbox).where(
                    ProjectionOutbox.status == "pending",
                    ProjectionOutbox.user_id == "user-a",
                )
            )
            assert refreshed.confidence == 0.98
            assert pending.attempts == 1
            assert pending.error == "RuntimeError"

        service.vector_store.upsert_many = original_upsert_many  # type: ignore[method-assign]
        with service.session_factory.begin() as session:
            session.execute(
                update(ProjectionOutbox)
                .where(ProjectionOutbox.status == "pending")
                .values(next_attempt_at=None)
            )
        assert service.process_projection_outbox() == 1
        assert service.drain_status("user-a")["drained"] is True
    finally:
        service.close()


def test_phase_b_frozen_policy_loads_exact_siliconflow_inputs() -> None:
    settings = Settings(
        semantic_routing_enabled=True,
        semantic_router_config_file=(
            "benchmarks/fixtures/semantic-router-phaseb-frozen-config.json"
        ),
        semantic_router_config_sha256=(
            "c67d3d0b1b279116910219d53be1b99ed053397eb8e739b56a9912711f9e73c9"
        ),
    )
    policy = FrozenSemanticRoutingPolicy.from_settings(
        settings, TestEmbedder(settings.embedding_dimension)
    )
    assert policy.score_threshold == 0.475
    assert policy.margin_threshold == 0.005
    assert policy.prototype_sha256 == (
        "d9e97ec67b5688b44728a1b2bcf73cc39d7625ef1a7baf39b83862cae7a136dc"
    )


def test_phase_b_frozen_policy_rejects_missing_pin() -> None:
    with pytest.raises(ValueError, match="pinned semantic router config"):
        FrozenSemanticRoutingPolicy.from_settings(
            Settings(semantic_routing_enabled=True), TestEmbedder(1024)
        )


def test_build_service_rejects_injected_embedder_dimension_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dimension does not match"):
        build_service(
            Settings(
                database_url=f"sqlite:///{tmp_path / 'mismatch.sqlite3'}",
                vector_mode="memory",
                embedding_dimension=1024,
            ),
            embedder=TestEmbedder(256),
        )


def test_frozen_policy_first_score_uses_one_candidate_anchor_batch() -> None:
    class RecordingEmbedder(TestEmbedder):
        def __init__(self) -> None:
            super().__init__(1024)
            self.batch_sizes: list[int] = []

        def embed_documents(self, texts):
            self.batch_sizes.append(len(texts))
            return super().embed_documents(texts)

    settings = Settings(
        semantic_routing_enabled=True,
        semantic_router_config_sha256=(
            "c67d3d0b1b279116910219d53be1b99ed053397eb8e739b56a9912711f9e73c9"
        ),
    )
    embedder = RecordingEmbedder()
    policy = FrozenSemanticRoutingPolicy.from_settings(settings, embedder)
    policy.score_many(
        [
            CandidateEnvelope(
                user_id="user-a",
                kind="preference",
                semantic_channel="preference.explicit",
                value="point-first prose",
                sensitive=False,
                injection_detected=False,
                deterministic_slot=None,
            )
        ]
    )
    assert embedder.batch_sizes == [29]


def test_outbox_rollback_reconciles_vector_projection(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    try:
        ingest(service, "e1", "I prefer tea")
        assert service.process_outbox() == 1
        original = active_claims(service)[0]
        store = service.vector_store
        assert set(store._points) == {original.id}  # type: ignore[attr-defined]

        original_audit = service._audit

        def fail_after_vector_write(*args, **kwargs):
            if kwargs.get("event_name") == "meno.outbox.processed":
                raise RuntimeError("forced canonical rollback")
            return original_audit(*args, **kwargs)

        service._audit = fail_after_vector_write  # type: ignore[method-assign]
        ingest(service, "e2", "I prefer coffee")
        assert service.process_outbox() == 0
        active = active_claims(service)
        assert [claim.id for claim in active] == [original.id]
        assert set(store._points) == {original.id}  # type: ignore[attr-defined]
    finally:
        service.close()


def test_build_service_runs_frozen_policy_end_to_end(tmp_path: Path) -> None:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'phase-b-real-policy.sqlite3'}",
        vector_mode="memory",
        semantic_routing_enabled=True,
        semantic_router_config_file=(
            "benchmarks/fixtures/semantic-router-phaseb-frozen-config.json"
        ),
        semantic_router_config_sha256=(
            "c67d3d0b1b279116910219d53be1b99ed053397eb8e739b56a9912711f9e73c9"
        ),
    )
    service = build_service(settings, embedder=TestEmbedder(1024))
    try:
        ingest(service, "e1", "I prefer point-first prose")
        assert service.process_outbox() == 1
        assert len(active_claims(service)) == 1
        assert len(routing_audits(service)) == 1
        assert audit_chain_summary(audit_rows(service))["passed"] is True
    finally:
        service.close()
