from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from meno.api import build_service, create_app
from meno.config import Settings
from meno.schemas import (
    ConsentRequest,
    DeletionRequest,
    FeedbackRequest,
    IngestRequest,
    RetrieveRequest,
)
from tests.fakes import TestEmbedder


def make_service(
    path: Path,
    *,
    enabled: bool = True,
    preference_distribution: bool = False,
    preference_distribution_v2: bool = False,
    clarification: bool = False,
):
    settings = Settings(
        database_url=f"sqlite:///{path}",
        vector_mode="memory",
        embedding_dimension=256,
        user_token_materialization_enabled=enabled,
        preference_distribution_enabled=preference_distribution,
        preference_distribution_v2_enabled=preference_distribution_v2,
        context_activation_enabled=clarification,
        clarification_opportunities_enabled=clarification,
    )
    return build_service(settings, embedder=TestEmbedder(256))


def ingest(service, event_id: str, text: str, occurred_at: str) -> None:
    service.ingest(
        IngestRequest.model_validate(
            {
                "user_id": "user-token-test",
                "event_id": event_id,
                "occurred_at": occurred_at,
                "source": {"type": "hermes_turn", "session_id": "s1"},
                "content": {"role": "user", "text": text},
                "consent_scope": ["personalization", "task_planning"],
            }
        ),
        idempotency_key=event_id,
    )
    service.process_outbox()


def test_materialized_token_contains_stable_active_state_and_lineage(tmp_path) -> None:
    service = make_service(tmp_path / "token.sqlite3")
    try:
        ingest(service, "token-pref", "I prefer concise answers", "2026-08-20T12:00:00Z")

        token = service.user_token("user-token-test")

        assert token["schema_version"] == "1.0.0"
        assert token["content_hash"].startswith("sha256:")
        assert token["payload"]["state_revision"] == token["state_revision"]
        assert token["payload"]["active_state"][0]["value"] == "concise answers"
        assert token["payload"]["active_state"][0]["evidence_ids"] == ["token-pref"]
        assert token["payload"]["uncertainty"]["unsupported_claim_ids"] == []
    finally:
        service.close()


def test_user_token_diff_tracks_preference_replacement(tmp_path) -> None:
    service = make_service(tmp_path / "diff.sqlite3")
    try:
        ingest(service, "token-old", "I prefer terse answers", "2026-01-10T12:00:00Z")
        before = service.user_token("user-token-test")
        claim_id = before["payload"]["active_state"][0]["claim_id"]

        service.feedback(
            FeedbackRequest(
                user_id="user-token-test",
                claim_id=claim_id,
                action="correct",
                correction="I prefer detailed answers now",
            )
        )
        after = service.user_token("user-token-test")
        diff = service.diff_user_tokens(
            "user-token-test", before["state_revision"], after["state_revision"]
        )

        assert diff["added"] == []
        assert diff["removed"] == []
        assert len(diff["changed"]) == 1
        assert diff["changed"][0]["before"]["value"] == "terse answers"
        assert diff["changed"][0]["after"]["value"] == "I prefer detailed answers now"
    finally:
        service.close()


@pytest.mark.parametrize("preference_distribution_v2", [False, True])
def test_same_event_replay_produces_same_token_hash(
    tmp_path, preference_distribution_v2: bool
) -> None:
    hashes = []
    for index in range(2):
        service = make_service(
            tmp_path / f"replay-{index}.sqlite3",
            preference_distribution=True,
            preference_distribution_v2=preference_distribution_v2,
        )
        try:
            ingest(
                service,
                "stable-event",
                "I prefer concise answers",
                "2026-08-20T12:00:00Z",
            )
            token = service.user_token("user-token-test")
            hashes.append(
                (
                    token["content_hash"],
                    token["payload"]["preference_distributions"][0]["content_hash"],
                )
            )
        finally:
            service.close()

    assert hashes[0] == hashes[1]


def test_consent_change_is_materialized_and_diffable(tmp_path) -> None:
    service = make_service(tmp_path / "consent.sqlite3")
    try:
        ingest(service, "consent-event", "I prefer tea", "2026-08-20T12:00:00Z")
        before = service.user_token("user-token-test")
        service.set_consent(
            ConsentRequest(
                user_id="user-token-test",
                source="obsidian",
                purpose="response_personalization",
                allowed_operations=["ingest"],
                status="revoked",
            )
        )
        after = service.user_token("user-token-test")
        diff = service.diff_user_tokens(
            "user-token-test", before["state_revision"], after["state_revision"]
        )

        assert diff["consent_changed"] is True
        assert diff["consent_before"] == []
        assert diff["consent_after"][0]["status"] == "revoked"
    finally:
        service.close()


def test_preference_distribution_updates_after_explicit_correction(tmp_path) -> None:
    service = make_service(tmp_path / "distribution.sqlite3", preference_distribution=True)
    try:
        ingest(service, "distribution-old", "I prefer terse answers", "2026-01-10T12:00:00Z")
        before = service.user_token("user-token-test")
        claim_id = before["payload"]["active_state"][0]["claim_id"]
        initial = before["payload"]["preference_distributions"][0]
        assert initial["distribution_type"] == "dirichlet"
        assert initial["calibration_status"] == "uncalibrated"
        assert initial["parameters"]["mode"] == "terse answers"

        service.feedback(
            FeedbackRequest(
                user_id="user-token-test",
                claim_id=claim_id,
                action="correct",
                correction="I prefer detailed answers now",
            )
        )
        after = service.user_token("user-token-test")
        distribution = after["payload"]["preference_distributions"][0]
        probabilities = distribution["parameters"]["probabilities"]
        diff = service.diff_user_tokens(
            "user-token-test", before["state_revision"], after["state_revision"]
        )

        assert distribution["parameters"]["mode"] == "I prefer detailed answers now"
        assert probabilities["I prefer detailed answers now"] > probabilities["terse answers"]
        assert sum(probabilities.values()) == pytest.approx(1.0)
        assert diff["preference_distributions_changed"] is True
    finally:
        service.close()


def test_rejecting_only_active_preference_removes_distribution(tmp_path) -> None:
    service = make_service(tmp_path / "reject.sqlite3", preference_distribution=True)
    try:
        ingest(service, "reject-event", "I prefer tea", "2026-08-20T12:00:00Z")
        before = service.user_token("user-token-test")
        claim_id = before["payload"]["active_state"][0]["claim_id"]

        service.feedback(
            FeedbackRequest(
                user_id="user-token-test",
                claim_id=claim_id,
                action="reject",
            )
        )
        after = service.user_token("user-token-test")

        assert after["payload"]["preference_distributions"] == []
    finally:
        service.close()


def test_preference_distribution_v2_preserves_unknown_mass_until_confirmed(
    tmp_path,
) -> None:
    service = make_service(
        tmp_path / "distribution-v2.sqlite3",
        preference_distribution=True,
        preference_distribution_v2=True,
    )
    try:
        ingest(
            service,
            "distribution-v2-event",
            "I prefer concise answers",
            "2026-08-20T12:00:00Z",
        )
        before = service.user_token("user-token-test")
        claim_id = before["payload"]["active_state"][0]["claim_id"]
        initial = before["payload"]["preference_distributions"][0]

        assert initial["distribution_type"] == "dirichlet_beta"
        assert initial["strategy_version"] == "preference-dirichlet-beta-v2"
        assert initial["calibration_status"] == "uncalibrated"
        assert initial["parameters"]["mode_probability"] == pytest.approx(1.0)
        assert initial["parameters"]["belief"]["probability"] == pytest.approx(0.5)
        assert initial["parameters"]["belief"]["unknown_mass"] == pytest.approx(0.5)
        assert initial["parameters"]["actionable_probability"] == pytest.approx(0.5)

        service.feedback(
            FeedbackRequest(user_id="user-token-test", claim_id=claim_id, action="confirm")
        )
        confirmed = service.user_token("user-token-test")["payload"]["preference_distributions"][0]

        assert confirmed["parameters"]["belief"]["probability"] == pytest.approx(0.75)
        assert confirmed["parameters"]["belief"]["unknown_mass"] == pytest.approx(0.25)
        assert confirmed["parameters"]["actionable_probability"] == pytest.approx(0.75)
    finally:
        service.close()


def test_ambiguous_preference_requires_clarification_until_confirmed(tmp_path) -> None:
    service = make_service(
        tmp_path / "clarify.sqlite3",
        preference_distribution=True,
        clarification=True,
    )
    try:
        ingest(
            service,
            "clarify-event",
            "I prefer maybe concise answers",
            "2026-08-20T12:00:00Z",
        )
        before = service.user_token("user-token-test")
        opportunity = before["payload"]["clarification_opportunities"][0]
        claim_id = before["payload"]["active_state"][0]["claim_id"]
        recall = service.retrieve(
            RetrieveRequest.model_validate(
                {
                    "user_id": "user-token-test",
                    "purpose": "response_personalization",
                    "context": {
                        "query": "How should you format answers for me?",
                        "task_type": "response_style",
                    },
                    "constraints": {"min_confidence": 0.5},
                }
            )
        )

        assert opportunity["reason"] == "ambiguous_expression"
        assert recall.facets == []
        assert recall.clarification_opportunities[0].claim_id == claim_id

        service.feedback(
            FeedbackRequest(user_id="user-token-test", claim_id=claim_id, action="confirm")
        )
        after = service.user_token("user-token-test")
        confirmed = service.retrieve(
            RetrieveRequest.model_validate(
                {
                    "user_id": "user-token-test",
                    "purpose": "response_personalization",
                    "context": {
                        "query": "How should you format answers for me?",
                        "task_type": "response_style",
                    },
                    "constraints": {"min_confidence": 0.5},
                }
            )
        )

        assert after["payload"]["clarification_opportunities"] == []
        assert confirmed.facets[0].value == "maybe concise answers"
        assert confirmed.clarification_opportunities == []
    finally:
        service.close()


def test_full_deletion_removes_materialized_token(tmp_path) -> None:
    service = make_service(tmp_path / "delete.sqlite3")
    try:
        ingest(service, "delete-event", "I prefer tea", "2026-08-20T12:00:00Z")
        service.delete(DeletionRequest(user_id="user-token-test", scope="all"))

        with pytest.raises(LookupError, match="not found"):
            service.user_token("user-token-test")
    finally:
        service.close()


def test_materialization_feature_defaults_to_no_snapshots(tmp_path) -> None:
    service = make_service(tmp_path / "disabled.sqlite3", enabled=False)
    try:
        ingest(service, "disabled-event", "I prefer tea", "2026-08-20T12:00:00Z")

        with pytest.raises(LookupError, match="not found"):
            service.user_token("user-token-test")
    finally:
        service.close()


def test_user_token_is_available_through_read_only_api(tmp_path) -> None:
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'api.sqlite3'}",
        vector_mode="memory",
        embedding_dimension=256,
        user_token_materialization_enabled=True,
    )
    app = create_app(settings, embedder=TestEmbedder(256))
    with TestClient(app) as client:
        response = client.post(
            "/v1/ingest",
            headers={"Idempotency-Key": "token-api-event"},
            json={
                "user_id": "api-user",
                "event_id": "token-api-event",
                "occurred_at": "2026-08-20T12:00:00Z",
                "source": {"type": "hermes_turn", "session_id": "s1"},
                "content": {"role": "user", "text": "I prefer concise answers"},
                "consent_scope": ["personalization"],
            },
        )
        assert response.status_code == 202
        app.state.meno.process_outbox()

        token = client.get("/v1/users/api-user/token")

        assert token.status_code == 200
        assert token.json()["payload"]["active_state"][0]["value"] == "concise answers"
