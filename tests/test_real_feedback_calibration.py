from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import run_real_feedback_calibration
from meno.api import build_service
from meno.config import Settings
from meno.schemas import FeedbackRequest, IngestRequest
from tests.fakes import TestEmbedder

GATE_CONFIG = Path("benchmarks/fixtures/real-feedback-calibration-gate-v1.json")


def _database_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_feedback_database(path: Path) -> None:
    service = build_service(
        Settings(
            database_url=f"sqlite:///{path}",
            vector_mode="memory",
            embedding_dimension=256,
            user_token_materialization_enabled=True,
            preference_distribution_enabled=True,
            preference_distribution_v2_enabled=True,
        ),
        embedder=TestEmbedder(256),
    )
    try:
        cases = [
            ("real-confirm", "I prefer concise answers", "confirm", None),
            ("real-reject", "I prefer tea", "reject", None),
            (
                "real-correct",
                "I prefer Python",
                "correct",
                "I prefer Rust now",
            ),
        ]
        for index, (user_id, preference, action, correction) in enumerate(cases):
            event_id = f"{user_id}-event"
            service.ingest(
                IngestRequest.model_validate(
                    {
                        "user_id": user_id,
                        "event_id": event_id,
                        "occurred_at": f"2026-08-0{index + 1}T12:00:00Z",
                        "source": {"type": "hermes_turn", "session_id": user_id},
                        "content": {"role": "user", "text": preference},
                        "consent_scope": ["personalization"],
                    }
                ),
                idempotency_key=event_id,
            )
            service.process_outbox()
            token = service.user_token(user_id)
            claim_id = token["payload"]["active_state"][0]["claim_id"]
            service.feedback(
                FeedbackRequest(
                    user_id=user_id,
                    claim_id=claim_id,
                    action=action,
                    correction=correction,
                )
            )
        service.ingest(
            IngestRequest.model_validate(
                {
                    "user_id": "real-episodic",
                    "event_id": "real-episodic-event",
                    "occurred_at": "2026-08-04T12:00:00Z",
                    "source": {"type": "hermes_turn"},
                    "content": {
                        "role": "user",
                        "text": "I am building the Meno project",
                    },
                    "consent_scope": ["personalization"],
                }
            ),
            idempotency_key="real-episodic-event",
        )
        service.process_outbox()
        episodic_claim_id = service.user_token("real-episodic")["payload"]["active_state"][0][
            "claim_id"
        ]
        service.feedback(
            FeedbackRequest(
                user_id="real-episodic",
                claim_id=episodic_claim_id,
                action="confirm",
            )
        )
    finally:
        service.close()


def test_real_feedback_extractor_pairs_prior_revisions_without_payload_leakage(
    tmp_path,
) -> None:
    database = tmp_path / "real-feedback.sqlite3"
    _build_feedback_database(database)
    before_sha = _database_sha(database)

    report = run_real_feedback_calibration.run_benchmark(f"sqlite:///{database}", GATE_CONFIG)

    assert _database_sha(database) == before_sha
    assert report["read_only"] is True
    assert report["source_feedback_count"] == 3
    assert report["paired_forecast_count"] == 3
    assert report["excluded"] == {}
    assert {sample["feedback_action"] for sample in report["samples"]} == {
        "confirm",
        "reject",
        "correct",
    }
    assert sorted(sample["mode_correct"] for sample in report["samples"]) == [
        False,
        False,
        True,
    ]
    assert {sample["candidate_probability"] for sample in report["samples"]} == {0.5}
    assert all(
        sample["outcome_revision"] == sample["forecast_revision"] + 1
        for sample in report["samples"]
    )
    serialized = json.dumps(report)
    assert "concise answers" not in serialized
    assert "I prefer tea" not in serialized
    assert "I prefer Rust now" not in serialized
    assert "feedback_created_at" not in serialized


def test_real_feedback_lane_fails_closed_when_data_is_insufficient(tmp_path) -> None:
    database = tmp_path / "insufficient.sqlite3"
    _build_feedback_database(database)

    report = run_real_feedback_calibration.run_benchmark(f"sqlite:///{database}", GATE_CONFIG)

    assert report["data_readiness"]["passed"] is False
    assert report["quality_gate"]["passed"] is False
    assert report["production_go_eligible"] is False
    assert report["production_decision"] == "NO-GO"


def test_real_feedback_gate_config_is_valid() -> None:
    payload = json.loads(GATE_CONFIG.read_text(encoding="utf-8"))

    run_real_feedback_calibration._validate_gate_config(payload)

    assert payload["data_readiness"]["minimum_total_samples"] == 100
    assert payload["development_fraction"] == 0.7


def test_read_only_lane_does_not_create_a_missing_sqlite_database(tmp_path) -> None:
    missing = tmp_path / "missing.sqlite3"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_real_feedback_calibration.run_benchmark(f"sqlite:///{missing}", GATE_CONFIG)

    assert not missing.exists()
