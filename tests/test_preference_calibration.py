from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks import run_preference_calibration

FIXTURE = Path("benchmarks/fixtures/preference-calibration-v1.json")


def test_fixture_separates_development_and_holdout_labels() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    run_preference_calibration._validate_fixture(payload)

    assert {case["split"] for case in payload["cases"]} == {
        "development",
        "holdout",
    }
    assert len(payload["cases"]) == 16
    assert sum(case["split"] == "holdout" for case in payload["cases"]) == 8


def test_fixture_rejects_duplicate_users() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["cases"][1]["user_id"] = payload["cases"][0]["user_id"]

    with pytest.raises(ValueError, match="user IDs must be unique"):
        run_preference_calibration._validate_fixture(payload)


def test_binary_metrics_calculate_brier_ece_and_nll() -> None:
    metrics = run_preference_calibration._binary_metrics(
        [
            {"probability": 0.8, "mode_correct": True},
            {"probability": 0.2, "mode_correct": False},
        ],
        "probability",
        10,
    )

    assert metrics["brier_score"] == pytest.approx(0.04)
    assert metrics["ece"] == pytest.approx(0.2)
    assert metrics["negative_log_likelihood"] == pytest.approx(-math.log(0.8))


def test_current_distribution_fails_synthetic_calibration_gate() -> None:
    report = run_preference_calibration.run_benchmark(FIXTURE)

    holdout = report["splits"]["holdout"]
    assert report["answer_model_calls"] == 0
    assert report["future_labels_ingested"] is False
    assert holdout["coverage"] == 1.0
    assert holdout["candidate"]["sample_count"] == 8
    assert holdout["candidate"]["positive_labels"] == 4
    assert holdout["candidate"]["negative_labels"] == 4
    assert report["calibration_statuses"] == ["uncalibrated"]
    assert report["diagnostic_gate"]["passed"] is False
    assert report["production_go_eligible"] is False
    assert report["production_decision"] == "NO-GO"


def test_single_label_certainty_is_detected_as_overconfident() -> None:
    report = run_preference_calibration.run_benchmark(FIXTURE)
    holdout_cases = [
        case
        for case in report["cases"]
        if case["split"] == "holdout" and "single" in case["case_id"]
    ]

    assert {case["candidate_probability"] for case in holdout_cases} == {1.0}
    assert any(not case["mode_correct"] for case in holdout_cases)
    assert report["splits"]["holdout"]["candidate"]["overconfidence"] > 0


def test_v2_unknown_mass_improves_diagnostic_but_does_not_grant_go() -> None:
    v1 = run_preference_calibration.run_benchmark(FIXTURE)
    v2 = run_preference_calibration.run_benchmark(FIXTURE, preference_distribution_v2_enabled=True)

    v1_metrics = v1["splits"]["holdout"]["candidate"]
    v2_metrics = v2["splits"]["holdout"]["candidate"]
    assert v2["strategy_versions"] == ["preference-dirichlet-beta-v2"]
    assert v2_metrics["brier_score"] < v1_metrics["brier_score"]
    assert v2_metrics["ece"] < v1_metrics["ece"]
    assert v2["diagnostic_gate"]["passed"] is False
    assert v2["production_go_eligible"] is False
