from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import run_ucm_oracle

FIXTURE = Path("benchmarks/fixtures/ucm-oracle-v1.json")


def test_fixture_has_exactly_the_four_spec_cohorts() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    run_ucm_oracle._validate_fixture(payload)

    assert {case["cohort"] for case in payload["cases"]} == {
        "static",
        "drifting",
        "contradictory",
        "adversarial",
    }


def test_fixture_rejects_duplicate_event_ids() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["cases"][1]["events"][0]["event_id"] = payload["cases"][0]["events"][0]["event_id"]

    with pytest.raises(ValueError, match="event IDs"):
        run_ucm_oracle._validate_fixture(payload)


def test_ucm_oracle_replays_state_without_answer_model() -> None:
    report = run_ucm_oracle.run_benchmark(FIXTURE)

    assert report["answer_model_calls"] == 0
    assert report["aggregate"]["cases"] == 5
    assert (
        report["aggregate"]["state_assertions_passed"]
        == report["aggregate"]["state_assertions_total"]
    )
    assert report["aggregate"]["query_cases_total"] == 10
    assert report["fixture_sha256"]


def test_context_activation_treatment_passes_v1_oracle() -> None:
    report = run_ucm_oracle.run_benchmark(FIXTURE, context_activation_enabled=True)

    assert report["context_activation_enabled"] is True
    assert report["all_passed"] is True


def test_materialized_tokens_match_canonical_state() -> None:
    report = run_ucm_oracle.run_benchmark(
        FIXTURE,
        context_activation_enabled=True,
        user_token_materialization_enabled=True,
    )

    assert report["aggregate"]["snapshot_checks_passed"] == 5
    assert report["aggregate"]["snapshot_checks_total"] == 5
    assert report["all_passed"] is True


def test_preference_distributions_match_oracle_modes() -> None:
    report = run_ucm_oracle.run_benchmark(
        FIXTURE,
        context_activation_enabled=True,
        user_token_materialization_enabled=True,
        preference_distribution_enabled=True,
    )

    aggregate = report["aggregate"]
    assert (
        aggregate["preference_distribution_checks_passed"]
        == aggregate["preference_distribution_checks_total"]
    )
    assert aggregate["preference_distribution_checks_total"] > 4
    assert report["all_passed"] is True


def test_ambiguous_preference_is_withheld_for_clarification() -> None:
    report = run_ucm_oracle.run_benchmark(
        FIXTURE,
        context_activation_enabled=True,
        user_token_materialization_enabled=True,
        preference_distribution_enabled=True,
        clarification_opportunities_enabled=True,
    )

    aggregate = report["aggregate"]
    assert (
        aggregate["clarification_opportunity_checks_passed"]
        == aggregate["clarification_opportunity_checks_total"]
    )
    assert aggregate["clarification_opportunity_checks_total"] > 5
    ambiguous = next(case for case in report["cases"] if case["case_id"] == "ambiguous-user")
    assert ambiguous["queries"][0]["actual_decision"] == "clarify"
    assert ambiguous["queries"][0]["selected_values"] == []
    assert report["all_passed"] is True


def test_metrics_count_query_checks_independently() -> None:
    result = run_ucm_oracle._metrics(
        [
            {
                "state_assertions": [{"passed": True}, {"passed": False}],
                "queries": [
                    {
                        "passed": False,
                        "checks": {"a": True, "b": False},
                        "context_precision": 0.5,
                        "degraded": False,
                    }
                ],
            }
        ]
    )

    assert result["state_assertions_passed"] == 1
    assert result["query_cases_passed"] == 0
    assert result["query_checks_passed"] == 1
    assert result["query_checks_total"] == 2
