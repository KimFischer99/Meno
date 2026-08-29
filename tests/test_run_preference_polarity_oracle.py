from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import run_preference_polarity_oracle as oracle

FIXTURE = Path("benchmarks/fixtures/preference-polarity-oracle-v1.json")


def test_fixture_covers_every_polarity_family() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    oracle._validate_fixture(payload)

    families = {case["family"] for case in payload["cases"]}
    # Affirmation alone is not enough: the defect this lane exists to catch was
    # negations and transitions losing their direction. The narrative family is
    # mandatory because an earlier revision covered only explicit phrasing the
    # implementation already handled and passed while real-data coverage was 2%.
    assert families == {"affirmation", "negation", "reversal", "transition", "narrative"}


def test_fixture_rejects_duplicate_user_ids() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["cases"][1]["user_id"] = payload["cases"][0]["user_id"]

    with pytest.raises(ValueError, match="user IDs"):
        oracle._validate_fixture(payload)


def test_fixture_requires_stance_on_every_expectation() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    del payload["cases"][0]["expected_active"][0]["stance"]

    with pytest.raises(ValueError, match="value and stance"):
        oracle._validate_fixture(payload)


def test_polarity_oracle_passes_without_answer_model_or_provider() -> None:
    report = oracle.run_benchmark(FIXTURE)

    assert report["answer_model_calls"] == 0
    assert report["embedding_provider_calls"] == 0
    assert report["all_passed"] is True
    aggregate = report["aggregate"]
    assert aggregate["cases_passed"] == aggregate["cases"]
    assert aggregate["stance_checks_passed"] == aggregate["stance_checks_total"]
    assert report["fixture_sha256"]


def test_reversal_supersedes_instead_of_accumulating() -> None:
    report = oracle.run_benchmark(FIXTURE)
    case = next(item for item in report["cases"] if item["case_id"] == "reversal-across-turns")

    # One active claim carrying the new direction, the old one superseded as a
    # contradiction: this is what makes preference evolution readable.
    assert case["observed_active"] == [
        {"value": "tea", "stance": "negative", "superseded_reason": None}
    ]
    assert case["observed_superseded"] == [
        {"value": "tea", "stance": "positive", "superseded_reason": "contradiction"}
    ]


def test_one_sentence_transition_yields_both_directions() -> None:
    report = oracle.run_benchmark(FIXTURE)
    case = next(item for item in report["cases"] if item["case_id"] == "transition-used-to")

    stances = {record["value"]: record["stance"] for record in case["observed_active"]}
    assert stances == {"book clubs": "negative", "solo reading": "positive"}


def test_same_slot_transition_supersedes_the_abandoned_side() -> None:
    report = oracle.run_benchmark(FIXTURE)
    case = next(
        item for item in report["cases"] if item["case_id"] == "transition-switched-same-slot"
    )

    # tea and coffee share the beverage slot, so the abandoned side must not stay active.
    assert case["observed_active"] == [
        {"value": "coffee", "stance": "positive", "superseded_reason": None}
    ]
    assert [record["value"] for record in case["observed_superseded"]] == ["tea"]
