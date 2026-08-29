from __future__ import annotations

from pathlib import Path

from benchmarks.run_semantic_router_shadow import (
    EXPECTED_DIMENSIONS,
    FixtureCase,
    _safe_endpoint,
    decision_metrics,
    load_fixture,
    recommend_thresholds,
    route_cases,
)
from meno.semantic_router import STRATEGY_VERSION, DimensionScore

FIXTURE = (
    Path(__file__).parent.parent / "benchmarks" / "fixtures" / "semantic-normalization-v2.json"
)


def _oracle_scores(
    cases: tuple[FixtureCase, ...],
) -> dict[str, DimensionScore | None]:
    scores: dict[str, DimensionScore | None] = {}
    for case in cases:
        if case.candidate.deterministic_slot is not None or case.expected_dimension is None:
            scores[case.case_id] = None
            continue
        low_confidence = case.category in {"lexical_collision", "entity_collision"}
        score = 0.1 if low_confidence else 1.0
        scores[case.case_id] = DimensionScore(
            dimension=case.expected_dimension,
            score=score,
            second_dimension="other",
            second_score=0.0,
            margin=score,
            strategy_version=STRATEGY_VERSION,
        )
    return scores


def test_v2_fixture_is_split_and_uses_all_controlled_dimensions() -> None:
    _, prototypes, cases = load_fixture(FIXTURE)

    assert len(cases) >= 44
    assert {prototype.name for prototype in prototypes} == EXPECTED_DIMENSIONS
    assert {case.split for case in cases} == {"calibration", "holdout"}
    assert {case.expected_dimension for case in cases} >= EXPECTED_DIMENSIONS


def test_fixture_contract_routes_under_oracle_dimension_scores() -> None:
    _, _, cases = load_fixture(FIXTURE)
    scores = _oracle_scores(cases)

    decisions = route_cases(
        cases,
        scores,
        score_threshold=0.5,
        margin_threshold=0.0,
    )
    metrics = decision_metrics(cases, decisions)

    assert metrics["false_merge_count"] == 0
    assert metrics["missed_merge_count"] == 0
    assert metrics["accuracy"] == 1.0


def test_calibration_recommendation_does_not_consume_holdout_labels() -> None:
    _, _, cases = load_fixture(FIXTURE)
    scores = _oracle_scores(cases)

    recommendation, grid = recommend_thresholds(
        cases,
        scores,
        score_thresholds=[0.0, 0.5, 0.9],
        margin_thresholds=[0.0],
        minimum_recall=0.5,
    )

    assert len(grid) == 3
    assert recommendation is not None
    assert recommendation["score_threshold"] == 0.9

    holdout = tuple(case for case in cases if case.split == "holdout")
    decisions = route_cases(
        holdout,
        scores,
        score_threshold=recommendation["score_threshold"],
        margin_threshold=recommendation["margin_threshold"],
    )
    metrics = decision_metrics(holdout, decisions)
    assert metrics["false_merge_count"] == 0
    assert metrics["recall"] >= 0.5


def test_high_confidence_collision_blocks_calibration() -> None:
    _, _, cases = load_fixture(FIXTURE)
    scores = _oracle_scores(cases)
    collision = next(case for case in cases if case.category == "lexical_collision")
    scores[collision.case_id] = DimensionScore(
        dimension=collision.expected_dimension or "beverage",
        score=1.0,
        second_dimension="other",
        second_score=0.0,
        margin=1.0,
        strategy_version=STRATEGY_VERSION,
    )

    recommendation, _ = recommend_thresholds(
        cases,
        scores,
        score_thresholds=[0.5, 1.0],
        margin_thresholds=[0.0, 1.0],
        minimum_recall=0.5,
    )

    assert recommendation is None


def test_provider_endpoint_metadata_strips_credentials_and_query() -> None:
    endpoint = "https://user:secret@example.com:8443/v1?token=secret#fragment"

    assert _safe_endpoint(endpoint) == "https://example.com:8443/v1"
