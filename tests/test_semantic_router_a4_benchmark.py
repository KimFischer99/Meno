from __future__ import annotations

import hashlib

import pytest

from benchmarks.run_semantic_router_a3_development import load_prototypes
from benchmarks.run_semantic_router_a4_development import (
    DEFAULT_A31_ARTIFACT,
    DEFAULT_A31_FIXTURE,
    DEFAULT_A31_MANIFEST,
    DEFAULT_MARGIN_THRESHOLDS,
    DEFAULT_PROTOTYPES,
    DEFAULT_V2_FIXTURE,
    EXPECTED_PROTOTYPE_SHA256,
    _contains_raw_artifact_text,
    evaluate_a4_variant,
    load_a4_development,
    provider_coverage,
    role_annotated_prototypes,
    route_a4_development,
    select_a4_variant,
)
from meno.semantic_router import A4_STRATEGY_VERSION, DimensionScore

# Every archive-backed test below replays the a31 provider-holdout run, whose
# result file under artifacts/ is local-only (see benchmarks/README.md).
requires_a31_artifact = pytest.mark.skipif(
    not DEFAULT_A31_ARTIFACT.is_file(),
    reason="Historical a31 holdout artifact is local-only; archive-backed tests skip offline",
)


def _oracle_scores(cases):  # type: ignore[no-untyped-def]
    scores: dict[str, DimensionScore | None] = {}
    for case in cases:
        if case.candidate.deterministic_slot is not None or case.expected_dimension is None:
            scores[case.case_id] = None
            continue
        score = 0.1 if case.category in {"lexical_collision", "entity_collision"} else 1.0
        scores[case.case_id] = DimensionScore(
            dimension=case.expected_dimension,
            score=score,
            second_dimension="other",
            second_score=0.0,
            margin=score,
            strategy_version=A4_STRATEGY_VERSION,
        )
    return scores


@requires_a31_artifact
def test_a4_development_is_exactly_the_two_revealed_sources() -> None:
    _, cases, sources = load_a4_development(
        DEFAULT_V2_FIXTURE,
        DEFAULT_A31_FIXTURE,
        DEFAULT_A31_ARTIFACT,
        DEFAULT_A31_MANIFEST,
    )

    assert len(cases) == 111
    assert len(sources["v2"]) == 62
    assert len(sources["a31"]) == 49
    assert len({case.case_id for case in cases}) == 111


def test_a4_prototypes_have_one_topic_and_three_polarity_anchors() -> None:
    raw, _, base_prototypes, metadata = load_prototypes(DEFAULT_PROTOTYPES)
    prototypes = role_annotated_prototypes(base_prototypes, metadata)

    assert hashlib.sha256(raw).hexdigest() == EXPECTED_PROTOTYPE_SHA256
    assert len(prototypes) == 28
    assert [prototype.role for prototype in prototypes].count("topic") == 7
    assert [prototype.role for prototype in prototypes].count("polarity") == 21
    for anchors in metadata["dimensions"].values():
        assert [anchor["role"] for anchor in anchors].count("topic") == 1
        assert [anchor["role"] for anchor in anchors].count("polarity") == 3


def test_a4_raw_artifact_guard_avoids_short_substring_false_positives() -> None:
    assert not _contains_raw_artifact_text(
        {"coverage_required": True, "provider_scored_count": 47},
        ("red",),
    )
    assert _contains_raw_artifact_text({"candidate": "red"}, ("red",))
    assert _contains_raw_artifact_text(
        {"message": "prefix private candidate suffix"},
        ("private candidate",),
    )


@requires_a31_artifact
def test_a4_gate_requires_each_source_and_answer_style_to_pass() -> None:
    _, cases, sources = load_a4_development(
        DEFAULT_V2_FIXTURE,
        DEFAULT_A31_FIXTURE,
        DEFAULT_A31_ARTIFACT,
        DEFAULT_A31_MANIFEST,
    )
    recommendation, grid = evaluate_a4_variant(
        "role_top2_mean",
        cases,
        sources,
        _oracle_scores(cases),
        score_thresholds=[0.5],
        margin_thresholds=[0.2],
        minimum_overall_recall=0.75,
        minimum_source_recall=0.75,
    )

    assert len(grid) == 1
    assert recommendation is not None
    assert recommendation["source_false_merge_count"] == {"v2": 0, "a31": 0}
    assert recommendation["a31_answer_style_failure_ids"] == []
    assert recommendation["safety_passed"] is True
    assert recommendation["robustness_passed"] is True
    assert provider_coverage(cases, _oracle_scores(cases))["passed"] is True


@requires_a31_artifact
def test_a4_high_confidence_collision_has_no_recommendation() -> None:
    _, cases, sources = load_a4_development(
        DEFAULT_V2_FIXTURE,
        DEFAULT_A31_FIXTURE,
        DEFAULT_A31_ARTIFACT,
        DEFAULT_A31_MANIFEST,
    )
    scores = _oracle_scores(cases)
    collision = next(case for case in cases if case.category == "lexical_collision")
    scores[collision.case_id] = DimensionScore(
        dimension=collision.expected_dimension or "beverage",
        score=1.0,
        second_dimension="other",
        second_score=0.0,
        margin=1.0,
        strategy_version=A4_STRATEGY_VERSION,
    )

    recommendation, _ = evaluate_a4_variant(
        "role_top2_mean",
        cases,
        sources,
        scores,
        score_thresholds=[0.5, 1.0],
        margin_thresholds=[0.0, 1.0],
        minimum_overall_recall=0.75,
        minimum_source_recall=0.75,
    )

    assert recommendation is None


@requires_a31_artifact
def test_a4_router_and_tie_break_use_role_aware_version() -> None:
    _, cases, _ = load_a4_development(
        DEFAULT_V2_FIXTURE,
        DEFAULT_A31_FIXTURE,
        DEFAULT_A31_ARTIFACT,
        DEFAULT_A31_MANIFEST,
    )
    decisions = route_a4_development(
        cases,
        _oracle_scores(cases),
        aggregation="role_top2_mean",
        score_threshold=0.5,
        margin_threshold=0.2,
    )
    base = {
        "recall": 0.9,
        "source_recall": {"v2": 0.9, "a31": 0.9},
        "accuracy": 0.95,
        "score_threshold": 0.5,
        "margin_threshold": 0.2,
    }

    assert all(decision.strategy_version == A4_STRATEGY_VERSION for decision in decisions.values())
    assert (
        select_a4_variant(
            [
                {**base, "aggregation": "top2_mean"},
                {**base, "aggregation": "role_top2_mean"},
            ]
        )["aggregation"]
        == "role_top2_mean"
    )
    assert (
        select_a4_variant(
            [
                {**base, "aggregation": "top2_mean"},
                {**base, "aggregation": "role_top2_mean", "accuracy": 0.9},
            ]
        )["aggregation"]
        == "top2_mean"
    )
    assert 0.02 in DEFAULT_MARGIN_THRESHOLDS
