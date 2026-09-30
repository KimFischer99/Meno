from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from benchmarks.run_semantic_router_a3_development import load_prototypes
from benchmarks.run_semantic_router_a4_development import (
    DEFAULT_A31_ARTIFACT,
    DEFAULT_A31_FIXTURE,
    DEFAULT_A31_MANIFEST,
    DEFAULT_PROTOTYPES,
    DEFAULT_V2_FIXTURE,
    EXPECTED_PROTOTYPE_SHA256,
    load_a4_development,
    provider_coverage,
    role_annotated_prototypes,
)
from benchmarks.run_semantic_router_a5_diagnostics import (
    _baseline_top2_mean,
    polarity_annotated_prototypes,
)
from meno.semantic_router import (
    A5_STRATEGY_VERSION,
    DimensionScore,
    PrototypeEnsembleStrategy,
    TopicGatedStrategy,
)
from tests.fakes import TestEmbedder

ROOT = Path(__file__).parent.parent

# The archive-backed tests below replay the a31 provider-holdout run, whose
# result file under artifacts/ is local-only (see benchmarks/README.md).
requires_a31_artifact = pytest.mark.skipif(
    not DEFAULT_A31_ARTIFACT.is_file(),
    reason="Historical a31 holdout artifact is local-only; archive-backed tests skip offline",
)


def _annotated_prototypes() -> tuple[object, ...]:
    _, _, base_prototypes, metadata = load_prototypes(DEFAULT_PROTOTYPES)
    return polarity_annotated_prototypes(
        role_annotated_prototypes(base_prototypes, metadata),
        metadata,
    )


def test_a5_diagnostics_annotates_every_anchor_with_role_and_polarity() -> None:
    prototypes = _annotated_prototypes()

    assert len(prototypes) == 28
    assert hashlib.sha256(Path(DEFAULT_PROTOTYPES).read_bytes()).hexdigest() == (
        EXPECTED_PROTOTYPE_SHA256
    )
    assert [prototype.role for prototype in prototypes].count("topic") == 7
    assert [prototype.role for prototype in prototypes].count("polarity") == 21
    assert all(prototype.polarity is not None for prototype in prototypes)
    for dimension in ("answer_style", "beverage", "color", "food", "music",
                      "programming_language", "sport"):
        anchors = [p for p in prototypes if p.name == dimension]
        assert [p.polarity for p in anchors if p.role == "topic"] == ["neutral"]
        assert {p.polarity for p in anchors if p.role == "polarity"} == {
            "affirmed",
            "negated",
            "contradiction",
        }


@requires_a31_artifact
def test_a5_diagnostics_baseline_matches_ensemble_top2_mean() -> None:
    prototypes = _annotated_prototypes()
    _, cases, _ = load_a4_development(
        DEFAULT_V2_FIXTURE,
        DEFAULT_A31_FIXTURE,
        DEFAULT_A31_ARTIFACT,
        DEFAULT_A31_MANIFEST,
    )
    candidate = next(
        case.candidate
        for case in cases
        if case.candidate.deterministic_slot is None
        and not case.candidate.sensitive
        and not case.candidate.injection_detected
    )

    ensemble = PrototypeEnsembleStrategy(
        TestEmbedder(dimension=128), prototypes, aggregation="top2_mean"
    )
    ensemble_score = ensemble.score(candidate)

    gated = TopicGatedStrategy(TestEmbedder(dimension=128), prototypes)
    detailed = gated.score_many_detailed([candidate])[0]

    assert ensemble_score is not None
    assert detailed is not None
    baseline = _baseline_top2_mean(dict(detailed.anchor_scores))
    ranked = sorted(baseline, key=lambda name: baseline[name], reverse=True)
    assert ranked[0] == ensemble_score.dimension
    assert abs(baseline[ranked[0]] - ensemble_score.score) < 1e-12
    assert abs(
        baseline[ranked[0]] - baseline[ranked[1]] - ensemble_score.margin
    ) < 1e-12


@requires_a31_artifact
def test_a5_topic_gated_strategy_scores_the_full_development_fixture() -> None:
    prototypes = _annotated_prototypes()
    _, cases, _ = load_a4_development(
        DEFAULT_V2_FIXTURE,
        DEFAULT_A31_FIXTURE,
        DEFAULT_A31_ARTIFACT,
        DEFAULT_A31_MANIFEST,
    )
    strategy = TopicGatedStrategy(TestEmbedder(dimension=128), prototypes)
    scored_cases = [
        case
        for case in cases
        if (
            case.candidate.deterministic_slot is None
            and not case.candidate.sensitive
            and not case.candidate.injection_detected
        )
    ]
    detailed = strategy.score_many_detailed([case.candidate for case in scored_cases])

    assert all(item is not None for item in detailed)
    scores = {
        case.case_id: item.dimension_score
        for case, item in zip(scored_cases, detailed, strict=True)
    }
    assert all(score.strategy_version == A5_STRATEGY_VERSION for score in scores.values())
    designed = {
        case.case_id: scores.get(case.case_id)
        for case in cases
        if (
            case.candidate.deterministic_slot is None
            and not case.candidate.sensitive
            and not case.candidate.injection_detected
            and case.expected_dimension
            in {
                "answer_style",
                "beverage",
                "color",
                "food",
                "music",
                "programming_language",
                "sport",
            }
        )
    }
    assert provider_coverage(cases, designed)["passed"] is True


@requires_a31_artifact
def test_a5_topic_gated_dimension_score_is_topic_only() -> None:
    prototypes = _annotated_prototypes()
    strategy = TopicGatedStrategy(TestEmbedder(dimension=128), prototypes)
    _, cases, _ = load_a4_development(
        DEFAULT_V2_FIXTURE,
        DEFAULT_A31_FIXTURE,
        DEFAULT_A31_ARTIFACT,
        DEFAULT_A31_MANIFEST,
    )
    candidate = next(
        case.candidate
        for case in cases
        if case.case_id == "a31-contradiction-as"
    )

    detailed = strategy.score_many_detailed([candidate])[0]

    assert detailed is not None
    score = detailed.dimension_score
    assert isinstance(score, DimensionScore)
    assert score.dimension in {
        "answer_style",
        "beverage",
        "color",
        "food",
        "music",
        "programming_language",
        "sport",
    }
    topic_anchor_scores = {
        dimension: labels["topic"] for dimension, labels in detailed.anchor_scores.items()
    }
    assert score.score == max(topic_anchor_scores.values())
    assert topic_anchor_scores[score.dimension] == score.score
