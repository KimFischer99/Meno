from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.run_semantic_router_a3_development import (
    evaluate_variant,
    load_prototypes,
    parse_aggregations,
    select_variant,
)
from benchmarks.run_semantic_router_a3_holdout import (
    _normalized_candidate_texts,
    load_fresh_holdout,
    load_frozen_config,
)
from benchmarks.run_semantic_router_a31_holdout import (
    _contains_raw_text,
    load_evaluation_manifest,
    load_provider_holdout,
    provider_coverage,
)
from benchmarks.run_semantic_router_shadow import FixtureCase, _sha256, load_fixture
from meno.semantic_router import ENSEMBLE_STRATEGY_VERSION, DimensionScore

ROOT = Path(__file__).parent.parent
V2_FIXTURE = ROOT / "benchmarks" / "fixtures" / "semantic-normalization-v2.json"
A3_PROTOTYPES = ROOT / "benchmarks" / "fixtures" / "semantic-prototypes-a3-v1.json"
A3_CONFIG = ROOT / "benchmarks" / "fixtures" / "semantic-router-a3-frozen-config.json"
A3_HOLDOUT = ROOT / "benchmarks" / "fixtures" / "semantic-normalization-v3-fresh-holdout.json"
A3_HOLDOUT_RUNNER = ROOT / "benchmarks" / "run_semantic_router_a3_holdout.py"
A31_CONFIG = ROOT / "benchmarks" / "fixtures" / "semantic-router-a31-frozen-config.json"
A31_HOLDOUT = ROOT / "benchmarks" / "fixtures" / "semantic-normalization-v4-provider-holdout.json"
SEALED_A3_ROUTER = ROOT / "benchmarks" / "sealed_sources" / "a3" / "semantic_router.py"
SEALED_A31_ROUTER = ROOT / "benchmarks" / "sealed_sources" / "a31" / "semantic_router.py"
SEALED_A3_DEVELOPMENT_RUNNER = (
    ROOT / "benchmarks" / "sealed_sources" / "a3_a31" / "run_semantic_router_a3_development.py"
)


def _development_scores(
    cases: tuple[FixtureCase, ...],
) -> dict[str, DimensionScore | None]:
    result: dict[str, DimensionScore | None] = {}
    for case in cases:
        if case.candidate.deterministic_slot is not None or case.expected_dimension is None:
            result[case.case_id] = None
            continue
        low_confidence = case.category in {"lexical_collision", "entity_collision"}
        score = 0.1 if low_confidence else 1.0
        result[case.case_id] = DimensionScore(
            dimension=case.expected_dimension,
            score=score,
            second_dimension="other",
            second_score=0.0,
            margin=score,
            strategy_version=ENSEMBLE_STRATEGY_VERSION,
        )
    return result


def test_a3_prototypes_have_four_anchors_per_controlled_dimension() -> None:
    _, version, prototypes, metadata = load_prototypes(A3_PROTOTYPES)

    assert version == "semantic-prototypes-a3-v1"
    assert len(prototypes) == 28
    assert metadata["anchor_count"] == 28
    assert all(len(anchors) == 4 for anchors in metadata["dimensions"].values())


def test_a3_frozen_config_disables_holdout_tuning() -> None:
    _, config = load_frozen_config(A3_CONFIG)

    assert config["strategy"]["aggregation"] == "top2_mean"
    assert config["strategy"]["score_threshold"] == 0.425
    assert config["strategy"]["margin_threshold"] == 0.0
    assert config["fresh_holdout_gate"]["threshold_search_allowed"] is False
    assert config["fresh_holdout_gate"]["prototype_changes_allowed"] is False
    assert config["fresh_holdout_gate"]["router_changes_allowed"] is False


def test_original_a3_holdout_is_structurally_valid_but_not_provider_scored() -> None:
    development_raw, _, development_cases = load_fixture(V2_FIXTURE)
    config_raw, config = load_frozen_config(A3_CONFIG)
    strategy = config["strategy"]
    expected_seal = {
        "router_sha256": strategy["router_sha256"],
        "prototype_sha256": strategy["prototype_sha256"],
        "frozen_config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "development_fixture_sha256": hashlib.sha256(development_raw).hexdigest(),
        "holdout_runner_sha256": _sha256(A3_HOLDOUT_RUNNER),
    }

    raw, cases = load_fresh_holdout(
        A3_HOLDOUT,
        development_candidate_texts=_normalized_candidate_texts(development_cases),
        expected_seal=expected_seal,
    )

    assert hashlib.sha256(raw).hexdigest() == (
        "28c694b8cc79bc4666128940f6ca19b4f3778c05ece3a400c31c43b2e65c28a1"
    )
    assert len(cases) == 49
    assert sum(case.expected_action != "reuse_key" for case in cases) == 13
    assert provider_coverage(cases)["provider_scored_count"] == 0


def test_a31_manifest_requires_out_of_band_exact_hash(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "manifest_version": "semantic-router-a31-evaluation-manifest-v1",
                "files": {},
                "policy": {},
                "coverage": {},
            }
        )
    )
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    raw, _ = load_evaluation_manifest(path, expected_sha256=actual_sha)
    assert hashlib.sha256(raw).hexdigest() == actual_sha

    try:
        load_evaluation_manifest(path, expected_sha256="0" * 64)
    except ValueError as exc:
        assert "out-of-band" in str(exc)
    else:
        raise AssertionError("manifest accepted the wrong out-of-band SHA-256")


def test_a31_fixture_exercises_provider_across_all_dimensions() -> None:
    _, _, development_cases = load_fixture(V2_FIXTURE)
    raw, cases = load_provider_holdout(
        A31_HOLDOUT,
        development_candidate_texts=_normalized_candidate_texts(development_cases),
    )
    coverage = provider_coverage(cases)

    assert hashlib.sha256(raw).hexdigest() == (
        "7868c19ccbd09a7799ac8d0c220147242040223e17a81e9745bd76f748a2a635"
    )
    assert len(cases) == 49
    assert coverage["provider_scored_count"] == 47
    assert min(coverage["provider_scored_by_dimension"].values()) >= 6


def test_a31_config_freezes_provider_coverage_gate() -> None:
    _, config = load_frozen_config(A31_CONFIG)

    assert config["freeze_revision"] == "a3.1-provider-scored"
    assert config["strategy"]["router_sha256"] == (
        "dd1fe5f08c91779c08849f508d0abc2febf02c4319dbc2d353a3277840a326bf"
    )
    assert config["fresh_holdout_gate"]["minimum_provider_scored_count"] == 35
    assert config["fresh_holdout_gate"]["minimum_provider_scored_per_dimension"] == 4


def test_historical_sealed_sources_remain_recoverable() -> None:
    assert _sha256(SEALED_A3_ROUTER) == (
        "c90fa64f887e104943d66ef9b21e7b03ad1cb8212fd9c5d8c1e9d5ad335809fd"
    )
    assert _sha256(SEALED_A31_ROUTER) == (
        "dd1fe5f08c91779c08849f508d0abc2febf02c4319dbc2d353a3277840a326bf"
    )
    assert _sha256(SEALED_A3_DEVELOPMENT_RUNNER) == (
        "1a3230b85f585e04b2a3ca26c3036a54746df6d33aa2e72f02b9252c772af118"
    )


def test_a31_raw_text_guard_checks_values_before_json_escaping() -> None:
    raw = 'quoted "value" with newline\ncontrol'

    assert _contains_raw_text({"nested": [raw]}, (raw,)) is True
    assert _contains_raw_text({"nested": ["redacted"]}, (raw,)) is False


def test_development_gate_requires_safety_and_robustness() -> None:
    _, _, cases = load_fixture(V2_FIXTURE)
    recommendation, grid = evaluate_variant(
        "top2_mean",
        cases,
        _development_scores(cases),
        score_thresholds=[0.5],
        margin_thresholds=[0.0],
        minimum_recall=0.5,
    )

    assert len(grid) == 1
    assert recommendation is not None
    assert recommendation["false_merge_count"] == 0
    assert recommendation["safety_passed"] is True
    assert recommendation["robustness_passed"] is True


def test_high_confidence_development_collision_has_no_recommendation() -> None:
    _, _, cases = load_fixture(V2_FIXTURE)
    scores = _development_scores(cases)
    collision = next(case for case in cases if case.category == "lexical_collision")
    scores[collision.case_id] = DimensionScore(
        dimension=collision.expected_dimension or "beverage",
        score=1.0,
        second_dimension="other",
        second_score=0.0,
        margin=1.0,
        strategy_version=ENSEMBLE_STRATEGY_VERSION,
    )

    recommendation, _ = evaluate_variant(
        "max",
        cases,
        scores,
        score_thresholds=[0.5, 1.0],
        margin_thresholds=[0.0, 1.0],
        minimum_recall=0.5,
    )

    assert recommendation is None


def test_safety_gate_measures_no_cross_scope_merge_and_protected_feedback() -> None:
    _, _, cases = load_fixture(V2_FIXTURE)
    scores = _development_scores(cases)
    cross_channel = next(case for case in cases if case.category == "cross_channel_filter")
    explicit_feedback = next(
        case for case in cases if case.category == "explicit_feedback_protected"
    )
    scores[cross_channel.case_id] = DimensionScore(
        dimension="beverage",
        score=1.0,
        second_dimension="food",
        second_score=0.0,
        margin=1.0,
        strategy_version=ENSEMBLE_STRATEGY_VERSION,
    )
    scores[explicit_feedback.case_id] = DimensionScore(
        dimension=explicit_feedback.expected_dimension or "color",
        score=0.1,
        second_dimension="food",
        second_score=0.0,
        margin=0.1,
        strategy_version=ENSEMBLE_STRATEGY_VERSION,
    )

    recommendation, grid = evaluate_variant(
        "top2_mean",
        cases,
        scores,
        score_thresholds=[0.5],
        margin_thresholds=[0.2],
        minimum_recall=0.5,
    )

    assert grid[0]["safety_passed"] is True
    assert grid[0]["safety_failure_ids"] == []
    assert recommendation is not None


def test_selection_prefers_top2_mean_only_after_metrics_tie() -> None:
    safe_max = {
        "recall": 0.9,
        "accuracy": 0.95,
        "score_threshold": 0.6,
        "margin_threshold": 0.1,
    }
    safe_top2 = dict(safe_max)

    selected = select_variant(
        [
            {**safe_max, "aggregation": "mean"},
            {**safe_top2, "aggregation": "top2_mean"},
            {**safe_max, "aggregation": "max"},
        ]
    )

    assert selected is not None
    assert selected["aggregation"] == "top2_mean"

    higher_accuracy = select_variant(
        [
            {**safe_max, "aggregation": "mean"},
            {**safe_max, "aggregation": "top2_mean", "accuracy": 0.9},
        ]
    )
    assert higher_accuracy is not None
    assert higher_accuracy["aggregation"] == "mean"


def test_aggregation_list_is_strict() -> None:
    assert parse_aggregations("top2_mean,mean") == ["top2_mean", "mean"]

    for invalid in ("", "mean,mean", "median"):
        try:
            parse_aggregations(invalid)
        except ValueError:
            continue
        raise AssertionError(f"invalid aggregation list accepted: {invalid}")
