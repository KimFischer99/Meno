from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import benchmarks.run_semantic_router_a5_development as a5_runner
from benchmarks.run_semantic_router_a3_development import load_prototypes
from benchmarks.run_semantic_router_a5_development import (
    DEFAULT_PROTOTYPES as DEFAULT_A5_PROTOTYPES,
)
from benchmarks.run_semantic_router_a5_development import (
    EXPECTED_PROTOTYPE_SHA256,
    evaluate_point,
    load_a5_development,
    load_prototypes_a5,
)
from meno.semantic_router import ENSEMBLE_STRATEGY_VERSION, DimensionScore, PolicyRouter
from tests.fakes import TestEmbedder

ROOT = Path(__file__).parent.parent
A4_PROTOTYPES = ROOT / "benchmarks" / "fixtures" / "semantic-prototypes-a3-v1.json"


def test_a5_prototypes_fixture_is_frozen_and_structurally_valid() -> None:
    raw, version, prototypes, metadata = load_prototypes_a5(DEFAULT_A5_PROTOTYPES)

    assert version == "semantic-prototypes-a5-v2"
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_PROTOTYPE_SHA256
    assert len(prototypes) == 28
    assert metadata["anchor_count"] == 28
    for anchors in metadata["dimensions"].values():
        assert [anchor["id"] for anchor in anchors] == [
            "topic",
            "affirmed",
            "negated",
            "contradiction",
        ]
    assert all(prototype.description.strip() for prototype in prototypes)
    # A5 v2 anchors are English-only by design.
    assert all(prototype.description.isascii() for prototype in prototypes)
    assert all(prototype.polarity is not None for prototype in prototypes)


def test_a5_prototypes_reject_tampered_fixtures(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_A5_PROTOTYPES.read_bytes())

    wrong_version = dict(payload, prototype_version="semantic-prototypes-a5-v1")
    path = tmp_path / "wrong-version.json"
    path.write_text(json.dumps(wrong_version))
    with pytest.raises(ValueError, match="version"):
        load_prototypes_a5(path)

    missing_anchor = json.loads(json.dumps(payload))
    missing_anchor["dimensions"]["beverage"] = missing_anchor["dimensions"]["beverage"][:3]
    path = tmp_path / "missing-anchor.json"
    path.write_text(json.dumps(missing_anchor))
    with pytest.raises(ValueError, match="four anchors"):
        load_prototypes_a5(path)

    swapped_role = json.loads(json.dumps(payload))
    swapped_role["dimensions"]["color"][0]["role"] = "polarity"
    path = tmp_path / "swapped-role.json"
    path.write_text(json.dumps(swapped_role))
    with pytest.raises(ValueError, match="must declare"):
        load_prototypes_a5(path)

    # The sealed A3 prototype fixture must be untouched by the A5 lane.
    _, a3_version, _, _ = load_prototypes(A4_PROTOTYPES)
    assert a3_version == "semantic-prototypes-a3-v1"
    assert hashlib.sha256(A4_PROTOTYPES.read_bytes()).hexdigest() == (
        "8e7a8e2e1b2c2f608d72c78289236e2883303039b5957bba343403f27fcd8fbe"
    )


def test_a5_development_loads_the_pinned_111_cases() -> None:
    raw, cases, sources = load_a5_development()

    assert len(cases) == 111
    assert len(sources["v2"]) == 62
    assert len(sources["a31"]) == 49
    assert set(raw) == {"v2_fixture", "a31_fixture", "a31_artifact", "a31_manifest"}


def test_a5_evaluate_point_enforces_all_frozen_conditions() -> None:
    _, cases, sources = load_a5_development()
    router = PolicyRouter(strategy_version=ENSEMBLE_STRATEGY_VERSION)

    def decisions_for(score_override=None):  # type: ignore[no-untyped-def]
        decisions = {}
        for case in cases:
            score = score_override(case) if score_override is not None else _oracle_score(case)
            decisions[case.case_id] = router.decide(
                case.candidate,
                case.references,
                score,
                0.5,
                0.2,
            )
        return decisions

    evaluation = evaluate_point(
        cases,
        sources,
        decisions_for(),
        minimum_overall_recall=0.75,
        minimum_source_recall=0.75,
    )
    assert evaluation["eligible"] is True
    assert evaluation["false_merge_count"] == 0
    assert evaluation["safety_passed"] is True
    assert evaluation["robustness_passed"] is True
    assert evaluation["a31_answer_style_failure_ids"] == []

    def collision_score(case):  # type: ignore[no-untyped-def]
        if case.category != "lexical_collision":
            return _oracle_score(case)
        return DimensionScore(
            dimension=case.expected_dimension or "beverage",
            score=1.0,
            second_dimension="other",
            second_score=0.0,
            margin=1.0,
            strategy_version=ENSEMBLE_STRATEGY_VERSION,
        )

    collided = evaluate_point(
        cases,
        sources,
        decisions_for(collision_score),
        minimum_overall_recall=0.75,
        minimum_source_recall=0.75,
    )
    assert collided["eligible"] is False
    assert collided["false_merge_count"] >= 1


def _oracle_score(case):  # type: ignore[no-untyped-def]
    if case.candidate.deterministic_slot is not None or case.expected_dimension is None:
        return None
    low_confidence = case.category in {"lexical_collision", "entity_collision"}
    score = 0.1 if low_confidence else 1.0
    return DimensionScore(
        dimension=case.expected_dimension,
        score=score,
        second_dimension="other",
        second_score=0.0,
        margin=score,
        strategy_version=ENSEMBLE_STRATEGY_VERSION,
    )


def test_a5_main_writes_artifact_with_deterministic_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(a5_runner, "make_embedder", lambda _settings: TestEmbedder(dimension=64))
    output = tmp_path / "a5-development.json"

    def run() -> None:
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_semantic_router_a5_development.py",
                "--output",
                str(output),
                "--score-thresholds",
                "0.0,0.5",
                "--margin-thresholds",
                "0.0,0.05",
                "--runs",
                "2",
            ],
        )
        a5_runner.main()

    try:
        run()
        exit_code = 0
    except SystemExit as excinfo:
        exit_code = excinfo.code
    assert exit_code in {0, 3}

    report = json.loads(output.read_text())
    assert report["prototypes"]["sha256"] == EXPECTED_PROTOTYPE_SHA256
    assert report["provider_policy"]["coverage_passed"] is True
    assert report["provider_policy"]["runs_per_variant"] == 2
    assert set(report["variants"]) == {"top2_mean", "topic_gated_v1"}
    assert report["gate"]["failure_reason"] in {None, "no_stable_policy"}
    if report["gate"]["passed"]:
        assert report["selected"] is not None
        assert len(report["selected_run_evaluations"]) == 2
        for evaluation in report["selected_run_evaluations"]:
            assert evaluation["eligible"] is True
            assert evaluation["false_merge_count"] == 0
