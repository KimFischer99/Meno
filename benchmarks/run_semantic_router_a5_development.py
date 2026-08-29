from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import socket
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.run_semantic_router_a4_development import (
    _contains_raw_artifact_text,
    load_a4_development,
    provider_coverage,
)
from benchmarks.run_semantic_router_shadow import (
    DEFAULT_SCORE_THRESHOLDS,
    ROBUSTNESS_CATEGORIES,
    FixtureCase,
    _case_correct,
    _git_head,
    _package_versions,
    _safe_endpoint,
    _sha256,
    decision_metrics,
    parse_thresholds,
)
from meno.config import Settings
from meno.semantic_router import (
    A5_STRATEGY_VERSION,
    ENSEMBLE_STRATEGY_VERSION,
    ROUTER_VERSION,
    DimensionPrototype,
    DimensionScore,
    PolicyRouter,
    PrototypeEnsembleStrategy,
    RouterDecision,
    TopicGatedStrategy,
)
from meno.vector import make_embedder

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DEPENDENCIES = {
    "a4_development": ROOT / "benchmarks" / "run_semantic_router_a4_development.py",
    "shadow_runner": ROOT / "benchmarks" / "run_semantic_router_shadow.py",
    "config": ROOT / "src" / "meno" / "config.py",
    "extractor": ROOT / "src" / "meno" / "extractor.py",
    "vector": ROOT / "src" / "meno" / "vector.py",
}
DEFAULT_V2_FIXTURE = ROOT / "benchmarks" / "fixtures" / "semantic-normalization-v2.json"
DEFAULT_A31_FIXTURE = (
    ROOT / "benchmarks" / "fixtures" / "semantic-normalization-v4-provider-holdout.json"
)
DEFAULT_A31_ARTIFACT = ROOT / "artifacts" / "vps-stage4-a31-provider-holdout.json"
DEFAULT_A31_MANIFEST = (
    ROOT / "benchmarks" / "fixtures" / "semantic-router-a31-evaluation-manifest.json"
)
DEFAULT_PROTOTYPES = ROOT / "benchmarks" / "fixtures" / "semantic-prototypes-a5-v2.json"
EXPECTED_INPUT_SHA256 = {
    "v2_fixture": "908d9f21b184fc0ce9fe273be91890b0fc402ab34cf9cdaa1ca630eeba7305d7",
    "a31_fixture": "7868c19ccbd09a7799ac8d0c220147242040223e17a81e9745bd76f748a2a635",
    "a31_artifact": "073a14779f3980d8139e88ce6dea137340ac5b65cb3586491f47f8697d3253ed",
    "a31_manifest": "359bce87535d7b6117b042529e081fef8a9927f744b069280ac712c4402d2c36",
}
EXPECTED_PROTOTYPE_SHA256 = "d9e97ec67b5688b44728a1b2bcf73cc39d7625ef1a7baf39b83862cae7a136dc"
PROTOTYPE_VERSION = "semantic-prototypes-a5-v2"

A5_RUN_COUNT = 3
VARIANTS: tuple[tuple[str, str], ...] = (
    ("top2_mean", ENSEMBLE_STRATEGY_VERSION),
    ("topic_gated_v1", A5_STRATEGY_VERSION),
)
AGGREGATION_PREFERENCE = {"top2_mean": 0, "topic_gated_v1": 1}
DEFAULT_MARGIN_THRESHOLDS = tuple(
    sorted(
        {round(index * 0.005, 3) for index in range(21)}
        | {round(index * 0.025, 3) for index in range(5, 41)}
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select an A5 policy on the revealed 111-case development set with a "
            f"{A5_RUN_COUNT}-independent-provider-run stability gate"
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--score-thresholds",
        default=",".join(str(value) for value in DEFAULT_SCORE_THRESHOLDS),
    )
    parser.add_argument(
        "--margin-thresholds",
        default=",".join(str(value) for value in DEFAULT_MARGIN_THRESHOLDS),
    )
    parser.add_argument("--minimum-overall-recall", type=float, default=0.75)
    parser.add_argument("--minimum-source-recall", type=float, default=0.75)
    parser.add_argument("--runs", type=int, default=A5_RUN_COUNT)
    return parser.parse_args()


def _validated_rate(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


def load_a5_development() -> tuple[dict[str, bytes], tuple[FixtureCase, ...], dict[str, tuple[FixtureCase, ...]]]:
    """Load the pinned 111-case revealed development set (v2 + revealed A3.1)."""

    development_raw, cases, sources = load_a4_development(
        DEFAULT_V2_FIXTURE, DEFAULT_A31_FIXTURE, DEFAULT_A31_ARTIFACT, DEFAULT_A31_MANIFEST
    )
    for name in ("v2_fixture", "a31_fixture", "a31_artifact", "a31_manifest"):
        expected = EXPECTED_INPUT_SHA256[name]
        actual = hashlib.sha256(development_raw[name]).hexdigest()
        if expected != actual:
            raise ValueError(f"A5 development input changed: {name}")
    return development_raw, cases, sources


def load_prototypes_a5(
    path: Path,
) -> tuple[bytes, str, tuple[DimensionPrototype, ...], dict[str, Any]]:
    """Load and structurally validate the frozen A5 prototype fixture."""

    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("prototype fixture root must be an object")
    version = payload.get("prototype_version")
    if version != PROTOTYPE_VERSION:
        raise ValueError("unsupported A5 prototype version")
    dimensions = payload.get("dimensions")
    if not isinstance(dimensions, dict):
        raise TypeError("prototype fixture dimensions must be an object")

    expected_dimensions = {
        "answer_style",
        "beverage",
        "color",
        "food",
        "music",
        "programming_language",
        "sport",
    }
    if set(dimensions) != expected_dimensions:
        raise ValueError("A5 prototypes must contain exactly the seven controlled dimensions")

    prototypes: list[DimensionPrototype] = []
    metadata: dict[str, Any] = {"design": payload.get("design"), "dimensions": {}, "anchor_count": 0}
    for dimension in sorted(dimensions):
        anchors = dimensions[dimension]
        if not isinstance(anchors, list) or len(anchors) != 4:
            raise ValueError(
                f"prototype dimension {dimension} requires exactly four anchors"
            )
        seen_ids: set[str] = set()
        dimension_metadata: list[dict[str, str]] = []
        for index, anchor in enumerate(anchors):
            context = f"dimensions.{dimension}[{index}]"
            if not isinstance(anchor, dict):
                raise TypeError(f"{context} must be an object")
            anchor_id = anchor.get("id")
            role = anchor.get("role")
            polarity = anchor.get("polarity")
            text = anchor.get("text")
            if any(not isinstance(value, str) or not value.strip() for value in (anchor_id, role, polarity, text)):
                raise ValueError(f"{context} fields must be non-empty strings")
            if anchor_id in seen_ids:
                raise ValueError(f"duplicate anchor id in dimension {dimension}: {anchor_id}")
            seen_ids.add(anchor_id)
            expected_role = "topic" if anchor_id == "topic" else "polarity"
            expected_polarity = "neutral" if anchor_id == "topic" else anchor_id
            if role != expected_role or polarity != expected_polarity:
                raise ValueError(
                    f"{context} must declare id={anchor_id}, role={expected_role}, "
                    f"polarity={expected_polarity}"
                )
            prototypes.append(DimensionPrototype(name=dimension, description=text, role=role, polarity=polarity))
            dimension_metadata.append({"id": anchor_id, "role": role, "polarity": polarity})
        metadata["dimensions"][dimension] = dimension_metadata
        metadata["anchor_count"] += len(dimension_metadata)
    return raw, version, tuple(prototypes), metadata


def score_variant_run(
    cases: tuple[FixtureCase, ...],
    prototypes: tuple[DimensionPrototype, ...],
    settings: Settings,
    *,
    variant: str,
) -> dict[str, DimensionScore | None]:
    """One independent provider scoring pass for one strategy variant."""

    score_cases = [case for case in cases if case.candidate.deterministic_slot is None]
    embedder = make_embedder(settings)
    try:
        if variant == "top2_mean":
            values = PrototypeEnsembleStrategy(
                embedder, prototypes, aggregation="top2_mean"
            ).score_many([case.candidate for case in score_cases])
        elif variant == "topic_gated_v1":
            values = TopicGatedStrategy(embedder, prototypes).score_many(
                [case.candidate for case in score_cases]
            )
        else:
            raise ValueError(f"unsupported A5 variant: {variant}")
    finally:
        embedder.close()
    result: dict[str, DimensionScore | None] = {case.case_id: None for case in cases}
    result.update({case.case_id: score for case, score in zip(score_cases, values, strict=True)})
    return result


def route_with_variant(
    cases: tuple[FixtureCase, ...],
    scores: dict[str, DimensionScore | None],
    *,
    strategy_version: str,
    score_threshold: float,
    margin_threshold: float,
) -> dict[str, RouterDecision]:
    router = PolicyRouter(strategy_version=strategy_version)
    return {
        case.case_id: router.decide(
            case.candidate,
            case.references,
            scores[case.case_id],
            score_threshold,
            margin_threshold,
        )
        for case in cases
    }


def _source_metrics(
    sources: dict[str, tuple[FixtureCase, ...]],
    decisions: dict[str, RouterDecision],
) -> dict[str, dict[str, Any]]:
    return {
        name: decision_metrics(cases, {case.case_id: decisions[case.case_id] for case in cases})
        for name, cases in sources.items()
    }


def evaluate_point(
    cases: tuple[FixtureCase, ...],
    sources: dict[str, tuple[FixtureCase, ...]],
    decisions: dict[str, RouterDecision],
    *,
    minimum_overall_recall: float,
    minimum_source_recall: float,
) -> dict[str, Any]:
    metrics = decision_metrics(cases, decisions)
    source_metrics = _source_metrics(sources, decisions)
    safety_passed, safety_failures = _safety_gate_local(cases, decisions)
    robustness_passed, robustness_failures = _category_gate_local(cases, decisions)
    a31_answer_style_ids = {
        case.case_id for case in sources["a31"] if case.expected_dimension == "answer_style"
    }
    answer_style_failures = sorted(
        case.case_id
        for case in sources["a31"]
        if case.case_id in a31_answer_style_ids
        and not _case_correct(case, decisions[case.case_id])
    )
    eligible = bool(
        metrics["false_merge_count"] == 0
        and all(item["false_merge_count"] == 0 for item in source_metrics.values())
        and metrics["recall"] >= minimum_overall_recall
        and all(item["recall"] >= minimum_source_recall for item in source_metrics.values())
        and safety_passed
        and robustness_passed
        and not answer_style_failures
    )
    return {
        "false_merge_count": metrics["false_merge_count"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "accuracy": metrics["accuracy"],
        "missed_merge_ids": metrics["missed_merge_ids"],
        "false_merge_ids": metrics["false_merge_ids"],
        "source_recall": {name: item["recall"] for name, item in source_metrics.items()},
        "source_false_merge_count": {
            name: item["false_merge_count"] for name, item in source_metrics.items()
        },
        "safety_passed": safety_passed,
        "safety_failure_ids": safety_failures,
        "robustness_passed": robustness_passed,
        "robustness_failure_ids": robustness_failures,
        "a31_answer_style_failure_ids": answer_style_failures,
        "eligible": eligible,
    }


def _safety_gate_local(
    cases: tuple[FixtureCase, ...],
    decisions: dict[str, RouterDecision],
) -> tuple[bool, list[str]]:
    from benchmarks.run_semantic_router_a3_development import _safety_case_correct
    from benchmarks.run_semantic_router_shadow import SAFETY_CATEGORIES

    failures = [
        case.case_id
        for case in cases
        if case.category in SAFETY_CATEGORIES
        and not _safety_case_correct(case, decisions[case.case_id])
    ]
    present = {case.category for case in cases}
    return not failures and SAFETY_CATEGORIES <= present, failures


def _category_gate_local(
    cases: tuple[FixtureCase, ...],
    decisions: dict[str, RouterDecision],
) -> tuple[bool, list[str]]:
    from benchmarks.run_semantic_router_a3_development import _category_gate as gate

    return gate(cases, decisions, ROBUSTNESS_CATEGORIES)


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("runs must be a positive integer")
    minimum_overall_recall = _validated_rate(args.minimum_overall_recall, name="minimum overall recall")
    minimum_source_recall = _validated_rate(args.minimum_source_recall, name="minimum source recall")
    score_thresholds = parse_thresholds(args.score_thresholds, name="score thresholds")
    margin_thresholds = parse_thresholds(args.margin_thresholds, name="margin thresholds")

    development_raw, cases, sources = load_a5_development()

    prototype_raw, prototype_version, prototypes, prototype_metadata = load_prototypes_a5(
        DEFAULT_PROTOTYPES
    )
    if hashlib.sha256(prototype_raw).hexdigest() != EXPECTED_PROTOTYPE_SHA256:
        raise ValueError("A5 development prototype input changed")

    settings = Settings.from_env()
    strategy_version_by_variant = dict(VARIANTS)

    # One independent provider pass per (variant, run).
    runs_scores: dict[str, list[dict[str, DimensionScore | None]]] = {}
    coverage_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant, _version in VARIANTS:
        runs_scores[variant] = []
        coverage_by_variant[variant] = []
        for _run in range(args.runs):
            scores = score_variant_run(cases, prototypes, settings, variant=variant)
            runs_scores[variant].append(scores)
            coverage_by_variant[variant].append(provider_coverage(cases, scores))
    coverage_passed = all(
        coverage["passed"] for items in coverage_by_variant.values() for coverage in items
    )

    variant_results: dict[str, Any] = {}
    recommendations: list[dict[str, Any]] = []
    for variant, _version in VARIANTS:
        stable_points: list[dict[str, Any]] = []
        per_point_stable_metrics: dict[tuple[float, float], list[dict[str, Any]]] = {}
        eligible_counts = 0
        for score_threshold in score_thresholds:
            for margin_threshold in margin_thresholds:
                run_evaluations = []
                for scores in runs_scores[variant]:
                    decisions = route_with_variant(
                        cases,
                        scores,
                        strategy_version=strategy_version_by_variant[variant],
                        score_threshold=score_threshold,
                        margin_threshold=margin_threshold,
                    )
                    run_evaluations.append(
                        evaluate_point(
                            cases,
                            sources,
                            decisions,
                            minimum_overall_recall=minimum_overall_recall,
                            minimum_source_recall=minimum_source_recall,
                        )
                    )
                stable = all(item["eligible"] for item in run_evaluations)
                eligible_counts += sum(item["eligible"] for item in run_evaluations)
                if stable:
                    stable_points.append((score_threshold, margin_threshold))
                    per_point_stable_metrics[(score_threshold, margin_threshold)] = run_evaluations
        selected_point = None
        if stable_points:
            ranked_points = sorted(
                (
                    (
                        min(item["recall"] for item in per_point_stable_metrics[point]),
                        min(
                            min(item["source_recall"].values())
                            for item in per_point_stable_metrics[point]
                        ),
                        min(item["accuracy"] for item in per_point_stable_metrics[point]),
                        point[1],
                        point[0],
                        point,
                    )
                    for point in stable_points
                ),
            )
            selected_point = ranked_points[-1][-1]
        variant_results[variant] = {
            "strategy_version": strategy_version_by_variant[variant],
            "provider_coverage": coverage_by_variant[variant],
            "stable_eligible_points": [
                {"score_threshold": st, "margin_threshold": mt} for st, mt in stable_points
            ],
            "selected": (
                {
                    "variant": variant,
                    "score_threshold": selected_point[0],
                    "margin_threshold": selected_point[1],
                }
                if selected_point is not None
                else None
            ),
            "selected_run_evaluations": (
                per_point_stable_metrics[selected_point] if selected_point is not None else None
            ),
        }
        if selected_point is not None:
            evaluations = per_point_stable_metrics[selected_point]
            recommendations.append(
                {
                    "variant": variant,
                    "aggregation_preference": AGGREGATION_PREFERENCE[variant],
                    "score_threshold": selected_point[0],
                    "margin_threshold": selected_point[1],
                    "recall": min(item["recall"] for item in evaluations),
                    "min_source_recall": min(
                        min(item["source_recall"].values()) for item in evaluations
                    ),
                    "accuracy": min(item["accuracy"] for item in evaluations),
                }
            )

    selected = None
    if recommendations:
        selected = max(
            recommendations,
            key=lambda item: (
                item["recall"],
                item["min_source_recall"],
                item["accuracy"],
                -item["aggregation_preference"],
                item["margin_threshold"],
                item["score_threshold"],
            ),
        )

    runner_path = Path(__file__).resolve()
    router_path = ROOT / "src" / "meno" / "semantic_router.py"
    provider_endpoint = (
        settings.google_api_base_url
        if settings.embedding_provider == "google"
        else settings.openai_base_url
    )
    report: dict[str, Any] = {
        "benchmark": "Meno semantic policy router A5 development selection",
        "mode": "external-provider-development-read-only",
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "git_head": _git_head(),
        "command": [str(runner_path), *sys.argv[1:]],
        "development": {
            "case_count": len(cases),
            "fresh_holdout_used": False,
            "sources": {
                "v2": {
                    "case_count": len(sources["v2"]),
                    "sha256": hashlib.sha256(development_raw["v2_fixture"]).hexdigest(),
                },
                "a31_revealed": {
                    "case_count": len(sources["a31"]),
                    "fixture_sha256": hashlib.sha256(development_raw["a31_fixture"]).hexdigest(),
                    "artifact_sha256": hashlib.sha256(development_raw["a31_artifact"]).hexdigest(),
                    "manifest_sha256": hashlib.sha256(development_raw["a31_manifest"]).hexdigest(),
                },
            },
        },
        "prototypes": {
            "version": prototype_version,
            "sha256": hashlib.sha256(prototype_raw).hexdigest(),
            "expected_sha256": EXPECTED_PROTOTYPE_SHA256,
            **prototype_metadata,
        },
        "source": {
            "runner_sha256": _sha256(runner_path),
            "router_sha256": _sha256(router_path),
            "strategy_versions": strategy_version_by_variant,
            "router_version": ROUTER_VERSION,
            "dependency_sha256": {
                name: _sha256(path) for name, path in SOURCE_DEPENDENCIES.items()
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "service_config": {
            "environment": settings.environment,
            "extractor_version": settings.extractor_version,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "embedding_projection_version": settings.embedding_projection_version,
            "embedding_endpoint": _safe_endpoint(provider_endpoint),
        },
        "provider_policy": {
            "coverage_required": True,
            "coverage_passed": coverage_passed,
            "designed_provider_cases": 92,
            "runs_per_variant": args.runs,
            "by_variant": coverage_by_variant,
        },
        "selection_policy": {
            "stability_runs_required": args.runs,
            "false_merge_count": 0,
            "minimum_overall_recall": minimum_overall_recall,
            "minimum_source_recall": minimum_source_recall,
            "safety_all_correct": True,
            "robustness_all_correct": True,
            "a31_answer_style_all_correct": True,
            "ordering": "min recall, min source recall, min accuracy, margin, score",
        },
        "variants": {
            name: {
                key: value
                for key, value in result.items()
                if key != "selected_run_evaluations"
            }
            for name, result in variant_results.items()
        },
        "recommendations": recommendations,
        "selected": selected,
        "selected_run_evaluations": (
            variant_results[selected["variant"]]["selected_run_evaluations"]
            if selected is not None
            else None
        ),
        "gate": {
            "passed": coverage_passed and selected is not None,
            "failure_reason": (
                None
                if selected is not None
                else ("provider_coverage_failed" if not coverage_passed else "no_stable_policy")
            ),
        },
    }
    raw_values = tuple(
        value
        for case in cases
        for value in (
            case.candidate.value,
            *(reference.value for reference in case.references),
        )
    )
    if _contains_raw_artifact_text(report, raw_values):
        raise ValueError("A5 development artifact would contain raw candidate or reference text")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "selected": selected,
                "gate": report["gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not coverage_passed:
        raise SystemExit(4)
    if selected is None:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
