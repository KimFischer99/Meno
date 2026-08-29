from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import socket
import sys
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.run_semantic_router_a3_development import (
    _category_gate,
    _safety_gate,
    _score_development,
    load_prototypes,
)
from benchmarks.run_semantic_router_a3_holdout import _normalized_candidate_texts
from benchmarks.run_semantic_router_a31_holdout import (
    load_provider_holdout,
)
from benchmarks.run_semantic_router_shadow import (
    DEFAULT_SCORE_THRESHOLDS,
    EXPECTED_DIMENSIONS,
    ROBUSTNESS_CATEGORIES,
    FixtureCase,
    _case_correct,
    _git_head,
    _package_versions,
    _safe_endpoint,
    _sha256,
    decision_metrics,
    load_fixture,
    parse_thresholds,
)
from meno.config import Settings
from meno.semantic_router import (
    A4_STRATEGY_VERSION,
    ENSEMBLE_STRATEGY_VERSION,
    ROUTER_VERSION,
    DimensionPrototype,
    DimensionScore,
    EnsembleAggregation,
    PolicyRouter,
    RouterDecision,
)

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DEPENDENCIES = {
    "a3_development": ROOT / "benchmarks" / "run_semantic_router_a3_development.py",
    "a3_holdout": ROOT / "benchmarks" / "run_semantic_router_a3_holdout.py",
    "a31_holdout": ROOT / "benchmarks" / "run_semantic_router_a31_holdout.py",
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
DEFAULT_PROTOTYPES = ROOT / "benchmarks" / "fixtures" / "semantic-prototypes-a3-v1.json"
EXPECTED_INPUT_SHA256 = {
    "v2_fixture": "908d9f21b184fc0ce9fe273be91890b0fc402ab34cf9cdaa1ca630eeba7305d7",
    "a31_fixture": "7868c19ccbd09a7799ac8d0c220147242040223e17a81e9745bd76f748a2a635",
    "a31_artifact": "073a14779f3980d8139e88ce6dea137340ac5b65cb3586491f47f8697d3253ed",
    "a31_manifest": "359bce87535d7b6117b042529e081fef8a9927f744b069280ac712c4402d2c36",
}
EXPECTED_PROTOTYPE_SHA256 = "8e7a8e2e1b2c2f608d72c78289236e2883303039b5957bba343403f27fcd8fbe"
A4_AGGREGATIONS: tuple[EnsembleAggregation, ...] = ("top2_mean", "role_top2_mean")
AGGREGATION_PREFERENCE = {"top2_mean": 0, "role_top2_mean": 1}
STRATEGY_VERSION_BY_AGGREGATION = {
    "top2_mean": ENSEMBLE_STRATEGY_VERSION,
    "role_top2_mean": A4_STRATEGY_VERSION,
}
DEFAULT_MARGIN_THRESHOLDS = tuple(
    sorted(
        {round(index * 0.005, 3) for index in range(21)}
        | {round(index * 0.025, 3) for index in range(5, 41)}
    )
)


def _contains_raw_artifact_text(value: object, raw_values: tuple[str, ...]) -> bool:
    """Reject exact raw fields and longer raw text embedded in report strings."""
    if isinstance(value, str):
        return value in raw_values or any(
            len(raw_value) >= 8 and raw_value in value for raw_value in raw_values
        )
    if isinstance(value, dict):
        return any(
            _contains_raw_artifact_text(key, raw_values)
            or _contains_raw_artifact_text(item, raw_values)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_raw_artifact_text(item, raw_values) for item in value)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select an A4 role-aware policy on pooled revealed development evidence"
    )
    parser.add_argument("--v2-fixtures", type=Path, default=DEFAULT_V2_FIXTURE)
    parser.add_argument("--a31-fixtures", type=Path, default=DEFAULT_A31_FIXTURE)
    parser.add_argument("--a31-artifact", type=Path, default=DEFAULT_A31_ARTIFACT)
    parser.add_argument("--a31-manifest", type=Path, default=DEFAULT_A31_MANIFEST)
    parser.add_argument("--prototypes", type=Path, default=DEFAULT_PROTOTYPES)
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
    return parser.parse_args()


def _validated_rate(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


def role_annotated_prototypes(
    prototypes: tuple[DimensionPrototype, ...],
    metadata: dict[str, Any],
) -> tuple[DimensionPrototype, ...]:
    annotated: list[DimensionPrototype] = []
    prototype_index = 0
    dimensions = metadata.get("dimensions")
    if not isinstance(dimensions, dict):
        raise TypeError("prototype metadata dimensions must be an object")
    for dimension in sorted(dimensions):
        anchors = dimensions[dimension]
        if not isinstance(anchors, list):
            raise TypeError("prototype metadata anchors must be a list")
        for anchor in anchors:
            if prototype_index >= len(prototypes) or not isinstance(anchor, dict):
                raise ValueError("prototype metadata does not align with prototypes")
            prototype = prototypes[prototype_index]
            role = anchor.get("role")
            if prototype.name != dimension or role not in {"topic", "polarity"}:
                raise ValueError("prototype role metadata does not align with prototypes")
            annotated.append(
                DimensionPrototype(
                    name=prototype.name,
                    description=prototype.description,
                    role=role,
                )
            )
            prototype_index += 1
    if prototype_index != len(prototypes):
        raise ValueError("prototype metadata does not cover every prototype")
    return tuple(annotated)


def load_a4_development(
    v2_path: Path,
    a31_path: Path,
    a31_artifact_path: Path,
    a31_manifest_path: Path,
) -> tuple[dict[str, bytes], tuple[FixtureCase, ...], dict[str, tuple[FixtureCase, ...]]]:
    paths = {
        "v2_fixture": v2_path,
        "a31_fixture": a31_path,
        "a31_artifact": a31_artifact_path,
        "a31_manifest": a31_manifest_path,
    }
    raw = {name: path.read_bytes() for name, path in paths.items()}
    for name, expected_sha in EXPECTED_INPUT_SHA256.items():
        if hashlib.sha256(raw[name]).hexdigest() != expected_sha:
            raise ValueError(f"A4 development input changed: {name}")

    _, _, v2_cases = load_fixture(v2_path)
    _, a31_cases = load_provider_holdout(
        a31_path,
        development_candidate_texts=_normalized_candidate_texts(v2_cases),
    )
    artifact = json.loads(raw["a31_artifact"])
    if not isinstance(artifact, dict):
        raise TypeError("A3.1 artifact root must be an object")
    if artifact.get("run_id") != "2ca272c4-1fba-4d5d-bca7-5ac574c1c53b":
        raise ValueError("unexpected A3.1 development artifact run")
    if artifact.get("passed") is not False:
        raise ValueError("A4 expects the revealed A3.1 NO-GO artifact")
    if artifact.get("holdout", {}).get("fixture_sha256") != EXPECTED_INPUT_SHA256["a31_fixture"]:
        raise ValueError("A3.1 artifact is not bound to the A4 development fixture")
    if len({case.case_id for case in (*v2_cases, *a31_cases)}) != len(v2_cases) + len(a31_cases):
        raise ValueError("A4 development case ids must be unique")
    sources = {"v2": v2_cases, "a31": a31_cases}
    return raw, (*v2_cases, *a31_cases), sources


def _source_metrics(
    sources: dict[str, tuple[FixtureCase, ...]],
    decisions: dict[str, RouterDecision],
) -> dict[str, dict[str, Any]]:
    return {
        name: decision_metrics(cases, {case.case_id: decisions[case.case_id] for case in cases})
        for name, cases in sources.items()
    }


def provider_coverage(
    cases: tuple[FixtureCase, ...],
    scores: dict[str, DimensionScore | None],
) -> dict[str, Any]:
    designed = tuple(
        case
        for case in cases
        if (
            case.candidate.deterministic_slot is None
            and not case.candidate.sensitive
            and not case.candidate.injection_detected
            and case.expected_dimension in EXPECTED_DIMENSIONS
        )
    )
    designed_by_dimension = Counter(case.expected_dimension for case in designed)
    returned = tuple(case for case in designed if scores[case.case_id] is not None)
    returned_by_dimension = Counter(case.expected_dimension for case in returned)
    expected_by_dimension = {
        dimension: designed_by_dimension.get(dimension, 0)
        for dimension in sorted(EXPECTED_DIMENSIONS)
    }
    actual_by_dimension = {
        dimension: returned_by_dimension.get(dimension, 0)
        for dimension in sorted(EXPECTED_DIMENSIONS)
    }
    return {
        "provider_scored_count": len(designed),
        "provider_scored_by_dimension": expected_by_dimension,
        "provider_score_returned_count": len(returned),
        "provider_score_returned_by_dimension": actual_by_dimension,
        "passed": len(returned) == len(designed) and actual_by_dimension == expected_by_dimension,
    }


def route_a4_development(
    cases: tuple[FixtureCase, ...],
    scores: dict[str, DimensionScore | None],
    *,
    aggregation: EnsembleAggregation,
    score_threshold: float,
    margin_threshold: float,
) -> dict[str, RouterDecision]:
    try:
        strategy_version = STRATEGY_VERSION_BY_AGGREGATION[aggregation]
    except KeyError as exc:
        raise ValueError("A4 development aggregation is unsupported") from exc
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


def evaluate_a4_variant(
    aggregation: str,
    cases: tuple[FixtureCase, ...],
    sources: dict[str, tuple[FixtureCase, ...]],
    scores: dict[str, DimensionScore | None],
    score_thresholds: list[float],
    margin_thresholds: list[float],
    *,
    minimum_overall_recall: float,
    minimum_source_recall: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    grid: list[dict[str, Any]] = []
    a31_answer_style_ids = {
        case.case_id for case in sources["a31"] if case.expected_dimension == "answer_style"
    }
    for score_threshold in score_thresholds:
        for margin_threshold in margin_thresholds:
            decisions = route_a4_development(
                cases,
                scores,
                aggregation=aggregation,
                score_threshold=score_threshold,
                margin_threshold=margin_threshold,
            )
            metrics = decision_metrics(cases, decisions)
            source_metrics = _source_metrics(sources, decisions)
            safety_passed, safety_failures = _safety_gate(cases, decisions)
            robustness_passed, robustness_failures = _category_gate(
                cases, decisions, ROBUSTNESS_CATEGORIES
            )
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
            grid.append(
                {
                    "aggregation": aggregation,
                    "score_threshold": score_threshold,
                    "margin_threshold": margin_threshold,
                    "false_merge_count": metrics["false_merge_count"],
                    "recall": metrics["recall"],
                    "precision": metrics["precision"],
                    "accuracy": metrics["accuracy"],
                    "source_recall": {
                        name: item["recall"] for name, item in source_metrics.items()
                    },
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
            )
    eligible_items = [item for item in grid if item["eligible"]]
    if not eligible_items:
        return None, grid
    return max(
        eligible_items,
        key=lambda item: (
            item["recall"],
            min(item["source_recall"].values()),
            item["accuracy"],
            item["margin_threshold"],
            item["score_threshold"],
        ),
    ), grid


def select_a4_variant(recommendations: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not recommendations:
        return None
    return max(
        recommendations,
        key=lambda item: (
            item["recall"],
            min(item["source_recall"].values()),
            item["accuracy"],
            AGGREGATION_PREFERENCE[item["aggregation"]],
            item["margin_threshold"],
            item["score_threshold"],
        ),
    )


def main() -> None:
    args = parse_args()
    minimum_overall_recall = _validated_rate(
        args.minimum_overall_recall, name="minimum overall recall"
    )
    minimum_source_recall = _validated_rate(
        args.minimum_source_recall, name="minimum source recall"
    )
    score_thresholds = parse_thresholds(args.score_thresholds, name="score thresholds")
    margin_thresholds = parse_thresholds(args.margin_thresholds, name="margin thresholds")
    development_raw, cases, sources = load_a4_development(
        args.v2_fixtures,
        args.a31_fixtures,
        args.a31_artifact,
        args.a31_manifest,
    )
    prototype_raw, prototype_version, base_prototypes, prototype_metadata = load_prototypes(
        args.prototypes
    )
    if hashlib.sha256(prototype_raw).hexdigest() != EXPECTED_PROTOTYPE_SHA256:
        raise ValueError("A4 development prototype input changed")
    prototypes = role_annotated_prototypes(base_prototypes, prototype_metadata)
    settings = Settings.from_env()

    scores_by_aggregation: dict[str, dict[str, DimensionScore | None]] = {}
    recommendations: list[dict[str, Any]] = []
    grids: dict[str, list[dict[str, Any]]] = {}
    provider_policy: dict[str, dict[str, Any]] = {}
    for aggregation in A4_AGGREGATIONS:
        scores = _score_development(cases, prototypes, settings, aggregation)
        scores_by_aggregation[aggregation] = scores
        coverage = provider_coverage(cases, scores)
        provider_policy[aggregation] = coverage
        if not coverage["passed"]:
            grids[aggregation] = []
            continue
        recommendation, grid = evaluate_a4_variant(
            aggregation,
            cases,
            sources,
            scores,
            score_thresholds,
            margin_thresholds,
            minimum_overall_recall=minimum_overall_recall,
            minimum_source_recall=minimum_source_recall,
        )
        grids[aggregation] = grid
        if recommendation is not None:
            recommendations.append(recommendation)
    selected = select_a4_variant(recommendations)
    provider_coverage_passed = all(policy["passed"] for policy in provider_policy.values()) and set(
        provider_policy
    ) == set(A4_AGGREGATIONS)

    selected_decisions = None
    selected_metrics = None
    selected_source_metrics = None
    selected_safety_failures: list[str] = []
    selected_robustness_failures: list[str] = []
    if selected is not None:
        selected_decisions = route_a4_development(
            cases,
            scores_by_aggregation[selected["aggregation"]],
            aggregation=selected["aggregation"],
            score_threshold=selected["score_threshold"],
            margin_threshold=selected["margin_threshold"],
        )
        selected_metrics = decision_metrics(cases, selected_decisions)
        selected_source_metrics = _source_metrics(sources, selected_decisions)
        _, selected_safety_failures = _safety_gate(cases, selected_decisions)
        _, selected_robustness_failures = _category_gate(
            cases, selected_decisions, ROBUSTNESS_CATEGORIES
        )

    runner_path = Path(__file__).resolve()
    router_path = ROOT / "src" / "meno" / "semantic_router.py"
    provider_endpoint = (
        settings.google_api_base_url
        if settings.embedding_provider == "google"
        else settings.openai_base_url
    )
    report: dict[str, Any] = {
        "benchmark": "Meno semantic policy router A4 development selection",
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
            "strategy_versions": STRATEGY_VERSION_BY_AGGREGATION,
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
            "embedding_batch_size": (
                settings.google_batch_size
                if settings.embedding_provider == "google"
                else settings.openai_batch_size
            ),
            "embedding_timeout_seconds": (
                settings.google_timeout_seconds
                if settings.embedding_provider == "google"
                else settings.openai_timeout_seconds
            ),
            "embedding_max_retries": (
                settings.google_max_retries
                if settings.embedding_provider == "google"
                else settings.openai_max_retries
            ),
        },
        "provider_policy": {
            "coverage_required": True,
            "coverage_passed": provider_coverage_passed,
            "by_aggregation": provider_policy,
        },
        "selection_policy": {
            "false_merge_count": 0,
            "minimum_overall_recall": minimum_overall_recall,
            "minimum_source_recall": minimum_source_recall,
            "safety_all_correct": True,
            "robustness_all_correct": True,
            "a31_answer_style_all_correct": True,
        },
        "recommendations": recommendations,
        "selected": selected,
        "grid": grids,
        "selected_metrics": selected_metrics,
        "selected_source_metrics": selected_source_metrics,
        "safety_failure_ids": selected_safety_failures,
        "robustness_failure_ids": selected_robustness_failures,
        "gate": {
            "passed": provider_coverage_passed and selected is not None,
            "failure_reason": (
                None
                if selected is not None
                else (
                    "provider_coverage_failed"
                    if not provider_coverage_passed
                    else "no_eligible_policy"
                )
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
        raise ValueError("A4 development artifact would contain raw candidate or reference text")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "selected": selected,
                "selected_metrics": selected_metrics,
                "selected_source_metrics": selected_source_metrics,
                "safety_failure_ids": selected_safety_failures,
                "robustness_failure_ids": selected_robustness_failures,
                "gate": report["gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not provider_coverage_passed:
        raise SystemExit(4)
    if selected is None:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
