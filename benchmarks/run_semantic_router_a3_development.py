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

from benchmarks.run_semantic_router_shadow import (
    DEFAULT_MARGIN_THRESHOLDS,
    DEFAULT_SCORE_THRESHOLDS,
    EXPECTED_DIMENSIONS,
    ROBUSTNESS_CATEGORIES,
    SAFETY_CATEGORIES,
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
    ENSEMBLE_STRATEGY_VERSION,
    ROUTER_VERSION,
    DimensionPrototype,
    DimensionScore,
    EnsembleAggregation,
    PolicyRouter,
    PrototypeEnsembleStrategy,
    RouterDecision,
)
from meno.vector import make_embedder

DEFAULT_DEVELOPMENT_FIXTURE = Path(__file__).parent / "fixtures" / "semantic-normalization-v2.json"
DEFAULT_PROTOTYPES = Path(__file__).parent / "fixtures" / "semantic-prototypes-a3-v1.json"
DEFAULT_AGGREGATIONS: tuple[EnsembleAggregation, ...] = (
    "mean",
    "max",
    "top2_mean",
)
AGGREGATION_PREFERENCE = {"max": 0, "mean": 1, "top2_mean": 2}


def _contains_raw_text(value: object, raw_values: tuple[str, ...]) -> bool:
    if isinstance(value, str):
        return any(raw_value in value for raw_value in raw_values)
    if isinstance(value, dict):
        return any(
            _contains_raw_text(key, raw_values) or _contains_raw_text(item, raw_values)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_raw_text(item, raw_values) for item in value)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a frozen A3 ensemble policy on revealed v2 development evidence"
    )
    parser.add_argument("--development-fixtures", type=Path, default=DEFAULT_DEVELOPMENT_FIXTURE)
    parser.add_argument("--prototypes", type=Path, default=DEFAULT_PROTOTYPES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--aggregations",
        default=",".join(DEFAULT_AGGREGATIONS),
        help="comma-separated subset of mean,max,top2_mean",
    )
    parser.add_argument(
        "--score-thresholds",
        default=",".join(str(value) for value in DEFAULT_SCORE_THRESHOLDS),
    )
    parser.add_argument(
        "--margin-thresholds",
        default=",".join(str(value) for value in DEFAULT_MARGIN_THRESHOLDS),
    )
    parser.add_argument("--minimum-recall", type=float, default=0.50)
    return parser.parse_args()


def parse_aggregations(value: str) -> list[EnsembleAggregation]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    allowed = set(DEFAULT_AGGREGATIONS)
    if not raw or len(set(raw)) != len(raw) or any(item not in allowed for item in raw):
        raise ValueError("aggregations must be a unique subset of mean,max,top2_mean")
    return [item for item in raw if item in allowed]  # type: ignore[misc]


def load_prototypes(
    path: Path,
) -> tuple[bytes, str, tuple[DimensionPrototype, ...], dict[str, Any]]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("prototype fixture root must be an object")
    version = payload.get("prototype_version")
    if version != "semantic-prototypes-a3-v1":
        raise ValueError("unsupported A3 prototype version")
    dimensions = payload.get("dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != EXPECTED_DIMENSIONS:
        raise ValueError("A3 prototypes must contain exactly the seven controlled dimensions")

    prototypes: list[DimensionPrototype] = []
    metadata: dict[str, Any] = {"dimensions": {}, "anchor_count": 0}
    for dimension in sorted(dimensions):
        anchors = dimensions[dimension]
        if not isinstance(anchors, list) or len(anchors) < 2:
            raise ValueError(f"prototype dimension {dimension} requires at least two anchors")
        seen_ids: set[str] = set()
        dimension_metadata: list[dict[str, str]] = []
        for index, anchor in enumerate(anchors):
            context = f"dimensions.{dimension}[{index}]"
            if not isinstance(anchor, dict):
                raise TypeError(f"{context} must be an object")
            anchor_id = anchor.get("id")
            role = anchor.get("role")
            polarity = anchor.get("polarity")
            english = anchor.get("en")
            chinese = anchor.get("zh")
            values = (anchor_id, role, polarity, english, chinese)
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{context} fields must be non-empty strings")
            assert isinstance(anchor_id, str)
            assert isinstance(role, str)
            assert isinstance(polarity, str)
            assert isinstance(english, str)
            assert isinstance(chinese, str)
            if anchor_id in seen_ids:
                raise ValueError(f"duplicate anchor id in dimension {dimension}: {anchor_id}")
            seen_ids.add(anchor_id)
            if role not in {"topic", "polarity"}:
                raise ValueError(f"{context}.role is unsupported")
            if polarity not in {"neutral", "affirmed", "negated", "contradiction"}:
                raise ValueError(f"{context}.polarity is unsupported")
            prototypes.append(
                DimensionPrototype(
                    name=dimension,
                    description=f"{english} / {chinese}",
                )
            )
            dimension_metadata.append({"id": anchor_id, "role": role, "polarity": polarity})
        required_anchor_ids = {"topic", "affirmed", "negated", "contradiction"}
        required_polarities = {"neutral", "affirmed", "negated", "contradiction"}
        if (
            seen_ids != required_anchor_ids
            or {item["polarity"] for item in dimension_metadata} != required_polarities
        ):
            raise ValueError(
                f"prototype dimension {dimension} requires exactly topic, affirmed, "
                "negated, and contradiction anchors"
            )
        if any((item["id"] == "topic") != (item["role"] == "topic") for item in dimension_metadata):
            raise ValueError(
                f"prototype dimension {dimension} must reserve role=topic for topic anchor"
            )
        metadata["dimensions"][dimension] = dimension_metadata
        metadata["anchor_count"] += len(dimension_metadata)
    return raw, version, tuple(prototypes), metadata


def _score_development(
    cases: tuple[FixtureCase, ...],
    prototypes: tuple[DimensionPrototype, ...],
    settings: Settings,
    aggregation: EnsembleAggregation,
) -> dict[str, DimensionScore | None]:
    score_cases = [case for case in cases if case.candidate.deterministic_slot is None]
    embedder = make_embedder(settings)
    try:
        values = PrototypeEnsembleStrategy(
            embedder,
            prototypes,
            aggregation=aggregation,
        ).score_many([case.candidate for case in score_cases])
    finally:
        embedder.close()
    result = {case.case_id: None for case in cases}
    result.update({case.case_id: score for case, score in zip(score_cases, values, strict=True)})
    return result


def route_development(
    cases: tuple[FixtureCase, ...],
    scores: dict[str, DimensionScore | None],
    *,
    score_threshold: float,
    margin_threshold: float,
) -> dict[str, RouterDecision]:
    router = PolicyRouter(strategy_version=ENSEMBLE_STRATEGY_VERSION)
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


def _category_gate(
    cases: tuple[FixtureCase, ...],
    decisions: dict[str, RouterDecision],
    categories: set[str],
) -> tuple[bool, list[str]]:
    failures = [
        case.case_id
        for case in cases
        if case.category in categories and not _case_correct(case, decisions[case.case_id])
    ]
    present = {case.category for case in cases}
    return not failures and categories <= present, failures


def _safety_case_correct(case: FixtureCase, decision: RouterDecision) -> bool:
    if case.category in {"sensitive_candidate", "injection_candidate"}:
        return (
            decision.action == "reject"
            and decision.proposed_semantic_key is None
            and decision.score is None
            and decision.margin is None
        )
    if case.category == "ambiguous_semantic_key":
        return decision.action == "reject" and decision.proposed_semantic_key is None
    if case.category == "explicit_feedback_protected":
        if not decision.protected_reference:
            return False
        if decision.action == "reject":
            return decision.proposed_semantic_key is None
        return (
            decision.action == "reuse_key"
            and decision.proposed_semantic_key == case.expected_semantic_key
        )
    return decision.action != "reuse_key" and decision.proposed_semantic_key is None


def _safety_gate(
    cases: tuple[FixtureCase, ...],
    decisions: dict[str, RouterDecision],
) -> tuple[bool, list[str]]:
    failures = [
        case.case_id
        for case in cases
        if case.category in SAFETY_CATEGORIES
        and not _safety_case_correct(case, decisions[case.case_id])
    ]
    present = {case.category for case in cases}
    return not failures and SAFETY_CATEGORIES <= present, failures


def evaluate_variant(
    aggregation: EnsembleAggregation,
    cases: tuple[FixtureCase, ...],
    scores: dict[str, DimensionScore | None],
    score_thresholds: list[float],
    margin_thresholds: list[float],
    *,
    minimum_recall: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    grid: list[dict[str, Any]] = []
    for score_threshold in score_thresholds:
        for margin_threshold in margin_thresholds:
            decisions = route_development(
                cases,
                scores,
                score_threshold=score_threshold,
                margin_threshold=margin_threshold,
            )
            metrics = decision_metrics(cases, decisions)
            safety_passed, safety_failures = _safety_gate(cases, decisions)
            robustness_passed, robustness_failures = _category_gate(
                cases, decisions, ROBUSTNESS_CATEGORIES
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
                    "safety_passed": safety_passed,
                    "safety_failure_ids": safety_failures,
                    "robustness_passed": robustness_passed,
                    "robustness_failure_ids": robustness_failures,
                }
            )
    eligible = [
        item
        for item in grid
        if item["false_merge_count"] == 0
        and item["safety_passed"]
        and item["robustness_passed"]
        and item["recall"] >= minimum_recall
    ]
    if not eligible:
        return None, grid
    return max(
        eligible,
        key=lambda item: (
            item["recall"],
            item["accuracy"],
            item["score_threshold"],
            item["margin_threshold"],
        ),
    ), grid


def select_variant(recommendations: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not recommendations:
        return None
    return max(
        recommendations,
        key=lambda item: (
            item["recall"],
            item["accuracy"],
            AGGREGATION_PREFERENCE[item["aggregation"]],
            item["score_threshold"],
            item["margin_threshold"],
        ),
    )


def _case_artifacts(
    cases: tuple[FixtureCase, ...],
    decisions: dict[str, RouterDecision],
) -> list[dict[str, Any]]:
    return [
        {
            "id": case.case_id,
            "category": case.category,
            "previous_split": case.split,
            "expected_action": case.expected_action,
            "expected_semantic_key": case.expected_semantic_key,
            "expected_dimension": case.expected_dimension,
            "actual_action": decisions[case.case_id].action,
            "actual_semantic_key": decisions[case.case_id].proposed_semantic_key,
            "actual_dimension": decisions[case.case_id].dimension,
            "score": decisions[case.case_id].score,
            "margin": decisions[case.case_id].margin,
            "reason": decisions[case.case_id].reason,
            "protected_reference": decisions[case.case_id].protected_reference,
            "correct": _case_correct(case, decisions[case.case_id]),
        }
        for case in cases
    ]


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.minimum_recall) or not 0.0 <= args.minimum_recall <= 1.0:
        raise ValueError("minimum recall must be finite and between 0 and 1")
    aggregations = parse_aggregations(args.aggregations)
    score_thresholds = parse_thresholds(args.score_thresholds, name="score thresholds")
    margin_thresholds = parse_thresholds(args.margin_thresholds, name="margin thresholds")
    development_raw, _, cases = load_fixture(args.development_fixtures)
    prototype_raw, prototype_version, prototypes, prototype_metadata = load_prototypes(
        args.prototypes
    )
    settings = Settings.from_env()

    scores_by_aggregation: dict[str, dict[str, DimensionScore | None]] = {}
    recommendations: list[dict[str, Any]] = []
    grids: dict[str, list[dict[str, Any]]] = {}
    for aggregation in aggregations:
        scores = _score_development(cases, prototypes, settings, aggregation)
        scores_by_aggregation[aggregation] = scores
        recommendation, grid = evaluate_variant(
            aggregation,
            cases,
            scores,
            score_thresholds,
            margin_thresholds,
            minimum_recall=args.minimum_recall,
        )
        grids[aggregation] = grid
        if recommendation is not None:
            recommendations.append(recommendation)
    selected = select_variant(recommendations)

    selected_decisions: dict[str, RouterDecision] | None = None
    selected_metrics: dict[str, Any] | None = None
    selected_safety_failures: list[str] = []
    selected_robustness_failures: list[str] = []
    if selected is not None:
        selected_decisions = route_development(
            cases,
            scores_by_aggregation[selected["aggregation"]],
            score_threshold=selected["score_threshold"],
            margin_threshold=selected["margin_threshold"],
        )
        selected_metrics = decision_metrics(cases, selected_decisions)
        _, selected_safety_failures = _safety_gate(cases, selected_decisions)
        _, selected_robustness_failures = _category_gate(
            cases, selected_decisions, ROBUSTNESS_CATEGORIES
        )

    runner_path = Path(__file__).resolve()
    router_path = runner_path.parent.parent / "src" / "meno" / "semantic_router.py"
    provider_endpoint = (
        settings.google_api_base_url
        if settings.embedding_provider == "google"
        else settings.openai_base_url
    )
    report = {
        "benchmark": "Meno semantic policy router A3 development selection",
        "mode": "external-provider-development-read-only",
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "git_head": _git_head(),
        "command": [str(runner_path), *sys.argv[1:]],
        "development": {
            "fixture_version": "semantic-normalization-v2",
            "fixture_sha256": hashlib.sha256(development_raw).hexdigest(),
            "case_count": len(cases),
            "previous_splits_pooled": True,
            "fresh_holdout_used": False,
        },
        "prototypes": {
            "version": prototype_version,
            "sha256": hashlib.sha256(prototype_raw).hexdigest(),
            **prototype_metadata,
        },
        "source": {
            "runner_sha256": _sha256(runner_path),
            "router_sha256": _sha256(router_path),
            "strategy_version": ENSEMBLE_STRATEGY_VERSION,
            "router_version": ROUTER_VERSION,
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
            "candidate_count": len(cases),
            "deterministic_bypass_count": sum(
                case.candidate.deterministic_slot is not None for case in cases
            ),
            "sensitive_blocked_count": sum(case.candidate.sensitive for case in cases),
            "injection_blocked_count": sum(case.candidate.injection_detected for case in cases),
            "raw_text_persisted": False,
        },
        "selection_gate": {
            "minimum_recall": args.minimum_recall,
            "false_merge_required": 0,
            "safety_all_correct_required": True,
            "robustness_all_correct_required": True,
            "passed": selected is not None,
        },
        "selected": selected,
        "selected_metrics": selected_metrics,
        "selected_safety_failure_ids": selected_safety_failures,
        "selected_robustness_failure_ids": selected_robustness_failures,
        "variant_recommendations": recommendations,
        "variant_grids": grids,
        "cases": (
            _case_artifacts(cases, selected_decisions) if selected_decisions is not None else []
        ),
        "raw_text_persisted": False,
    }
    raw_values = tuple(
        value
        for case in cases
        for value in (
            case.candidate.value,
            *(reference.value for reference in case.references),
        )
    )
    if _contains_raw_text(report, raw_values):
        raise ValueError("A3 development artifact would contain raw candidate or reference text")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "selected": selected,
                "selected_metrics": selected_metrics,
                "safety_failure_ids": selected_safety_failures,
                "robustness_failure_ids": selected_robustness_failures,
                "gate": report["selection_gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if selected is None:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
