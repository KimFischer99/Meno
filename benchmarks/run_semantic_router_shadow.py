from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import socket
import subprocess
import sys
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from meno.config import Settings
from meno.extractor import preference_slot
from meno.semantic_router import (
    ROUTER_VERSION,
    STRATEGY_VERSION,
    CandidateEnvelope,
    DimensionPrototype,
    DimensionScore,
    PolicyRouter,
    PrototypeDimensionStrategy,
    ReferenceEnvelope,
    RouterDecision,
)
from meno.vector import make_embedder

DEFAULT_FIXTURE = Path(__file__).parent / "fixtures" / "semantic-normalization-v2.json"
DEFAULT_SCORE_THRESHOLDS = tuple(round(index * 0.025, 3) for index in range(41))
DEFAULT_MARGIN_THRESHOLDS = tuple(round(index * 0.025, 3) for index in range(41))
EXPECTED_DIMENSIONS = {
    "answer_style",
    "beverage",
    "programming_language",
    "music",
    "food",
    "color",
    "sport",
}
SAFETY_CATEGORIES = {
    "cross_user_filter",
    "cross_channel_filter",
    "inactive_reference_filter",
    "sensitive_reference_filter",
    "sensitive_candidate",
    "injection_candidate",
    "ambiguous_semantic_key",
    "explicit_feedback_protected",
}
ROBUSTNESS_CATEGORIES = {"negation", "contradiction"}


@dataclass(frozen=True)
class FixtureCase:
    case_id: str
    split: str
    category: str
    candidate: CandidateEnvelope
    references: tuple[ReferenceEnvelope, ...]
    expected_action: str
    expected_semantic_key: str | None
    expected_dimension: str | None
    expected_protected_reference: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate and hold out the shadow-only semantic policy router"
    )
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
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


def parse_thresholds(value: str, *, name: str) -> list[float]:
    try:
        thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated numbers") from exc
    if not thresholds or any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in thresholds):
        raise ValueError(f"{name} must contain finite values between 0 and 1")
    return thresholds


def _required_string(payload: dict[str, Any], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


def _required_bool(payload: dict[str, Any], key: str, *, context: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise TypeError(f"{context}.{key} must be a boolean")
    return value


def _required_optional_string(
    payload: dict[str, Any], key: str, *, context: str
) -> str | None:
    if key not in payload:
        raise ValueError(f"{context}.{key} is required")
    value = payload[key]
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise TypeError(f"{context}.{key} must be a non-empty string or null")
    return value


def _load_candidate(payload: Any, *, context: str) -> CandidateEnvelope:
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be an object")
    return CandidateEnvelope(
        user_id=_required_string(payload, "user_id", context=context),
        kind=_required_string(payload, "kind", context=context),
        semantic_channel=_required_string(payload, "semantic_channel", context=context),
        value=_required_string(payload, "value", context=context),
        sensitive=_required_bool(payload, "sensitive", context=context),
        injection_detected=_required_bool(payload, "injection_detected", context=context),
        deterministic_slot=_required_optional_string(
            payload, "deterministic_slot", context=context
        ),
    )


def _load_reference(payload: Any, *, context: str) -> ReferenceEnvelope:
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be an object")
    return ReferenceEnvelope(
        claim_id=_required_string(payload, "claim_id", context=context),
        user_id=_required_string(payload, "user_id", context=context),
        semantic_key=_required_string(payload, "semantic_key", context=context),
        kind=_required_string(payload, "kind", context=context),
        semantic_channel=_required_string(payload, "semantic_channel", context=context),
        value=_required_string(payload, "value", context=context),
        sensitive=_required_bool(payload, "sensitive", context=context),
        status=_required_string(payload, "status", context=context),
        source_type=_required_string(payload, "source_type", context=context),
        deterministic_slot=_required_optional_string(
            payload, "deterministic_slot", context=context
        ),
    )


def load_fixture(
    path: Path,
) -> tuple[bytes, tuple[DimensionPrototype, ...], tuple[FixtureCase, ...]]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("fixture root must be an object")
    if payload.get("fixture_version") != "semantic-normalization-v2":
        raise ValueError("unsupported semantic normalization fixture version")
    if payload.get("mode") != "multi-reference-router":
        raise ValueError("fixture mode must be multi-reference-router")

    raw_prototypes = payload.get("prototypes")
    if not isinstance(raw_prototypes, dict):
        raise TypeError("fixture prototypes must be an object keyed by dimension")
    prototypes_list: list[DimensionPrototype] = []
    for name, item in raw_prototypes.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(item, dict):
            raise TypeError("fixture prototype entries must contain a dimension object")
        english = _required_string(item, "en", context=f"prototypes.{name}")
        chinese = _required_string(item, "zh", context=f"prototypes.{name}")
        prototypes_list.append(
            DimensionPrototype(name=name, description=f"{english} / {chinese}")
        )
    prototypes = tuple(prototypes_list)
    if {prototype.name for prototype in prototypes} != EXPECTED_DIMENSIONS:
        raise ValueError("fixture prototypes must contain exactly the seven controlled dimensions")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) < 44:
        raise ValueError("fixture must contain at least 44 cases")
    cases: list[FixtureCase] = []
    seen_ids: set[str] = set()
    text_splits: dict[str, str] = {}
    for index, item in enumerate(raw_cases):
        context = f"cases[{index}]"
        if not isinstance(item, dict):
            raise TypeError(f"{context} must be an object")
        case_id = _required_string(item, "id", context=context)
        if case_id in seen_ids:
            raise ValueError(f"duplicate fixture case id: {case_id}")
        seen_ids.add(case_id)
        split = _required_string(item, "split", context=context)
        if split not in {"calibration", "holdout"}:
            raise ValueError(f"{context}.split must be calibration or holdout")
        category = _required_string(item, "category", context=context)
        candidate = _load_candidate(item.get("candidate"), context=f"{context}.candidate")
        raw_references = item.get("references")
        if not isinstance(raw_references, list):
            raise TypeError(f"{context}.references must be a list")
        references = tuple(
            _load_reference(reference, context=f"{context}.references[{ref_index}]")
            for ref_index, reference in enumerate(raw_references)
        )
        if len({reference.claim_id for reference in references}) != len(references):
            raise ValueError(f"{context}.reference claim ids must be unique")
        expected_action = _required_string(item, "expected_action", context=context)
        if expected_action not in {"reuse_key", "new_key", "reject"}:
            raise ValueError(f"{context}.expected_action is unsupported")
        expected_semantic_key = _required_optional_string(
            item, "expected_semantic_key", context=context
        )
        expected_dimension = _required_optional_string(
            item, "expected_dimension", context=context
        )
        expected_protected = _required_bool(
            item, "expected_protected_reference", context=context
        )
        if expected_action == "reuse_key" and expected_semantic_key is None:
            raise ValueError(f"{context} reuse_key requires expected_semantic_key")
        if expected_action != "reuse_key" and expected_semantic_key is not None:
            raise ValueError(f"{context} non-reuse action cannot expect a semantic key")
        if candidate.deterministic_slot is None and preference_slot(candidate.value) is not None:
            raise ValueError(f"{context} slot-less candidate hits the current SLOT_LEXICON")
        normalized_text = " ".join(candidate.value.casefold().split())
        previous_split = text_splits.setdefault(normalized_text, split)
        if previous_split != split:
            raise ValueError("candidate text is reused across calibration and holdout")
        if (candidate.sensitive or candidate.injection_detected) and expected_action != "reject":
            raise ValueError(f"{context} unsafe candidate must expect reject")
        cases.append(
            FixtureCase(
                case_id=case_id,
                split=split,
                category=category,
                candidate=candidate,
                references=references,
                expected_action=expected_action,
                expected_semantic_key=expected_semantic_key,
                expected_dimension=expected_dimension,
                expected_protected_reference=expected_protected,
            )
        )

    for split in ("calibration", "holdout"):
        split_cases = [case for case in cases if case.split == split]
        if not split_cases:
            raise ValueError(f"fixture requires {split} cases")
        if not any(case.expected_action == "reuse_key" for case in split_cases):
            raise ValueError(f"fixture {split} split requires a reuse_key case")
        if not any(case.expected_action != "reuse_key" for case in split_cases):
            raise ValueError(f"fixture {split} split requires a non-reuse case")
    categories = {case.category for case in cases}
    missing_categories = (SAFETY_CATEGORIES | ROBUSTNESS_CATEGORIES) - categories
    if missing_categories:
        raise ValueError(f"fixture is missing required categories: {sorted(missing_categories)}")
    return raw, prototypes, tuple(cases)


def _case_correct(case: FixtureCase, decision: RouterDecision) -> bool:
    return (
        decision.action == case.expected_action
        and decision.proposed_semantic_key == case.expected_semantic_key
        and (
            case.expected_dimension is None
            or decision.dimension == case.expected_dimension
        )
        and decision.protected_reference == case.expected_protected_reference
    )


def decision_metrics(
    cases: Iterable[FixtureCase],
    decisions: dict[str, RouterDecision],
) -> dict[str, Any]:
    case_list = list(cases)
    correct_ids = [case.case_id for case in case_list if _case_correct(case, decisions[case.case_id])]
    positive_cases = [case for case in case_list if case.expected_action == "reuse_key"]
    correct_reuse_ids = [
        case.case_id
        for case in positive_cases
        if decisions[case.case_id].action == "reuse_key"
        and decisions[case.case_id].proposed_semantic_key == case.expected_semantic_key
    ]
    false_merge_ids = [
        case.case_id
        for case in case_list
        if decisions[case.case_id].action == "reuse_key"
        and (
            case.expected_action != "reuse_key"
            or decisions[case.case_id].proposed_semantic_key != case.expected_semantic_key
        )
    ]
    missed_merge_ids = [case.case_id for case in positive_cases if case.case_id not in correct_reuse_ids]
    predicted_reuse = sum(decision.action == "reuse_key" for decision in decisions.values())
    recall = len(correct_reuse_ids) / len(positive_cases) if positive_cases else 1.0
    precision = len(correct_reuse_ids) / predicted_reuse if predicted_reuse else 1.0
    category_metrics: dict[str, dict[str, Any]] = {}
    for category in sorted({case.category for case in case_list}):
        category_cases = [case for case in case_list if case.category == category]
        category_correct = sum(
            _case_correct(case, decisions[case.case_id]) for case in category_cases
        )
        category_metrics[category] = {
            "count": len(category_cases),
            "correct": category_correct,
            "accuracy": category_correct / len(category_cases),
        }
    return {
        "case_count": len(case_list),
        "positive_count": len(positive_cases),
        "correct_count": len(correct_ids),
        "accuracy": len(correct_ids) / len(case_list) if case_list else 1.0,
        "precision": precision,
        "recall": recall,
        "false_merge_count": len(false_merge_ids),
        "false_merge_ids": false_merge_ids,
        "missed_merge_count": len(missed_merge_ids),
        "missed_merge_ids": missed_merge_ids,
        "category_metrics": category_metrics,
    }


def route_cases(
    cases: Iterable[FixtureCase],
    scores: dict[str, DimensionScore | None],
    *,
    score_threshold: float,
    margin_threshold: float,
) -> dict[str, RouterDecision]:
    router = PolicyRouter()
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


def recommend_thresholds(
    cases: tuple[FixtureCase, ...],
    scores: dict[str, DimensionScore | None],
    score_thresholds: list[float],
    margin_thresholds: list[float],
    *,
    minimum_recall: float,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    calibration_cases = tuple(case for case in cases if case.split == "calibration")
    grid: list[dict[str, Any]] = []
    for score_threshold in score_thresholds:
        for margin_threshold in margin_thresholds:
            decisions = route_cases(
                calibration_cases,
                scores,
                score_threshold=score_threshold,
                margin_threshold=margin_threshold,
            )
            metrics = decision_metrics(calibration_cases, decisions)
            grid.append(
                {
                    "score_threshold": score_threshold,
                    "margin_threshold": margin_threshold,
                    "false_merge_count": metrics["false_merge_count"],
                    "recall": metrics["recall"],
                    "precision": metrics["precision"],
                    "accuracy": metrics["accuracy"],
                }
            )
    eligible = [
        item
        for item in grid
        if item["false_merge_count"] == 0 and item["recall"] >= minimum_recall
    ]
    if not eligible:
        return None, grid
    return max(
        eligible,
        key=lambda item: (
            item["recall"],
            item["score_threshold"],
            item["margin_threshold"],
            item["accuracy"],
        ),
    ), grid


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("meno-memory", "httpx", "pydantic", "qdrant-client", "sqlalchemy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _safe_endpoint(value: str) -> str:
    """Retain provider identity/path without persisting credentials or query data."""

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return "configured-invalid-url"
    if not parsed.scheme or not hostname:
        return "configured-non-url"
    rendered_host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = f"{rendered_host}:{port}" if port is not None else rendered_host
    return f"{parsed.scheme}://{netloc}{parsed.path}".rstrip("/")


def _score_cases(
    cases: tuple[FixtureCase, ...],
    prototypes: tuple[DimensionPrototype, ...],
    settings: Settings,
) -> dict[str, DimensionScore | None]:
    score_cases = [
        case
        for case in cases
        if case.candidate.deterministic_slot is None
    ]
    embedder = make_embedder(settings)
    try:
        values = PrototypeDimensionStrategy(embedder, prototypes).score_many(
            [case.candidate for case in score_cases]
        )
    finally:
        embedder.close()
    scores = {case.case_id: None for case in cases}
    scores.update(
        {
            case.case_id: score
            for case, score in zip(score_cases, values, strict=True)
        }
    )
    return scores


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.minimum_recall) or not 0.0 <= args.minimum_recall <= 1.0:
        raise ValueError("minimum recall must be finite and between 0 and 1")
    score_thresholds = parse_thresholds(args.score_thresholds, name="score thresholds")
    margin_thresholds = parse_thresholds(args.margin_thresholds, name="margin thresholds")
    fixture_raw, prototypes, cases = load_fixture(args.fixtures)
    settings = Settings.from_env()
    scores = _score_cases(cases, prototypes, settings)
    recommendation, grid = recommend_thresholds(
        cases,
        scores,
        score_thresholds,
        margin_thresholds,
        minimum_recall=args.minimum_recall,
    )

    selected_score = recommendation["score_threshold"] if recommendation else 1.0
    selected_margin = recommendation["margin_threshold"] if recommendation else 1.0
    calibration_cases = tuple(case for case in cases if case.split == "calibration")
    holdout_cases = tuple(case for case in cases if case.split == "holdout")
    calibration_decisions = route_cases(
        calibration_cases,
        scores,
        score_threshold=selected_score,
        margin_threshold=selected_margin,
    )
    holdout_decisions = route_cases(
        holdout_cases,
        scores,
        score_threshold=selected_score,
        margin_threshold=selected_margin,
    )
    calibration_metrics = decision_metrics(calibration_cases, calibration_decisions)
    holdout_metrics = decision_metrics(holdout_cases, holdout_decisions)
    all_decisions = {**calibration_decisions, **holdout_decisions}

    safety_case_ids = [case.case_id for case in cases if case.category in SAFETY_CATEGORIES]
    safety_passed = all(
        _case_correct(case, all_decisions[case.case_id])
        for case in cases
        if case.case_id in safety_case_ids
    )
    robustness = {
        category: holdout_metrics["category_metrics"].get(
            category, {"count": 0, "correct": 0, "accuracy": 0.0}
        )
        for category in sorted(ROBUSTNESS_CATEGORIES)
    }
    robustness_passed = all(
        item["count"] > 0 and item["correct"] == item["count"]
        for item in robustness.values()
    )
    passed = bool(
        recommendation
        and holdout_metrics["false_merge_count"] == 0
        and holdout_metrics["recall"] >= args.minimum_recall
        and safety_passed
        and robustness_passed
    )

    runner_path = Path(__file__).resolve()
    router_path = runner_path.parent.parent / "src" / "meno" / "semantic_router.py"
    provider_endpoint = (
        settings.google_api_base_url
        if settings.embedding_provider == "google"
        else settings.openai_base_url
    )
    report = {
        "benchmark": "Meno semantic policy router shadow calibration",
        "mode": "external-provider-shadow-read-only",
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "git_head": _git_head(),
        "command": [str(runner_path), *sys.argv[1:]],
        "fixture": {
            "version": "semantic-normalization-v2",
            "sha256": hashlib.sha256(fixture_raw).hexdigest(),
            "case_count": len(cases),
            "calibration_count": len(calibration_cases),
            "holdout_count": len(holdout_cases),
        },
        "source": {
            "runner_sha256": _sha256(runner_path),
            "router_sha256": _sha256(router_path),
            "strategy_version": STRATEGY_VERSION,
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
            "injection_blocked_count": sum(
                case.candidate.injection_detected for case in cases
            ),
            "raw_text_persisted": False,
        },
        "gate": {
            "minimum_recall": args.minimum_recall,
            "calibration_recommendation_found": recommendation is not None,
            "holdout_zero_false_merge": holdout_metrics["false_merge_count"] == 0,
            "holdout_minimum_recall": holdout_metrics["recall"] >= args.minimum_recall,
            "safety_passed": safety_passed,
            "robustness_passed": robustness_passed,
            "passed": passed,
        },
        "recommended_thresholds": recommendation,
        "calibration_metrics": calibration_metrics,
        "holdout_metrics": holdout_metrics,
        "robustness_holdout": robustness,
        "calibration_grid": grid,
        "cases": [
            {
                "id": case.case_id,
                "split": case.split,
                "category": case.category,
                "expected_action": case.expected_action,
                "expected_semantic_key": case.expected_semantic_key,
                "expected_dimension": case.expected_dimension,
                "actual_action": all_decisions[case.case_id].action,
                "actual_semantic_key": all_decisions[case.case_id].proposed_semantic_key,
                "actual_dimension": all_decisions[case.case_id].dimension,
                "score": all_decisions[case.case_id].score,
                "margin": all_decisions[case.case_id].margin,
                "reason": all_decisions[case.case_id].reason,
                "protected_reference": all_decisions[case.case_id].protected_reference,
                "correct": _case_correct(case, all_decisions[case.case_id]),
            }
            for case in cases
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "recommended_thresholds": recommendation,
                "calibration_metrics": calibration_metrics,
                "holdout_metrics": holdout_metrics,
                "gate": report["gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
