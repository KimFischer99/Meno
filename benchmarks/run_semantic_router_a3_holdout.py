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

from benchmarks.run_semantic_router_a3_development import (
    _category_gate,
    _safety_gate,
    _score_development,
    load_prototypes,
    route_development,
)
from benchmarks.run_semantic_router_shadow import (
    EXPECTED_DIMENSIONS,
    ROBUSTNESS_CATEGORIES,
    SAFETY_CATEGORIES,
    FixtureCase,
    _case_correct,
    _git_head,
    _load_candidate,
    _load_reference,
    _package_versions,
    _required_bool,
    _required_optional_string,
    _required_string,
    _safe_endpoint,
    _sha256,
    decision_metrics,
    load_fixture,
)
from meno.config import Settings
from meno.extractor import preference_slot
from meno.semantic_router import ENSEMBLE_STRATEGY_VERSION, ROUTER_VERSION

DEFAULT_HOLDOUT_FIXTURE = (
    Path(__file__).parent / "fixtures" / "semantic-normalization-v3-fresh-holdout.json"
)
DEFAULT_DEVELOPMENT_FIXTURE = Path(__file__).parent / "fixtures" / "semantic-normalization-v2.json"
DEFAULT_PROTOTYPES = Path(__file__).parent / "fixtures" / "semantic-prototypes-a3-v1.json"
DEFAULT_CONFIG = Path(__file__).parent / "fixtures" / "semantic-router-a3-frozen-config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen A3 policy once on a sealed fresh holdout"
    )
    parser.add_argument("--holdout-fixtures", type=Path, default=DEFAULT_HOLDOUT_FIXTURE)
    parser.add_argument(
        "--development-fixtures",
        type=Path,
        default=DEFAULT_DEVELOPMENT_FIXTURE,
        help="revealed v2 fixture used only for exact candidate-overlap rejection",
    )
    parser.add_argument("--prototypes", type=Path, default=DEFAULT_PROTOTYPES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_frozen_config(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("frozen config root must be an object")
    if payload.get("config_version") != "semantic-router-a3-frozen-v1":
        raise ValueError("unsupported A3 frozen config version")
    strategy = payload.get("strategy")
    development = payload.get("development_evidence")
    dependencies = payload.get("dependencies")
    gate = payload.get("fresh_holdout_gate")
    if not all(isinstance(item, dict) for item in (strategy, development, dependencies, gate)):
        raise TypeError("frozen config requires evidence, strategy, dependencies, and gate objects")
    assert isinstance(strategy, dict)
    assert isinstance(development, dict)
    assert isinstance(dependencies, dict)
    assert isinstance(gate, dict)
    expected_strategy = {
        "strategy_version": ENSEMBLE_STRATEGY_VERSION,
        "router_version": ROUTER_VERSION,
        "prototype_version": "semantic-prototypes-a3-v1",
        "aggregation": "top2_mean",
    }
    for key, value in expected_strategy.items():
        if strategy.get(key) != value:
            raise ValueError(f"frozen config strategy.{key} does not match A3 code")
    for key in ("router_sha256", "prototype_sha256"):
        value = strategy.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"frozen config strategy.{key} must be SHA-256")
    dependency_hashes = {
        "development_evidence.runner_sha256": development.get("runner_sha256"),
        "development_evidence.fixture_sha256": development.get("fixture_sha256"),
        "dependencies.semantic_router_shadow_runner_sha256": dependencies.get(
            "semantic_router_shadow_runner_sha256"
        ),
        "dependencies.extractor_sha256": dependencies.get("extractor_sha256"),
        "dependencies.vector_sha256": dependencies.get("vector_sha256"),
    }
    for name, value in dependency_hashes.items():
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"frozen config {name} must be SHA-256")
    for key in ("score_threshold", "margin_threshold"):
        value = strategy.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"frozen config strategy.{key} must be finite in [0, 1]")
    expected_gate = {
        "false_merge_count": 0,
        "safety_all_correct": True,
        "robustness_all_correct": True,
        "threshold_search_allowed": False,
        "prototype_changes_allowed": False,
        "router_changes_allowed": False,
    }
    for key, value in expected_gate.items():
        if gate.get(key) != value:
            raise ValueError(f"frozen config fresh_holdout_gate.{key} is invalid")
    minimum_recall = gate.get("minimum_recall")
    if (
        isinstance(minimum_recall, bool)
        or not isinstance(minimum_recall, (int, float))
        or not math.isfinite(minimum_recall)
        or not 0.0 <= minimum_recall <= 1.0
    ):
        raise ValueError("frozen config minimum_recall must be finite in [0, 1]")
    return raw, payload


def _normalized_candidate_texts(cases: tuple[FixtureCase, ...]) -> set[str]:
    return {" ".join(case.candidate.value.casefold().split()) for case in cases}


def load_fresh_holdout(
    path: Path,
    *,
    development_candidate_texts: set[str],
    expected_seal: dict[str, str],
) -> tuple[bytes, tuple[FixtureCase, ...]]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("fresh holdout fixture root must be an object")
    if payload.get("fixture_version") != "semantic-normalization-v3-fresh-holdout":
        raise ValueError("unsupported fresh holdout fixture version")
    if payload.get("mode") != "fresh-holdout-router":
        raise ValueError("fresh holdout fixture mode is invalid")
    if payload.get("sealed_against") != expected_seal:
        raise ValueError("fresh holdout seal does not match frozen A3 sources")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) < 35:
        raise ValueError("fresh holdout requires at least 35 cases")

    cases: list[FixtureCase] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    for index, item in enumerate(raw_cases):
        context = f"cases[{index}]"
        if not isinstance(item, dict):
            raise TypeError(f"{context} must be an object")
        case_id = _required_string(item, "id", context=context)
        if case_id in seen_ids:
            raise ValueError(f"duplicate fresh holdout case id: {case_id}")
        seen_ids.add(case_id)
        split = _required_string(item, "split", context=context)
        if split != "fresh_holdout":
            raise ValueError(f"{context}.split must be fresh_holdout")
        category = _required_string(item, "category", context=context)
        candidate = _load_candidate(item.get("candidate"), context=f"{context}.candidate")
        if (
            candidate.deterministic_slot is not None
            and candidate.deterministic_slot not in EXPECTED_DIMENSIONS
        ):
            raise ValueError(f"{context}.candidate deterministic slot is unsupported")
        normalized_text = " ".join(candidate.value.casefold().split())
        if normalized_text in development_candidate_texts:
            raise ValueError("fresh holdout candidate text overlaps revealed v2 development")
        if normalized_text in seen_texts:
            raise ValueError("fresh holdout candidate texts must be unique")
        seen_texts.add(normalized_text)
        if candidate.deterministic_slot is None and preference_slot(candidate.value) is not None:
            raise ValueError(f"{context} slot-less candidate hits the current SLOT_LEXICON")

        raw_references = item.get("references")
        if not isinstance(raw_references, list) or not raw_references:
            raise ValueError(f"{context}.references must be a non-empty list")
        references = tuple(
            _load_reference(reference, context=f"{context}.references[{ref_index}]")
            for ref_index, reference in enumerate(raw_references)
        )
        if len({reference.claim_id for reference in references}) != len(references):
            raise ValueError(f"{context}.reference claim ids must be unique")
        if any(reference.deterministic_slot not in EXPECTED_DIMENSIONS for reference in references):
            raise ValueError(f"{context}.references require controlled deterministic slots")

        expected_action = _required_string(item, "expected_action", context=context)
        if expected_action not in {"reuse_key", "new_key", "reject"}:
            raise ValueError(f"{context}.expected_action is unsupported")
        expected_semantic_key = _required_optional_string(
            item, "expected_semantic_key", context=context
        )
        expected_dimension = _required_optional_string(item, "expected_dimension", context=context)
        if expected_dimension is not None and expected_dimension not in EXPECTED_DIMENSIONS:
            raise ValueError(f"{context}.expected_dimension is unsupported")
        expected_protected = _required_bool(item, "expected_protected_reference", context=context)
        if expected_action == "reuse_key" and expected_semantic_key is None:
            raise ValueError(f"{context} reuse_key requires expected_semantic_key")
        if expected_action != "reuse_key" and expected_semantic_key is not None:
            raise ValueError(f"{context} non-reuse action cannot expect a semantic key")
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

    case_tuple = tuple(cases)
    categories = {case.category for case in case_tuple}
    missing_categories = (SAFETY_CATEGORIES | ROBUSTNESS_CATEGORIES) - categories
    if missing_categories:
        raise ValueError(f"fresh holdout is missing categories: {sorted(missing_categories)}")
    dimensions = {case.expected_dimension for case in case_tuple if case.expected_dimension}
    if dimensions != EXPECTED_DIMENSIONS:
        raise ValueError("fresh holdout must cover all seven controlled dimensions")
    if sum(case.expected_action == "reuse_key" for case in case_tuple) < 21:
        raise ValueError("fresh holdout requires at least 21 reuse_key cases")
    if sum(case.expected_action != "reuse_key" for case in case_tuple) < 10:
        raise ValueError("fresh holdout requires at least 10 non-reuse cases")
    for category in ROBUSTNESS_CATEGORIES:
        if sum(case.category == category for case in case_tuple) < 2:
            raise ValueError(f"fresh holdout requires at least two {category} cases")
    return raw, case_tuple


def _case_artifacts(
    cases: tuple[FixtureCase, ...], decisions: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "id": case.case_id,
            "category": case.category,
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
    config_raw, config = load_frozen_config(args.config)
    prototype_raw, prototype_version, prototypes, prototype_metadata = load_prototypes(
        args.prototypes
    )
    strategy = config["strategy"]
    project_root = Path(__file__).resolve().parent.parent
    router_path = project_root / "src" / "meno" / "semantic_router.py"
    if _sha256(router_path) != strategy["router_sha256"]:
        raise ValueError("router source changed after A3 config freeze")
    if hashlib.sha256(prototype_raw).hexdigest() != strategy["prototype_sha256"]:
        raise ValueError("prototype fixture changed after A3 config freeze")
    dependency_paths = {
        config["development_evidence"]["runner_sha256"]: (
            project_root / "benchmarks" / "run_semantic_router_a3_development.py"
        ),
        config["dependencies"]["semantic_router_shadow_runner_sha256"]: (
            project_root / "benchmarks" / "run_semantic_router_shadow.py"
        ),
        config["dependencies"]["extractor_sha256"]: (
            project_root / "src" / "meno" / "extractor.py"
        ),
        config["dependencies"]["vector_sha256"]: (project_root / "src" / "meno" / "vector.py"),
    }
    for expected_sha, dependency_path in dependency_paths.items():
        if _sha256(dependency_path) != expected_sha:
            raise ValueError(f"frozen A3 dependency changed: {dependency_path.name}")

    development_raw, _, development_cases = load_fixture(args.development_fixtures)
    development_fixture_sha = hashlib.sha256(development_raw).hexdigest()
    if development_fixture_sha != config["development_evidence"]["fixture_sha256"]:
        raise ValueError("revealed development fixture changed after A3 config freeze")
    runner_path = Path(__file__).resolve()
    expected_seal = {
        "router_sha256": strategy["router_sha256"],
        "prototype_sha256": strategy["prototype_sha256"],
        "frozen_config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "development_fixture_sha256": development_fixture_sha,
        "holdout_runner_sha256": _sha256(runner_path),
    }
    holdout_raw, cases = load_fresh_holdout(
        args.holdout_fixtures,
        development_candidate_texts=_normalized_candidate_texts(development_cases),
        expected_seal=expected_seal,
    )

    settings = Settings.from_env()
    scores = _score_development(cases, prototypes, settings, strategy["aggregation"])
    decisions = route_development(
        cases,
        scores,
        score_threshold=strategy["score_threshold"],
        margin_threshold=strategy["margin_threshold"],
    )
    metrics = decision_metrics(cases, decisions)
    safety_passed, safety_failures = _safety_gate(cases, decisions)
    robustness_passed, robustness_failures = _category_gate(cases, decisions, ROBUSTNESS_CATEGORIES)
    gate = config["fresh_holdout_gate"]
    passed = bool(
        metrics["false_merge_count"] == gate["false_merge_count"]
        and metrics["recall"] >= gate["minimum_recall"]
        and safety_passed
        and robustness_passed
    )

    provider_endpoint = (
        settings.google_api_base_url
        if settings.embedding_provider == "google"
        else settings.openai_base_url
    )
    report = {
        "benchmark": "Meno semantic policy router A3 sealed fresh holdout",
        "mode": "external-provider-fresh-holdout-read-only",
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "git_head": _git_head(),
        "command": [str(runner_path), *sys.argv[1:]],
        "holdout": {
            "fixture_version": "semantic-normalization-v3-fresh-holdout",
            "fixture_sha256": hashlib.sha256(holdout_raw).hexdigest(),
            "case_count": len(cases),
            "sealed_against": expected_seal,
            "threshold_search_performed": False,
        },
        "frozen_config": {
            "version": config["config_version"],
            "sha256": hashlib.sha256(config_raw).hexdigest(),
            "aggregation": strategy["aggregation"],
            "score_threshold": strategy["score_threshold"],
            "margin_threshold": strategy["margin_threshold"],
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
        "metrics": metrics,
        "safety_passed": safety_passed,
        "safety_failure_ids": safety_failures,
        "robustness_passed": robustness_passed,
        "robustness_failure_ids": robustness_failures,
        "passed": passed,
        "cases": _case_artifacts(cases, decisions),
    }
    serialized = json.dumps(report, ensure_ascii=False)
    raw_values = [
        value
        for case in cases
        for value in (
            case.candidate.value,
            *(reference.value for reference in case.references),
        )
    ]
    if any(value in serialized for value in raw_values):
        raise ValueError("holdout artifact would contain raw candidate or reference text")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "frozen_config": report["frozen_config"],
                "metrics": metrics,
                "safety_failure_ids": safety_failures,
                "robustness_failure_ids": robustness_failures,
                "passed": passed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
