from __future__ import annotations

import argparse
import hashlib
import json
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
    route_development,
)
from benchmarks.run_semantic_router_a3_holdout import (
    _case_artifacts,
    _normalized_candidate_texts,
    load_frozen_config,
)
from benchmarks.run_semantic_router_shadow import (
    EXPECTED_DIMENSIONS,
    ROBUSTNESS_CATEGORIES,
    SAFETY_CATEGORIES,
    FixtureCase,
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
    Path(__file__).parent / "fixtures" / "semantic-normalization-v4-provider-holdout.json"
)
DEFAULT_DEVELOPMENT_FIXTURE = Path(__file__).parent / "fixtures" / "semantic-normalization-v2.json"
DEFAULT_PROTOTYPES = Path(__file__).parent / "fixtures" / "semantic-prototypes-a3-v1.json"
DEFAULT_CONFIG = Path(__file__).parent / "fixtures" / "semantic-router-a31-frozen-config.json"
DEFAULT_MANIFEST = (
    Path(__file__).parent / "fixtures" / "semantic-router-a31-evaluation-manifest.json"
)
MANIFEST_VERSION = "semantic-router-a31-evaluation-manifest-v1"
FIXTURE_VERSION = "semantic-normalization-v4-provider-holdout"
MIN_PROVIDER_SCORED_COUNT = 35
MIN_PROVIDER_SCORED_PER_DIMENSION = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen A3.1 policy once on a provider-scored sealed holdout"
    )
    parser.add_argument("--holdout-fixtures", type=Path, default=DEFAULT_HOLDOUT_FIXTURE)
    parser.add_argument("--development-fixtures", type=Path, default=DEFAULT_DEVELOPMENT_FIXTURE)
    parser.add_argument("--prototypes", type=Path, default=DEFAULT_PROTOTYPES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--expected-manifest-sha256",
        required=True,
        help="out-of-band SHA-256 published by the independent manifest author",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _require_sha(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


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


def load_evaluation_manifest(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    expected_sha = _require_sha(expected_sha256, name="expected manifest SHA-256")
    raw = path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError("evaluation manifest does not match the out-of-band SHA-256")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("evaluation manifest root must be an object")
    if payload.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("unsupported A3.1 evaluation manifest version")
    files = payload.get("files")
    policy = payload.get("policy")
    coverage = payload.get("coverage")
    if not all(isinstance(item, dict) for item in (files, policy, coverage)):
        raise TypeError("evaluation manifest requires files, policy, and coverage objects")
    assert isinstance(files, dict)
    for name, value in files.items():
        _require_sha(value, name=f"manifest files.{name}")
    return raw, payload


def _provider_scored_cases(cases: tuple[FixtureCase, ...]) -> tuple[FixtureCase, ...]:
    return tuple(
        case
        for case in cases
        if (
            case.candidate.deterministic_slot is None
            and not case.candidate.sensitive
            and not case.candidate.injection_detected
            and case.expected_dimension is not None
        )
    )


def provider_coverage(cases: tuple[FixtureCase, ...]) -> dict[str, Any]:
    scored = _provider_scored_cases(cases)
    by_dimension = Counter(case.expected_dimension for case in scored)
    return {
        "provider_scored_count": len(scored),
        "provider_scored_by_dimension": {
            dimension: by_dimension.get(dimension, 0) for dimension in sorted(EXPECTED_DIMENSIONS)
        },
    }


def load_provider_holdout(
    path: Path,
    *,
    development_candidate_texts: set[str],
) -> tuple[bytes, tuple[FixtureCase, ...]]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("provider holdout fixture root must be an object")
    if payload.get("fixture_version") != FIXTURE_VERSION:
        raise ValueError("unsupported provider holdout fixture version")
    if payload.get("mode") != "fresh-holdout-provider-scored":
        raise ValueError("provider holdout fixture mode is invalid")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) < 42:
        raise ValueError("provider holdout requires at least 42 cases")

    cases: list[FixtureCase] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    for index, item in enumerate(raw_cases):
        context = f"cases[{index}]"
        if not isinstance(item, dict):
            raise TypeError(f"{context} must be an object")
        case_id = _required_string(item, "id", context=context)
        if case_id in seen_ids:
            raise ValueError(f"duplicate provider holdout case id: {case_id}")
        seen_ids.add(case_id)
        if _required_string(item, "split", context=context) != "fresh_holdout":
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
            raise ValueError("provider holdout candidate overlaps revealed development text")
        if normalized_text in seen_texts:
            raise ValueError("provider holdout candidate texts must be unique")
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
                split="fresh_holdout",
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
        raise ValueError(f"provider holdout is missing categories: {sorted(missing_categories)}")
    dimensions = {case.expected_dimension for case in case_tuple if case.expected_dimension}
    if dimensions != EXPECTED_DIMENSIONS:
        raise ValueError("provider holdout must cover all seven controlled dimensions")
    if sum(case.expected_action == "reuse_key" for case in case_tuple) < 21:
        raise ValueError("provider holdout requires at least 21 reuse_key cases")
    if sum(case.expected_action != "reuse_key" for case in case_tuple) < 10:
        raise ValueError("provider holdout requires at least 10 non-reuse cases")
    for category in ROBUSTNESS_CATEGORIES:
        if sum(case.category == category for case in case_tuple) < 2:
            raise ValueError(f"provider holdout requires at least two {category} cases")

    coverage = provider_coverage(case_tuple)
    if coverage["provider_scored_count"] < MIN_PROVIDER_SCORED_COUNT:
        raise ValueError("provider holdout has insufficient provider-scored cases")
    if any(
        count < MIN_PROVIDER_SCORED_PER_DIMENSION
        for count in coverage["provider_scored_by_dimension"].values()
    ):
        raise ValueError("provider holdout has insufficient provider-scored dimension coverage")
    return raw, case_tuple


def _expected_manifest_files(
    *,
    project_root: Path,
    config_raw: bytes,
    prototype_raw: bytes,
    development_raw: bytes,
    holdout_raw: bytes,
) -> dict[str, str]:
    return {
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "config_module_sha256": _sha256(project_root / "src" / "meno" / "config.py"),
        "development_fixture_sha256": hashlib.sha256(development_raw).hexdigest(),
        "development_runner_sha256": _sha256(
            project_root / "benchmarks" / "run_semantic_router_a3_development.py"
        ),
        "extractor_sha256": _sha256(project_root / "src" / "meno" / "extractor.py"),
        "holdout_base_runner_sha256": _sha256(
            project_root / "benchmarks" / "run_semantic_router_a3_holdout.py"
        ),
        "holdout_fixture_sha256": hashlib.sha256(holdout_raw).hexdigest(),
        "holdout_runner_sha256": _sha256(Path(__file__).resolve()),
        "prototype_sha256": hashlib.sha256(prototype_raw).hexdigest(),
        "router_sha256": _sha256(project_root / "src" / "meno" / "semantic_router.py"),
        "shadow_runner_sha256": _sha256(
            project_root / "benchmarks" / "run_semantic_router_shadow.py"
        ),
        "vector_sha256": _sha256(project_root / "src" / "meno" / "vector.py"),
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    config_raw, config = load_frozen_config(args.config)
    if config.get("freeze_revision") != "a3.1-provider-scored":
        raise ValueError("A3.1 requires the provider-scored frozen config")
    prototype_raw, prototype_version, prototypes, prototype_metadata = load_prototypes(
        args.prototypes
    )
    development_raw, _, development_cases = load_fixture(args.development_fixtures)
    holdout_raw, cases = load_provider_holdout(
        args.holdout_fixtures,
        development_candidate_texts=_normalized_candidate_texts(development_cases),
    )
    manifest_raw, manifest = load_evaluation_manifest(
        args.manifest,
        expected_sha256=args.expected_manifest_sha256,
    )

    expected_files = _expected_manifest_files(
        project_root=project_root,
        config_raw=config_raw,
        prototype_raw=prototype_raw,
        development_raw=development_raw,
        holdout_raw=holdout_raw,
    )
    if manifest["files"] != expected_files:
        mismatches = sorted(
            name
            for name in set(manifest["files"]) | set(expected_files)
            if manifest["files"].get(name) != expected_files.get(name)
        )
        raise ValueError(f"evaluation manifest file mismatch: {mismatches}")

    strategy = config["strategy"]
    gate = config["fresh_holdout_gate"]
    expected_policy = {
        "aggregation": strategy["aggregation"],
        "score_threshold": strategy["score_threshold"],
        "margin_threshold": strategy["margin_threshold"],
        "minimum_recall": gate["minimum_recall"],
        "false_merge_count": gate["false_merge_count"],
    }
    if manifest["policy"] != expected_policy:
        raise ValueError("evaluation manifest policy does not match frozen config")
    coverage = provider_coverage(cases)
    expected_coverage = {
        "minimum_provider_scored_count": MIN_PROVIDER_SCORED_COUNT,
        "minimum_provider_scored_per_dimension": MIN_PROVIDER_SCORED_PER_DIMENSION,
        **coverage,
    }
    if manifest["coverage"] != expected_coverage:
        raise ValueError("evaluation manifest coverage does not match holdout fixture")
    if gate.get("minimum_provider_scored_count") != MIN_PROVIDER_SCORED_COUNT:
        raise ValueError("frozen config weakens provider-scored count")
    if gate.get("minimum_provider_scored_per_dimension") != MIN_PROVIDER_SCORED_PER_DIMENSION:
        raise ValueError("frozen config weakens provider-scored dimension coverage")

    dependency_expectations = {
        config["development_evidence"]["fixture_sha256"]: hashlib.sha256(
            development_raw
        ).hexdigest(),
        config["development_evidence"]["runner_sha256"]: expected_files[
            "development_runner_sha256"
        ],
        config["dependencies"]["semantic_router_shadow_runner_sha256"]: expected_files[
            "shadow_runner_sha256"
        ],
        config["dependencies"]["config_module_sha256"]: expected_files["config_module_sha256"],
        config["dependencies"]["extractor_sha256"]: expected_files["extractor_sha256"],
        config["dependencies"]["vector_sha256"]: expected_files["vector_sha256"],
        strategy["router_sha256"]: expected_files["router_sha256"],
        strategy["prototype_sha256"]: expected_files["prototype_sha256"],
    }
    if any(expected != actual for expected, actual in dependency_expectations.items()):
        raise ValueError("frozen config dependency changed before A3.1 evaluation")

    settings = Settings.from_env()
    scores = _score_development(cases, prototypes, settings, strategy["aggregation"])
    provider_cases = _provider_scored_cases(cases)
    returned_by_dimension = Counter(
        case.expected_dimension for case in provider_cases if scores[case.case_id] is not None
    )
    runtime_provider_coverage = {
        "provider_score_returned_count": sum(
            scores[case.case_id] is not None for case in provider_cases
        ),
        "provider_score_returned_by_dimension": {
            dimension: returned_by_dimension.get(dimension, 0)
            for dimension in sorted(EXPECTED_DIMENSIONS)
        },
    }
    runtime_provider_coverage_passed = bool(
        runtime_provider_coverage["provider_score_returned_count"] >= MIN_PROVIDER_SCORED_COUNT
        and all(
            count >= MIN_PROVIDER_SCORED_PER_DIMENSION
            for count in runtime_provider_coverage["provider_score_returned_by_dimension"].values()
        )
    )
    decisions = route_development(
        cases,
        scores,
        score_threshold=strategy["score_threshold"],
        margin_threshold=strategy["margin_threshold"],
    )
    metrics = decision_metrics(cases, decisions)
    safety_passed, safety_failures = _safety_gate(cases, decisions)
    robustness_passed, robustness_failures = _category_gate(cases, decisions, ROBUSTNESS_CATEGORIES)
    passed = bool(
        metrics["false_merge_count"] == gate["false_merge_count"]
        and metrics["recall"] >= gate["minimum_recall"]
        and safety_passed
        and robustness_passed
        and runtime_provider_coverage_passed
    )

    provider_endpoint = (
        settings.google_api_base_url
        if settings.embedding_provider == "google"
        else settings.openai_base_url
    )
    report = {
        "benchmark": "Meno semantic policy router A3.1 provider-scored fresh holdout",
        "mode": "external-provider-fresh-holdout-read-only",
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "git_head": _git_head(),
        "command": [str(Path(__file__).resolve()), *sys.argv[1:]],
        "manifest": {
            "version": MANIFEST_VERSION,
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "out_of_band_sha_required": True,
        },
        "holdout": {
            "fixture_version": FIXTURE_VERSION,
            "fixture_sha256": hashlib.sha256(holdout_raw).hexdigest(),
            "case_count": len(cases),
            "threshold_search_performed": False,
        },
        "frozen_config": {
            "version": config["config_version"],
            "freeze_revision": config["freeze_revision"],
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
            **expected_files,
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
            **coverage,
            **runtime_provider_coverage,
            "runtime_provider_coverage_passed": runtime_provider_coverage_passed,
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
    raw_values = tuple(
        value
        for case in cases
        for value in (
            case.candidate.value,
            *(reference.value for reference in case.references),
        )
    )
    if _contains_raw_text(report, raw_values):
        raise ValueError("holdout artifact would contain raw candidate or reference text")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "manifest": report["manifest"],
                "provider_policy": report["provider_policy"],
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
