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
)
from benchmarks.run_semantic_router_a3_holdout import _normalized_candidate_texts
from benchmarks.run_semantic_router_a4_development import provider_coverage
from benchmarks.run_semantic_router_a5_development import (
    DEFAULT_PROTOTYPES,
    EXPECTED_PROTOTYPE_SHA256,
    load_prototypes_a5,
)
from benchmarks.run_semantic_router_shadow import (
    EXPECTED_DIMENSIONS,
    ROBUSTNESS_CATEGORIES,
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
from meno.semantic_router import PolicyRouter, PrototypeEnsembleStrategy
from meno.vector import make_embedder

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOLDOUT_FIXTURE = (
    ROOT / "benchmarks" / "fixtures" / "semantic-normalization-v6-a5-fresh-holdout.json"
)
DEFAULT_DEVELOPMENT_FIXTURE = ROOT / "benchmarks" / "fixtures" / "semantic-normalization-v2.json"
DEFAULT_CONFIG = ROOT / "benchmarks" / "fixtures" / "semantic-router-a5-frozen-config.json"
DEFAULT_MANIFEST = ROOT / "benchmarks" / "fixtures" / "semantic-router-a5-evaluation-manifest.json"

MANIFEST_VERSION = "semantic-router-a5-evaluation-manifest-v1"
FIXTURE_VERSION = "semantic-normalization-v6-a5-fresh-holdout"
CONFIG_VERSION = "semantic-router-a5-frozen-config-v1"
FREEZE_REVISION = "a5-anchor-redesign-stable"
MIN_PROVIDER_SCORED_COUNT = 35
MIN_PROVIDER_SCORED_PER_DIMENSION = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen A5 policy once on a provider-scored fresh holdout"
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


def _expected_manifest_files(
    *,
    project_root: Path,
    config_raw: bytes,
    prototype_raw: bytes,
    development_raw: bytes,
    holdout_raw: bytes,
) -> dict[str, str]:
    return {
        "a3_development_runner_sha256": _sha256(
            project_root / "benchmarks" / "run_semantic_router_a3_development.py"
        ),
        "a4_development_runner_sha256": _sha256(
            project_root / "benchmarks" / "run_semantic_router_a4_development.py"
        ),
        "a5_development_runner_sha256": _sha256(
            project_root / "benchmarks" / "run_semantic_router_a5_development.py"
        ),
        "config_module_sha256": _sha256(project_root / "src" / "meno" / "config.py"),
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "development_fixture_sha256": hashlib.sha256(development_raw).hexdigest(),
        "extractor_sha256": _sha256(project_root / "src" / "meno" / "extractor.py"),
        "holdout_fixture_sha256": hashlib.sha256(holdout_raw).hexdigest(),
        "holdout_runner_sha256": _sha256(Path(__file__).resolve()),
        "prototype_sha256": hashlib.sha256(prototype_raw).hexdigest(),
        "router_sha256": _sha256(project_root / "src" / "meno" / "semantic_router.py"),
        "shadow_runner_sha256": _sha256(
            project_root / "benchmarks" / "run_semantic_router_shadow.py"
        ),
        "vector_sha256": _sha256(project_root / "src" / "meno" / "vector.py"),
    }


def load_frozen_config(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("frozen config root must be an object")
    if payload.get("config_version") != CONFIG_VERSION:
        raise ValueError("unsupported A5 frozen config version")
    if payload.get("freeze_revision") != FREEZE_REVISION:
        raise ValueError("A5 requires the anchor-redesign stable freeze revision")
    strategy = payload.get("strategy")
    if not isinstance(strategy, dict):
        raise TypeError("frozen config requires a strategy object")
    if strategy.get("variant") != "top2_mean" or strategy.get("aggregation") != "top2_mean":
        raise ValueError("A5 frozen strategy must be the selected top2_mean ensemble")
    gate = payload.get("fresh_holdout_gate")
    if not isinstance(gate, dict):
        raise TypeError("frozen config requires a fresh_holdout_gate object")
    if strategy.get("threshold_search_allowed_after_freeze") is not False:
        raise ValueError("frozen config must disable post-freeze threshold search")
    if gate.get("holdout_runs_allowed") != 1:
        raise ValueError("A5 fresh holdout is single-run")
    return raw, payload


def load_a5_evaluation_manifest(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    """Load the A5 evaluation manifest against its out-of-band SHA-256."""

    expected_sha = expected_sha256.strip().lower()
    if (
        len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
    ):
        raise ValueError("expected manifest SHA-256 must be a lowercase SHA-256")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("evaluation manifest does not match the out-of-band SHA-256")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("evaluation manifest root must be an object")
    if payload.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("unsupported A5 evaluation manifest version")
    files = payload.get("files")
    policy = payload.get("policy")
    coverage = payload.get("coverage")
    if not all(isinstance(item, dict) for item in (files, policy, coverage)):
        raise TypeError("evaluation manifest requires files, policy, and coverage objects")
    assert isinstance(files, dict)
    for name, value in files.items():
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"manifest files.{name} must be a lowercase SHA-256")
    return raw, payload


def load_a5_fresh_holdout(
    path: Path,
    *,
    development_candidate_texts: set[str],
) -> tuple[bytes, tuple[FixtureCase, ...]]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("fresh holdout fixture root must be an object")
    if payload.get("fixture_version") != FIXTURE_VERSION:
        raise ValueError("unsupported A5 fresh holdout fixture version")
    if payload.get("mode") != "fresh-holdout-provider-scored":
        raise ValueError("fresh holdout fixture mode is invalid")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) < 42:
        raise ValueError("fresh holdout requires at least 42 cases")

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
            raise ValueError("fresh holdout candidate overlaps revealed development text")
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
    missing_categories = (SAFETY_CATEGORY_SET | ROBUSTNESS_CATEGORIES) - categories
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

    coverage = designed_coverage(case_tuple)
    if coverage["provider_scored_count"] < MIN_PROVIDER_SCORED_COUNT:
        raise ValueError("fresh holdout has insufficient provider-scored cases")
    if any(
        count < MIN_PROVIDER_SCORED_PER_DIMENSION
        for count in coverage["provider_scored_by_dimension"].values()
    ):
        raise ValueError("fresh holdout has insufficient provider-scored dimension coverage")
    return raw, case_tuple


SAFETY_CATEGORY_SET = {
    "cross_user_filter",
    "cross_channel_filter",
    "inactive_reference_filter",
    "sensitive_reference_filter",
    "sensitive_candidate",
    "injection_candidate",
    "ambiguous_semantic_key",
    "explicit_feedback_protected",
}


def designed_coverage(cases: tuple[FixtureCase, ...]) -> dict[str, Any]:
    scored = tuple(
        case
        for case in cases
        if (
            case.candidate.deterministic_slot is None
            and not case.candidate.sensitive
            and not case.candidate.injection_detected
            and case.expected_dimension is not None
        )
    )
    by_dimension = Counter(case.expected_dimension for case in scored)
    return {
        "provider_scored_count": len(scored),
        "provider_scored_by_dimension": {
            dimension: by_dimension.get(dimension, 0) for dimension in sorted(EXPECTED_DIMENSIONS)
        },
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    config_raw, config = load_frozen_config(args.config)
    prototype_raw, prototype_version, prototypes, prototype_metadata = load_prototypes_a5(
        args.prototypes
    )
    if hashlib.sha256(prototype_raw).hexdigest() != EXPECTED_PROTOTYPE_SHA256:
        raise ValueError("A5 holdout prototype input changed")
    development_raw, _, development_cases = load_fixture(args.development_fixtures)
    holdout_raw, cases = load_a5_fresh_holdout(
        args.holdout_fixtures,
        development_candidate_texts=_normalized_candidate_texts(development_cases),
    )
    manifest_raw, manifest = load_a5_evaluation_manifest(
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

    # Continuity: the frozen strategy must still be bound to today's bytes.
    if strategy["router_sha256"] != expected_files["router_sha256"]:
        raise ValueError("router changed since the A5 development freeze")
    if strategy["prototype_sha256"] != expected_files["prototype_sha256"]:
        raise ValueError("prototypes changed since the A5 development freeze")

    expected_policy = {
        "aggregation": strategy["aggregation"],
        "score_threshold": strategy["score_threshold"],
        "margin_threshold": strategy["margin_threshold"],
        "minimum_recall": gate["minimum_recall"],
        "false_merge_count": gate["false_merge_count"],
    }
    if manifest["policy"] != expected_policy:
        raise ValueError("evaluation manifest policy does not match frozen config")
    holdout_coverage = designed_coverage(cases)
    expected_coverage = {
        "minimum_provider_scored_count": MIN_PROVIDER_SCORED_COUNT,
        "minimum_provider_scored_per_dimension": MIN_PROVIDER_SCORED_PER_DIMENSION,
        **holdout_coverage,
    }
    if manifest.get("coverage") != expected_coverage:
        raise ValueError("evaluation manifest coverage does not match holdout fixture")
    if gate.get("minimum_provider_scored_count") != MIN_PROVIDER_SCORED_COUNT:
        raise ValueError("frozen config weakens provider-scored count")
    if gate.get("minimum_provider_scored_per_dimension") != MIN_PROVIDER_SCORED_PER_DIMENSION:
        raise ValueError("frozen config weakens provider-scored dimension coverage")

    settings = Settings.from_env()
    score_cases = [case for case in cases if case.candidate.deterministic_slot is None]
    embedder = make_embedder(settings)
    try:
        values = PrototypeEnsembleStrategy(
            embedder, prototypes, aggregation=strategy["aggregation"]
        ).score_many([case.candidate for case in score_cases])
    finally:
        embedder.close()
    scores: dict[str, Any] = {case.case_id: None for case in cases}
    scores.update({case.case_id: score for case, score in zip(score_cases, values, strict=True)})

    coverage = provider_coverage(cases, scores)
    runtime_coverage_passed = bool(coverage["passed"])

    router = PolicyRouter(strategy_version=strategy["strategy_version"])
    decisions = {
        case.case_id: router.decide(
            case.candidate,
            case.references,
            scores[case.case_id],
            strategy["score_threshold"],
            strategy["margin_threshold"],
        )
        for case in cases
    }
    metrics = decision_metrics(cases, decisions)
    safety_passed, safety_failures = _safety_gate(cases, decisions)
    robustness_passed, robustness_failures = _category_gate(cases, decisions, ROBUSTNESS_CATEGORIES)
    answer_style_ids = {
        case.case_id for case in cases if case.expected_dimension == "answer_style"
    }
    answer_style_failures = sorted(
        case.case_id
        for case in cases
        if case.case_id in answer_style_ids
        and not _case_correct(case, decisions[case.case_id])
    )
    passed = bool(
        runtime_coverage_passed
        and metrics["false_merge_count"] == gate["false_merge_count"]
        and metrics["recall"] >= gate["minimum_recall"]
        and safety_passed
        and robustness_passed
        and not answer_style_failures
    )

    provider_endpoint = (
        settings.google_api_base_url
        if settings.embedding_provider == "google"
        else settings.openai_base_url
    )
    report: dict[str, Any] = {
        "benchmark": "Meno semantic policy router A5 fresh holdout",
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
            "strategy_version": strategy["strategy_version"],
            "score_threshold": strategy["score_threshold"],
            "margin_threshold": strategy["margin_threshold"],
        },
        "prototypes": {
            "version": prototype_version,
            "sha256": hashlib.sha256(prototype_raw).hexdigest(),
            **prototype_metadata,
        },
        "source": expected_files,
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
            
                "runtime_provider_coverage_passed": runtime_coverage_passed
            ,
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
        "answer_style_failure_ids": answer_style_failures,
        "passed": passed,
        "cases": [
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
        ],
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
                "answer_style_failure_ids": answer_style_failures,
                "passed": passed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(3)


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


if __name__ == "__main__":
    main()
