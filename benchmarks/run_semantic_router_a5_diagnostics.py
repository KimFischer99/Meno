from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.run_semantic_router_a3_development import load_prototypes
from benchmarks.run_semantic_router_a4_development import (
    _contains_raw_artifact_text,
    load_a4_development,
    provider_coverage,
    role_annotated_prototypes,
)
from benchmarks.run_semantic_router_shadow import (
    DimensionScore,
    _git_head,
    _package_versions,
    _safe_endpoint,
    _sha256,
)
from meno.config import Settings
from meno.semantic_router import (
    A5_STRATEGY_VERSION,
    ENSEMBLE_STRATEGY_VERSION,
    ROUTER_VERSION,
    DimensionPrototype,
    TopicGatedStrategy,
)
from meno.vector import make_embedder

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DEPENDENCIES = {
    "a3_development": ROOT / "benchmarks" / "run_semantic_router_a3_development.py",
    "a3_holdout": ROOT / "benchmarks" / "run_semantic_router_a3_holdout.py",
    "a31_holdout": ROOT / "benchmarks" / "run_semantic_router_a31_holdout.py",
    "a4_development": ROOT / "benchmarks" / "run_semantic_router_a4_development.py",
    "shadow_runner": ROOT / "benchmarks" / "run_semantic_router_shadow.py",
    "config": ROOT / "src" / "meno" / "config.py",
    "extractor": ROOT / "src" / "meno" / "extractor.py",
    "vector": ROOT / "src" / "meno" / "vector.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dump per-anchor/per-role score diagnostics for the A5 topic-gated "
            "design on the revealed 111-case development set"
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def polarity_annotated_prototypes(
    prototypes: tuple[DimensionPrototype, ...],
    metadata: dict[str, Any],
) -> tuple[DimensionPrototype, ...]:
    """Attach declared polarities to the role-annotated anchor prototypes."""

    annotated: list[DimensionPrototype] = []
    prototype_index = 0
    for dimension in sorted(metadata["dimensions"]):
        for anchor in metadata["dimensions"][dimension]:
            if prototype_index >= len(prototypes):
                raise ValueError("prototype metadata does not align with prototypes")
            prototype = prototypes[prototype_index]
            polarity = anchor.get("polarity")
            if prototype.name != dimension or polarity is None:
                raise ValueError("prototype polarity metadata does not align with prototypes")
            annotated.append(
                DimensionPrototype(
                    name=prototype.name,
                    description=prototype.description,
                    role=prototype.role,
                    polarity=polarity,
                )
            )
            prototype_index += 1
    if prototype_index != len(prototypes):
        raise ValueError("prototype metadata does not cover every prototype")
    return tuple(annotated)


def _baseline_top2_mean(anchor_scores: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        dimension: sum(sorted(labels.values(), reverse=True)[:2])
        / min(2, len(labels))
        for dimension, labels in anchor_scores.items()
    }


def main() -> None:
    args = parse_args()
    development_raw, cases, sources = load_a4_development(
        ROOT / "benchmarks" / "fixtures" / "semantic-normalization-v2.json",
        ROOT / "benchmarks" / "fixtures" / "semantic-normalization-v4-provider-holdout.json",
        ROOT / "artifacts" / "vps-stage4-a31-provider-holdout.json",
        ROOT / "benchmarks" / "fixtures" / "semantic-router-a31-evaluation-manifest.json",
    )
    prototype_raw, prototype_version, base_prototypes, prototype_metadata = load_prototypes(
        ROOT / "benchmarks" / "fixtures" / "semantic-prototypes-a3-v1.json"
    )
    prototypes = polarity_annotated_prototypes(
        role_annotated_prototypes(base_prototypes, prototype_metadata),
        prototype_metadata,
    )

    settings = Settings.from_env()
    strategy = TopicGatedStrategy(make_embedder(settings), prototypes)
    detailed = strategy.score_many_detailed([case.candidate for case in cases])
    strategy.embedder.close()

    source_by_case_id = {
        case.case_id: name for name, split_cases in sources.items() for case in split_cases
    }
    case_diagnostics: dict[str, dict[str, Any]] = {}
    for case, item in zip(cases, detailed, strict=True):
        if item is None:
            continue
        dimension_score = item.dimension_score
        baseline_scores = _baseline_top2_mean(dict(item.anchor_scores))
        ranked_baseline = sorted(baseline_scores, key=lambda name: baseline_scores[name], reverse=True)
        case_diagnostics[case.case_id] = {
            "source": source_by_case_id[case.case_id],
            "category": case.category,
            "expected_action": case.expected_action,
            "expected_dimension": case.expected_dimension,
            "anchor_scores": dict(item.anchor_scores),
            "topic_dimension": dimension_score.dimension,
            "topic_score": dimension_score.score,
            "topic_second_dimension": dimension_score.second_dimension,
            "topic_second_score": dimension_score.second_score,
            "topic_margin": dimension_score.margin,
            "top2_mean_dimension": ranked_baseline[0],
            "top2_mean_score": baseline_scores[ranked_baseline[0]],
            "top2_mean_margin": baseline_scores[ranked_baseline[0]]
            - baseline_scores[ranked_baseline[1]],
        }

    # Provider coverage uses the same designed-case contract as the A4 gate,
    # evaluated on the topic-gated ranking.
    designed_scores: dict[str, DimensionScore | None] = {
        case.case_id: (
            DimensionScore(
                dimension=case_diagnostics[case.case_id]["topic_dimension"],
                score=case_diagnostics[case.case_id]["topic_score"],
                second_dimension=case_diagnostics[case.case_id]["topic_second_dimension"],
                second_score=case_diagnostics[case.case_id]["topic_second_score"],
                margin=case_diagnostics[case.case_id]["topic_margin"],
                strategy_version=A5_STRATEGY_VERSION,
            )
            if case.case_id in case_diagnostics
            else None
        )
        for case in cases
    }
    coverage = provider_coverage(cases, designed_scores)

    runner_path = Path(__file__).resolve()
    router_path = ROOT / "src" / "meno" / "semantic_router.py"
    provider_endpoint = (
        settings.google_api_base_url
        if settings.embedding_provider == "google"
        else settings.openai_base_url
    )
    report: dict[str, Any] = {
        "benchmark": "Meno semantic router A5 topic-gated diagnostics",
        "mode": "external-provider-development-read-only-diagnostics",
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
                    "fixture_sha256": hashlib.sha256(
                        development_raw["a31_fixture"]
                    ).hexdigest(),
                    "artifact_sha256": hashlib.sha256(
                        development_raw["a31_artifact"]
                    ).hexdigest(),
                    "manifest_sha256": hashlib.sha256(
                        development_raw["a31_manifest"]
                    ).hexdigest(),
                },
            },
        },
        "prototypes": {
            "version": prototype_version,
            "sha256": hashlib.sha256(prototype_raw).hexdigest(),
            **prototype_metadata,
        },
        "source": {
            "runner_sha256": _sha256(runner_path),
            "router_sha256": _sha256(router_path),
            "strategy_versions": {
                "topic_gated": A5_STRATEGY_VERSION,
                "top2_mean_baseline": ENSEMBLE_STRATEGY_VERSION,
            },
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
            "coverage_passed": coverage["passed"],
            "coverage": coverage,
            "deterministic_bypass_count": sum(
                case.candidate.deterministic_slot is not None for case in cases
            ),
            "sensitive_blocked_count": sum(case.candidate.sensitive for case in cases),
            "injection_blocked_count": sum(case.candidate.injection_detected for case in cases),
            "raw_text_persisted": False,
        },
        "cases": case_diagnostics,
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
        raise ValueError("A5 diagnostics artifact would contain raw candidate or reference text")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "provider_policy": report["provider_policy"],
                "scored_case_count": len(case_diagnostics),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not coverage["passed"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
