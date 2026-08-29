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

from benchmarks.run_semantic_router_a4_development import (
    _contains_raw_artifact_text,
    load_a4_development,
    provider_coverage,
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
    _cosine_similarity,
    _validate_vector,
)
from meno.vector import make_embedder

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VARIANTS = ROOT / "benchmarks" / "fixtures" / "semantic-prototypes-a5-anchor-search.json"
SOURCE_DEPENDENCIES = {
    "a4_development": ROOT / "benchmarks" / "run_semantic_router_a4_development.py",
    "shadow_runner": ROOT / "benchmarks" / "run_semantic_router_shadow.py",
    "config": ROOT / "src" / "meno" / "config.py",
    "extractor": ROOT / "src" / "meno" / "extractor.py",
    "vector": ROOT / "src" / "meno" / "vector.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-batch anchor-redesign search: embed the 111-case development set "
            "against multiple A5 prototype variants and dump per-anchor cosines"
        )
    )
    parser.add_argument("--variants", type=Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_variants(path: Path) -> tuple[bytes, dict[str, dict[str, dict[str, str]]]]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("anchor search fixture root must be an object")
    if payload.get("fixture_version") != "semantic-prototypes-a5-anchor-search-v1":
        raise ValueError("unsupported anchor search fixture version")
    variants = payload.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise TypeError("anchor search fixture requires a variants object")
    return raw, variants


def variant_anchors(
    name: str, dimensions: dict[str, Any]
) -> list[tuple[str, str, str]]:
    """Return (dimension, role_label, text) rows for one variant."""

    if not isinstance(dimensions, dict) or set(dimensions) != {
        "answer_style",
        "beverage",
        "color",
        "food",
        "music",
        "programming_language",
        "sport",
    }:
        raise ValueError(f"variant {name} must contain exactly the seven controlled dimensions")
    rows: list[tuple[str, str, str]] = []
    for dimension in sorted(dimensions):
        anchors = dimensions[dimension]
        if not isinstance(anchors, dict):
            raise TypeError(f"variant {name} dimension {dimension} must be an object")
        for label, text in anchors.items():
            role = (
                "topic"
                if label.startswith("topic")
                else (
                    "polarity"
                    if label.split("_")[0] in {"affirmed", "negated", "contradiction"}
                    else None
                )
            )
            if role is None:
                raise ValueError(f"variant {name} has unsupported anchor label {label!r}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"variant {name} anchor {dimension}.{label} must be non-empty")
            rows.append((dimension, label, text))
    return rows


def main() -> None:
    args = parse_args()
    variants_raw, variants = load_variants(args.variants)
    development_raw, cases, sources = load_a4_development(
        ROOT / "benchmarks" / "fixtures" / "semantic-normalization-v2.json",
        ROOT / "benchmarks" / "fixtures" / "semantic-normalization-v4-provider-holdout.json",
        ROOT / "artifacts" / "vps-stage4-a31-provider-holdout.json",
        ROOT / "benchmarks" / "fixtures" / "semantic-router-a31-evaluation-manifest.json",
    )

    parsed: dict[str, list[tuple[str, str, str]]] = {}
    for name, variant in variants.items():
        if not isinstance(variant, dict) or not isinstance(variant.get("dimensions"), dict):
            raise TypeError(f"variant {name} must contain a dimensions object")
        parsed[name] = variant_anchors(name, variant["dimensions"])

    safe_indices = [
        index
        for index, case in enumerate(cases)
        if (
            case.candidate.deterministic_slot is None
            and not case.candidate.sensitive
            and not case.candidate.injection_detected
        )
    ]
    settings = Settings.from_env()
    texts = [cases[index].candidate.value for index in safe_indices]
    anchor_row_offsets: dict[str, tuple[int, int]] = {}
    for name, rows in parsed.items():
        anchor_row_offsets[name] = (len(texts), len(texts) + len(rows))
        texts.extend(row[2] for row in rows)

    embedder = make_embedder(settings)
    try:
        raw_vectors = embedder.embed_documents(texts)
    finally:
        embedder.close()
    vectors = [
        _validate_vector(vector, label=f"vector {index}")
        for index, vector in enumerate(raw_vectors)
    ]

    source_by_case_id = {
        case.case_id: name for name, split_cases in sources.items() for case in split_cases
    }

    def coverage_for(scores: dict[str, DimensionScore | None]) -> dict[str, Any]:
        return provider_coverage(cases, scores)

    variant_reports: dict[str, Any] = {}
    coverage_all_passed = True
    for name, rows in parsed.items():
        start, end = anchor_row_offsets[name]
        anchor_vectors = vectors[start:end]
        case_diagnostics: dict[str, Any] = {}
        designed_scores: dict[str, DimensionScore | None] = {}
        for result_index, case_index in enumerate(safe_indices):
            case = cases[case_index]
            candidate_vector = vectors[result_index]
            anchor_scores: dict[str, dict[str, float]] = {}
            for row_index, (dimension, label, _text) in enumerate(rows):
                anchor_scores.setdefault(dimension, {})[label] = _cosine_similarity(
                    candidate_vector, anchor_vectors[row_index]
                )
            ranked = sorted(
                anchor_scores,
                key=lambda dim: max(anchor_scores[dim].values()),
                reverse=True,
            )
            best, second = ranked[0], ranked[1]
            score = max(anchor_scores[best].values())
            second_score = max(anchor_scores[second].values())
            dimension_score = DimensionScore(
                dimension=best,
                score=score,
                second_dimension=second,
                second_score=second_score,
                margin=score - second_score,
                strategy_version=A5_STRATEGY_VERSION,
            )
            designed_scores[case.case_id] = dimension_score
            case_diagnostics[case.case_id] = {
                "source": source_by_case_id[case.case_id],
                "category": case.category,
                "expected_action": case.expected_action,
                "expected_dimension": case.expected_dimension,
                "anchor_scores": anchor_scores,
            }
        coverage = coverage_for(designed_scores)
        coverage_all_passed = coverage_all_passed and bool(coverage["passed"])
        variant_reports[name] = {
            "design": variants[name].get("design"),
            "anchor_count": len(rows),
            "provider_coverage": coverage,
            "sha256_of_variant_payload": hashlib.sha256(
                json.dumps(variants[name], sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
            "cases": case_diagnostics,
        }

    runner_path = Path(__file__).resolve()
    router_path = ROOT / "src" / "meno" / "semantic_router.py"
    provider_endpoint = (
        settings.google_api_base_url
        if settings.embedding_provider == "google"
        else settings.openai_base_url
    )
    report: dict[str, Any] = {
        "benchmark": "Meno semantic router A5 anchor redesign search",
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
        "variants_fixture": {
            "path": str(DEFAULT_VARIANTS),
            "sha256": hashlib.sha256(variants_raw).hexdigest(),
            "variant_names": sorted(parsed),
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
            "all_variants_covered": coverage_all_passed,
            "raw_text_persisted": False,
        },
        "variants": variant_reports,
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
        raise ValueError("A5 anchor search artifact would contain raw candidate or reference text")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "provider_policy": report["provider_policy"],
                "variants": {
                    name: report_item["provider_coverage"]["passed"]
                    for name, report_item in variant_reports.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not coverage_all_passed:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
