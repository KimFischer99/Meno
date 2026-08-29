from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from meno.config import Settings
from meno.extractor import preference_slot
from meno.semantic_resolver import RESOLVER_VERSION, cosine_similarity
from meno.vector import make_embedder

DEFAULT_FIXTURE = Path(__file__).parent / "fixtures" / "semantic-normalization-v1.json"
DEFAULT_THRESHOLDS = tuple(round(0.50 + index * 0.025, 3) for index in range(21))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the read-only semantic-key shadow resolver"
    )
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
        help="comma-separated cosine thresholds",
    )
    parser.add_argument("--minimum-recall", type=float, default=0.50)
    return parser.parse_args()


def threshold_metrics(scored_pairs: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    true_positive = false_positive = true_negative = false_negative = 0
    for pair in scored_pairs:
        predicted = pair["score"] is not None and pair["score"] >= threshold
        expected = pair["expected_merge"]
        if predicted and expected:
            true_positive += 1
        elif predicted:
            false_positive += 1
        elif expected:
            false_negative += 1
        else:
            true_negative += 1
    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive
    predicted_positive = true_positive + false_positive
    precision = true_positive / predicted_positive if predicted_positive else 1.0
    recall = true_positive / positive_count if positive_count else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_merge_rate": false_positive / negative_count if negative_count else 0.0,
        "missed_merge_rate": false_negative / positive_count if positive_count else 0.0,
    }


def recommend_threshold(
    metrics: list[dict[str, Any]], *, minimum_recall: float
) -> dict[str, Any] | None:
    eligible = [
        item for item in metrics if item["false_positive"] == 0 and item["recall"] >= minimum_recall
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda item: (item["threshold"], item["recall"]))


def recall_frontier(
    metrics: list[dict[str, Any]], *, minimum_recall: float
) -> dict[str, Any] | None:
    eligible = [item for item in metrics if item["recall"] >= minimum_recall]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            item["false_positive"],
            -item["precision"],
            -item["threshold"],
        ),
    )


def load_fixture(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("fixture root must be an object")
    if payload.get("fixture_version") != "semantic-normalization-v1":
        raise ValueError("unsupported semantic normalization fixture version")
    required_scope = {
        "candidate_slot": "missing",
        "reference_status": "active",
        "same_user": True,
        "same_kind": True,
        "same_semantic_channel": True,
    }
    if payload.get("scope") != required_scope:
        raise ValueError("fixture scope does not match the Phase A resolver boundary")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("fixture must contain a non-empty pairs list")
    seen_ids: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            raise TypeError("each fixture pair must be an object")
        pair_id = pair.get("id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in seen_ids:
            raise ValueError("fixture pair ids must be unique non-empty strings")
        seen_ids.add(pair_id)
        if not all(
            isinstance(pair.get(key), str) and pair[key].strip()
            for key in ("category", "candidate", "reference")
        ):
            raise ValueError(f"fixture pair {pair_id} has invalid text fields")
        if not isinstance(pair.get("expected_merge"), bool):
            raise TypeError(f"fixture pair {pair_id} requires expected_merge boolean")
        if preference_slot(pair["candidate"]):
            raise ValueError(
                f"fixture pair {pair_id} candidate is not slot-less under the current extractor"
            )
        if not isinstance(pair.get("provider_allowed", True), bool):
            raise TypeError(f"fixture pair {pair_id} provider_allowed must be boolean")
    if not any(pair["expected_merge"] for pair in pairs):
        raise ValueError("fixture requires at least one positive pair")
    if not any(not pair["expected_merge"] for pair in pairs):
        raise ValueError("fixture requires at least one negative pair")
    return raw, payload


def parse_thresholds(value: str) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not thresholds or any(not 0.0 <= item <= 1.0 for item in thresholds):
        raise ValueError("thresholds must contain values between 0 and 1")
    return thresholds


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.minimum_recall <= 1.0:
        raise ValueError("minimum recall must be between 0 and 1")
    fixture_raw, fixture = load_fixture(args.fixtures)
    thresholds = parse_thresholds(args.thresholds)
    pairs: list[dict[str, Any]] = fixture["pairs"]
    unique_texts = list(
        dict.fromkeys(
            text
            for pair in pairs
            if pair.get("provider_allowed", True)
            for text in (pair["candidate"], pair["reference"])
        )
    )

    settings = Settings.from_env()
    embedder = make_embedder(settings)
    try:
        vectors = embedder.embed_documents(unique_texts)
    finally:
        embedder.close()
    if len(vectors) != len(unique_texts):
        raise ValueError("embedder returned an unexpected number of vectors")
    vector_by_text = dict(zip(unique_texts, vectors, strict=True))
    scored_pairs = [
        {
            "id": pair["id"],
            "category": pair["category"],
            "expected_merge": pair["expected_merge"],
            "provider_allowed": pair.get("provider_allowed", True),
            "score": (
                cosine_similarity(
                    vector_by_text[pair["candidate"]],
                    vector_by_text[pair["reference"]],
                )
                if pair.get("provider_allowed", True)
                else None
            ),
        }
        for pair in pairs
    ]
    metrics = [threshold_metrics(scored_pairs, threshold) for threshold in thresholds]
    recommendation = recommend_threshold(metrics, minimum_recall=args.minimum_recall)
    safe_frontier = recommend_threshold(metrics, minimum_recall=0.0)
    minimum_recall_frontier = recall_frontier(metrics, minimum_recall=args.minimum_recall)
    diagnostic = recommendation or minimum_recall_frontier or safe_frontier
    diagnostic_threshold = diagnostic["threshold"] if diagnostic is not None else None
    false_merge_ids = [
        pair["id"]
        for pair in scored_pairs
        if diagnostic_threshold is not None
        and pair["score"] is not None
        and pair["score"] >= diagnostic_threshold
        and not pair["expected_merge"]
    ]
    missed_merge_ids = [
        pair["id"]
        for pair in scored_pairs
        if diagnostic_threshold is not None
        and (pair["score"] is None or pair["score"] < diagnostic_threshold)
        and pair["expected_merge"]
    ]

    report = {
        "benchmark": "Meno semantic-key shadow calibration",
        "mode": "offline-read-only",
        "created_at": datetime.now(UTC).isoformat(),
        "fixture_version": fixture["fixture_version"],
        "fixture_sha256": hashlib.sha256(fixture_raw).hexdigest(),
        "resolver_version": RESOLVER_VERSION,
        "service_config": {
            "environment": settings.environment,
            "extractor_version": settings.extractor_version,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "embedding_projection_version": settings.embedding_projection_version,
        },
        "pair_count": len(scored_pairs),
        "positive_count": sum(pair["expected_merge"] for pair in scored_pairs),
        "negative_count": sum(not pair["expected_merge"] for pair in scored_pairs),
        "provider_blocked_count": sum(not pair["provider_allowed"] for pair in scored_pairs),
        "minimum_recall": args.minimum_recall,
        "passed": recommendation is not None,
        "recommended_threshold": recommendation,
        "safe_frontier": safe_frontier,
        "minimum_recall_frontier": minimum_recall_frontier,
        "diagnostic_threshold": diagnostic_threshold,
        "false_merge_ids": false_merge_ids,
        "missed_merge_ids": missed_merge_ids,
        "positive_score_range": {
            "minimum": min(
                pair["score"]
                for pair in scored_pairs
                if pair["expected_merge"] and pair["score"] is not None
            ),
            "maximum": max(
                pair["score"]
                for pair in scored_pairs
                if pair["expected_merge"] and pair["score"] is not None
            ),
        },
        "negative_score_range": {
            "minimum": min(
                pair["score"]
                for pair in scored_pairs
                if not pair["expected_merge"] and pair["score"] is not None
            ),
            "maximum": max(
                pair["score"]
                for pair in scored_pairs
                if not pair["expected_merge"] and pair["score"] is not None
            ),
        },
        "threshold_metrics": metrics,
        "scored_pairs": scored_pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in report.items() if key != "scored_pairs"}, indent=2)
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
