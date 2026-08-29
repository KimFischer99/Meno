"""Score preference mode probabilities against prospective binary outcomes.

The future outcome label in each fixture case is never ingested into Meno. This
lane diagnoses probability calibration; synthetic evidence cannot grant a
production GO decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

from meno.api import build_service
from meno.config import Settings
from meno.schemas import IngestRequest
from tests.fakes import TestEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("benchmarks/fixtures/preference-calibration-v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enable-v2", action="store_true")
    return parser.parse_args()


def _validate_fixture(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "preference-calibration-v1":
        raise ValueError("unsupported preference calibration schema")
    if payload.get("evidence_grade") != "synthetic_diagnostic":
        raise ValueError("v1 calibration fixture must remain synthetic diagnostic evidence")
    bins = payload.get("ece_bins")
    if not isinstance(bins, int) or not 2 <= bins <= 100:
        raise ValueError("ece_bins must be an integer between 2 and 100")
    gate = payload.get("diagnostic_gate")
    if not isinstance(gate, dict):
        raise TypeError("diagnostic_gate must be an object")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("calibration fixture must contain cases")
    if {case.get("split") for case in cases} != {"development", "holdout"}:
        raise ValueError("calibration fixture must contain development and holdout splits")
    case_ids = [case.get("case_id") for case in cases]
    user_ids = [case.get("user_id") for case in cases]
    event_ids = [event.get("event_id") for case in cases for event in case.get("events", [])]
    for name, values in (
        ("case IDs", case_ids),
        ("user IDs", user_ids),
        ("event IDs", event_ids),
    ):
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"calibration {name} must be non-empty strings")
        if len(values) != len(set(values)):
            raise ValueError(f"calibration {name} must be unique")
    for case in cases:
        if not case.get("events"):
            raise ValueError("each calibration case requires evidence events")
        if not isinstance(case.get("mode_correct"), bool):
            raise TypeError("mode_correct must be boolean")
        if case.get("label_source") != "synthetic_future_feedback":
            raise ValueError("labels must be prospective synthetic future feedback")


def _binary_metrics(
    forecasts: list[dict[str, Any]], probability_key: str, bins: int
) -> dict[str, Any]:
    if not forecasts:
        return {
            "sample_count": 0,
            "positive_labels": 0,
            "negative_labels": 0,
            "brier_score": None,
            "ece": None,
            "negative_log_likelihood": None,
            "mean_confidence": None,
            "empirical_accuracy": None,
            "overconfidence": None,
            "bins": [],
        }
    probabilities = [float(item[probability_key]) for item in forecasts]
    labels = [int(item["mode_correct"]) for item in forecasts]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("forecast probabilities must be finite and in [0, 1]")
    brier = sum(
        (probability - label) ** 2 for probability, label in zip(probabilities, labels)
    ) / len(labels)
    epsilon = 1e-15
    nll = -sum(
        label * math.log(min(max(probability, epsilon), 1.0 - epsilon))
        + (1 - label) * math.log(min(max(1.0 - probability, epsilon), 1.0 - epsilon))
        for probability, label in zip(probabilities, labels)
    ) / len(labels)
    populated_bins: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        members = [
            (probability, label)
            for probability, label in zip(probabilities, labels)
            if min(int(probability * bins), bins - 1) == index
        ]
        if not members:
            continue
        average_confidence = sum(item[0] for item in members) / len(members)
        empirical_accuracy = sum(item[1] for item in members) / len(members)
        gap = abs(average_confidence - empirical_accuracy)
        ece += len(members) / len(labels) * gap
        populated_bins.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": len(members),
                "average_confidence": average_confidence,
                "empirical_accuracy": empirical_accuracy,
                "absolute_gap": gap,
            }
        )
    mean_confidence = sum(probabilities) / len(probabilities)
    empirical_accuracy = sum(labels) / len(labels)
    return {
        "sample_count": len(labels),
        "positive_labels": sum(labels),
        "negative_labels": len(labels) - sum(labels),
        "brier_score": brier,
        "ece": ece,
        "negative_log_likelihood": nll,
        "mean_confidence": mean_confidence,
        "empirical_accuracy": empirical_accuracy,
        "overconfidence": mean_confidence - empirical_accuracy,
        "bins": populated_bins,
    }


def _diagnostic_gate(holdout: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    candidate = holdout["candidate"]
    baseline = holdout["extractor_baseline"]
    conditions = {
        "coverage_complete": holdout["coverage"] == 1.0,
        "minimum_holdout_samples": candidate["sample_count"]
        >= thresholds["minimum_holdout_samples"],
        "minimum_positive_labels": candidate["positive_labels"]
        >= thresholds["minimum_positive_labels"],
        "minimum_negative_labels": candidate["negative_labels"]
        >= thresholds["minimum_negative_labels"],
        "maximum_brier_score": candidate["brier_score"] <= thresholds["maximum_brier_score"],
        "maximum_ece": candidate["ece"] <= thresholds["maximum_ece"],
        "minimum_brier_improvement": baseline["brier_score"] - candidate["brier_score"]
        >= thresholds["minimum_brier_improvement"],
        "minimum_ece_improvement": baseline["ece"] - candidate["ece"]
        >= thresholds["minimum_ece_improvement"],
    }
    return {
        "thresholds": thresholds,
        "conditions": conditions,
        "passed": all(conditions.values()),
        "failure_reasons": [name for name, passed in conditions.items() if not passed],
    }


def run_benchmark(
    fixture_path: Path, *, preference_distribution_v2_enabled: bool = False
) -> dict[str, Any]:
    raw = fixture_path.read_bytes()
    payload = json.loads(raw)
    _validate_fixture(payload)
    forecasts: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="meno-preference-calibration-") as directory:
        settings = Settings(
            database_url=f"sqlite:///{Path(directory) / 'meno.sqlite3'}",
            vector_mode="memory",
            embedding_dimension=256,
            user_token_materialization_enabled=True,
            preference_distribution_enabled=True,
            preference_distribution_v2_enabled=preference_distribution_v2_enabled,
        )
        settings.validate()
        service = build_service(settings, embedder=TestEmbedder(256))
        try:
            for case in payload["cases"]:
                for event in case["events"]:
                    request = IngestRequest.model_validate(
                        {
                            "user_id": case["user_id"],
                            "event_id": event["event_id"],
                            "occurred_at": event["occurred_at"],
                            "source": {
                                "type": "hermes_turn",
                                "profile": "preference-calibration-v1",
                                "session_id": case["case_id"],
                            },
                            "content": {"role": "user", "text": event["text"]},
                            "consent_scope": ["personalization"],
                            "metadata": {"benchmark": "preference-calibration-v1"},
                        }
                    )
                    service.ingest(request, idempotency_key=event["event_id"])
                while service.process_outbox(limit=1000):
                    pass
                token = service.user_token(case["user_id"])["payload"]
                distributions = token["preference_distributions"]
                matching_distributions = [
                    item
                    for item in distributions
                    if item["parameters"]["mode"] == case["expected_mode"]
                ]
                active_claims = [
                    item
                    for item in token["active_state"]
                    if item["kind"] == "preference" and item["value"] == case["expected_mode"]
                ]
                valid = (
                    len(distributions) == len(matching_distributions) == 1
                    and len(active_claims) == 1
                )
                forecast = {
                    "case_id": case["case_id"],
                    "split": case["split"],
                    "expected_mode": case["expected_mode"],
                    "mode_correct": case["mode_correct"],
                    "label_source": case["label_source"],
                    "valid": valid,
                }
                if valid:
                    distribution = matching_distributions[0]
                    forecast.update(
                        {
                            "candidate_probability": distribution["parameters"][
                                "actionable_probability"
                                if preference_distribution_v2_enabled
                                else "mode_probability"
                            ],
                            "conditional_mode_probability": distribution["parameters"][
                                "mode_probability"
                            ],
                            "extractor_probability": active_claims[0]["confidence"]["calibrated"],
                            "strategy_version": distribution["strategy_version"],
                            "calibration_status": distribution["calibration_status"],
                            "support_event_ids": distribution["parameters"]["support_event_ids"],
                        }
                    )
                forecasts.append(forecast)
        finally:
            service.close()

    split_reports: dict[str, Any] = {}
    for split in ("development", "holdout"):
        split_cases = [item for item in forecasts if item["split"] == split]
        valid_cases = [item for item in split_cases if item["valid"]]
        split_reports[split] = {
            "case_count": len(split_cases),
            "valid_forecast_count": len(valid_cases),
            "coverage": len(valid_cases) / len(split_cases),
            "candidate": _binary_metrics(valid_cases, "candidate_probability", payload["ece_bins"]),
            "extractor_baseline": _binary_metrics(
                valid_cases, "extractor_probability", payload["ece_bins"]
            ),
        }
    gate = _diagnostic_gate(split_reports["holdout"], payload["diagnostic_gate"])
    strategy_versions = sorted({item["strategy_version"] for item in forecasts if item["valid"]})
    calibration_statuses = sorted(
        {item["calibration_status"] for item in forecasts if item["valid"]}
    )
    return {
        "benchmark": "Meno preference distribution calibration diagnostic",
        "schema_version": payload["schema_version"],
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "evidence_grade": payload["evidence_grade"],
        "answer_model_calls": 0,
        "future_labels_ingested": False,
        "preference_distribution_v2_enabled": preference_distribution_v2_enabled,
        "strategy_versions": strategy_versions,
        "calibration_statuses": calibration_statuses,
        "splits": split_reports,
        "diagnostic_gate": gate,
        "production_go_eligible": False,
        "production_decision": "NO-GO",
        "production_blockers": [
            "synthetic labels cannot establish real-world calibration",
            *([] if gate["passed"] else ["diagnostic calibration gate failed"]),
        ],
        "cases": forecasts,
    }


def main() -> None:
    args = parse_args()
    report = run_benchmark(
        args.fixture,
        preference_distribution_v2_enabled=args.enable_v2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
