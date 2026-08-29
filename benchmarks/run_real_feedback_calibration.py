"""Read-only calibration evaluation from canonical prospective feedback.

Each feedback audit revision is paired with revision - 1 of the user's
materialized token. No user text, correction text, claim value, or token payload
is emitted in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from benchmarks.run_preference_calibration import _binary_metrics
from meno.db import AuditEvent, Claim, Feedback, UserTokenSnapshot

REQUIRED_TABLES = {
    "meno_audit_events",
    "meno_claims",
    "meno_feedback",
    "meno_user_token_snapshots",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=Path("benchmarks/fixtures/real-feedback-calibration-gate-v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _validate_gate_config(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "real-feedback-calibration-gate-v1":
        raise ValueError("unsupported real feedback calibration gate schema")
    bins = payload.get("ece_bins")
    if not isinstance(bins, int) or not 2 <= bins <= 100:
        raise ValueError("ece_bins must be an integer between 2 and 100")
    development_fraction = payload.get("development_fraction")
    if not isinstance(development_fraction, (int, float)) or not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be between 0 and 1")
    for section in ("data_readiness", "quality_gate"):
        if not isinstance(payload.get(section), dict):
            raise TypeError(f"{section} must be an object")


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _feedback_audit_matches(
    feedback: Feedback,
    audit: AuditEvent,
    claims_by_id: dict[str, Claim],
) -> bool:
    if audit.user_hash != _sha(feedback.user_id):
        return False
    if audit.decision.get("feedback_action") != feedback.action:
        return False
    if feedback.action in {"confirm", "reject"}:
        return audit.claim_id == feedback.claim_id
    replacement = claims_by_id.get(audit.claim_id or "")
    return replacement is not None and replacement.supersedes_id == feedback.claim_id


def _distribution_for_claim(
    snapshot: UserTokenSnapshot, claim_id: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    active_claims = [
        claim
        for claim in snapshot.payload.get("active_state", [])
        if claim.get("claim_id") == claim_id and claim.get("kind") == "preference"
    ]
    if len(active_claims) != 1:
        return None
    semantic_key = active_claims[0].get("semantic_key")
    distributions = [
        distribution
        for distribution in snapshot.payload.get("preference_distributions", [])
        if distribution.get("semantic_key") == semantic_key
    ]
    if len(distributions) != 1:
        return None
    return active_claims[0], distributions[0]


def extract_forecasts(session: Session) -> tuple[list[dict[str, Any]], dict[str, int]]:
    all_feedback_rows = list(
        session.scalars(select(Feedback).order_by(Feedback.created_at, Feedback.id)).all()
    )
    audit_rows = list(
        session.scalars(
            select(AuditEvent)
            .where(AuditEvent.event_name == "meno.feedback.applied")
            .order_by(AuditEvent.state_revision, AuditEvent.created_at, AuditEvent.id)
        ).all()
    )
    claims_by_id = {claim.id: claim for claim in session.scalars(select(Claim)).all()}
    feedback_rows = [
        feedback
        for feedback in all_feedback_rows
        if claims_by_id.get(feedback.claim_id) is not None
        and claims_by_id[feedback.claim_id].kind == "preference"
    ]
    snapshots = {
        (snapshot.user_id, snapshot.state_revision): snapshot
        for snapshot in session.scalars(select(UserTokenSnapshot)).all()
    }
    consumed_audit_ids: set[str] = set()
    excluded: Counter[str] = Counter()
    forecasts: list[dict[str, Any]] = []
    for feedback in feedback_rows:
        matching_audits = [
            audit
            for audit in audit_rows
            if audit.id not in consumed_audit_ids
            and _feedback_audit_matches(feedback, audit, claims_by_id)
        ]
        if not matching_audits:
            excluded["missing_feedback_audit"] += 1
            continue
        audit = matching_audits[0]
        consumed_audit_ids.add(audit.id)
        prior_revision = audit.state_revision - 1
        snapshot = snapshots.get((feedback.user_id, prior_revision))
        if snapshot is None:
            excluded["missing_prior_snapshot"] += 1
            continue
        matched = _distribution_for_claim(snapshot, feedback.claim_id)
        if matched is None:
            excluded["missing_prior_preference_distribution"] += 1
            continue
        active_claim, distribution = matched
        parameters = distribution.get("parameters", {})
        candidate_probability = parameters.get("actionable_probability")
        if candidate_probability is None:
            candidate_probability = parameters.get("mode_probability")
        extractor_probability = active_claim.get("confidence", {}).get("calibrated")
        if any(
            not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
            for probability in (candidate_probability, extractor_probability)
        ):
            excluded["invalid_probability"] += 1
            continue
        forecasts.append(
            {
                "sample_id": _sha(feedback.id),
                "feedback_action": feedback.action,
                "_feedback_created_at": feedback.created_at.isoformat(),
                "forecast_revision": prior_revision,
                "outcome_revision": audit.state_revision,
                "mode_correct": feedback.action == "confirm",
                "candidate_probability": float(candidate_probability),
                "extractor_probability": float(extractor_probability),
                "strategy_version": distribution.get("strategy_version"),
                "calibration_status": distribution.get("calibration_status"),
            }
        )
    forecasts.sort(key=lambda item: (item["_feedback_created_at"], item["sample_id"]))
    for forecast in forecasts:
        del forecast["_feedback_created_at"]
    return forecasts, dict(sorted(excluded.items()))


def _readiness_gate(
    forecasts: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    source_feedback_count: int,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    positives = sum(item["mode_correct"] for item in holdout)
    pairing_coverage = len(forecasts) / source_feedback_count if source_feedback_count else 0.0
    conditions = {
        "minimum_total_samples": len(forecasts) >= thresholds["minimum_total_samples"],
        "minimum_holdout_samples": len(holdout) >= thresholds["minimum_holdout_samples"],
        "minimum_holdout_positive_labels": positives
        >= thresholds["minimum_holdout_positive_labels"],
        "minimum_holdout_negative_labels": len(holdout) - positives
        >= thresholds["minimum_holdout_negative_labels"],
        "minimum_pairing_coverage": pairing_coverage >= thresholds["minimum_pairing_coverage"],
    }
    return {
        "thresholds": thresholds,
        "pairing_coverage": pairing_coverage,
        "conditions": conditions,
        "passed": all(conditions.values()),
        "failure_reasons": [name for name, passed in conditions.items() if not passed],
    }


def _quality_gate(metrics: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    candidate = metrics["candidate"]
    baseline = metrics["extractor_baseline"]
    available = candidate["sample_count"] > 0
    conditions = {
        "maximum_brier_score": available
        and candidate["brier_score"] <= thresholds["maximum_brier_score"],
        "maximum_ece": available and candidate["ece"] <= thresholds["maximum_ece"],
        "minimum_brier_improvement": available
        and baseline["brier_score"] - candidate["brier_score"]
        >= thresholds["minimum_brier_improvement"],
        "minimum_ece_improvement": available
        and baseline["ece"] - candidate["ece"] >= thresholds["minimum_ece_improvement"],
    }
    return {
        "thresholds": thresholds,
        "conditions": conditions,
        "passed": all(conditions.values()),
        "failure_reasons": [name for name, passed in conditions.items() if not passed],
    }


def run_benchmark(database_url: str, gate_config_path: Path) -> dict[str, Any]:
    raw_config = gate_config_path.read_bytes()
    config = json.loads(raw_config)
    _validate_gate_config(config)
    parsed_url = make_url(database_url)
    if parsed_url.get_backend_name() == "sqlite" and parsed_url.database not in {
        None,
        ":memory:",
    }:
        database_path = Path(parsed_url.database)
        if not database_path.is_file():
            raise FileNotFoundError("read-only calibration database does not exist")
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            transaction = connection.begin()
            if engine.dialect.name == "sqlite":
                connection.execute(text("PRAGMA query_only = ON"))
            elif engine.dialect.name == "postgresql":
                connection.execute(text("SET TRANSACTION READ ONLY"))
            tables = set(inspect(connection).get_table_names())
            missing_tables = sorted(REQUIRED_TABLES - tables)
            if missing_tables:
                raise ValueError(f"database is missing required tables: {missing_tables}")
            try:
                with Session(bind=connection) as session:
                    source_feedback_count = session.scalar(
                        select(func.count(Feedback.id))
                        .join(Claim, Claim.id == Feedback.claim_id)
                        .where(Claim.kind == "preference")
                    )
                    forecasts, excluded = extract_forecasts(session)
            finally:
                transaction.rollback()
    finally:
        engine.dispose()
    split_index = math.floor(len(forecasts) * config["development_fraction"])
    development = forecasts[:split_index]
    holdout = forecasts[split_index:]
    bins = config["ece_bins"]
    split_metrics = {
        split: {
            "candidate": _binary_metrics(rows, "candidate_probability", bins),
            "extractor_baseline": _binary_metrics(rows, "extractor_probability", bins),
        }
        for split, rows in (("development", development), ("holdout", holdout))
    }
    readiness = _readiness_gate(
        forecasts,
        holdout,
        int(source_feedback_count or 0),
        config["data_readiness"],
    )
    quality = _quality_gate(split_metrics["holdout"], config["quality_gate"])
    statuses = sorted(
        {item["calibration_status"] for item in forecasts if item["calibration_status"]}
    )
    production_go_eligible = (
        readiness["passed"] and quality["passed"] and statuses == ["calibrated"]
    )
    return {
        "benchmark": "Meno real prospective feedback calibration",
        "schema_version": config["schema_version"],
        "gate_config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "read_only": True,
        "privacy": {
            "raw_user_text_exported": False,
            "correction_text_exported": False,
            "claim_values_exported": False,
            "token_payloads_exported": False,
            "feedback_timestamps_exported": False,
        },
        "source_feedback_count": int(source_feedback_count or 0),
        "paired_forecast_count": len(forecasts),
        "excluded": excluded,
        "strategy_versions": sorted(
            {item["strategy_version"] for item in forecasts if item["strategy_version"]}
        ),
        "calibration_statuses": statuses,
        "splits": {
            "development_count": len(development),
            "holdout_count": len(holdout),
            **split_metrics,
        },
        "data_readiness": readiness,
        "quality_gate": quality,
        "production_go_eligible": production_go_eligible,
        "production_decision": "GO" if production_go_eligible else "NO-GO",
        "production_blockers": [
            *([] if readiness["passed"] else ["real feedback data readiness gate failed"]),
            *([] if quality["passed"] else ["real feedback quality gate failed"]),
            *([] if statuses == ["calibrated"] else ["strategy is not marked calibrated"]),
        ],
        "samples": forecasts,
    }


def main() -> None:
    args = parse_args()
    report = run_benchmark(args.database_url, args.gate_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
