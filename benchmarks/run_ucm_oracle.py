"""Replay deterministic user trajectories and score Meno's user-state model.

This benchmark evaluates canonical state and context activation directly. It
does not call an answer LLM and does not treat retrieval recall alone as User
Context Modeling success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from meno.api import build_service
from meno.config import Settings
from meno.db import Claim
from meno.schemas import IngestRequest, RetrieveRequest
from tests.fakes import TestEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("benchmarks/fixtures/ucm-oracle-v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enable-context-activation", action="store_true")
    parser.add_argument("--enable-user-token-materialization", action="store_true")
    parser.add_argument("--enable-preference-distributions", action="store_true")
    parser.add_argument("--enable-clarification-opportunities", action="store_true")
    return parser.parse_args()


def _validate_fixture(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "ucm-oracle-v1":
        raise ValueError("unsupported UCM oracle schema")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("UCM oracle fixture must contain cases")
    expected_cohorts = {"static", "drifting", "contradictory", "adversarial"}
    cohorts = {case.get("cohort") for case in cases}
    if cohorts != expected_cohorts:
        raise ValueError("UCM oracle fixture must contain exactly four cohorts")
    case_ids = [case.get("case_id") for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("UCM oracle case IDs must be unique")
    event_ids = [event.get("event_id") for case in cases for event in case.get("events", [])]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("UCM oracle event IDs must be unique")


def build_ingest_request(case: dict[str, Any], event: dict[str, Any]) -> IngestRequest:
    """The fixture's event-log contract, shared with the full-state replay lane."""
    return IngestRequest.model_validate(
        {
            "user_id": case["user_id"],
            "event_id": event["event_id"],
            "occurred_at": event["occurred_at"],
            "source": {
                "type": event["source"],
                "profile": "ucm-oracle",
                "session_id": case["case_id"],
            },
            "content": {"role": event["role"], "text": event["text"]},
            "consent_scope": ["personalization", "task_planning"],
            "metadata": {"benchmark": "ucm-oracle-v1"},
        }
    )


def build_retrieve_request(case: dict[str, Any], oracle: dict[str, Any]) -> RetrieveRequest:
    """The fixture's query contract, shared with the full-state replay lane."""
    return RetrieveRequest.model_validate(
        {
            "user_id": oracle.get("user_id", case["user_id"]),
            "purpose": oracle.get("purpose", "response_personalization"),
            "context": {
                "query": oracle["query"],
                "task_type": oracle["task_type"],
                "as_of": oracle["as_of"],
                "platform": "ucm-oracle",
            },
            "constraints": {
                "max_facets": oracle.get("max_facets", 8),
                "min_confidence": oracle.get("min_confidence", 0.5),
                "allow_sensitive": oracle.get("allow_sensitive", False),
            },
        }
    )


def _claim_record(claim: Claim, claims_by_id: dict[str, Claim]) -> dict[str, Any]:
    successor = claims_by_id.get(claim.superseded_by_id or "")
    return {
        "kind": claim.kind,
        "value": claim.value,
        # Stance makes a reversal assertable: both directions of a preference share
        # the object, so "value" alone cannot distinguish them.
        "stance": claim.stance,
        "status": claim.status,
        "sensitive": claim.sensitive,
        "origin_role": claim.origin_role,
        "evidence_ids": sorted(evidence.event_id for evidence in claim.evidence),
        "superseded_by_value": successor.value if successor else None,
        "superseded_by_stance": successor.stance if successor else None,
    }


def _matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def _state_assertions(
    actual: list[dict[str, Any]], expected: dict[str, Any]
) -> list[dict[str, Any]]:
    assertions: list[dict[str, Any]] = []
    for status in ("active", "superseded"):
        candidates = [record for record in actual if record["status"] == status]
        for oracle in expected.get(status, []):
            passed = any(_matches(record, oracle) for record in candidates)
            label = oracle["value"]
            if oracle.get("stance"):
                label = f"{label}[{oracle['stance']}]"
            assertions.append(
                {
                    "name": f"state.{status}.{label}",
                    "passed": passed,
                    "expected": oracle,
                }
            )
    values = {record["value"] for record in actual}
    for value in expected.get("absent_values", []):
        assertions.append(
            {
                "name": f"state.absent.{value}",
                "passed": value not in values,
                "expected": value,
            }
        )
    return assertions


def _facet_values(facets: list[Any]) -> list[str]:
    values: list[str] = []
    for facet in facets:
        if isinstance(facet.value, dict):
            values.append(str(facet.value.get("current", "")))
        else:
            values.append(str(facet.value))
    return values


def _query_result(service, case: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    response = service.retrieve(build_retrieve_request(case, oracle))
    values = _facet_values(response.facets)
    clarification_enabled = service.settings.clarification_opportunities_enabled
    expected_decision = (
        oracle["expected_decision"]
        if clarification_enabled
        else oracle.get("expected_decision_without_clarification", oracle["expected_decision"])
    )
    expected_values = (
        oracle.get("expected_values", [])
        if clarification_enabled
        else oracle.get("expected_values_without_clarification", oracle.get("expected_values", []))
    )
    forbidden_values = (
        oracle.get("forbidden_values", [])
        if clarification_enabled
        else oracle.get(
            "forbidden_values_without_clarification",
            oracle.get("forbidden_values", []),
        )
    )
    clarification_reasons = [
        opportunity.reason for opportunity in response.clarification_opportunities
    ]
    expected_clarification_reasons = oracle.get("expected_clarification_reasons", [])
    decision = (
        "clarify"
        if response.clarification_opportunities
        else "inject"
        if response.facets
        else "abstain"
    )
    checks = {
        "decision": decision == expected_decision,
        "expected_values": all(value in values for value in expected_values),
        "forbidden_values": all(value not in values for value in forbidden_values),
        "clarification_reasons": (
            sorted(clarification_reasons) == sorted(expected_clarification_reasons)
            if clarification_enabled
            else not clarification_reasons
        ),
        "lineage": all(facet.evidence_ids for facet in response.facets)
        and all(opportunity.evidence_ids for opportunity in response.clarification_opportunities),
    }
    allowed_values = set(expected_values)
    context_precision = (
        sum(value in allowed_values for value in values) / len(values) if values else 1.0
    )
    return {
        "query_id": oracle["query_id"],
        "expected_decision": expected_decision,
        "actual_decision": decision,
        "selected_values": values,
        "clarification_reasons": clarification_reasons,
        "clarification_claim_ids": [
            opportunity.claim_id for opportunity in response.clarification_opportunities
        ],
        "checks": checks,
        "passed": all(checks.values()),
        "context_precision": context_precision,
        "facet_count": len(values),
        "degraded": response.degraded,
    }


def _metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    state = [item for case in case_results for item in case["state_assertions"]]
    queries = [item for case in case_results for item in case["queries"]]
    checks = [passed for query in queries for passed in query["checks"].values()]
    snapshot_checks = [
        case["snapshot_matches_canonical"]
        for case in case_results
        if case.get("snapshot_matches_canonical") is not None
    ]
    distribution_checks = [
        check for case in case_results for check in case.get("preference_distribution_checks", [])
    ]
    clarification_checks = [
        check for case in case_results for check in case.get("clarification_opportunity_checks", [])
    ]
    return {
        "cases": len(case_results),
        "state_assertions_passed": sum(item["passed"] for item in state),
        "state_assertions_total": len(state),
        "query_cases_passed": sum(item["passed"] for item in queries),
        "query_cases_total": len(queries),
        "query_checks_passed": sum(checks),
        "query_checks_total": len(checks),
        "mean_context_precision": (
            sum(item["context_precision"] for item in queries) / len(queries) if queries else 0.0
        ),
        "unexpected_degraded": sum(item["degraded"] for item in queries),
        "snapshot_checks_passed": sum(snapshot_checks),
        "snapshot_checks_total": len(snapshot_checks),
        "preference_distribution_checks_passed": sum(distribution_checks),
        "preference_distribution_checks_total": len(distribution_checks),
        "clarification_opportunity_checks_passed": sum(clarification_checks),
        "clarification_opportunity_checks_total": len(clarification_checks),
    }


def run_benchmark(
    fixture_path: Path,
    *,
    context_activation_enabled: bool = False,
    user_token_materialization_enabled: bool = False,
    preference_distribution_enabled: bool = False,
    clarification_opportunities_enabled: bool = False,
) -> dict[str, Any]:
    raw = fixture_path.read_bytes()
    payload = json.loads(raw)
    _validate_fixture(payload)
    with tempfile.TemporaryDirectory(prefix="meno-ucm-oracle-") as directory:
        settings = Settings(
            database_url=f"sqlite:///{Path(directory) / 'meno.sqlite3'}",
            vector_mode="memory",
            embedding_dimension=256,
            worker_poll_seconds=0.01,
            context_activation_enabled=context_activation_enabled,
            user_token_materialization_enabled=user_token_materialization_enabled,
            preference_distribution_enabled=preference_distribution_enabled,
            clarification_opportunities_enabled=clarification_opportunities_enabled,
        )
        settings.validate()
        service = build_service(settings, embedder=TestEmbedder(256))
        try:
            case_results: list[dict[str, Any]] = []
            for case in payload["cases"]:
                for event in case["events"]:
                    service.ingest(
                        build_ingest_request(case, event), idempotency_key=event["event_id"]
                    )
                while service.process_outbox(limit=1000):
                    pass
                service.process_projection_outbox(limit=1000)

                with service.session_factory() as session:
                    claims = list(
                        session.scalars(
                            select(Claim)
                            .options(selectinload(Claim.evidence))
                            .where(Claim.user_id == case["user_id"])
                        ).all()
                    )
                    claims_by_id = {claim.id: claim for claim in claims}
                    actual_state = sorted(
                        (_claim_record(claim, claims_by_id) for claim in claims),
                        key=lambda record: (record["status"], record["value"]),
                    )
                state_assertions = _state_assertions(actual_state, case["expected_state"])
                snapshot_matches_canonical: bool | None = None
                preference_distribution_checks: list[bool] = []
                clarification_opportunity_checks: list[bool] = []
                if user_token_materialization_enabled:
                    token = service.user_token(case["user_id"])
                    snapshot_state = token["payload"]["active_state"]
                    canonical_active = [
                        record for record in actual_state if record["status"] == "active"
                    ]
                    snapshot_matches_canonical = {
                        (
                            item["kind"],
                            item["value"],
                            tuple(item["evidence_ids"]),
                        )
                        for item in snapshot_state
                    } == {
                        (
                            item["kind"],
                            item["value"],
                            tuple(item["evidence_ids"]),
                        )
                        for item in canonical_active
                    }
                    if preference_distribution_enabled:
                        actual_distributions = token["payload"]["preference_distributions"]
                        expected_distributions = case["expected_preference_distributions"]
                        preference_distribution_checks.append(
                            len(actual_distributions) == len(expected_distributions)
                        )
                        for expected in expected_distributions:
                            matching = [
                                item
                                for item in actual_distributions
                                if item["parameters"]["mode"] == expected["mode"]
                            ]
                            preference_distribution_checks.append(bool(matching))
                            if not matching:
                                continue
                            actual = matching[0]
                            preference_distribution_checks.extend(
                                [
                                    set(actual["parameters"]["labels"]) == set(expected["labels"]),
                                    actual["calibration_status"] == expected["calibration_status"],
                                    actual["parameters"]["mode_probability"]
                                    >= expected.get("mode_probability_min", 0.0),
                                ]
                            )
                    if clarification_opportunities_enabled:
                        actual_opportunities = token["payload"]["clarification_opportunities"]
                        expected_opportunities = case.get(
                            "expected_clarification_opportunities", []
                        )
                        clarification_opportunity_checks.append(
                            len(actual_opportunities) == len(expected_opportunities)
                        )
                        for expected in expected_opportunities:
                            clarification_opportunity_checks.append(
                                any(
                                    all(actual.get(key) == value for key, value in expected.items())
                                    for actual in actual_opportunities
                                )
                            )
                queries = [_query_result(service, case, query) for query in case["queries"]]
                case_results.append(
                    {
                        "case_id": case["case_id"],
                        "cohort": case["cohort"],
                        "state_assertions": state_assertions,
                        "state_passed": all(item["passed"] for item in state_assertions),
                        "snapshot_matches_canonical": snapshot_matches_canonical,
                        "preference_distribution_checks": (preference_distribution_checks),
                        "clarification_opportunity_checks": (clarification_opportunity_checks),
                        "queries": queries,
                    }
                )
        finally:
            service.close()

    metrics = _metrics(case_results)
    by_cohort = {
        cohort: _metrics([case for case in case_results if case["cohort"] == cohort])
        for cohort in sorted({case["cohort"] for case in case_results})
    }
    return {
        "benchmark": "Meno deterministic User Context Modeling oracle",
        "schema_version": payload["schema_version"],
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "answer_model_calls": 0,
        "context_activation_enabled": context_activation_enabled,
        "user_token_materialization_enabled": user_token_materialization_enabled,
        "preference_distribution_enabled": preference_distribution_enabled,
        "clarification_opportunities_enabled": clarification_opportunities_enabled,
        "evaluation_scope": [
            "canonical_state",
            "supersede",
            "context_activation",
            "abstain",
            "evidence_lineage",
            *(["materialized_snapshot_consistency"] if user_token_materialization_enabled else []),
            *(["preference_distribution"] if preference_distribution_enabled else []),
            *(["clarification_opportunity"] if clarification_opportunities_enabled else []),
        ],
        "aggregate": metrics,
        "by_cohort": by_cohort,
        "all_passed": (
            metrics["state_assertions_passed"] == metrics["state_assertions_total"]
            and metrics["query_cases_passed"] == metrics["query_cases_total"]
            and metrics["unexpected_degraded"] == 0
            and metrics["snapshot_checks_passed"] == metrics["snapshot_checks_total"]
            and metrics["preference_distribution_checks_passed"]
            == metrics["preference_distribution_checks_total"]
            and metrics["clarification_opportunity_checks_passed"]
            == metrics["clarification_opportunity_checks_total"]
        ),
        "cases": case_results,
    }


def main() -> None:
    args = parse_args()
    report = run_benchmark(
        args.fixture,
        context_activation_enabled=args.enable_context_activation,
        user_token_materialization_enabled=args.enable_user_token_materialization,
        preference_distribution_enabled=args.enable_preference_distributions,
        clarification_opportunities_enabled=args.enable_clarification_opportunities,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
