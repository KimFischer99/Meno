"""Deterministic preference-polarity oracle.

Scores whether the canonical store holds the right stance per preference
dimension, and whether a reversal supersedes instead of accumulating. This
isolates the capability that PersonaMem's ``track_full_preference_evolution``
depends on, without the token-overlap noise of the multiple-choice scorer:
those options are ~80% token-identical and turn on a handful of polarity words,
so an option-ranking metric cannot tell a stance error from a wording accident.

No answer model and no embedding provider: SQLite plus the deterministic test
embedder. A component oracle can block a production decision but never grant one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from sqlalchemy import select

from meno.api import build_service
from meno.config import Settings
from meno.db import Claim
from meno.schemas import IngestRequest
from tests.fakes import TestEmbedder

EMBEDDING_DIMENSION = 256
SCHEMA_VERSION = "preference-polarity-oracle-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("benchmarks/fixtures/preference-polarity-oracle-v1.json"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _validate_fixture(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported fixture schema: {payload.get('schema_version')!r}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("fixture must contain cases")
    case_ids = [case.get("case_id") for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case IDs must be unique")
    user_ids = [case.get("user_id") for case in cases]
    if len(set(user_ids)) != len(user_ids):
        raise ValueError("user IDs must be unique so cases stay isolated")
    event_ids = [event.get("event_id") for case in cases for event in case.get("events", [])]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("event IDs must be unique")
    for case in cases:
        for group in ("expected_active", "expected_superseded"):
            for item in case.get(group, []):
                if "value" not in item or "stance" not in item:
                    raise ValueError(f"{case.get('case_id')}: {group} entries need value and stance")


def _observed(claims: list[Claim], status: str) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "value": claim.value,
                "stance": claim.stance,
                "superseded_reason": claim.superseded_reason,
            }
            for claim in claims
            if claim.status == status and claim.kind == "preference"
        ),
        key=lambda item: (item["value"], item["stance"] or ""),
    )


def _match(observed: list[dict[str, Any]], expected: dict[str, Any]) -> bool:
    return any(
        all(record.get(key) == value for key, value in expected.items()) for record in observed
    )


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    directory = Path(tempfile.mkdtemp(prefix="meno-polarity-"))
    settings = Settings(
        database_url=f"sqlite:///{directory / 'meno.sqlite3'}",
        vector_mode="memory",
        embedding_dimension=EMBEDDING_DIMENSION,
    )
    service = build_service(settings, embedder=TestEmbedder(EMBEDDING_DIMENSION))
    try:
        for event in case["events"]:
            service.ingest(
                IngestRequest(
                    user_id=case["user_id"],
                    event_id=event["event_id"],
                    source={"type": "hermes_turn"},
                    content={"role": "user", "text": event["text"]},
                    consent_scope=["personalization", "task_planning"],
                ),
                f"key-{event['event_id']}",
            )
            service.process_outbox()
        with service.session_factory() as session:
            claims = list(
                session.scalars(select(Claim).where(Claim.user_id == case["user_id"])).all()
            )
            active = _observed(claims, "active")
            superseded = _observed(claims, "superseded")
    finally:
        service.close()
        shutil.rmtree(directory, ignore_errors=True)

    checks: list[dict[str, Any]] = []
    for group, observed in (("active", active), ("superseded", superseded)):
        for expected in case.get(f"expected_{group}", []):
            checks.append(
                {
                    "name": f"{group}.{expected['value']}[{expected['stance']}]",
                    "passed": _match(observed, expected),
                    "expected": expected,
                }
            )
    # Extra active preference claims mean the store accumulated instead of superseding.
    expected_active_count = len(case.get("expected_active", []))
    checks.append(
        {
            "name": "active.no_extra_claims",
            "passed": len(active) == expected_active_count,
            "expected": {"active_count": expected_active_count},
        }
    )
    stance_checks = [check for check in checks if check["name"] != "active.no_extra_claims"]
    return {
        "case_id": case["case_id"],
        "family": case.get("family"),
        "observed_active": active,
        "observed_superseded": superseded,
        "checks": checks,
        "checks_passed": sum(check["passed"] for check in checks),
        "checks_total": len(checks),
        "stance_checks_passed": sum(check["passed"] for check in stance_checks),
        "stance_checks_total": len(stance_checks),
        "passed": all(check["passed"] for check in checks),
    }


def run_benchmark(fixture_path: Path) -> dict[str, Any]:
    raw = fixture_path.read_bytes()
    payload = json.loads(raw)
    _validate_fixture(payload)
    results = [run_case(case) for case in payload["cases"]]
    by_family: dict[str, dict[str, int]] = {}
    for result in results:
        family = by_family.setdefault(
            result["family"] or "unspecified", {"cases": 0, "passed": 0}
        )
        family["cases"] += 1
        family["passed"] += int(result["passed"])
    return {
        "benchmark": "Meno preference polarity oracle",
        "schema_version": SCHEMA_VERSION,
        "evidence_grade": payload.get("evidence_grade", "deterministic_component_oracle"),
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "answer_model_calls": 0,
        "embedding_provider_calls": 0,
        "aggregate": {
            "cases": len(results),
            "cases_passed": sum(result["passed"] for result in results),
            "checks_passed": sum(result["checks_passed"] for result in results),
            "checks_total": sum(result["checks_total"] for result in results),
            "stance_checks_passed": sum(result["stance_checks_passed"] for result in results),
            "stance_checks_total": sum(result["stance_checks_total"] for result in results),
        },
        "by_family": dict(sorted(by_family.items())),
        "all_passed": all(result["passed"] for result in results),
        "cases": results,
    }


def main() -> None:
    args = parse_args()
    report = run_benchmark(args.fixture)
    aggregate = report["aggregate"]
    print(f"{'case':38} {'family':12} result")
    print("-" * 62)
    for result in report["cases"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"{result['case_id']:38} {result['family']!s:12} {status} "
            f"({result['checks_passed']}/{result['checks_total']})"
        )
    print()
    print(
        f"cases {aggregate['cases_passed']}/{aggregate['cases']} | "
        f"stance checks {aggregate['stance_checks_passed']}/{aggregate['stance_checks_total']} | "
        f"all checks {aggregate['checks_passed']}/{aggregate['checks_total']}"
    )
    print(f"all_passed: {report['all_passed']}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
