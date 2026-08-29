"""Full-state replay determinism lane (Meno_SPEC.md:1393, Gate Charter #12).

SPEC pins a frozen event log per release, replays it from an empty store, and
compares the *whole* semantic state rather than an aggregate score. The value is
stated in the SPEC itself: it detects whether swapping the extractor, the
embedding, or the decay policy quietly changed one person's model. Aggregate
benchmark deltas cannot show that; a state-level diff can.

Eight comparison dimensions are named in SPEC :1393. Seven are implemented here;
``community_memberships`` has no table, projection, or facet path in the codebase
and is reported as **uncovered** rather than silently digested as empty.

Two properties make this lane hard to satisfy vacuously:

1. Wall-clock and per-process identifiers (``uuid4`` audit ids, trace ids,
   ``created_at``, supersede ``valid_to``) cannot be compared across replays, so
   they are excluded. Every exclusion is listed per dimension in the report,
   so the normalization surface is auditable instead of implicit.
2. A digest that excludes too much would pass forever. ``--self-check`` (on by
   default) therefore perturbs the inputs and *requires* the named dimensions to
   change. A digest that no longer responds to its own perturbation fails the
   lane exactly like a drifted state would.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from benchmarks.run_ucm_oracle import (
    _validate_fixture,
    build_ingest_request,
    build_retrieve_request,
)
from benchmarks.verify_invariants import audit_chain_summary
from meno.api import build_service
from meno.config import Settings
from meno.db import AuditEvent, Claim, PreferenceDistribution, UserTokenSnapshot
from meno.service import _aware
from tests.fakes import TestEmbedder

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE = ROOT / "benchmarks" / "fixtures" / "ucm-oracle-v1.json"
DEFAULT_BASELINE = ROOT / "benchmarks" / "fixtures" / "state-replay-baseline-v1.json"
BASELINE_SCHEMA = "state-replay-baseline-v1"

# SPEC :1393 comparison surface, in the SPEC's own order.
DIMENSIONS = (
    "claim_set",
    "claim_statuses",
    "preference_distributions",
    "community_memberships",
    "token_revision",
    "retrieval_results",
    "audit_lineage",
    "rendered_context",
)

# Fields a replay cannot reproduce, recorded per dimension so that the
# normalization surface is reviewable rather than buried in the digest code.
EXCLUDED_FIELDS: dict[str, tuple[str, ...]] = {
    "claim_set": ("created_at", "updated_at"),
    "claim_statuses": ("valid_to", "updated_at"),
    "preference_distributions": ("id", "updated_at"),
    "community_memberships": (),
    "token_revision": ("id", "created_at"),
    "retrieval_results": ("trace_id",),
    "audit_lineage": ("id", "trace_id", "prev_hash", "current_hash", "created_at"),
    "rendered_context": (),
}

UNCOVERED_DIMENSIONS: dict[str, str] = {
    "community_memberships": (
        "community_memberships is not implemented: no table, no projection, and no "
        "retrieval path exists (Gate Charter #17). Reporting it as uncovered rather "
        "than digesting an empty set, which would read as a passing comparison."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Freeze this run's digests as the release baseline instead of comparing",
    )
    parser.add_argument(
        "--no-self-check",
        action="store_true",
        help="Skip the digest sensitivity probes (they are the anti-vacuity guard)",
    )
    parser.add_argument(
        "--emit-state",
        action="store_true",
        help="Include the full per-dimension state in the report, not only digests",
    )
    return parser.parse_args()


def _canonical(value: Any) -> Any:
    """Round floats so identical state cannot differ by representation alone."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical(item) for item in value]
    return value


def _digest(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _ordered_claims(session) -> list[Claim]:
    return list(
        session.scalars(
            select(Claim)
            .options(selectinload(Claim.evidence))
            .order_by(Claim.user_id, Claim.semantic_key, Claim.id)
        ).all()
    )


def _claim_set(claims: list[Claim]) -> list[dict[str, Any]]:
    """Identity and content of every claim. Status lives in its own dimension."""
    return [
        {
            # uuid5 of the derivation key, so this is reproducible and worth comparing.
            "claim_id": claim.id,
            "derivation_key": claim.derivation_key,
            "user_id": claim.user_id,
            "kind": claim.kind,
            "semantic_channel": claim.semantic_channel,
            "semantic_key": claim.semantic_key,
            "value": claim.value,
            "stance": claim.stance,
            "sensitive": bool(claim.sensitive),
            "origin_role": claim.origin_role,
            "source_type": claim.source_type,
            "confidence": claim.confidence,
            "half_life_days": claim.half_life_days,
            "allowed_purposes": sorted(claim.allowed_purposes),
            "extractor_version": claim.extractor_version,
            "routing_slot": claim.routing_slot,
            "routing_basis": claim.routing_basis,
            "router_version": claim.router_version,
            "valid_from": _aware(claim.valid_from).isoformat(),
            "evidence": sorted(
                [item.event_id, item.relation] for item in claim.evidence
            ),
        }
        for claim in claims
    ]


def _claim_statuses(claims: list[Claim]) -> list[dict[str, Any]]:
    """Status and supersede topology.

    ``valid_to`` is stamped with wall clock on supersede, so only its presence is
    comparable; the link and the reason carry the semantics.
    """
    return [
        {
            "claim_id": claim.id,
            "semantic_key": claim.semantic_key,
            "status": claim.status,
            "supersedes_id": claim.supersedes_id,
            "superseded_by_id": claim.superseded_by_id,
            "superseded_reason": claim.superseded_reason,
            "valid_to_set": claim.valid_to is not None,
        }
        for claim in claims
    ]


def _preference_distributions(session) -> list[dict[str, Any]]:
    return [
        {
            "user_id": distribution.user_id,
            "semantic_key": distribution.semantic_key,
            "distribution_type": distribution.distribution_type,
            "strategy_version": distribution.strategy_version,
            "calibration_status": distribution.calibration_status,
            "parameters": distribution.parameters,
            "state_revision": distribution.state_revision,
            "content_hash": distribution.content_hash,
        }
        for distribution in session.scalars(
            select(PreferenceDistribution).order_by(
                PreferenceDistribution.user_id, PreferenceDistribution.semantic_key
            )
        ).all()
    ]


def _token_revisions(session) -> list[dict[str, Any]]:
    return [
        {
            "user_id": snapshot.user_id,
            "state_revision": snapshot.state_revision,
            "schema_version": snapshot.schema_version,
            "policy_version": snapshot.policy_version,
            "extractor_version": snapshot.extractor_version,
            "content_hash": snapshot.content_hash,
            "payload": snapshot.payload,
        }
        for snapshot in session.scalars(
            select(UserTokenSnapshot).order_by(
                UserTokenSnapshot.user_id, UserTokenSnapshot.state_revision
            )
        ).all()
    ]


def _audit_rows(session) -> list[AuditEvent]:
    # _audit forces strictly increasing created_at, so this ordering is total and
    # does not need the uuid4 primary key as a tie-break.
    return list(session.scalars(select(AuditEvent).order_by(AuditEvent.created_at)).all())


def _audit_lineage(rows: list[AuditEvent]) -> list[dict[str, Any]]:
    """Ordered decision lineage.

    The hash chain itself cannot be compared across replays: ``prev_hash`` folds in
    a ``uuid4`` trace id. Chain *integrity* is verified per replay by
    ``audit_chain_summary``; this digest covers what each entry claims.
    """
    return [
        {
            "position": position,
            "event_name": row.event_name,
            "user_hash": row.user_hash,
            "claim_id": row.claim_id,
            "action": row.action,
            "purpose": row.purpose,
            "decision": row.decision,
            "state_revision": row.state_revision,
            "source_event_ids": row.source_event_ids,
        }
        for position, row in enumerate(rows)
    ]


def _retrieval_results(
    service, payload: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay every fixture query; return retrieval state and rendered context.

    Rendered context is kept separate because SPEC :1393 lists it as its own
    comparison dimension, and a byte-exact render diff is the cheapest signal that
    injection wording moved without the underlying state moving.
    """
    retrieval: list[dict[str, Any]] = []
    rendered: list[dict[str, Any]] = []
    for case in payload["cases"]:
        for oracle in case["queries"]:
            response = service.retrieve(build_retrieve_request(case, oracle))
            retrieval.append(
                {
                    "case_id": case["case_id"],
                    "query_id": oracle["query_id"],
                    "state_revision": response.state_revision,
                    "token_revision_id": response.token_revision_id,
                    "degraded": response.degraded,
                    "policy_version": response.policy_version,
                    "facets": [
                        {
                            "claim_id": facet.claim_id,
                            "kind": facet.kind,
                            "value": facet.value,
                            "stance": facet.stance,
                            "relevance": facet.relevance,
                            "confidence": facet.confidence,
                            "evidence_ids": facet.evidence_ids,
                            "why_selected": facet.why_selected,
                        }
                        for facet in response.facets
                    ],
                    "clarification_opportunities": [
                        opportunity.model_dump()
                        for opportunity in response.clarification_opportunities
                    ],
                }
            )
            rendered.append(
                {
                    "case_id": case["case_id"],
                    "query_id": oracle["query_id"],
                    # Byte-exact: SPEC :1393 compares rendered context verbatim.
                    "rendered_context": response.rendered_context,
                }
            )
    return retrieval, rendered


def replay(
    payload: dict[str, Any],
    settings: Settings,
    *,
    embedding_dimension: int,
    event_limit_per_case: int | None = None,
) -> dict[str, Any]:
    """Replay the frozen event log into an empty store and capture full state."""
    with tempfile.TemporaryDirectory(prefix="meno-state-replay-") as directory:
        run_settings = dataclasses.replace(
            settings,
            database_url=f"sqlite:///{Path(directory) / 'meno.sqlite3'}",
            embedding_dimension=embedding_dimension,
        )
        run_settings.validate()
        service = build_service(run_settings, embedder=TestEmbedder(embedding_dimension))
        try:
            for case in payload["cases"]:
                events = case["events"]
                if event_limit_per_case is not None:
                    events = events[:event_limit_per_case]
                for event in events:
                    service.ingest(
                        build_ingest_request(case, event), idempotency_key=event["event_id"]
                    )
                while service.process_outbox(limit=1000):
                    pass
                service.process_projection_outbox(limit=1000)

            retrieval, rendered = _retrieval_results(service, payload)
            # Retrieval audits are buffered; they must reach the table before the
            # lineage dimension is read, or the digest would omit them entirely.
            service.flush_audit_buffer()

            with service.session_factory() as session:
                claims = _ordered_claims(session)
                audit_rows = _audit_rows(session)
                state = {
                    "claim_set": _claim_set(claims),
                    "claim_statuses": _claim_statuses(claims),
                    "preference_distributions": _preference_distributions(session),
                    "community_memberships": None,
                    "token_revision": _token_revisions(session),
                    "retrieval_results": retrieval,
                    "audit_lineage": _audit_lineage(audit_rows),
                    "rendered_context": rendered,
                }
                chain = audit_chain_summary(
                    [
                        {
                            "event_name": row.event_name,
                            "trace_id": row.trace_id,
                            "user_hash": row.user_hash,
                            "claim_id": row.claim_id,
                            "action": row.action,
                            "purpose": row.purpose,
                            "decision": row.decision,
                            "state_revision": row.state_revision,
                            "source_event_ids": row.source_event_ids,
                            "prev_hash": row.prev_hash,
                            "current_hash": row.current_hash,
                        }
                        for row in audit_rows
                    ]
                )
        finally:
            service.close()

    digests = {
        dimension: (None if dimension in UNCOVERED_DIMENSIONS else _digest(state[dimension]))
        for dimension in DIMENSIONS
    }
    return {"digests": digests, "state": state, "audit_chain": chain}


def _covered_dimensions() -> tuple[str, ...]:
    return tuple(item for item in DIMENSIONS if item not in UNCOVERED_DIMENSIONS)


def _changed_dimensions(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    return [
        dimension
        for dimension in _covered_dimensions()
        if left[dimension] != right[dimension]
    ]


# Each probe perturbs the replay and names the dimensions that MUST respond. A
# digest that stops responding to its own perturbation has been normalized into
# uselessness, and that is a lane failure, not a pass.
SELF_CHECK_PROBES: tuple[dict[str, Any], ...] = (
    {
        "probe": "truncated_event_log",
        "rationale": (
            "Dropping each case's last event must change the person's model. If it "
            "does not, the digests are not reading the event log."
        ),
        "must_change": (
            "claim_set",
            "claim_statuses",
            "token_revision",
            "retrieval_results",
            "rendered_context",
        ),
    },
    {
        "probe": "extractor_version",
        "rationale": (
            "SPEC :1393's stated purpose: detect that swapping the extractor changed "
            "stored state. Claim identity derives from the extractor version."
        ),
        "must_change": ("claim_set",),
    },
    {
        "probe": "embedding_dimension",
        "rationale": (
            "Swapping the embedding must be visible. Scores and ordering come from "
            "the embedder, so retrieval state must respond even when claims do not."
        ),
        "must_change": ("retrieval_results",),
    },
)


def run_self_check(
    payload: dict[str, Any], settings: Settings, baseline_digests: dict[str, Any]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for probe in SELF_CHECK_PROBES:
        name = probe["probe"]
        if name == "truncated_event_log":
            perturbed = replay(
                payload, settings, embedding_dimension=256, event_limit_per_case=1
            )
        elif name == "extractor_version":
            perturbed = replay(
                payload,
                dataclasses.replace(settings, extractor_version="meno-extractor-probe-0.0.0"),
                embedding_dimension=256,
            )
        elif name == "embedding_dimension":
            perturbed = replay(payload, settings, embedding_dimension=192)
        else:  # pragma: no cover - probe table is closed
            raise ValueError(f"unknown self-check probe: {name}")
        changed = _changed_dimensions(baseline_digests, perturbed["digests"])
        missing = [item for item in probe["must_change"] if item not in changed]
        results.append(
            {
                "probe": name,
                "rationale": probe["rationale"],
                "must_change": list(probe["must_change"]),
                "changed_dimensions": changed,
                "unresponsive_dimensions": missing,
                "passed": not missing,
            }
        )
    return results


def run_benchmark(
    fixture_path: Path,
    baseline_path: Path,
    *,
    self_check: bool = True,
    write_baseline: bool = False,
    emit_state: bool = False,
) -> dict[str, Any]:
    raw = fixture_path.read_bytes()
    payload = json.loads(raw)
    _validate_fixture(payload)

    # The full state surface only exists with these four flags on; a default-off
    # replay would silently skip token, distribution, and clarification state.
    settings = Settings(
        database_url="sqlite:///:memory:",
        vector_mode="memory",
        embedding_dimension=256,
        worker_poll_seconds=0.01,
        context_activation_enabled=True,
        user_token_materialization_enabled=True,
        preference_distribution_enabled=True,
        clarification_opportunities_enabled=True,
    )

    first = replay(payload, settings, embedding_dimension=256)
    second = replay(payload, settings, embedding_dimension=256)
    drifted = _changed_dimensions(first["digests"], second["digests"])

    baseline_status: dict[str, Any]
    if write_baseline:
        baseline_status = {"mode": "written", "path": str(baseline_path), "drift": []}
    elif baseline_path.is_file():
        frozen = json.loads(baseline_path.read_text(encoding="utf-8"))
        if frozen.get("schema_version") != BASELINE_SCHEMA:
            raise ValueError("unsupported state replay baseline schema")
        baseline_status = {
            "mode": "compared",
            "path": str(baseline_path),
            "fixture_sha256_matches": frozen.get("fixture_sha256")
            == hashlib.sha256(raw).hexdigest(),
            "drift": [
                dimension
                for dimension in _covered_dimensions()
                if frozen["digests"].get(dimension) != first["digests"][dimension]
            ],
        }
    else:
        baseline_status = {
            "mode": "absent",
            "path": str(baseline_path),
            "drift": [],
        }

    self_check_results = (
        run_self_check(payload, settings, first["digests"]) if self_check else []
    )

    chain_passed = bool(first["audit_chain"]["passed"]) and bool(second["audit_chain"]["passed"])
    baseline_ok = baseline_status["mode"] != "compared" or (
        not baseline_status["drift"] and baseline_status["fixture_sha256_matches"]
    )
    report = {
        "benchmark": "Meno full-state replay determinism (Meno_SPEC.md:1393)",
        "gate_charter_item": 12,
        "schema_version": BASELINE_SCHEMA,
        "fixture": str(fixture_path),
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "answer_model_calls": 0,
        "cohorts": sorted({case["cohort"] for case in payload["cases"]}),
        "dimensions": {
            dimension: {
                "covered": dimension not in UNCOVERED_DIMENSIONS,
                "digest": first["digests"][dimension],
                "excluded_fields": list(EXCLUDED_FIELDS[dimension]),
                **(
                    {"uncovered_reason": UNCOVERED_DIMENSIONS[dimension]}
                    if dimension in UNCOVERED_DIMENSIONS
                    else {}
                ),
            }
            for dimension in DIMENSIONS
        },
        "dimensions_covered": len(_covered_dimensions()),
        "dimensions_total": len(DIMENSIONS),
        "replay_determinism": {
            "passes": 2,
            "drifted_dimensions": drifted,
            "passed": not drifted,
        },
        "audit_chain": {
            "first_pass": first["audit_chain"],
            "second_pass": second["audit_chain"],
            "passed": chain_passed,
        },
        "baseline": baseline_status,
        "self_check": {
            "ran": self_check,
            "probes": self_check_results,
            "passed": all(item["passed"] for item in self_check_results) if self_check else False,
        },
        "passed": (
            not drifted
            and chain_passed
            and baseline_ok
            and self_check
            and all(item["passed"] for item in self_check_results)
        ),
        # Passing every covered dimension is not SPEC :1393 compliance while one
        # dimension has no implementation behind it.
        "spec_1393_fully_covered": not UNCOVERED_DIMENSIONS,
    }
    if emit_state:
        report["state"] = first["state"]
    if write_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(
                {
                    "schema_version": BASELINE_SCHEMA,
                    "description": (
                        "Frozen per-dimension state digests for the SPEC :1393 replay "
                        "lane. Regenerate only with a recorded reason: a changed digest "
                        "means the stored user model changed."
                    ),
                    "fixture": fixture_path.name,
                    "fixture_sha256": report["fixture_sha256"],
                    "uncovered_dimensions": sorted(UNCOVERED_DIMENSIONS),
                    "digests": first["digests"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    return report


def main() -> None:
    args = parse_args()
    report = run_benchmark(
        args.fixture,
        args.baseline,
        self_check=not args.no_self_check,
        write_baseline=args.write_baseline,
        emit_state=args.emit_state,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "state"},
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
