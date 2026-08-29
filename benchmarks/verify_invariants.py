"""Verify Meno's canonical, audit-chain, outbox, and vector invariants read-only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text

from meno.config import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-vector",
        action="store_true",
        help="Fail unless a Qdrant projection is configured and consistent",
    )
    return parser.parse_args()


def _audit_hash(row: Mapping[str, Any]) -> str:
    payload = {
        "event_name": row["event_name"],
        "trace_id": row["trace_id"],
        "user_hash": row["user_hash"],
        "claim_id": row["claim_id"],
        "action": row["action"],
        "purpose": row["purpose"],
        "decision": row["decision"],
        "state_revision": row["state_revision"],
        "source_event_ids": row["source_event_ids"],
        "prev_hash": row["prev_hash"],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def audit_chain_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | bool]:
    hashes = {str(row["current_hash"]) for row in rows}
    predecessors = [str(row["prev_hash"]) for row in rows if row["prev_hash"]]
    roots = sum(row["prev_hash"] is None for row in rows)
    dangling = sum(previous not in hashes for previous in predecessors)
    forks = sum(count - 1 for count in Counter(predecessors).values() if count > 1)
    mismatches = sum(_audit_hash(row) != row["current_hash"] for row in rows)
    valid_roots = roots == 1 if rows else roots == 0
    return {
        "rows": len(rows),
        "roots": roots,
        "dangling": dangling,
        "forks": forks,
        "hash_mismatch": mismatches,
        "passed": valid_roots and dangling == 0 and forks == 0 and mismatches == 0,
    }


def verify(settings: Settings, *, require_vector: bool) -> dict[str, Any]:
    engine = create_engine(settings.database_url)
    try:
        with engine.connect() as connection:
            duplicate_groups = int(
                connection.execute(
                    text(
                        """
                        SELECT COUNT(*) FROM (
                            SELECT user_id, semantic_key
                            FROM meno_claims
                            WHERE status = :status
                            GROUP BY user_id, semantic_key
                            HAVING COUNT(*) > 1
                        ) AS duplicates
                        """
                    ),
                    {"status": "active"},
                ).scalar_one()
            )
            outbox_counts = {
                str(status): int(count)
                for status, count in connection.execute(
                    text("SELECT status, COUNT(*) FROM meno_outbox GROUP BY status")
                ).all()
            }
            projection_counts = {
                str(status): int(count)
                for status, count in connection.execute(
                    text(
                        "SELECT status, COUNT(*) FROM meno_projection_outbox "
                        "GROUP BY status"
                    )
                ).all()
            }
            active_ids = set(
                connection.execute(
                    text(
                        "SELECT id FROM meno_claims "
                        "WHERE status = :status AND sensitive = false"
                    ),
                    {"status": "active"},
                ).scalars()
            )
            audit_rows = connection.execute(
                text(
                    """
                    SELECT id, event_name, trace_id, user_hash, claim_id, action,
                           purpose, decision, state_revision, source_event_ids,
                           prev_hash, current_hash, created_at
                    FROM meno_audit_events
                    ORDER BY created_at, id
                    """
                )
            ).mappings().all()
    finally:
        engine.dispose()

    outbox = {
        "processed": outbox_counts.get("processed", 0),
        "pending": outbox_counts.get("pending", 0),
        "failed": outbox_counts.get("failed", 0),
        "cancelled": outbox_counts.get("cancelled", 0),
    }
    projection_outbox = {
        "processed": projection_counts.get("processed", 0),
        "pending": projection_counts.get("pending", 0),
        "failed": projection_counts.get("failed", 0),
    }
    audit = audit_chain_summary(audit_rows)
    vector = _vector_summary(settings, active_ids)
    vector_required_passed = (
        bool(vector["passed"]) if vector["checked"] else not require_vector
    )
    passed = (
        duplicate_groups == 0
        and outbox["pending"] == 0
        and outbox["failed"] == 0
        and projection_outbox["pending"] == 0
        and projection_outbox["failed"] == 0
        and bool(audit["passed"])
        and vector_required_passed
    )
    return {
        "check": "Meno Stage 4 invariant verification",
        "checked_at": datetime.now(UTC).isoformat(),
        "service_config": {
            "environment": settings.environment,
            "extractor_version": settings.extractor_version,
            "policy_version": settings.policy_version,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "embedding_projection_version": settings.embedding_projection_version,
            "vector_mode": settings.vector_mode,
            "qdrant_collection": settings.qdrant_collection,
        },
        "outbox": outbox,
        "projection_outbox": projection_outbox,
        "active_semantic_key_duplicate_groups": duplicate_groups,
        "audit": audit,
        "vector": vector,
        "passed": passed,
    }


def _vector_summary(settings: Settings, active_ids: set[str]) -> dict[str, Any]:
    if settings.vector_mode != "qdrant":
        return {
            "checked": False,
            "reason": "vector_mode is not qdrant",
            "active_nonsensitive_claims": len(active_ids),
            "passed": False,
        }

    client = QdrantClient(url=settings.qdrant_url)
    vector_ids: set[str] = set()
    offset: Any = None
    try:
        try:
            while True:
                points, offset = client.scroll(
                    collection_name=settings.qdrant_collection,
                    limit=256,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                vector_ids.update(str(point.id) for point in points)
                if offset is None:
                    break
        except Exception as exc:  # noqa: BLE001 - verifier reports provider failures
            return {
                "checked": True,
                "error": type(exc).__name__,
                "active_nonsensitive_claims": len(active_ids),
                "passed": False,
            }
    finally:
        client.close()

    orphan = len(vector_ids - active_ids)
    missing = len(active_ids - vector_ids)
    return {
        "checked": True,
        "points": len(vector_ids),
        "active_nonsensitive_claims": len(active_ids),
        "orphan": orphan,
        "missing": missing,
        "passed": orphan == 0 and missing == 0,
    }


def main() -> None:
    args = parse_args()
    report = verify(Settings.from_env(), require_vector=args.require_vector)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
