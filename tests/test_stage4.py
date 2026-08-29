from __future__ import annotations

from benchmarks.run_personamem import _metrics
from benchmarks.verify_invariants import _audit_hash, audit_chain_summary


def _audit_row(index: int, previous: str | None) -> dict:
    row = {
        "event_name": "meno.test",
        "trace_id": f"trace-{index}",
        "user_hash": "sha256:user",
        "claim_id": None,
        "action": "test",
        "purpose": None,
        "decision": {"allowed": True},
        "state_revision": index,
        "source_event_ids": [],
        "prev_hash": previous,
    }
    row["current_hash"] = _audit_hash(row)
    return row


def test_personamem_metrics_include_tail_latency():
    results = [
        {
            "correct": index % 2 == 0,
            "correct_supported": True,
            "reciprocal_rank": 1.0,
            "latency_ms": float(index),
        }
        for index in range(1, 101)
    ]

    metrics = _metrics(results)

    assert metrics["mean_latency_ms"] == 50.5
    assert metrics["p95_latency_ms"] == 95.0
    assert metrics["max_latency_ms"] == 100.0


def test_audit_chain_summary_accepts_one_valid_chain():
    root = _audit_row(1, None)
    child = _audit_row(2, root["current_hash"])

    assert audit_chain_summary([root, child]) == {
        "rows": 2,
        "roots": 1,
        "dangling": 0,
        "forks": 0,
        "hash_mismatch": 0,
        "passed": True,
    }


def test_audit_chain_summary_rejects_fork_and_hash_mismatch():
    root = _audit_row(1, None)
    first = _audit_row(2, root["current_hash"])
    second = _audit_row(3, root["current_hash"])
    second["decision"] = {"allowed": False}

    summary = audit_chain_summary([root, first, second])

    assert summary["forks"] == 1
    assert summary["hash_mismatch"] == 1
    assert summary["passed"] is False

