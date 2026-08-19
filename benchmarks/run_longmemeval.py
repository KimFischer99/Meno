from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Meno evidence retrieval on LongMemEval")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8765")
    parser.add_argument("--api-token", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-facets", type=int, default=10)
    parser.add_argument("--ingest-batch-size", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--cleanup", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * quantile) - 1))
    return ordered[index]


def main() -> None:
    args = parse_args()
    raw = args.data.read_bytes()
    items = json.loads(raw)
    if args.limit:
        items = items[: args.limit]
    headers = {"Authorization": f"Bearer {args.api_token}"} if args.api_token else {}
    client = httpx.Client(
        base_url=args.api_url, headers=headers, timeout=args.timeout, trust_env=False
    )
    results: list[dict[str, Any]] = []
    latencies: list[float] = []

    for item_index, item in enumerate(items, start=1):
        user_id = f"lme:{args.run_id}:{item['question_id']}"
        event_to_session: dict[str, str] = {}
        ingest_events: list[dict[str, Any]] = []
        for session_index, session in enumerate(item["haystack_sessions"]):
            session_id = item["haystack_session_ids"][session_index]
            occurred_at = _parse_date(item["haystack_dates"][session_index])
            for turn_index, turn in enumerate(session):
                if turn["role"] not in {"user", "assistant"}:
                    continue
                event_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"meno:{args.run_id}:{item['question_id']}:{session_id}:{turn_index}",
                    )
                )
                event_to_session[event_id] = session_id
                ingest_events.append(
                    {
                        "user_id": user_id,
                        "event_id": event_id,
                        "occurred_at": occurred_at,
                        "source": {
                            "type": "longmemeval",
                            "profile": "benchmark",
                            "session_id": session_id,
                        },
                        "content": {"role": turn["role"], "text": turn["content"]},
                        "consent_scope": ["personalization", "task_planning"],
                        "metadata": {
                            "benchmark": "LongMemEval",
                            "question_id": item["question_id"],
                            "has_answer": bool(turn.get("has_answer")),
                        },
                    }
                )

        accepted = 0
        for batch_index, offset in enumerate(
            range(0, len(ingest_events), args.ingest_batch_size)
        ):
            batch = ingest_events[offset : offset + args.ingest_batch_size]
            response = client.post(
                "/v1/ingest/batch",
                headers={
                    "Idempotency-Key": (
                        f"batch:{args.run_id}:{item['question_id']}:{batch_index}"
                    )
                },
                json={"events": batch},
            )
            response.raise_for_status()
            accepted += int(response.json()["accepted"])

        _wait_for_processing(client, user_id, expected_revision=accepted * 2, timeout=args.timeout)
        started = time.perf_counter()
        response = client.post(
            "/v1/retrieve",
            json={
                "user_id": user_id,
                "purpose": "response_personalization",
                "context": {
                    "query": item["question"],
                    "as_of": _parse_date(item["question_date"]),
                    "task_type": "conversation_recall",
                    "platform": "benchmark",
                },
                "constraints": {"max_facets": args.max_facets, "min_confidence": 0.5},
            },
        )
        latency_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        payload = response.json()
        latencies.append(latency_ms)
        retrieved_sessions = []
        for facet in payload["facets"]:
            retrieved_sessions.extend(
                event_to_session[event_id]
                for event_id in facet["evidence_ids"]
                if event_id in event_to_session
            )
        relevant = set(item["answer_session_ids"])
        retrieved = set(retrieved_sessions)
        hit = bool(relevant & retrieved)
        precision = len(relevant & retrieved) / len(retrieved) if retrieved else 0.0
        recall = len(relevant & retrieved) / len(relevant) if relevant else float(not retrieved)
        results.append(
            {
                "question_id": item["question_id"],
                "question_type": item["question_type"],
                "hit_at_k": hit,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "retrieved_session_ids": sorted(retrieved),
                "answer_session_ids": sorted(relevant),
                "facet_count": len(payload["facets"]),
                "degraded": payload["degraded"],
                "latency_ms": latency_ms,
            }
        )
        if args.cleanup:
            delete_response = client.post(
                "/v1/deletions",
                headers={"Idempotency-Key": f"cleanup:{args.run_id}:{item['question_id']}"},
                json={"user_id": user_id, "scope": "all"},
            )
            delete_response.raise_for_status()
        print(f"[{item_index}/{len(items)}] {item['question_id']} hit={hit} latency={latency_ms:.1f}ms")

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_type[result["question_type"]].append(result)
    report = {
        "benchmark": "LongMemEval oracle evidence retrieval",
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "run_id": args.run_id,
        "items": len(results),
        "max_facets": args.max_facets,
        "aggregate": _metrics(results),
        "by_question_type": {key: _metrics(value) for key, value in sorted(by_type.items())},
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "degraded_count": Counter(result["degraded"] for result in results).get(True, 0),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


def _wait_for_processing(
    client: httpx.Client, user_id: str, expected_revision: int, timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/v1/revisions/{user_id}")
        response.raise_for_status()
        if response.json()["state_revision"] >= expected_revision:
            return
        time.sleep(0.05)
    raise TimeoutError(f"Meno outbox did not process {user_id} before timeout")


def _parse_date(value: str) -> str:
    parsed = datetime.strptime(value, "%Y/%m/%d (%a) %H:%M").replace(tzinfo=UTC)
    return parsed.isoformat()


def _metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {"hit_at_k": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0}
    return {
        "hit_at_k": statistics.fmean(float(result["hit_at_k"]) for result in results),
        "precision_at_k": statistics.fmean(result["precision_at_k"] for result in results),
        "recall_at_k": statistics.fmean(result["recall_at_k"] for result in results),
    }


if __name__ == "__main__":
    main()
