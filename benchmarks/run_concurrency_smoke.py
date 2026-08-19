from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Meno concurrent ingest stability smoke")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8765")
    parser.add_argument("--api-token", default="")
    parser.add_argument("--events", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    user_id = f"concurrency:{args.run_id}"
    headers = {"Authorization": f"Bearer {args.api_token}"} if args.api_token else {}
    client = httpx.Client(
        base_url=args.api_url, headers=headers, timeout=args.timeout, trust_env=False
    )

    def ingest(index: int) -> tuple[int, float]:
        event_id = str(uuid.uuid4())
        started = time.perf_counter()
        response = client.post(
            "/v1/ingest",
            headers={"Idempotency-Key": event_id},
            json={
                "user_id": user_id,
                "event_id": event_id,
                "source": {
                    "type": "concurrency_smoke",
                    "profile": "benchmark",
                    "session_id": args.run_id,
                },
                "content": {
                    "role": "user",
                    "text": f"I prefer evidence-backed report style number {index}",
                },
                "consent_scope": ["personalization"],
            },
        )
        latency = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        return response.status_code, latency

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        outcomes = list(executor.map(ingest, range(args.events)))
    expected_revision = args.events * 2
    deadline = time.monotonic() + args.timeout
    revision = 0
    while time.monotonic() < deadline:
        response = client.get(f"/v1/revisions/{user_id}")
        response.raise_for_status()
        revision = response.json()["state_revision"]
        if revision >= expected_revision:
            break
        time.sleep(0.05)

    response = client.post(
        "/v1/retrieve",
        json={
            "user_id": user_id,
            "purpose": "response_personalization",
            "context": {"query": "evidence backed report preferences"},
            "constraints": {"max_facets": min(16, args.events)},
        },
    )
    response.raise_for_status()
    retrieval = response.json()
    latencies = [latency for _, latency in outcomes]
    passed = (
        all(status == 202 for status, _ in outcomes)
        and revision == expected_revision
        and len(retrieval["facets"]) >= min(8, args.events)
        and not retrieval["degraded"]
    )
    report = {
        "benchmark": "Meno concurrent ingest stability smoke",
        "run_id": args.run_id,
        "events": args.events,
        "workers": args.workers,
        "accepted": sum(status == 202 for status, _ in outcomes),
        "expected_revision": expected_revision,
        "final_revision": revision,
        "facet_count": len(retrieval["facets"]),
        "degraded": retrieval["degraded"],
        "ingest_latency_ms": {
            "mean": statistics.fmean(latencies),
            "max": max(latencies),
        },
        "passed": passed,
    }
    if args.cleanup:
        response = client.post(
            "/v1/deletions",
            headers={"Idempotency-Key": f"cleanup:{args.run_id}"},
            json={"user_id": user_id, "scope": "all"},
        )
        response.raise_for_status()
    client.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
