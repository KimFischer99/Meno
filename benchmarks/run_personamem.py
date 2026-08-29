from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import time
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been", "being",
    "could", "does", "for", "from", "great", "have", "here", "imagine", "into",
    "just", "like", "must", "not", "quite", "really", "see", "since", "such",
    "that", "the", "their", "there", "they", "this", "through", "user", "very",
    "was", "were", "what", "when", "where", "which", "with", "would", "you", "your",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Meno's retrieval-only PersonaMem 32k option-ranking adaptation"
    )
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8765")
    parser.add_argument("--api-token", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-facets", type=int, default=12)
    parser.add_argument("--ingest-batch-size", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--cleanup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question_raw = args.questions.read_bytes()
    context_raw = args.contexts.read_bytes()
    with args.questions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]
    rows.sort(key=lambda row: (row["shared_context_id"], int(row["end_index_in_shared_context"])))
    contexts: dict[str, list[dict[str, str]]] = {}
    for line in context_raw.decode("utf-8").splitlines():
        contexts.update(json.loads(line))

    headers = {"Authorization": f"Bearer {args.api_token}"} if args.api_token else {}
    results: list[dict[str, Any]] = []
    progress: dict[str, int] = defaultdict(int)
    revisions: dict[str, int] = defaultdict(int)
    user_ids: set[str] = set()
    service_config: dict[str, Any]
    with httpx.Client(
        base_url=args.api_url, headers=headers, timeout=args.timeout, trust_env=False
    ) as client:
        ready = client.get("/health/ready")
        ready.raise_for_status()
        ready_payload = ready.json()
        service_config = {
            key: ready_payload.get(key)
            for key in (
                "environment",
                "policy_version",
                "extractor_version",
                "embedding_provider",
                "embedding_model",
                "embedding_projection_version",
            )
        }
        for index, row in enumerate(rows, start=1):
            context_id = row["shared_context_id"]
            user_id = f"personamem:{args.run_id}:{context_id[:16]}"
            user_ids.add(user_id)
            end = int(row["end_index_in_shared_context"])
            messages = contexts[context_id]
            ingest_events: list[dict[str, Any]] = []
            for message_index in range(progress[context_id], min(end, len(messages))):
                message = messages[message_index]
                if message.get("role") != "user":
                    continue
                event_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"meno:personamem:{args.run_id}:{context_id}:{message_index}",
                    )
                )
                ingest_events.append(
                    {
                        "user_id": user_id,
                        "event_id": event_id,
                        "source": {
                            "type": "personamem",
                            "profile": "benchmark",
                            "session_id": context_id,
                        },
                        "content": {"role": "user", "text": message["content"]},
                        "consent_scope": ["personalization", "task_planning"],
                        "metadata": {
                            "benchmark": "PersonaMem-32k",
                            "message_index": message_index,
                        },
                    }
                )
            newly_accepted = 0
            last_ingest_revision = revisions[user_id]
            for batch_index, offset in enumerate(
                range(0, len(ingest_events), args.ingest_batch_size)
            ):
                batch = ingest_events[offset : offset + args.ingest_batch_size]
                response = client.post(
                    "/v1/ingest/batch",
                    headers={
                        "Idempotency-Key": (
                            f"batch:{args.run_id}:{context_id}:{end}:{batch_index}"
                        )
                    },
                    json={"events": batch},
                )
                response.raise_for_status()
                payload = response.json()
                newly_accepted += sum(
                    not event["idempotent_replay"] for event in payload["events"]
                )
                last_ingest_revision = max(
                    last_ingest_revision,
                    *(event["state_revision"] for event in payload["events"]),
                )
            if newly_accepted:
                revisions[user_id] = last_ingest_revision + newly_accepted
            progress[context_id] = max(progress[context_id], end)
            _wait_for_processing(client, user_id, revisions[user_id], args.timeout)

            started = time.perf_counter()
            response = client.post(
                "/v1/retrieve",
                json={
                    "user_id": user_id,
                    "purpose": "response_personalization",
                    "context": {
                        "query": row["user_question_or_message"],
                        "task_type": "personalization",
                        "platform": "benchmark",
                    },
                    "constraints": {"max_facets": args.max_facets, "min_confidence": 0.5},
                },
            )
            latency_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            payload = response.json()
            options = _parse_options(row["all_options"])
            scores = _rank_options(options, payload["facets"])
            predicted = max(range(len(scores)), key=scores.__getitem__) if max(scores) > 0 else None
            correct = ord(row["correct_answer"].strip("()").lower()) - ord("a")
            ranking = sorted(range(len(scores)), key=lambda item: (-scores[item], item))
            rank = ranking.index(correct) + 1
            result = {
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "correct_option": correct,
                "predicted_option": predicted,
                "correct": predicted == correct,
                "correct_supported": scores[correct] > 0,
                "reciprocal_rank": 1 / rank,
                "option_scores": scores,
                "facet_count": len(payload["facets"]),
                "degraded": payload["degraded"],
                "latency_ms": latency_ms,
            }
            results.append(result)
            print(
                f"[{index}/{len(rows)}] {row['question_id']} "
                f"correct={result['correct']} supported={result['correct_supported']}"
            )

        if args.cleanup:
            for user_id in user_ids:
                response = client.post(
                    "/v1/deletions",
                    headers={"Idempotency-Key": f"cleanup:{args.run_id}:{user_id}"},
                    json={"user_id": user_id, "scope": "all"},
                )
                response.raise_for_status()

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_type[result["question_type"]].append(result)
    report = {
        "benchmark": "PersonaMem-32k retrieval-only option-ranking adaptation",
        "comparability": "Not comparable to the official end-to-end LLM accuracy metric",
        "questions_sha256": hashlib.sha256(question_raw).hexdigest(),
        "contexts_sha256": hashlib.sha256(context_raw).hexdigest(),
        "run_id": args.run_id,
        "items": len(results),
        "max_facets": args.max_facets,
        "ingest_batch_size": args.ingest_batch_size,
        "service_config": service_config,
        "aggregate": _metrics(results),
        "by_question_type": {key: _metrics(value) for key, value in sorted(by_type.items())},
        "degraded_count": Counter(result["degraded"] for result in results).get(True, 0),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 2 and token not in STOPWORDS
    }


def _parse_options(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or len(parsed) != 4:
        raise ValueError("PersonaMem all_options must contain four options")
    return [str(option) for option in parsed]


def _rank_options(options: list[str], facets: list[dict[str, Any]]) -> list[float]:
    memory_tokens = _tokens(" ".join(str(facet["value"]) for facet in facets))
    option_tokens = [_tokens(re.sub(r"^\([a-d]\)\s*", "", option)) for option in options]
    frequency = Counter(token for tokens in option_tokens for token in tokens)
    return [
        sum(math.log((len(options) + 1) / (frequency[token] + 0.5)) for token in tokens & memory_tokens)
        for tokens in option_tokens
    ]


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
    raise TimeoutError(user_id)


def _metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {
            "option_accuracy": 0.0,
            "correct_support_at_k": 0.0,
            "mrr": 0.0,
            "mean_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "max_latency_ms": 0.0,
        }
    latencies = sorted(float(result["latency_ms"]) for result in results)
    p95_index = min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)
    return {
        "option_accuracy": statistics.fmean(result["correct"] for result in results),
        "correct_support_at_k": statistics.fmean(
            result["correct_supported"] for result in results
        ),
        "mrr": statistics.fmean(result["reciprocal_rank"] for result in results),
        "mean_latency_ms": statistics.fmean(latencies),
        "p95_latency_ms": latencies[p95_index],
        "max_latency_ms": latencies[-1],
    }


if __name__ == "__main__":
    main()
