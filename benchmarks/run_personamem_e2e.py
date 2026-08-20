"""End-to-end LLM-answer PersonaMem evaluation for Meno.

Unlike ``run_personamem.py`` (retrieval-only option ranking with a fixed non-LLM
scorer), this script measures the full personalization path:

    ingest context -> /v1/retrieve -> inject rendered_context -> LLM answers

The LLM only sees Meno's retrieved ``<user_context>`` block, never the raw
conversation, so accuracy reflects what Meno memory actually contributes to an
answer. Retrieval quality, answer quality, latency, and token cost are recorded
separately (Meno_SPEC evaluation groups; test report P1 item 4).

For every question the legacy option-ranking scorer is also computed on the
*same* retrieved facets, which isolates "retrieval failure" from "answer-method
failure" on identical evidence.

Credentials are never hardcoded: the Meno bearer token and the LLM API key are
read from files or environment variables at runtime and are never written to
the result JSON.
"""

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
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

# Same instruction string as the official PersonaMem inference pipeline
# (bowen-upenn/PersonaMem inference.py), so answer formatting/extraction stays
# aligned with the official end-to-end protocol.
INSTRUCTIONS = (
    "Find the most appropriate model response and give your final answer "
    "(a), (b), (c), or (d) after the special token <final_answer>."
)

SYSTEM_PROMPT = (
    "You are a personalized assistant. The following evidence-backed context about "
    "the current user was retrieved from memory. Choose the response that is most "
    "appropriate for this specific user.\n\n"
    "{user_context}\n\n"
    "If the retrieved context is insufficient, choose the most generally "
    "appropriate response."
)

STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been", "being",
    "could", "does", "for", "from", "great", "have", "here", "imagine", "into",
    "just", "like", "must", "not", "quite", "really", "see", "since", "such",
    "that", "the", "their", "there", "they", "this", "through", "user", "very",
    "was", "were", "what", "when", "where", "which", "with", "would", "you", "your",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Meno's end-to-end LLM-answer PersonaMem 32k evaluation"
    )
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8765")
    parser.add_argument("--api-token", default="")
    parser.add_argument("--api-token-file", type=Path, default=None)
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8317/v1")
    parser.add_argument("--llm-model", default="glm-5.2")
    parser.add_argument("--llm-api-key", default="")
    parser.add_argument("--llm-api-key-file", type=Path, default=None)
    parser.add_argument("--answer-max-tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--question-types", default="")
    parser.add_argument("--max-facets", type=int, default=12)
    parser.add_argument("--ingest-batch-size", type=int, default=128)
    parser.add_argument(
        "--ingest-batch-interval-seconds",
        type=float,
        default=0,
        help="sleep between ingest batch submissions to pace embedding provider quota",
    )
    parser.add_argument(
        "--retrieve-retries",
        type=int,
        default=3,
        help="extra attempts when /v1/retrieve responds degraded=true",
    )
    parser.add_argument(
        "--retrieve-retry-wait-seconds",
        type=float,
        default=20,
        help="backoff between degraded retrieve retries",
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=600,
        help="max seconds to wait for outbox drain after each ingest",
    )
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--retrieval-cache", type=Path, default=None)
    parser.add_argument("--cleanup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    question_raw = args.questions.read_bytes()
    context_raw = args.contexts.read_bytes()
    with args.questions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.question_types:
        allowed = {item.strip() for item in args.question_types.split(",") if item.strip()}
        rows = [row for row in rows if row["question_type"] in allowed]
    if args.limit:
        rows = rows[: args.limit]
    rows.sort(key=lambda row: (row["shared_context_id"], int(row["end_index_in_shared_context"])))
    contexts: dict[str, list[dict[str, str]]] = {}
    for line in context_raw.decode("utf-8").splitlines():
        contexts.update(json.loads(line))

    api_token = _read_secret(args.api_token, args.api_token_file)
    llm_api_key = _read_secret(args.llm_api_key, args.llm_api_key_file)
    headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
    questions_sha256 = hashlib.sha256(question_raw).hexdigest()
    contexts_sha256 = hashlib.sha256(context_raw).hexdigest()

    user_ids: set[str] = set()
    with httpx.Client(
        base_url=args.api_url, headers=headers, timeout=args.timeout, trust_env=False
    ) as client:
        service_versions = _fetch_service_versions(client)
        retrieval_records = _load_retrieval_cache(
            args.retrieval_cache,
            args.run_id,
            rows,
            extractor_version=service_versions.get("extractor_version", ""),
            questions_sha256=questions_sha256,
            contexts_sha256=contexts_sha256,
        )
        if retrieval_records is None:
            retrieval_records, user_ids = _run_retrieval_phase(args, client, rows, contexts)
            _save_retrieval_cache(
                args.retrieval_cache,
                args.run_id,
                retrieval_records,
                extractor_version=service_versions.get("extractor_version", ""),
                questions_sha256=questions_sha256,
                contexts_sha256=contexts_sha256,
            )
        elif args.cleanup:
            user_ids = {record["user_id"] for record in retrieval_records}

        answers = _run_answer_phase(args, retrieval_records, llm_api_key)

        if args.cleanup:
            for user_id in sorted(user_ids):
                response = client.post(
                    "/v1/deletions",
                    headers={"Idempotency-Key": f"cleanup:{args.run_id}:{user_id}"},
                    json={"user_id": user_id, "scope": "all"},
                )
                response.raise_for_status()

    results = [_merge(record, answer) for record, answer in zip(retrieval_records, answers)]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_type[result["question_type"]].append(result)
    report = {
        "benchmark": "PersonaMem-32k end-to-end LLM-answer evaluation",
        "comparability": (
            "LLM answers from Meno retrieved context only; comparable across Meno "
            "revisions, not to the official full-context PersonaMem leaderboard"
        ),
        "questions_sha256": questions_sha256,
        "contexts_sha256": contexts_sha256,
        "run_id": args.run_id,
        "items": len(results),
        "config": {
            "api_url": args.api_url,
            "llm_base_url": args.llm_base_url,
            "llm_model": args.llm_model,
            "answer_max_tokens": args.answer_max_tokens,
            "max_facets": args.max_facets,
            "concurrency": args.concurrency,
            "ingest_batch_size": args.ingest_batch_size,
            "ingest_batch_interval_seconds": args.ingest_batch_interval_seconds,
            "retrieve_retries": args.retrieve_retries,
            "retrieve_retry_wait_seconds": args.retrieve_retry_wait_seconds,
            "extractor_version": service_versions.get("extractor_version"),
            "embedding_projection_version": service_versions.get(
                "embedding_projection_version"
            ),
            "policy_version": service_versions.get("policy_version"),
        },
        "aggregate": _metrics(results),
        "by_question_type": {key: _metrics(value) for key, value in sorted(by_type.items())},
        "degraded_count": Counter(result["degraded"] for result in results).get(True, 0),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


def _read_secret(value: str, path: Path | None) -> str:
    if value:
        return value
    if path is not None:
        return path.read_text(encoding="utf-8").strip()
    return ""


def _run_retrieval_phase(
    args: argparse.Namespace,
    client: httpx.Client,
    rows: list[dict[str, str]],
    contexts: dict[str, list[dict[str, str]]],
) -> tuple[list[dict[str, Any]], set[str]]:
    records: list[dict[str, Any]] = []
    progress: dict[str, int] = defaultdict(int)
    user_ids: set[str] = set()
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
        for batch_index, offset in enumerate(range(0, len(ingest_events), args.ingest_batch_size)):
            if batch_index and args.ingest_batch_interval_seconds:
                time.sleep(args.ingest_batch_interval_seconds)
            batch = ingest_events[offset : offset + args.ingest_batch_size]
            response = client.post(
                "/v1/ingest/batch",
                headers={"Idempotency-Key": f"batch:{args.run_id}:{context_id}:{end}:{batch_index}"},
                json={"events": batch},
            )
            response.raise_for_status()
            payload = response.json()
            newly_accepted += sum(not event["idempotent_replay"] for event in payload["events"])
        progress[context_id] = max(progress[context_id], end)
        if newly_accepted:
            _wait_for_processing(client, user_id, args.wait_timeout)

        started = time.perf_counter()
        retrieve_attempts = 0
        while True:
            retrieve_attempts += 1
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
            response.raise_for_status()
            payload = response.json()
            if (
                not payload["degraded"]
                or retrieve_attempts > args.retrieve_retries
            ):
                break
            time.sleep(args.retrieve_retry_wait_seconds)
        latency_ms = (time.perf_counter() - started) * 1000
        options = _parse_options(row["all_options"])
        scores = _rank_options(options, payload["facets"])
        correct = ord(row["correct_answer"].strip("()").lower()) - ord("a")
        ranking = sorted(range(len(scores)), key=lambda item: (-scores[item], item))
        rank = ranking.index(correct) + 1
        records.append(
            {
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "user_id": user_id,
                "question": row["user_question_or_message"],
                "correct_option": correct,
                "options": options,
                "rendered_context": payload["rendered_context"],
                "facet_count": len(payload["facets"]),
                "degraded": payload["degraded"],
                "state_revision": payload["state_revision"],
                "retrieval_latency_ms": latency_ms,
                "retrieve_attempts": retrieve_attempts,
                "ranking_predicted_option": (
                    max(range(len(scores)), key=scores.__getitem__) if max(scores) > 0 else None
                ),
                "ranking_correct_supported": scores[correct] > 0,
                "ranking_reciprocal_rank": 1 / rank,
            }
        )
        print(f"[retrieve {index}/{len(rows)}] {row['question_id']} facets={len(payload['facets'])}")
    return records, user_ids


def _run_answer_phase(
    args: argparse.Namespace, records: list[dict[str, Any]], llm_api_key: str
) -> list[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {llm_api_key}"} if llm_api_key else {}
    answers: list[dict[str, Any] | None] = [None] * len(records)
    with httpx.Client(
        base_url=args.llm_base_url, headers=headers, timeout=args.timeout, trust_env=False
    ) as client, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(_answer_one, client, args, record): position
            for position, record in enumerate(records)
        }
        for done, (future, position) in enumerate(futures.items(), start=1):
            answers[position] = future.result()
            print(
                f"[answer {done}/{len(records)}] {records[position]['question_id']} "
                f"correct={answers[position]['answer_correct']}"
            )
    return [answer for answer in answers if answer is not None]


def _answer_one(
    client: httpx.Client, args: argparse.Namespace, record: dict[str, Any]
) -> dict[str, Any]:
    user_context = record["rendered_context"] or "No user context available."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(user_context=user_context)},
        {
            "role": "user",
            "content": (
                record["question"]
                + "\n\n"
                + INSTRUCTIONS
                + "\n\n"
                + "\n".join(record["options"])
            ),
        },
    ]
    started = time.perf_counter()
    payload: dict[str, Any] = {}
    error: str | None = None
    for attempt in range(4):
        try:
            response = client.post(
                "/chat/completions",
                json={
                    "model": args.llm_model,
                    "messages": messages,
                    "max_tokens": args.answer_max_tokens,
                },
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                raise httpx.HTTPStatusError(
                    f"status {response.status_code}", request=response.request, response=response
                )
            response.raise_for_status()
            payload = response.json()
            error = None
            break
        except (httpx.HTTPError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"[:200]
            time.sleep(min(2**attempt, 8))
    latency_ms = (time.perf_counter() - started) * 1000
    answer_text = ""
    usage: dict[str, Any] = {}
    if payload:
        answer_text = str(payload["choices"][0]["message"].get("content") or "")
        usage = payload.get("usage") or {}
    correct_letter = "abcd"[record["correct_option"]]
    answer_correct, predicted_letter = _extract_answer(answer_text, correct_letter)
    return {
        "answer_correct": answer_correct,
        "predicted_option": (
            ord(predicted_letter) - ord("a") if predicted_letter is not None else None
        ),
        "answer_parse_ok": predicted_letter is not None,
        "answer_error": error,
        "answer_latency_ms": latency_ms,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "answer_text": answer_text[:400],
    }


def _extract_answer(predicted_answer: str, correct: str) -> tuple[bool, str | None]:
    """Official PersonaMem extraction: option set after <final_answer> must equal {correct}."""

    def _extract_only_options(text: str) -> set[str]:
        text = text.lower()
        in_parens = re.findall(r"\(([a-d])\)", text)
        if in_parens:
            return set(in_parens)
        return set(re.findall(r"\b([a-d])\b", text))

    full_response = predicted_answer
    stripped = predicted_answer.strip()
    if "<final_answer>" in stripped:
        stripped = stripped.split("<final_answer>")[-1].strip()
    if stripped.endswith("</final_answer>"):
        stripped = stripped[: -len("</final_answer>")].strip()

    pred_options = _extract_only_options(stripped)
    predicted_letter = next(iter(pred_options), None) if len(pred_options) == 1 else None
    if pred_options == {correct}:
        return True, predicted_letter
    response_options = _extract_only_options(full_response)
    if predicted_letter is None and len(response_options) == 1:
        predicted_letter = next(iter(response_options), None)
    return response_options == {correct}, predicted_letter


def _merge(record: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "question_id": record["question_id"],
        "question_type": record["question_type"],
        "correct_option": record["correct_option"],
        "answer_correct": answer["answer_correct"],
        "predicted_option": answer["predicted_option"],
        "answer_parse_ok": answer["answer_parse_ok"],
        "answer_error": answer["answer_error"],
        "ranking_predicted_option": record["ranking_predicted_option"],
        "ranking_correct": record["ranking_predicted_option"] == record["correct_option"],
        "correct_supported": record["ranking_correct_supported"],
        "reciprocal_rank": record["ranking_reciprocal_rank"],
        "facet_count": record["facet_count"],
        "degraded": record["degraded"],
        "state_revision": record["state_revision"],
        "retrieval_latency_ms": record["retrieval_latency_ms"],
        "retrieve_attempts": record["retrieve_attempts"],
        "answer_latency_ms": answer["answer_latency_ms"],
        "prompt_tokens": answer["prompt_tokens"],
        "completion_tokens": answer["completion_tokens"],
        "total_tokens": answer["total_tokens"],
        "answer_text": answer["answer_text"],
    }
    return merged


def _fetch_service_versions(client: httpx.Client) -> dict[str, Any]:
    """Bind the report to the running service's version stamps (design 5.2)."""
    try:
        response = client.get("/health/ready")
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return {}


def _load_retrieval_cache(
    path: Path | None,
    run_id: str,
    rows: list[dict[str, str]],
    *,
    extractor_version: str,
    questions_sha256: str,
    contexts_sha256: str,
) -> list[dict[str, Any]] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_id") != run_id:
        return None
    # Cross-version cache reuse would contaminate A/B extractor comparisons.
    if payload.get("extractor_version") != extractor_version:
        return None
    if payload.get("questions_sha256") != questions_sha256:
        return None
    if payload.get("contexts_sha256") != contexts_sha256:
        return None
    expected = [row["question_id"] for row in rows]
    records = payload.get("records", [])
    if [record["question_id"] for record in records] != expected:
        return None
    print(f"reusing retrieval cache: {path} ({len(records)} records)")
    return records


def _save_retrieval_cache(
    path: Path | None,
    run_id: str,
    records: list[dict[str, Any]],
    *,
    extractor_version: str,
    questions_sha256: str,
    contexts_sha256: str,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "extractor_version": extractor_version,
                "questions_sha256": questions_sha256,
                "contexts_sha256": contexts_sha256,
                "records": records,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


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
    # Same fixed non-LLM scorer as run_personamem.py, applied to the same facets
    # so retrieval-only and end-to-end answer methods are directly comparable.
    memory_tokens = _tokens(" ".join(str(facet["value"]) for facet in facets))
    option_tokens = [_tokens(re.sub(r"^\([a-d]\)\s*", "", option)) for option in options]
    frequency = Counter(token for tokens in option_tokens for token in tokens)
    return [
        sum(math.log((len(options) + 1) / (frequency[token] + 0.5)) for token in tokens & memory_tokens)
        for tokens in option_tokens
    ]


def _wait_for_processing(client: httpx.Client, user_id: str, timeout: float) -> None:
    """Wait until the user's outbox queue reports zero pending rows.

    Outbox rows are created in the same transaction as the ingest response,
    so ``pending_outbox == 0`` from the drain endpoint means every accepted
    event has been processed (or terminally failed) — no stability window.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/v1/users/{user_id}/drain")
        response.raise_for_status()
        if response.json()["pending_outbox"] == 0:
            return
        time.sleep(0.5)
    raise TimeoutError(user_id)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    prompt_tokens = [r["prompt_tokens"] for r in results if r["prompt_tokens"] is not None]
    completion_tokens = [
        r["completion_tokens"] for r in results if r["completion_tokens"] is not None
    ]
    retrieval_latencies = [r["retrieval_latency_ms"] for r in results]
    answer_latencies = [r["answer_latency_ms"] for r in results]
    return {
        "answer_accuracy": statistics.fmean(r["answer_correct"] for r in results),
        "option_ranking_accuracy": statistics.fmean(r["ranking_correct"] for r in results),
        "correct_support_at_k": statistics.fmean(r["correct_supported"] for r in results),
        "mrr": statistics.fmean(r["reciprocal_rank"] for r in results),
        "answer_parse_failures": sum(not r["answer_parse_ok"] for r in results),
        "answer_errors": sum(r["answer_error"] is not None for r in results),
        "degraded_after_retries": sum(r["degraded"] for r in results),
        "retrieve_retries_total": sum(r["retrieve_attempts"] - 1 for r in results),
        "retrieval_mean_latency_ms": statistics.fmean(retrieval_latencies),
        "retrieval_p95_latency_ms": _p95(retrieval_latencies),
        "answer_mean_latency_ms": statistics.fmean(answer_latencies),
        "answer_p95_latency_ms": _p95(answer_latencies),
        "prompt_tokens_total": sum(prompt_tokens),
        "completion_tokens_total": sum(completion_tokens),
        "prompt_tokens_mean": statistics.fmean(prompt_tokens) if prompt_tokens else 0.0,
        "completion_tokens_mean": (
            statistics.fmean(completion_tokens) if completion_tokens else 0.0
        ),
    }


if __name__ == "__main__":
    main()
