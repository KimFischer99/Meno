"""Run a bounded answer-model qualification on frozen retrieval records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from benchmarks.run_personamem_e2e import (
    INSTRUCTIONS,
    SYSTEM_PROMPT,
    _extract_answer,
    _p95,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-cache", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--samples-per-type", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=1024)
    return parser.parse_args()


def _select(records: list[dict[str, Any]], per_type: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["question_type"]].append(record)
    selected: list[dict[str, Any]] = []
    for rows in groups.values():
        ordered = sorted(
            rows,
            key=lambda row: hashlib.sha256(row["question_id"].encode()).hexdigest(),
        )
        selected.extend(ordered[:per_type])
    return sorted(selected, key=lambda row: (row["question_type"], row["question_id"]))


def _answer_one(
    args: argparse.Namespace,
    api_key: str,
    record: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                user_context=record["rendered_context"] or "No user context available."
            ),
        },
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
    status: int | None = None
    for attempt in range(3):
        try:
            with httpx.Client(
                base_url=args.llm_base_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=args.timeout,
                trust_env=False,
            ) as client:
                response = client.post(
                    "/chat/completions",
                    json={
                        "model": args.llm_model,
                        "messages": messages,
                        "max_tokens": args.max_tokens,
                        "temperature": 0,
                        "enable_thinking": False,
                    },
                )
                status = response.status_code
                if status in {429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        f"status {status}", request=response.request, response=response
                    )
                response.raise_for_status()
                payload = response.json()
                error = None
                break
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            error = f"{type(exc).__name__}: {exc}"[:200]
            if attempt < 2:
                time.sleep(min(2**attempt, 4))

    latency_ms = (time.perf_counter() - started) * 1000
    answer_text = ""
    usage: dict[str, Any] = {}
    if payload:
        answer_text = str(payload.get("choices", [{}])[0].get("message", {}).get("content") or "")
        usage = payload.get("usage") or {}
    correct_letter = "abcd"[record["correct_option"]]
    answer_correct, predicted = _extract_answer(answer_text, correct_letter)
    return {
        "question_id": record["question_id"],
        "question_type": record["question_type"],
        "answer_correct": answer_correct,
        "answer_parse_ok": predicted is not None,
        "predicted_option": ord(predicted) - ord("a") if predicted else None,
        "answer_error": error,
        "http_status": status,
        "answer_latency_ms": latency_ms,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "response_model": payload.get("model"),
        "response_provider": payload.get("provider"),
        "baseline_answer_correct": baseline["answer_correct"],
        "baseline_parse_ok": baseline["answer_parse_ok"],
        "answer_text": answer_text[:400],
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [row["answer_latency_ms"] for row in rows]
    return {
        "items": len(rows),
        "answer_accuracy": statistics.fmean(row["answer_correct"] for row in rows),
        "baseline_accuracy_same_items": statistics.fmean(
            row["baseline_answer_correct"] for row in rows
        ),
        "improved": sum(
            row["answer_correct"] and not row["baseline_answer_correct"] for row in rows
        ),
        "regressed": sum(
            not row["answer_correct"] and row["baseline_answer_correct"] for row in rows
        ),
        "parse_failures": sum(not row["answer_parse_ok"] for row in rows),
        "answer_errors": sum(row["answer_error"] is not None for row in rows),
        "mean_latency_ms": statistics.fmean(latencies),
        "p95_latency_ms": _p95(latencies),
        "prompt_tokens_total": sum(row["prompt_tokens"] or 0 for row in rows),
        "completion_tokens_total": sum(row["completion_tokens"] or 0 for row in rows),
        "response_models": dict(Counter(str(row["response_model"]) for row in rows)),
        "response_providers": dict(Counter(str(row["response_provider"]) for row in rows)),
    }


def main() -> None:
    args = parse_args()
    if args.samples_per_type < 1 or args.concurrency < 1:
        raise SystemExit("sample and concurrency values must be positive")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if len(api_key) < 20:
        raise SystemExit("OPENROUTER_API_KEY is missing")

    cache = json.loads(args.retrieval_cache.read_text(encoding="utf-8"))
    selected = _select(cache["records"], args.samples_per_type)
    baseline = {
        row["question_id"]: row
        for row in json.loads(args.baseline_report.read_text(encoding="utf-8"))["results"]
    }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(_answer_one, args, api_key, record, baseline[record["question_id"]]): record
            for record in selected
        }
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"[{index}/{len(selected)}] {result['question_type']} "
                f"correct={result['answer_correct']} parse={result['answer_parse_ok']} "
                f"error={bool(result['answer_error'])}",
                flush=True,
            )
    results.sort(key=lambda row: (row["question_type"], row["question_id"]))
    question_types = sorted({row["question_type"] for row in results})
    report = {
        "benchmark": "Answer-model qualification on frozen Meno retrieval contexts",
        "created_at": datetime.now(UTC).isoformat(),
        "model_requested": args.llm_model,
        "base_url": args.llm_base_url,
        "selection": f"SHA-256(question_id), first {args.samples_per_type} per type",
        "source_retrieval_cache_sha256": hashlib.sha256(
            args.retrieval_cache.read_bytes()
        ).hexdigest(),
        "aggregate": _metrics(results),
        "by_question_type": {
            kind: _metrics([row for row in results if row["question_type"] == kind])
            for kind in question_types
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "aggregate": report["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
