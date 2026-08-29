"""Convert the LongMemEval oracle dataset into Meno's external-corpus contract.

Companion to `MENO_DATA_REQUIREMENTS.md` §1.3 and `validate_external_corpus.py`.
This exists to answer one question cheaply: does LongMemEval's `query_signal`
clear the bar that PersonaMem failed (0.087), so that a curated set over it could
evidence Gate Charter #13/#14 at all.

It emits no labels of its own. Every `relevant_message_indices` entry comes from
the dataset's own per-message `has_answer` flag, which is dataset provenance
rather than answer-reading, and is therefore admissible under the Charter §3
annotation constraint.

Four conversion decisions change what the measurement means. Each is recorded in
the emitted `*-conversion.json` so a reader can check them without reading this
file:

1. **One question becomes one session.** The contract binds a query to a single
   session_id, while a LongMemEval question spans several haystack sessions. The
   haystack sessions are concatenated in `haystack_dates` order into one session
   whose `user_key` is the question_id. This matches how Meno actually sees a
   user -- cross-session state under one identity -- but it means "session" here
   means "one user's whole history for one question".

2. **Timestamps are synthesized within a session.** `haystack_dates` is
   per-session, so every message in a session would share one timestamp, which
   the validator correctly flags as an export that used its own clock. Each turn
   is offset by one minute from its session date to preserve ordering. Absolute
   times are therefore not real; relative order within a session is.

3. **The `_abs` subset becomes the abstain queries.** LongMemEval marks 30
   questions with an `_abs` question_id suffix whose gold answer is "The
   information provided is not enough" -- the haystack deliberately lacks the
   deciding evidence. These map onto `expected_decision: abstain` with an empty
   label, which is exactly what the contract's abstain path means. Their
   `has_answer` flags (present on 9 of 30) mark near-miss evidence that answers a
   *related* question, so they are deliberately discarded rather than kept.

4. **Non-`_abs` questions with no `has_answer` message are dropped.** These have
   answers in the haystack but no usable pointer to them, so labelling them
   `abstain` would manufacture negatives the corpus does not contain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# One minute per turn: enough to order turns inside a session without implying
# that the gap between two messages is known.
TURN_INTERVAL = timedelta(minutes=1)

# LongMemEval's own abstention subset: the haystack deliberately omits the
# deciding evidence and the gold answer says so.
ABSTAIN_SUFFIX = "_abs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="longmemeval_*.json")
    parser.add_argument("--sessions-out", type=Path, required=True)
    parser.add_argument("--queries-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def parse_session_date(value: str) -> datetime:
    """LongMemEval dates look like '2023/05/20 (Sat) 02:21'."""
    cleaned = " ".join(part for part in value.split() if not part.startswith("("))
    for pattern in ("%Y/%m/%d %H:%M", "%Y/%m/%d"):
        try:
            return datetime.strptime(cleaned, pattern).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError(f"unrecognized session date: {value!r}")


def convert(items: list[dict[str, Any]]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    sessions: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    dropped_no_flag: list[str] = []
    abstain_ids: list[str] = []
    haystack_equals_answer = 0
    role_skipped = 0

    for item in items:
        question_id = item["question_id"]
        is_abstain = question_id.endswith(ABSTAIN_SUFFIX)
        if set(item["haystack_session_ids"]) == set(item["answer_session_ids"]):
            haystack_equals_answer += 1

        # Order the haystack by its own dates so the merged index is chronological.
        order = sorted(
            range(len(item["haystack_sessions"])),
            key=lambda position: parse_session_date(item["haystack_dates"][position]),
        )
        messages: list[dict[str, Any]] = []
        relevant: list[int] = []
        for position in order:
            session_start = parse_session_date(item["haystack_dates"][position])
            for turn_index, turn in enumerate(item["haystack_sessions"][position]):
                if turn["role"] not in {"user", "assistant", "system"}:
                    role_skipped += 1
                    continue
                index = len(messages)
                messages.append(
                    {
                        "index": index,
                        "role": turn["role"],
                        "text": turn["content"],
                        "occurred_at": (
                            session_start + turn_index * TURN_INTERVAL
                        ).isoformat().replace("+00:00", "Z"),
                    }
                )
                # `has_answer` appears as both bool and the string "True". On an
                # `_abs` item it marks near-miss evidence for a related question,
                # so it must not become a label.
                if not is_abstain and turn.get("has_answer") in (True, "True"):
                    relevant.append(index)

        if not messages:
            dropped_no_flag.append(question_id)
            continue
        if is_abstain:
            abstain_ids.append(question_id)
            relevant = []
        elif not relevant:
            # Decision 4: drop rather than fabricate an abstain.
            dropped_no_flag.append(question_id)
            continue

        sessions.append(
            {
                "session_id": question_id,
                "user_key": question_id,
                "messages": messages,
            }
        )
        queries.append(
            {
                "query_id": question_id,
                "session_id": question_id,
                "user_key": question_id,
                # The question is asked after the entire history, so every labeled
                # message precedes it and the causality rule holds by construction.
                "asked_at_index": len(messages),
                "query": item["question"],
                "task_type": item["question_type"],
                "relevant_message_indices": relevant,
                "expected_decision": "abstain" if is_abstain else "inject",
            }
        )

    report = {
        "converter": "longmemeval -> meno external-corpus-v1",
        "items_read": len(items),
        "sessions_emitted": len(sessions),
        "queries_emitted": len(queries),
        "dropped_no_has_answer_flag": len(dropped_no_flag),
        "dropped_question_ids": sorted(dropped_no_flag),
        "turns_skipped_unknown_role": role_skipped,
        "abstain_queries": len(abstain_ids),
        "oracle_check": {
            "items_where_haystack_equals_answer_sessions": haystack_equals_answer,
            "note": (
                "A haystack identical to the answer set means the dataset variant "
                "carries no distractor sessions. Retrieval precision measured on it "
                "is an upper bound, not a deployable number."
            ),
        },
        "conversion_decisions": {
            "one_question_is_one_session": (
                "haystack sessions concatenated in date order under user_key="
                "question_id; 'session' here means one user's whole history"
            ),
            "timestamps_synthesized": (
                f"per-session date plus {int(TURN_INTERVAL.total_seconds() // 60)} "
                "minute(s) per turn; absolute times are not real, order is"
            ),
            "abs_subset_is_abstain": (
                f"question_ids ending in {ABSTAIN_SUFFIX} are LongMemEval's own "
                "abstention set (gold answer: information not enough); their "
                "has_answer flags mark near-miss evidence and are discarded"
            ),
            "unflagged_questions_dropped": (
                "non-abstain questions with no has_answer message are dropped, not "
                "relabeled abstain, to avoid manufacturing negatives"
            ),
        },
        "labels_are_dataset_provenance": (
            "relevant_message_indices comes from the dataset's per-message "
            "has_answer flag, not from reading answers (Charter §3 rule 2)"
        ),
    }
    return sessions, queries, report


def main() -> None:
    args = parse_args()
    raw = args.data.read_bytes()
    items = json.loads(raw)
    if args.limit:
        items = items[: args.limit]
    sessions, queries, report = convert(items)
    report["source"] = {
        "file": args.data.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "limit": args.limit,
    }

    for path, rows in ((args.sessions_out, sessions), (args.queries_out, queries)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "dropped_question_ids"}, indent=2))


if __name__ == "__main__":
    main()
