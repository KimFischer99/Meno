"""Convert LoCoMo into Meno's external-corpus contract.

Companion to `MENO_DATA_REQUIREMENTS.md` §1.3 and `validate_external_corpus.py`.
Screens LoCoMo as a Gate Charter #13/#14 candidate corpus: the decisive output is
`query_signal`, which predicts whether a curated set over it can evidence
anything. Emits no labels of its own -- `relevant_message_indices` comes from
LoCoMo's own per-turn `evidence` pointers (`dia_id`), which is dataset provenance
rather than answer-reading, and so is admissible under Charter §3 rule 2.

LoCoMo's labels are the cleanest of the screened corpora: `evidence` names exact
`dia_id`s and every turn carries one, so no window-widening or region-guessing is
needed. Four structural facts change what the measurement means, all recorded in
the emitted report:

1. **Both speakers are human.** LoCoMo is a two-person conversation with no
   assistant. Meno only extracts claims from `user` turns by design, so mapping
   both speakers to `user` would let one person's statements become the other's
   claims. Speaker A maps to `user` and speaker B to `assistant`, and queries are
   kept only where the evidence lies on speaker A's turns -- otherwise the label
   points at evidence Meno deliberately refuses to extract from.

2. **Category 5 is adversarial and carries `adversarial_answer`, not `answer`.**
   These questions are unanswerable from the conversation, which makes them the
   natural abstain set. OmniMemEval excludes them from its retrieval numbers; here
   they become `expected_decision: abstain` with an empty label, because an
   abstain query is exactly a question whose evidence is not there.

3. **One conversation is one session.** A LoCoMo conversation already spans many
   dated sessions between the same two people, so concatenating them in session
   order under one `user_key` matches how Meno sees a user: cross-session state
   under one identity.

4. **Timestamps are synthesized within a session.** Session dates are per-session,
   so each turn is offset by one minute to preserve order. Absolute times are not
   real; relative order is. Without this the validator's
   `sessions_with_one_timestamp` check fires, and correctly so.

Nine of 2815 evidence pointers in the published file are malformed (embedded
spaces or semicolons, or ids for sessions that do not exist). They are dropped and
counted rather than guessed at.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

TURN_INTERVAL = timedelta(minutes=1)

# LoCoMo's adversarial split: the answer is not in the conversation.
ADVERSARIAL_CATEGORY = 5

# "3:47 pm on 17 March, 2022"
_DATE = re.compile(
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<meridiem>am|pm)\s+on\s+"
    r"(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+),?\s+(?P<year>\d{4})",
    re.IGNORECASE,
)
_MONTHS = {
    month: number
    for number, month in enumerate(
        [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ],
        start=1,
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="locomo10.json")
    parser.add_argument("--sessions-out", type=Path, required=True)
    parser.add_argument("--queries-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    return parser.parse_args()


def parse_session_date(value: str) -> datetime:
    match = _DATE.search(value)
    if match is None:
        raise ValueError(f"unrecognized LoCoMo session date: {value!r}")
    hour = int(match["hour"]) % 12
    if match["meridiem"].lower() == "pm":
        hour += 12
    month = _MONTHS.get(match["month"].lower())
    if month is None:
        raise ValueError(f"unrecognized month in session date: {value!r}")
    return datetime(
        int(match["year"]), month, int(match["day"]), hour, int(match["minute"]), tzinfo=UTC
    )


def _session_keys(conversation: dict[str, Any]) -> list[str]:
    """Session keys ordered by their trailing number, not lexicographically.

    `session_10` sorts before `session_2` as a string, which would scramble the
    merged index and silently invalidate every label.
    """
    keys = [
        key
        for key in conversation
        if key.startswith("session_") and not key.endswith("date_time")
    ]
    return sorted(keys, key=lambda key: int(key.rsplit("_", 1)[1]))


def convert(items: list[dict[str, Any]]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    sessions: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    malformed_evidence: list[str] = []
    dropped_assistant_only: list[str] = []
    dropped_no_evidence: list[str] = []
    abstain_ids: list[str] = []
    turns_total = 0

    for item in items:
        sample_id = item["sample_id"]
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]

        messages: list[dict[str, Any]] = []
        # dia_id -> merged index, and whether that turn is speaker A's.
        index_of: dict[str, int] = {}
        for key in _session_keys(conversation):
            start = parse_session_date(conversation[f"{key}_date_time"])
            for turn_index, turn in enumerate(conversation[key]):
                index = len(messages)
                turns_total += 1
                messages.append(
                    {
                        "index": index,
                        # Decision 1: speaker A is the modeled user; speaker B is
                        # the counterpart, mapped to assistant so Meno does not
                        # extract claims about A from B's utterances.
                        "role": "user" if turn["speaker"] == speaker_a else "assistant",
                        "text": turn["text"],
                        "occurred_at": (
                            start + turn_index * TURN_INTERVAL
                        ).isoformat().replace("+00:00", "Z"),
                    }
                )
                index_of[turn["dia_id"]] = index

        if not messages:
            continue

        for position, qa in enumerate(item["qa"]):
            query_id = f"{sample_id}:q{position}"
            is_adversarial = qa.get("category") == ADVERSARIAL_CATEGORY
            raw_evidence = qa.get("evidence") or []
            resolved: list[int] = []
            for pointer in raw_evidence:
                if pointer not in index_of:
                    malformed_evidence.append(f"{query_id}:{pointer}")
                    continue
                resolved.append(index_of[pointer])

            if is_adversarial:
                abstain_ids.append(query_id)
                relevant: list[int] = []
            else:
                # Decision 1 continued: a label on a speaker-B turn points at
                # evidence Meno refuses to extract from, so the query would be
                # unanswerable for reasons unrelated to retrieval quality.
                relevant = sorted(
                    {index for index in resolved if messages[index]["role"] == "user"}
                )
                if not resolved:
                    dropped_no_evidence.append(query_id)
                    continue
                if not relevant:
                    dropped_assistant_only.append(query_id)
                    continue

            asked_at = len(messages)
            queries.append(
                {
                    "query_id": query_id,
                    "session_id": sample_id,
                    "user_key": sample_id,
                    "asked_at_index": asked_at,
                    "query": qa["question"],
                    "task_type": f"category_{qa.get('category')}",
                    "relevant_message_indices": relevant,
                    "expected_decision": "abstain" if is_adversarial else "inject",
                }
            )

        sessions.append(
            {"session_id": sample_id, "user_key": sample_id, "messages": messages}
        )

    report = {
        "converter": "locomo -> meno external-corpus-v1",
        "items_read": len(items),
        "sessions_emitted": len(sessions),
        "queries_emitted": len(queries),
        "messages_emitted": turns_total,
        "abstain_queries": len(abstain_ids),
        "dropped_no_resolvable_evidence": len(dropped_no_evidence),
        "dropped_evidence_only_on_counterpart_turns": len(dropped_assistant_only),
        "malformed_evidence_pointers": len(malformed_evidence),
        "malformed_evidence_samples": sorted(malformed_evidence)[:10],
        "conversion_decisions": {
            "speaker_roles": (
                "speaker_a -> user (the modeled subject), speaker_b -> assistant; "
                "queries whose evidence lies only on speaker_b turns are dropped "
                "because Meno does not extract user claims from counterpart turns"
            ),
            "adversarial_is_abstain": (
                f"category {ADVERSARIAL_CATEGORY} carries adversarial_answer and is "
                "unanswerable from the conversation, so it maps to "
                "expected_decision=abstain with an empty label"
            ),
            "one_conversation_is_one_session": (
                "dated sessions concatenated in session order under "
                "user_key=sample_id; 'session' means one user's whole history"
            ),
            "timestamps_synthesized": (
                f"per-session date plus {int(TURN_INTERVAL.total_seconds() // 60)} "
                "minute(s) per turn; absolute times are not real, order is"
            ),
        },
        "labels_are_dataset_provenance": (
            "relevant_message_indices comes from LoCoMo's own per-turn evidence "
            "dia_id pointers, not from reading answers (Charter §3 rule 2)"
        ),
    }
    return sessions, queries, report


def main() -> None:
    args = parse_args()
    raw = args.data.read_bytes()
    items = json.loads(raw)
    sessions, queries, report = convert(items)
    report["source"] = {
        "file": args.data.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "license": "CC BY-NC 4.0 (not redistributed by this repository)",
    }
    for path, rows in ((args.sessions_out, sessions), (args.queries_out, queries)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "malformed_evidence_samples"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
