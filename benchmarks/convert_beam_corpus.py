"""Convert BEAM into Meno's external-corpus contract.

Companion to `MENO_DATA_REQUIREMENTS.md` §1.3 and `validate_external_corpus.py`.
Screens BEAM as a Gate Charter #13/#14 candidate corpus. Emits no labels of its
own: `relevant_message_indices` comes from BEAM's own `source_chat_ids` pointers,
which name exact chat turn ids and resolved 55/55 on a first pass -- dataset
provenance, admissible under Charter §3 rule 2.

BEAM has the cleanest label pointers of the screened corpora, but four structural
facts change what the measurement means. All are recorded in the emitted report:

1. **`probing_questions` is a Python repr string, not JSON.** It is parsed with
   `ast.literal_eval`, which evaluates literals only. The field arrives from a
   third-party dataset, so `json.loads` is tried first and `literal_eval` is the
   fallback -- never `eval`.

2. **`source_chat_ids` has two shapes.** Some categories give a flat list
   (`[4, 60, 116]`); the ones that test a change over time give a labeled dict
   (`{'first_statement': [58], 'second_statement': [24]}`). Both are flattened,
   and the labeling is discarded -- the contract wants the evidence set, not which
   side of a contradiction each item is on.

3. **Labels on assistant turns are dropped.** BEAM alternates user/assistant, and
   Meno only extracts claims from user turns by design. A query whose evidence sits
   only on assistant turns would be unanswerable for reasons unrelated to retrieval
   quality. `instruction_following` and `preference_following` are the categories
   most affected, since the instruction is often restated by the assistant.

4. **Timestamps come from `time_anchor`, forward-filled.** The field is populated
   only on the turn that introduces a new anchor and is `None` elsewhere, so the
   last seen anchor carries forward and each turn is offset by one minute to keep
   order. Absolute times are as-published at day resolution; within-day order is
   synthesized.

The `abstention` category is BEAM's own unanswerable set (it carries
`why_unanswerable` and no `source_chat_ids`), so it maps to
`expected_decision: abstain` with an empty label.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

TURN_INTERVAL = timedelta(minutes=1)

# BEAM's own unanswerable split.
ABSTENTION_CATEGORY = "abstention"

# "March-15-2024"
_ANCHOR = re.compile(r"^(?P<month>[A-Za-z]+)-(?P<day>\d{1,2})-(?P<year>\d{4})$")
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
# Used only when a conversation opens before any anchor appears, so that the
# emitted timestamps stay ordered and parseable.
FALLBACK_EPOCH = datetime(2024, 1, 1, tzinfo=UTC)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="BEAM *.parquet")
    parser.add_argument("--sessions-out", type=Path, required=True)
    parser.add_argument("--queries-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    return parser.parse_args()


def parse_anchor(value: str | None) -> datetime | None:
    if not value:
        return None
    match = _ANCHOR.match(value.strip())
    if match is None:
        return None
    month = _MONTHS.get(match["month"].lower())
    if month is None:
        return None
    return datetime(int(match["year"]), month, int(match["day"]), tzinfo=UTC)


def parse_probing_questions(raw: str) -> dict[str, list[dict[str, Any]]]:
    """BEAM publishes this column as a Python repr string.

    `json.loads` is tried first so a future JSON-clean release needs no change;
    `ast.literal_eval` handles the current single-quoted form and evaluates
    literals only, so untrusted dataset content cannot execute.
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = ast.literal_eval(raw)
    if not isinstance(parsed, dict):
        raise TypeError("probing_questions did not parse to a mapping")
    return parsed


def flatten_source_ids(value: Any) -> list[str]:
    """`source_chat_ids` is a flat list, or a dict of labeled lists."""
    if value is None:
        return []
    if isinstance(value, dict):
        flat: list[str] = []
        for member in value.values():
            flat.extend(
                str(item) for item in (member if isinstance(member, list) else [member])
            )
        return flat
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def convert(rows: list[dict[str, Any]]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    sessions: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    abstain_ids: list[str] = []
    dropped_no_evidence: list[str] = []
    dropped_assistant_only: list[str] = []
    unresolved_pointers: list[str] = []
    turns_total = 0
    anchors_forward_filled = 0

    for row in rows:
        conversation_id = str(row["conversation_id"])
        messages: list[dict[str, Any]] = []
        index_of: dict[str, int] = {}
        anchor = None
        offset = 0

        for block in row["chat"]:
            for turn in block:
                parsed = parse_anchor(turn.get("time_anchor"))
                if parsed is not None:
                    if parsed != anchor:
                        offset = 0
                    anchor = parsed
                else:
                    anchors_forward_filled += 1
                base = anchor or FALLBACK_EPOCH
                index = len(messages)
                turns_total += 1
                messages.append(
                    {
                        "index": index,
                        "role": turn["role"],
                        "text": turn["content"],
                        "occurred_at": (
                            base + offset * TURN_INTERVAL
                        ).isoformat().replace("+00:00", "Z"),
                    }
                )
                offset += 1
                index_of[str(turn["id"])] = index

        if not messages:
            continue

        probing = parse_probing_questions(row["probing_questions"])
        for category, items in sorted(probing.items()):
            for position, item in enumerate(items):
                query_id = f"{conversation_id}:{category}:{position}"
                question = item.get("question")
                if not isinstance(question, str) or not question.strip():
                    continue
                is_abstain = category == ABSTENTION_CATEGORY
                resolved: list[int] = []
                for pointer in flatten_source_ids(item.get("source_chat_ids")):
                    if pointer not in index_of:
                        unresolved_pointers.append(f"{query_id}:{pointer}")
                        continue
                    resolved.append(index_of[pointer])

                if is_abstain:
                    abstain_ids.append(query_id)
                    relevant: list[int] = []
                else:
                    # Decision 3: Meno does not extract user claims from assistant
                    # turns, so an assistant-only label is unreachable by design.
                    relevant = sorted(
                        {
                            index
                            for index in resolved
                            if messages[index]["role"] == "user"
                        }
                    )
                    if not resolved:
                        dropped_no_evidence.append(query_id)
                        continue
                    if not relevant:
                        dropped_assistant_only.append(query_id)
                        continue

                queries.append(
                    {
                        "query_id": query_id,
                        "session_id": conversation_id,
                        "user_key": conversation_id,
                        "asked_at_index": len(messages),
                        "query": question,
                        "task_type": category,
                        "relevant_message_indices": relevant,
                        "expected_decision": "abstain" if is_abstain else "inject",
                    }
                )

        sessions.append(
            {
                "session_id": conversation_id,
                "user_key": conversation_id,
                "messages": messages,
            }
        )

    report = {
        "converter": "beam -> meno external-corpus-v1",
        "items_read": len(rows),
        "sessions_emitted": len(sessions),
        "queries_emitted": len(queries),
        "messages_emitted": turns_total,
        "abstain_queries": len(abstain_ids),
        "dropped_no_resolvable_evidence": len(dropped_no_evidence),
        "dropped_evidence_only_on_assistant_turns": len(dropped_assistant_only),
        "unresolved_source_chat_ids": len(unresolved_pointers),
        "unresolved_samples": sorted(unresolved_pointers)[:10],
        "turns_with_forward_filled_anchor": anchors_forward_filled,
        "conversion_decisions": {
            "probing_questions_is_a_python_repr": (
                "parsed with json.loads first, ast.literal_eval as fallback; "
                "literal-only evaluation, never eval"
            ),
            "source_chat_ids_two_shapes": (
                "flat list, or dict of labeled lists for change-over-time "
                "categories; both flattened and the labels discarded"
            ),
            "assistant_only_labels_dropped": (
                "Meno extracts user claims from user turns only, so a query whose "
                "evidence sits only on assistant turns is unreachable by design"
            ),
            "timestamps_from_forward_filled_time_anchor": (
                f"time_anchor is set only when it changes; last anchor carries "
                f"forward plus {int(TURN_INTERVAL.total_seconds() // 60)} minute(s) "
                "per turn. Day resolution is as-published; intra-day order is synthesized"
            ),
            "abstention_is_abstain": (
                f"the {ABSTENTION_CATEGORY} category carries why_unanswerable and no "
                "source_chat_ids, so it maps to expected_decision=abstain"
            ),
        },
        "labels_are_dataset_provenance": (
            "relevant_message_indices comes from BEAM's own source_chat_ids turn "
            "pointers, not from reading answers (Charter §3 rule 2)"
        ),
    }
    return sessions, queries, report


def main() -> None:
    args = parse_args()
    from pyarrow import parquet

    raw = args.data.read_bytes()
    rows = parquet.read_table(args.data).to_pylist()
    sessions, queries, report = convert(rows)
    report["source"] = {
        "file": args.data.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "license": "CC BY-SA 4.0 (not redistributed by this repository)",
    }
    for path, emitted in ((args.sessions_out, sessions), (args.queries_out, queries)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in emitted)
        )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "unresolved_samples"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
