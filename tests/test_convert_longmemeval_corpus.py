"""Tests for the LongMemEval -> Meno external-corpus converter.

These lock the decisions that change what the downstream `query_signal`
measurement means. A converter that silently mislabels the abstention subset, or
that emits one timestamp per session, produces a number that looks valid and is
not -- the same failure mode the Gate Charter's anti-vacuity rules exist for.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.convert_longmemeval_corpus import convert, parse_session_date

ROOT = Path(__file__).resolve().parents[1]


def _item(
    question_id: str,
    *,
    sessions: list[list[dict[str, object]]],
    dates: list[str],
    question: str = "which one came first?",
    question_type: str = "temporal-reasoning",
) -> dict[str, object]:
    session_ids = [f"{question_id}_s{index}" for index in range(len(sessions))]
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": question,
        "answer": "some answer",
        "question_date": "2023/06/01 (Thu) 10:00",
        "haystack_dates": dates,
        "haystack_session_ids": session_ids,
        "haystack_sessions": sessions,
        "answer_session_ids": session_ids,
    }


def test_parse_session_date_strips_weekday():
    parsed = parse_session_date("2023/05/20 (Sat) 02:21")
    assert (parsed.year, parsed.month, parsed.day) == (2023, 5, 20)
    assert (parsed.hour, parsed.minute) == (2, 21)


def test_parse_session_date_rejects_unknown_format():
    with pytest.raises(ValueError):
        parse_session_date("20th of May")


def test_has_answer_flags_become_message_level_labels():
    item = _item(
        "q1",
        sessions=[
            [
                {"role": "user", "content": "unrelated chatter", "has_answer": False},
                {"role": "user", "content": "I bought a bike", "has_answer": True},
            ]
        ],
        dates=["2023/05/20 (Sat) 02:21"],
    )
    _, queries, report = convert([item])
    assert queries[0]["relevant_message_indices"] == [1]
    assert queries[0]["expected_decision"] == "inject"
    assert report["dropped_no_has_answer_flag"] == 0


def test_has_answer_accepts_the_string_form():
    """The dataset stores this flag as both bool True and the string "True"."""
    item = _item(
        "q1",
        sessions=[
            [{"role": "user", "content": "I bought a bike", "has_answer": "True"}]
        ],
        dates=["2023/05/20 (Sat) 02:21"],
    )
    _, queries, _ = convert([item])
    assert queries[0]["relevant_message_indices"] == [0]


def test_abs_subset_becomes_abstain_and_discards_near_miss_flags():
    """`_abs` items are LongMemEval's own abstention set.

    Their `has_answer` flags point at evidence for a *related* question, so
    keeping them would turn an unanswerable question into a labeled positive.
    """
    item = _item(
        "q1_abs",
        sessions=[
            [{"role": "user", "content": "I booked San Francisco", "has_answer": True}]
        ],
        dates=["2023/05/20 (Sat) 02:21"],
    )
    _, queries, report = convert([item])
    assert queries[0]["expected_decision"] == "abstain"
    assert queries[0]["relevant_message_indices"] == []
    assert report["abstain_queries"] == 1
    assert report["dropped_no_has_answer_flag"] == 0


def test_unflagged_non_abstain_question_is_dropped_not_relabeled():
    """Dropping keeps the abstain share honest; relabeling would inflate it."""
    item = _item(
        "q1",
        sessions=[[{"role": "user", "content": "nothing decisive", "has_answer": False}]],
        dates=["2023/05/20 (Sat) 02:21"],
    )
    sessions, queries, report = convert([item])
    assert sessions == []
    assert queries == []
    assert report["dropped_question_ids"] == ["q1"]
    assert report["abstain_queries"] == 0


def test_sessions_are_merged_in_date_order_and_timestamps_advance():
    """Per-session dates alone would give every message in a session one clock
    reading, which the validator flags as an export that used its own clock."""
    item = _item(
        "q1",
        sessions=[
            [
                {"role": "user", "content": "later one", "has_answer": True},
                {"role": "assistant", "content": "later reply", "has_answer": False},
            ],
            [{"role": "user", "content": "earlier one", "has_answer": False}],
        ],
        dates=["2023/05/20 (Sat) 02:21", "2023/01/02 (Mon) 08:00"],
    )
    sessions, queries, _ = convert([item])
    texts = [message["text"] for message in sessions[0]["messages"]]
    assert texts == ["earlier one", "later one", "later reply"]
    stamps = [message["occurred_at"] for message in sessions[0]["messages"]]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)
    # The label must follow the merged index, not the original per-session one.
    assert queries[0]["relevant_message_indices"] == [1]


def test_indices_are_contiguous_and_labels_precede_the_question():
    """Both are validator contract rules; violating either voids the measurement."""
    item = _item(
        "q1",
        sessions=[
            [
                {"role": "user", "content": "a", "has_answer": True},
                {"role": "tool", "content": "skipped", "has_answer": False},
                {"role": "assistant", "content": "b", "has_answer": False},
            ]
        ],
        dates=["2023/05/20 (Sat) 02:21"],
    )
    sessions, queries, report = convert([item])
    indices = [message["index"] for message in sessions[0]["messages"]]
    assert indices == list(range(len(indices)))
    assert report["turns_skipped_unknown_role"] == 1
    asked_at = queries[0]["asked_at_index"]
    assert asked_at == len(sessions[0]["messages"])
    assert all(index < asked_at for index in queries[0]["relevant_message_indices"])


def test_oracle_check_counts_haystacks_identical_to_the_answer_set():
    """The reason the existing LongMemEval retrieval numbers are an upper bound."""
    item = _item(
        "q1",
        sessions=[[{"role": "user", "content": "a", "has_answer": True}]],
        dates=["2023/05/20 (Sat) 02:21"],
    )
    distractor = _item(
        "q2",
        sessions=[[{"role": "user", "content": "b", "has_answer": True}]],
        dates=["2023/05/20 (Sat) 02:21"],
    )
    distractor["answer_session_ids"] = []
    _, _, report = convert([item, distractor])
    assert report["oracle_check"]["items_where_haystack_equals_answer_sessions"] == 1


def test_converted_output_satisfies_the_validator_contract(tmp_path):
    """End-to-end: the emitted files must pass structural validation, otherwise
    the query_signal number downstream is measured on a malformed corpus."""
    items = [
        _item(
            f"q{n}",
            sessions=[
                [
                    {"role": "user", "content": f"my bike is model {n}", "has_answer": True},
                    {"role": "assistant", "content": "noted", "has_answer": False},
                ]
            ],
            dates=["2023/05/20 (Sat) 02:21"],
        )
        for n in range(3)
    ]
    items.append(
        _item(
            "q9_abs",
            sessions=[[{"role": "user", "content": "unrelated", "has_answer": False}]],
            dates=["2023/05/20 (Sat) 02:21"],
        )
    )
    data = tmp_path / "lme.json"
    data.write_text(json.dumps(items))
    sessions_out = tmp_path / "sessions.jsonl"
    queries_out = tmp_path / "queries.jsonl"

    subprocess.run(
        [
            sys.executable,
            "benchmarks/convert_longmemeval_corpus.py",
            "--data", str(data),
            "--sessions-out", str(sessions_out),
            "--queries-out", str(queries_out),
            "--report-out", str(tmp_path / "conversion.json"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
    )
    validation = tmp_path / "validation.json"
    subprocess.run(
        [
            sys.executable,
            "benchmarks/validate_external_corpus.py",
            "--sessions", str(sessions_out),
            "--queries", str(queries_out),
            "--output", str(validation),
            "--sample",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
    )
    report = json.loads(validation.read_text())
    assert report["structure"]["passed"], report["structure"]
    assert report["structure"]["sessions"]["sessions_with_one_timestamp"] == 0
