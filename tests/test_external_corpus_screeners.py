"""Tests for the LoCoMo, BEAM, and HaluMem corpus screeners.

Each locks the decisions that change what the downstream `query_signal` means. A
screener that mislabels a dataset's abstain split, scrambles session order, or
turns counterpart utterances into user claims yields a number that looks valid and
is not -- the failure mode the Gate Charter's anti-vacuity rules exist for.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.convert_beam_corpus import (
    convert as beam_convert,
)
from benchmarks.convert_beam_corpus import (
    flatten_source_ids,
    parse_anchor,
    parse_probing_questions,
)
from benchmarks.convert_locomo_corpus import _session_keys, parse_session_date
from benchmarks.convert_locomo_corpus import convert as locomo_convert
from benchmarks.screen_halumem_label_source import screen as halumem_screen

# --------------------------------------------------------------------------- LoCoMo


def _locomo_item(
    sample_id: str = "conv-1",
    *,
    speaker_a: str = "Alice",
    speaker_b: str = "Bob",
    sessions: dict[str, list[dict[str, str]]] | None = None,
    dates: dict[str, str] | None = None,
    qa: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    sessions = sessions or {
        "session_1": [
            {"speaker": speaker_a, "dia_id": "D1:1", "text": "I adopted a cat"},
            {"speaker": speaker_b, "dia_id": "D1:2", "text": "nice"},
        ]
    }
    dates = dates or {"session_1_date_time": "1:56 pm on 8 May, 2023"}
    conversation: dict[str, object] = {"speaker_a": speaker_a, "speaker_b": speaker_b}
    conversation.update(sessions)
    conversation.update(dates)
    return {
        "sample_id": sample_id,
        "conversation": conversation,
        "qa": qa if qa is not None else [
            {"question": "what pet?", "answer": "a cat", "evidence": ["D1:1"], "category": 1}
        ],
    }


def test_locomo_session_date_parses_meridiem():
    parsed = parse_session_date("1:56 pm on 8 May, 2023")
    assert (parsed.year, parsed.month, parsed.day) == (2023, 5, 8)
    assert (parsed.hour, parsed.minute) == (13, 56)
    assert parse_session_date("12:30 am on 1 January, 2022").hour == 0


def test_locomo_session_date_rejects_unknown_format():
    with pytest.raises(ValueError):
        parse_session_date("sometime last spring")


def test_locomo_session_keys_sort_numerically_not_lexicographically():
    """`session_10` sorts before `session_2` as a string, which would scramble the
    merged index and silently invalidate every label."""
    conversation = {f"session_{n}": [] for n in (1, 2, 10, 11)}
    conversation.update({f"session_{n}_date_time": "x" for n in (1, 2, 10, 11)})
    assert _session_keys(conversation) == [
        "session_1", "session_2", "session_10", "session_11",
    ]


def test_locomo_speaker_a_is_user_and_speaker_b_is_assistant():
    """Both LoCoMo speakers are human; mapping both to `user` would let one
    person's statements become the other's claims."""
    sessions, _, _ = locomo_convert([_locomo_item()])
    roles = [message["role"] for message in sessions[0]["messages"]]
    assert roles == ["user", "assistant"]


def test_locomo_evidence_only_on_counterpart_turns_is_dropped():
    item = _locomo_item(
        qa=[{"question": "q", "evidence": ["D1:2"], "category": 1}]
    )
    _, queries, report = locomo_convert([item])
    assert queries == []
    assert report["dropped_evidence_only_on_counterpart_turns"] == 1


def test_locomo_category_5_becomes_abstain_with_empty_label():
    """Category 5 carries `adversarial_answer` and is unanswerable, so it is the
    natural abstain set rather than a positive with evidence."""
    item = _locomo_item(
        qa=[
            {
                "question": "q",
                "adversarial_answer": "made up",
                "evidence": ["D1:1"],
                "category": 5,
            }
        ]
    )
    _, queries, report = locomo_convert([item])
    assert queries[0]["expected_decision"] == "abstain"
    assert queries[0]["relevant_message_indices"] == []
    assert report["abstain_queries"] == 1


def test_locomo_malformed_evidence_is_counted_not_guessed():
    item = _locomo_item(
        qa=[{"question": "q", "evidence": ["D9:1 D4:4", "D1:1"], "category": 1}]
    )
    _, queries, report = locomo_convert([item])
    assert report["malformed_evidence_pointers"] == 1
    assert queries[0]["relevant_message_indices"] == [0]


def test_locomo_merges_sessions_in_order_with_advancing_timestamps():
    item = _locomo_item(
        sessions={
            "session_2": [{"speaker": "Alice", "dia_id": "D2:1", "text": "later"}],
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "earlier"},
                {"speaker": "Bob", "dia_id": "D1:2", "text": "ok"},
            ],
        },
        dates={
            "session_1_date_time": "1:00 pm on 8 May, 2023",
            "session_2_date_time": "1:00 pm on 9 May, 2023",
        },
        qa=[{"question": "q", "evidence": ["D2:1"], "category": 1}],
    )
    sessions, queries, _ = locomo_convert([item])
    texts = [message["text"] for message in sessions[0]["messages"]]
    assert texts == ["earlier", "ok", "later"]
    stamps = [message["occurred_at"] for message in sessions[0]["messages"]]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)
    # The label must follow the merged index, not the per-session one.
    assert queries[0]["relevant_message_indices"] == [2]


def test_locomo_labels_precede_the_question():
    sessions, queries, _ = locomo_convert([_locomo_item()])
    asked_at = queries[0]["asked_at_index"]
    assert asked_at == len(sessions[0]["messages"])
    assert all(index < asked_at for index in queries[0]["relevant_message_indices"])


# ----------------------------------------------------------------------------- BEAM


def _beam_row(
    conversation_id: str = "1",
    *,
    turns: list[dict[str, object]] | None = None,
    probing: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, object]:
    turns = turns or [
        {
            "id": 0, "role": "user", "content": "my sprint ends March 29",
            "time_anchor": "March-15-2024", "index": "1,1", "question_type": "main",
        },
        {
            "id": 1, "role": "assistant", "content": "noted",
            "time_anchor": None, "index": None, "question_type": None,
        },
    ]
    probing = probing if probing is not None else {
        "information_extraction": [
            {"question": "when does my sprint end?", "source_chat_ids": [0]}
        ]
    }
    return {
        "conversation_id": conversation_id,
        "chat": [turns],
        "probing_questions": repr(probing),
    }


def test_beam_anchor_parsing():
    parsed = parse_anchor("March-15-2024")
    assert (parsed.year, parsed.month, parsed.day) == (2024, 3, 15)
    assert parse_anchor(None) is None
    assert parse_anchor("not-a-date") is None


def test_beam_probing_questions_accepts_python_repr_and_json():
    payload = {"abstention": [{"question": "q"}]}
    assert parse_probing_questions(repr(payload)) == payload
    assert parse_probing_questions(json.dumps(payload)) == payload


def test_beam_probing_questions_rejects_non_mapping():
    with pytest.raises(TypeError):
        parse_probing_questions("[1, 2, 3]")


def test_beam_source_chat_ids_flattens_both_shapes():
    assert flatten_source_ids([4, 60]) == ["4", "60"]
    assert sorted(
        flatten_source_ids({"first_statement": [58], "second_statement": [24]})
    ) == ["24", "58"]
    assert flatten_source_ids(None) == []


def test_beam_abstention_category_becomes_abstain():
    row = _beam_row(
        probing={
            "abstention": [
                {"question": "unanswerable?", "why_unanswerable": "not discussed"}
            ]
        }
    )
    _, queries, report = beam_convert([row])
    assert queries[0]["expected_decision"] == "abstain"
    assert queries[0]["relevant_message_indices"] == []
    assert report["abstain_queries"] == 1
    # An abstention item has no source_chat_ids; that must not count as a drop.
    assert report["dropped_no_resolvable_evidence"] == 0


def test_beam_assistant_only_evidence_is_dropped():
    row = _beam_row(
        probing={"information_extraction": [{"question": "q", "source_chat_ids": [1]}]}
    )
    _, queries, report = beam_convert([row])
    assert queries == []
    assert report["dropped_evidence_only_on_assistant_turns"] == 1


def test_beam_unresolved_pointer_is_counted_not_guessed():
    row = _beam_row(
        probing={
            "information_extraction": [{"question": "q", "source_chat_ids": [0, 999]}]
        }
    )
    _, queries, report = beam_convert([row])
    assert report["unresolved_source_chat_ids"] == 1
    assert queries[0]["relevant_message_indices"] == [0]


def test_beam_forward_fills_anchors_and_keeps_timestamps_distinct():
    """`time_anchor` is populated only when it changes; without forward-filling
    plus a per-turn offset the validator's one-timestamp check fires."""
    row = _beam_row(
        turns=[
            {"id": 0, "role": "user", "content": "a", "time_anchor": "March-15-2024"},
            {"id": 1, "role": "user", "content": "b", "time_anchor": None},
            {"id": 2, "role": "user", "content": "c", "time_anchor": "April-05-2024"},
        ],
        probing={"information_extraction": [{"question": "q", "source_chat_ids": [1]}]},
    )
    sessions, _, report = beam_convert([row])
    stamps = [message["occurred_at"] for message in sessions[0]["messages"]]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == 3
    assert stamps[0].startswith("2024-03-15")
    assert stamps[2].startswith("2024-04-05")
    assert report["turns_with_forward_filled_anchor"] == 1


# -------------------------------------------------------------------------- HaluMem


def _halumem_user(
    *,
    memory_content: str = "Alice's birth date is 1996-08-02",
    include_event_source: bool = False,
    dialogue: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    dialogue = dialogue or [
        {"role": "user", "content": "I was born on 1996-08-02", "timestamp": "t"},
        {"role": "assistant", "content": "noted", "timestamp": "t"},
    ]
    point: dict[str, object] = {"index": 1, "memory_content": memory_content}
    if include_event_source:
        point["event_source"] = 0
    return {
        "uuid": "u1",
        "sessions": [
            {
                "dialogue": dialogue,
                "memory_points": [point],
                "questions": [
                    {
                        "question": "when was Alice born?",
                        "question_type": "Basic Fact Recall",
                        "evidence": [{"memory_content": memory_content}],
                    },
                    {
                        "question": "what is her middle name?",
                        "question_type": "Memory Boundary",
                        "evidence": [],
                    },
                ],
            }
        ],
    }


def test_halumem_screen_rejects_when_pointer_field_is_unusable():
    """`event_source` would be the principled mapping if it indexed turns; it is
    absent from most memory points, so text matching is the only fallback."""
    report = halumem_screen([_halumem_user()])
    assert report["dataset_pointer"]["usable_as_message_index"] is False
    assert report["verdict"]["recommendation"] == "CORPUS_UNSUITABLE_NO_LABEL_SOURCE"
    assert report["verdict"]["label_source_usable"] is False


def test_halumem_screen_counts_boundary_questions_as_abstain_candidates():
    report = halumem_screen([_halumem_user()])
    assert report["corpus"]["questions"] == 2
    assert report["corpus"]["questions_without_evidence"] == 1
    assert report["corpus"]["abstain_candidate_share"] == 0.5


def test_halumem_screen_measures_overlap_against_user_messages_only():
    """An assistant turn that echoes the memory must not count as its source."""
    report = halumem_screen(
        [
            _halumem_user(
                dialogue=[
                    {"role": "user", "content": "unrelated chatter", "timestamp": "t"},
                    {
                        "role": "assistant",
                        "content": "Alice's birth date is 1996-08-02",
                        "timestamp": "t",
                    },
                ]
            )
        ]
    )
    assert report["text_matching_fallback"]["evidence_items_scored"] == 1
    assert report["text_matching_fallback"]["mean_best_overlap"] < 0.8


def test_halumem_screen_reports_no_answer_model_or_embedding_calls():
    """The screen must stay deterministic and offline, like every other probe."""
    report = halumem_screen([_halumem_user()])
    assert report["answer_model_calls"] == 0
    assert report["embedding_provider_calls"] == 0
