from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import validate_external_corpus as validator


def _session(
    session_id: str = "s1",
    user_key: str = "u1",
    *,
    count: int = 6,
    evidence_text: str = "I set pgbouncer max_connections to 250 for the pool",
    evidence_index: int = 2,
    same_timestamp: bool = False,
) -> dict:
    messages = []
    for index in range(count):
        text = evidence_text if index == evidence_index else f"unrelated filler {index}"
        messages.append(
            {
                "index": index,
                "role": "user" if index % 2 == 0 else "assistant",
                "text": text,
                "occurred_at": (
                    "2026-05-01T09:00:00Z"
                    if same_timestamp
                    else f"2026-05-0{index + 1}T09:00:00Z"
                ),
            }
        )
    return {"session_id": session_id, "user_key": user_key, "messages": messages}


def _query(**overrides) -> dict:
    base = {
        "query_id": "q1",
        "session_id": "s1",
        "user_key": "u1",
        "asked_at_index": 5,
        "query": "What pgbouncer max_connections did I set for the pool?",
        "task_type": "technical_recall",
        "relevant_message_indices": [2],
        "expected_decision": "inject",
    }
    return {**base, **overrides}


def _write(tmp_path: Path, sessions: list[dict], queries: list[dict]) -> tuple[Path, Path]:
    sessions_path = tmp_path / "sessions.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    sessions_path.write_text(
        "\n".join(json.dumps(item) for item in sessions) + "\n", encoding="utf-8"
    )
    queries_path.write_text(
        "\n".join(json.dumps(item) for item in queries) + "\n", encoding="utf-8"
    )
    return sessions_path, queries_path


def test_contract_minimums_match_the_requirements_document() -> None:
    """139 is derived from the SPEC threshold, not chosen."""
    assert validator.MIN_QUERIES == 139
    assert (validator.MIN_ABSTAIN_SHARE, validator.MAX_ABSTAIN_SHARE) == (0.20, 0.40)
    assert validator.MIN_SESSIONS == 30
    assert validator.MIN_USERS == 10
    # The stop band must sit above the signal level that made PersonaMem unusable.
    assert validator.SIGNAL_STOP > validator.PERSONAMEM_SIGNAL


def test_query_naming_its_evidence_scores_signal(tmp_path: Path) -> None:
    sessions, queries = _write(tmp_path, [_session()], [_query()])

    report = validator.validate(sessions, queries, sample=True)

    assert report["structure"]["passed"] is True
    assert report["query_signal"]["query_signal"] > 0.35
    assert report["query_signal"]["band"] == "GOOD_PROCEED_TO_ANNOTATION"


def test_vague_query_is_flagged_stop(tmp_path: Path) -> None:
    """The PersonaMem failure mode: the query names nothing in its evidence."""
    sessions, queries = _write(
        tmp_path,
        [_session()],
        [_query(query="I recently had a memorable experience worth discussing.")],
    )

    report = validator.validate(sessions, queries, sample=True)

    assert report["query_signal"]["query_signal"] == 0.0
    assert report["query_signal"]["band"] == "STOP_DO_NOT_ANNOTATE"
    assert "Do not annotate" in report["verdict"]["next_step"]


def test_label_citing_a_future_message_is_rejected(tmp_path: Path) -> None:
    """A live system answering at turn N cannot see turn N+1."""
    sessions, queries = _write(
        tmp_path, [_session()], [_query(relevant_message_indices=[5], asked_at_index=5)]
    )

    report = validator.validate(sessions, queries, sample=True)

    assert report["structure"]["queries"]["causality_violations"] == 1
    assert report["structure"]["passed"] is False
    assert any("must precede" in error for error in report["structure"]["errors"])


def test_abstain_with_relevant_indices_is_rejected(tmp_path: Path) -> None:
    sessions, queries = _write(
        tmp_path,
        [_session()],
        [_query(expected_decision="abstain", relevant_message_indices=[2])],
    )

    report = validator.validate(sessions, queries, sample=True)

    assert report["structure"]["queries"]["abstain_with_relevant"] == 1
    assert report["structure"]["passed"] is False


def test_inject_without_relevant_indices_is_rejected(tmp_path: Path) -> None:
    sessions, queries = _write(
        tmp_path, [_session()], [_query(relevant_message_indices=[])]
    )

    report = validator.validate(sessions, queries, sample=True)

    assert report["structure"]["queries"]["inject_without_relevant"] == 1
    assert report["structure"]["passed"] is False


def test_index_gaps_are_rejected(tmp_path: Path) -> None:
    session = _session()
    session["messages"] = [m for m in session["messages"] if m["index"] != 3]
    sessions, queries = _write(tmp_path, [session], [_query()])

    report = validator.validate(sessions, queries, sample=True)

    assert report["structure"]["passed"] is False
    assert any("without gaps" in error for error in report["structure"]["errors"])


def test_single_timestamp_across_a_session_is_reported(tmp_path: Path) -> None:
    """Export-time clocks make decay and stale-active meaningless."""
    sessions, queries = _write(tmp_path, [_session(same_timestamp=True)], [_query()])

    report = validator.validate(sessions, queries, sample=True)

    assert report["structure"]["sessions"]["sessions_with_one_timestamp"] == 1


def test_missing_timestamps_are_counted(tmp_path: Path) -> None:
    session = _session()
    del session["messages"][1]["occurred_at"]
    sessions, queries = _write(tmp_path, [session], [_query()])

    report = validator.validate(sessions, queries, sample=True)

    assert report["structure"]["sessions"]["messages_missing_occurred_at"] == 1


def test_query_referencing_an_unknown_session_is_rejected(tmp_path: Path) -> None:
    sessions, queries = _write(tmp_path, [_session()], [_query(session_id="missing")])

    report = validator.validate(sessions, queries, sample=True)

    assert report["structure"]["queries"]["queries_accepted"] == 0
    assert report["structure"]["passed"] is False


def test_user_key_disagreement_is_rejected(tmp_path: Path) -> None:
    """Cross-session identity is the whole point; a mismatch is a real defect."""
    sessions, queries = _write(tmp_path, [_session()], [_query(user_key="someone-else")])

    report = validator.validate(sessions, queries, sample=True)

    assert report["structure"]["passed"] is False
    assert any("user_key" in error for error in report["structure"]["errors"])


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    sessions, queries = _write(
        tmp_path, [_session(), _session()], [_query(), _query()]
    )

    report = validator.validate(sessions, queries, sample=True)

    assert report["structure"]["passed"] is False
    errors = " ".join(report["structure"]["errors"])
    assert "duplicate session_id" in errors
    assert "duplicate query_id" in errors


def test_malformed_json_line_is_reported_not_crashed(tmp_path: Path) -> None:
    sessions_path = tmp_path / "sessions.jsonl"
    sessions_path.write_text(json.dumps(_session()) + "\n{not json\n", encoding="utf-8")
    queries_path = tmp_path / "queries.jsonl"
    queries_path.write_text(json.dumps(_query()) + "\n", encoding="utf-8")

    report = validator.validate(sessions_path, queries_path, sample=True)

    assert report["structure"]["passed"] is False
    assert any("invalid JSON" in error for error in report["structure"]["errors"])


def test_abstain_queries_are_excluded_from_the_signal_measurement(tmp_path: Path) -> None:
    """An abstain query has no evidence to locate; scoring it would dilute the mean."""
    sessions, queries = _write(
        tmp_path,
        [_session()],
        [
            _query(),
            _query(
                query_id="q2",
                query="What is my preferred airline?",
                relevant_message_indices=[],
                expected_decision="abstain",
            ),
        ],
    )

    report = validator.validate(sessions, queries, sample=True)

    assert report["structure"]["queries"]["queries_accepted"] == 2
    assert report["query_signal"]["scored_queries"] == 1


def test_privacy_scan_finds_credentials_and_reports_locations_only(
    tmp_path: Path,
) -> None:
    session = _session()
    session["messages"][1]["text"] = (
        "key sk-abcdefghij0123456789 and reach me at ops@example.com or 13800138000"
    )
    sessions, queries = _write(tmp_path, [session], [_query()])

    report = validator.validate(sessions, queries, sample=True)
    scan = report["privacy_scan"]

    assert scan["clean"] is False
    assert {"openai_key", "email", "cn_mobile"} <= set(scan["patterns_matched"])
    # The report must be shareable even when the corpus is not.
    serialized = json.dumps(report)
    assert "sk-abcdefghij0123456789" not in serialized
    assert "ops@example.com" not in serialized
    assert scan["locations"][0]["session_id"] == "s1"


def test_clean_corpus_reports_a_clean_scan_with_its_caveat(tmp_path: Path) -> None:
    sessions, queries = _write(tmp_path, [_session()], [_query()])

    scan = validator.validate(sessions, queries, sample=True)["privacy_scan"]

    assert scan["clean"] is True
    assert "not a compliance guarantee" in scan["caveat"]


def test_sample_mode_tolerates_volume_but_not_structure(tmp_path: Path) -> None:
    """Sample mode exists to read the signal early, not to waive correctness."""
    sessions, queries = _write(tmp_path, [_session()], [_query()])

    sample = validator.validate(sessions, queries, sample=True)
    full = validator.validate(sessions, queries, sample=False)

    assert sample["volume"]["passed"] is False
    assert sample["verdict"]["contract_satisfied"] is True
    # The same undersized corpus fails the full contract.
    assert full["verdict"]["contract_satisfied"] is False


def test_structural_failure_blocks_the_contract_even_in_sample_mode(
    tmp_path: Path,
) -> None:
    sessions, queries = _write(
        tmp_path, [_session()], [_query(relevant_message_indices=[5])]
    )

    report = validator.validate(sessions, queries, sample=True)

    assert report["verdict"]["contract_satisfied"] is False
    assert "Fix the structural errors first" in report["verdict"]["next_step"]


def test_signal_ignores_tokens_common_to_the_whole_session(tmp_path: Path) -> None:
    """Shared session vocabulary must not inflate the score.

    Otherwise any query repeating ordinary words from its context looks informative.
    """
    session = _session(evidence_text="unrelated filler 2", evidence_index=2)
    sessions, queries = _write(
        tmp_path, [session], [_query(query="tell me about unrelated filler")]
    )

    report = validator.validate(sessions, queries, sample=True)

    # The labeled message is indistinguishable from its background, so there is
    # nothing to score rather than a falsely high score.
    assert report["query_signal"]["scored_queries"] == 0


def test_validation_is_deterministic(tmp_path: Path) -> None:
    sessions, queries = _write(tmp_path, [_session()], [_query()])

    first = validator.validate(sessions, queries, sample=True)
    second = validator.validate(sessions, queries, sample=True)

    assert first == second


def test_report_states_it_does_not_move_the_gate(tmp_path: Path) -> None:
    sessions, queries = _write(tmp_path, [_session()], [_query()])

    report = validator.validate(sessions, queries, sample=True)

    assert "does not score Recall@10" in report["scope"]
    assert report["answer_model_calls"] == 0
    assert report["requirements_document"] == "MENO_DATA_REQUIREMENTS.md"


def test_provenance_pins_both_input_files(tmp_path: Path) -> None:
    sessions, queries = _write(tmp_path, [_session()], [_query()])

    report = validator.validate(sessions, queries, sample=True)
    provenance = report["provenance"]

    assert len(provenance["sessions_sha256"]) == 64
    assert len(provenance["queries_sha256"]) == 64
    assert provenance["sessions_sha256"] != provenance["queries_sha256"]


@pytest.mark.parametrize(
    ("role", "accepted"),
    [("user", True), ("assistant", True), ("system", True), ("tool", False)],
)
def test_only_contract_roles_are_accepted(
    tmp_path: Path, role: str, accepted: bool
) -> None:
    session = _session()
    session["messages"][1]["role"] = role
    sessions, queries = _write(tmp_path, [session], [_query()])

    report = validator.validate(sessions, queries, sample=True)

    assert report["structure"]["passed"] is accepted
