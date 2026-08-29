from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import run_personamem_e2e as harness


def make_args(answer_cache: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        answer_cache=answer_cache,
        run_id="run-1",
        llm_base_url="http://llm.test/v1",
        llm_model="model-a",
        temperature=0.2,
        enable_thinking=True,
        answer_max_tokens=128,
        concurrency=3,
        timeout=1.0,
    )


def make_records(count: int = 3) -> list[dict[str, object]]:
    return [
        {
            "question_id": f"q-{index}",
            "question_type": "preference",
            "user_id": "user-1",
            "question": f"Question {index}",
            "correct_option": 0,
            "options": ["(a) one", "(b) two", "(c) three", "(d) four"],
            "rendered_context": f"Context {index}",
            "facet_count": 1,
            "degraded": False,
            "state_revision": index + 1,
            "retrieval_latency_ms": 1.0,
            "retrieve_attempts": 1,
            "ranking_predicted_option": 0,
            "ranking_correct_supported": True,
            "ranking_reciprocal_rank": 1.0,
        }
        for index in range(count)
    ]


def make_answer(question_id: str) -> dict[str, object]:
    return {
        "answer_correct": True,
        "predicted_option": 0,
        "answer_parse_ok": True,
        "answer_error": None,
        "answer_latency_ms": 1.0,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "answer_text": question_id,
    }


def cache_parameters() -> dict[str, object]:
    return {
        "questions_sha256": "q" * 64,
        "contexts_sha256": "c" * 64,
        "llm_base_url": "http://llm.test/v1",
        "llm_model": "model-a",
        "temperature": 0.2,
        "enable_thinking": True,
        "answer_max_tokens": 128,
    }


def phase_parameters() -> dict[str, str]:
    return {
        "questions_sha256": "q" * 64,
        "contexts_sha256": "c" * 64,
    }


def load_cache(
    path: Path,
    records: list[dict[str, object]],
    **overrides: object,
) -> list[dict[str, object] | None] | None:
    parameters = cache_parameters()
    parameters.update(overrides)
    run_id = str(parameters.pop("run_id", "run-1"))
    return harness._load_answer_cache(path, run_id, records, **parameters)  # type: ignore[arg-type]


def test_answer_progress_uses_completion_order_but_returns_record_order(monkeypatch, capsys):
    records = make_records()
    args = make_args()

    def reverse_completion_order(futures):
        return iter(reversed(list(futures)))

    def record_answer(_client, _args, record):
        question_id = str(record["question_id"])
        return make_answer(question_id)

    monkeypatch.setattr(harness, "as_completed", reverse_completion_order)
    monkeypatch.setattr(harness, "_answer_one", record_answer)

    answers = harness._run_answer_phase(args, records, "", **phase_parameters())

    assert [answer["answer_text"] for answer in answers] == ["q-0", "q-1", "q-2"]
    progress_lines = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("[answer ")
    ]
    assert [line.split()[2] for line in progress_lines] == ["q-2", "q-1", "q-0"]


def test_answer_cache_checkpoints_resume_and_excludes_credentials(tmp_path, monkeypatch):
    records = make_records()
    cache_path = tmp_path / "answers.json"
    args = make_args(cache_path)
    calls: list[str] = []
    save_counts: list[int] = []
    real_save = harness._save_answer_cache

    def fake_answer(_client, _args, record):
        question_id = str(record["question_id"])
        calls.append(question_id)
        return make_answer(question_id)

    def tracked_save(path, run_id, saved_records, answers, **kwargs):
        save_counts.append(sum(answer is not None for answer in answers))
        real_save(path, run_id, saved_records, answers, **kwargs)

    monkeypatch.setattr(harness, "_answer_one", fake_answer)
    monkeypatch.setattr(harness, "_save_answer_cache", tracked_save)
    first = harness._run_answer_phase(args, records, "api-secret", **phase_parameters())

    assert [answer["answer_text"] for answer in first] == ["q-0", "q-1", "q-2"]
    assert sorted(calls) == ["q-0", "q-1", "q-2"]
    assert save_counts == [0, 1, 2, 3]
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert all(answer is not None for answer in payload["answers"])
    assert "api-secret" not in cache_path.read_text(encoding="utf-8")
    assert "api_key" not in payload
    assert list(tmp_path.glob(".answers.json.*.tmp")) == []

    calls.clear()
    save_counts.clear()
    resumed = harness._run_answer_phase(args, records, "api-secret", **phase_parameters())
    assert [answer["answer_text"] for answer in resumed] == ["q-0", "q-1", "q-2"]
    assert calls == []
    assert save_counts == []


def test_answer_error_is_reported_but_not_checkpointed_for_resume(tmp_path, monkeypatch):
    records = make_records(2)
    cache_path = tmp_path / "answers.json"
    args = make_args(cache_path)
    transient_error = make_answer("q-0")
    transient_error.update(
        answer_correct=False,
        answer_parse_ok=False,
        answer_error="HTTPError: status 503",
    )
    first_calls: list[str] = []

    def first_answer(_client, _args, record):
        question_id = str(record["question_id"])
        first_calls.append(question_id)
        return transient_error if question_id == "q-0" else make_answer(question_id)

    monkeypatch.setattr(harness, "_answer_one", first_answer)
    first = harness._run_answer_phase(args, records, "", **phase_parameters())

    assert first[0]["answer_error"] == "HTTPError: status 503"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["answers"] == [None, make_answer("q-1")]

    retry_calls: list[str] = []

    def retry_answer(_client, _args, record):
        question_id = str(record["question_id"])
        retry_calls.append(question_id)
        return make_answer(question_id)

    monkeypatch.setattr(harness, "_answer_one", retry_answer)
    resumed = harness._run_answer_phase(args, records, "", **phase_parameters())

    assert retry_calls == ["q-0"]
    assert [answer["answer_text"] for answer in resumed] == ["q-0", "q-1"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "other-run"),
        ("questions_sha256", "x" * 64),
        ("contexts_sha256", "x" * 64),
        ("llm_base_url", "http://other.test/v1"),
        ("llm_model", "other-model"),
        ("temperature", 0.7),
        ("enable_thinking", False),
        ("answer_max_tokens", 256),
    ],
)
def test_answer_cache_metadata_mismatch_fails_closed(tmp_path, field, value):
    records = make_records()
    cache_path = tmp_path / "answers.json"
    harness._save_answer_cache(
        cache_path,
        "run-1",
        records,
        [make_answer(f"q-{index}") for index in range(3)],
        **cache_parameters(),
    )

    with pytest.raises(ValueError, match=field):
        load_cache(cache_path, records, **{field: value})


def test_answer_cache_record_identity_mismatch_fails_closed(tmp_path):
    records = make_records()
    cache_path = tmp_path / "answers.json"
    harness._save_answer_cache(
        cache_path,
        "run-1",
        records,
        [None, None, None],
        **cache_parameters(),
    )
    runtime_changed_records = [dict(record) for record in records]
    runtime_changed_records[0]["retrieval_latency_ms"] = 999.0
    runtime_changed_records[0]["retrieve_attempts"] = 2
    assert load_cache(cache_path, runtime_changed_records) == [None, None, None]

    changed_records = [dict(record) for record in records]
    changed_records[1]["rendered_context"] = "changed context"

    with pytest.raises(ValueError, match="retrieval_records"):
        load_cache(cache_path, changed_records)


def test_retrieval_cache_is_bound_to_full_service_fingerprint(tmp_path):
    cache_path = tmp_path / "retrieval.json"
    rows = [{"question_id": f"q-{index}"} for index in range(3)]
    records = make_records()
    fingerprint = harness._retrieval_service_fingerprint(
        {
            "extractor_version": "extractor-1",
            "policy_version": "policy-2",
            "embedding_projection_version": "projection-3",
            "semantic_routing_enabled": True,
            "semantic_router_config_sha256": "a" * 64,
            "preference_history_retrieval_enabled": True,
            "preference_history_max_facets": 2,
            "preference_history_max_events": 4,
        }
    )
    harness._save_retrieval_cache(
        cache_path,
        "run-1",
        records,
        service_fingerprint=fingerprint,
        questions_sha256="q" * 64,
        contexts_sha256="c" * 64,
    )

    loaded = harness._load_retrieval_cache(
        cache_path,
        "run-1",
        rows,
        service_fingerprint=fingerprint,
        questions_sha256="q" * 64,
        contexts_sha256="c" * 64,
    )
    assert loaded == records

    changed = dict(fingerprint, preference_history_retrieval_enabled=False)
    assert (
        harness._load_retrieval_cache(
            cache_path,
            "run-1",
            rows,
            service_fingerprint=changed,
            questions_sha256="q" * 64,
            contexts_sha256="c" * 64,
        )
        is None
    )
