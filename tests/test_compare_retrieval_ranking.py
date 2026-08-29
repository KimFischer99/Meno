from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import compare_retrieval_ranking

BASELINE = Path("artifacts/benchmarks/results/vps-phaseb-baseline-retrieval-cache-formal-hy3.json")
TREATMENT = Path(
    "artifacts/benchmarks/results/vps-phaseb-treatment-retrieval-cache-formal-hy3.json"
)
PHASEB2 = Path("artifacts/benchmarks/results/vps-phaseb2-retrieval-cache-development-hy3.json")


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _minimal_payload(**overrides) -> dict:
    payload = {
        "run_id": "unit",
        "questions_sha256": "a" * 64,
        "contexts_sha256": "b" * 64,
        "records": [
            {
                "question_id": "q1",
                "question_type": "track_full_preference_evolution",
                "correct_option": 1,
                "rendered_context": '<user_context user_id="u" revision="1">\n'
                "- [preference] concise answers (confidence=0.900, evidence=e1)\n"
                "</user_context>",
                "facet_count": 1,
                "ranking_predicted_option": 1,
                "ranking_correct_supported": True,
                "ranking_reciprocal_rank": 1.0,
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_comparison_reproduces_the_phase_b_product_gate_regression() -> None:
    report = compare_retrieval_ranking.compare(BASELINE, TREATMENT)

    assert report["answer_model_calls"] == 0
    assert report["dataset"]["question_count"] == 589
    track = report["by_question_type"]["track_full_preference_evolution"]
    # The frozen gate requires +0.05 absolute on this subset; the measured delta is negative.
    assert track["delta"]["ranking_accuracy"] == pytest.approx(-0.0288, abs=5e-4)
    assert report["overall"]["delta"]["ranking_accuracy"] < 0
    # Semantic routing merges same-dimension preferences, so fewer reach the prompt.
    assert track["treatment"]["mean_preference_facets_per_question"] < (
        track["baseline"]["mean_preference_facets_per_question"]
    )


def test_phaseb2_regresses_the_same_subset() -> None:
    report = compare_retrieval_ranking.compare(BASELINE, PHASEB2)

    track = report["by_question_type"]["track_full_preference_evolution"]
    assert track["delta"]["ranking_accuracy"] == pytest.approx(-0.0288, abs=5e-4)


def test_missing_service_fingerprint_is_reported_not_silently_passed() -> None:
    report = compare_retrieval_ranking.compare(BASELINE, TREATMENT)

    # The formal double run predates the runner recording flag state.
    assert report["service_fingerprint_available"] is False
    assert len(report["fingerprint_warnings"]) == 2


def test_fingerprint_available_when_both_artifacts_record_it(tmp_path) -> None:
    fingerprint = {"semantic_routing_enabled": True}
    left = _write(tmp_path / "l.json", _minimal_payload(service_fingerprint=fingerprint))
    right = _write(tmp_path / "r.json", _minimal_payload(service_fingerprint=fingerprint))

    report = compare_retrieval_ranking.compare(left, right)

    assert report["service_fingerprint_available"] is True
    assert report["fingerprint_warnings"] == []


def test_mismatched_dataset_hashes_are_refused(tmp_path) -> None:
    left = _write(tmp_path / "l.json", _minimal_payload())
    right = _write(tmp_path / "r.json", _minimal_payload(questions_sha256="c" * 64))

    with pytest.raises(ValueError, match="different datasets"):
        compare_retrieval_ranking.compare(left, right)


def test_missing_dataset_hash_is_refused(tmp_path) -> None:
    left = _write(tmp_path / "l.json", _minimal_payload())
    right = _write(tmp_path / "r.json", _minimal_payload(contexts_sha256=None))

    with pytest.raises(ValueError, match="contexts_sha256"):
        compare_retrieval_ranking.compare(left, right)


def test_divergent_question_sets_are_refused(tmp_path) -> None:
    payload = _minimal_payload()
    extra = _minimal_payload()
    extra["records"] = [*extra["records"], {**extra["records"][0], "question_id": "q2"}]
    left = _write(tmp_path / "l.json", payload)
    right = _write(tmp_path / "r.json", extra)

    with pytest.raises(ValueError, match="different question sets"):
        compare_retrieval_ranking.compare(left, right)


def test_allow_subset_compares_the_intersection_of_a_narrow_band_run(tmp_path) -> None:
    """A --question-types run covers fewer questions but the same pinned dataset."""
    full = _minimal_payload()
    full["records"] = [
        *full["records"],
        {**full["records"][0], "question_id": "q2", "question_type": "suggest_new_ideas"},
    ]
    narrow = _minimal_payload()
    left = _write(tmp_path / "l.json", full)
    right = _write(tmp_path / "r.json", narrow)

    report = compare_retrieval_ranking.compare(left, right, allow_subset=True)

    assert report["dataset"]["question_count"] == 1
    assert report["dataset"]["compared_subset"] is True
    assert report["dataset"]["baseline_record_count"] == 2
    assert set(report["by_question_type"]) == {"track_full_preference_evolution"}


def test_allow_subset_still_refuses_a_different_dataset(tmp_path) -> None:
    left = _write(tmp_path / "l.json", _minimal_payload())
    right = _write(tmp_path / "r.json", _minimal_payload(questions_sha256="c" * 64))

    with pytest.raises(ValueError, match="different datasets"):
        compare_retrieval_ranking.compare(left, right, allow_subset=True)


def test_full_comparison_is_not_flagged_as_a_subset() -> None:
    report = compare_retrieval_ranking.compare(BASELINE, TREATMENT)

    assert report["dataset"]["compared_subset"] is False
    assert report["dataset"]["question_count"] == 589


def test_records_missing_ranking_fields_are_refused(tmp_path) -> None:
    payload = _minimal_payload()
    del payload["records"][0]["ranking_reciprocal_rank"]
    path = _write(tmp_path / "l.json", payload)

    with pytest.raises(ValueError, match="ranking_reciprocal_rank"):
        compare_retrieval_ranking.compare(path, path)


def test_identical_retrieval_is_detected_as_the_noise_floor(tmp_path) -> None:
    # Same context, different run IDs in the user_id/revision attributes.
    left = _write(tmp_path / "l.json", _minimal_payload(run_id="left"))
    right_payload = _minimal_payload(run_id="right")
    right_payload["records"][0]["rendered_context"] = right_payload["records"][0][
        "rendered_context"
    ].replace('user_id="u" revision="1"', 'user_id="other" revision="9"')
    right = _write(tmp_path / "r.json", right_payload)

    report = compare_retrieval_ranking.compare(left, right)

    assert report["retrieval_identical_questions"] == 1
    assert report["retrieval_changed_questions"] == 0
