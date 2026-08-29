from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.generate_fresh_product_holdout import (
    CSV_FIELDS,
    FOCUS_QUESTION_TYPE,
    FOCUS_SLICE_QUOTAS,
    OFFICIAL_QUESTION_TYPES,
    QUOTAS,
    TOTAL_ITEMS,
    HoldoutError,
    generate_candidate,
    main,
    seal_bundle,
    validate_bundle,
)


def _generate(path: Path, seed: int = 20260823) -> None:
    generate_candidate(path, seed)


def _attestation(path: Path, bundle_sha256: str) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "attestation_id": "test-oracle-pass",
                "bundle_sha256": bundle_sha256,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_same_seed_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _generate(first)
    _generate(second)

    for name in ("questions.csv", "contexts.jsonl", "labels.jsonl", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_schema_quota_and_position_balance(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    _generate(output)
    manifest = validate_bundle(output)
    assert manifest["item_count"] == TOTAL_ITEMS
    assert manifest["quotas"] == QUOTAS
    assert manifest["leakage_status"] == "LEAKAGE_UNVERIFIED"

    import csv

    with (output / "questions.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == CSV_FIELDS
    assert len(rows) == TOTAL_ITEMS
    expected_types = {
        "generalizing_to_new_scenarios",
        "provide_preference_aligned_recommendations",
        "recall_user_shared_facts",
        "recalling_facts_mentioned_by_the_user",
        "recalling_the_reasons_behind_previous_updates",
        "suggest_new_ideas",
        "track_full_preference_evolution",
    }
    assert set(OFFICIAL_QUESTION_TYPES) == expected_types
    assert {row["question_type"] for row in rows} == expected_types

    label_records = [
        json.loads(line)
        for line in (output / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    focus_counts = {slice_name: 0 for slice_name in FOCUS_SLICE_QUOTAS}
    for label in label_records:
        if label["question_type"] == FOCUS_QUESTION_TYPE:
            focus_counts[label["focus_slice"]] += 1
    assert focus_counts == FOCUS_SLICE_QUOTAS


def test_tampering_fails_hash_validation(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    _generate(output)
    with (output / "questions.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(HoldoutError, match="hash mismatch"):
        validate_bundle(output)


def test_sealed_output_cannot_be_overwritten_or_resealed(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    _generate(output)
    attestation = tmp_path / "oracle.json"
    manifest = validate_bundle(output)
    _attestation(attestation, manifest["bundle_sha256"])
    sealed = seal_bundle(output, attestation)
    assert sealed["seal"]["status"] == "sealed"
    assert sealed["leakage_status"] == "LEAKAGE_ATTESTED"

    with pytest.raises(HoldoutError, match="sealed output"):
        generate_candidate(output, 20260823)
    with pytest.raises(HoldoutError, match="sealed output"):
        seal_bundle(output, attestation)


def test_seal_requires_external_oracle_attestation(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    _generate(output)
    with pytest.raises(HoldoutError, match="oracle-attestation"):
        seal_bundle(output, None)
    assert validate_bundle(output)["seal"]["status"] == "candidate"

    assert main(["seal", "--input", str(output)]) == 2
