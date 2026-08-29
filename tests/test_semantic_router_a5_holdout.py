from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import benchmarks.run_semantic_router_a5_holdout as holdout_runner
from benchmarks.run_semantic_router_a5_holdout import (
    DEFAULT_CONFIG,
    DEFAULT_MANIFEST,
    load_a5_evaluation_manifest,
    load_frozen_config,
)
from tests.fakes import TestEmbedder

ROOT = Path(__file__).parent.parent
HOLDOUT_FIXTURE = ROOT / "benchmarks" / "fixtures" / "semantic-normalization-v6-a5-fresh-holdout.json"


def test_frozen_config_pins_the_selected_a5_policy() -> None:
    _raw, config = load_frozen_config(DEFAULT_CONFIG)

    strategy = config["strategy"]
    assert strategy["variant"] == "top2_mean"
    assert strategy["score_threshold"] == 0.475
    assert strategy["margin_threshold"] == 0.005
    assert strategy["threshold_search_allowed_after_freeze"] is False
    assert config["fresh_holdout_gate"]["holdout_runs_allowed"] == 1
    assert config["fresh_holdout_gate"]["minimum_recall"] == 0.75


def test_frozen_config_rejects_tampering(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_CONFIG.read_bytes())

    reopened_search = json.loads(json.dumps(payload))
    reopened_search["strategy"]["threshold_search_allowed_after_freeze"] = True
    path = tmp_path / "reopened.json"
    path.write_text(json.dumps(reopened_search))
    with pytest.raises(ValueError, match="threshold search"):
        load_frozen_config(path)

    two_runs = json.loads(json.dumps(payload))
    two_runs["fresh_holdout_gate"]["holdout_runs_allowed"] = 2
    path = tmp_path / "two-runs.json"
    path.write_text(json.dumps(two_runs))
    with pytest.raises(ValueError, match="single-run"):
        load_frozen_config(path)


def test_a5_evaluation_manifest_requires_exact_out_of_band_sha(tmp_path: Path) -> None:
    raw_bytes = DEFAULT_MANIFEST.read_bytes()
    good_sha = hashlib.sha256(raw_bytes).hexdigest()

    loaded_raw, payload = load_a5_evaluation_manifest(
        DEFAULT_MANIFEST, expected_sha256=good_sha
    )
    assert hashlib.sha256(loaded_raw).hexdigest() == good_sha
    assert payload["manifest_version"] == "semantic-router-a5-evaluation-manifest-v1"

    with pytest.raises(ValueError, match="out-of-band"):
        load_a5_evaluation_manifest(DEFAULT_MANIFEST, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_a5_evaluation_manifest(DEFAULT_MANIFEST, expected_sha256="nothex")


def test_fresh_holdout_fixture_satisfies_sealed_protocol() -> None:
    raw_bytes = HOLDOUT_FIXTURE.read_bytes()
    payload = json.loads(raw_bytes)
    assert payload["fixture_version"] == "semantic-normalization-v6-a5-fresh-holdout"
    assert payload["mode"] == "fresh-holdout-provider-scored"

    cases_payload = payload["cases"]
    scored = [
        case
        for case in cases_payload
        if (
            case["candidate"]["deterministic_slot"] is None
            and not case["candidate"]["sensitive"]
            and not case["candidate"]["injection_detected"]
            and case["expected_dimension"] is not None
        )
    ]
    by_dimension: dict[str, int] = {}
    for case in scored:
        by_dimension[case["expected_dimension"]] = (
            by_dimension.get(case["expected_dimension"], 0) + 1
        )
    assert len(cases_payload) >= 42
    assert len(scored) >= 35
    assert min(by_dimension.values()) >= 4
    assert set(by_dimension) == {
        "answer_style",
        "beverage",
        "color",
        "food",
        "music",
        "programming_language",
        "sport",
    }
    # The manifest must pin this exact fixture content.
    manifest = json.loads(DEFAULT_MANIFEST.read_bytes())
    assert manifest["files"]["holdout_fixture_sha256"] == hashlib.sha256(raw_bytes).hexdigest()


def test_holdout_manifest_rejects_reuse_after_phaseb_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        holdout_runner, "make_embedder", lambda _settings: TestEmbedder(dimension=64)
    )
    output = tmp_path / "a5-holdout.json"

    def run() -> int:
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_semantic_router_a5_holdout.py",
                "--output",
                str(output),
                "--expected-manifest-sha256",
                hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest(),
            ],
        )
        try:
            holdout_runner.main()
            return 0
        except SystemExit as excinfo:
            return excinfo.code  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="evaluation manifest file mismatch"):
        run()
    assert not output.exists()
