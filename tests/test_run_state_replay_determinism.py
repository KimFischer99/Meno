from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from benchmarks import run_state_replay_determinism as lane
from meno.config import Settings

FIXTURE = Path("benchmarks/fixtures/ucm-oracle-v1.json")
BASELINE = Path("benchmarks/fixtures/state-replay-baseline-v1.json")


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings(
        database_url="sqlite:///:memory:",
        vector_mode="memory",
        embedding_dimension=256,
        worker_poll_seconds=0.01,
        context_activation_enabled=True,
        user_token_materialization_enabled=True,
        preference_distribution_enabled=True,
        clarification_opportunities_enabled=True,
    )


@pytest.fixture(scope="module")
def first_replay(payload: dict, settings: Settings) -> dict:
    return lane.replay(payload, settings, embedding_dimension=256)


def test_lane_covers_every_spec_1393_dimension(payload: dict, first_replay: dict) -> None:
    """SPEC :1393 names eight dimensions; all eight must be accounted for."""
    assert lane.DIMENSIONS == (
        "claim_set",
        "claim_statuses",
        "preference_distributions",
        "community_memberships",
        "token_revision",
        "retrieval_results",
        "audit_lineage",
        "rendered_context",
    )
    assert set(first_replay["state"]) == set(lane.DIMENSIONS)
    assert set(lane.EXCLUDED_FIELDS) == set(lane.DIMENSIONS)


def test_unimplemented_dimension_is_uncovered_not_empty(first_replay: dict) -> None:
    """An unimplemented dimension must not digest as an empty set.

    An empty list hashes to a stable value and would compare equal forever,
    reading as a passing check for something that does not exist.
    """
    assert lane.UNCOVERED_DIMENSIONS == {
        "community_memberships": lane.UNCOVERED_DIMENSIONS["community_memberships"]
    }
    assert first_replay["digests"]["community_memberships"] is None
    assert first_replay["state"]["community_memberships"] is None


def test_replay_is_deterministic_across_independent_stores(
    payload: dict, settings: Settings, first_replay: dict
) -> None:
    second = lane.replay(payload, settings, embedding_dimension=256)

    assert lane._changed_dimensions(first_replay["digests"], second["digests"]) == []


def test_state_captures_all_four_cohorts(payload: dict, first_replay: dict) -> None:
    users = {item["user_id"] for item in first_replay["state"]["claim_set"]}

    assert users == {case["user_id"] for case in payload["cases"]}
    assert len(first_replay["state"]["rendered_context"]) == sum(
        len(case["queries"]) for case in payload["cases"]
    )


def test_supersede_topology_is_captured(first_replay: dict) -> None:
    """The contradictory cohort must show a real supersede link in the digest input."""
    superseded = [
        item
        for item in first_replay["state"]["claim_statuses"]
        if item["status"] == "superseded"
    ]

    assert superseded
    assert any(item["superseded_by_id"] for item in superseded)


def test_retrieval_audits_reach_the_lineage_dimension(first_replay: dict) -> None:
    """Retrieve buffers its audits, so the lane must flush before reading them."""
    names = {item["event_name"] for item in first_replay["state"]["audit_lineage"]}

    assert "meno.retrieve.facet_selected" in names
    assert "meno.ingest.accepted" in names


def test_digest_ignores_float_representation() -> None:
    assert lane._digest({"score": 0.1 + 0.2}) == lane._digest({"score": 0.3})


def test_truncated_event_log_changes_the_user_model(
    payload: dict, settings: Settings, first_replay: dict
) -> None:
    truncated = lane.replay(payload, settings, embedding_dimension=256, event_limit_per_case=1)
    changed = lane._changed_dimensions(first_replay["digests"], truncated["digests"])

    assert "claim_set" in changed
    assert "rendered_context" in changed


def test_swapping_the_extractor_is_visible(
    payload: dict, settings: Settings, first_replay: dict
) -> None:
    """SPEC :1393's stated purpose for this lane."""
    swapped = lane.replay(
        payload,
        dataclasses.replace(settings, extractor_version="meno-extractor-probe-0.0.0"),
        embedding_dimension=256,
    )

    assert "claim_set" in lane._changed_dimensions(first_replay["digests"], swapped["digests"])


def test_swapping_the_embedding_is_visible(
    payload: dict, settings: Settings, first_replay: dict
) -> None:
    swapped = lane.replay(payload, settings, embedding_dimension=192)

    assert "retrieval_results" in lane._changed_dimensions(
        first_replay["digests"], swapped["digests"]
    )


def test_self_check_fails_when_a_digest_stops_responding(
    payload: dict, settings: Settings, first_replay: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The anti-vacuity guard must itself be able to fail.

    Freezing every digest simulates over-normalization: the probes perturb the
    inputs but nothing moves, which has to be reported as a failure rather than
    as three passing probes.
    """
    monkeypatch.setattr(
        lane, "replay", lambda *args, **kwargs: {"digests": first_replay["digests"]}
    )

    results = lane.run_self_check(payload, settings, first_replay["digests"])

    assert results
    assert all(not item["passed"] for item in results)
    assert all(item["unresponsive_dimensions"] for item in results)


def test_frozen_baseline_matches_current_state() -> None:
    report = lane.run_benchmark(FIXTURE, BASELINE, self_check=False)

    assert report["baseline"]["mode"] == "compared"
    assert report["baseline"]["fixture_sha256_matches"] is True
    assert report["baseline"]["drift"] == []
    assert report["replay_determinism"]["passed"] is True
    assert report["audit_chain"]["passed"] is True
    assert report["answer_model_calls"] == 0


def test_skipping_the_self_check_cannot_pass_the_lane() -> None:
    """--no-self-check is a diagnostic mode, not a cheaper way to pass."""
    report = lane.run_benchmark(FIXTURE, BASELINE, self_check=False)

    assert report["self_check"]["ran"] is False
    assert report["passed"] is False


def test_full_lane_passes_with_the_self_check() -> None:
    report = lane.run_benchmark(FIXTURE, BASELINE)

    assert report["self_check"]["passed"] is True
    assert report["passed"] is True
    # Seven of eight dimensions is not SPEC :1393 compliance, and the report
    # must keep saying so until community_memberships exists.
    assert report["dimensions_covered"] == 7
    assert report["dimensions_total"] == 8
    assert report["spec_1393_fully_covered"] is False


def test_baseline_drift_fails_the_lane(tmp_path: Path) -> None:
    frozen = json.loads(BASELINE.read_text(encoding="utf-8"))
    frozen["digests"]["rendered_context"] = "sha256:" + "0" * 64
    tampered = tmp_path / "baseline.json"
    tampered.write_text(json.dumps(frozen), encoding="utf-8")

    report = lane.run_benchmark(FIXTURE, tampered, self_check=False)

    assert report["baseline"]["drift"] == ["rendered_context"]
    assert report["passed"] is False


def test_baseline_bound_to_a_fixture_revision(tmp_path: Path) -> None:
    """Digests are only meaningful for the event log that produced them."""
    frozen = json.loads(BASELINE.read_text(encoding="utf-8"))
    frozen["fixture_sha256"] = "0" * 64
    stale = tmp_path / "baseline.json"
    stale.write_text(json.dumps(frozen), encoding="utf-8")

    report = lane.run_benchmark(FIXTURE, stale, self_check=False)

    assert report["baseline"]["fixture_sha256_matches"] is False
    assert report["passed"] is False


def test_unknown_baseline_schema_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "baseline.json"
    bad.write_text(json.dumps({"schema_version": "nope", "digests": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="baseline schema"):
        lane.run_benchmark(FIXTURE, bad, self_check=False)
