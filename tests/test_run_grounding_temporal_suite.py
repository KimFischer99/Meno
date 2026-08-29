from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import run_grounding_temporal_suite as lane

FIXTURE = Path("benchmarks/fixtures/ucm-oracle-v1.json")
CONTEXTS = Path("artifacts/benchmarks/raw/shared_contexts_32k.jsonl")
QUESTIONS = Path("artifacts/benchmarks/raw/questions_32k.csv")

corpus_available = pytest.mark.skipif(
    not (CONTEXTS.is_file() and QUESTIONS.is_file()),
    reason="PersonaMem corpus is gitignored; run the corpus lane where it is present",
)


def test_thresholds_match_the_spec_table() -> None:
    """SPEC :1340-1372 values, including each metric's comparison direction."""
    assert lane.THRESHOLDS["evidence_attribution_coverage"] == {
        "bound": 0.99,
        "direction": "min",
        "charter_item": 8,
    }
    assert lane.THRESHOLDS["unsupported_injected_claim_rate"] == {
        "bound": 0.01,
        "direction": "max",
        "charter_item": 9,
    }
    assert lane.THRESHOLDS["stale_active_rate"] == {
        "bound": 0.02,
        "direction": "max",
        "charter_item": 10,
    }


def test_truncated_episodic_tail_is_not_counted_unsupported() -> None:
    """Episodic values are cut mid-word, so the fragment matches no event.

    Without this rule the measured unsupported rate was 157/285 -- all artifacts.
    """
    assert "experie" not in lane._asserted_tokens("I had a great experie…")
    assert "experience" in lane._asserted_tokens("I had a great experience")
    # An untruncated value keeps its last token.
    assert "tea" in lane._asserted_tokens("green tea")


def test_assistant_prefix_is_not_treated_as_asserted_content() -> None:
    tokens = lane._asserted_tokens("Assistant previously responded: green tea")

    assert "assistant" not in tokens
    assert "tea" in tokens


def test_horizon_uses_the_shortest_marker() -> None:
    assert lane._horizon("I am in Tokyo today") == ("today", 1)
    assert lane._horizon("this week and this month") == ("this week", 7)
    assert lane._horizon("我本周在东京") == ("this week (zh)", 7)
    assert lane._horizon("I like green tea") is None


def test_metric_refuses_thresholds_finer_than_the_sample() -> None:
    """A "< 1%" claim from 5 observations is not evidence."""
    coarse = lane._metric("unsupported_injected_claim_rate", 0, 5)

    assert coarse["rate"] == 0.0
    assert coarse["meets_threshold"] is True
    assert coarse["resolvable"] is False
    # Meeting the bound is not enough when the bound is unresolvable.
    assert coarse["passed"] is False

    fine = lane._metric("unsupported_injected_claim_rate", 0, 500)
    assert fine["resolvable"] is True
    assert fine["passed"] is True


def test_metric_direction_is_per_metric() -> None:
    """Coverage is a floor; the other two are ceilings."""
    coverage = lane._metric("evidence_attribution_coverage", 100, 100)
    assert coverage["passed"] is True
    assert lane._metric("evidence_attribution_coverage", 50, 100)["meets_threshold"] is False

    assert lane._metric("stale_active_rate", 1, 1000)["meets_threshold"] is True
    assert lane._metric("stale_active_rate", 100, 1000)["meets_threshold"] is False


def test_zero_denominator_never_passes() -> None:
    """No injected facets means no measurement, not a clean sheet."""
    empty = lane._metric("evidence_attribution_coverage", 0, 0)

    assert empty["denominator"] == 0
    assert empty["resolvable"] is False
    assert empty["passed"] is False


def test_stale_counts_supersede_shadowing() -> None:
    observation = lane.FacetObservation(
        user_id="u",
        query_id="q",
        claim_id="c",
        kind="preference",
        value="terse answers",
        evidence_ids=["e"],
        has_evidence=True,
        evidence_resolves=True,
        unsupported_tokens=[],
        horizon_label=None,
        horizon_days=None,
        age_days=1.0,
        superseded_by_newer=True,
    )

    assert observation.stale is True
    assert observation.attribution_ok is True
    assert observation.unsupported is False


def test_expired_horizon_is_stale_but_fresh_one_is_not() -> None:
    def observe(age_days: float) -> lane.FacetObservation:
        return lane.FacetObservation(
            user_id="u",
            query_id="q",
            claim_id="c",
            kind="episodic",
            value="I am in Tokyo today",
            evidence_ids=["e"],
            has_evidence=True,
            evidence_resolves=True,
            unsupported_tokens=[],
            horizon_label="today",
            horizon_days=1,
            age_days=age_days,
            superseded_by_newer=False,
        )

    assert observe(60.0).stale is True
    assert observe(0.5).stale is False


def test_self_check_probes_detect_their_own_violations() -> None:
    """Each metric must catch a synthetic defect, or it is not a gate.

    #9 cannot fail on real corpus data (measured 0/449) because the deterministic
    extractor copies event text, and reflection -- the one writer of non-verbatim
    values -- produces nothing on single-session PersonaMem contexts. So both the
    extractor route (`fabricated_value`) and the reflection route
    (`fabricated_pattern`) have to be probed synthetically.
    """
    probes = lane.run_self_check()

    assert {item["probe"] for item in probes} == {
        "fabricated_value",
        "stripped_evidence",
        "expired_horizon",
        "fabricated_pattern",
    }
    for probe in probes:
        assert probe["injected_facets"] > 0, probe["probe"]
        assert probe["detected"] is True, probe["probe"]
        assert probe["passed"] is True, probe["probe"]


def test_fixture_lane_reports_unresolvable_thresholds() -> None:
    """The committed fixture is too small to evidence < 1%, and says so."""
    report = lane.run_benchmark(FIXTURE, self_check=False)

    assert report["injected_facets"] > 0
    assert report["metrics"]["evidence_attribution_coverage"]["rate"] == 1.0
    assert "unsupported_injected_claim_rate" in report["unresolvable_metrics"]
    assert report["passed"] is False


def test_skipping_the_self_check_cannot_pass_the_lane() -> None:
    report = lane.run_benchmark(FIXTURE, self_check=False)

    assert report["self_check"]["ran"] is False
    assert report["passed"] is False


def test_corpus_requires_questions() -> None:
    """Invented queries measure the probe, not the system."""
    with pytest.raises(ValueError, match="--questions"):
        lane.run_benchmark(FIXTURE, contexts_path=CONTEXTS, self_check=False)


@corpus_available
def test_corpus_lane_resolves_every_threshold() -> None:
    report = lane.run_benchmark(
        FIXTURE,
        contexts_path=CONTEXTS,
        questions_path=QUESTIONS,
        contexts_limit=4,
        self_check=False,
    )

    assert report["provenance"]["source"] == "corpus"
    assert report["provenance"]["questions_asked"] > 0
    # Real questions yield an order of magnitude more facets than the fixture.
    assert report["injected_facets"] > 100
    assert report["unresolvable_metrics"] == []
    for name in lane.THRESHOLDS:
        assert report["metrics"][name]["meets_threshold"] is True, name


@corpus_available
def test_corpus_lane_is_deterministic() -> None:
    """A fixed clock and a frozen question set must give a repeatable rate."""
    first = lane.run_benchmark(
        FIXTURE,
        contexts_path=CONTEXTS,
        questions_path=QUESTIONS,
        contexts_limit=2,
        self_check=False,
    )
    second = lane.run_benchmark(
        FIXTURE,
        contexts_path=CONTEXTS,
        questions_path=QUESTIONS,
        contexts_limit=2,
        self_check=False,
    )

    assert first["metrics"] == second["metrics"]


def test_report_records_the_structural_caveat() -> None:
    """#9 must not be read as evidence that inference is grounded."""
    report = lane.run_benchmark(FIXTURE, self_check=False)
    note = report["notes"]["unsupported_rate_is_structurally_low"]

    assert "by construction" in note
    assert json.dumps(report)  # report must stay serializable
