from __future__ import annotations

import argparse

from benchmarks import run_layered_retrieval_diagnostic as diagnostic
from meno.service import STATE_LAYER_KINDS


def _args(**overrides) -> argparse.Namespace:
    values = {
        "questions": 4,
        "episodic_per_question": 14,
        "max_facets": 12,
        "max_rendered_tokens": 800,
        "output": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_preference_utterances_cover_distinct_extractor_slots() -> None:
    from meno.extractor import preference_slot

    slots = [preference_slot(value) for _slot, value in diagnostic.PREFERENCE_UTTERANCES]

    # Each utterance must land on its declared slot, otherwise they supersede each
    # other and the corpus cannot hold several concurrent preferences.
    assert slots == [slot for slot, _value in diagnostic.PREFERENCE_UTTERANCES]
    assert len(set(slots)) == len(slots)


def test_episodic_templates_match_the_measured_production_length_band() -> None:
    lengths = [len(template) for template in diagnostic.EPISODIC_TEMPLATES]

    # Production episodic values averaged 263.3 chars and truncate at 280.
    assert all(200 <= length <= 280 for length in lengths), lengths


def test_baseline_render_budget_drops_state_facets() -> None:
    args = _args()
    cases = diagnostic._build_cases(args)

    baseline = diagnostic._run(cases, args)

    # Reproduces the production shape: 12 facets chosen, not all rendered, and the
    # injected context dominated by long episodic values.
    assert baseline["facets_selected"] == args.questions * args.max_facets
    assert baseline["facets_dropped_by_render"] > 0
    assert baseline["state_layer_share_rendered"] < 0.2


def test_layered_retrieval_keeps_state_facets_and_stops_dropping_the_tail() -> None:
    args = _args()
    cases = diagnostic._build_cases(args)

    baseline = diagnostic._run(cases, args)
    layered = diagnostic._run(cases, args, layered_retrieval_enabled=True)

    assert layered["facets_selected"] == baseline["facets_selected"]
    assert layered["facets_dropped_by_render"] == 0
    assert layered["state_layer_share_rendered"] > baseline["state_layer_share_rendered"]
    assert layered["mean_state_facets_rendered_per_question"] > (
        baseline["mean_state_facets_rendered_per_question"]
    )
    assert set(layered["rendered_kind_totals"]) & STATE_LAYER_KINDS
