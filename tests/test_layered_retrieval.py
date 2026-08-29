"""Layered retrieval: state-layer claims must survive the injection budget.

Meno_SPEC.md requires the canonical representation to distinguish abstraction
levels rather than treating every memory as the same chunk. These tests pin the
three places where a kind-blind read path let long verbatim episodic values crowd
short canonical state claims out of the injected context, and pin byte-identical
behavior when the flag is off.
"""

from __future__ import annotations

from meno.config import Settings
from meno.schemas import Facet, RetrieveRequest
from meno.service import STATE_LAYER_KINDS, _episodic_length_discount
from tests.test_stage1 import ingest_event, make_service

# Long enough to exceed the render char budget on its own, which is what made the
# original `_render` discard every facet behind it. Kept just under the extractor's
# EPISODIC_MAX_CHARS so the leading text is not truncated away.
LONG_EPISODIC = (
    "I spent the whole weekend at a hands-on cooking class downtown where we worked "
    "through regional braises, and the instructor kept circling back to how much the "
    "resting time matters, which surprised me because I had rushed that step at home"
)


def _episodic_variant(index: int) -> str:
    """A distinct long episodic utterance.

    The extractor truncates episodic values at EPISODIC_MAX_CHARS and keys claims on
    the normalized value, so a shared prefix would collapse every event onto one
    claim. Varying the opening keeps them distinct at production-like length.
    """
    return f"Session {index}: on day {index} of the term, {LONG_EPISODIC}"


def _facet(kind: str, relevance: float, value: str = "v") -> Facet:
    return Facet(
        claim_id=f"claim-{kind}-{relevance}",
        kind=kind,
        value=value,
        relevance=relevance,
        confidence=0.9,
        evidence_ids=["event-1"],
        why_selected=[],
    )


def test_state_layer_kinds_cover_the_spec_stable_layers() -> None:
    assert "preference" in STATE_LAYER_KINDS
    assert "trait" in STATE_LAYER_KINDS
    # Transient conversational layers must not be reserved.
    assert "episodic" not in STATE_LAYER_KINDS
    assert "state" not in STATE_LAYER_KINDS


def test_episodic_length_discount_is_neutral_below_canonical_band() -> None:
    assert _episodic_length_discount("concise answers") == 1.0
    # Decreasing past the canonical band, then floored so episodic claims stay
    # retrievable rather than being effectively banned.
    assert _episodic_length_discount("x" * 200) < 1.0
    assert _episodic_length_discount("x" * 300) < _episodic_length_discount("x" * 200)
    assert _episodic_length_discount("x" * 2000) == 0.6


def test_facet_budget_reserves_slots_for_state_layers(tmp_path) -> None:
    service = make_service(tmp_path, layered_retrieval_enabled=True)
    try:
        # Episodic facets outrank every preference facet on raw relevance.
        facets = [_facet("episodic", 0.90 - index * 0.01) for index in range(10)]
        facets += [_facet("preference", 0.40 - index * 0.01) for index in range(4)]

        selected = service._apply_facet_budget(facets, 8)

        assert len(selected) == 8
        kinds = [facet.kind for facet in selected]
        assert kinds.count("preference") == 4, kinds
        # Still ordered by relevance for the render step.
        assert selected == sorted(selected, key=lambda item: item.relevance, reverse=True)
    finally:
        service.close()


def test_facet_budget_returns_unused_reserve_to_the_pool(tmp_path) -> None:
    service = make_service(tmp_path, layered_retrieval_enabled=True)
    try:
        # Only one state-layer facet exists; the rest of the reserve must not be wasted.
        facets = [_facet("episodic", 0.90 - index * 0.01) for index in range(10)]
        facets += [_facet("preference", 0.20)]

        selected = service._apply_facet_budget(facets, 8)

        assert len(selected) == 8
        assert [facet.kind for facet in selected].count("preference") == 1
    finally:
        service.close()


def test_facet_budget_is_a_flat_cut_when_layering_is_disabled(tmp_path) -> None:
    service = make_service(tmp_path)
    try:
        facets = [_facet("episodic", 0.90 - index * 0.01) for index in range(10)]
        facets += [_facet("preference", 0.40)]

        selected = service._apply_facet_budget(facets, 8)

        assert selected == facets[:8]
        assert all(facet.kind == "episodic" for facet in selected)
    finally:
        service.close()


def test_render_skips_one_oversized_value_instead_of_dropping_the_tail(tmp_path) -> None:
    service = make_service(tmp_path, layered_retrieval_enabled=True)
    try:
        facets = [
            _facet("episodic", 0.9, LONG_EPISODIC),
            _facet("preference", 0.5, "prefers concise answers"),
            _facet("trait", 0.4, "detail oriented"),
        ]

        rendered = service._render("user-a", 1, facets, max_tokens=64)

        # The oversized episodic value is skipped, and both short state claims survive.
        assert "prefers concise answers" in rendered
        assert "detail oriented" in rendered
    finally:
        service.close()


def test_render_default_behavior_truncates_and_stops(tmp_path) -> None:
    service = make_service(tmp_path)
    try:
        facets = [
            _facet("episodic", 0.9, LONG_EPISODIC),
            _facet("preference", 0.5, "prefers concise answers"),
        ]

        rendered = service._render("user-a", 1, facets, max_tokens=64)

        # Unchanged legacy behavior: truncate the first value, discard the rest.
        assert "…" in rendered
        assert "prefers concise answers" not in rendered
    finally:
        service.close()


def test_layered_retrieval_admits_more_state_context_end_to_end(tmp_path) -> None:
    """The whole point: state claims reach the prompt where episodic values crowded them out.

    The corpus mirrors the distribution measured in the production PersonaMem cache:
    long verbatim episodic values (~263 chars) vastly outnumbering short canonical
    preference values (~88 chars).
    """
    preferences = [
        "I prefer concise answers that lead with the conclusion",
        "I prefer dark roast coffee in the morning",
        "I prefer Python for data analysis work",
        "I prefer jazz when I am concentrating",
    ]

    def injected(label: str, **overrides) -> tuple[int, list[str]]:
        database_dir = tmp_path / label
        database_dir.mkdir()
        service = make_service(database_dir, **overrides)
        try:
            for index, preference in enumerate(preferences):
                ingest_event(service, f"pref-{index}", preference, user_id="user-a")
            for index in range(14):
                ingest_event(
                    service,
                    f"ep-{index}",
                    _episodic_variant(index),
                    user_id="user-a",
                )
            service.process_outbox()
            response = service.retrieve(
                RetrieveRequest(
                    user_id="user-a",
                    purpose="response_personalization",
                    context={"query": "How should you format answers and what do I like?"},
                    constraints={
                        "max_facets": 12,
                        "max_rendered_tokens": 800,
                        "min_confidence": 0.5,
                    },
                )
            )
            kinds = [
                line.removeprefix("- [").split("]")[0].split(" ")[0].split(":")[0]
                for line in response.rendered_context.splitlines()
                if line.startswith("- [")
            ]
            return len(response.facets), kinds
        finally:
            service.close()

    baseline_selected, baseline_kinds = injected("baseline")
    layered_selected, layered_kinds = injected("layered", layered_retrieval_enabled=True)

    # Same number of facets chosen, but the layered path actually renders them all.
    assert layered_selected == baseline_selected
    assert len(baseline_kinds) < baseline_selected, "expected the render budget to drop facets"
    assert len(layered_kinds) == layered_selected

    baseline_state = sum(kind in STATE_LAYER_KINDS for kind in baseline_kinds)
    layered_state = sum(kind in STATE_LAYER_KINDS for kind in layered_kinds)
    assert layered_state > baseline_state, (baseline_kinds, layered_kinds)


def test_layered_settings_default_off_and_validate_reserve() -> None:
    settings = Settings()
    assert settings.layered_retrieval_enabled is False

    Settings(layered_state_facet_reserve=0.0).validate()
    Settings(layered_state_facet_reserve=1.0).validate()
    for invalid in (-0.1, 1.5):
        try:
            Settings(layered_state_facet_reserve=invalid).validate()
        except ValueError:
            continue
        raise AssertionError(f"reserve {invalid} must be rejected")
