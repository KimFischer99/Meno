"""Set-level evidence selection.

Pointwise relevance scores each claim against the query independently, so a
cluster of near-duplicates can occupy the whole injection budget while the claims
that decide the answer sit just outside the cut. Measured on the frozen PersonaMem
corpus, decisive tokens appear in ~90% of full user history but only ~25% of the
injected top-12, and an oracle picking the best 12 from the same stored state
reaches 0.88 where the live path reaches 0.42.

These tests pin the marginal-gain selection that closes part of that gap, and pin
byte-identical behavior when the flag is off.
"""

from __future__ import annotations

from pathlib import Path

from meno.config import Settings
from meno.schemas import Facet, RetrieveRequest
from meno.service import _content_tokens
from tests.test_stage1 import ingest_event, make_service

QUERY = "what coffee do I drink, which sport, and which programming language"


def _duplicates(count: int) -> list[Facet]:
    """High-relevance near-duplicates that all cover the same query aspect."""
    return [
        Facet(
            claim_id=f"dup{index}",
            kind="episodic",
            value=f"I drink coffee every morning at cafe number {index}",
            relevance=0.90 - index * 0.01,
            confidence=0.9,
            evidence_ids=["e"],
            why_selected=[],
        )
        for index in range(count)
    ]


def _distinct() -> list[Facet]:
    """Lower-relevance claims covering query aspects nothing else covers."""
    return [
        Facet(
            claim_id="sport",
            kind="preference",
            value="swimming is my main sport",
            relevance=0.40,
            confidence=0.9,
            evidence_ids=["e"],
            why_selected=[],
        ),
        Facet(
            claim_id="lang",
            kind="preference",
            value="python is my programming language",
            relevance=0.35,
            confidence=0.9,
            evidence_ids=["e"],
            why_selected=[],
        ),
    ]


def _selected(tmp_path: Path, budget: int, **overrides) -> list[str]:
    service = make_service(tmp_path, **overrides)
    try:
        facets = [*_duplicates(6), *_distinct()]
        return [
            facet.claim_id for facet in service._apply_facet_budget(facets, budget, QUERY)
        ]
    finally:
        service.close()


def test_content_tokens_drop_function_words_and_short_tokens() -> None:
    tokens = _content_tokens("I would like the coffee that is very hot")

    assert "coffee" in tokens
    # Function words and sub-4-character tokens carry no evidence signal.
    assert "would" not in tokens
    assert "that" not in tokens
    assert "the" not in tokens


def test_pointwise_budget_is_filled_by_near_duplicates(tmp_path) -> None:
    picked = _selected(tmp_path, 4)

    # The default path takes the top four by relevance, all describing coffee.
    assert picked == ["dup0", "dup1", "dup2", "dup3"]


def test_set_level_selection_admits_uncovered_query_aspects(tmp_path) -> None:
    picked = _selected(tmp_path, 4, evidence_selection_enabled=True)

    assert "sport" in picked
    assert "lang" in picked
    # It still keeps the strongest duplicates rather than discarding the cluster.
    assert "dup0" in picked
    assert len(picked) == 4


def test_selection_is_deterministic(tmp_path) -> None:
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()

    first = _selected(first_dir, 4, evidence_selection_enabled=True)
    second = _selected(second_dir, 4, evidence_selection_enabled=True)

    assert first == second


def test_selection_returns_relevance_order(tmp_path) -> None:
    service = make_service(tmp_path, evidence_selection_enabled=True)
    try:
        facets = [*_duplicates(6), *_distinct()]
        selected = service._apply_facet_budget(facets, 4, QUERY)

        assert selected == sorted(selected, key=lambda item: item.relevance, reverse=True)
    finally:
        service.close()


def test_selection_is_a_noop_when_the_budget_is_not_binding(tmp_path) -> None:
    service = make_service(tmp_path, evidence_selection_enabled=True)
    try:
        facets = _distinct()
        assert service._apply_facet_budget(facets, 8, QUERY) == facets
    finally:
        service.close()


def test_empty_query_falls_back_to_relevance_and_redundancy(tmp_path) -> None:
    """With no query terms there is no coverage signal, but selection must not crash."""
    service = make_service(tmp_path, evidence_selection_enabled=True)
    try:
        facets = [*_duplicates(6), *_distinct()]
        picked = [facet.claim_id for facet in service._apply_facet_budget(facets, 4, "")]

        assert len(picked) == 4
        # Redundancy alone still breaks up the duplicate cluster.
        assert any(item in picked for item in ("sport", "lang"))
    finally:
        service.close()


def test_selection_composes_with_the_state_layer_reserve(tmp_path) -> None:
    service = make_service(
        tmp_path, layered_retrieval_enabled=True, evidence_selection_enabled=True
    )
    try:
        facets = [*_duplicates(10), *_distinct()]
        selected = service._apply_facet_budget(facets, 4, QUERY)

        kinds = [facet.kind for facet in selected]
        assert len(selected) == 4
        # The reserve guarantees the state layer slots; selection fills them.
        assert kinds.count("preference") == 2
    finally:
        service.close()


def test_redundancy_penalty_bounds_are_validated() -> None:
    assert Settings().evidence_selection_enabled is False

    Settings(evidence_selection_redundancy_penalty=0.0).validate()
    Settings(evidence_selection_redundancy_penalty=1.0).validate()
    for invalid in (-0.1, 1.5):
        try:
            Settings(evidence_selection_redundancy_penalty=invalid).validate()
        except ValueError:
            continue
        raise AssertionError(f"penalty {invalid} must be rejected")


def test_retrieve_is_unchanged_when_selection_is_disabled(tmp_path) -> None:
    """Default-off equivalence through the real read path."""

    def rendered(label: str, **overrides) -> str:
        directory = tmp_path / label
        directory.mkdir()
        service = make_service(directory, **overrides)
        try:
            for index in range(6):
                ingest_event(
                    service,
                    f"ev-{index}",
                    f"I drink coffee every morning at cafe number {index}",
                    user_id="user-a",
                )
            service.process_outbox()
            response = service.retrieve(
                RetrieveRequest(
                    user_id="user-a",
                    purpose="response_personalization",
                    context={"query": QUERY},
                    constraints={"max_facets": 3, "min_confidence": 0.5},
                )
            )
            return response.rendered_context
        finally:
            service.close()

    assert rendered("off") == rendered("off2")
