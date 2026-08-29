"""Tests for deterministic reflection (`pattern` claim derivation).

Each test pins a rule whose violation would break an already-passing gate:
consent intersection (#1 #2 #3), token-level groundedness (#9), replay
determinism (#12), or the SPEC No-Go that an inference must never override an
explicit user statement.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

from meno.reflection import (
    PATTERN_CHANNEL,
    PATTERN_MIN_EVIDENCE,
    PATTERN_TEMPLATE_TOKENS,
    derive_patterns,
    pattern_derivation_basis,
)

BASE = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass
class FakeClaim:
    id: str
    user_id: str = "u1"
    kind: str = "preference"
    semantic_channel: str = "preference.explicit"
    semantic_key: str = "key-tea"
    value: str = "green tea"
    status: str = "active"
    stance: str | None = "positive"
    sensitive: bool = False
    source_type: str = "chat"
    routing_slot: str | None = "beverage"


@dataclass
class FakeEvent:
    id: str
    occurred_at: datetime
    consent_scope: list[str]


def _group(
    count: int = PATTERN_MIN_EVIDENCE,
    *,
    scopes: list[list[str]] | None = None,
    span_days: int = 30,
    claim_overrides: list[dict] | None = None,
):
    """A group of preference claims on one key, each with its own event."""
    claims = []
    events = {}
    evidence = {}
    for index in range(count):
        claim = FakeClaim(id=f"c{index}")
        if claim_overrides and index < len(claim_overrides):
            claim = replace(claim, **claim_overrides[index])
        claims.append(claim)
        event_id = f"e{index}"
        scope = scopes[index] if scopes else ["personalization", "task_planning"]
        # Spread events across the window so the span requirement is met.
        offset = timedelta(days=span_days * index / max(1, count - 1)) if count > 1 else timedelta()
        events[event_id] = FakeEvent(event_id, BASE + offset, scope)
        evidence[claim.id] = [event_id]
    return claims, events, evidence


def test_derives_a_pattern_from_repeated_preferences():
    claims, events, evidence = _group()
    result = derive_patterns(claims, events, evidence)
    assert len(result) == 1
    pattern = result[0]
    assert pattern.semantic_channel == PATTERN_CHANNEL
    assert pattern.evidence_count == PATTERN_MIN_EVIDENCE
    assert pattern.source_claim_ids == ("c0", "c1", "c2")
    assert pattern.stance == "positive"


def test_value_uses_only_evidence_words_plus_whitelisted_templates():
    """Charter #9 checks token-level support; anything outside the template
    whitelist must come from the evidence itself."""
    claims, events, evidence = _group()
    value = derive_patterns(claims, events, evidence)[0].value
    evidence_words = set()
    for claim in claims:
        evidence_words.update(claim.value.lower().split())
    for token in value.lower().split():
        assert token in evidence_words or token in PATTERN_TEMPLATE_TOKENS, token


def test_value_uses_the_shared_core_of_the_restatements():
    """The repeated part is the pattern; the varying tails are occasion detail.
    Picking one member would also make the value depend on arrival order."""
    claims, events, evidence = _group(
        claim_overrides=[
            {"value": "green tea in the morning"},
            {"value": "green tea when working"},
            {"value": "green tea over soda"},
        ]
    )
    assert derive_patterns(claims, events, evidence)[0].value == "repeatedly prefers green tea"


def test_value_is_stable_when_a_restatement_is_added():
    """A growing support set must not rewrite the summary, because the value feeds
    the semantic key -- an unstable value would churn the supersede chain."""
    three = derive_patterns(
        *_group(
            claim_overrides=[
                {"value": "green tea in the morning"},
                {"value": "green tea when working"},
                {"value": "green tea over soda"},
            ]
        )
    )[0]
    four = derive_patterns(
        *_group(
            count=4,
            claim_overrides=[
                {"value": "green tea in the morning"},
                {"value": "green tea when working"},
                {"value": "green tea over soda"},
                {"value": "green tea after lunch"},
            ],
        )
    )[0]
    assert three.value == four.value


def test_value_falls_back_when_restatements_share_no_prefix():
    """Different wording for one slot has no shared core, so the shortest member
    is used -- deterministically, so the value does not depend on arrival order."""
    claims, events, evidence = _group(
        claim_overrides=[
            {"value": "black coffee"},
            {"value": "espresso strong"},
            {"value": "latte warm"},
        ]
    )
    value = derive_patterns(claims, events, evidence)[0].value
    assert value == "repeatedly prefers latte warm"
    reversed_value = derive_patterns(list(reversed(claims)), events, evidence)[0].value
    assert reversed_value == value


def test_negative_stance_selects_the_avoids_template():
    claims, events, evidence = _group(
        claim_overrides=[{"stance": "negative"}] * PATTERN_MIN_EVIDENCE
    )
    pattern = derive_patterns(claims, events, evidence)[0]
    assert pattern.stance == "negative"
    assert "avoids" in pattern.value
    assert "prefers" not in pattern.value


def test_below_the_evidence_floor_produces_nothing():
    claims, events, evidence = _group(count=PATTERN_MIN_EVIDENCE - 1)
    assert derive_patterns(claims, events, evidence) == []


def test_evidence_inside_one_exchange_is_not_a_pattern():
    """Three statements minutes apart are one opinion voiced three times.
    SPEC :91 requires repeated evidence *across windows*."""
    claims, events, evidence = _group(span_days=0)
    assert derive_patterns(claims, events, evidence) == []


def test_mixed_stances_produce_nothing():
    """A reversal is already represented by the supersede chain; calling it a
    pattern would assert a stability that is not there."""
    claims, events, evidence = _group(
        claim_overrides=[{"stance": "positive"}, {"stance": "negative"}, {"stance": "positive"}]
    )
    assert derive_patterns(claims, events, evidence) == []


def test_allowed_purposes_is_the_intersection_not_the_union():
    """Retrieval admits a claim if either the scope or the purpose appears in
    `allowed_purposes`, so a union would surface a pattern under a purpose its
    narrowest source never consented to."""
    claims, events, evidence = _group(
        scopes=[
            ["personalization", "task_planning"],
            ["personalization"],
            ["personalization", "analytics"],
        ]
    )
    pattern = derive_patterns(claims, events, evidence)[0]
    assert pattern.allowed_purposes == ("personalization",)


def test_empty_purpose_intersection_produces_nothing():
    """With no shared scope there is no purpose under which the pattern could be
    served without over-disclosing."""
    claims, events, evidence = _group(
        scopes=[["personalization"], ["task_planning"], ["analytics"]]
    )
    assert derive_patterns(claims, events, evidence) == []


def test_any_sensitive_source_makes_the_pattern_sensitive():
    """The summary discloses that the underlying statements were made, so it
    inherits the strictest sensitivity of its sources (#2 depends on this)."""
    claims, events, evidence = _group(claim_overrides=[{}, {"sensitive": True}, {}])
    assert derive_patterns(claims, events, evidence)[0].sensitive is True


def test_explicitly_corrected_dimension_is_skipped():
    """An inferred pattern must never override an explicit statement (SPEC No-Go)."""
    claims, events, evidence = _group()
    protected = frozenset({claims[0].semantic_key})
    assert derive_patterns(claims, events, evidence, protected_semantic_keys=protected) == []


def test_explicit_feedback_in_the_group_is_skipped():
    """Same rule via the other route: the user has spoken on this dimension."""
    claims, events, evidence = _group(
        claim_overrides=[{}, {"source_type": "explicit_feedback"}, {}]
    )
    assert derive_patterns(claims, events, evidence) == []


def test_superseded_members_count_toward_the_pattern():
    """Required, not incidental: slot-keyed preferences share a semantic_key and a
    paraphrased restatement supersedes rather than merges, so the repetition lives
    in the chain. Reading only active claims would miss a pattern exactly where the
    user repeated themselves most."""
    claims, events, evidence = _group(
        claim_overrides=[{"status": "superseded"}, {"status": "superseded"}, {}]
    )
    result = derive_patterns(claims, events, evidence)
    assert len(result) == 1
    assert result[0].evidence_count == PATTERN_MIN_EVIDENCE


def test_a_fully_retired_dimension_produces_nothing():
    """With no active member there is no current preference to summarize as an
    ongoing pattern."""
    claims, events, evidence = _group(
        claim_overrides=[{"status": "superseded"}] * PATTERN_MIN_EVIDENCE
    )
    assert derive_patterns(claims, events, evidence) == []


def test_deleted_or_rejected_claims_are_ignored():
    claims, events, evidence = _group(claim_overrides=[{}, {"status": "rejected"}, {}])
    assert derive_patterns(claims, events, evidence) == []


def test_non_preference_kinds_are_ignored():
    claims, events, evidence = _group(claim_overrides=[{"kind": "episodic"}, {}, {}])
    assert derive_patterns(claims, events, evidence) == []


def test_valid_from_is_the_newest_source_event():
    """`_effective_confidence` decays from `valid_from`; taking the oldest event
    would make a freshly derived pattern already half-decayed."""
    claims, events, evidence = _group(span_days=40)
    pattern = derive_patterns(claims, events, evidence)[0]
    assert pattern.valid_from == max(event.occurred_at for event in events.values())


def test_unresolvable_evidence_ids_are_dropped_not_guessed():
    claims, events, evidence = _group()
    evidence[claims[0].id] = ["missing-event"]
    # Two resolvable events remain, below the floor.
    assert derive_patterns(claims, events, evidence) == []


def test_confidence_grows_with_evidence_but_stays_below_explicit():
    small = derive_patterns(*_group(count=3))[0]
    large = derive_patterns(*_group(count=8))[0]
    assert large.confidence > small.confidence
    assert large.confidence < 0.99  # explicit feedback confidence


def test_derivation_basis_is_order_independent():
    """Replay determinism (#12): the same event set in any order is one claim."""
    forward = pattern_derivation_basis("u1", "key", ["e3", "e1", "e2"], "v1")
    backward = pattern_derivation_basis("u1", "key", ["e2", "e3", "e1"], "v1")
    assert forward == backward


def test_derivation_basis_changes_when_the_evidence_set_changes():
    """This is the staleness mechanism: a changed support set is a new claim,
    superseding the old one through the existing semantic_key index."""
    before = pattern_derivation_basis("u1", "key", ["e1", "e2", "e3"], "v1")
    after = pattern_derivation_basis("u1", "key", ["e1", "e2", "e3", "e4"], "v1")
    assert before != after


def test_derivation_basis_is_scoped_by_user_and_extractor_version():
    base = pattern_derivation_basis("u1", "key", ["e1"], "v1")
    assert pattern_derivation_basis("u2", "key", ["e1"], "v1") != base
    assert pattern_derivation_basis("u1", "key", ["e1"], "v2") != base


def test_output_is_deterministic_across_input_orderings():
    claims, events, evidence = _group(count=4)
    forward = derive_patterns(claims, events, evidence)
    backward = derive_patterns(list(reversed(claims)), events, evidence)
    assert forward == backward


def test_separate_dimensions_yield_separate_patterns():
    claims_a, events_a, evidence_a = _group()
    claims_b, events_b, evidence_b = _group()
    claims_b = [
        replace(claim, id=f"b{index}", semantic_key="key-coffee", value="black coffee")
        for index, claim in enumerate(claims_b)
    ]
    events_b = {f"x{key[1:]}": replace(event, id=f"x{key[1:]}") for key, event in events_b.items()}
    evidence_b = {claim.id: [f"x{index}"] for index, claim in enumerate(claims_b)}
    result = derive_patterns(
        [*claims_a, *claims_b], {**events_a, **events_b}, {**evidence_a, **evidence_b}
    )
    assert {pattern.value for pattern in result} == {
        "repeatedly prefers green tea",
        "repeatedly prefers black coffee",
    }
