"""Deterministic reflection: derive `pattern` claims from repeated preferences.

Meno's extractor produces two claim kinds (`preference`, `episodic`) while
`schemas.py` declares seven and `Meno_SPEC.md` :119 asks for five state layers.
This module fills the first of the missing ones. The retrieval side already
expects it: `pattern` is listed in `service.STATE_LAYER_KINDS`, so layered
injection reserves budget for these facets as soon as something produces them.

The shape is borrowed from generative-agent memory (observation -> reflection ->
plan), where reflections are higher-order judgements synthesized from several
observations. Two deliberate departures:

**The synthesis is deterministic, not generative.** A pattern's value is
assembled from words that already appear in the source evidence, plus a small
fixed set of template words. This keeps Gate Charter #9 (unsupported injected
claim rate) meaningful: that lane checks, token by token, that a claim's value
appears in its evidence events. A generative reflection would assert words found
in no event and fail it by construction, so the abstraction is traded away to
keep the audit property. `PATTERN_TEMPLATE_TOKENS` is the entire vocabulary this
module can introduce, and the grounding lane whitelists exactly that set.

**Reflections carry lineage.** A generative agent records which observations a
reflection came from as a convenience. Here the source events become real
`ClaimEvidence` rows, so a pattern is subject to the same provenance, consent,
and supersede machinery as any other claim -- which is the property Meno is
actually built to defend.

Every function here is pure: callers pass claims and events in and get candidates
out. No session, no I/O, no clock. That is what makes the write-path integration
replayable (Charter #12) -- `derivation_key` depends only on the user, the
semantic key, the sorted source event ids, and the extractor version.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

# Fewer than this many distinct evidence events is a repetition, not a pattern.
# SPEC :91 requires "repeated evidence across windows" for behavioral patterns.
PATTERN_MIN_EVIDENCE = 3

# The evidence must also span time. Three statements inside one exchange are one
# opinion voiced three times; the same statement weeks apart is a pattern.
PATTERN_MIN_SPAN = timedelta(days=1)

# The only words this module may introduce that are not copied from evidence.
# The grounding lane (Charter #9) whitelists exactly this set, so any addition
# here widens what that gate tolerates and must be recorded there too.
PATTERN_TEMPLATE_TOKENS = frozenset({"repeatedly", "prefers", "avoids"})

PATTERN_CHANNEL = "pattern.repeated_preference"

# Patterns are slower to form and slower to fade than the preferences they
# summarize (SPEC :91 puts long-term behavioral patterns at the month scale).
PATTERN_HALF_LIFE_DAYS = 180.0

# A pattern is more than any single observation but is still an inference, so it
# stays below the confidence of an explicit statement.
PATTERN_BASE_CONFIDENCE = 0.72
PATTERN_CONFIDENCE_STEP = 0.04
PATTERN_MAX_CONFIDENCE = 0.9

STANCE_POSITIVE = "positive"
STANCE_NEGATIVE = "negative"


class ClaimLike(Protocol):
    """The subset of `db.Claim` this module reads."""

    id: str
    user_id: str
    kind: str
    semantic_channel: str
    semantic_key: str
    value: str
    status: str
    stance: str | None
    sensitive: bool
    source_type: str
    routing_slot: str | None


class EventLike(Protocol):
    """The subset of `db.Event` this module reads."""

    id: str
    occurred_at: datetime
    consent_scope: list[str]


@dataclass(frozen=True)
class PatternCandidate:
    """A `pattern` claim ready to be written, with its provenance resolved."""

    user_id: str
    semantic_channel: str
    value: str
    stance: str | None
    confidence: float
    half_life_days: float
    sensitive: bool
    # Intersection of the source events' consent scopes. See `_purpose_intersection`.
    allowed_purposes: tuple[str, ...]
    # Sorted, so the derivation key does not depend on processing order.
    source_event_ids: tuple[str, ...]
    source_claim_ids: tuple[str, ...]
    # Newest source event: a pattern asserts "this holds as of now", and
    # `_effective_confidence` decays from `valid_from`.
    valid_from: datetime
    routing_slot: str | None

    @property
    def evidence_count(self) -> int:
        return len(self.source_event_ids)


def pattern_derivation_basis(
    user_id: str,
    semantic_key: str,
    source_event_ids: Sequence[str],
    extractor_version: str,
) -> str:
    """Idempotency basis for a pattern claim.

    Deliberately excludes wall-clock time and processing order so that replaying
    the same event log yields the same claim id (Charter #12). Including the event
    set also gives the staleness behavior for free: when the supporting set
    changes, the derivation key changes, so the result is a *new* claim that
    supersedes the old one through the existing `semantic_key` uniqueness index
    rather than through a bespoke invalidation path.
    """
    joined = ",".join(sorted(source_event_ids))
    return f"reflection:{user_id}:{semantic_key}:{extractor_version}:{joined}"


def _purpose_intersection(events: Sequence[EventLike]) -> tuple[str, ...]:
    """Consent scopes shared by *every* source event.

    This must be an intersection, not a union. Retrieval admits a claim when
    either the request scope or its purpose appears in `allowed_purposes`
    (`service._facet_decision`), so a union would let a pattern surface
    information under a purpose its narrowest source never consented to -- a
    privilege-escalation channel dressed up as an inference.
    """
    if not events:
        return ()
    shared: set[str] | None = None
    for event in events:
        scopes = set(event.consent_scope or ())
        shared = scopes if shared is None else (shared & scopes)
    return tuple(sorted(shared or ()))


def _pattern_value(stance: str | None, objects: Sequence[str]) -> str:
    """Assemble the value from evidence words plus whitelisted template words.

    `objects` are preference values copied verbatim from source claims, which the
    extractor in turn copied from event text -- so every non-template token in the
    result is grounded, which is what keeps Charter #9 meaningful.

    The subject is the longest leading run of words shared by every restatement.
    "green tea in the morning" / "green tea when working" / "green tea over soda"
    reduce to "green tea": the part the user repeated is the pattern, while the
    trailing clauses are what varied. Picking any single member instead would
    assert one occasion's phrasing as the standing preference, and would also make
    the value -- and therefore the claim's semantic key -- depend on which
    restatement happened to arrive last.
    """
    verb = "avoids" if stance == STANCE_NEGATIVE else "prefers"
    tokenized = [item.split() for item in objects]
    shared: list[str] = []
    for position in range(min(len(words) for words in tokenized)):
        candidate = tokenized[0][position]
        if any(words[position] != candidate for words in tokenized):
            break
        shared.append(candidate)
    # No shared prefix (different wording for the same slot): fall back to the
    # shortest member, deterministically tie-broken.
    subject = " ".join(shared) if shared else min(objects, key=lambda item: (len(item), item))
    return f"repeatedly {verb} {subject}"


def derive_patterns(
    claims: Sequence[ClaimLike],
    events: Mapping[str, EventLike],
    claim_evidence: Mapping[str, Sequence[str]],
    *,
    protected_semantic_keys: frozenset[str] = frozenset(),
    min_evidence: int = PATTERN_MIN_EVIDENCE,
    min_span: timedelta = PATTERN_MIN_SPAN,
) -> list[PatternCandidate]:
    """Group preference claims into `pattern` candidates.

    A group is one `semantic_key` -- one preference dimension -- and includes
    superseded members, not just the active one. This is required rather than
    optional: slot-keyed preferences share a `semantic_key`, and the write path
    only merges evidence onto a single claim when a restatement is byte-identical
    (`_coordinate_candidate`). A paraphrase supersedes instead, so "I prefer green
    tea in the morning" followed by "I prefer green tea when working" leaves one
    active claim holding one event. Reading only active claims would therefore
    miss a pattern precisely where the user repeated themselves most -- the
    opposite of the intent.

    Superseded members are admitted regardless of supersede reason, because the
    write path labels every value change on a slot `"contradiction"` even when the
    stance is unchanged -- "green tea in the morning" -> "green tea when working"
    is recorded that way. The reason field therefore cannot distinguish a change of
    mind from a rephrasing. `stance` can, and does: a group with mixed stances is
    dropped, so a genuine reversal ("prefers tea" -> "avoids tea") never becomes a
    pattern, while restatements of the same preference do.

    `protected_semantic_keys` names dimensions the user has explicitly corrected.
    Those are skipped: an inferred pattern must never overwrite an explicit
    statement (`Meno_SPEC.md` No-Go), and the candidate-level guard in the write
    path does not cover this route.

    Returns candidates ordered by semantic key for deterministic output.
    """
    by_key: dict[str, list[ClaimLike]] = defaultdict(list)
    for claim in claims:
        if claim.kind != "preference":
            continue
        if claim.status not in ("active", "superseded"):
            continue
        if claim.semantic_key in protected_semantic_keys:
            continue
        by_key[claim.semantic_key].append(claim)

    candidates: list[PatternCandidate] = []
    for semantic_key in sorted(by_key):
        members = by_key[semantic_key]
        if not any(claim.status == "active" for claim in members):
            # Every member is history; there is no current preference to
            # summarize as an ongoing pattern.
            continue
        stances = {claim.stance for claim in members}
        if len(stances) > 1:
            continue
        stance = next(iter(stances))

        # An explicit correction anywhere in the group means the user has spoken
        # on this dimension; do not summarize over it.
        if any(claim.source_type == "explicit_feedback" for claim in members):
            continue

        event_ids: set[str] = set()
        for claim in members:
            event_ids.update(claim_evidence.get(claim.id, ()))
        resolved = [events[event_id] for event_id in sorted(event_ids) if event_id in events]
        if len(resolved) < min_evidence:
            continue

        occurred = sorted(event.occurred_at for event in resolved)
        if occurred[-1] - occurred[0] < min_span:
            continue

        purposes = _purpose_intersection(resolved)
        if not purposes:
            # No purpose is shared by every source, so there is no scope under
            # which this pattern could be served without over-disclosing.
            continue

        confidence = min(
            PATTERN_MAX_CONFIDENCE,
            PATTERN_BASE_CONFIDENCE + PATTERN_CONFIDENCE_STEP * (len(resolved) - min_evidence),
        )
        slots = {claim.routing_slot for claim in members if claim.routing_slot}
        candidates.append(
            PatternCandidate(
                user_id=members[0].user_id,
                semantic_channel=PATTERN_CHANNEL,
                value=_pattern_value(stance, sorted(claim.value for claim in members)),
                stance=stance,
                confidence=confidence,
                half_life_days=PATTERN_HALF_LIFE_DAYS,
                # Any sensitive source makes the summary sensitive: the pattern
                # discloses that the underlying statements were made.
                sensitive=any(bool(claim.sensitive) for claim in members),
                allowed_purposes=purposes,
                source_event_ids=tuple(
                    event.id for event in sorted(resolved, key=lambda item: item.id)
                ),
                source_claim_ids=tuple(sorted(claim.id for claim in members)),
                valid_from=occurred[-1],
                routing_slot=next(iter(sorted(slots))) if len(slots) == 1 else None,
            )
        )
    return candidates
