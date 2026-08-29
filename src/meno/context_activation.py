"""Deterministic context activation after candidate retrieval.

Retrieval produces candidates; this policy decides whether a candidate is
relevant enough to enter the current User Context. Sensitive canonical claims
use lexical evidence only and are never sent to the embedding provider.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .extractor import preference_slot

SEMANTIC_THRESHOLD = 0.25
LEXICAL_THRESHOLD = 0.34
# Kinds whose claims carry a preference slot and can therefore be activated by a
# slot match. `pattern` claims are reflection summaries of slotted preferences and
# inherit the slot, so they belong here too.
SLOT_KEYED_KINDS = frozenset({"preference", "pattern"})
TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "for",
    "how",
    "i",
    "is",
    "me",
    "my",
    "of",
    "should",
    "the",
    "to",
    "what",
    "which",
    "you",
}


@dataclass(frozen=True)
class ActivationDecision:
    allowed: bool
    lexical_score: float
    reasons: tuple[str, ...]


def lexical_overlap(query: str, value: str) -> float:
    query_tokens = {
        token for token in TOKEN_RE.findall(query.casefold()) if token not in STOPWORDS
    }
    value_tokens = {
        token for token in TOKEN_RE.findall(value.casefold()) if token not in STOPWORDS
    }
    if not query_tokens or not value_tokens:
        return 0.0
    return len(query_tokens & value_tokens) / len(value_tokens)


class ContextActivationPolicy:
    def decide(
        self,
        *,
        query: str,
        task_type: str | None,
        kind: str,
        value: str,
        routing_slot: str | None,
        sensitive: bool,
        semantic_score: float | None,
    ) -> ActivationDecision:
        context = f"{task_type or ''} {query}".strip()
        query_slot = preference_slot(context.replace("_", " "))
        overlap = lexical_overlap(context, value)

        # Slot matching applies to any slot-keyed state claim, not only to raw
        # preferences. A reflection-derived `pattern` carries the slot of the
        # preferences it summarizes, so gating this on `preference` alone would
        # route every pattern to the lexical path -- where a canonical summary
        # ("repeatedly prefers green tea") rarely overlaps a natural question
        # enough to activate, making the layer unreachable in practice.
        if kind in SLOT_KEYED_KINDS and routing_slot and query_slot:
            if routing_slot == query_slot:
                return ActivationDecision(True, overlap, ("context slot match",))
            return ActivationDecision(False, overlap, ("context slot mismatch",))

        if sensitive:
            allowed = overlap >= LEXICAL_THRESHOLD
            return ActivationDecision(
                allowed,
                overlap,
                (
                    "sensitive canonical lexical match"
                    if allowed
                    else "insufficient sensitive lexical evidence"
                ,),
            )

        semantic = (
            semantic_score
            if semantic_score is not None and math.isfinite(semantic_score)
            else -1.0
        )
        allowed = semantic >= SEMANTIC_THRESHOLD or overlap >= LEXICAL_THRESHOLD
        reasons = []
        if semantic >= SEMANTIC_THRESHOLD:
            reasons.append("semantic activation threshold")
        if overlap >= LEXICAL_THRESHOLD:
            reasons.append("lexical activation threshold")
        if not reasons:
            reasons.append("insufficient context relevance")
        return ActivationDecision(allowed, overlap, tuple(reasons))
