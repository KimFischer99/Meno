from __future__ import annotations

import math
from dataclasses import dataclass

from .vector import Embedder, EmbeddingError

RESOLVER_VERSION = "meno-semantic-shadow-1.0.0"


@dataclass(frozen=True)
class SemanticReference:
    semantic_key: str
    value: str
    sensitive: bool = False


@dataclass(frozen=True)
class SemanticProposal:
    resolver_version: str
    proposed_semantic_key: str | None
    score: float | None
    proposed_merge: bool
    reason: str


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity without assuming provider-side normalization."""
    if not left or len(left) != len(right):
        raise ValueError("embedding vectors must have the same non-zero dimension")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not math.isfinite(left_norm) or not math.isfinite(right_norm):
        raise ValueError("embedding vectors must be finite")
    if left_norm == 0 or right_norm == 0:
        raise ValueError("embedding vectors must have non-zero norm")
    score = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, score))


class ShadowSemanticResolver:
    """Read-only semantic-key proposal engine.

    Callers remain responsible for restricting references to the same user,
    claim kind, and semantic channel. This class never reads or writes Meno's
    canonical store and therefore cannot change claim coordination semantics.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        threshold: float,
        resolver_version: str = RESOLVER_VERSION,
        tie_epsilon: float = 1e-9,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if tie_epsilon < 0:
            raise ValueError("tie_epsilon must be non-negative")
        self.embedder = embedder
        self.threshold = threshold
        self.resolver_version = resolver_version
        self.tie_epsilon = tie_epsilon

    def propose(
        self,
        candidate_value: str,
        references: list[SemanticReference],
        *,
        candidate_sensitive: bool = False,
    ) -> SemanticProposal:
        candidate = candidate_value.strip()
        if not candidate or not references:
            return SemanticProposal(
                resolver_version=self.resolver_version,
                proposed_semantic_key=None,
                score=None,
                proposed_merge=False,
                reason="no_candidate" if not candidate else "no_reference",
            )
        if any(not item.semantic_key or not item.value.strip() for item in references):
            raise ValueError("references require non-empty semantic_key and value")
        if candidate_sensitive or any(item.sensitive for item in references):
            return SemanticProposal(
                resolver_version=self.resolver_version,
                proposed_semantic_key=None,
                score=None,
                proposed_merge=False,
                reason="sensitive_input",
            )

        try:
            vectors = self.embedder.embed_documents(
                [candidate, *(item.value for item in references)]
            )
        except EmbeddingError:
            return SemanticProposal(
                resolver_version=self.resolver_version,
                proposed_semantic_key=None,
                score=None,
                proposed_merge=False,
                reason="provider_unavailable",
            )
        if len(vectors) != len(references) + 1:
            raise ValueError("embedder returned an unexpected number of vectors")
        scores = [cosine_similarity(vectors[0], vector) for vector in vectors[1:]]
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        best_index = ranked_indices[0]
        best_score = scores[best_index]
        if len(ranked_indices) > 1 and best_score - scores[ranked_indices[1]] <= self.tie_epsilon:
            return SemanticProposal(
                resolver_version=self.resolver_version,
                proposed_semantic_key=None,
                score=best_score,
                proposed_merge=False,
                reason="ambiguous_reference",
            )
        proposed_merge = best_score >= self.threshold
        return SemanticProposal(
            resolver_version=self.resolver_version,
            proposed_semantic_key=(references[best_index].semantic_key if proposed_merge else None),
            score=best_score,
            proposed_merge=proposed_merge,
            reason="threshold_match" if proposed_merge else "below_threshold",
        )
