"""Shadow-only semantic dimension routing.

This module deliberately contains no persistence, service, or write-path
integration.  It turns typed candidate/reference envelopes into an auditable
proposal that a later caller may choose to apply.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .vector import Embedder, EmbeddingError

STRATEGY_VERSION = "meno-prototype-dimension-1.0.0"
ENSEMBLE_STRATEGY_VERSION = "meno-prototype-ensemble-2.0.0"
ROUTER_VERSION = "meno-policy-router-2.0.0"

RouterAction = Literal["reuse_key", "new_key", "reject"]
EnsembleAggregation = Literal["mean", "max", "top2_mean"]


def _require_text(value: object, *, name: str, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        suffix = " or None" if optional else ""
        raise TypeError(f"{name} must be a non-empty string{suffix}")


def _require_bool(value: object, *, name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")


@dataclass(frozen=True)
class CandidateEnvelope:
    """A candidate presented to the shadow router.

    Security-relevant flags are intentionally required arguments.  A caller
    must make an explicit decision about sensitivity and prompt injection.
    """

    user_id: str
    kind: str
    semantic_channel: str
    value: str
    sensitive: bool
    injection_detected: bool
    deterministic_slot: str | None

    def __post_init__(self) -> None:
        _require_text(self.user_id, name="candidate.user_id")
        _require_text(self.kind, name="candidate.kind")
        _require_text(self.semantic_channel, name="candidate.semantic_channel")
        _require_text(self.value, name="candidate.value")
        _require_bool(self.sensitive, name="candidate.sensitive")
        _require_bool(self.injection_detected, name="candidate.injection_detected")
        _require_text(
            self.deterministic_slot,
            name="candidate.deterministic_slot",
            optional=True,
        )


@dataclass(frozen=True)
class ReferenceEnvelope:
    """A read-only reference claim supplied by the caller."""

    claim_id: str
    user_id: str
    semantic_key: str
    kind: str
    semantic_channel: str
    value: str
    sensitive: bool
    status: str
    source_type: str
    deterministic_slot: str | None

    def __post_init__(self) -> None:
        _require_text(self.claim_id, name="reference.claim_id")
        _require_text(self.user_id, name="reference.user_id")
        _require_text(self.semantic_key, name="reference.semantic_key")
        _require_text(self.kind, name="reference.kind")
        _require_text(self.semantic_channel, name="reference.semantic_channel")
        _require_text(self.value, name="reference.value")
        _require_bool(self.sensitive, name="reference.sensitive")
        _require_text(self.status, name="reference.status")
        _require_text(self.source_type, name="reference.source_type")
        _require_text(
            self.deterministic_slot,
            name="reference.deterministic_slot",
            optional=True,
        )


@dataclass(frozen=True)
class DimensionPrototype:
    """A semantic dimension name and the text used to represent it."""

    name: str
    description: str


@dataclass(frozen=True)
class DimensionScore:
    """Top-two prototype scores for one candidate."""

    dimension: str
    score: float
    second_dimension: str | None
    second_score: float | None
    margin: float


@dataclass(frozen=True)
class RouterDecision:
    """A shadow-only routing proposal."""

    action: RouterAction
    proposed_semantic_key: str | None
    matched_claim_id: str | None
    dimension: str | None
    score: float | None
    margin: float | None
    strategy_version: str
    router_version: str
    reason: str
    protected_reference: bool


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return a finite cosine similarity for two non-zero vectors."""

    if not left or len(left) != len(right):
        raise ValueError("embedding vectors must have the same non-zero dimension")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not math.isfinite(left_norm) or not math.isfinite(right_norm):
        raise ValueError("embedding vectors must be finite")
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("embedding vectors must have non-zero norm")
    score = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    if not math.isfinite(score):
        raise ValueError("embedding similarity must be finite")
    return max(-1.0, min(1.0, score))


def _validate_vector(vector: object, *, label: str) -> list[float]:
    """Convert a provider vector to finite floats, or fail closed loudly."""

    try:
        values = [float(value) for value in vector]  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} embedding vector is invalid") from exc
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} embedding vector must be finite and non-empty")
    return values


class PrototypeDimensionStrategy:
    """Score safe candidates against semantic dimension prototypes.

    The provider receives one batch containing only non-sensitive,
    non-injection candidate values followed by the prototype descriptions.
    Unsafe candidates are never sent to the provider and receive ``None``.
    """

    def __init__(
        self,
        embedder: Embedder,
        prototypes: Sequence[DimensionPrototype],
        *,
        strategy_version: str = STRATEGY_VERSION,
    ) -> None:
        prototype_tuple = tuple(prototypes)
        if not prototype_tuple:
            raise ValueError("at least one dimension prototype is required")
        if any(not prototype.name.strip() for prototype in prototype_tuple):
            raise ValueError("dimension prototype names must be non-empty")
        if any(not prototype.description.strip() for prototype in prototype_tuple):
            raise ValueError("dimension prototype descriptions must be non-empty")
        if len({prototype.name for prototype in prototype_tuple}) != len(prototype_tuple):
            raise ValueError("dimension prototype names must be unique")
        if not strategy_version:
            raise ValueError("strategy_version must be non-empty")
        self.embedder = embedder
        self.prototypes = prototype_tuple
        self.strategy_version = strategy_version

    def score_many(
        self,
        candidates: Sequence[CandidateEnvelope],
    ) -> list[DimensionScore | None]:
        """Return one top-two score (or ``None``) per candidate.

        ``EmbeddingError`` is intentionally converted into unavailable scores
        for all candidates in this provider batch.  Shape and finite-value
        violations are programming/provider contract errors and raise
        ``ValueError`` instead of being silently treated as a match.
        """

        candidate_list = list(candidates)
        if not candidate_list:
            return []

        safe_indices = [
            index
            for index, candidate in enumerate(candidate_list)
            if not candidate.sensitive and not candidate.injection_detected
        ]
        results: list[DimensionScore | None] = [None] * len(candidate_list)
        if not safe_indices:
            return results

        texts = [candidate_list[index].value for index in safe_indices]
        texts.extend(prototype.description for prototype in self.prototypes)
        try:
            raw_vectors = self.embedder.embed_documents(texts)
        except EmbeddingError:
            return results

        expected_count = len(safe_indices) + len(self.prototypes)
        if len(raw_vectors) != expected_count:
            raise ValueError(
                "embedder returned an unexpected number of vectors for candidates and prototypes"
            )

        vectors = [
            _validate_vector(vector, label=f"vector {index}")
            for index, vector in enumerate(raw_vectors)
        ]
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError("embedding vectors must have a consistent dimension")

        prototype_vectors = vectors[len(safe_indices) :]
        for result_index, candidate_index in enumerate(safe_indices):
            candidate_vector = vectors[result_index]
            scores = [
                _cosine_similarity(candidate_vector, prototype_vector)
                for prototype_vector in prototype_vectors
            ]
            ranked_indices = sorted(
                range(len(scores)),
                key=lambda index: scores[index],
                reverse=True,
            )
            best_index = ranked_indices[0]
            best_score = scores[best_index]
            if len(ranked_indices) > 1:
                second_index = ranked_indices[1]
                second_dimension = self.prototypes[second_index].name
                second_score = scores[second_index]
            else:
                second_dimension = None
                # A single prototype has no competing dimension.  Zero is a
                # finite neutral baseline, keeping margin usable by PolicyRouter.
                second_score = 0.0
            margin = best_score - second_score
            if not math.isfinite(margin):
                raise ValueError("dimension score margin must be finite")
            results[candidate_index] = DimensionScore(
                dimension=self.prototypes[best_index].name,
                score=best_score,
                second_dimension=second_dimension,
                second_score=second_score,
                margin=margin,
            )
        return results

    def score(self, candidate: CandidateEnvelope) -> DimensionScore | None:
        """Score one candidate through the same batch implementation."""

        return self.score_many([candidate])[0]


class PrototypeEnsembleStrategy:
    """Score dimensions represented by one or more controlled anchors.

    ``DimensionPrototype`` entries with the same name form one dimension
    ensemble.  The provider receives one batch containing safe candidate
    values followed by every anchor description; reference values are never
    sent to the provider.  Aggregation is deliberately limited to the small
    set of deterministic policies supported by the shadow contract.
    """

    _AGGREGATIONS = frozenset({"mean", "max", "top2_mean"})

    def __init__(
        self,
        embedder: Embedder,
        prototypes: Sequence[DimensionPrototype],
        *,
        aggregation: EnsembleAggregation = "mean",
        strategy_version: str = ENSEMBLE_STRATEGY_VERSION,
    ) -> None:
        prototype_tuple = tuple(prototypes)
        if not prototype_tuple:
            raise ValueError("at least one dimension prototype is required")
        if any(not isinstance(prototype, DimensionPrototype) for prototype in prototype_tuple):
            raise TypeError("ensemble prototypes must be DimensionPrototype values")
        if any(not prototype.name.strip() for prototype in prototype_tuple):
            raise ValueError("dimension prototype names must be non-empty")
        if any(not prototype.description.strip() for prototype in prototype_tuple):
            raise ValueError("dimension prototype descriptions must be non-empty")
        if not isinstance(aggregation, str) or aggregation not in self._AGGREGATIONS:
            raise ValueError("aggregation must be one of: mean, max, top2_mean")
        if not strategy_version:
            raise ValueError("strategy_version must be non-empty")

        anchors_by_dimension: dict[str, list[int]] = {}
        for index, prototype in enumerate(prototype_tuple):
            anchors_by_dimension.setdefault(prototype.name, []).append(index)

        self.embedder = embedder
        self.prototypes = prototype_tuple
        self.anchors_by_dimension = {
            name: tuple(indices) for name, indices in anchors_by_dimension.items()
        }
        self.aggregation = aggregation
        self.strategy_version = strategy_version

    def score_many(
        self,
        candidates: Sequence[CandidateEnvelope],
    ) -> list[DimensionScore | None]:
        """Return one top-two dimension score (or ``None``) per candidate."""

        candidate_list = list(candidates)
        if not candidate_list:
            return []

        safe_indices = [
            index
            for index, candidate in enumerate(candidate_list)
            if (
                not candidate.sensitive
                and not candidate.injection_detected
                and candidate.deterministic_slot is None
            )
        ]
        results: list[DimensionScore | None] = [None] * len(candidate_list)
        if not safe_indices:
            return results

        texts = [candidate_list[index].value for index in safe_indices]
        texts.extend(prototype.description for prototype in self.prototypes)
        try:
            raw_vectors = self.embedder.embed_documents(texts)
        except EmbeddingError:
            return results

        expected_count = len(safe_indices) + len(self.prototypes)
        if len(raw_vectors) != expected_count:
            raise ValueError(
                "embedder returned an unexpected number of vectors for candidates and prototypes"
            )

        vectors = [
            _validate_vector(vector, label=f"vector {index}")
            for index, vector in enumerate(raw_vectors)
        ]
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError("embedding vectors must have a consistent dimension")

        prototype_vectors = vectors[len(safe_indices) :]
        dimension_names = tuple(self.anchors_by_dimension)
        for result_index, candidate_index in enumerate(safe_indices):
            candidate_vector = vectors[result_index]
            dimension_scores = {
                name: self._aggregate(
                    [
                        _cosine_similarity(candidate_vector, prototype_vectors[anchor_index])
                        for anchor_index in anchor_indices
                    ]
                )
                for name, anchor_indices in self.anchors_by_dimension.items()
            }
            ranked_dimensions = sorted(
                dimension_names,
                key=lambda name: dimension_scores[name],
                reverse=True,
            )
            best_dimension = ranked_dimensions[0]
            best_score = dimension_scores[best_dimension]
            if len(ranked_dimensions) > 1:
                second_dimension = ranked_dimensions[1]
                second_score = dimension_scores[second_dimension]
            else:
                second_dimension = None
                # A single dimension has no competing dimension.  Zero is a
                # finite neutral baseline, matching PrototypeDimensionStrategy.
                second_score = 0.0
            margin = best_score - second_score
            if not math.isfinite(margin):
                raise ValueError("dimension score margin must be finite")
            results[candidate_index] = DimensionScore(
                dimension=best_dimension,
                score=best_score,
                second_dimension=second_dimension,
                second_score=second_score,
                margin=margin,
            )
        return results

    def score(self, candidate: CandidateEnvelope) -> DimensionScore | None:
        """Score one candidate through the same batch implementation."""

        return self.score_many([candidate])[0]

    def _aggregate(self, scores: Sequence[float]) -> float:
        if not scores or any(not math.isfinite(score) for score in scores):
            raise ValueError("dimension anchor scores must be finite and non-empty")
        if self.aggregation == "max":
            return max(scores)
        if self.aggregation == "top2_mean":
            selected = sorted(scores, reverse=True)[:2]
            return sum(selected) / len(selected)
        return sum(scores) / len(scores)


class PolicyRouter:
    """Apply scope, safety, dimension, and ambiguity policy to a proposal."""

    def __init__(
        self,
        *,
        strategy_version: str = STRATEGY_VERSION,
        router_version: str = ROUTER_VERSION,
    ) -> None:
        if not strategy_version:
            raise ValueError("strategy_version must be non-empty")
        if not router_version:
            raise ValueError("router_version must be non-empty")
        self.strategy_version = strategy_version
        self.router_version = router_version

    def decide(
        self,
        candidate: CandidateEnvelope,
        references: Sequence[ReferenceEnvelope],
        dimension_score: DimensionScore | None,
        score_threshold: float,
        margin_threshold: float,
    ) -> RouterDecision:
        """Return a fail-closed routing decision without touching persistence."""

        score_threshold = self._validate_threshold(score_threshold, "score_threshold")
        margin_threshold = self._validate_threshold(margin_threshold, "margin_threshold")

        if candidate.sensitive or candidate.injection_detected:
            reason = "injection_detected" if candidate.injection_detected else "sensitive_candidate"
            return self._decision(
                action="reject",
                dimension=None,
                score=None,
                margin=None,
                reason=reason,
            )

        scoped_references = [
            reference
            for reference in references
            if (
                reference.user_id == candidate.user_id
                and reference.kind == candidate.kind
                and reference.semantic_channel == candidate.semantic_channel
                and reference.status == "active"
                and not reference.sensitive
            )
        ]

        if candidate.deterministic_slot is not None:
            dimension = candidate.deterministic_slot
            score = None
            margin = None
            reason = "deterministic_slot"
        else:
            if dimension_score is None:
                return self._decision(
                    action="new_key",
                    dimension=None,
                    score=None,
                    margin=None,
                    reason="provider_unavailable",
                )
            dimension = dimension_score.dimension
            score = self._finite_or_none(dimension_score.score)
            margin = self._finite_or_none(dimension_score.margin)
            if score is None or margin is None:
                return self._decision(
                    action="new_key",
                    dimension=dimension,
                    score=None,
                    margin=None,
                    reason="invalid_dimension_score",
                )
            reason = "dimension_match"

        matching_references = [
            reference
            for reference in scoped_references
            if reference.deterministic_slot == dimension
        ]
        semantic_keys = {reference.semantic_key for reference in matching_references}
        if len(semantic_keys) > 1:
            return self._decision(
                action="reject",
                dimension=dimension,
                score=score,
                margin=margin,
                reason="ambiguous_semantic_key",
            )

        matched_reference = (
            min(
                matching_references,
                key=lambda reference: (
                    reference.source_type != "explicit_feedback",
                    reference.claim_id,
                ),
            )
            if matching_references
            else None
        )
        if (
            matched_reference is not None
            and matched_reference.source_type == "explicit_feedback"
            and score is not None
            and margin is not None
            and (score < score_threshold or margin < margin_threshold)
        ):
            return self._decision(
                action="reject",
                dimension=dimension,
                score=score,
                margin=margin,
                reason="explicit_feedback_protected",
                matched_claim_id=matched_reference.claim_id,
                protected_reference=True,
            )

        if score is not None and margin is not None:
            if score < score_threshold:
                return self._decision(
                    action="new_key",
                    dimension=dimension,
                    score=score,
                    margin=margin,
                    reason="below_score_threshold",
                )
            if margin < margin_threshold:
                return self._decision(
                    action="new_key",
                    dimension=dimension,
                    score=score,
                    margin=margin,
                    reason="below_margin_threshold",
                )

        if matched_reference is None:
            return self._decision(
                action="new_key",
                dimension=dimension,
                score=score,
                margin=margin,
                reason="no_matching_reference",
            )

        semantic_key = next(iter(semantic_keys))
        return self._decision(
            action="reuse_key",
            proposed_semantic_key=semantic_key,
            matched_claim_id=matched_reference.claim_id,
            dimension=dimension,
            score=score,
            margin=margin,
            reason=reason,
            protected_reference=matched_reference.source_type == "explicit_feedback",
        )

    def _decision(
        self,
        *,
        action: RouterAction,
        dimension: str | None,
        score: float | None,
        margin: float | None,
        reason: str,
        proposed_semantic_key: str | None = None,
        matched_claim_id: str | None = None,
        protected_reference: bool = False,
    ) -> RouterDecision:
        return RouterDecision(
            action=action,
            proposed_semantic_key=proposed_semantic_key,
            matched_claim_id=matched_claim_id,
            dimension=dimension,
            score=score,
            margin=margin,
            strategy_version=self.strategy_version,
            router_version=self.router_version,
            reason=reason,
            protected_reference=protected_reference,
        )

    @staticmethod
    def _finite_or_none(value: float) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value) if math.isfinite(value) else None

    @staticmethod
    def _validate_threshold(value: float, name: str) -> float:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and between 0 and 1")
        return value


__all__ = [
    "ENSEMBLE_STRATEGY_VERSION",
    "ROUTER_VERSION",
    "STRATEGY_VERSION",
    "CandidateEnvelope",
    "DimensionPrototype",
    "DimensionScore",
    "EnsembleAggregation",
    "PolicyRouter",
    "PrototypeDimensionStrategy",
    "PrototypeEnsembleStrategy",
    "ReferenceEnvelope",
    "RouterAction",
    "RouterDecision",
]
