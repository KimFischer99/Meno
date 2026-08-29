"""Shadow-only semantic dimension routing.

This module deliberately contains no persistence, service, or write-path
integration.  It turns typed candidate/reference envelopes into an auditable
proposal that a later caller may choose to apply.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .vector import Embedder, EmbeddingError

STRATEGY_VERSION = "meno-prototype-dimension-1.0.0"
ENSEMBLE_STRATEGY_VERSION = "meno-prototype-ensemble-2.0.0"
A4_STRATEGY_VERSION = "meno-prototype-ensemble-role-aware-3.0.0"
# Descriptive alias for callers that identify the A4 strategy by capability.
ROLE_AWARE_ENSEMBLE_STRATEGY_VERSION = A4_STRATEGY_VERSION
A5_STRATEGY_VERSION = "meno-topic-gated-ensemble-4.0.0"
ROUTER_VERSION = "meno-policy-router-2.0.0"

RouterAction = Literal["reuse_key", "new_key", "reject"]
PrototypeRole = Literal["topic", "polarity"]
PrototypePolarity = Literal["neutral", "affirmed", "negated", "contradiction"]
EnsembleAggregation = Literal["mean", "max", "top2_mean", "role_top2_mean"]
CONTROLLED_DIMENSIONS = frozenset(
    {
        "answer_style",
        "beverage",
        "color",
        "food",
        "music",
        "programming_language",
        "sport",
    }
)


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
    """A semantic dimension name and the text used to represent it.

    ``role`` is optional so the A3 prototype shape remains valid.  A4's
    role-aware aggregation requires it to be either ``topic`` or ``polarity``
    for every anchor in every dimension.  ``polarity`` is optional metadata
    used by the topic-gated strategy to label mutation-state diagnostics; it
    never influences dimension ranking.
    """

    name: str
    description: str
    role: PrototypeRole | None = None
    polarity: PrototypePolarity | None = None

    def __post_init__(self) -> None:
        _require_text(self.name, name="dimension prototype name")
        _require_text(self.description, name="dimension prototype description")
        if self.role is not None and (
            not isinstance(self.role, str) or self.role not in {"topic", "polarity"}
        ):
            raise ValueError("dimension prototype role must be topic, polarity, or None")
        if self.polarity is not None and (
            not isinstance(self.polarity, str)
            or self.polarity not in {"neutral", "affirmed", "negated", "contradiction"}
        ):
            raise ValueError(
                "dimension prototype polarity must be neutral, affirmed, negated, "
                "contradiction, or None"
            )


@dataclass(frozen=True)
class DimensionScore:
    """Top-two prototype scores for one candidate."""

    dimension: str
    score: float
    second_dimension: str | None
    second_score: float | None
    margin: float
    strategy_version: str

    def __post_init__(self) -> None:
        _require_text(self.dimension, name="dimension score dimension")
        _require_text(
            self.second_dimension,
            name="dimension score second_dimension",
            optional=True,
        )
        _require_text(self.strategy_version, name="dimension score strategy_version")
        if self.second_dimension == self.dimension:
            raise ValueError("dimension score dimensions must be distinct")
        for name, value, optional in (
            ("score", self.score, False),
            ("second_score", self.second_score, True),
            ("margin", self.margin, False),
        ):
            if optional and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"dimension score {name} must be a number")
            if not math.isfinite(value):
                raise ValueError(f"dimension score {name} must be finite")
        if not -1.0 <= self.score <= 1.0:
            raise ValueError("dimension score score must be in [-1, 1]")
        if self.second_score is not None:
            if not -1.0 <= self.second_score <= 1.0:
                raise ValueError("dimension score second_score must be in [-1, 1]")
            if not math.isclose(
                self.margin,
                self.score - self.second_score,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("dimension score margin must equal score minus second_score")
        if not -2.0 <= self.margin <= 2.0:
            raise ValueError("dimension score margin must be in [-2, 2]")


@dataclass(frozen=True)
class TopicGatedScore:
    """Topic-gated ranking with per-anchor mutation-state diagnostics.

    ``dimension_score`` is produced from topic anchors only.  ``anchor_scores``
    maps dimension -> anchor label -> cosine similarity for every anchor in
    every dimension; topic anchors are labelled ``"topic"`` and polarity
    anchors by their declared polarity (or ``polarity-N`` when unlabelled).
    Polarity scores are diagnostics for the selected dimension and never
    contribute to the ranking.
    """

    dimension_score: DimensionScore
    anchor_scores: Mapping[str, Mapping[str, float]]

    def __post_init__(self) -> None:
        if not isinstance(self.anchor_scores, Mapping):
            raise TypeError("topic gated score anchor_scores must be a mapping")
        observed_dimensions = set(self.anchor_scores)
        ranked_dimensions = {
            dimension
            for dimension in (
                self.dimension_score.dimension,
                self.dimension_score.second_dimension,
            )
            if dimension is not None
        }
        if not ranked_dimensions <= observed_dimensions:
            raise ValueError("topic gated score must report anchors for ranked dimensions")
        for dimension, labels in self.anchor_scores.items():
            _require_text(dimension, name="topic gated score anchor dimension")
            if not isinstance(labels, Mapping) or not labels:
                raise ValueError(
                    "topic gated score anchor labels must be a non-empty mapping"
                )
            for label, score in labels.items():
                _require_text(label, name="topic gated score anchor label")
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise TypeError("topic gated score anchor scores must be numbers")
                if not math.isfinite(score) or not -1.0 <= score <= 1.0:
                    raise ValueError(
                        "topic gated score anchor scores must be finite and in [-1, 1]"
                    )


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


def _embedder_dimension(embedder: Embedder) -> int:
    dimension = getattr(embedder, "dimension", None)
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("embedder dimension must be a positive integer")
    return dimension


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
        if any(not isinstance(prototype, DimensionPrototype) for prototype in prototype_tuple):
            raise TypeError("prototypes must be DimensionPrototype values")
        if len({prototype.name for prototype in prototype_tuple}) != len(prototype_tuple):
            raise ValueError("dimension prototype names must be unique")
        _require_text(strategy_version, name="strategy_version")
        self.embedder = embedder
        self.embedding_dimension = _embedder_dimension(embedder)
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
        if any(len(vector) != self.embedding_dimension for vector in vectors):
            raise ValueError("embedding vectors must match the configured dimension")

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
                strategy_version=self.strategy_version,
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

    _AGGREGATIONS = frozenset({"mean", "max", "top2_mean", "role_top2_mean"})
    _DEFAULT_STRATEGY_VERSION = object()

    def __init__(
        self,
        embedder: Embedder,
        prototypes: Sequence[DimensionPrototype],
        *,
        aggregation: EnsembleAggregation = "mean",
        strategy_version: str | object = _DEFAULT_STRATEGY_VERSION,
        cache_prototypes: bool = False,
    ) -> None:
        prototype_tuple = tuple(prototypes)
        if not prototype_tuple:
            raise ValueError("at least one dimension prototype is required")
        if any(not isinstance(prototype, DimensionPrototype) for prototype in prototype_tuple):
            raise TypeError("ensemble prototypes must be DimensionPrototype values")
        if not isinstance(aggregation, str) or aggregation not in self._AGGREGATIONS:
            raise ValueError("aggregation must be one of: mean, max, top2_mean, role_top2_mean")
        if strategy_version is self._DEFAULT_STRATEGY_VERSION:
            strategy_version = (
                A4_STRATEGY_VERSION
                if aggregation == "role_top2_mean"
                else ENSEMBLE_STRATEGY_VERSION
            )
        if not isinstance(strategy_version, str) or not strategy_version:
            raise ValueError("strategy_version must be non-empty")

        anchors_by_dimension: dict[str, list[int]] = {}
        for index, prototype in enumerate(prototype_tuple):
            anchors_by_dimension.setdefault(prototype.name, []).append(index)

        if aggregation == "role_top2_mean":
            required_roles = {"topic", "polarity"}
            for dimension, anchor_indices in anchors_by_dimension.items():
                roles = {prototype_tuple[index].role for index in anchor_indices}
                if not required_roles <= roles:
                    missing_roles = ", ".join(sorted(required_roles - roles))
                    raise ValueError(
                        "role_top2_mean requires topic and polarity anchors for every "
                        f"dimension; {dimension!r} is missing: {missing_roles}"
                    )

        self.embedder = embedder
        self.embedding_dimension = _embedder_dimension(embedder)
        self.prototypes = prototype_tuple
        self.anchors_by_dimension = {
            name: tuple(indices) for name, indices in anchors_by_dimension.items()
        }
        self.aggregation = aggregation
        self.strategy_version = strategy_version
        self.cache_prototypes = cache_prototypes
        self._prototype_vectors: tuple[list[float], ...] | None = None
        self._prototype_lock = threading.Lock()

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
        if not self.cache_prototypes:
            texts.extend(prototype.description for prototype in self.prototypes)
        try:
            raw_vectors = self.embedder.embed_documents(texts)
            prototype_vectors = self._load_prototype_vectors()
        except EmbeddingError:
            return results

        expected_count = len(safe_indices) + (
            0 if self.cache_prototypes else len(self.prototypes)
        )
        if len(raw_vectors) != expected_count:
            raise ValueError(
                "embedder returned an unexpected number of vectors for candidates and prototypes"
            )

        vectors = [
            _validate_vector(vector, label=f"vector {index}")
            for index, vector in enumerate(raw_vectors)
        ]
        if any(len(vector) != self.embedding_dimension for vector in vectors):
            raise ValueError("embedding vectors must match the configured dimension")

        if not self.cache_prototypes:
            prototype_vectors = tuple(vectors[len(safe_indices) :])
        dimension_names = tuple(self.anchors_by_dimension)
        for result_index, candidate_index in enumerate(safe_indices):
            candidate_vector = vectors[result_index]
            dimension_scores = {
                name: self._aggregate(
                    [
                        _cosine_similarity(candidate_vector, prototype_vectors[anchor_index])
                        for anchor_index in anchor_indices
                    ],
                    roles=[self.prototypes[anchor_index].role for anchor_index in anchor_indices],
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
                strategy_version=self.strategy_version,
            )
        return results

    def _load_prototype_vectors(self) -> tuple[list[float], ...]:
        if not self.cache_prototypes:
            return ()
        with self._prototype_lock:
            if self._prototype_vectors is not None:
                return self._prototype_vectors
            raw_vectors = self.embedder.embed_documents(
                [prototype.description for prototype in self.prototypes]
            )
            if len(raw_vectors) != len(self.prototypes):
                raise ValueError(
                    "embedder returned an unexpected number of vectors for prototypes"
                )
            vectors = tuple(
                _validate_vector(vector, label=f"prototype vector {index}")
                for index, vector in enumerate(raw_vectors)
            )
            if any(len(vector) != self.embedding_dimension for vector in vectors):
                raise ValueError("prototype vectors must match the configured dimension")
            self._prototype_vectors = vectors
            return vectors

    def score(self, candidate: CandidateEnvelope) -> DimensionScore | None:
        """Score one candidate through the same batch implementation."""

        return self.score_many([candidate])[0]

    def _aggregate(
        self,
        scores: Sequence[float],
        *,
        roles: Sequence[PrototypeRole | None] | None = None,
    ) -> float:
        if not scores or any(not math.isfinite(score) for score in scores):
            raise ValueError("dimension anchor scores must be finite and non-empty")
        if self.aggregation == "max":
            return max(scores)
        if self.aggregation == "top2_mean":
            selected = sorted(scores, reverse=True)[:2]
            return sum(selected) / len(selected)
        if self.aggregation == "role_top2_mean":
            if roles is None or len(roles) != len(scores):
                raise ValueError("role_top2_mean requires one valid role per anchor")
            role_scores: dict[PrototypeRole, float] = {}
            for role, score in zip(roles, scores, strict=True):
                if role not in {"topic", "polarity"}:
                    raise ValueError("role_top2_mean requires topic and polarity anchors")
                role_scores[role] = max(role_scores.get(role, -math.inf), score)
            if set(role_scores) != {"topic", "polarity"}:
                raise ValueError("role_top2_mean requires topic and polarity anchors")
            return sum(role_scores.values()) / 2.0
        return sum(scores) / len(scores)


class TopicGatedStrategy:
    """Two-stage topic-gated dimension scoring.

    Stage 1 ranks dimensions using only ``topic``-role anchors, so generic
    polarity wording such as "I no longer ..." cannot inflate an unrelated
    dimension.  Stage 2 reads the selected dimension's polarity anchors for
    mutation-state diagnostics; those scores never feed the ranking.  The
    provider receives one batch containing safe candidate values followed by
    every anchor description, matching the cost and leakage surface of the
    ensemble strategies.
    """

    def __init__(
        self,
        embedder: Embedder,
        prototypes: Sequence[DimensionPrototype],
        *,
        strategy_version: str = A5_STRATEGY_VERSION,
    ) -> None:
        prototype_tuple = tuple(prototypes)
        if not prototype_tuple:
            raise ValueError("at least one dimension prototype is required")
        if any(not isinstance(prototype, DimensionPrototype) for prototype in prototype_tuple):
            raise TypeError("topic-gated prototypes must be DimensionPrototype values")
        _require_text(strategy_version, name="strategy_version")

        anchors_by_dimension: dict[str, list[int]] = {}
        for index, prototype in enumerate(prototype_tuple):
            anchors_by_dimension.setdefault(prototype.name, []).append(index)
        missing_topic = sorted(
            name
            for name, indices in anchors_by_dimension.items()
            if not any(prototype_tuple[index].role == "topic" for index in indices)
        )
        if missing_topic:
            raise ValueError(
                "topic-gated strategy requires a topic anchor for every dimension; "
                f"missing: {', '.join(missing_topic)}"
            )

        self.embedder = embedder
        self.embedding_dimension = _embedder_dimension(embedder)
        self.prototypes = prototype_tuple
        self.anchors_by_dimension = {
            name: tuple(indices) for name, indices in anchors_by_dimension.items()
        }
        self.strategy_version = strategy_version

    def score_many_detailed(
        self,
        candidates: Sequence[CandidateEnvelope],
    ) -> list[TopicGatedScore | None]:
        """Return topic-gated ranking plus per-anchor diagnostics per candidate.

        ``EmbeddingError`` is converted into unavailable scores for the whole
        provider batch, matching the ensemble strategies.  Shape and
        finite-value violations raise ``ValueError``.
        """

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
        results: list[TopicGatedScore | None] = [None] * len(candidate_list)
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
        if any(len(vector) != self.embedding_dimension for vector in vectors):
            raise ValueError("embedding vectors must match the configured dimension")

        prototype_vectors = vectors[len(safe_indices) :]
        dimension_names = tuple(self.anchors_by_dimension)
        for result_index, candidate_index in enumerate(safe_indices):
            candidate_vector = vectors[result_index]
            anchor_scores: dict[str, dict[str, float]] = {}
            topic_scores: dict[str, float] = {}
            for name, anchor_indices in self.anchors_by_dimension.items():
                labels: dict[str, float] = {}
                topic_values: list[float] = []
                unlabeled_polarity_count = 0
                for anchor_index in anchor_indices:
                    prototype = self.prototypes[anchor_index]
                    score = _cosine_similarity(
                        candidate_vector, prototype_vectors[anchor_index]
                    )
                    if prototype.role == "topic":
                        label = "topic"
                        topic_values.append(score)
                    elif prototype.polarity is not None:
                        label = prototype.polarity
                    else:
                        label = f"polarity-{unlabeled_polarity_count}"
                        unlabeled_polarity_count += 1
                    labels[label] = score
                anchor_scores[name] = labels
                if topic_values:
                    topic_scores[name] = sum(topic_values) / len(topic_values)

            ranked_dimensions = sorted(
                dimension_names,
                key=lambda name: topic_scores[name],
                reverse=True,
            )
            best_dimension = ranked_dimensions[0]
            best_score = topic_scores[best_dimension]
            if len(ranked_dimensions) > 1:
                second_dimension = ranked_dimensions[1]
                second_score = topic_scores[second_dimension]
            else:
                second_dimension = None
                # A single dimension has no competing dimension.  Zero is a
                # finite neutral baseline, matching the other strategies.
                second_score = 0.0
            margin = best_score - second_score
            if not math.isfinite(margin):
                raise ValueError("dimension score margin must be finite")
            results[candidate_index] = TopicGatedScore(
                dimension_score=DimensionScore(
                    dimension=best_dimension,
                    score=best_score,
                    second_dimension=second_dimension,
                    second_score=second_score,
                    margin=margin,
                    strategy_version=self.strategy_version,
                ),
                anchor_scores=anchor_scores,
            )
        return results

    def score_many(
        self,
        candidates: Sequence[CandidateEnvelope],
    ) -> list[DimensionScore | None]:
        """Return the topic-only dimension score (or ``None``) per candidate."""

        detailed = self.score_many_detailed(candidates)
        return [item.dimension_score if item is not None else None for item in detailed]

    def score(self, candidate: CandidateEnvelope) -> DimensionScore | None:
        """Score one candidate through the same batch implementation."""

        return self.score_many([candidate])[0]


class PolicyRouter:
    """Apply scope, safety, dimension, and ambiguity policy to a proposal."""

    def __init__(
        self,
        *,
        strategy_version: str = STRATEGY_VERSION,
        router_version: str = ROUTER_VERSION,
        allowed_dimensions: Collection[str] = CONTROLLED_DIMENSIONS,
    ) -> None:
        _require_text(strategy_version, name="strategy_version")
        _require_text(router_version, name="router_version")
        dimension_set = frozenset(allowed_dimensions)
        if not dimension_set or any(
            not isinstance(dimension, str) or not dimension.strip() for dimension in dimension_set
        ):
            raise ValueError("allowed_dimensions must contain non-empty strings")
        if not dimension_set <= CONTROLLED_DIMENSIONS:
            raise ValueError("allowed_dimensions must be controlled dimensions")
        self.strategy_version = strategy_version
        self.router_version = router_version
        self.allowed_dimensions = dimension_set

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
            if dimension_score.strategy_version != self.strategy_version:
                return self._decision(
                    action="reject",
                    dimension=None,
                    score=None,
                    margin=None,
                    reason="strategy_version_mismatch",
                )
            dimension = dimension_score.dimension
            if dimension not in self.allowed_dimensions:
                return self._decision(
                    action="reject",
                    dimension=None,
                    score=None,
                    margin=None,
                    reason="unsupported_dimension",
                )
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

        if dimension not in self.allowed_dimensions:
            return self._decision(
                action="reject",
                dimension=None,
                score=score,
                margin=margin,
                reason="unsupported_dimension",
            )

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
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and between 0 and 1")
        return float(value)


__all__ = [
    "A4_STRATEGY_VERSION",
    "A5_STRATEGY_VERSION",
    "ENSEMBLE_STRATEGY_VERSION",
    "ROLE_AWARE_ENSEMBLE_STRATEGY_VERSION",
    "ROUTER_VERSION",
    "STRATEGY_VERSION",
    "CandidateEnvelope",
    "DimensionPrototype",
    "DimensionScore",
    "EnsembleAggregation",
    "PolicyRouter",
    "PrototypeDimensionStrategy",
    "PrototypeEnsembleStrategy",
    "PrototypePolarity",
    "PrototypeRole",
    "ReferenceEnvelope",
    "RouterAction",
    "RouterDecision",
    "TopicGatedScore",
    "TopicGatedStrategy",
]
