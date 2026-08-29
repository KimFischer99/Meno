from __future__ import annotations

from benchmarks.run_semantic_shadow import (
    DEFAULT_FIXTURE,
    DEFAULT_THRESHOLDS,
    load_fixture,
    recall_frontier,
    recommend_threshold,
    threshold_metrics,
)
from meno.extractor import preference_slot
from meno.semantic_resolver import (
    SemanticReference,
    ShadowSemanticResolver,
    cosine_similarity,
)
from meno.vector import EmbeddingError


class StaticEmbedder:
    dimension = 2

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self.vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vectors[text]

    def close(self) -> None:
        return None


class FailingEmbedder(StaticEmbedder):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        raise EmbeddingError("provider unavailable")


def test_shadow_resolver_proposes_best_reference_without_writing() -> None:
    embedder = StaticEmbedder(
        {
            "brief replies": [1.0, 0.0],
            "answer style": [0.9, 0.1],
            "beverage": [0.0, 1.0],
        }
    )
    references = [
        SemanticReference("preference:answer_style", "answer style"),
        SemanticReference("preference:beverage", "beverage"),
    ]

    proposal = ShadowSemanticResolver(embedder, threshold=0.9).propose("brief replies", references)

    assert proposal.proposed_merge is True
    assert proposal.proposed_semantic_key == "preference:answer_style"
    assert proposal.reason == "threshold_match"
    assert references[0].value == "answer style"
    assert embedder.calls == [["brief replies", "answer style", "beverage"]]


def test_shadow_resolver_fails_closed_below_threshold_or_without_reference() -> None:
    embedder = StaticEmbedder({"candidate": [1.0, 0.0], "reference": [0.5, 0.5]})
    resolver = ShadowSemanticResolver(embedder, threshold=0.9)

    proposal = resolver.propose("candidate", [SemanticReference("preference:other", "reference")])
    empty = resolver.propose("candidate", [])

    assert proposal.proposed_merge is False
    assert proposal.proposed_semantic_key is None
    assert proposal.reason == "below_threshold"
    assert empty.proposed_merge is False
    assert empty.reason == "no_reference"


def test_shadow_resolver_blocks_sensitive_input_before_embedding() -> None:
    embedder = StaticEmbedder({})
    resolver = ShadowSemanticResolver(embedder, threshold=0.9)

    proposal = resolver.propose(
        "medical diagnosis",
        [SemanticReference("preference:answer_style", "concise answers")],
        candidate_sensitive=True,
    )

    assert proposal.proposed_merge is False
    assert proposal.reason == "sensitive_input"
    assert embedder.calls == []


def test_shadow_resolver_fails_closed_when_provider_is_unavailable() -> None:
    embedder = FailingEmbedder({})

    proposal = ShadowSemanticResolver(embedder, threshold=0.9).propose(
        "candidate", [SemanticReference("preference:other", "reference")]
    )

    assert proposal.proposed_merge is False
    assert proposal.reason == "provider_unavailable"


def test_shadow_resolver_rejects_tied_references() -> None:
    embedder = StaticEmbedder({"candidate": [1.0, 0.0], "first": [1.0, 0.0], "second": [1.0, 0.0]})

    proposal = ShadowSemanticResolver(embedder, threshold=0.9).propose(
        "candidate",
        [
            SemanticReference("preference:first", "first"),
            SemanticReference("preference:second", "second"),
        ],
    )

    assert proposal.proposed_merge is False
    assert proposal.reason == "ambiguous_reference"
    assert proposal.proposed_semantic_key is None


def test_cosine_similarity_validates_dimensions() -> None:
    assert cosine_similarity([2.0, 0.0], [4.0, 0.0]) == 1.0
    try:
        cosine_similarity([1.0], [1.0, 0.0])
    except ValueError as exc:
        assert "same non-zero dimension" in str(exc)
    else:
        raise AssertionError("dimension mismatch should fail")


def test_threshold_calibration_requires_zero_false_merges() -> None:
    pairs = [
        {"expected_merge": True, "score": 0.91},
        {"expected_merge": True, "score": 0.80},
        {"expected_merge": False, "score": 0.84},
        {"expected_merge": False, "score": 0.20},
    ]
    metrics = [threshold_metrics(pairs, value) for value in (0.80, 0.85, 0.95)]

    recommendation = recommend_threshold(metrics, minimum_recall=0.5)

    assert recommendation is not None
    assert recommendation["threshold"] == 0.85
    assert recommendation["false_positive"] == 0
    assert recommendation["recall"] == 0.5


def test_threshold_recommendation_prefers_safer_high_threshold() -> None:
    metrics = [
        {"threshold": 0.5, "false_positive": 0, "recall": 1.0},
        {"threshold": 0.9, "false_positive": 0, "recall": 0.5},
    ]

    recommendation = recommend_threshold(metrics, minimum_recall=0.5)

    assert recommendation is metrics[1]


def test_recall_frontier_reports_least_unsafe_fallback_without_recommending_it() -> None:
    pairs = [
        {"expected_merge": True, "score": 0.91},
        {"expected_merge": True, "score": 0.80},
        {"expected_merge": False, "score": 0.92},
    ]
    metrics = [threshold_metrics(pairs, value) for value in (0.80, 0.90, 0.95)]

    frontier = recall_frontier(metrics, minimum_recall=0.5)

    assert frontier is not None
    assert frontier["threshold"] == 0.8
    assert frontier["false_positive"] == 1
    assert recommend_threshold(metrics, minimum_recall=0.5) is None


def test_frozen_fixture_contains_only_currently_slotless_values() -> None:
    _, fixture = load_fixture(DEFAULT_FIXTURE)

    assert len(fixture["pairs"]) >= 30
    assert all(preference_slot(pair["candidate"]) is None for pair in fixture["pairs"])
    assert fixture["scope"]["reference_status"] == "active"
    assert sum(not pair.get("provider_allowed", True) for pair in fixture["pairs"]) == 2


def test_default_threshold_grid_includes_fail_closed_upper_bound() -> None:
    assert DEFAULT_THRESHOLDS[-1] == 1.0


def test_threshold_metrics_treats_provider_block_as_fail_closed() -> None:
    metrics = threshold_metrics(
        [
            {"expected_merge": False, "score": None},
            {"expected_merge": True, "score": 0.9},
        ],
        0.8,
    )

    assert metrics["true_negative"] == 1
    assert metrics["true_positive"] == 1
