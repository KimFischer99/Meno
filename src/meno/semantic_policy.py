from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import Settings
from .semantic_router import (
    CandidateEnvelope,
    DimensionPrototype,
    DimensionScore,
    PolicyRouter,
    PrototypeEnsembleStrategy,
    ReferenceEnvelope,
    RouterDecision,
)
from .vector import Embedder

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
PHASE_B_CONFIG_SHA256 = (
    "c67d3d0b1b279116910219d53be1b99ed053397eb8e739b56a9912711f9e73c9"
)
PHASE_B_CONFIG_VERSION = "semantic-router-phaseb-frozen-config-v1"
PHASE_B_STRATEGY_VERSION = "meno-prototype-ensemble-2.0.0"
PHASE_B_PROTOTYPE_SHA256 = (
    "d9e97ec67b5688b44728a1b2bcf73cc39d7625ef1a7baf39b83862cae7a136dc"
)
PHASE_B_SCORE_THRESHOLD = 0.475
PHASE_B_MARGIN_THRESHOLD = 0.005
PHASE_B_PROVIDER = {
    "name": "siliconflow",
    "model": "BAAI/bge-m3",
    "dimension": 1024,
    "projection_version": "siliconflow-bge-m3-1024-v1",
}


class FrozenSemanticRoutingPolicy:
    """Validated Phase B runtime wrapper around the frozen A5 policy."""

    def __init__(
        self,
        *,
        strategy: PrototypeEnsembleStrategy,
        router: PolicyRouter,
        score_threshold: float,
        margin_threshold: float,
        config_sha256: str,
        prototype_sha256: str,
    ) -> None:
        self.strategy = strategy
        self.router = router
        self.score_threshold = score_threshold
        self.margin_threshold = margin_threshold
        self.config_sha256 = config_sha256
        self.prototype_sha256 = prototype_sha256

    @classmethod
    def from_settings(
        cls, settings: Settings, embedder: Embedder
    ) -> FrozenSemanticRoutingPolicy:
        config_path = _resolve_config_path(settings.semantic_router_config_file)
        config_raw = config_path.read_bytes()
        config_sha256 = hashlib.sha256(config_raw).hexdigest()
        if settings.semantic_router_config_sha256 != PHASE_B_CONFIG_SHA256:
            raise ValueError("Phase B requires the pinned semantic router config SHA-256")
        if config_sha256 != PHASE_B_CONFIG_SHA256:
            raise ValueError("semantic router frozen config SHA-256 mismatch")
        payload = json.loads(config_raw)
        if not isinstance(payload, dict):
            raise TypeError("semantic router frozen config must be an object")
        provider = payload.get("provider")
        strategy_config = payload.get("strategy")
        if not isinstance(provider, dict) or not isinstance(strategy_config, dict):
            raise TypeError("semantic router frozen config is incomplete")
        if payload.get("config_version") != PHASE_B_CONFIG_VERSION:
            raise ValueError("unsupported semantic router frozen config version")
        if payload.get("source_checkpoint") != "stage4-phase-a5":
            raise ValueError("semantic router source checkpoint mismatch")
        expected_provider = {
            "name": settings.embedding_provider,
            "model": settings.embedding_model,
            "dimension": settings.embedding_dimension,
            "projection_version": settings.embedding_projection_version,
        }
        if provider != expected_provider:
            raise ValueError("semantic router provider pin does not match service settings")
        if provider != PHASE_B_PROVIDER:
            raise ValueError("Phase B provider pin was modified")
        if strategy_config.get("variant") != "top2_mean":
            raise ValueError("Phase B requires the frozen top2_mean strategy")
        if strategy_config.get("aggregation") != "top2_mean":
            raise ValueError("Phase B requires top2_mean aggregation")
        if strategy_config.get("strategy_version") != PHASE_B_STRATEGY_VERSION:
            raise ValueError("Phase B strategy version was modified")
        if strategy_config.get("score_threshold") != PHASE_B_SCORE_THRESHOLD:
            raise ValueError("Phase B score threshold was modified")
        if strategy_config.get("margin_threshold") != PHASE_B_MARGIN_THRESHOLD:
            raise ValueError("Phase B margin threshold was modified")
        if strategy_config.get("threshold_search_allowed_after_freeze") is not False:
            raise ValueError("post-freeze threshold search must remain disabled")

        prototype_path = _resolve_prototype_path(
            config_path, strategy_config.get("prototype_fixture")
        )
        prototype_raw = prototype_path.read_bytes()
        prototype_sha256 = hashlib.sha256(prototype_raw).hexdigest()
        if prototype_sha256 != PHASE_B_PROTOTYPE_SHA256 or (
            prototype_sha256 != strategy_config.get("prototype_sha256")
        ):
            raise ValueError("semantic router prototype SHA-256 mismatch")
        prototypes = _load_prototypes(prototype_raw)
        strategy_version = _require_text(
            strategy_config.get("strategy_version"), "strategy_version"
        )
        score_threshold = _require_probability(
            strategy_config.get("score_threshold"), "score_threshold"
        )
        margin_threshold = _require_probability(
            strategy_config.get("margin_threshold"), "margin_threshold"
        )
        strategy = PrototypeEnsembleStrategy(
            embedder,
            prototypes,
            aggregation="top2_mean",
            strategy_version=strategy_version,
            cache_prototypes=False,
        )
        router = PolicyRouter(
            strategy_version=strategy_version,
            allowed_dimensions=CONTROLLED_DIMENSIONS,
        )
        return cls(
            strategy=strategy,
            router=router,
            score_threshold=score_threshold,
            margin_threshold=margin_threshold,
            config_sha256=config_sha256,
            prototype_sha256=prototype_sha256,
        )

    def score_many(
        self, candidates: Sequence[CandidateEnvelope]
    ) -> list[DimensionScore | None]:
        return self.strategy.score_many(candidates)

    def decide(
        self,
        candidate: CandidateEnvelope,
        references: Sequence[ReferenceEnvelope],
        score: DimensionScore | None,
    ) -> RouterDecision:
        return self.router.decide(
            candidate,
            references,
            score,
            self.score_threshold,
            self.margin_threshold,
        )


def _resolve_prototype_path(config_path: Path, value: object) -> Path:
    relative = Path(_require_text(value, "prototype_fixture"))
    if relative.is_absolute():
        return relative
    cwd_path = relative.resolve()
    if cwd_path.is_file():
        return cwd_path
    sibling = config_path.parent / relative.name
    if sibling.is_file():
        return sibling.resolve()
    raise FileNotFoundError(f"semantic router prototype fixture not found: {relative}")


def _resolve_config_path(value: str) -> Path:
    configured = Path(value).expanduser()
    if configured.is_file():
        return configured.resolve()
    packaged = Path(__file__).resolve().parent / "data" / configured.name
    if packaged.is_file():
        return packaged
    raise FileNotFoundError(f"semantic router frozen config not found: {configured}")


def _load_prototypes(raw: bytes) -> tuple[DimensionPrototype, ...]:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("prototype_version") != (
        "semantic-prototypes-a5-v2"
    ):
        raise ValueError("unsupported semantic router prototype fixture")
    dimensions = payload.get("dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != CONTROLLED_DIMENSIONS:
        raise ValueError("semantic router prototypes must contain seven controlled dimensions")
    prototypes: list[DimensionPrototype] = []
    for dimension in sorted(dimensions):
        anchors = dimensions[dimension]
        if not isinstance(anchors, list) or len(anchors) != 4:
            raise ValueError(f"prototype dimension {dimension} requires four anchors")
        seen_ids: set[str] = set()
        for index, anchor in enumerate(anchors):
            if not isinstance(anchor, dict):
                raise TypeError(f"prototype {dimension}[{index}] must be an object")
            anchor_id = _require_text(anchor.get("id"), "prototype id")
            role = _require_text(anchor.get("role"), "prototype role")
            polarity = _require_text(anchor.get("polarity"), "prototype polarity")
            description = _require_text(anchor.get("text"), "prototype text")
            expected_role = "topic" if anchor_id == "topic" else "polarity"
            expected_polarity = "neutral" if anchor_id == "topic" else anchor_id
            if role != expected_role or polarity != expected_polarity:
                raise ValueError(f"prototype {dimension}[{index}] metadata is inconsistent")
            if anchor_id in seen_ids:
                raise ValueError(f"duplicate prototype id for {dimension}: {anchor_id}")
            seen_ids.add(anchor_id)
            prototypes.append(
                DimensionPrototype(
                    name=dimension,
                    description=description,
                    role=role,  # type: ignore[arg-type]
                    polarity=polarity,  # type: ignore[arg-type]
                )
            )
        if seen_ids != {"topic", "affirmed", "negated", "contradiction"}:
            raise ValueError(f"prototype dimension {dimension} has incomplete anchors")
    return tuple(prototypes)


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _require_probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    resolved = float(value)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return resolved
