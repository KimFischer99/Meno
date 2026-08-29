from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from benchmarks.run_semantic_router_a4_development import _contains_raw_artifact_text
from benchmarks.run_semantic_router_a5_development import (
    DEFAULT_PROTOTYPES,
    EXPECTED_PROTOTYPE_SHA256,
    load_prototypes_a5,
)
from benchmarks.run_semantic_router_shadow import (
    _git_head,
    _package_versions,
    _safe_endpoint,
    _sha256,
)
from benchmarks.verify_invariants import audit_chain_summary
from meno.config import Settings
from meno.db import AuditEvent, Claim, Event
from meno.extractor import extract_claims, preference_slot
from meno.semantic_router import (
    ENSEMBLE_STRATEGY_VERSION,
    ROUTER_VERSION,
    CandidateEnvelope,
    DimensionScore,
    PolicyRouter,
    PrototypeEnsembleStrategy,
    ReferenceEnvelope,
    RouterDecision,
)
from meno.service import _semantic_key
from meno.vector import EmbeddingError, make_embedder

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FROZEN_CONFIG = ROOT / "benchmarks" / "fixtures" / "semantic-router-a5-frozen-config.json"
SOURCE_DEPENDENCIES = {
    "a5_development_runner": ROOT / "benchmarks" / "run_semantic_router_a5_development.py",
    "a5_holdout_runner": ROOT / "benchmarks" / "run_semantic_router_a5_holdout.py",
    "invariants_runner": ROOT / "benchmarks" / "verify_invariants.py",
    "shadow_runner": ROOT / "benchmarks" / "run_semantic_router_shadow.py",
    "config_module": ROOT / "src" / "meno" / "config.py",
    "db_module": ROOT / "src" / "meno" / "db.py",
    "extractor_module": ROOT / "src" / "meno" / "extractor.py",
    "service_module": ROOT / "src" / "meno" / "service.py",
    "vector_module": ROOT / "src" / "meno" / "vector.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "A5.4 read-only integration shadow: replay stored events through the "
            "frozen A5 policy beside the canonical deterministic writer"
        )
    )
    parser.add_argument("--frozen-config", type=Path, default=DEFAULT_FROZEN_CONFIG)
    parser.add_argument("--prototypes", type=Path, default=DEFAULT_PROTOTYPES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-limit", type=int, default=500)
    return parser.parse_args()


def load_frozen_strategy(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    strategy = payload.get("strategy") if isinstance(payload, dict) else None
    if not isinstance(strategy, dict) or strategy.get("variant") != "top2_mean":
        raise ValueError("replay shadow requires the frozen A5 top2_mean strategy")
    if strategy.get("threshold_search_allowed_after_freeze") is not False:
        raise ValueError("frozen config must disable post-freeze threshold search")
    return raw, strategy


def read_only_session_factory(database_url: str) -> sessionmaker:
    """Open the canonical store without running any DDL or writes."""

    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def reference_from_claim(claim: Claim) -> ReferenceEnvelope:
    """Derive the routing slot read-only from the stored claim value.

    The Claim table has no persisted slot column (tracked as a Phase B P1
    schema item); deriving it from the value keeps this replay write-free.
    """

    return ReferenceEnvelope(
        claim_id=claim.id,
        user_id=claim.user_id,
        semantic_key=claim.semantic_key,
        kind=claim.kind,
        semantic_channel=claim.semantic_channel,
        value=claim.value,
        sensitive=bool(claim.sensitive),
        status=claim.status,
        source_type=claim.source_type,
        deterministic_slot=preference_slot(claim.value),
    )


class _ReplayEmbedder:
    """Serve precomputed vectors for the ensemble's fixed batch contract.

    ``PrototypeEnsembleStrategy`` always embeds one batch of safe candidate
    values followed by every anchor description, so this double replays
    exactly those precomputed vectors without contacting the provider.
    """

    def __init__(
        self,
        candidate_vectors: list[list[float]],
        anchor_vectors: list[list[float]],
        dimension: int,
    ) -> None:
        self._candidate_vectors = candidate_vectors
        self._anchor_vectors = anchor_vectors
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        expected = len(self._candidate_vectors) + len(self._anchor_vectors)
        if len(texts) != expected:
            raise ValueError("replay embedder received an unexpected batch shape")
        return [*self._candidate_vectors, *self._anchor_vectors]

    def embed_query(self, text: str) -> list[float]:
        raise ValueError("replay embedder does not serve queries")

    def close(self) -> None:
        return None


def main() -> None:
    args = parse_args()
    config_raw, strategy = load_frozen_strategy(args.frozen_config)
    prototype_raw, prototype_version, prototypes, prototype_metadata = load_prototypes_a5(
        args.prototypes
    )
    if hashlib.sha256(prototype_raw).hexdigest() != EXPECTED_PROTOTYPE_SHA256:
        raise ValueError("replay shadow prototype input changed")

    settings = Settings.from_env()
    session_factory = read_only_session_factory(settings.database_url)
    session = session_factory()
    try:
        claims = session.scalars(select(Claim)).all()
        events = (
            session.scalars(select(Event).order_by(Event.created_at).limit(args.event_limit))
            .all()
        )
        audit_rows = session.execute(select(AuditEvent.__table__)).mappings().all()

        references = [reference_from_claim(claim) for claim in claims]

        replay_candidates: list[dict[str, Any]] = []
        for event in events:
            for candidate_index, candidate in enumerate(extract_claims(event)):
                canonical_key = _semantic_key(
                    event.user_id,
                    candidate.kind,
                    candidate.semantic_channel,
                    candidate.value,
                    candidate.slot,
                )
                replay_candidates.append(
                    {
                        "event_id": event.id,
                        "candidate_index": candidate_index,
                        "user_id": event.user_id,
                        "kind": candidate.kind,
                        "semantic_channel": candidate.semantic_channel,
                        "value": candidate.value,
                        "sensitive": bool(candidate.sensitive),
                        "slot": candidate.slot,
                        "canonical_key": canonical_key,
                        "envelope": CandidateEnvelope(
                            user_id=event.user_id,
                            kind=candidate.kind,
                            semantic_channel=candidate.semantic_channel,
                            value=candidate.value,
                            sensitive=bool(candidate.sensitive),
                            injection_detected=False,
                            deterministic_slot=candidate.slot,
                        ),
                    }
                )

        scored_indices = [
            index
            for index, item in enumerate(replay_candidates)
            if item["envelope"].deterministic_slot is None
            and not item["envelope"].sensitive
            and not item["envelope"].injection_detected
        ]
        texts = [replay_candidates[index]["value"] for index in scored_indices]
        texts.extend(prototype.description for prototype in prototypes)

        provider_failed = False
        embed_started = time.perf_counter()
        embedder = make_embedder(settings)
        try:
            raw_vectors = embedder.embed_documents(texts)
        except EmbeddingError:
            provider_failed = True
            raw_vectors = []
        finally:
            embedder.close()
        embed_seconds = time.perf_counter() - embed_started

        scores: dict[int, DimensionScore | None] = {}
        if provider_failed:
            # Fail-closed: every designed candidate abstains via provider_unavailable.
            decisions_reason_override = "provider_unavailable"
        else:
            decisions_reason_override = None
            from meno.semantic_router import _validate_vector

            expected_count = len(scored_indices) + len(prototypes)
            if len(raw_vectors) != expected_count:
                raise ValueError("embedder returned an unexpected number of vectors")
            vectors = [
                _validate_vector(vector, label=f"vector {position}")
                for position, vector in enumerate(raw_vectors)
            ]
            if any(len(vector) != settings.embedding_dimension for vector in vectors):
                raise ValueError("embedding vectors must match the configured dimension")
            candidate_vectors = vectors[: len(scored_indices)]
            anchor_vectors = vectors[len(scored_indices):]
            replay_embedder = _ReplayEmbedder(
                candidate_vectors,
                anchor_vectors,
                settings.embedding_dimension,
            )
            ensemble = PrototypeEnsembleStrategy(
                replay_embedder,
                prototypes,
                aggregation=strategy["aggregation"],
            )
            local_scores = ensemble.score_many(
                [replay_candidates[index]["envelope"] for index in scored_indices]
            )
            scores = dict(zip(scored_indices, local_scores, strict=True))

        router = PolicyRouter(strategy_version=strategy["strategy_version"])
        route_started = time.perf_counter()
        decision_records: list[dict[str, Any]] = []
        for index, item in enumerate(replay_candidates):
            decision: RouterDecision | None = None
            if decisions_reason_override is None:
                decision = router.decide(
                    item["envelope"],
                    references,
                    scores.get(index),
                    strategy["score_threshold"],
                    strategy["margin_threshold"],
                )
            else:
                decision = router.decide(
                    item["envelope"], references, None, strategy["score_threshold"],
                    strategy["margin_threshold"],
                )
            decision_records.append(_decision_record(item, decision))
        route_seconds = time.perf_counter() - route_started

        probe_envelope = CandidateEnvelope(
            user_id="shadow-probe",
            kind="preference",
            semantic_channel="preference.explicit",
            value="probe-value",
            sensitive=False,
            injection_detected=False,
            deterministic_slot=None,
        )
        fail_closed_decision = router.decide(
            probe_envelope,
            references,
            None,
            strategy["score_threshold"],
            strategy["margin_threshold"],
        )
        fail_closed_ok = bool(
            fail_closed_decision.action == "new_key"
            and fail_closed_decision.reason == "provider_unavailable"
        )

        metrics = agreement_metrics(decision_records)
        audit_summary = audit_chain_summary(audit_rows)
        event_count = len(events)
        claim_count = len(claims)
    finally:
        session.close()

    report: dict[str, Any] = {
        "benchmark": "Meno A5.4 read-only integration replay shadow",
        "mode": "isolated-replay-shadow-read-only",
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "git_head": _git_head(),
        "command": [str(Path(__file__).resolve()), *sys.argv[1:]],
        "frozen_config": {
            "path": str(args.frozen_config),
            "sha256": hashlib.sha256(config_raw).hexdigest(),
            "strategy_version": strategy["strategy_version"],
            "aggregation": strategy["aggregation"],
            "score_threshold": strategy["score_threshold"],
            "margin_threshold": strategy["margin_threshold"],
        },
        "prototypes": {
            "version": prototype_version,
            "sha256": hashlib.sha256(prototype_raw).hexdigest(),
            **prototype_metadata,
        },
        "source": {
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "router_sha256": _sha256(ROOT / "src" / "meno" / "semantic_router.py"),
            "router_version": ROUTER_VERSION,
            "ensemble_strategy_version": ENSEMBLE_STRATEGY_VERSION,
            "dependency_sha256": {
                name: _sha256(path) for name, path in SOURCE_DEPENDENCIES.items()
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "service_config": {
            "environment": settings.environment,
            "extractor_version": settings.extractor_version,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "embedding_projection_version": settings.embedding_projection_version,
            "embedding_endpoint": _safe_endpoint(
                settings.google_api_base_url
                if settings.embedding_provider == "google"
                else settings.openai_base_url
            ),
            "provider_failed_during_replay": provider_failed,
        },
        "replay": {
            "event_limit": args.event_limit,
            "event_count": event_count,
            "claim_count_total": claim_count,
            "reference_count_total": len(references),
            "active_reference_count": sum(r.status == "active" for r in references),
            "candidate_count": len(replay_candidates),
            "scored_candidate_count": len(scored_indices),
            "audit_rows_checked": len(audit_rows),
        },
        "latency": {
            "embed_seconds_total": round(embed_seconds, 6),
            "route_seconds_total": round(route_seconds, 6),
            "route_seconds_per_candidate": round(
                route_seconds / max(1, len(decision_records)), 9
            ),
        },
        "fail_closed": {
            "probe_action": fail_closed_decision.action,
            "probe_reason": fail_closed_decision.reason,
            "passed": fail_closed_ok,
        },
        "audit_schema_summary": audit_summary,
        "metrics": metrics,
        "decisions": decision_records,
        "raw_text_persisted": False,
    }
    raw_values = tuple(item["value"] for item in replay_candidates) + tuple(
        reference.value for reference in references
    )
    if _contains_raw_artifact_text(report, raw_values):
        raise ValueError("replay shadow artifact would contain raw candidate or reference text")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "replay": report["replay"],
                "metrics": metrics,
                "fail_closed": report["fail_closed"],
                "latency": report["latency"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not fail_closed_ok:
        raise SystemExit(3)


def _decision_record(item: dict[str, Any], decision: RouterDecision) -> dict[str, Any]:
    """Audit-schema record: identifiers and scores only, never raw values."""

    has_slot = item["slot"] is not None
    proposed_matches_canonical = decision.proposed_semantic_key == item["canonical_key"]
    record: dict[str, Any] = {
        "event_id": item["event_id"],
        "candidate_index": item["candidate_index"],
        "kind": item["kind"],
        "semantic_channel": item["semantic_channel"],
        "has_deterministic_slot": has_slot,
        "sensitive": item["sensitive"],
        "action": decision.action,
        "proposed_key_matches_canonical": proposed_matches_canonical,
        "matched_claim_known": decision.matched_claim_id is not None,
        "dimension": decision.dimension,
        "score": decision.score,
        "margin": decision.margin,
        "reason": decision.reason,
        "protected_reference": decision.protected_reference,
        "shadow_would_change_key": bool(
            has_slot is False
            and decision.action == "reuse_key"
            and not proposed_matches_canonical
        ),
    }
    return record


def agreement_metrics(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    slotted = [item for item in decisions if item["has_deterministic_slot"]]
    slotless = [item for item in decisions if not item["has_deterministic_slot"]]
    slot_agreement = (
        sum(item["proposed_key_matches_canonical"] or item["action"] == "new_key" for item in slotted)
        / len(slotted)
        if slotted
        else 1.0
    )
    unsafe = [item for item in decisions if item["sensitive"]]
    unsafe_ok = all(item["action"] == "reject" for item in unsafe) if unsafe else True
    return {
        "slotted_candidate_count": len(slotted),
        "slotless_candidate_count": len(slotless),
        "unsafe_candidate_count": len(unsafe),
        "slot_agreement_rate": round(slot_agreement, 6),
        "unsafe_all_rejected": unsafe_ok,
        "shadow_merge_proposal_count": sum(
            item["action"] == "reuse_key" and not item["proposed_key_matches_canonical"]
            for item in decisions
        ),
        "shadow_would_change_key_count": sum(
            item["shadow_would_change_key"] for item in decisions
        ),
        "reject_count": sum(item["action"] == "reject" for item in decisions),
        "new_key_count": sum(item["action"] == "new_key" for item in decisions),
        "reuse_key_count": sum(item["action"] == "reuse_key" for item in decisions),
    }


if __name__ == "__main__":
    main()
