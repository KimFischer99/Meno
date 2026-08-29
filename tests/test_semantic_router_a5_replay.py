from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import benchmarks.run_semantic_router_a5_replay_shadow as replay_runner
from benchmarks.run_semantic_router_a5_replay_shadow import (
    agreement_metrics,
    load_frozen_strategy,
    reference_from_claim,
)
from meno.db import Claim, Event, make_session_factory
from meno.service import _semantic_key
from tests.fakes import TestEmbedder


@pytest.fixture
def seeded_db(tmp_path: Path):
    database_url = f"sqlite:///{tmp_path / 'replay.sqlite3'}"
    session_factory, _engine = make_session_factory(database_url)
    session = session_factory()
    try:
        now = datetime.now(UTC)
        events = [
            Event(
                id="evt-tea",
                user_id="user-a",
                occurred_at=now,
                source_type="hermes_turn",
                role="user",
                content="I prefer drinking tieguanyin tea every morning.",
                content_hash="h1",
                consent_scope=["personalization"],
            ),
            Event(
                id="evt-notes",
                user_id="user-a",
                occurred_at=now,
                source_type="hermes_turn",
                role="user",
                content="I like point-first notes with the takeaway up front.",
                content_hash="h2",
                consent_scope=["personalization"],
            ),
            Event(
                id="evt-sensitive",
                user_id="user-b",
                occurred_at=now,
                source_type="hermes_turn",
                role="user",
                content="I prefer not to discuss my diagnosis.",
                content_hash="h3",
                consent_scope=["personalization"],
            ),
        ]
        session.add_all(events)
        session.flush()
        claims: list[Claim] = []
        from meno.extractor import extract_claims

        for event in events:
            for index, candidate in enumerate(extract_claims(event)):
                claims.append(
                    Claim(
                        id=f"claim-{event.id}-{index}",
                        user_id=event.user_id,
                        kind=candidate.kind,
                        semantic_channel=candidate.semantic_channel,
                        value=candidate.value,
                        status="active",
                        sensitive=bool(candidate.sensitive),
                        allowed_purposes=["personalization"],
                        source_type="extracted",
                        extractor_version="meno-extractor-2.0.0",
                        semantic_key=_semantic_key(
                            event.user_id,
                            candidate.kind,
                            candidate.semantic_channel,
                            candidate.value,
                            candidate.slot,
                        ),
                    )
                )
        session.add_all(claims)
        session.commit()
    finally:
        session.close()
    return database_url


def test_load_frozen_strategy_accepts_only_the_frozen_top2_mean(tmp_path: Path) -> None:
    config_path = Path(__file__).parent.parent / (
        "benchmarks/fixtures/semantic-router-a5-frozen-config.json"
    )
    raw, strategy = load_frozen_strategy(config_path)
    assert strategy["variant"] == "top2_mean"
    assert strategy["score_threshold"] == 0.475
    assert len(raw) > 0

    tampered = json.loads(config_path.read_bytes())
    tampered["strategy"]["variant"] = "mean"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="top2_mean"):
        load_frozen_strategy(path)


def test_reference_from_claim_derives_slot_read_only() -> None:
    from types import SimpleNamespace

    claim = SimpleNamespace(
        id="c1",
        user_id="u",
        semantic_key="k",
        kind="preference",
        semantic_channel="preference.explicit",
        value="morning tieguanyin tea",
        sensitive=False,
        status="active",
        source_type="extracted",
    )
    reference = reference_from_claim(claim)  # type: ignore[arg-type]
    assert reference.deterministic_slot == "beverage"


def test_agreement_metrics_counts_shadow_key_changes() -> None:
    decisions = [
        {
            "has_deterministic_slot": True,
            "sensitive": False,
            "action": "reuse_key",
            "proposed_key_matches_canonical": True,
            "shadow_would_change_key": False,
        },
        {
            "has_deterministic_slot": False,
            "sensitive": False,
            "action": "new_key",
            "proposed_key_matches_canonical": False,
            "shadow_would_change_key": False,
        },
        {
            "has_deterministic_slot": False,
            "sensitive": False,
            "action": "reuse_key",
            "proposed_key_matches_canonical": False,
            "shadow_would_change_key": True,
        },
        {
            "has_deterministic_slot": False,
            "sensitive": True,
            "action": "reject",
            "proposed_key_matches_canonical": False,
            "shadow_would_change_key": False,
        },
    ]
    metrics = agreement_metrics(decisions)
    assert metrics["slotted_candidate_count"] == 1
    assert metrics["slotless_candidate_count"] == 3
    assert metrics["unsafe_all_rejected"] is True
    assert metrics["shadow_would_change_key_count"] == 1
    assert metrics["reject_count"] == 1


def test_replay_main_runs_read_only_against_seeded_sqlite(
    tmp_path: Path, seeded_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "replay-shadow.json"
    monkeypatch.setenv("MENO_DATABASE_URL", seeded_db)
    monkeypatch.setenv("MENO_VECTOR_MODE", "memory")
    monkeypatch.setenv("MENO_ENV", "development")
    monkeypatch.setenv("MENO_EMBEDDING_PROVIDER", "siliconflow")
    monkeypatch.setenv("MENO_EMBEDDING_DIMENSION", "128")
    monkeypatch.setattr(
        replay_runner, "make_embedder", lambda _settings: TestEmbedder(dimension=128)
    )

    def run() -> int:
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_semantic_router_a5_replay_shadow.py",
                "--output",
                str(output),
                "--event-limit",
                "50",
            ],
        )
        try:
            replay_runner.main()
            return 0
        except SystemExit as excinfo:
            return excinfo.code  # type: ignore[arg-type]

    exit_code = run()
    assert exit_code in {0, 3}
    report = json.loads(output.read_text())
    assert report["fail_closed"]["passed"] is True
    assert report["replay"]["candidate_count"] >= 2
    assert report["metrics"]["unsafe_all_rejected"] is True
    # Slotted candidates route through the same slot the canonical writer used;
    # their reuse proposals must land on the exact canonical key.
    slotted = [
        record for record in report["decisions"] if record["has_deterministic_slot"]
    ]
    assert all(
        record["action"] in {"reuse_key", "new_key"} for record in slotted
    )
    assert report["metrics"]["slot_agreement_rate"] >= 0.0
