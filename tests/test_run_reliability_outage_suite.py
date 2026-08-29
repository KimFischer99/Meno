from __future__ import annotations

import time

from benchmarks import run_reliability_outage_suite as lane
from meno.vector import MemoryVectorStore, VectorDocument
from tests.fakes import TestEmbedder


def test_qdrant_outage_degrades_instead_of_failing() -> None:
    """SPEC requires fallback or degraded retrieval, not an exception."""
    result = lane.qdrant_outage()
    outage = result["outage"]

    assert outage["retrieve_raised"] is None
    assert outage["degraded_flag"] is True
    # Degraded must still serve from canonical: returning nothing would block
    # personalization just as an exception would.
    assert outage["facets_returned"] > 0
    assert outage["health_status"] == "degraded"
    assert outage["health_vector"] == "unavailable"
    assert result["passed"] is True


def test_healthy_vector_is_not_reported_degraded() -> None:
    """Otherwise the outage assertion passes for a permanently degraded service."""
    nominal = lane._vector_scenario(fail=False)

    assert nominal["degraded_flag"] is False
    assert nominal["facets_returned"] > 0
    assert nominal["health_status"] == "ok"


def test_dead_sidecar_never_blocks_the_agent_loop() -> None:
    scenario = lane._exercise_provider(lane._free_port(), "dead_port")

    assert scenario["blocked_calls"] == []
    assert scenario["hermes_blocked"] is False
    for call in scenario["calls"]:
        assert call["raised"] is None, call["call"]


def test_dead_sidecar_still_retains_the_turn() -> None:
    """Non-blocking must not mean silently dropped.

    The durable spool is the difference between "Hermes kept going" and "the user's
    turn was lost", so both halves are asserted together.
    """
    scenario = lane._exercise_provider(lane._free_port(), "dead_port")

    assert scenario["spooled_events_retained"] == 2


def test_hanging_sidecar_is_bounded_by_the_provider_timeout() -> None:
    """A refused connection fails fast; a hang is the real threat to the loop."""
    with lane._server(lane._HangingHandler) as port:
        scenario = lane._exercise_provider(port, "hanging_server")

    assert scenario["hermes_blocked"] is False
    assert scenario["spooled_events_retained"] == 2
    prefetch = next(item for item in scenario["calls"] if item["call"] == "prefetch")
    # Bounded by MENO_PROVIDER_TIMEOUT_SECONDS (0.8s), not by the 30s hang.
    assert prefetch["raised"] is None
    assert prefetch["elapsed_seconds"] < lane.BLOCKING_BUDGET_SECONDS


def test_healthy_sidecar_actually_retrieves() -> None:
    """If this fails, the outage scenarios prove nothing."""
    with lane._server(lane._OkHandler) as port:
        scenario = lane._exercise_provider(port, "healthy_sidecar")

    prefetch = next(item for item in scenario["calls"] if item["call"] == "prefetch")
    assert prefetch["raised"] is None
    assert prefetch["returned_type"] == "str"


def test_timed_reports_a_raise_rather_than_swallowing_it() -> None:
    def boom() -> None:
        raise ConnectionError("down")

    result = lane._timed("boom", boom)

    assert result["raised"] == "ConnectionError"
    assert result["non_blocking"] is False


def test_timed_flags_a_call_that_exceeds_the_budget(monkeypatch) -> None:
    """The budget must be enforced, not merely recorded."""
    monkeypatch.setattr(lane, "BLOCKING_BUDGET_SECONDS", 0.01)

    result = lane._timed("slow", lambda: time.sleep(0.05))

    assert result["raised"] is None
    assert result["within_budget"] is False
    assert result["non_blocking"] is False


def test_failing_store_breaks_only_the_read_path() -> None:
    """Ingestion must keep working so canonical fallback has something to serve."""
    inner = MemoryVectorStore(TestEmbedder(256))
    store = lane._FailingVectorStore(inner)
    store.upsert_many([VectorDocument("c1", "u1", "active", "green tea", None, None)])

    assert store.health() is False
    try:
        store.search("u1", "tea", 8)
    except ConnectionError:
        pass
    else:  # pragma: no cover - the point of the double is that it fails
        raise AssertionError("search should fail")
    assert store.search_calls == 1
    # The write landed in the real store despite the read path being down.
    assert inner.search("u1", "tea", 8)


def test_skipping_the_self_check_cannot_pass_the_lane() -> None:
    report = lane.run_benchmark(self_check=False)

    assert report["self_check"]["ran"] is False
    assert report["passed"] is False


def test_report_records_that_this_is_component_level_evidence() -> None:
    """No Hermes deployment consumes Meno today; the lane must not imply otherwise."""
    report = lane.run_benchmark(self_check=False)

    assert "does not run inside Hermes" in report["notes"]["scope"]
    assert report["gate_charter_items"] == [11]
