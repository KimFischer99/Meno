"""Reliability outage lane (Gate Charter #11).

SPEC :1340-1372 sets two reliability conditions, and the v1.0 No-Go list makes the
first one explicit: "Sidecar outage 阻断 Agent" is a No-Go.

| condition                     | SPEC threshold           |
|-------------------------------|--------------------------|
| Sidecar outage blocks Hermes  | 0%                       |
| Qdrant outage                 | fallback or degraded      |

The two are tested at the layers where they actually happen:

**Sidecar outage** is exercised through `MenoMemoryProvider`, the real Hermes
integration surface, against a dead port and a hanging server. What matters is not
that Meno returns an error but that Hermes keeps going: every provider entry point
must return a usable value, never raise, and never block past its timeout. A turn
recorded during the outage must still reach Meno afterwards — the durable spool is
what makes "non-blocking" different from "silently dropped".

**Qdrant outage** is exercised by failing the vector store inside a live service.
Retrieval must fall back to the canonical store and report `degraded: true`. Both
halves matter: serving facets while claiming to be healthy would hide the outage,
and reporting degraded while returning nothing would block personalization.

`--self-check` (on by default) verifies the harness can observe a real failure:
a provider pointed at a working server must succeed, and a healthy vector store
must report `degraded: false`. Without that, "nothing raised" is indistinguishable
from "nothing was tested" — the failure mode this repo has hit before.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import tempfile
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from meno.api import build_service
from meno.config import Settings
from meno.hermes_plugin import MenoMemoryProvider
from meno.schemas import IngestRequest, RetrieveRequest
from meno.vector import MemoryVectorStore
from tests.fakes import TestEmbedder

# A provider call that takes longer than this has blocked the agent loop, which is
# the No-Go condition itself. The provider's own timeout defaults to 0.8s.
BLOCKING_BUDGET_SECONDS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-self-check", action="store_true")
    return parser.parse_args()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _HangingHandler(BaseHTTPRequestHandler):
    """Accepts the connection, then never answers: the worst outage for a sidecar.

    A refused connection fails fast; a hang is what actually threatens an agent
    loop, so the lane must cover it explicitly.
    """

    def do_POST(self) -> None:
        time.sleep(30)

    def do_GET(self) -> None:
        time.sleep(30)

    def log_message(self, *args: Any) -> None:
        return None


class _OkHandler(BaseHTTPRequestHandler):
    """Minimal healthy sidecar, used only by the self-check."""

    def _respond(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if self.path.startswith("/v1/retrieve"):
            self._respond(
                {
                    "rendered_context": "<user_context>ok</user_context>",
                    "facets": [{"claim_id": "c1"}],
                }
            )
        else:
            self._respond({"accepted": True, "state_revision": 1})

    def do_GET(self) -> None:
        self._respond({"claim_id": "c1"})

    def log_message(self, *args: Any) -> None:
        return None


class _ThreadedServer(ThreadingHTTPServer):
    """Serves each request on a daemon thread.

    The hanging handler sleeps for 30s by design. On a single-threaded server that
    sleep also blocks ``shutdown()``, so tearing the lane down would cost 30s per
    scenario; daemon threads let the still-sleeping handler be abandoned.
    """

    daemon_threads = True


@contextlib.contextmanager
def _server(handler: type[BaseHTTPRequestHandler]):
    server = _ThreadedServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


@contextlib.contextmanager
def _provider(port: int, home: Path):
    provider = MenoMemoryProvider()
    provider._base_url = f"http://127.0.0.1:{port}"
    provider._token = "reliability-lane-token"
    provider.initialize("session-reliability", hermes_home=str(home), user_id="outage-user")
    try:
        yield provider
    finally:
        provider.shutdown()


def _timed(label: str, call) -> dict[str, Any]:
    """Run one provider entry point; record whether it raised or blocked."""
    started = time.perf_counter()
    raised: str | None = None
    result: Any = None
    try:
        result = call()
    except Exception as exc:  # noqa: BLE001 - a raise here is the finding
        raised = type(exc).__name__
    elapsed = time.perf_counter() - started
    return {
        "call": label,
        "raised": raised,
        "elapsed_seconds": round(elapsed, 3),
        "within_budget": elapsed <= BLOCKING_BUDGET_SECONDS,
        # Hermes keeps running only if the call neither raised nor hung.
        "non_blocking": raised is None and elapsed <= BLOCKING_BUDGET_SECONDS,
        "returned_type": type(result).__name__,
    }


def _exercise_provider(port: int, scenario: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"meno-outage-{scenario}-") as directory:
        home = Path(directory)
        with _provider(port, home) as provider:
            calls = [
                _timed("prefetch", lambda: provider.prefetch("What do I prefer?")),
                _timed(
                    "sync_turn",
                    lambda: provider.sync_turn("I prefer tea", "Noted."),
                ),
                _timed(
                    "handle_tool_call.recall",
                    lambda: provider.handle_tool_call("meno_recall", {"query": "tea"}),
                ),
                _timed(
                    "handle_tool_call.audit",
                    lambda: provider.handle_tool_call("meno_audit", {"claim_id": "x"}),
                ),
                _timed("on_pre_compress", lambda: provider.on_pre_compress([])),
                _timed("recall_status", provider.recall_status),
                _timed("system_prompt_block", provider.system_prompt_block),
            ]
            # A turn taken during the outage must survive it. If the spool dropped
            # the event, "non-blocking" would just mean "lost the data".
            spooled = provider._spool.count() if provider._spool is not None else 0
    return {
        "scenario": scenario,
        "calls": calls,
        "blocked_calls": [item["call"] for item in calls if not item["non_blocking"]],
        "spooled_events_retained": spooled,
        "hermes_blocked": any(not item["non_blocking"] for item in calls),
    }


def sidecar_outage() -> dict[str, Any]:
    """Both outage shapes: nothing listening, and something that never answers."""
    dead = _exercise_provider(_free_port(), "dead_port")
    with _server(_HangingHandler) as port:
        hanging = _exercise_provider(port, "hanging_server")
    scenarios = [dead, hanging]
    # SPEC threshold is 0%: no scenario may block, and durable turns must survive.
    blocked = sum(item["hermes_blocked"] for item in scenarios)
    return {
        "charter_item": 11,
        "condition": "sidecar outage blocks Hermes",
        "spec_threshold": "0%",
        "scenarios": scenarios,
        "blocked_scenarios": blocked,
        "blocked_rate": blocked / len(scenarios),
        "durable_spool_retained_turns": all(
            item["spooled_events_retained"] > 0 for item in scenarios
        ),
        "passed": blocked == 0
        and all(item["spooled_events_retained"] > 0 for item in scenarios),
    }


class _FailingVectorStore:
    """Wraps a live store and fails only the read path, like a Qdrant outage.

    Writes keep succeeding so the canonical store still holds the claims: the point
    is that retrieval must fall back to canonical, not that ingestion stops.
    """

    def __init__(self, inner: MemoryVectorStore) -> None:
        self.inner = inner
        self.embedder = inner.embedder
        self.search_calls = 0

    def upsert(self, *args: Any, **kwargs: Any) -> None:
        self.inner.upsert(*args, **kwargs)

    def upsert_many(self, documents: list[Any]) -> None:
        self.inner.upsert_many(documents)

    def search(self, *args: Any, **kwargs: Any):
        self.search_calls += 1
        raise ConnectionError("simulated Qdrant outage")

    def delete_claim(self, claim_id: str) -> None:
        self.inner.delete_claim(claim_id)

    def delete_user(self, user_id: str) -> None:
        self.inner.delete_user(user_id)

    def health(self) -> bool:
        return False

    def close(self) -> None:
        self.inner.close()


def _vector_scenario(*, fail: bool) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="meno-outage-vector-") as directory:
        settings = Settings(
            database_url=f"sqlite:///{Path(directory) / 'meno.sqlite3'}",
            vector_mode="memory",
            embedding_dimension=256,
            worker_poll_seconds=0.01,
        )
        settings.validate()
        embedder = TestEmbedder(256)
        inner = MemoryVectorStore(embedder)
        store = _FailingVectorStore(inner) if fail else inner
        service = build_service(settings, vector_store=store, embedder=embedder)
        try:
            service.ingest(
                IngestRequest.model_validate(
                    {
                        "user_id": "outage-user",
                        "event_id": "outage-event",
                        "occurred_at": datetime(2026, 8, 1, 12, 0, tzinfo=UTC).isoformat(),
                        "source": {
                            "type": "hermes_turn",
                            "profile": "reliability",
                            "session_id": "s",
                        },
                        "content": {"role": "user", "text": "I prefer green tea"},
                        "consent_scope": ["personalization", "task_planning"],
                    }
                ),
                idempotency_key="outage-event",
            )
            while service.process_outbox(limit=100):
                pass
            service.process_projection_outbox(limit=100)

            raised: str | None = None
            response = None
            try:
                response = service.retrieve(
                    RetrieveRequest.model_validate(
                        {
                            "user_id": "outage-user",
                            "purpose": "response_personalization",
                            "context": {
                                "query": "Should I drink tea or coffee?",
                                "task_type": "personalization",
                            },
                            "constraints": {"max_facets": 8, "min_confidence": 0.0},
                        }
                    )
                )
            except Exception as exc:  # noqa: BLE001 - a raise here is the finding
                raised = type(exc).__name__
            health = service.health()
        finally:
            service.close()
    facets = len(response.facets) if response is not None else 0
    return {
        "vector_failing": fail,
        "retrieve_raised": raised,
        "degraded_flag": bool(response.degraded) if response is not None else None,
        "facets_returned": facets,
        "health_status": health["status"],
        "health_vector": health["vector"],
    }


def qdrant_outage() -> dict[str, Any]:
    outage = _vector_scenario(fail=True)
    # Degraded must mean "served from canonical", not "served nothing": returning
    # zero facets would block personalization just as an exception would.
    passed = (
        outage["retrieve_raised"] is None
        and outage["degraded_flag"] is True
        and outage["facets_returned"] > 0
        and outage["health_status"] == "degraded"
    )
    return {
        "charter_item": 11,
        "condition": "Qdrant outage falls back or degrades",
        "spec_threshold": "fallback or degraded retrieval",
        "outage": outage,
        "passed": passed,
    }


def run_self_check() -> list[dict[str, Any]]:
    """Prove the harness can tell working from broken.

    Without this, every assertion in the lane is satisfied by a provider that does
    nothing at all.
    """
    results: list[dict[str, Any]] = []

    with _server(_OkHandler) as port:
        healthy = _exercise_provider(port, "healthy_sidecar")
    prefetch = next(item for item in healthy["calls"] if item["call"] == "prefetch")
    results.append(
        {
            "probe": "healthy_sidecar_succeeds",
            "rationale": (
                "A provider pointed at a working sidecar must actually retrieve. If "
                "this fails, the outage scenarios prove nothing: the provider would "
                "return empty regardless of whether Meno is up."
            ),
            "detail": prefetch,
            "passed": prefetch["raised"] is None and prefetch["returned_type"] == "str",
        }
    )

    nominal = _vector_scenario(fail=False)
    results.append(
        {
            "probe": "healthy_vector_not_degraded",
            "rationale": (
                "A healthy vector store must report degraded=false. Otherwise the "
                "Qdrant assertion is satisfied by a service that always claims to be "
                "degraded."
            ),
            "detail": nominal,
            "passed": nominal["degraded_flag"] is False
            and nominal["facets_returned"] > 0
            and nominal["health_status"] == "ok",
        }
    )
    return results


def run_benchmark(*, self_check: bool = True) -> dict[str, Any]:
    sidecar = sidecar_outage()
    vector = qdrant_outage()
    probes = run_self_check() if self_check else []
    return {
        "benchmark": "Meno reliability outage suite",
        "gate_charter_items": [11],
        "schema_version": "reliability-outage-suite-v1",
        "answer_model_calls": 0,
        "blocking_budget_seconds": BLOCKING_BUDGET_SECONDS,
        "sidecar_outage": sidecar,
        "qdrant_outage": vector,
        "self_check": {
            "ran": self_check,
            "probes": probes,
            "passed": all(item["passed"] for item in probes) if self_check else False,
        },
        "notes": {
            "scope": (
                "Exercises MenoMemoryProvider, the real Hermes integration surface, "
                "against a dead port and a hanging server. It does not run inside "
                "Hermes: no Hermes deployment consumes Meno today (the plugin is "
                "registered but not wired), so this is component-level evidence."
            )
        },
        "passed": (
            sidecar["passed"]
            and vector["passed"]
            and self_check
            and all(item["passed"] for item in probes)
        ),
    }


def main() -> None:
    args = parse_args()
    report = run_benchmark(self_check=not args.no_self_check)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
