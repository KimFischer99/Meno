from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

try:  # Available inside Hermes Agent.
    from agent.memory_provider import MemoryProvider, RecallStatus
except ImportError:  # Allows standalone Meno packaging and tests.
    class MemoryProvider:  # type: ignore[no-redef]
        pass

    @dataclass(frozen=True)
    class RecallStatus:  # type: ignore[no-redef]
        provider_label: str
        count: int
        glyph: str = "🧠"


log = logging.getLogger(__name__)


AUDIT_SCHEMA = {
    "name": "meno_audit",
    "description": "Explain the evidence and validity behind a recalled Meno claim.",
    "parameters": {
        "type": "object",
        "properties": {"claim_id": {"type": "string"}},
        "required": ["claim_id"],
    },
}

RECALL_SCHEMA = {
    "name": "meno_recall",
    "description": "Recall evidence-backed personal context from Meno.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_facets": {"type": "integer", "minimum": 1, "maximum": 16},
        },
        "required": ["query"],
    },
}


class DurableSpool:
    def __init__(self, path: Path, max_pending: int = 5000) -> None:
        self.path = path
        self.max_pending = max_pending
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with closing(self._connect()) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS pending ("
                "id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at REAL NOT NULL)"
            )
            connection.commit()
        self.path.chmod(0o600)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def enqueue(self, payload: dict[str, Any]) -> bool:
        with closing(self._connect()) as connection:
            count = connection.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
            if count >= self.max_pending:
                return False
            connection.execute(
                "INSERT OR IGNORE INTO pending(id, payload, created_at) VALUES (?, ?, ?)",
                (payload["event_id"], json.dumps(payload, ensure_ascii=False), time.time()),
            )
            connection.commit()
        return True

    def peek(self) -> tuple[str, dict[str, Any]] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT id, payload FROM pending ORDER BY created_at LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return row[0], json.loads(row[1])

    def complete(self, event_id: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute("DELETE FROM pending WHERE id = ?", (event_id,))
            connection.commit()

    def count(self) -> int:
        with closing(self._connect()) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM pending").fetchone()[0])


class MenoMemoryProvider(MemoryProvider):
    def __init__(self) -> None:
        self._base_url = os.environ.get("MENO_API_URL", "http://127.0.0.1:8765").rstrip("/")
        self._token = os.environ.get("MENO_API_TOKEN", "")
        self._timeout = float(os.environ.get("MENO_PROVIDER_TIMEOUT_SECONDS", "0.8"))
        self._user_id = ""
        self._profile = ""
        self._session_id = ""
        self._agent_context = "primary"
        self._client: httpx.Client | None = None
        self._spool: DurableSpool | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._worker: threading.Thread | None = None
        self._last_recall_count = 0

    @property
    def name(self) -> str:
        return "meno"

    def is_available(self) -> bool:
        return bool(self._base_url and self._token)

    def unavailable_reason(self) -> str:
        if not self._token:
            return "Set MENO_API_TOKEN in the active Hermes profile .env"
        return "Set MENO_API_URL for the local Meno sidecar"

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        hermes_home = Path(kwargs["hermes_home"]).expanduser().resolve()
        self._profile = str(kwargs.get("agent_identity") or hermes_home.name)
        runtime_user = str(kwargs.get("user_id") or kwargs.get("user_id_alt") or "")
        self._user_id = os.environ.get("MENO_USER_ID") or runtime_user or self._profile
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        self._session_id = session_id
        self._spool = DurableSpool(hermes_home / "meno" / "provider-spool.sqlite3")
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={"Authorization": f"Bearer {self._token}"},
            trust_env=False,
        )
        self._worker = threading.Thread(
            target=self._drain_loop,
            name="meno-provider-spool",
            daemon=True,
        )
        self._worker.start()

    def system_prompt_block(self) -> str:
        return (
            "Meno supplies evidence-backed personal context. Treat recalled facets as "
            "fallible context, never as authorization for sensitive or irreversible actions."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._client is None:
            return ""
        try:
            response = self._client.post(
                "/v1/retrieve",
                json={
                    "user_id": self._user_id,
                    "session_id": session_id or self._session_id,
                    "purpose": "response_personalization",
                    "context": {"query": query, "platform": "hermes"},
                    "constraints": {"max_facets": 8, "max_rendered_tokens": 800},
                },
            )
            response.raise_for_status()
            payload = response.json()
            self._last_recall_count = len(payload.get("facets") or [])
            return str(payload.get("rendered_context") or "")
        except Exception:
            self._last_recall_count = 0
            log.warning("Meno retrieve degraded", exc_info=True)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        return None

    def recall_status(self):
        if self._last_recall_count <= 0:
            return None
        return RecallStatus(provider_label="Meno", count=self._last_recall_count, glyph="🧠")

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages=None,
    ) -> None:
        if self._agent_context != "primary" or self._spool is None:
            return
        active_session = session_id or self._session_id
        for role, content in (("user", user_content), ("assistant", assistant_content)):
            if not content:
                continue
            event_id = str(uuid.uuid4())
            accepted = self._spool.enqueue(
                {
                    "user_id": self._user_id,
                    "event_id": event_id,
                    "source": {
                        "type": "hermes_turn",
                        "profile": self._profile,
                        "session_id": active_session,
                    },
                    "content": {"role": role, "text": content},
                    "consent_scope": ["personalization", "task_planning"],
                    "metadata": {"hermes_profile": self._profile},
                }
            )
            if not accepted:
                log.error("Meno durable spool is full; turn was not accepted")
        self._wake.set()

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        self._session_id = new_session_id

    def on_pre_compress(self, messages) -> str:
        deadline = time.monotonic() + 2.0
        self._wake.set()
        while self._spool is not None and self._spool.count() and time.monotonic() < deadline:
            time.sleep(0.02)
        pending = self._spool.count() if self._spool is not None else 0
        if pending:
            return f"Meno has {pending} durable profile-local event(s) queued for ingestion."
        return "Meno turn events were durably accepted before context compression."

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._spool is None:
            return
        event_id = str(uuid.uuid4())
        self._spool.enqueue(
            {
                "user_id": self._user_id,
                "event_id": event_id,
                "source": {
                    "type": "hermes_memory_write",
                    "profile": self._profile,
                    "session_id": str((metadata or {}).get("session_id") or self._session_id),
                },
                "content": {
                    "role": "system",
                    "text": f"{action}:{target}:{content}",
                },
                "consent_scope": ["personalization"],
                "metadata": {"provenance": metadata or {}, "action": action, "target": target},
            }
        )
        self._wake.set()

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [RECALL_SCHEMA, AUDIT_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        if self._client is None:
            return json.dumps({"error": "Meno is not initialized"})
        try:
            if tool_name == "meno_audit":
                response = self._client.get(f"/v1/audit/{args['claim_id']}")
            elif tool_name == "meno_recall":
                response = self._client.post(
                    "/v1/retrieve",
                    json={
                        "user_id": self._user_id,
                        "purpose": "response_personalization",
                        "context": {"query": args["query"], "platform": "hermes-tool"},
                        "constraints": {"max_facets": int(args.get("max_facets", 8))},
                    },
                )
            else:
                return json.dumps({"error": f"unknown tool: {tool_name}"})
            response.raise_for_status()
            return json.dumps(response.json(), ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001 - tool failures must remain fail-open
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "api_url",
                "description": "Loopback URL for the Meno sidecar",
                "required": True,
                "default": "http://127.0.0.1:8765",
                "env_var": "MENO_API_URL",
            },
            {
                "key": "api_token",
                "description": "Bearer token for the Meno sidecar",
                "secret": True,
                "required": True,
                "env_var": "MENO_API_TOKEN",
            },
            {
                "key": "user_id",
                "description": "Stable Meno user identifier for this profile",
                "required": True,
                "env_var": "MENO_USER_ID",
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        return None

    def backup_paths(self) -> list[str]:
        return []

    def shutdown(self) -> None:
        deadline = time.monotonic() + 3.0
        self._wake.set()
        while self._spool is not None and self._spool.count() and time.monotonic() < deadline:
            time.sleep(0.02)
        self._stop.set()
        self._wake.set()
        if self._worker is not None:
            self._worker.join(timeout=2)
        if self._client is not None:
            self._client.close()

    def _drain_loop(self) -> None:
        while not self._stop.is_set():
            item = self._spool.peek() if self._spool is not None else None
            if item is None:
                self._wake.wait(timeout=0.5)
                self._wake.clear()
                continue
            event_id, payload = item
            try:
                if self._client is None:
                    raise RuntimeError("Meno client is not initialized")
                response = self._client.post(
                    "/v1/ingest",
                    headers={"Idempotency-Key": event_id},
                    json=payload,
                )
                response.raise_for_status()
                if self._spool is not None:
                    self._spool.complete(event_id)
            except Exception:
                log.warning("Meno ingest deferred; event remains in durable spool", exc_info=True)
                self._stop.wait(0.5)


def register(ctx) -> None:
    ctx.register_memory_provider(MenoMemoryProvider())
