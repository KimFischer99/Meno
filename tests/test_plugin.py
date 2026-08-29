from __future__ import annotations

import stat

from meno.hermes_plugin import DurableSpool


def test_durable_spool_round_trip_and_permissions(tmp_path):
    path = tmp_path / "profile" / "meno" / "spool.sqlite3"
    spool = DurableSpool(path)
    payload = {"event_id": "event-1", "content": {"role": "user", "text": "hello"}}
    assert spool.enqueue(payload)
    assert spool.enqueue(payload)
    assert spool.count() == 1
    event_id, restored = spool.peek()
    assert event_id == "event-1"
    assert restored == payload
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    spool.complete(event_id)
    assert spool.count() == 0
