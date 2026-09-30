"""Small, lossless delta format for user-token snapshots (stdlib only)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

PAYLOAD_FIELDS = {
    "user_id",
    "state_revision",
    "active_state",
    "consent_state",
    "preference_distributions",
    "clarification_opportunities",
    "uncertainty",
}
ACTIVE_FIELDS = {
    "claim_id",
    "kind",
    "semantic_channel",
    "semantic_key",
    "routing_slot",
    "value",
    "confidence",
    "sensitive",
    "allowed_purposes",
    "source_type",
    "valid_from",
    "valid_to",
    "evidence_ids",
}
REPLACE_FIELDS = PAYLOAD_FIELDS - {"user_id", "state_revision", "active_state"}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _supported(payload: Any) -> bool:
    if (
        not isinstance(payload, dict)
        or payload.keys() != PAYLOAD_FIELDS
        or not isinstance(payload.get("user_id"), str)
        or type(payload.get("state_revision")) is not int
        or not isinstance(payload.get("active_state"), list)
    ):
        return False
    keys = []
    for item in payload["active_state"]:
        if (
            not isinstance(item, dict)
            or item.keys() != ACTIVE_FIELDS
            or not isinstance(item.get("semantic_key"), str)
            or not isinstance(item.get("confidence"), dict)
            or item["confidence"].keys() != {"calibrated", "half_life_days", "support_count"}
        ):
            return False
        keys.append(item["semantic_key"])
    return keys == sorted(set(keys))


def make_delta(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any] | None:
    """Return a smaller representable patch, or None to store a full base."""
    if not _supported(previous) or not _supported(current):
        return None
    if previous["user_id"] != current["user_id"]:
        return None
    if current["state_revision"] <= previous["state_revision"]:
        return None

    before = {item["semantic_key"]: item for item in previous["active_state"]}
    after = {item["semantic_key"]: item for item in current["active_state"]}
    active = {
        "set": [
            after[key]
            for key in sorted(after)
            if key not in before or canonical_json(before[key]) != canonical_json(after[key])
        ],
        "delete": sorted(before.keys() - after.keys()),
    }
    replace = {
        key: current[key]
        for key in sorted(REPLACE_FIELDS)
        if canonical_json(previous[key]) != canonical_json(current[key])
    }
    delta = {"version": 1, "active_state": active, "replace": replace}
    if len(canonical_json(delta)) >= len(canonical_json(current)):
        return None
    return delta


def apply_delta(
    previous: dict[str, Any], delta: dict[str, Any], user_id: str, revision: int
) -> dict[str, Any]:
    if not _supported(previous):
        raise ValueError("unsupported snapshot base payload")
    if previous["user_id"] != user_id or type(revision) is not int or revision <= previous["state_revision"]:
        raise ValueError("invalid snapshot delta revision or user")
    if (
        not isinstance(delta, dict)
        or delta.keys() != {"version", "active_state", "replace"}
        or delta.get("version") != 1
        or not isinstance(delta.get("active_state"), dict)
        or delta["active_state"].keys() != {"set", "delete"}
        or not isinstance(delta["active_state"]["set"], list)
        or not isinstance(delta["active_state"]["delete"], list)
        or not isinstance(delta.get("replace"), dict)
        or delta["replace"].keys() - REPLACE_FIELDS
    ):
        raise ValueError("invalid snapshot delta")

    items = {item["semantic_key"]: item for item in previous["active_state"]}
    for key in delta["active_state"]["delete"]:
        if not isinstance(key, str) or key not in items:
            raise ValueError("invalid snapshot delta deletion")
        del items[key]
    for item in delta["active_state"]["set"]:
        if not isinstance(item, dict) or item.keys() != ACTIVE_FIELDS:
            raise ValueError("unsupported snapshot delta item")
        key = item.get("semantic_key")
        if not isinstance(key, str):
            raise TypeError("invalid snapshot delta key")
        items[key] = item

    payload = dict(previous)
    payload["active_state"] = [items[key] for key in sorted(items)]
    payload.update(delta["replace"])
    payload["user_id"] = user_id
    payload["state_revision"] = revision
    if not _supported(payload):
        raise ValueError("snapshot delta produced unsupported payload")
    return payload


def self_check() -> None:
    base = {
        "user_id": "u",
        "state_revision": 10,
        "active_state": [
            {
                "claim_id": "a",
                "kind": "fact",
                "semantic_channel": "x",
                "semantic_key": "a",
                "routing_slot": None,
                "value": 30,
                "confidence": {"calibrated": 0.5, "half_life_days": None, "support_count": 1},
                "sensitive": False,
                "allowed_purposes": [],
                "source_type": "test",
                "valid_from": "2026-01-01T00:00:00+00:00",
                "valid_to": None,
                "evidence_ids": [],
            }
        ],
        "consent_state": [],
        "preference_distributions": [],
        "clarification_opportunities": [],
        "uncertainty": {
            "low_confidence_claim_ids": [],
            "unsupported_claim_ids": [],
            "ambiguous_claim_ids": [],
        },
    }
    current = json.loads(canonical_json(base))
    current["state_revision"] = 12  # A revision gap is linked by previous_revision.
    current["active_state"][0]["value"] = 30.0
    delta = make_delta(base, current)
    assert delta is not None and content_hash(base) != content_hash(current)
    rebuilt = apply_delta(base, delta, "u", 12)
    assert canonical_json(rebuilt) == canonical_json(current)
    assert type(rebuilt["active_state"][0]["value"]) is float
    assert make_delta(base, {**current, "future_field": True}) is None
    try:
        apply_delta(base, delta, "u", 9)
    except ValueError:
        pass
    else:
        raise AssertionError("backwards revision was accepted")


if __name__ == "__main__":
    self_check()
    print("snapshot delta self-check: ok")
