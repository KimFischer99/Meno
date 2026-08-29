"""Validate an externally exported corpus against Meno's evaluation contract.

Companion to `MENO_DATA_REQUIREMENTS.md`. Run this on a small sample (20-30
queries) *before* annotating the full set: the decisive output is
``query_signal``, which predicts whether a curated set over this corpus can
evidence anything at all.

Why that gate exists: PersonaMem's decisive tokens appear in the query for only
0.087 of questions, and four architecture interventions each improved their
mechanism while the score stayed flat, because no query-conditioned selector can
find evidence the query does not point at. A corpus that scores the same is not
worth annotating -- see `probe_curated_viability.py` for that measurement.

This validator checks three things, in order of what can waste the most effort:

1. **Structure** -- fields, types, index continuity, and the causality rule that
   evidence must precede the question. A corpus that violates these cannot be
   ingested at all.
2. **Query signal** -- can the deciding evidence be located from the query? This
   is the go/no-go for annotation.
3. **Privacy** -- a best-effort scan for credentials and direct identifiers. This
   is a backstop, not a compliance guarantee.

Deterministic, read-only, no answer model and no embedding provider. It emits no
message text: findings are reported by location, so the report is safe to share
even when the corpus is not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from meno.service import _content_tokens

# Contract minimums from MENO_DATA_REQUIREMENTS.md §1.5. Sample runs are expected
# to fall short of these; the report separates "malformed" from "too small".
MIN_QUERIES = 139
MIN_ABSTAIN_SHARE = 0.20
MAX_ABSTAIN_SHARE = 0.40
MIN_SESSIONS = 30
MIN_USERS = 10

# Interpretation bands for query_signal (§1.6). The lower bound sits above the
# 0.087 that made PersonaMem unusable, with room for sampling noise.
SIGNAL_STOP = 0.15
SIGNAL_PROMISING = 0.35
PERSONAMEM_SIGNAL = 0.087

VALID_ROLES = {"user", "assistant", "system"}
VALID_DECISIONS = {"inject", "abstain"}

# Best-effort credential and direct-identifier patterns. Deliberately narrow: this
# is a backstop that must not lull the operator into skipping their own review.
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{16,}")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack_token", re.compile(r"\bxox[abpsr]-[A-Za-z0-9-]{10,}")),
    ("bearer_header", re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+\S{8,}")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("connection_string", re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb)://[^\s/@]+:[^\s/@]+@")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("cn_mobile", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("cn_id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("credit_card", re.compile(r"(?<!\d)(?:\d[ -]?){15,18}(?!\d)")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Pre-annotation sample: report volume shortfalls as informational "
        "rather than as contract violations",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str], str]:
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{number}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(parsed, dict):
            errors.append(f"{path.name}:{number}: expected an object")
            continue
        rows.append(parsed)
    return rows, errors, hashlib.sha256(raw).hexdigest()


def validate_sessions(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    sessions: dict[str, dict[str, Any]] = {}
    users: set[str] = set()
    role_counts: Counter[str] = Counter()
    missing_timestamps = 0
    identical_timestamps: Counter[str] = Counter()

    for position, row in enumerate(rows, start=1):
        session_id = row.get("session_id")
        user_key = row.get("user_key")
        messages = row.get("messages")
        where = f"sessions[{position}]"
        if not isinstance(session_id, str) or not session_id:
            errors.append(f"{where}: session_id must be a non-empty string")
            continue
        if not isinstance(user_key, str) or not user_key:
            errors.append(f"{where} ({session_id}): user_key must be a non-empty string")
            continue
        if session_id in sessions:
            errors.append(f"{where}: duplicate session_id")
            continue
        if not isinstance(messages, list) or not messages:
            errors.append(f"{where} ({session_id}): messages must be a non-empty list")
            continue

        by_index: dict[int, dict[str, Any]] = {}
        timestamps: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                errors.append(f"{session_id}: message entries must be objects")
                continue
            index = message.get("index")
            role = message.get("role")
            text = message.get("text")
            if not isinstance(index, int) or index < 0:
                errors.append(f"{session_id}: message index must be a non-negative int")
                continue
            if index in by_index:
                errors.append(f"{session_id}: duplicate message index {index}")
                continue
            if role not in VALID_ROLES:
                errors.append(
                    f"{session_id}[{index}]: role must be one of {sorted(VALID_ROLES)}"
                )
                continue
            if not isinstance(text, str) or not text.strip():
                errors.append(f"{session_id}[{index}]: text must be a non-empty string")
                continue
            occurred_at = message.get("occurred_at")
            if not isinstance(occurred_at, str) or not occurred_at:
                missing_timestamps += 1
            else:
                timestamps.append(occurred_at)
            by_index[index] = message
            role_counts[role] += 1

        if by_index:
            expected = set(range(len(by_index)))
            if set(by_index) != expected:
                errors.append(
                    f"{session_id}: message indices must run 0..{len(by_index) - 1} "
                    "without gaps"
                )
        # Every message sharing one timestamp means the export used its own clock.
        # Time decay, stale-active, and supersede ordering all become meaningless.
        if len(timestamps) > 1 and len(set(timestamps)) == 1:
            identical_timestamps[session_id] += 1

        sessions[session_id] = {"user_key": user_key, "messages": by_index}
        users.add(user_key)

    return {
        "sessions": len(sessions),
        "users": len(users),
        "messages": sum(role_counts.values()),
        "role_counts": dict(sorted(role_counts.items())),
        "messages_missing_occurred_at": missing_timestamps,
        "sessions_with_one_timestamp": len(identical_timestamps),
        "_index": sessions,
    }, errors


def validate_queries(
    rows: list[dict[str, Any]], sessions: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    seen: set[str] = set()
    accepted: list[dict[str, Any]] = []
    decisions: Counter[str] = Counter()
    causality_violations = 0
    empty_relevant_on_inject = 0
    nonempty_relevant_on_abstain = 0

    for position, row in enumerate(rows, start=1):
        where = f"queries[{position}]"
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            errors.append(f"{where}: query_id must be a non-empty string")
            continue
        if query_id in seen:
            errors.append(f"{where}: duplicate query_id {query_id}")
            continue
        seen.add(query_id)

        session_id = row.get("session_id")
        session = sessions.get(session_id) if isinstance(session_id, str) else None
        if session is None:
            errors.append(f"{query_id}: session_id does not match any exported session")
            continue
        user_key = row.get("user_key")
        if user_key != session["user_key"]:
            errors.append(f"{query_id}: user_key disagrees with its session")
            continue

        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            errors.append(f"{query_id}: query must be a non-empty string")
            continue
        asked_at = row.get("asked_at_index")
        if not isinstance(asked_at, int) or asked_at <= 0:
            errors.append(f"{query_id}: asked_at_index must be a positive int")
            continue
        decision = row.get("expected_decision")
        if decision not in VALID_DECISIONS:
            errors.append(
                f"{query_id}: expected_decision must be one of {sorted(VALID_DECISIONS)}"
            )
            continue
        relevant = row.get("relevant_message_indices")
        if not isinstance(relevant, list) or any(
            not isinstance(item, int) for item in relevant
        ):
            errors.append(f"{query_id}: relevant_message_indices must be a list of ints")
            continue

        unknown = [index for index in relevant if index not in session["messages"]]
        if unknown:
            errors.append(f"{query_id}: relevant_message_indices not in session: {unknown}")
            continue
        # The rule that makes the label honest: a live system answering at turn N
        # cannot see turn N+1. Labels citing future messages measure nothing.
        future = [index for index in relevant if index >= asked_at]
        if future:
            causality_violations += 1
            errors.append(
                f"{query_id}: relevant_message_indices {future} are at or after "
                f"asked_at_index {asked_at}; evidence must precede the question"
            )
            continue
        if decision == "inject" and not relevant:
            empty_relevant_on_inject += 1
            errors.append(f"{query_id}: expected_decision=inject requires relevant indices")
            continue
        if decision == "abstain" and relevant:
            nonempty_relevant_on_abstain += 1
            errors.append(
                f"{query_id}: expected_decision=abstain must have no relevant indices"
            )
            continue

        decisions[decision] += 1
        accepted.append(
            {
                "query_id": query_id,
                "session_id": session_id,
                "query": query,
                "asked_at_index": asked_at,
                "relevant": relevant,
                "expected_decision": decision,
                "task_type": row.get("task_type"),
            }
        )

    total = len(accepted)
    abstain = decisions.get("abstain", 0)
    return {
        "queries_accepted": total,
        "decisions": dict(sorted(decisions.items())),
        "abstain_share": abstain / total if total else 0.0,
        "causality_violations": causality_violations,
        "inject_without_relevant": empty_relevant_on_inject,
        "abstain_with_relevant": nonempty_relevant_on_abstain,
        "_accepted": accepted,
    }, errors


def measure_query_signal(
    queries: list[dict[str, Any]], sessions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Share of the labeled evidence's distinctive tokens that the query names.

    Only ``inject`` queries are scored: an abstain query has no evidence to locate,
    so including them would dilute the measurement toward zero and make a good
    corpus look like a bad one.

    "Distinctive" means tokens in the labeled messages but not in the rest of the
    session. Common session vocabulary would otherwise inflate the score for any
    query that shares ordinary words with its context.
    """
    per_query: list[float] = []
    zero_signal = 0
    scored: list[dict[str, Any]] = []
    for item in queries:
        if item["expected_decision"] != "inject":
            continue
        session = sessions[item["session_id"]]
        messages = session["messages"]
        relevant_tokens: set[str] = set()
        for index in item["relevant"]:
            relevant_tokens |= _content_tokens(messages[index]["text"])
        background: set[str] = set()
        for index, message in messages.items():
            if index in item["relevant"] or index >= item["asked_at_index"]:
                continue
            background |= _content_tokens(message["text"])
        distinctive = relevant_tokens - background
        if not distinctive:
            # The labeled evidence says nothing the rest of the session does not.
            # That is a labeling problem, not a signal measurement.
            continue
        signal = len(distinctive & _content_tokens(item["query"])) / len(distinctive)
        per_query.append(signal)
        if signal == 0.0:
            zero_signal += 1
        scored.append({"query_id": item["query_id"], "query_signal": signal})

    mean_signal = statistics.mean(per_query) if per_query else 0.0
    if mean_signal < SIGNAL_STOP:
        band = "STOP_DO_NOT_ANNOTATE"
    elif mean_signal < SIGNAL_PROMISING:
        band = "WEAK_EXPAND_SAMPLE_FIRST"
    else:
        band = "GOOD_PROCEED_TO_ANNOTATION"
    return {
        "scored_queries": len(per_query),
        "query_signal": mean_signal,
        "median_query_signal": statistics.median(per_query) if per_query else 0.0,
        "queries_with_zero_signal": zero_signal,
        "zero_signal_share": zero_signal / len(per_query) if per_query else 0.0,
        "personamem_reference": PERSONAMEM_SIGNAL,
        "stop_below": SIGNAL_STOP,
        "promising_above": SIGNAL_PROMISING,
        "band": band,
        "interpretation": (
            f"Mean {mean_signal:.4f} vs {PERSONAMEM_SIGNAL} on the corpus that proved "
            "unusable. Below "
            f"{SIGNAL_STOP} the deciding evidence is not reachable from the query and "
            "no labeling scheme rescues it."
        ),
        "per_query": scored[:200],
    }


def scan_privacy(sessions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Best-effort credential/identifier scan. Locations only, never content."""
    hits: Counter[str] = Counter()
    locations: list[dict[str, Any]] = []
    for session_id, session in sessions.items():
        for index, message in session["messages"].items():
            for name, pattern in SECRET_PATTERNS:
                if pattern.search(message["text"]):
                    hits[name] += 1
                    if len(locations) < 200:
                        locations.append(
                            {
                                "session_id": session_id,
                                "index": index,
                                "pattern": name,
                            }
                        )
    return {
        "patterns_matched": dict(sorted(hits.items())),
        "total_matches": sum(hits.values()),
        # Locations, not text: the report must be shareable when the corpus is not.
        "locations": locations,
        "clean": not hits,
        "caveat": (
            "A backstop, not a compliance guarantee. It cannot detect paraphrased "
            "personal information, sensitive topics, or identifiers it has no pattern "
            "for. The operator's own review is what protects the data."
        ),
    }


def _volume(sessions: dict[str, Any], queries: dict[str, Any]) -> dict[str, Any]:
    total = queries["queries_accepted"]
    abstain_share = queries["abstain_share"]
    conditions = {
        "minimum_queries": total >= MIN_QUERIES,
        "abstain_share_in_range": MIN_ABSTAIN_SHARE <= abstain_share <= MAX_ABSTAIN_SHARE,
        "minimum_sessions": sessions["sessions"] >= MIN_SESSIONS,
        "minimum_users": sessions["users"] >= MIN_USERS,
    }
    return {
        "requirements": {
            "minimum_queries": MIN_QUERIES,
            "abstain_share_range": [MIN_ABSTAIN_SHARE, MAX_ABSTAIN_SHARE],
            "minimum_sessions": MIN_SESSIONS,
            "minimum_users": MIN_USERS,
        },
        "conditions": conditions,
        "passed": all(conditions.values()),
        "shortfalls": [name for name, ok in conditions.items() if not ok],
        "rationale": (
            f"{MIN_QUERIES} queries separates Recall@10 {0.90} from {0.85} at 95% "
            "confidence. An abstain share in range is what makes this a negative "
            "benchmark rather than a recall-only test SPEC warns against."
        ),
    }


def validate(
    sessions_path: Path, queries_path: Path, *, sample: bool = False
) -> dict[str, Any]:
    session_rows, session_read_errors, sessions_sha = _read_jsonl(sessions_path)
    query_rows, query_read_errors, queries_sha = _read_jsonl(queries_path)
    session_summary, session_errors = validate_sessions(session_rows)
    index = session_summary.pop("_index")
    query_summary, query_errors = validate_queries(query_rows, index)
    accepted = query_summary.pop("_accepted")

    signal = measure_query_signal(accepted, index)
    privacy = scan_privacy(index)
    volume = _volume(session_summary, query_summary)
    errors = [*session_read_errors, *query_read_errors, *session_errors, *query_errors]

    structure_ok = not errors
    # In sample mode the point is the signal reading, so volume shortfalls are
    # expected and do not fail the run. Structure and causality always must hold.
    contract_ok = structure_ok and (sample or volume["passed"])
    return {
        "check": "Meno external corpus contract validation",
        "schema_version": "external-corpus-validation-v1",
        "requirements_document": "MENO_DATA_REQUIREMENTS.md",
        "mode": "sample" if sample else "full",
        "answer_model_calls": 0,
        "provenance": {
            "sessions_file": sessions_path.name,
            "sessions_sha256": sessions_sha,
            "queries_file": queries_path.name,
            "queries_sha256": queries_sha,
        },
        "structure": {
            "sessions": session_summary,
            "queries": query_summary,
            "errors": errors[:200],
            "error_count": len(errors),
            "passed": structure_ok,
        },
        "volume": volume,
        "query_signal": signal,
        "privacy_scan": privacy,
        "verdict": {
            "contract_satisfied": contract_ok,
            "annotation_recommendation": signal["band"],
            "next_step": (
                "Fix the structural errors first; the signal reading is not "
                "trustworthy until every query parses and respects causality."
                if not structure_ok
                else "Do not annotate this corpus. Re-sample toward queries that name "
                "what they ask about (MENO_DATA_REQUIREMENTS.md §1.7), then re-run."
                if signal["band"] == "STOP_DO_NOT_ANNOTATE"
                else "Expand the sample to ~50 queries and re-measure before "
                "committing the full annotation budget."
                if signal["band"] == "WEAK_EXPAND_SAMPLE_FIRST"
                else f"Signal is sufficient. Proceed to {MIN_QUERIES} labeled queries "
                "under the no-answer-peeking constraint (§1.4)."
            ),
        },
        "scope": (
            "Validates the export contract and predicts annotation viability. It does "
            "not score Recall@10 or nDCG@10, and passing it does not move Gate Charter "
            "#13/#14 -- those need the annotated set itself."
        ),
    }


def main() -> None:
    args = parse_args()
    report = validate(args.sessions, args.queries, sample=args.sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    structure = report["structure"]
    print(
        f"sessions={structure['sessions']['sessions']} "
        f"users={structure['sessions']['users']} "
        f"messages={structure['sessions']['messages']} "
        f"queries={structure['queries']['queries_accepted']}"
    )
    if not structure["passed"]:
        print(f"\nSTRUCTURAL ERRORS ({structure['error_count']}):")
        for error in structure["errors"][:20]:
            print(f"  - {error}")
        if structure["error_count"] > 20:
            print(f"  ... and {structure['error_count'] - 20} more")

    if structure["sessions"]["messages_missing_occurred_at"]:
        print(
            f"\nWARNING: {structure['sessions']['messages_missing_occurred_at']} messages "
            "lack occurred_at; time decay and stale-active become meaningless"
        )
    if structure["sessions"]["sessions_with_one_timestamp"]:
        print(
            f"WARNING: {structure['sessions']['sessions_with_one_timestamp']} sessions "
            "share a single timestamp across all messages; use real message times"
        )

    signal = report["query_signal"]
    print(
        f"\nquery_signal = {signal['query_signal']:.4f} "
        f"(median {signal['median_query_signal']:.4f}, "
        f"n={signal['scored_queries']}, "
        f"zero-signal {signal['zero_signal_share']:.1%})"
    )
    print(f"  PersonaMem reference: {signal['personamem_reference']}")
    print(f"  band: {signal['band']}")

    if not report["volume"]["passed"]:
        print(f"\nvolume shortfalls: {report['volume']['shortfalls']}")

    privacy = report["privacy_scan"]
    if not privacy["clean"]:
        print(f"\nPRIVACY SCAN matched {privacy['total_matches']}:")
        for name, count in privacy["patterns_matched"].items():
            print(f"  - {name}: {count}")
        print("  (locations in the report; review before sharing this corpus)")

    print(f"\n{report['verdict']['next_step']}")
    print(f"wrote {args.output}")
    if not report["verdict"]["contract_satisfied"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
