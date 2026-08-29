"""Grounding and temporal gate lanes (Gate Charter #8, #9, #10).

Three SPEC :1340-1372 metrics measured over one shared surface — the facets Meno
actually injects — because all three are properties of that same set:

| # | metric                          | SPEC threshold           |
|---|---------------------------------|--------------------------|
| 8 | evidence attribution coverage   | >= 99% of active injected |
| 9 | unsupported injected claim rate | < 1%                     |
|10 | stale-active rate               | < 2%                     |

Two honesty constraints are built in, both learned from earlier lanes in this
repo that passed while measuring nothing.

**Sample resolution is reported, not assumed.** With N injected facets the finest
rate this lane can distinguish from zero is 1/N. Asserting "< 1%" from 15 facets
is not evidence, so when 1/N exceeds a threshold the lane reports
``resolvable: false`` for that metric and refuses to call it passed. The committed
`ucm-oracle-v1.json` fixture is small by design; pass `--contexts` to scale up on
a real corpus and get a resolving denominator.

**The deterministic extractor copies text from events, so #9 cannot fail by
construction** — measured directly, unsupported rate is 0/449 on real data. A
metric that cannot fail is not a gate, so `--self-check` (on by default) injects
three synthetic violations and requires each metric to catch its own: a fabricated
value, a stripped evidence link, and an expired short-horizon claim. If a metric
does not catch its violation, the lane fails.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import re
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from benchmarks.run_ucm_oracle import build_ingest_request, build_retrieve_request
from meno.api import build_service
from meno.config import Settings
from meno.db import Claim, ClaimEvidence, Event
from meno.extractor import EPISODIC_MAX_CHARS
from meno.schemas import IngestRequest, RetrieveRequest
from meno.service import _aware
from tests.fakes import TestEmbedder

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE = ROOT / "benchmarks" / "fixtures" / "ucm-oracle-v1.json"

# SPEC :1340-1372 thresholds. Direction differs per metric, so each carries its own
# comparison rather than a shared ">= threshold" assumption.
THRESHOLDS = {
    "evidence_attribution_coverage": {"bound": 0.99, "direction": "min", "charter_item": 8},
    "unsupported_injected_claim_rate": {"bound": 0.01, "direction": "max", "charter_item": 9},
    "stale_active_rate": {"bound": 0.02, "direction": "max", "charter_item": 10},
}

_TOKEN = re.compile(r"[\w]+|[㐀-鿿]")
_ASSISTANT_PREFIX = "Assistant previously responded: "

# Short-horizon time markers and how long the statement stays true, in days. SPEC's
# temporal example is exactly this shape: "本周在东京" must not enter ordinary
# retrieval weeks later.
HORIZON_MARKERS: tuple[tuple[re.Pattern[str], str, int], ...] = (
    (re.compile(r"\btonight\b", re.IGNORECASE), "tonight", 1),
    (re.compile(r"\btoday\b", re.IGNORECASE), "today", 1),
    (re.compile(r"\btomorrow\b", re.IGNORECASE), "tomorrow", 2),
    (re.compile(r"\bright\s+now\b", re.IGNORECASE), "right now", 1),
    (re.compile(r"\bthis\s+weekend\b", re.IGNORECASE), "this weekend", 3),
    (re.compile(r"\bthis\s+week\b", re.IGNORECASE), "this week", 7),
    (re.compile(r"\bnext\s+week\b", re.IGNORECASE), "next week", 14),
    (re.compile(r"\bthis\s+month\b", re.IGNORECASE), "this month", 31),
    (re.compile(r"(?:今晚|今天)"), "today (zh)", 1),
    (re.compile(r"(?:本周|这周)"), "this week (zh)", 7),
    (re.compile(r"(?:本月|这个月)"), "this month (zh)", 31),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--contexts",
        type=Path,
        default=None,
        help="PersonaMem shared_contexts JSONL; scales the denominator so the "
        "SPEC thresholds become resolvable",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="PersonaMem questions CSV; required with --contexts so queries come "
        "from the frozen question set rather than invented probes",
    )
    parser.add_argument("--contexts-limit", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-self-check", action="store_true")
    return parser.parse_args()


def _asserted_tokens(value: str) -> set[str]:
    """Tokens a claim value actually asserts.

    Episodic values are truncated at ``EPISODIC_MAX_CHARS``, which cuts mid-word and
    leaves a fragment ("experie") that appears in no event. Measured on real data
    that artifact alone produced 157 false violations out of 285 claims, so the
    trailing partial token of a truncated value is dropped.
    """
    text = value
    text = text.removeprefix(_ASSISTANT_PREFIX)
    truncated = text.endswith("…")
    tokens = _TOKEN.findall(text.rstrip("…").casefold())
    if truncated and tokens:
        tokens = tokens[:-1]
    return set(tokens)


def _event_tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.casefold()))


def _horizon(value: str) -> tuple[str, int] | None:
    """The shortest validity horizon any marker in the value implies."""
    best: tuple[str, int] | None = None
    for pattern, label, days in HORIZON_MARKERS:
        if pattern.search(value) and (best is None or days < best[1]):
            best = (label, days)
    return best


@dataclasses.dataclass
class FacetObservation:
    """One injected facet, with everything the three metrics need."""

    user_id: str
    query_id: str
    claim_id: str
    kind: str
    value: str
    evidence_ids: list[str]
    # #8: evidence must exist AND resolve to stored events.
    has_evidence: bool
    evidence_resolves: bool
    # #9: value tokens not present in any referenced event.
    unsupported_tokens: list[str]
    # #10: short-horizon marker already expired at query time.
    horizon_label: str | None
    horizon_days: int | None
    age_days: float | None
    superseded_by_newer: bool

    @property
    def attribution_ok(self) -> bool:
        return self.has_evidence and self.evidence_resolves

    @property
    def unsupported(self) -> bool:
        return bool(self.unsupported_tokens)

    @property
    def stale(self) -> bool:
        if self.superseded_by_newer:
            return True
        if self.horizon_days is None or self.age_days is None:
            return False
        return self.age_days > self.horizon_days


def _observe(
    service, user_id: str, query_id: str, response, session
) -> list[FacetObservation]:
    observations: list[FacetObservation] = []
    if not response.facets:
        return observations
    claim_ids = [facet.claim_id for facet in response.facets]
    claims = {
        claim.id: claim
        for claim in session.scalars(
            select(Claim).options(selectinload(Claim.evidence)).where(Claim.id.in_(claim_ids))
        ).all()
    }
    for facet in response.facets:
        claim = claims.get(facet.claim_id)
        evidence_ids = list(facet.evidence_ids)
        events = {
            event.id: event
            for event in session.scalars(
                select(Event).where(Event.id.in_(evidence_ids))
            ).all()
        }
        support: set[str] = set()
        for event_id in evidence_ids:
            event = events.get(event_id)
            if event is not None:
                support |= _event_tokens(event.content)

        # An enriched preference facet renders current + history together, so its
        # asserted tokens legitimately span every event in the chain.
        if isinstance(facet.value, dict):
            asserted = _asserted_tokens(str(facet.value.get("current", "")))
            for historical in facet.value.get("history_oldest_to_newest", []):
                asserted |= _asserted_tokens(str(historical))
        else:
            asserted = _asserted_tokens(str(facet.value))

        newest = max(
            (_aware(event.occurred_at) for event in events.values()),
            default=None,
        )
        as_of = _aware(response_as_of(response, session, user_id))
        age_days = (as_of - newest).total_seconds() / 86400 if newest else None
        horizon = _horizon(str(facet.value)) if not isinstance(facet.value, dict) else None
        superseded_by_newer = bool(
            claim is not None
            and session.scalar(
                select(Claim.id)
                .where(
                    Claim.user_id == claim.user_id,
                    Claim.semantic_key == claim.semantic_key,
                    Claim.status == "active",
                    Claim.id != claim.id,
                    Claim.valid_from > claim.valid_from,
                )
                .limit(1)
            )
        )
        observations.append(
            FacetObservation(
                user_id=user_id,
                query_id=query_id,
                claim_id=facet.claim_id,
                kind=facet.kind,
                value=str(facet.value)[:200],
                evidence_ids=evidence_ids,
                has_evidence=bool(evidence_ids),
                evidence_resolves=bool(evidence_ids) and len(events) == len(set(evidence_ids)),
                unsupported_tokens=sorted(asserted - support)[:8],
                horizon_label=horizon[0] if horizon else None,
                horizon_days=horizon[1] if horizon else None,
                age_days=age_days,
                superseded_by_newer=superseded_by_newer,
            )
        )
    return observations


def response_as_of(response, session, user_id: str) -> datetime:
    """The instant the facets were judged against; carried on the request."""
    return getattr(response, "_as_of", None) or datetime.now(UTC)


def _metric(
    name: str, numerator: int, denominator: int
) -> dict[str, Any]:
    spec = THRESHOLDS[name]
    bound = float(spec["bound"])
    rate = numerator / denominator if denominator else 0.0
    # Finest rate distinguishable from zero at this sample size.
    resolution = 1 / denominator if denominator else 1.0
    resolvable = denominator > 0 and resolution <= bound
    if spec["direction"] == "min":
        meets = rate >= bound
    else:
        meets = rate < bound
    return {
        "charter_item": spec["charter_item"],
        "spec_threshold": bound,
        "direction": spec["direction"],
        "numerator": numerator,
        "denominator": denominator,
        "rate": rate,
        "sample_resolution": resolution,
        # A threshold finer than the sample can resolve is not evidence.
        "resolvable": resolvable,
        "meets_threshold": meets,
        "passed": meets and resolvable,
    }


def _summarize(observations: list[FacetObservation]) -> dict[str, Any]:
    total = len(observations)
    return {
        "evidence_attribution_coverage": _metric(
            "evidence_attribution_coverage",
            sum(item.attribution_ok for item in observations),
            total,
        ),
        "unsupported_injected_claim_rate": _metric(
            "unsupported_injected_claim_rate",
            sum(item.unsupported for item in observations),
            total,
        ),
        "stale_active_rate": _metric(
            "stale_active_rate", sum(item.stale for item in observations), total
        ),
    }


def _build_settings(directory: str) -> Settings:
    settings = Settings(
        database_url=f"sqlite:///{Path(directory) / 'meno.sqlite3'}",
        vector_mode="memory",
        embedding_dimension=256,
        worker_poll_seconds=0.01,
        context_activation_enabled=True,
        user_token_materialization_enabled=True,
        preference_distribution_enabled=True,
        # Reflection is the only writer of non-verbatim claim values, so #9 is
        # only a real constraint while it is on. Forced here rather than read
        # from the environment: a lane that silently skips the one thing that
        # can fail it would report a pass it did not earn.
        reflection_enabled=True,
    )
    settings.validate()
    return settings


def _drain(service) -> None:
    while service.process_outbox(limit=1000):
        pass
    service.process_projection_outbox(limit=1000)


def _retrieve_observed(service, request: RetrieveRequest, query_id: str, as_of: datetime):
    response = service.retrieve(request)
    # The observation needs the instant the request was judged at; RetrieveResponse
    # does not carry it back, so attach it for _observe rather than re-deriving.
    object.__setattr__(response, "_as_of", as_of)
    with service.session_factory() as session:
        return _observe(service, request.user_id, query_id, response, session)


def run_fixture_lane(fixture_path: Path) -> tuple[list[FacetObservation], dict[str, Any]]:
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    observations: list[FacetObservation] = []
    with tempfile.TemporaryDirectory(prefix="meno-grounding-") as directory:
        service = build_service(_build_settings(directory), embedder=TestEmbedder(256))
        try:
            for case in payload["cases"]:
                for event in case["events"]:
                    service.ingest(
                        build_ingest_request(case, event), idempotency_key=event["event_id"]
                    )
                _drain(service)
                for oracle in case["queries"]:
                    request = build_retrieve_request(case, oracle)
                    observations.extend(
                        _retrieve_observed(
                            service,
                            request,
                            f"{case['case_id']}:{oracle['query_id']}",
                            _aware(request.context.as_of),
                        )
                    )
        finally:
            service.close()
    return observations, {
        "source": "fixture",
        "fixture": fixture_path.name,
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "cases": len(payload["cases"]),
    }


def run_corpus_lane(
    contexts_path: Path, questions_path: Path, limit: int
) -> tuple[list[FacetObservation], dict[str, Any]]:
    """Scale the denominator on a real corpus so the SPEC thresholds resolve.

    Queries come from the frozen PersonaMem question set rather than hand-written
    probes: invented queries measure whichever wording happens to clear the lexical
    activation threshold, which is a property of the probe, not of the system. On
    one context, 15 invented queries yielded 4 facets while 10 real questions
    yielded 80.
    """
    raw = contexts_path.read_bytes()
    contexts: dict[str, list[dict[str, str]]] = {}
    for line in raw.decode("utf-8").splitlines():
        contexts.update(json.loads(line))
    question_raw = questions_path.read_bytes()
    with questions_path.open(newline="", encoding="utf-8") as handle:
        questions = list(csv.DictReader(handle))
    by_context: dict[str, list[dict[str, str]]] = {}
    for row in questions:
        by_context.setdefault(row["shared_context_id"], []).append(row)

    selected = sorted(set(contexts) & set(by_context))[:limit]
    observations: list[FacetObservation] = []
    # Fixed clock: ages and horizons must not depend on when the lane ran.
    base = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    ingested = 0
    asked = 0
    with tempfile.TemporaryDirectory(prefix="meno-grounding-corpus-") as directory:
        service = build_service(_build_settings(directory), embedder=TestEmbedder(256))
        try:
            for context_id in selected:
                user_id = f"grounding:{context_id[:16]}"
                messages = contexts[context_id]
                for index, message in enumerate(messages):
                    if message.get("role") != "user":
                        continue
                    event_id = f"{context_id[:16]}-{index}"
                    service.ingest(
                        IngestRequest.model_validate(
                            {
                                "user_id": user_id,
                                "event_id": event_id,
                                # Spread events over time so horizon expiry is real
                                # rather than every claim sharing one timestamp.
                                "occurred_at": (base + timedelta(hours=index)).isoformat(),
                                "source": {
                                    "type": "personamem",
                                    "profile": "grounding-suite",
                                    "session_id": context_id,
                                },
                                "content": {"role": "user", "text": message["content"]},
                                "consent_scope": ["personalization", "task_planning"],
                            }
                        ),
                        idempotency_key=event_id,
                    )
                    ingested += 1
                _drain(service)
                # Just after the last event: the realistic moment a turn is served.
                as_of = base + timedelta(hours=len(messages) + 1)
                for row in by_context[context_id]:
                    request = RetrieveRequest.model_validate(
                        {
                            "user_id": user_id,
                            "purpose": "response_personalization",
                            "context": {
                                "query": row["user_question_or_message"],
                                "task_type": "personalization",
                                "as_of": as_of.isoformat(),
                                "platform": "grounding-suite",
                            },
                            "constraints": {"max_facets": 8, "min_confidence": 0.5},
                        }
                    )
                    asked += 1
                    observations.extend(
                        _retrieve_observed(service, request, row["question_id"], as_of)
                    )
        finally:
            service.close()
    return observations, {
        "source": "corpus",
        "contexts_file": contexts_path.name,
        "contexts_sha256": hashlib.sha256(raw).hexdigest(),
        "questions_file": questions_path.name,
        "questions_sha256": hashlib.sha256(question_raw).hexdigest(),
        "contexts_used": len(selected),
        "events_ingested": ingested,
        "questions_asked": asked,
    }


# Each probe writes one synthetic violation directly into the store and names the
# metric that must catch it. Injecting at the storage layer is deliberate: the
# deterministic extractor cannot produce these states, which is exactly why the
# metrics would otherwise never be exercised.
def run_self_check() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for probe, metric, rationale in (
        (
            "fabricated_value",
            "unsupported_injected_claim_rate",
            (
                "A claim asserting content absent from its evidence must be counted. "
                "The extractor copies event text, so unsupported rate is 0 by "
                "construction (measured 0/449) and this metric is otherwise untested."
            ),
        ),
        (
            "stripped_evidence",
            "evidence_attribution_coverage",
            "A claim whose evidence link is gone must drop coverage below 100%.",
        ),
        (
            "expired_horizon",
            "stale_active_rate",
            "A 'today' claim still injected 60 days later must be counted stale.",
        ),
        (
            "fabricated_pattern",
            "unsupported_injected_claim_rate",
            (
                "Reflection is the only writer of non-verbatim claim values, so it is "
                "the one route by which #9 can genuinely fail. This probe forms a "
                "real pattern claim from a multi-session series, then fabricates "
                "its value. The corpus run cannot cover this: PersonaMem contexts "
                "are single-session, so no pattern clears the span requirement and "
                "#9 would report a pass over episodic claims alone."
            ),
        ),
    ):
        observations = _run_violation(probe)
        summary = _summarize(observations)
        detected = (
            summary[metric]["numerator"] > 0
            if metric != "evidence_attribution_coverage"
            else summary[metric]["numerator"] < summary[metric]["denominator"]
        )
        results.append(
            {
                "probe": probe,
                "metric": metric,
                "rationale": rationale,
                "injected_facets": len(observations),
                "detected": detected,
                "passed": detected,
            }
        )
    return results


def _run_violation(probe: str) -> list[FacetObservation]:
    base = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    as_of = base + timedelta(days=60)
    # `fabricated_pattern` needs several restatements spread across sessions so a
    # pattern claim actually forms; the others need one event.
    texts = {
        "fabricated_value": ["I prefer green tea"],
        "stripped_evidence": ["I prefer green tea"],
        "expired_horizon": ["I am in Tokyo today"],
        "fabricated_pattern": [
            "I prefer green tea in the morning",
            "I prefer green tea when working",
            "I prefer green tea over soda",
        ],
    }[probe]
    with tempfile.TemporaryDirectory(prefix=f"meno-grounding-{probe}-") as directory:
        service = build_service(_build_settings(directory), embedder=TestEmbedder(256))
        try:
            user_id = f"probe:{probe}"
            for index, text in enumerate(texts):
                service.ingest(
                    IngestRequest.model_validate(
                        {
                            "user_id": user_id,
                            "event_id": f"{probe}-event-{index}",
                            # Spread across days so reflection's span requirement
                            # is met for the pattern probe.
                            "occurred_at": (base + timedelta(days=index * 3)).isoformat(),
                            "source": {
                                "type": "hermes_turn",
                                "profile": "grounding-probe",
                                "session_id": f"{probe}-{index}",
                            },
                            "content": {"role": "user", "text": text},
                            "consent_scope": ["personalization", "task_planning"],
                        }
                    ),
                    idempotency_key=f"{probe}-event-{index}",
                )
                _drain(service)

            if probe in {"fabricated_value", "stripped_evidence"}:
                with service.session_factory.begin() as session:
                    claim = session.scalar(
                        select(Claim).where(
                            Claim.user_id == user_id, Claim.status == "active"
                        )
                    )
                    if probe == "fabricated_value":
                        # Content no event supports: the defect #9 exists to catch.
                        claim.value = "cardiology appointment in Reykjavik"
                    else:
                        session.execute(
                            ClaimEvidence.__table__.delete().where(
                                ClaimEvidence.claim_id == claim.id
                            )
                        )
                        # Keep it retrievable: _admit lets explicit_feedback through
                        # without evidence, which is the state #8 must report.
                        claim.source_type = "explicit_feedback"
                service.rebuild_projection()
            elif probe == "fabricated_pattern":
                with service.session_factory.begin() as session:
                    claim = session.scalar(
                        select(Claim).where(
                            Claim.user_id == user_id,
                            Claim.kind == "pattern",
                            Claim.status == "active",
                        )
                    )
                    if claim is None:
                        raise RuntimeError(
                            "fabricated_pattern probe formed no pattern claim; "
                            "reflection must be enabled and its thresholds met, "
                            "otherwise this probe silently tests nothing"
                        )
                    claim.value = "repeatedly prefers cardiology appointments"
                service.rebuild_projection()

            request = RetrieveRequest.model_validate(
                {
                    "user_id": user_id,
                    "purpose": "response_personalization",
                    "context": {
                        # Context activation gates on slot match or lexical overlap,
                        # so each probe query must genuinely activate its claim —
                        # otherwise the probe measures activation, not the metric.
                        "query": {
                            "fabricated_value": "Should I drink tea or coffee?",
                            "stripped_evidence": "Should I drink tea or coffee?",
                            "expired_horizon": "I am in Tokyo today",
                            "fabricated_pattern": "Should I drink tea or coffee?",
                        }[probe],
                        "task_type": "personalization",
                        "as_of": as_of.isoformat(),
                        "platform": "grounding-probe",
                    },
                    "constraints": {"max_facets": 8, "min_confidence": 0.0},
                }
            )
            return _retrieve_observed(service, request, probe, as_of)
        finally:
            service.close()


def run_benchmark(
    fixture_path: Path,
    *,
    contexts_path: Path | None = None,
    questions_path: Path | None = None,
    contexts_limit: int = 8,
    self_check: bool = True,
) -> dict[str, Any]:
    if contexts_path is not None:
        if questions_path is None:
            raise ValueError("--contexts requires --questions")
        observations, provenance = run_corpus_lane(
            contexts_path, questions_path, contexts_limit
        )
    else:
        observations, provenance = run_fixture_lane(fixture_path)
    metrics = _summarize(observations)
    self_check_results = run_self_check() if self_check else []

    unresolvable = [name for name, item in metrics.items() if not item["resolvable"]]
    report = {
        "benchmark": "Meno grounding and temporal gate suite",
        "gate_charter_items": [8, 9, 10],
        "schema_version": "grounding-temporal-suite-v1",
        "answer_model_calls": 0,
        "provenance": provenance,
        "injected_facets": len(observations),
        "metrics": metrics,
        # Thresholds finer than 1/N cannot be evidenced at this sample size.
        "unresolvable_metrics": unresolvable,
        "self_check": {
            "ran": self_check,
            "probes": self_check_results,
            "passed": all(item["passed"] for item in self_check_results)
            if self_check
            else False,
        },
        "notes": {
            "unsupported_rate_is_structurally_low": (
                "The deterministic extractor copies event text into claim values, so "
                "token-level support holds by construction (measured 0/449 on the "
                "PersonaMem corpus). Treat #9 as a regression guard against a future "
                "generative extractor, not as evidence that inference is grounded."
            ),
            "episodic_truncation": (
                f"Episodic values are cut at {EPISODIC_MAX_CHARS} chars mid-word; the "
                "trailing partial token is excluded from the support check because it "
                "appears in no event by construction (157/285 false positives without "
                "this rule)."
            ),
        },
        "passed": (
            not unresolvable
            and all(item["passed"] for item in metrics.values())
            and self_check
            and all(item["passed"] for item in self_check_results)
        ),
        "violations": [
            dataclasses.asdict(item)
            for item in observations
            if not item.attribution_ok or item.unsupported or item.stale
        ][:50],
    }
    return report


def main() -> None:
    args = parse_args()
    report = run_benchmark(
        args.fixture,
        contexts_path=args.contexts,
        questions_path=args.questions,
        contexts_limit=args.contexts_limit,
        self_check=not args.no_self_check,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "violations"},
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
