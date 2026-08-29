"""Deterministic retrieval ranking comparison between two Meno revisions.

Both inputs are retrieval-cache artifacts produced by ``run_personamem_e2e.py``.
Every metric is derived from the retrieval phase only: ``ranking_predicted_option``,
``ranking_correct_supported``, and ``ranking_reciprocal_rank`` are computed before
any answer model is called, so this lane has no answer-model variance.

The end-to-end lane remains useful for product feel, but its per-subset resolution
is below its own answer-model noise floor, so architectural deltas are judged here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

FACET_KIND_PATTERN = re.compile(r"^- \[(\w+)(?: timeline)?\]", re.MULTILINE)
TIMELINE_PATTERN = re.compile(r"^- \[\w+ timeline\]", re.MULTILINE)
STANCE_TOKENS = frozenset(
    {
        "dislike",
        "disliked",
        "disliking",
        "prefers",
        "longer",
        "evolved",
        "away",
        "changed",
        "change",
        "progression",
        "shifted",
        "stopped",
    }
)
REQUIRED_RECORD_FIELDS = (
    "question_id",
    "question_type",
    "correct_option",
    "rendered_context",
    "facet_count",
    "ranking_predicted_option",
    "ranking_correct_supported",
    "ranking_reciprocal_rank",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help=(
            "compare only the questions present in both artifacts, for narrow-band runs "
            "produced with the runner's --question-types or --limit flags"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} contains no retrieval records")
    for record in records:
        missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
        if missing:
            raise ValueError(f"{path} record {record.get('question_id')!r} is missing {missing}")
    question_ids = [record["question_id"] for record in records]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError(f"{path} contains duplicate question IDs")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "run_id": payload.get("run_id"),
        "questions_sha256": payload.get("questions_sha256"),
        "contexts_sha256": payload.get("contexts_sha256"),
        "service_fingerprint": payload.get("service_fingerprint"),
        "records": {record["question_id"]: record for record in records},
    }


def _require_comparable(
    baseline: dict[str, Any], treatment: dict[str, Any], *, allow_subset: bool = False
) -> None:
    for field in ("questions_sha256", "contexts_sha256"):
        left, right = baseline[field], treatment[field]
        if not left or not right:
            raise ValueError(f"both artifacts must record {field}")
        if left != right:
            raise ValueError(
                f"{field} differs between artifacts; the runs used different datasets "
                "and are not comparable"
            )
    shared = set(baseline["records"]) & set(treatment["records"])
    if not shared:
        raise ValueError("artifacts share no question IDs")
    if allow_subset:
        # Narrow-band runs (--question-types / --limit) cover fewer questions; the
        # dataset hashes still pin the source, so the intersection is comparable.
        return
    if len(shared) != len(baseline["records"]) or len(shared) != len(treatment["records"]):
        raise ValueError(
            "artifacts cover different question sets: "
            f"baseline={len(baseline['records'])} treatment={len(treatment['records'])} "
            f"shared={len(shared)}; pass --allow-subset to compare the intersection"
        )


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _length_normalized_ranking(record: dict[str, Any]) -> bool | None:
    """Whether a length-normalized scorer would pick the correct option.

    The runner's own scorer sums IDF over matched tokens without normalizing, so
    it structurally favors long options. On subsets where the correct answer is
    shorter than its distractors that penalty dominates, which reads as a memory
    failure when it is a scoring artifact. Reported alongside the runner metric,
    never as a replacement, since normalizing hurts subsets where the correct
    option is the long one.
    """
    options = record.get("options")
    correct = record.get("correct_option")
    if not isinstance(options, list) or not isinstance(correct, int):
        return None
    if not 0 <= correct < len(options):
        return None
    memory = _tokens(record.get("rendered_context", ""))
    option_tokens = [_tokens(re.sub(r"^\([a-d]\)\s*", "", option)) for option in options]
    frequency = Counter(token for tokens in option_tokens for token in tokens)
    scores = [
        sum(
            math.log((len(options) + 1) / (frequency[token] + 0.5))
            for token in tokens & memory
        )
        / max(1, len(tokens))
        for tokens in option_tokens
    ]
    if max(scores) <= 0:
        return False
    return max(range(len(scores)), key=scores.__getitem__) == correct


def _facet_mix(rendered_context: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for kind in FACET_KIND_PATTERN.findall(rendered_context):
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    if not total:
        raise ValueError("cannot summarize an empty question subset")
    ranking_correct = sum(
        record["ranking_predicted_option"] == record["correct_option"] for record in records
    )
    supported = sum(bool(record["ranking_correct_supported"]) for record in records)
    facet_kinds: dict[str, int] = {}
    for record in records:
        for kind, count in _facet_mix(record["rendered_context"]).items():
            facet_kinds[kind] = facet_kinds.get(kind, 0) + count
    injected = sum(facet_kinds.values())
    timeline_facets = sum(
        len(TIMELINE_PATTERN.findall(record["rendered_context"])) for record in records
    )
    normalized = [_length_normalized_ranking(record) for record in records]
    scored = [value for value in normalized if value is not None]
    stance_visible = sum(
        bool(_tokens(record["rendered_context"]) & STANCE_TOKENS) for record in records
    )
    return {
        "question_count": total,
        "ranking_accuracy": ranking_correct / total,
        # Same scorer, length-normalized. Diagnostic only: it is reported next to
        # the runner metric so a length artifact cannot be read as a memory result.
        "ranking_accuracy_length_normalized": (
            sum(scored) / len(scored) if scored else None
        ),
        # Whether the injected context expresses preference direction at all. A
        # stance-dependent question cannot be answered from a context without it.
        "stance_token_coverage": stance_visible / total,
        "correct_support_rate": supported / total,
        "mrr": sum(record["ranking_reciprocal_rank"] for record in records) / total,
        "mean_facet_count": sum(record["facet_count"] for record in records) / total,
        "degraded_count": sum(bool(record.get("degraded")) for record in records),
        "facet_kind_totals": dict(sorted(facet_kinds.items())),
        "facet_kind_share": {
            kind: count / injected for kind, count in sorted(facet_kinds.items())
        }
        if injected
        else {},
        "mean_preference_facets_per_question": facet_kinds.get("preference", 0) / total,
        "timeline_facets": timeline_facets,
        "timeline_question_coverage": sum(
            bool(TIMELINE_PATTERN.search(record["rendered_context"])) for record in records
        )
        / total,
    }


def _delta(baseline: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    normalized_delta = None
    if (
        baseline["ranking_accuracy_length_normalized"] is not None
        and treatment["ranking_accuracy_length_normalized"] is not None
    ):
        normalized_delta = (
            treatment["ranking_accuracy_length_normalized"]
            - baseline["ranking_accuracy_length_normalized"]
        )
    return {
        "ranking_accuracy": treatment["ranking_accuracy"] - baseline["ranking_accuracy"],
        "ranking_accuracy_length_normalized": normalized_delta,
        "stance_token_coverage": (
            treatment["stance_token_coverage"] - baseline["stance_token_coverage"]
        ),
        "correct_support_rate": treatment["correct_support_rate"]
        - baseline["correct_support_rate"],
        "mrr": treatment["mrr"] - baseline["mrr"],
        "mean_preference_facets_per_question": (
            treatment["mean_preference_facets_per_question"]
            - baseline["mean_preference_facets_per_question"]
        ),
    }


def _identical_retrieval_questions(
    baseline: dict[str, Any], treatment: dict[str, Any], question_ids: list[str]
) -> list[str]:
    """Question IDs whose injected context is byte-identical across both runs.

    ``user_id`` and ``revision`` carry the run ID and per-run revision counter, so
    they are normalized out before comparison.
    """
    identical: list[str] = []
    for question_id in question_ids:
        left = re.sub(
            r'(user_id="[^"]*"|revision="[^"]*")',
            "",
            baseline["records"][question_id]["rendered_context"],
        )
        right = re.sub(
            r'(user_id="[^"]*"|revision="[^"]*")',
            "",
            treatment["records"][question_id]["rendered_context"],
        )
        if left == right:
            identical.append(question_id)
    return identical


def compare(
    baseline_path: Path, treatment_path: Path, *, allow_subset: bool = False
) -> dict[str, Any]:
    baseline = _load(baseline_path)
    treatment = _load(treatment_path)
    _require_comparable(baseline, treatment, allow_subset=allow_subset)
    question_ids = sorted(set(baseline["records"]) & set(treatment["records"]))
    question_types = sorted(
        {baseline["records"][question_id]["question_type"] for question_id in question_ids}
    )

    by_question_type: dict[str, Any] = {}
    for question_type in question_types:
        subset = [
            question_id
            for question_id in question_ids
            if baseline["records"][question_id]["question_type"] == question_type
        ]
        baseline_summary = _summarize([baseline["records"][item] for item in subset])
        treatment_summary = _summarize([treatment["records"][item] for item in subset])
        by_question_type[question_type] = {
            "baseline": baseline_summary,
            "treatment": treatment_summary,
            "delta": _delta(baseline_summary, treatment_summary),
        }

    overall_baseline = _summarize([baseline["records"][item] for item in question_ids])
    overall_treatment = _summarize([treatment["records"][item] for item in question_ids])
    identical = _identical_retrieval_questions(baseline, treatment, question_ids)
    fingerprints = {
        "baseline": baseline["service_fingerprint"],
        "treatment": treatment["service_fingerprint"],
    }
    return {
        "comparison": "Meno deterministic retrieval ranking comparison",
        "schema_version": "retrieval-ranking-comparison-v1",
        "answer_model_calls": 0,
        "comparability": (
            "Ranking metrics are computed during retrieval, before any answer model call, "
            "so this lane carries no answer-model variance."
        ),
        "dataset": {
            "questions_sha256": baseline["questions_sha256"],
            "contexts_sha256": baseline["contexts_sha256"],
            "question_count": len(question_ids),
            "baseline_record_count": len(baseline["records"]),
            "treatment_record_count": len(treatment["records"]),
            "compared_subset": len(question_ids) != len(baseline["records"])
            or len(question_ids) != len(treatment["records"]),
        },
        "inputs": {
            "baseline": {
                key: baseline[key] for key in ("path", "sha256", "run_id", "service_fingerprint")
            },
            "treatment": {
                key: treatment[key] for key in ("path", "sha256", "run_id", "service_fingerprint")
            },
        },
        "service_fingerprint_available": all(value is not None for value in fingerprints.values()),
        "fingerprint_warnings": [
            f"{side} artifact does not record service_fingerprint; the feature flag state of "
            "this run is not self-evidenced"
            for side, value in sorted(fingerprints.items())
            if value is None
        ],
        "retrieval_identical_questions": len(identical),
        "retrieval_changed_questions": len(question_ids) - len(identical),
        "overall": {
            "baseline": overall_baseline,
            "treatment": overall_treatment,
            "delta": _delta(overall_baseline, overall_treatment),
        },
        "by_question_type": by_question_type,
    }


def _format_table(report: dict[str, Any]) -> str:
    header = (
        f"{'question_type':46} {'n':>4} {'base':>7} {'treat':>7} {'delta':>8} "
        f"{'norm_b':>7} {'norm_t':>7} {'stance_b':>8} {'stance_t':>8}"
    )
    lines = [header, "-" * len(header)]
    rows = [
        *sorted(report["by_question_type"].items()),
        ("OVERALL", report["overall"]),
    ]

    def show(value: float | None) -> str:
        return "    n/a" if value is None else f"{value:>7.4f}"

    for name, entry in rows:
        baseline, treatment, delta = entry["baseline"], entry["treatment"], entry["delta"]
        lines.append(
            f"{name:46} {baseline['question_count']:>4} "
            f"{baseline['ranking_accuracy']:>7.4f} {treatment['ranking_accuracy']:>7.4f} "
            f"{delta['ranking_accuracy']:>+8.4f} "
            f"{show(baseline['ranking_accuracy_length_normalized'])} "
            f"{show(treatment['ranking_accuracy_length_normalized'])} "
            f"{baseline['stance_token_coverage']:>8.3f} "
            f"{treatment['stance_token_coverage']:>8.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = compare(args.baseline, args.treatment, allow_subset=args.allow_subset)
    print(_format_table(report))
    print()
    if report["dataset"]["compared_subset"]:
        print(
            f"compared the {report['dataset']['question_count']}-question intersection "
            f"(baseline={report['dataset']['baseline_record_count']}, "
            f"treatment={report['dataset']['treatment_record_count']})"
        )
    print(
        f"retrieval identical on {report['retrieval_identical_questions']}"
        f"/{report['dataset']['question_count']} questions; "
        f"changed on {report['retrieval_changed_questions']}"
    )
    for warning in report["fingerprint_warnings"]:
        print(f"warning: {warning}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
