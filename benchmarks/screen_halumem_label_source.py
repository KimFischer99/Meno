"""Screen HaluMem as a Gate Charter #13/#14 candidate corpus.

Companion to `MENO_DATA_REQUIREMENTS.md` §1.3. Unlike the LongMemEval and LoCoMo
converters, this does **not** emit a corpus, because HaluMem cannot supply the
label the contract requires without the screener inventing it.

Why it stops at the label-source check, before query signal is even measured:

HaluMem's per-question `evidence` names `memory_content` strings -- third-person
summaries the dataset authors wrote ("Martin Mark's birth date is 1996-08-02") --
not message indices. The contract's `relevant_message_indices` needs indices, so a
converter must map each summary back to the source message. That mapping is the
screener's own construction, which Charter §3 rule 2 rules out as a label source:
labels must come from the dataset's provenance or from annotation that never sees
the answer.

The dataset does carry an `event_source` field on memory points, which would be the
principled mapping if it indexed dialogue turns. It does not: it is present on only
100 of 718 memory points and its value is always 0.

This script measures how bad a text-matching fallback would be, so the exclusion
rests on a number rather than an assertion. It reports, over every evidence item:

- **best token overlap** between the memory summary and any user message -- how
  well the source can be recognized at all;
- **top-1 vs top-2 margin** -- whether the best match is *distinguishable* from
  the runner-up. A tie means the constructed label is a coin flip.

A corpus whose mapping is ambiguous for a third of its evidence would produce a
Recall@10 that tracks the screener's matching heuristic. That is the same failure
`probe_curated_viability.py` §3.2 found in PersonaMem's reference metadata, one
step earlier in the pipeline.

Deterministic, read-only, no answer model and no embedding provider. Emits no
message text: findings are counts, so the report is safe to share even when the
corpus is not. HaluMem is CC BY-NC-ND 4.0 and is not redistributed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from meno.service import _content_tokens

# A constructed label is only defensible if the intended source stands out from
# the next candidate. Below this margin the choice is the matcher's, not the data's.
CLEAR_MARGIN = 0.15
# Overlap high enough to call a message the verbatim source of the summary.
VERBATIM_OVERLAP = 0.8
# Ambiguity above this share means a text-matched label set is mostly guesswork.
MAX_TOLERABLE_AMBIGUITY = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="HaluMem-*.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="users to read (0 = all)")
    return parser.parse_args()


def screen(users: list[dict[str, Any]]) -> dict[str, Any]:
    overlaps: list[float] = []
    margins: list[float] = []
    question_types: Counter[str] = Counter()
    evidence_item_types: Counter[str] = Counter()
    questions_total = 0
    questions_without_evidence = 0
    memory_points = 0
    memory_points_with_event_source = 0
    event_source_values: Counter[str] = Counter()
    messages_total = 0

    for user in users:
        user_messages: dict[int, set[str]] = {}
        for session in user["sessions"]:
            for turn in session["dialogue"]:
                index = messages_total
                messages_total += 1
                if turn["role"] == "user":
                    user_messages[index] = _content_tokens(turn["content"])
            for point in session.get("memory_points", []):
                memory_points += 1
                if "event_source" in point:
                    memory_points_with_event_source += 1
                    event_source_values[str(point["event_source"])] += 1

        candidates = [tokens for tokens in user_messages.values() if tokens]
        for session in user["sessions"]:
            for question in session.get("questions", []):
                questions_total += 1
                question_types[question.get("question_type", "unknown")] += 1
                evidence = question.get("evidence") or []
                if not evidence:
                    questions_without_evidence += 1
                    continue
                for item in evidence:
                    evidence_item_types[type(item).__name__] += 1
                    if not isinstance(item, dict):
                        continue
                    summary = _content_tokens(str(item.get("memory_content", "")))
                    if not summary or not candidates:
                        continue
                    scores = sorted(
                        (len(summary & tokens) / len(summary) for tokens in candidates),
                        reverse=True,
                    )
                    overlaps.append(scores[0])
                    margins.append(scores[0] - (scores[1] if len(scores) > 1 else 0.0))

    scored = len(overlaps)
    ambiguous = sum(1 for margin in margins if margin <= 0.0)
    clear = sum(1 for margin in margins if margin >= CLEAR_MARGIN)
    verbatim = sum(1 for overlap in overlaps if overlap >= VERBATIM_OVERLAP)
    ambiguous_share = ambiguous / scored if scored else 1.0
    verbatim_share = verbatim / scored if scored else 0.0

    # The dataset's own pointer would beat text matching, if it pointed anywhere.
    event_source_usable = (
        memory_points_with_event_source == memory_points
        and len(event_source_values) > 1
    )
    # A low ambiguity rate only means something if the summaries are actually
    # recognizable in the dialogue. When most of them match nothing well, a clear
    # top-1 margin just means one message happened to share a stray token.
    matching_is_recognizable = verbatim_share >= 0.5
    usable = event_source_usable or (
        matching_is_recognizable and ambiguous_share <= MAX_TOLERABLE_AMBIGUITY
    )

    reasons = []
    if not event_source_usable:
        reasons.append(
            f"`event_source` cannot serve as the message pointer: present on "
            f"{memory_points_with_event_source}/{memory_points} memory points with "
            f"{len(event_source_values)} distinct value(s) "
            f"{dict(event_source_values.most_common(5))}."
        )
    if scored and ambiguous_share > MAX_TOLERABLE_AMBIGUITY:
        reasons.append(
            f"Text matching is ambiguous for {ambiguous}/{scored} evidence items "
            f"({ambiguous_share:.1%}): the best-matching user message ties with the "
            "runner-up, so the resulting index would be the matcher's choice rather "
            "than the dataset's. Charter §3 rule 2 excludes screener-invented labels."
        )
    if scored and not matching_is_recognizable:
        reasons.append(
            f"Only {verbatim}/{scored} summaries overlap a single user message at "
            f">={VERBATIM_OVERLAP}; `memory_content` is a third-person rewrite of "
            "first-person dialogue, so overlap is low even when the match is right. "
            "With recognition this weak, a clear top-1 margin is not evidence that "
            "the right message was found."
        )

    return {
        "screen": "HaluMem label-source viability",
        "schema_version": "halumem-label-source-screen-v1",
        "requirements_document": "MENO_DATA_REQUIREMENTS.md",
        "answer_model_calls": 0,
        "embedding_provider_calls": 0,
        "corpus": {
            "users": len(users),
            "messages": messages_total,
            "memory_points": memory_points,
            "questions": questions_total,
            "questions_without_evidence": questions_without_evidence,
            "question_types": dict(question_types.most_common()),
            "evidence_item_types": dict(evidence_item_types),
            # Memory Boundary questions are unanswerable by construction, which is
            # what the contract's abstain path needs -- worth recording even though
            # the corpus is excluded for a different reason.
            "abstain_candidate_share": (
                questions_without_evidence / questions_total if questions_total else 0.0
            ),
        },
        "dataset_pointer": {
            "field": "event_source",
            "memory_points_with_field": memory_points_with_event_source,
            "distinct_values": dict(event_source_values.most_common(5)),
            "usable_as_message_index": event_source_usable,
        },
        "text_matching_fallback": {
            "evidence_items_scored": scored,
            "mean_best_overlap": statistics.fmean(overlaps) if overlaps else 0.0,
            "median_best_overlap": statistics.median(overlaps) if overlaps else 0.0,
            "verbatim_matches": verbatim,
            "verbatim_share": verbatim_share,
            "matching_is_recognizable": matching_is_recognizable,
            "ambiguous_top1_ties_top2": ambiguous,
            "ambiguous_share": ambiguous_share,
            "clear_by_margin": clear,
            "clear_share": clear / scored if scored else 0.0,
            "median_margin": statistics.median(margins) if margins else 0.0,
            "clear_margin_threshold": CLEAR_MARGIN,
            "max_tolerable_ambiguity": MAX_TOLERABLE_AMBIGUITY,
        },
        "verdict": {
            "label_source_usable": usable,
            "recommendation": (
                "PROCEED_TO_QUERY_SIGNAL" if usable else "CORPUS_UNSUITABLE_NO_LABEL_SOURCE"
            ),
            "reasons": reasons,
            "next_step": (
                "Convert to the §1.3 contract and run validate_external_corpus.py."
                if usable
                else "Do not convert. The contract needs message indices and this "
                "dataset only supplies rewritten summaries; any index would be "
                "authored by the screener, which Charter §3 rule 2 excludes. Query "
                "signal is deliberately not measured -- it would be measured against "
                "invented labels."
            ),
        },
        "scope": (
            "Label-source viability only. Says nothing about HaluMem's fitness for "
            "its own purpose (memory hallucination), only that it cannot supply "
            "Meno's Recall@10 gold labels."
        ),
        "license_note": "HaluMem is CC BY-NC-ND 4.0 and is not redistributed here.",
    }


def main() -> None:
    args = parse_args()
    raw = args.data.read_bytes()
    users = [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        users = users[: args.limit]
    report = screen(users)
    report["provenance"] = {
        "file": args.data.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "users_read": len(users),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    fallback = report["text_matching_fallback"]
    print(
        f"users={report['corpus']['users']} questions={report['corpus']['questions']} "
        f"evidence_items={fallback['evidence_items_scored']}"
    )
    print(
        f"\nambiguous (top-1 ties top-2) = {fallback['ambiguous_share']:.1%}"
        f"  verbatim = {fallback['verbatim_share']:.1%}"
        f"  median margin = {fallback['median_margin']:.3f}"
    )
    print(f"\nrecommendation: {report['verdict']['recommendation']}")
    for reason in report["verdict"]["reasons"]:
        print(f"  - {reason}")
    print(f"\n{report['verdict']['next_step']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
