"""Curated retrieval set viability probe (Gate Charter #13, #14 pre-check).

Charter §3 requires this before any annotation: a curated set built on a corpus
whose deciding evidence is not findable from the query would reproduce the 0.087
discriminability trap, and the annotation cost would buy nothing.

This probe does **not** produce relevance labels and does not score Recall@10 or
nDCG@10. It answers one question — *is a curated set on this corpus worth building,
and at what cost* — and reports the three facts that decide it:

1. **Can the corpus supply labels without reading the answer?** PersonaMem ships
   `distance_to_ref_proportion_in_context`, which resolves to a message index for
   all 589 questions with exact `context_length_in_letters` agreement. That is
   dataset provenance, not answer-reading, so it is admissible under Charter §3 rule 2 —
   but only if it is actually precise enough to be a label.
2. **How precise is it?** Measured against a random-message baseline over the same
   contexts. A pointer that barely beats random is a noisy label: Recall@10 computed
   from it would mostly measure label noise, not retrieval quality.
3. **What would annotation cost?** Derived from the resolution requirement, not
   guessed: distinguishing 0.90 from 0.85 at the SPEC threshold needs a specific
   number of labeled queries.

Deterministic, read-only, no answer model and no embedding provider. Costs nothing
but the corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any

from benchmarks.run_personamem_e2e import _parse_options
from meno.service import _content_tokens

ROOT = Path(__file__).resolve().parent.parent

# SPEC :1340-1372 target for #13. The probe reports what sample size is needed to
# tell this apart from a nearby value, rather than assuming any size is fine.
RECALL_TARGET = 0.90
# The effect a curated set must be able to resolve: passing at 0.90 is meaningless
# if the set cannot distinguish 0.90 from 0.85.
RESOLVABLE_DELTA = 0.05

# Below this, a label pointer is too noisy to annotate against: a Recall@10 built on
# it would move with label error rather than with retrieval quality. Set at roughly
# halfway between the observed random baseline and a usable pointer.
USABLE_LABEL_RECALL = 0.60
# The 0.087 figure from probe_discriminability.py: the share of questions whose
# decisive tokens appear in the query at all. Recorded so this probe's own numbers
# are read against the trap they exist to detect.
KNOWN_QUERY_SIGNAL = 0.087

# Window sizes tried around the resolved reference index. A pointer that only works
# when widened is a region, not a label.
WINDOW_SIZES = (0, 1, 2, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contexts",
        type=Path,
        default=ROOT / "artifacts" / "benchmarks" / "raw" / "shared_contexts_32k.jsonl",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "artifacts" / "benchmarks" / "raw" / "questions_32k.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_corpus(
    contexts_path: Path, questions_path: Path
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]], dict[str, str]]:
    context_raw = contexts_path.read_bytes()
    contexts: dict[str, list[dict[str, str]]] = {}
    for line in context_raw.decode("utf-8").splitlines():
        contexts.update(json.loads(line))
    question_raw = questions_path.read_bytes()
    with questions_path.open(newline="", encoding="utf-8") as handle:
        questions = list(csv.DictReader(handle))
    provenance = {
        "contexts_file": contexts_path.name,
        "contexts_sha256": hashlib.sha256(context_raw).hexdigest(),
        "questions_file": questions_path.name,
        "questions_sha256": hashlib.sha256(question_raw).hexdigest(),
    }
    return contexts, questions, provenance


def resolve_reference_index(
    row: dict[str, str], messages: list[dict[str, str]], end: int
) -> tuple[int | None, bool]:
    """Locate the dataset's reference message from its own distance metadata.

    Returns the index and whether ``context_length_in_letters`` agreed. The
    agreement check is what makes this a resolution rather than a guess: if the
    letter count does not match, the proportion cannot be trusted to locate
    anything.
    """
    letters = sum(len(message["content"]) for message in messages[:end])
    declared = int(row["context_length_in_letters"])
    consistent = letters == declared
    raw_proportion = row["distance_to_ref_proportion_in_context"].strip().rstrip("%")
    try:
        proportion = float(raw_proportion) / 100
    except ValueError:
        return None, consistent
    # The distance is measured backwards from the question, so the reference sits
    # at (1 - proportion) of the way through the context.
    target = letters * (1 - proportion)
    cumulative = 0
    for index, message in enumerate(messages[:end]):
        cumulative += len(message["content"])
        if cumulative >= target:
            return index, consistent
    return (end - 1 if end else None), consistent


def decisive_tokens(row: dict[str, str]) -> set[str]:
    """Tokens unique to the correct option.

    These are what a retrieved set must supply for the correct option to win, so
    they are the only tokens whose presence indicates real evidence.
    """
    options = _parse_options(row["all_options"])
    correct = ord(row["correct_answer"].strip("()").lower()) - ord("a")
    if not 0 <= correct < len(options):
        return set()
    per_option = [
        _content_tokens(re.sub(r"^\([a-d]\)\s*", "", option)) for option in options
    ]
    rivals = set().union(*[t for i, t in enumerate(per_option) if i != correct])
    return per_option[correct] - rivals


def _window_tokens(
    messages: list[dict[str, str]], index: int, radius: int, end: int
) -> set[str]:
    tokens: set[str] = set()
    for position in range(index - radius, index + radius + 1):
        if 0 <= position < end:
            tokens |= _content_tokens(messages[position]["content"])
    return tokens


def measure_label_quality(
    contexts: dict[str, list[dict[str, str]]],
    questions: list[dict[str, str]],
    seed: int,
) -> dict[str, Any]:
    """How precisely the dataset's own metadata points at the deciding evidence."""
    rng = random.Random(seed)
    resolved = 0
    inconsistent = 0
    missing_context = 0
    no_decisive = 0
    per_window: dict[int, list[float]] = {radius: [] for radius in WINDOW_SIZES}
    random_baseline: list[float] = []
    beats_random: dict[int, int] = dict.fromkeys(WINDOW_SIZES, 0)
    reference_roles: dict[str, int] = {}
    query_signal: list[float] = []

    for row in questions:
        context_id = row["shared_context_id"]
        messages = contexts.get(context_id)
        if messages is None:
            missing_context += 1
            continue
        end = int(row["end_index_in_shared_context"])
        index, consistent = resolve_reference_index(row, messages, end)
        if not consistent:
            inconsistent += 1
        if index is None:
            continue
        resolved += 1
        role = str(messages[index].get("role", "unknown"))
        reference_roles[role] = reference_roles.get(role, 0) + 1

        decisive = decisive_tokens(row)
        if not decisive:
            no_decisive += 1
            continue
        # Same denominator for every measurement, so windows and the baseline are
        # directly comparable.
        baseline = (
            len(decisive & _content_tokens(messages[rng.randrange(end)]["content"]))
            / len(decisive)
            if end
            else 0.0
        )
        random_baseline.append(baseline)
        for radius in WINDOW_SIZES:
            recall = (
                len(decisive & _window_tokens(messages, index, radius, end))
                / len(decisive)
            )
            per_window[radius].append(recall)
            if recall > baseline:
                beats_random[radius] += 1
        query_signal.append(
            len(decisive & _content_tokens(row["user_question_or_message"]))
            / len(decisive)
        )

    scored = len(random_baseline)
    windows = {
        f"radius_{radius}": {
            "messages_per_label": 2 * radius + 1,
            "mean_decisive_recall": (
                statistics.mean(values) if values else 0.0
            ),
            "beats_random_baseline": beats_random[radius],
            "beats_random_share": beats_random[radius] / scored if scored else 0.0,
            # A pointer only usable when widened labels a region, not a message.
            "usable_as_label": (
                statistics.mean(values) >= USABLE_LABEL_RECALL if values else False
            ),
        }
        for radius, values in per_window.items()
    }
    return {
        "questions_total": len(questions),
        "reference_resolved": resolved,
        "resolution_rate": resolved / len(questions) if questions else 0.0,
        "letter_count_inconsistent": inconsistent,
        "missing_context": missing_context,
        "questions_without_decisive_tokens": no_decisive,
        "questions_scored": scored,
        "reference_role_distribution": dict(sorted(reference_roles.items())),
        "random_baseline_decisive_recall": (
            statistics.mean(random_baseline) if random_baseline else 0.0
        ),
        "windows": windows,
        "query_signal_decisive_recall": (
            statistics.mean(query_signal) if query_signal else 0.0
        ),
        "known_query_signal_reference": KNOWN_QUERY_SIGNAL,
    }


def annotation_cost(label_quality: dict[str, Any]) -> dict[str, Any]:
    """Labeled-query count needed to resolve the SPEC threshold.

    Normal approximation for a proportion at ``RECALL_TARGET``: to separate 0.90
    from 0.85 with 95% confidence, n >= (1.96/delta)^2 * p(1-p).
    """
    variance = RECALL_TARGET * (1 - RECALL_TARGET)
    required = int((1.96 / RESOLVABLE_DELTA) ** 2 * variance) + 1
    # Under Charter §3 rule 2 an annotator may see only the query and the user's history,
    # so each label means reading that history rather than checking a marked span.
    best = max(
        label_quality["windows"].values(), key=lambda item: item["mean_decisive_recall"]
    )
    return {
        "recall_target": RECALL_TARGET,
        "resolvable_delta": RESOLVABLE_DELTA,
        "required_labeled_queries": required,
        "rationale": (
            f"Separating Recall@10 {RECALL_TARGET} from "
            f"{RECALL_TARGET - RESOLVABLE_DELTA:.2f} at 95% confidence needs "
            f"{required} labeled queries. Fewer cannot evidence the SPEC threshold, "
            "the same sample-resolution rule the grounding lane applies."
        ),
        "best_metadata_window_recall": best["mean_decisive_recall"],
        "best_metadata_window_messages": best["messages_per_label"],
        "metadata_can_replace_annotation": best["usable_as_label"],
    }


def verdict(label_quality: dict[str, Any], cost: dict[str, Any]) -> dict[str, Any]:
    """Recommend, with the reason stated as a measurement rather than a preference.

    Charter §3 rule 3 is decisive and is therefore checked first: if the deciding
    information is not reachable from the query, no labeling scheme rescues the set.
    Label quality only matters once that gate is passed, so reporting them in the
    other order would bury the finding that settles the question.
    """
    best_recall = cost["best_metadata_window_recall"]
    baseline = label_quality["random_baseline_decisive_recall"]
    signal_above_random = best_recall > baseline
    query_signal = label_quality["query_signal_decisive_recall"]
    # At or below the level that produced the original trap, better labels cannot
    # help: the ceiling is on the query side.
    query_signal_at_trap = query_signal <= KNOWN_QUERY_SIGNAL * 1.5
    reasons: list[str] = []

    if query_signal_at_trap:
        reasons.append(
            f"Decisive tokens appear in the query for only {query_signal:.4f} of "
            f"questions -- indistinguishable from the {KNOWN_QUERY_SIGNAL} that made "
            "PersonaMem unusable as a gate (Charter §0). Charter §3 rule 3 says to stop "
            "here: this is the same corpus, so a curated set over it inherits the "
            "same ceiling no matter how the labels are produced."
        )

    if not signal_above_random:
        reasons.append(
            "The dataset's reference metadata does not beat a random message from "
            "the same context, so it carries no usable localization signal."
        )
    elif not cost["metadata_can_replace_annotation"]:
        reasons.append(
            f"Reference metadata localizes evidence better than random "
            f"({best_recall:.4f} vs {baseline:.4f}) but below the "
            f"{USABLE_LABEL_RECALL} needed to stand in for a label, and only at a "
            f"{cost['best_metadata_window_messages']}-message window -- that labels a "
            "region, not a claim. Recall@10 built on it would track label noise more "
            "than retrieval quality."
        )
    else:
        reasons.append(
            f"Reference metadata reaches {best_recall:.4f} decisive recall, enough to "
            "seed labels for human confirmation rather than annotation from scratch."
        )

    if query_signal_at_trap:
        recommendation = "CORPUS_UNSUITABLE_QUERY_SIGNAL"
    elif cost["metadata_can_replace_annotation"]:
        recommendation = "ANNOTATE_WITH_SEEDS"
    elif signal_above_random:
        recommendation = "MANUAL_ANNOTATION_REQUIRED"
    else:
        recommendation = "CORPUS_UNSUITABLE_NO_LABEL_SOURCE"

    return {
        "query_signal_at_trap_level": bool(query_signal_at_trap),
        "metadata_usable_as_gold_labels": bool(cost["metadata_can_replace_annotation"]),
        "metadata_beats_random": bool(signal_above_random),
        "annotation_required": not cost["metadata_can_replace_annotation"],
        "reasons": reasons,
        "recommendation": recommendation,
        "next_step": (
            "Source a corpus whose questions name what they ask about, then re-run "
            "this probe before annotating. Spending the annotation budget on this "
            "corpus would repeat the four interventions in Charter §0, which each "
            "improved their mechanism while the score stayed flat."
            if query_signal_at_trap
            else f"Budget {cost['required_labeled_queries']} labeled queries under the "
            "Charter §3 rule 2 constraint (annotator sees query + user history only)."
        ),
        "decision_is_not_engineering": (
            "Whether to spend "
            f"{cost['required_labeled_queries']} manual annotations is a cost "
            "decision, not an engineering one. This probe supplies the inputs; it "
            "deliberately does not build a set or emit labels."
        ),
    }


def run_probe(
    contexts_path: Path, questions_path: Path, seed: int = 0
) -> dict[str, Any]:
    contexts, questions, provenance = load_corpus(contexts_path, questions_path)
    label_quality = measure_label_quality(contexts, questions, seed)
    cost = annotation_cost(label_quality)
    return {
        "probe": "Meno curated retrieval set viability",
        "gate_charter_items": [13, 14],
        "schema_version": "curated-viability-probe-v1",
        "answer_model_calls": 0,
        "embedding_provider_calls": 0,
        "seed": seed,
        "provenance": provenance,
        "label_quality": label_quality,
        "annotation_cost": cost,
        "verdict": verdict(label_quality, cost),
        "scope": (
            "A pre-check, not a gate. It emits no relevance labels and computes no "
            "Recall@10 or nDCG@10; #13 and #14 stay BUILD until a curated set with "
            "admissible labels exists."
        ),
    }


def main() -> None:
    args = parse_args()
    report = run_probe(args.contexts, args.questions, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    quality = report["label_quality"]
    print(
        f"resolved {quality['reference_resolved']}/{quality['questions_total']} "
        f"references, {quality['letter_count_inconsistent']} letter-count mismatches"
    )
    print(f"random baseline decisive recall: {quality['random_baseline_decisive_recall']:.4f}")
    header = f"{'window':>10} {'msgs':>5} {'recall':>8} {'beats_random':>13} {'usable':>7}"
    print(header)
    print("-" * len(header))
    for name, row in quality["windows"].items():
        print(
            f"{name:>10} {row['messages_per_label']:>5} "
            f"{row['mean_decisive_recall']:>8.4f} "
            f"{row['beats_random_share']:>12.1%} {row['usable_as_label']!s:>7}"
        )
    print(f"\nquery signal: {quality['query_signal_decisive_recall']:.4f}")
    print(f"recommendation: {report['verdict']['recommendation']}")
    for reason in report["verdict"]["reasons"]:
        print(f"  - {reason}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
