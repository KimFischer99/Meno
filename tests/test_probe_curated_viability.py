from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmarks import probe_curated_viability as probe

CONTEXTS = Path("artifacts/benchmarks/raw/shared_contexts_32k.jsonl")
QUESTIONS = Path("artifacts/benchmarks/raw/questions_32k.csv")

corpus_available = pytest.mark.skipif(
    not (CONTEXTS.is_file() and QUESTIONS.is_file()),
    reason="PersonaMem corpus is gitignored; run where it is present",
)


def _quality(
    *, query_signal: float, label_recall: float, usable: bool, messages: int = 1
) -> dict:
    return {
        "random_baseline_decisive_recall": 0.07,
        "query_signal_decisive_recall": query_signal,
        "windows": {
            "radius_0": {
                "mean_decisive_recall": label_recall,
                "messages_per_label": messages,
                "usable_as_label": usable,
            }
        },
    }


def test_query_signal_at_trap_level_overrides_label_quality() -> None:
    """Charter §3.3: if the query cannot reach the evidence, labels cannot help.

    This must be checked before label quality, or a corpus with clean labels and no
    query signal would be recommended for annotation.
    """
    quality = _quality(query_signal=0.087, label_recall=0.95, usable=True)

    verdict = probe.verdict(quality, probe.annotation_cost(quality))

    assert verdict["query_signal_at_trap_level"] is True
    assert verdict["recommendation"] == "CORPUS_UNSUITABLE_QUERY_SIGNAL"
    # Even though the labels themselves would have been usable.
    assert verdict["metadata_usable_as_gold_labels"] is True
    assert "0.087" in verdict["reasons"][0]


def test_high_signal_with_clean_labels_recommends_seeded_annotation() -> None:
    quality = _quality(query_signal=0.62, label_recall=0.81, usable=True)

    verdict = probe.verdict(quality, probe.annotation_cost(quality))

    assert verdict["query_signal_at_trap_level"] is False
    assert verdict["recommendation"] == "ANNOTATE_WITH_SEEDS"


def test_high_signal_with_noisy_labels_requires_manual_annotation() -> None:
    quality = _quality(query_signal=0.62, label_recall=0.30, usable=False)

    verdict = probe.verdict(quality, probe.annotation_cost(quality))

    assert verdict["recommendation"] == "MANUAL_ANNOTATION_REQUIRED"
    assert verdict["annotation_required"] is True
    assert "labeled queries" in verdict["next_step"]


def test_no_label_source_is_distinguished_from_no_query_signal() -> None:
    """The two unsuitable verdicts have different remedies and must not be merged."""
    quality = _quality(query_signal=0.62, label_recall=0.02, usable=False)

    verdict = probe.verdict(quality, probe.annotation_cost(quality))

    assert verdict["recommendation"] == "CORPUS_UNSUITABLE_NO_LABEL_SOURCE"
    assert verdict["metadata_beats_random"] is False


def test_required_sample_size_follows_the_resolvable_delta() -> None:
    """Sample size is derived, not chosen: same rule as the grounding lane."""
    cost = probe.annotation_cost(_quality(query_signal=0.5, label_recall=0.5, usable=False))

    assert cost["recall_target"] == 0.90
    assert cost["resolvable_delta"] == 0.05
    # (1.96/0.05)^2 * 0.9 * 0.1 = 138.3 -> 139
    assert cost["required_labeled_queries"] == 139


def test_decisive_tokens_exclude_shared_option_wording() -> None:
    """Only tokens unique to the correct option indicate real evidence."""
    row = {
        "all_options": json.dumps(
            [
                "(a) shared preamble about music with saxophone",
                "(b) shared preamble about music with trumpet",
                "(c) shared preamble about music with clarinet",
                "(d) shared preamble about music with bassoon",
            ]
        ),
        "correct_answer": "(c)",
    }

    decisive = probe.decisive_tokens(row)

    assert "clarinet" in decisive
    assert "saxophone" not in decisive
    # Wording every option shares carries no information about the answer.
    assert "shared" not in decisive
    assert "music" not in decisive


def test_reference_resolution_reports_letter_count_disagreement() -> None:
    """Without the letter-count check the proportion locates nothing reliably."""
    messages = [{"role": "user", "content": "abcde"}, {"role": "user", "content": "fghij"}]
    row = {
        "context_length_in_letters": "999",
        "distance_to_ref_proportion_in_context": "50.00%",
        "end_index_in_shared_context": "2",
    }

    index, consistent = probe.resolve_reference_index(row, messages, 2)

    assert consistent is False
    assert index is not None


def test_reference_resolution_locates_the_proportional_position() -> None:
    messages = [
        {"role": "user", "content": "a" * 100},
        {"role": "user", "content": "b" * 100},
        {"role": "user", "content": "c" * 100},
    ]
    # 66.67% back from the question => one third into the context => index 0.
    row = {
        "context_length_in_letters": "300",
        "distance_to_ref_proportion_in_context": "66.67%",
        "end_index_in_shared_context": "3",
    }

    index, consistent = probe.resolve_reference_index(row, messages, 3)

    assert consistent is True
    assert index == 0


def test_malformed_proportion_is_not_silently_resolved() -> None:
    messages = [{"role": "user", "content": "abc"}]
    row = {
        "context_length_in_letters": "3",
        "distance_to_ref_proportion_in_context": "n/a",
        "end_index_in_shared_context": "1",
    }

    index, consistent = probe.resolve_reference_index(row, messages, 1)

    assert index is None
    assert consistent is True


@corpus_available
def test_probe_reports_the_measured_verdict_on_the_real_corpus() -> None:
    """The frozen finding: this corpus is unsuitable for a curated set."""
    report = probe.run_probe(CONTEXTS, QUESTIONS)
    quality = report["label_quality"]

    assert quality["questions_total"] == 589
    # Dataset metadata is internally consistent, which is why it was worth testing.
    assert quality["reference_resolved"] == 589
    assert quality["letter_count_inconsistent"] == 0
    # But the query signal sits at the trap level measured by probe_discriminability.
    assert quality["query_signal_decisive_recall"] < 0.10
    assert report["verdict"]["recommendation"] == "CORPUS_UNSUITABLE_QUERY_SIGNAL"


@corpus_available
def test_metadata_beats_random_but_only_as_a_region() -> None:
    """A pointer that needs a 9-message window labels a region, not a claim."""
    report = probe.run_probe(CONTEXTS, QUESTIONS)
    quality = report["label_quality"]
    windows = quality["windows"]

    assert (
        windows["radius_0"]["mean_decisive_recall"]
        > quality["random_baseline_decisive_recall"]
    )
    # Recall only rises by widening, and never reaches label quality.
    assert (
        windows["radius_0"]["mean_decisive_recall"]
        < windows["radius_4"]["mean_decisive_recall"]
    )
    assert all(not row["usable_as_label"] for row in windows.values())


@corpus_available
def test_probe_is_deterministic() -> None:
    first = probe.run_probe(CONTEXTS, QUESTIONS, seed=0)
    second = probe.run_probe(CONTEXTS, QUESTIONS, seed=0)

    assert first["label_quality"] == second["label_quality"]
    assert first["verdict"] == second["verdict"]


@corpus_available
def test_probe_emits_no_labels_and_no_retrieval_scores() -> None:
    """It is a pre-check: #13/#14 stay BUILD until an admissible set exists."""
    report = probe.run_probe(CONTEXTS, QUESTIONS)

    assert report["answer_model_calls"] == 0
    assert report["embedding_provider_calls"] == 0
    # No scored metric or label payload anywhere in the structure. Checked over keys
    # rather than the serialized text, because the scope note mentions Recall@10 and
    # nDCG@10 precisely to disclaim computing them.
    def keys(node: object) -> set[str]:
        if isinstance(node, dict):
            return set(node) | {key for value in node.values() for key in keys(value)}
        if isinstance(node, list):
            return {key for value in node for key in keys(value)}
        return set()

    emitted = keys(report)
    assert not {"labels", "relevance_labels", "recall_at_10", "ndcg_at_10"} & emitted
    assert json.dumps(report)  # must stay serializable


@corpus_available
def test_every_question_row_carries_the_metadata_the_probe_relies_on() -> None:
    with QUESTIONS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        assert row["distance_to_ref_proportion_in_context"].endswith("%")
        assert row["context_length_in_letters"].isdigit()
