"""Local diagnostic for the layered-retrieval injection budget.

The read path assembles the injected context in three stages, each of which was
kind-blind: similarity scoring, the ``max_facets`` cut, and the render char budget.
This lane builds a corpus calibrated to the value-length distribution measured in the
production PersonaMem retrieval cache, then retrieves with layered retrieval off and
on to isolate what each stage does to state-layer visibility.

Calibration targets (``vps-phaseb-baseline-retrieval-cache-formal-hy3.json``, 589
questions, 5894 rendered facets):

* episodic value length: mean 263.3 chars (median 281)
* preference value length: mean 87.5 chars (median 77)
* injected mix: 76.8% episodic
* 12.0 facets selected per question, 10.0 actually rendered

Deterministic component diagnostic: no answer model, no embedding provider. It can
block a production decision but never grant one.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from meno.api import build_service
from meno.config import Settings
from meno.schemas import IngestRequest, RetrieveRequest
from meno.service import STATE_LAYER_KINDS
from tests.fakes import TestEmbedder

EMBEDDING_DIMENSION = 256

# Preference utterances phrased so the extractor recognizes them, one per slot in
# SLOT_LEXICON so they occupy distinct semantic keys instead of superseding.
PREFERENCE_UTTERANCES = (
    ("answer_style", "I prefer concise answers that lead with the conclusion"),
    ("beverage", "I prefer dark roast coffee in the morning"),
    ("programming_language", "I prefer Python for data analysis work"),
    ("music", "I prefer jazz when I am concentrating"),
    ("food", "I prefer spicy food when eating out"),
    ("sport", "I prefer swimming over running for exercise"),
    ("color", "I prefer blue for interface accents"),
)

# Verbatim-style conversational turns padded to the measured episodic length band.
EPISODIC_TEMPLATES = (
    (
        "I spent the weekend at a hands-on cooking class downtown and the instructor kept "
        "returning to how much the resting step matters, which caught me off guard because I "
        "had always rushed it at home and assumed it changed very little about the result"
    ),
    (
        "Over the last few months I have been rearranging my morning routine so that the "
        "quiet block comes before any meetings, and the difference in how much I actually "
        "finish before noon has been larger than I expected it to be when I started"
    ),
    (
        "I joined a reading group that meets every other week, and while the discussions are "
        "good I have noticed the pace is faster than I can comfortably keep up with given "
        "everything else going on, so I am still deciding whether to continue with it"
    ),
    (
        "The team switched the review process around so that design feedback lands earlier, "
        "and it has meant fewer late reversals, though it does front-load a lot more reading "
        "onto the first couple of days of any given cycle than the old arrangement did"
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=40)
    parser.add_argument("--episodic-per-question", type=int, default=14)
    parser.add_argument("--max-facets", type=int, default=12)
    parser.add_argument("--max-rendered-tokens", type=int, default=800)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _build_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index in range(args.questions):
        utterances = [value for _slot, value in PREFERENCE_UTTERANCES]
        for episode in range(args.episodic_per_question):
            template = EPISODIC_TEMPLATES[episode % len(EPISODIC_TEMPLATES)]
            # Vary the tail so each episode is a distinct claim, not a duplicate.
            utterances.append(f"{template}, and that was session {index}-{episode}")
        cases.append(
            {
                "case_id": f"case-{index:04d}",
                "question": (
                    "How should you format your answers for me, and what should you "
                    "recommend given what I like?"
                ),
                "utterances": utterances,
            }
        )
    return cases


def _run(cases: list[dict[str, Any]], args: argparse.Namespace, **overrides: Any) -> dict[str, Any]:
    directory = Path(tempfile.mkdtemp(prefix="meno-layered-"))
    settings = Settings(
        database_url=f"sqlite:///{directory / 'meno.sqlite3'}",
        vector_mode="memory",
        embedding_dimension=EMBEDDING_DIMENSION,
        **overrides,
    )
    service = build_service(settings, embedder=TestEmbedder(EMBEDDING_DIMENSION))
    selected = 0
    rendered = 0
    selected_kinds: Counter[str] = Counter()
    rendered_kinds: Counter[str] = Counter()
    truncated_questions = 0
    value_lengths: dict[str, list[int]] = {"episodic": [], "preference": []}
    try:
        for case in cases:
            user_id = f"diag:{case['case_id']}"
            for index, utterance in enumerate(case["utterances"]):
                service.ingest(
                    IngestRequest(
                        user_id=user_id,
                        event_id=f"{case['case_id']}-{index}",
                        source={"type": "hermes_turn"},
                        content={"role": "user", "text": utterance},
                        consent_scope=["personalization", "task_planning"],
                    ),
                    f"key-{case['case_id']}-{index}",
                )
            service.process_outbox()
            response = service.retrieve(
                RetrieveRequest(
                    user_id=user_id,
                    purpose="response_personalization",
                    context={"query": case["question"]},
                    constraints={
                        "max_facets": args.max_facets,
                        "max_rendered_tokens": args.max_rendered_tokens,
                        "min_confidence": 0.5,
                    },
                )
            )
            selected += len(response.facets)
            for facet in response.facets:
                selected_kinds[facet.kind] += 1
                if facet.kind in value_lengths:
                    value_lengths[facet.kind].append(len(str(facet.value)))
            lines = [
                line for line in response.rendered_context.splitlines() if line.startswith("- [")
            ]
            rendered += len(lines)
            if len(lines) < len(response.facets):
                truncated_questions += 1
            for line in lines:
                kind = line.removeprefix("- [").split("]")[0].split(" ")[0].split(":")[0]
                rendered_kinds[kind] += 1
    finally:
        service.close()
        shutil.rmtree(directory, ignore_errors=True)
    injected = sum(rendered_kinds.values())
    state_rendered = sum(
        count for kind, count in rendered_kinds.items() if kind in STATE_LAYER_KINDS
    )
    return {
        "questions": len(cases),
        "facets_selected": selected,
        "facets_rendered": rendered,
        "facets_dropped_by_render": selected - rendered,
        "render_drop_rate": (selected - rendered) / selected if selected else 0.0,
        "questions_with_dropped_facets": truncated_questions,
        "selected_kind_totals": dict(sorted(selected_kinds.items())),
        "rendered_kind_totals": dict(sorted(rendered_kinds.items())),
        "state_layer_share_rendered": state_rendered / injected if injected else 0.0,
        "mean_state_facets_rendered_per_question": state_rendered / len(cases),
        "mean_value_length": {
            kind: (statistics.mean(lengths) if lengths else 0.0)
            for kind, lengths in sorted(value_lengths.items())
        },
    }


def main() -> None:
    args = parse_args()
    cases = _build_cases(args)
    baseline = _run(cases, args)
    layered = _run(cases, args, layered_retrieval_enabled=True)
    report = {
        "diagnostic": "Meno layered retrieval injection budget",
        "schema_version": "layered-retrieval-diagnostic-v1",
        "evidence_grade": "deterministic_component_diagnostic",
        "answer_model_calls": 0,
        "embedding_provider_calls": 0,
        "calibration_reference": {
            "artifact": (
                "artifacts/benchmarks/results/vps-phaseb-baseline-retrieval-cache-formal-hy3.json"
            ),
            "episodic_mean_value_chars": 263.3,
            "preference_mean_value_chars": 87.5,
            "episodic_share_of_injected": 0.768,
            "facets_selected_per_question": 12.0,
            "facets_rendered_per_question": 10.0,
        },
        "constraints": {
            "max_facets": args.max_facets,
            "max_rendered_tokens": args.max_rendered_tokens,
            "episodic_per_question": args.episodic_per_question,
            "preference_slots": len(PREFERENCE_UTTERANCES),
        },
        "baseline": baseline,
        "layered": layered,
        "delta": {
            "render_drop_rate": layered["render_drop_rate"] - baseline["render_drop_rate"],
            "state_layer_share_rendered": (
                layered["state_layer_share_rendered"] - baseline["state_layer_share_rendered"]
            ),
            "mean_state_facets_rendered_per_question": (
                layered["mean_state_facets_rendered_per_question"]
                - baseline["mean_state_facets_rendered_per_question"]
            ),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
