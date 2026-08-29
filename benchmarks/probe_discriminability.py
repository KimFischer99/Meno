"""Discriminability probe: is the deciding evidence findable from the query alone?

``probe_selection_ceiling.py`` showed that an oracle picking the best 12 claims
from the same stored state reaches 0.88 where the live path reaches 0.42. That
oracle reads the answer key, so it bounds what *some* selector could achieve, not
what a deployable one can: a selector sees only the query and the user's history.

This probe compares two selections over the same candidate pool:

* ``oracle``  -- maximizes the correct option's scoring margin (reads the answer)
* ``blind``   -- maximizes query coverage while penalizing redundancy, which is
                 exactly the deployable objective in ``MenoService._select_evidence``

If the two pick largely the same claims, the query carries enough signal and
better selection is worth pursuing. If they diverge, the ceiling is unreachable in
principle and the bottleneck is neither claim representation nor ranking, but the
absence of a query-side signal pointing at the deciding evidence.

Deterministic, read-only, no answer model and no embedding provider.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from meno.api import build_service
from meno.config import Settings
from meno.db import Claim
from meno.service import _content_tokens

BUDGET = 12
REDUNDANCY_PENALTY = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-cache", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _option_tokens(record: dict[str, Any]) -> list[set[str]]:
    return [
        _content_tokens(re.sub(r"^\([a-d]\)\s*", "", option)) for option in record["options"]
    ]


def _margin(selected: list[str], options: list[set[str]], frequency: Counter, correct: int) -> float:
    injected = _content_tokens(" ".join(selected))
    scores = [
        sum(math.log((len(options) + 1) / (frequency[t] + 0.5)) for t in tokens & injected)
        / max(1, len(tokens))
        for tokens in options
    ]
    rival = max(score for index, score in enumerate(scores) if index != correct)
    return scores[correct] - rival


def _pick_oracle(
    pool: list[str], options: list[set[str]], frequency: Counter, correct: int
) -> list[str]:
    """Greedy on the answer key: upper bound, not deployable."""
    selected: list[str] = []
    remaining = list(dict.fromkeys(pool))
    for _ in range(BUDGET):
        best, best_value = None, float("-inf")
        for candidate in remaining:
            value = _margin([*selected, candidate], options, frequency, correct)
            if value > best_value:
                best_value, best = value, candidate
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
    return selected


def _pick_blind(pool: list[str], query: str) -> list[str]:
    """Greedy on query coverage minus redundancy: the deployable objective."""
    query_terms = _content_tokens(query)
    tokens = {value: _content_tokens(value) for value in dict.fromkeys(pool)}
    selected: list[str] = []
    covered: set[str] = set()
    chosen_tokens: set[str] = set()
    remaining = list(dict.fromkeys(pool))
    for _ in range(BUDGET):
        best, best_gain = None, float("-inf")
        for candidate in remaining:
            candidate_tokens = tokens[candidate]
            if not candidate_tokens:
                gain = 0.0
            else:
                new_terms = len((candidate_tokens & query_terms) - covered)
                coverage = new_terms / len(query_terms) if query_terms else 0.0
                overlap = len(candidate_tokens & chosen_tokens) / len(candidate_tokens)
                gain = coverage - REDUNDANCY_PENALTY * overlap
            if gain > best_gain:
                best_gain, best = gain, candidate
        if best is None:
            break
        selected.append(best)
        covered |= tokens[best] & query_terms
        chosen_tokens |= tokens[best]
        remaining.remove(best)
    return selected


def run(cache_path: Path, limit: int) -> dict[str, Any]:
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    records = cache["records"][: limit or None]
    service = build_service(Settings.from_env())
    per_type: dict[str, list[dict[str, float]]] = {}
    try:
        with service.session_factory() as session:
            for record in records:
                claims = session.scalars(
                    select(Claim).where(
                        Claim.user_id == record["user_id"], Claim.status == "active"
                    )
                ).all()
                pool = [claim.value for claim in claims]
                if not pool:
                    continue
                options = _option_tokens(record)
                correct = record["correct_option"]
                decisive = options[correct] - set().union(
                    *[t for i, t in enumerate(options) if i != correct]
                )
                if not decisive:
                    continue
                frequency = Counter(t for tokens in options for t in tokens)
                oracle = _pick_oracle(pool, options, frequency, correct)
                blind = _pick_blind(pool, record["question"])
                oracle_set, blind_set = set(oracle), set(blind)
                union = oracle_set | blind_set
                per_type.setdefault(record["question_type"], []).append(
                    {
                        "jaccard": len(oracle_set & blind_set) / len(union) if union else 0.0,
                        "oracle_decisive_recall": len(
                            decisive & _content_tokens(" ".join(oracle))
                        )
                        / len(decisive),
                        "blind_decisive_recall": len(
                            decisive & _content_tokens(" ".join(blind))
                        )
                        / len(decisive),
                        "query_signal": len(decisive & _content_tokens(record["question"]))
                        / len(decisive),
                    }
                )
    finally:
        service.close()

    def mean(rows: list[dict[str, float]], key: str) -> float:
        return statistics.mean(row[key] for row in rows)

    by_type = {
        question_type: {
            "questions": len(rows),
            "oracle_blind_jaccard": mean(rows, "jaccard"),
            "oracle_decisive_recall": mean(rows, "oracle_decisive_recall"),
            "blind_decisive_recall": mean(rows, "blind_decisive_recall"),
            "query_signal": mean(rows, "query_signal"),
        }
        for question_type, rows in sorted(per_type.items())
    }
    everything = [row for rows in per_type.values() for row in rows]
    return {
        "probe": "Meno evidence discriminability",
        "schema_version": "discriminability-probe-v1",
        "answer_model_calls": 0,
        "budget": BUDGET,
        "interpretation": (
            "High oracle_blind_jaccard means the query identifies the same evidence the "
            "answer key does, so better selection has headroom. Low jaccard with high "
            "oracle_decisive_recall means the ceiling is unreachable from the query alone."
        ),
        "by_question_type": by_type,
        "overall": {
            "questions": len(everything),
            "oracle_blind_jaccard": mean(everything, "jaccard"),
            "oracle_decisive_recall": mean(everything, "oracle_decisive_recall"),
            "blind_decisive_recall": mean(everything, "blind_decisive_recall"),
            "query_signal": mean(everything, "query_signal"),
        },
    }


def main() -> None:
    args = parse_args()
    report = run(args.retrieval_cache, args.limit)
    header = (
        f"{'question_type':46} {'n':>4} {'jaccard':>8} {'oracle_rec':>11} "
        f"{'blind_rec':>10} {'query_sig':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, row in [*report["by_question_type"].items(), ("OVERALL", report["overall"])]:
        print(
            f"{name:46} {row['questions']:>4} {row['oracle_blind_jaccard']:>8.4f} "
            f"{row['oracle_decisive_recall']:>11.4f} {row['blind_decisive_recall']:>10.4f} "
            f"{row['query_signal']:>10.4f}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
