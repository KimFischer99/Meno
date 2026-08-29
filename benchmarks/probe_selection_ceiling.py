"""Oracle selection ceiling for the injected-context budget.

If the read path could pick the best 12 claims from a user's whole active state,
how far would the current option-ranking metric go? This separates two very
different diagnoses:

* a high ceiling means the evidence is present and the ranker is the bottleneck
  (a selection problem, addressable by better retrieval scoring);
* a low ceiling means the stored representation cannot express what the questions
  turn on (a representation problem, needing a different claim model).

Greedy oracle: repeatedly add whichever active claim most increases the correct
option's scoring margin. Upper bound only -- it reads the answer key, so it can
never be a product metric.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from meno.api import build_service
from meno.config import Settings
from meno.db import Claim

BUDGET = 12
STOPWORDS = frozenset(
    (
        "user", "assistant", "would", "could", "should", "there", "their",
        "these", "those", "about", "which", "where",
    )
)


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z]{4,}", text.lower()) if token not in STOPWORDS}


def _margin(selected: list[str], option_tokens: list[set[str]], frequency: Counter, correct: int) -> float:
    injected = _tokens(" ".join(selected))
    scores = [
        sum(math.log((len(option_tokens) + 1) / (frequency[t] + 0.5)) for t in tokens & injected)
        / max(1, len(tokens))
        for tokens in option_tokens
    ]
    rival = max(score for index, score in enumerate(scores) if index != correct)
    return scores[correct] - rival


def _greedy(pool: list[str], option_tokens: list[set[str]], frequency: Counter, correct: int) -> list[str]:
    selected: list[str] = []
    remaining = list(dict.fromkeys(pool))
    for _ in range(BUDGET):
        best: str | None = None
        best_value = float("-inf")
        for candidate in remaining:
            value = _margin([*selected, candidate], option_tokens, frequency, correct)
            if value > best_value:
                best_value = value
                best = candidate
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
    return selected


def main() -> None:
    cache = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    questions = Path(sys.argv[2])
    rows = {row["question_id"]: row for row in csv.DictReader(questions.open(encoding="utf-8"))}
    service = build_service(Settings.from_env())
    results: dict[str, list[int]] = {}
    actual: dict[str, list[int]] = {}
    try:
        with service.session_factory() as session:
            for record in cache["records"]:
                if record["question_id"] not in rows:
                    continue
                claims = session.scalars(
                    select(Claim).where(
                        Claim.user_id == record["user_id"], Claim.status == "active"
                    )
                ).all()
                pool = [claim.value for claim in claims]
                if not pool:
                    continue
                options = [re.sub(r"^\([a-d]\)\s*", "", o) for o in record["options"]]
                option_tokens = [_tokens(o) for o in options]
                frequency = Counter(t for tokens in option_tokens for t in tokens)
                correct = record["correct_option"]
                selected = _greedy(pool, option_tokens, frequency, correct)
                bucket = results.setdefault(record["question_type"], [0, 0])
                bucket[1] += 1
                if _margin(selected, option_tokens, frequency, correct) > 0:
                    bucket[0] += 1
                live = actual.setdefault(record["question_type"], [0, 0])
                live[1] += 1
                if record["ranking_predicted_option"] == correct:
                    live[0] += 1
    finally:
        service.close()

    print(f"{'question_type':46} {'n':>4} {'actual':>8} {'oracle_selection':>17}")
    print("-" * 78)
    totals = [0, 0, 0]
    for question_type in sorted(results):
        hits, count = results[question_type]
        live_hits = actual[question_type][0]
        totals[0] += count
        totals[1] += live_hits
        totals[2] += hits
        print(
            f"{question_type:46} {count:>4} {live_hits / count:>8.4f} {hits / count:>17.4f}"
        )
    print(
        f"{'OVERALL':46} {totals[0]:>4} {totals[1] / totals[0]:>8.4f} "
        f"{totals[2] / totals[0]:>17.4f}"
    )


if __name__ == "__main__":
    main()
