"""Read-only scene-level RRF shadow for PersonaMem retrieval diagnostics.

This deliberately does not call Meno or an LLM. It groups the raw conversation
into overlapping scenes, ranks scenes with deterministic BM25 and entity
channels plus an optional existing BGE embedding channel, fuses ranks with RRF,
and compares the resulting context with an existing Meno retrieval cache using
the benchmark's fixed option scorer.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from benchmarks.run_personamem_e2e import _rank_options
from meno.config import Settings
from meno.vector import OpenAICompatEmbedder

WORD_RE = re.compile(r"[a-z0-9]+")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
ENTITY_RE = re.compile(r"\b(?:[A-Z][A-Za-z'-]{2,}|\d{2,})\b")
LEXICAL_STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been",
    "being", "could", "did", "does", "for", "from", "had", "has", "have",
    "how", "into", "its", "not", "that", "the", "their", "then", "there",
    "they", "this", "through", "user", "very", "was", "were", "what",
    "when", "where", "which", "with", "would", "you", "your",
}


@dataclass(frozen=True)
class Scene:
    scene_id: int
    start_index: int
    end_index: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--question-types", default="")
    parser.add_argument("--scene-turns", type=int, default=4)
    parser.add_argument("--scene-overlap", type=int, default=1)
    parser.add_argument("--top-scenes", type=int, default=6)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--recent-bonus", type=float, default=0.002)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-base-url", default="https://api.siliconflow.cn/v1")
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    return parser.parse_args()


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in WORD_RE.findall(text.casefold())
        if len(token) > 2 and token not in LEXICAL_STOPWORDS
    ]


def _entities(text: str) -> set[str]:
    return {match.casefold() for match in ENTITY_RE.findall(text)}


def _parse_options(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or len(parsed) != 4:
        raise ValueError("PersonaMem all_options must contain four options")
    return [str(option) for option in parsed]


def build_scenes(
    messages: list[dict[str, str]],
    *,
    end_index: int,
    scene_turns: int,
    overlap: int,
) -> tuple[str, list[Scene]]:
    if scene_turns < 1 or overlap < 0 or overlap >= scene_turns:
        raise ValueError("scene_turns must be positive and overlap smaller than scene_turns")
    bounded = messages[:end_index]
    profile = "\n".join(
        message.get("content", "") for message in bounded if message.get("role") == "system"
    )
    exchanges: list[tuple[int, int, list[str]]] = []
    current: tuple[int, int, list[str]] | None = None
    for index, message in enumerate(bounded):
        role = message.get("role", "")
        if role == "system":
            continue
        text = message.get("content", "").strip()
        if not text:
            continue
        if role == "user" or current is None:
            if current is not None:
                exchanges.append(current)
            current = (index, index, [f"{role}: {text}"])
        else:
            start, _end, parts = current
            current = (start, index, [*parts, f"{role}: {text}"])
    if current is not None:
        exchanges.append(current)

    step = scene_turns - overlap
    scenes: list[Scene] = []
    for offset in range(0, len(exchanges), step):
        chunk = exchanges[offset : offset + scene_turns]
        if not chunk:
            continue
        scenes.append(
            Scene(
                scene_id=len(scenes),
                start_index=chunk[0][0],
                end_index=chunk[-1][1],
                text="\n".join(part for _start, _end, parts in chunk for part in parts),
            )
        )
        if offset + scene_turns >= len(exchanges):
            break
    return profile, scenes


def _bm25_scores(query: str, scenes: list[Scene]) -> dict[int, float]:
    query_tokens = _tokens(query)
    if not query_tokens or not scenes:
        return {}
    documents = [Counter(_tokens(scene.text)) for scene in scenes]
    lengths = [sum(document.values()) for document in documents]
    average_length = statistics.fmean(lengths) or 1.0
    document_frequency = Counter(
        token for document in documents for token in set(document) if token in query_tokens
    )
    scores: dict[int, float] = {}
    for scene, document, length in zip(scenes, documents, lengths):
        score = 0.0
        for token in query_tokens:
            frequency = document[token]
            if not frequency:
                continue
            inverse = math.log(
                1 + (len(scenes) - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5)
            )
            denominator = frequency + 1.2 * (1 - 0.75 + 0.75 * length / average_length)
            score += inverse * frequency * 2.2 / denominator
        if score > 0:
            scores[scene.scene_id] = score
    return scores


def _entity_scores(query: str, scenes: list[Scene]) -> dict[int, float]:
    query_entities = _entities(query)
    if not query_entities:
        return {}
    return {
        scene.scene_id: float(len(query_entities & _entities(scene.text)))
        for scene in scenes
        if query_entities & _entities(scene.text)
    }


def _ranked_ids(scores: dict[int, float]) -> list[int]:
    return [scene_id for scene_id, _score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def _vector_scores(
    scenes: list[Scene],
    *,
    scene_vectors: dict[int, list[float]] | None,
    query_vector: list[float] | None,
) -> dict[int, float]:
    if scene_vectors is None or query_vector is None:
        return {}
    scores: dict[int, float] = {}
    for scene in scenes:
        vector = scene_vectors.get(scene.scene_id)
        if vector is None or len(vector) != len(query_vector):
            continue
        scores[scene.scene_id] = sum(
            left * right for left, right in zip(query_vector, vector)
        )
    return scores


def rrf_rank(
    query: str,
    scenes: list[Scene],
    *,
    rrf_k: float,
    recent_bonus: float,
    scene_vectors: dict[int, list[float]] | None = None,
    query_vector: list[float] | None = None,
) -> list[tuple[Scene, float]]:
    if rrf_k <= 0 or recent_bonus < 0:
        raise ValueError("rrf_k must be positive and recent_bonus non-negative")
    fused: dict[int, float] = defaultdict(float)
    for ranking in (
        _ranked_ids(_bm25_scores(query, scenes)),
        _ranked_ids(_entity_scores(query, scenes)),
        _ranked_ids(
            _vector_scores(
                scenes,
                scene_vectors=scene_vectors,
                query_vector=query_vector,
            )
        ),
    ):
        for rank, scene_id in enumerate(ranking, 1):
            fused[scene_id] += 1.0 / (rrf_k + rank)
    if not fused:
        return []
    newest = max(scene.end_index for scene in scenes)
    by_id = {scene.scene_id: scene for scene in scenes}
    for scene_id in fused:
        fused[scene_id] += recent_bonus * (by_id[scene_id].end_index / max(1, newest))
    return sorted(
        ((by_id[scene_id], score) for scene_id, score in fused.items()),
        key=lambda item: (-item[1], item[0].scene_id),
    )


def retrieve_context(
    question: str,
    profile: str,
    scenes: list[Scene],
    *,
    question_type: str,
    top_scenes: int,
    rrf_k: float,
    recent_bonus: float,
    max_chars: int,
    scene_vectors: dict[int, list[float]] | None = None,
    query_vector: list[float] | None = None,
) -> tuple[str, list[Scene]]:
    ranked = rrf_rank(
        question,
        scenes,
        rrf_k=rrf_k,
        recent_bonus=recent_bonus,
        scene_vectors=scene_vectors,
        query_vector=query_vector,
    )
    selected = [scene for scene, _score in ranked[:top_scenes]]
    if question_type == "track_full_preference_evolution":
        selected.sort(key=lambda scene: (scene.start_index, scene.scene_id))
    sections = [f"[profile]\n{profile.strip()}"] if profile.strip() else []
    sections.extend(
        f"[scene {scene.scene_id} messages {scene.start_index}-{scene.end_index}]\n{scene.text}"
        for scene in selected
    )
    rendered = "\n\n".join(sections)
    return rendered[:max_chars], selected


def _option_metrics(options: list[str], correct: int, context: str) -> dict[str, Any]:
    scores = _rank_options(options, [{"value": context}])
    ranking = sorted(range(4), key=lambda index: (-scores[index], index))
    predicted = ranking[0] if scores[ranking[0]] > 0 else None
    best_distractor = max(score for index, score in enumerate(scores) if index != correct)
    return {
        "scores": scores,
        "predicted_option": predicted,
        "ranking_correct": predicted == correct,
        "correct_supported": scores[correct] > 0,
        "reciprocal_rank": 1 / (ranking.index(correct) + 1),
        "contamination_margin": scores[correct] - best_distractor,
    }


def _year_recall(correct_option: str, context: str) -> float | None:
    expected = set(YEAR_RE.findall(correct_option))
    if not expected:
        return None
    return len(expected & set(YEAR_RE.findall(context))) / len(expected)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    year_values = [row["temporal_year_recall"] for row in rows if row["temporal_year_recall"] is not None]
    return {
        "items": len(rows),
        "ranking_accuracy": statistics.fmean(row["ranking_correct"] for row in rows),
        "baseline_ranking_accuracy_same_items": statistics.fmean(
            row["baseline_ranking_correct"] for row in rows
        ),
        "improved": sum(row["ranking_correct"] and not row["baseline_ranking_correct"] for row in rows),
        "regressed": sum(not row["ranking_correct"] and row["baseline_ranking_correct"] for row in rows),
        "correct_support_at_k": statistics.fmean(row["correct_supported"] for row in rows),
        "mrr": statistics.fmean(row["reciprocal_rank"] for row in rows),
        "mean_contamination_margin": statistics.fmean(row["contamination_margin"] for row in rows),
        "mean_selected_scenes": statistics.fmean(row["selected_scene_count"] for row in rows),
        "mean_context_chars": statistics.fmean(row["context_chars"] for row in rows),
        "temporal_year_recall": statistics.fmean(year_values) if year_values else None,
    }


def _embedding_key(kind: str, text: str) -> str:
    return hashlib.sha256(f"{kind}\0{text}".encode()).hexdigest()


def _load_or_build_embeddings(
    texts: dict[str, str],
    *,
    cache_path: Path,
    model: str,
    base_url: str,
    dimension: int,
    batch_size: int,
) -> dict[str, list[float]]:
    metadata = {
        "version": 1,
        "model": model,
        "base_url": base_url.rstrip("/"),
        "dimension": dimension,
    }
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("metadata") != metadata:
            raise ValueError("embedding cache metadata does not match requested configuration")
        vectors = payload.get("vectors", {})
        if not isinstance(vectors, dict):
            raise ValueError("embedding cache vectors must be an object")
    else:
        vectors = {}

    missing = [(key, text) for key, text in texts.items() if key not in vectors]
    if missing:
        settings = Settings(
            embedding_model=model,
            embedding_dimension=dimension,
            openai_api_key=os.environ.get("MENO_OPENAI_API_KEY", ""),
            openai_base_url=base_url.rstrip("/"),
            openai_batch_size=batch_size,
        )
        embedder = OpenAICompatEmbedder(settings)
        try:
            generated = embedder.embed_documents([text for _key, text in missing])
        finally:
            embedder.close()
        vectors.update(
            {key: vector for (key, _text), vector in zip(missing, generated)}
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            delete=False,
        ) as handle:
            json.dump({"metadata": metadata, "vectors": vectors}, handle)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, cache_path)
    return vectors


def main() -> None:
    args = parse_args()
    if args.top_scenes < 1 or args.max_context_chars < 1:
        raise SystemExit("top-scenes and max-context-chars must be positive")
    with args.questions.open(newline="", encoding="utf-8") as handle:
        questions = list(csv.DictReader(handle))
    allowed = {item.strip() for item in args.question_types.split(",") if item.strip()}
    if allowed:
        questions = [row for row in questions if row["question_type"] in allowed]
    contexts: dict[str, list[dict[str, str]]] = {}
    for line in args.contexts.read_text(encoding="utf-8").splitlines():
        contexts.update(json.loads(line))
    baseline_payload = json.loads(args.baseline_cache.read_text(encoding="utf-8"))
    baseline = {row["question_id"]: row for row in baseline_payload["records"]}

    prepared: list[tuple[dict[str, str], str, list[Scene]]] = []
    embedding_texts: dict[str, str] = {}
    for row in questions:
        end_index = int(row["end_index_in_shared_context"])
        profile, scenes = build_scenes(
            contexts[row["shared_context_id"]],
            end_index=end_index,
            scene_turns=args.scene_turns,
            overlap=args.scene_overlap,
        )
        prepared.append((row, profile, scenes))
        if args.embedding_cache:
            question = row["user_question_or_message"]
            embedding_texts[_embedding_key("query", question)] = question
            for scene in scenes:
                embedding_texts[_embedding_key("scene", scene.text)] = scene.text

    vectors: dict[str, list[float]] = {}
    if args.embedding_cache:
        vectors = _load_or_build_embeddings(
            embedding_texts,
            cache_path=args.embedding_cache,
            model=args.embedding_model,
            base_url=args.embedding_base_url,
            dimension=args.embedding_dimension,
            batch_size=args.embedding_batch_size,
        )

    results: list[dict[str, Any]] = []
    for row, profile, scenes in prepared:
        question_id = row["question_id"]
        question = row["user_question_or_message"]
        scene_vectors = (
            {
                scene.scene_id: vectors[_embedding_key("scene", scene.text)]
                for scene in scenes
            }
            if args.embedding_cache
            else None
        )
        query_vector = (
            vectors[_embedding_key("query", question)] if args.embedding_cache else None
        )
        rendered, selected = retrieve_context(
            question,
            profile,
            scenes,
            question_type=row["question_type"],
            top_scenes=args.top_scenes,
            rrf_k=args.rrf_k,
            recent_bonus=args.recent_bonus,
            max_chars=args.max_context_chars,
            scene_vectors=scene_vectors,
            query_vector=query_vector,
        )
        options = _parse_options(row["all_options"])
        correct = ord(row["correct_answer"].strip("()").casefold()) - ord("a")
        metrics = _option_metrics(options, correct, rendered)
        baseline_record = baseline[question_id]
        selected_span = (
            (max(scene.end_index for scene in selected) - min(scene.start_index for scene in selected))
            / max(1, end_index)
            if selected
            else 0.0
        )
        results.append(
            {
                "question_id": question_id,
                "question_type": row["question_type"],
                **metrics,
                "baseline_ranking_correct": (
                    baseline_record["ranking_predicted_option"] == baseline_record["correct_option"]
                ),
                "selected_scene_count": len(selected),
                "selected_scene_ids": [scene.scene_id for scene in selected],
                "selected_scene_span_ratio": selected_span,
                "chronological_output": all(
                    left.start_index <= right.start_index for left, right in pairwise(selected)
                ),
                "temporal_year_recall": _year_recall(options[correct], rendered),
                "context_chars": len(rendered),
            }
        )

    by_type = {
        question_type: _aggregate(
            [row for row in results if row["question_type"] == question_type]
        )
        for question_type in sorted({row["question_type"] for row in results})
    }
    report = {
        "benchmark": "PersonaMem deterministic scene-RRF retrieval shadow",
        "comparability": "Same questions and fixed option scorer as the Meno retrieval cache; no LLM answers.",
        "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "contexts_sha256": hashlib.sha256(args.contexts.read_bytes()).hexdigest(),
        "baseline_cache_sha256": hashlib.sha256(args.baseline_cache.read_bytes()).hexdigest(),
        "config": {
            "channels": ["bm25", "entity", *(["vector"] if args.embedding_cache else [])],
            "scene_turns": args.scene_turns,
            "scene_overlap": args.scene_overlap,
            "top_scenes": args.top_scenes,
            "rrf_k": args.rrf_k,
            "recent_bonus": args.recent_bonus,
            "max_context_chars": args.max_context_chars,
            "question_types": sorted(allowed),
            "embedding": (
                {
                    "model": args.embedding_model,
                    "base_url": args.embedding_base_url.rstrip("/"),
                    "dimension": args.embedding_dimension,
                    "cached_vectors": len(vectors),
                }
                if args.embedding_cache
                else None
            ),
        },
        "aggregate": _aggregate(results),
        "by_question_type": by_type,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "aggregate": report["aggregate"], "by_question_type": by_type}, indent=2))


if __name__ == "__main__":
    main()
