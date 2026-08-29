from __future__ import annotations

from benchmarks import run_scene_rrf_shadow as shadow


def messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "User prefers concise answers."},
        {"role": "user", "content": "In 2019 I started painting landscapes."},
        {"role": "assistant", "content": "Painting sounds rewarding."},
        {"role": "user", "content": "I practiced watercolor every weekend."},
        {"role": "assistant", "content": "Watercolor can capture light."},
        {"role": "user", "content": "In 2023 I switched to ceramic sculpture."},
        {"role": "assistant", "content": "Ceramic sculpture is tactile."},
        {"role": "user", "content": "I joined a local pottery studio."},
        {"role": "assistant", "content": "A studio provides useful equipment."},
    ]


def test_build_scenes_preserves_profile_and_overlaps() -> None:
    profile, scenes = shadow.build_scenes(messages(), end_index=9, scene_turns=2, overlap=1)

    assert profile == "User prefers concise answers."
    assert len(scenes) == 3
    assert scenes[0].start_index == 1
    assert scenes[1].start_index == 3
    assert "watercolor" in scenes[0].text.casefold()
    assert "watercolor" in scenes[1].text.casefold()


def test_rrf_retrieval_selects_matching_scene_without_options() -> None:
    profile, scenes = shadow.build_scenes(messages(), end_index=9, scene_turns=2, overlap=0)

    rendered, selected = shadow.retrieve_context(
        "What kind of pottery equipment might help me?",
        profile,
        scenes,
        question_type="suggest_new_ideas",
        top_scenes=1,
        rrf_k=60,
        recent_bonus=0,
        max_chars=2000,
    )

    assert len(selected) == 1
    assert "pottery studio" in rendered.casefold()
    assert "landscapes" not in rendered.casefold()


def test_evolution_output_is_chronological_after_relevance_selection() -> None:
    profile, scenes = shadow.build_scenes(messages(), end_index=9, scene_turns=1, overlap=0)

    rendered, selected = shadow.retrieve_context(
        "How did my creative interests change from painting to sculpture?",
        profile,
        scenes,
        question_type="track_full_preference_evolution",
        top_scenes=3,
        rrf_k=60,
        recent_bonus=0,
        max_chars=3000,
    )

    assert [scene.start_index for scene in selected] == sorted(
        scene.start_index for scene in selected
    )
    assert rendered.index("painting") < rendered.index("sculpture")


def test_option_metrics_exposes_distractor_contamination() -> None:
    options = [
        "(a) Continue watercolor painting",
        "(b) Explore ceramic sculpture",
        "(c) Train for a marathon",
        "(d) Learn accounting",
    ]

    result = shadow._option_metrics(options, 1, "ceramic sculpture pottery studio")

    assert result["ranking_correct"] is True
    assert result["contamination_margin"] > 0


def test_vector_channel_can_retrieve_without_lexical_overlap() -> None:
    profile, scenes = shadow.build_scenes(messages(), end_index=9, scene_turns=2, overlap=0)
    vectors = {
        scene.scene_id: ([1.0, 0.0] if "pottery studio" in scene.text else [0.0, 1.0])
        for scene in scenes
    }

    rendered, selected = shadow.retrieve_context(
        "What new activity fits me?",
        profile,
        scenes,
        question_type="suggest_new_ideas",
        top_scenes=1,
        rrf_k=60,
        recent_bonus=0,
        max_chars=2000,
        scene_vectors=vectors,
        query_vector=[1.0, 0.0],
    )

    assert len(selected) == 1
    assert "pottery studio" in rendered.casefold()


def test_embedding_cache_metadata_mismatch_fails_closed(tmp_path) -> None:
    cache = tmp_path / "vectors.json"
    cache.write_text(
        '{"metadata":{"version":1,"model":"wrong"},"vectors":{}}',
        encoding="utf-8",
    )

    try:
        shadow._load_or_build_embeddings(
            {},
            cache_path=cache,
            model="BAAI/bge-m3",
            base_url="https://api.siliconflow.cn/v1",
            dimension=1024,
            batch_size=32,
        )
    except ValueError as exc:
        assert "metadata" in str(exc)
    else:
        raise AssertionError("cache mismatch must fail closed")
