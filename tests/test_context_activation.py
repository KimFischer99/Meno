from __future__ import annotations

from meno.context_activation import ContextActivationPolicy, lexical_overlap


def decide(**overrides):
    values = {
        "query": "How should you format your answer?",
        "task_type": "response_style",
        "kind": "preference",
        "value": "concise answers",
        "routing_slot": "answer_style",
        "sensitive": False,
        "semantic_score": 0.01,
    }
    values.update(overrides)
    return ContextActivationPolicy().decide(**values)


def test_preference_slot_match_overrides_low_vector_score() -> None:
    assert decide().allowed is True


def test_preference_slot_mismatch_blocks_irrelevant_claim() -> None:
    result = decide(
        query="What food should you order for me?",
        task_type="food_recommendation",
    )

    assert result.allowed is False
    assert result.reasons == ("context slot mismatch",)


def test_unslotted_episode_requires_semantic_or_lexical_evidence() -> None:
    assert decide(
        kind="episodic",
        value="I am building the Meno project",
        routing_slot=None,
        semantic_score=0.65,
    ).allowed
    assert not decide(
        query="What food should you order?",
        task_type="food_recommendation",
        kind="episodic",
        value="I am building the Meno project",
        routing_slot=None,
        semantic_score=0.15,
    ).allowed


def test_sensitive_activation_uses_lexical_evidence_only() -> None:
    result = decide(
        query="What is my medical diagnosis?",
        task_type="personalization",
        kind="episodic",
        value="My medical diagnosis is cancer",
        routing_slot=None,
        sensitive=True,
        semantic_score=None,
    )

    assert result.allowed is True
    assert lexical_overlap("medical diagnosis", "My medical diagnosis is cancer") >= 0.34
