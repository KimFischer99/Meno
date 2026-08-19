from __future__ import annotations

import re
from dataclasses import dataclass

from .db import Event


@dataclass(frozen=True)
class ClaimCandidate:
    kind: str
    semantic_channel: str
    value: str
    confidence: float
    half_life_days: float | None
    sensitive: bool = False


PREFERENCE_PATTERNS = [
    re.compile(r"(?:i\s+(?:prefer|like|want)|please\s+always)\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:我(?:更)?喜欢|我希望|以后(?:请)?默认|请记住)\s*(.+)"),
]

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(?:the\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"忽略(?:所有)?(?:之前|先前)指令"),
    re.compile(r"泄露.*系统提示"),
]

SENSITIVE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(?:hiv|cancer|diagnosis|religion|ethnicity|biometric|political party)\b",
        r"(?:癌症|诊断|宗教|民族|生物识别|政治倾向)",
    ]
]


def extract_claims(event: Event) -> list[ClaimCandidate]:
    if event.role not in {"user", "assistant"}:
        return []
    if any(pattern.search(event.content) for pattern in INJECTION_PATTERNS):
        return []

    sensitive = any(pattern.search(event.content) for pattern in SENSITIVE_PATTERNS)
    if event.role == "assistant":
        return [
            ClaimCandidate(
                kind="episodic",
                semantic_channel="episode.assistant_response",
                value=f"Assistant previously responded: {event.content}",
                confidence=0.6,
                half_life_days=30,
                sensitive=sensitive,
            )
        ]
    for pattern in PREFERENCE_PATTERNS:
        match = pattern.search(event.content)
        if match:
            value = match.group(1).strip("。.!！ ")
            if value:
                return [
                    ClaimCandidate(
                        kind="preference",
                        semantic_channel="preference.explicit",
                        value=value,
                        confidence=0.9,
                        half_life_days=180,
                        sensitive=sensitive,
                    )
                ]
    return [
        ClaimCandidate(
            kind="episodic",
            semantic_channel="episode.user_statement",
            value=event.content,
            confidence=0.68,
            half_life_days=90,
            sensitive=sensitive,
        )
    ]
