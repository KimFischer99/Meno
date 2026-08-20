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
    slot: str | None = None


# Preference statements are matched per sentence so one message can yield
# several candidates. Each entry is (pattern, keep_full_match): affirmations
# extract only the preference object; negations keep the full statement so the
# value still reads as a negation. Under the slot key a negation collides with
# the earlier affirmed value and coordination supersedes it (contradiction).
PREFERENCE_PATTERNS: list[tuple[re.Pattern[str], bool]] = [
    (
        re.compile(
            r"(?:i\s+(?:really\s+)?(?:prefer|like|love|want|enjoy)"
            r"|please\s+always|i\s+usually\s+go\s+with)\s+([^.。!!?？\n]+)",
            re.IGNORECASE,
        ),
        False,
    ),
    (
        re.compile(
            r"(i\s+(?:no\s+longer|don'?t|do\s+not)\s+(?:like|prefer|want)\s+[^.。!!?？\n]+"
            r"|i\s+stopped\s+liking\s+[^.。!!?？\n]+)",
            re.IGNORECASE,
        ),
        True,
    ),
    (
        re.compile(r"(?:我(?:更|最)?(?:喜欢|偏好|爱)|我希望|以后(?:请)?默认|请记住)\s*([^.。!!?？\n]+)"),
        False,
    ),
    (
        re.compile(r"((?:我不再(?:喜欢|偏好|爱)|别再用)\s*[^.。!!?？\n]+)"),
        True,
    ),
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

# Slot lexicon: preference values mentioning a keyword collapse onto one
# semantic slot, so evolution of the same preference shares a semantic_key
# (merge/compress) and a changed value supersedes the old one (contradiction).
SLOT_LEXICON: dict[str, tuple[str, ...]] = {
    "beverage": ("tea", "coffee", "espresso", "latte", "matcha", "茶", "咖啡", "奶茶"),
    "answer_style": (
        "answer", "answers", "response", "responses", "reply", "replies",
        "terse", "detailed", "concise", "verbose", "summary", "format",
        "structure", "回答", "答案", "回复", "结论", "细节", "格式", "结构",
    ),
    "programming_language": (
        "python", "rust", "golang", "javascript", "typescript", "java",
        "kotlin", "swift", "c++", "c#", "ruby", "编程", "语言",
    ),
    "music": ("music", "jazz", "rock", "pop", "classical", "hip-hop", "音乐", "爵士", "摇滚"),
    "food": ("food", "sushi", "pizza", "pasta", "spicy", "sweet", "sour", "菜", "食物", "辣", "甜"),
    "color": ("color", "colour", "red", "blue", "green", "颜色", "红", "蓝", "绿"),
    "sport": (
        "sport", "running", "swimming", "yoga", "football", "basketball",
        "运动", "跑步", "游泳", "瑜伽",
    ),
}

EPISODIC_MAX_CHARS = 280
MIN_EPISODIC_TOKENS = 2

_TOKEN_PATTERN = re.compile(r"[\w-]+|[一-鿿]")
_SENTENCE_SPLIT = re.compile(r"[。!!?？\n]+|(?<=[.!?])\s+")


def preference_slot(text: str) -> str | None:
    """Map free-form preference text to a canonical slot, if any keyword hits."""
    lowered = text.casefold()
    for slot, keywords in SLOT_LEXICON.items():
        for keyword in keywords:
            if keyword.isascii():
                if re.search(rf"\b{re.escape(keyword)}\b", lowered):
                    return slot
            elif keyword in lowered:
                return slot
    return None


def extract_claims(event: Event) -> list[ClaimCandidate]:
    if event.role not in {"user", "assistant"}:
        return []
    if any(pattern.search(event.content) for pattern in INJECTION_PATTERNS):
        return []

    sensitive = any(pattern.search(event.content) for pattern in SENSITIVE_PATTERNS)
    if event.role == "assistant":
        return _episodic_fallback(event, sensitive, assistant=True)

    candidates: list[ClaimCandidate] = []
    for sentence in _sentences(event.content):
        for pattern, keep_full_match in PREFERENCE_PATTERNS:
            match = pattern.search(sentence)
            if not match:
                continue
            value = match.group(1).strip("。.!！?？ ")
            if not value:
                continue
            candidates.append(
                ClaimCandidate(
                    kind="preference",
                    semantic_channel="preference.explicit",
                    value=value,
                    confidence=0.9,
                    half_life_days=180,
                    sensitive=sensitive,
                    slot=preference_slot(value),
                )
            )
            break
    if candidates:
        return candidates
    return _episodic_fallback(event, sensitive, assistant=False)


def _sentences(content: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split(content) if part.strip()]


def _episodic_fallback(
    event: Event, sensitive: bool, *, assistant: bool
) -> list[ClaimCandidate]:
    """Graceful degradation: only persist statements with minimal substance,
    and truncate instead of storing the whole raw message."""
    text = event.content.strip()
    if len(_TOKEN_PATTERN.findall(text)) < MIN_EPISODIC_TOKENS:
        return []
    if assistant:
        value = f"Assistant previously responded: {text}"
        channel = "episode.assistant_response"
        confidence = 0.6
        half_life_days = 30
    else:
        value = text
        channel = "episode.user_statement"
        confidence = 0.68
        half_life_days = 90
    if len(value) > EPISODIC_MAX_CHARS:
        value = value[:EPISODIC_MAX_CHARS].rstrip() + "…"
    return [
        ClaimCandidate(
            kind="episodic",
            semantic_channel=channel,
            value=value,
            confidence=confidence,
            half_life_days=half_life_days,
            sensitive=sensitive,
        )
    ]
