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
    # Direction of the preference on its dimension. "positive" and "negative"
    # share a semantic_key so a reversal supersedes rather than co-exists, which
    # is what makes preference evolution readable from the claim chain.
    stance: str | None = None
    # Set when a single sentence expresses a transition ("used to X, now Y"):
    # the superseded side is emitted first and marked so the write path can
    # order them deterministically.
    transition_role: str | None = None


STANCE_POSITIVE = "positive"
STANCE_NEGATIVE = "negative"

# Affirmations capture only the preference object; the stance is carried in the
# dedicated field instead of being implied by the surrounding words. Negations
# previously kept the whole sentence, which made the two directions structurally
# incomparable and hid reversals from the supersede chain.
PREFERENCE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"(?:i\s+(?:really\s+)?(?:prefer|like|love|want|enjoy)"
            r"|please\s+always|i\s+usually\s+go\s+with)\s+([^.。!!?？\n]+)",
            re.IGNORECASE,
        ),
        STANCE_POSITIVE,
    ),
    (
        re.compile(
            r"i\s+(?:no\s+longer|don'?t|do\s+not)\s+(?:like|prefer|want|enjoy)\s+"
            r"([^.。!!?？\n]+)",
            re.IGNORECASE,
        ),
        STANCE_NEGATIVE,
    ),
    (
        re.compile(
            r"i\s+(?:stopped|quit)\s+(?:liking|enjoying|using)\s+([^.。!!?？\n]+)",
            re.IGNORECASE,
        ),
        STANCE_NEGATIVE,
    ),
    (
        re.compile(
            r"i\s+(?:dislike|disliked|hate|hated)\s+([^.。!!?？\n]+)",
            re.IGNORECASE,
        ),
        STANCE_NEGATIVE,
    ),
    (
        re.compile(r"(?:我(?:更|最)?(?:喜欢|偏好|爱)|我希望|以后(?:请)?默认|请记住)\s*([^.。!!?？\n]+)"),
        STANCE_POSITIVE,
    ),
    (
        re.compile(r"(?:我不再(?:喜欢|偏好|爱)|我讨厌|别再用)\s*([^.。!!?？\n]+)"),
        STANCE_NEGATIVE,
    ),
]

# Transition phrasings state both sides of an evolution in one sentence. These
# are the dominant construction in preference-evolution evaluation, and without
# them the sentence degrades to an episodic blob that loses the change entirely.
# Each entry maps to (old_group, new_group, old_stance, new_stance).
TRANSITION_PATTERNS: list[tuple[re.Pattern[str], int, int, str, str]] = [
    (
        re.compile(
            r"i\s+used\s+to\s+(?:like|love|enjoy|prefer)\s+([^,.;。!!?？\n]+)"
            r"\s*(?:,|but|though|however)\s*(?:but\s+)?(?:now|these\s+days|lately)\s+"
            r"i\s+(?:find|prefer|like|enjoy|switched\s+to)\s+([^.。!!?？\n]+)",
            re.IGNORECASE,
        ),
        1,
        2,
        STANCE_NEGATIVE,
        STANCE_POSITIVE,
    ),
    (
        re.compile(
            r"i\s+(?:switched|moved|shifted)\s+(?:away\s+)?from\s+([^,.;。!!?？\n]+?)"
            r"\s+to\s+([^.。!!?？\n]+)",
            re.IGNORECASE,
        ),
        1,
        2,
        STANCE_NEGATIVE,
        STANCE_POSITIVE,
    ),
    (
        re.compile(
            r"initially\s+i\s+(?:liked|enjoyed|preferred)\s+([^,.;。!!?？\n]+)"
            r"\s*(?:,|but|though)\s*(?:but\s+)?(?:later|now|eventually|then)\s+"
            r"i\s+(?:prefer|like|enjoy|switched\s+to|moved\s+to)\s+([^.。!!?？\n]+)",
            re.IGNORECASE,
        ),
        1,
        2,
        STANCE_NEGATIVE,
        STANCE_POSITIVE,
    ),
    (
        re.compile(
            r"我(?:以前|原来|之前)(?:喜欢|偏好)\s*([^,.;。!!?？\n]+)"
            r"\s*(?:,|，|但是?|不过)\s*(?:现在|后来)\s*(?:我)?(?:喜欢|偏好|改用)\s*([^.。!!?？\n]+)"
        ),
        1,
        2,
        STANCE_NEGATIVE,
        STANCE_POSITIVE,
    ),
]

# Reversals that name only the abandoned side, with no replacement. "Initially I
# liked X, but later I found it tedious" is a negative on X, not a preference for
# tedium, so the object is taken from the first clause.
REVERSAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"initially\s+i\s+(?:liked|enjoyed|preferred)\s+([^,.;。!!?？\n]+)"
        r"\s*(?:,|but|though)\s*(?:but\s+)?(?:later|now|eventually|then)\s+"
        r"i\s+(?:found|find|felt|considered)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"i\s+used\s+to\s+(?:like|love|enjoy|prefer)\s+([^,.;。!!?？\n]+)"
        r"\s*(?:,|but|though|however)\s*(?:but\s+)?(?:now|these\s+days|lately)\s+"
        r"i\s+(?:find|found|felt)\b",
        re.IGNORECASE,
    ),
]

# Sentences that only announce an abandonment, with no replacement.
ABANDON_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"i\s+(?:moved|stepped|shifted)\s+away\s+from\s+([^.。!!?？\n]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"i\s+(?:gave\s+up|abandoned)\s+(?:on\s+)?([^.。!!?？\n]+)",
        re.IGNORECASE,
    ),
]

# Narrative abandonment: "I tried X, but it was too demanding". Real conversation
# expresses most preference reversals this way rather than with an explicit
# "I no longer like X", so without this shape the negative stance is never
# recorded and preference evolution stays invisible.
TRIED_BUT_PATTERN = re.compile(
    r"\bi\s+(?:tried|started|joined|explored|attempted|took\s+up|signed\s+up\s+for)\s+"
    r"(?P<object>.{3,90}?)\s*,?\s*\bbut\b(?P<tail>.{0,120})",
    re.IGNORECASE | re.DOTALL,
)

# Negative sentiment in the trailing clause. Requiring an explicit marker keeps
# "I tried X but stuck with it" from being read as a rejection.
NEGATIVE_SENTIMENT = re.compile(
    r"\b(?:too\s+\w+|tedious|uninspiring|overwhelming|overwhelmed|unproductive|"
    r"unsatisfying|unfulfilling|frustrat\w*|stress\w*|chore|rigid|lacking|"
    r"time-consuming|did\s*n[o']?t\s+(?:resonate|stick|work|last)|"
    r"was\s*n[o']?t\s+(?:for\s+me|worth)|lost\s+interest|gave\s+up|"
    r"felt\s+(?:so\s+)?(?:lost|stuck|bored|drained))\b",
    re.IGNORECASE,
)

# Leading filler to drop from a narrative object ("to harmonize our efforts").
_OBJECT_LEAD = re.compile(r"^(?:to|the|a|an|my|some)\s+", re.IGNORECASE)
# A bare time reference is not a preference object.
_TIME_ONLY = re.compile(r"^(?:in\s+)?(?:19|20)\d{2}$|^(?:recently|lately|last\s+\w+)$", re.IGNORECASE)

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

# Trailing time adverbs belong to the stance, not to the preference object:
# "tea anymore" and "tea" are the same dimension.
_TRAILING_ADVERBS = re.compile(
    r"\s+(?:anymore|any\s+more|any\s+longer|no\s+more|now|these\s+days|nowadays)$",
    re.IGNORECASE,
)


def _normalize_object(value: str) -> str:
    """Strip punctuation and trailing stance adverbs from a preference object."""
    cleaned = value.strip("。.!！?？, ，")
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = _TRAILING_ADVERBS.sub("", cleaned).strip("。.!！?？, ，")
    return cleaned


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


def _preference(
    value: str, stance: str, sensitive: bool, *, transition_role: str | None = None
) -> ClaimCandidate | None:
    cleaned = _normalize_object(value)
    if not cleaned:
        return None
    return ClaimCandidate(
        kind="preference",
        semantic_channel="preference.explicit",
        value=cleaned,
        confidence=0.9,
        half_life_days=180,
        sensitive=sensitive,
        slot=preference_slot(cleaned),
        stance=stance,
        transition_role=transition_role,
    )


def _transition_candidates(sentence: str, sensitive: bool) -> list[ClaimCandidate]:
    """Both sides of a one-sentence preference change, oldest first.

    Returning the superseded side first lets the write path apply them in order
    so the supersede chain records the reversal instead of two unrelated claims.
    """
    for pattern, old_group, new_group, old_stance, new_stance in TRANSITION_PATTERNS:
        match = pattern.search(sentence)
        if match is None:
            continue
        old = _preference(
            match.group(old_group), old_stance, sensitive, transition_role="superseded"
        )
        new = _preference(
            match.group(new_group), new_stance, sensitive, transition_role="current"
        )
        candidates = [item for item in (old, new) if item is not None]
        if candidates:
            return candidates
    return []


def _narrative_abandonment(sentence: str, sensitive: bool) -> ClaimCandidate | None:
    """Negative stance from "I tried X, but <negative>" narrative phrasing."""
    match = TRIED_BUT_PATTERN.search(sentence)
    if match is None:
        return None
    if not NEGATIVE_SENTIMENT.search(match.group("tail")):
        return None
    obj = _OBJECT_LEAD.sub("", match.group("object").strip(" ,;"))
    obj = re.sub(r"\s+in\s+(?:19|20)\d{2}$", "", obj).strip(" ,;")
    if not obj or _TIME_ONLY.match(obj) or len(_TOKEN_PATTERN.findall(obj)) < 2:
        return None
    return _preference(obj, STANCE_NEGATIVE, sensitive)


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
        # Reversals are checked before transitions: "used to like X, now I find it
        # tedious" matches both shapes, but only the reversal reading is correct --
        # the trailing clause is a judgement about X, not a new preference.
        matched = False
        for pattern in REVERSAL_PATTERNS:
            match = pattern.search(sentence)
            if match is None:
                continue
            candidate = _preference(match.group(1), STANCE_NEGATIVE, sensitive)
            if candidate is not None:
                candidates.append(candidate)
                matched = True
            break
        if matched:
            continue
        # Transitions subsume the single-stance patterns, which would otherwise
        # match only one half of the change.
        transition = _transition_candidates(sentence, sensitive)
        if transition:
            candidates.extend(transition)
            continue
        for pattern, stance in PREFERENCE_PATTERNS:
            match = pattern.search(sentence)
            if match is None:
                continue
            candidate = _preference(match.group(1), stance, sensitive)
            if candidate is not None:
                candidates.append(candidate)
                matched = True
            break
        if matched:
            continue
        for pattern in ABANDON_PATTERNS:
            match = pattern.search(sentence)
            if match is None:
                continue
            candidate = _preference(match.group(1), STANCE_NEGATIVE, sensitive)
            if candidate is not None:
                candidates.append(candidate)
                matched = True
            break
        if matched:
            continue
        narrative = _narrative_abandonment(sentence, sensitive)
        if narrative is not None:
            candidates.append(narrative)
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
