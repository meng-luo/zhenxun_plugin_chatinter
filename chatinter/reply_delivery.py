"""Reply delivery planning for ChatInter conversation turns."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re

HARD_MESSAGE_CHAR_LIMIT = 3_500
CONVERSATIONAL_MIN_CHARS = 48
CONVERSATIONAL_TARGET_CHARS = 48
CONVERSATIONAL_MAX_SEGMENTS = 6
CONVERSATIONAL_MIN_SEGMENT_CHARS = 12

_END_PUNCTUATION = frozenset("。！？!?…")
_SOFT_PUNCTUATION = frozenset("，；;")
_CLOSING_PUNCTUATION = frozenset("\"'”’」』）》】")
_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_STRUCTURED_LINE_PATTERN = re.compile(
    r"(?m)^\s*(?:#{1,6}\s|[-*+•]\s|\d+[.)]\s|>\s|\|.*\|\s*$)"
)


@dataclass(frozen=True, slots=True)
class ReplyDeliveryPlan:
    canonical_text: str
    segments: tuple[str, ...]
    conversational: bool = False
    planned_attachments: int = 0


@dataclass(frozen=True, slots=True)
class DeliveryReceipt:
    canonical_text: str
    planned_segments: int
    delivered_segments: tuple[str, ...] = ()
    planned_attachments: int = 0
    delivered_attachments: int = 0
    complete: bool = False

    @classmethod
    def from_plan(
        cls,
        plan: ReplyDeliveryPlan,
        delivered_segments: tuple[str, ...] | list[str] = (),
        *,
        delivered_attachments: int = 0,
    ) -> DeliveryReceipt:
        delivered = tuple(delivered_segments)
        attachment_count = min(
            max(int(delivered_attachments), 0),
            max(int(plan.planned_attachments), 0),
        )
        planned_total = len(plan.segments) + max(int(plan.planned_attachments), 0)
        delivered_total = len(delivered) + attachment_count
        return cls(
            canonical_text=plan.canonical_text,
            planned_segments=len(plan.segments),
            delivered_segments=delivered,
            planned_attachments=max(int(plan.planned_attachments), 0),
            delivered_attachments=attachment_count,
            complete=planned_total > 0 and delivered_total == planned_total,
        )

    @property
    def delivered_count(self) -> int:
        return len(self.delivered_segments) + self.delivered_attachments

    @property
    def delivered_text(self) -> str:
        if self.complete:
            return self.canonical_text
        return "\n\n".join(self.delivered_segments).strip()


def build_reply_delivery_plan(
    text: str,
    *,
    conversational: bool,
    hard_limit: int = HARD_MESSAGE_CHAR_LIMIT,
    max_segments: int = CONVERSATIONAL_MAX_SEGMENTS,
    attachment_count: int = 0,
) -> ReplyDeliveryPlan:
    canonical = str(text or "").strip()
    if not canonical:
        return ReplyDeliveryPlan(
            canonical_text="",
            segments=(),
            planned_attachments=max(int(attachment_count), 0),
        )

    base_segments = (canonical,)
    natural_split = False
    if conversational and not _has_structured_content(canonical):
        candidate = _conversational_segments(
            canonical,
            max_segments=max(int(max_segments), 0),
        )
        if len(candidate) > 1:
            base_segments = candidate
            natural_split = True

    bounded = tuple(
        chunk
        for segment in base_segments
        for chunk in _split_to_hard_limit(segment, max(int(hard_limit), 256))
        if chunk
    )
    return ReplyDeliveryPlan(
        canonical_text=canonical,
        segments=bounded or (canonical,),
        conversational=natural_split,
        planned_attachments=max(int(attachment_count), 0),
    )


def conversational_send_interval(
    text: str,
    *,
    method: str,
    interval: tuple[float, float],
    log_base: float,
) -> float:
    if method == "log":
        value = str(text or "")
        word_count = (
            len(value.split())
            if value.isascii()
            else sum(char.isalnum() for char in value)
        )
        lower = math.log(word_count + 1, log_base)
        return random.uniform(lower, lower + 0.5)
    return random.uniform(*interval)


def _has_structured_content(text: str) -> bool:
    return bool(
        "`" in text
        or _URL_PATTERN.search(text)
        or _STRUCTURED_LINE_PATTERN.search(text)
    )


def _conversational_segments(
    text: str,
    *,
    max_segments: int,
) -> tuple[str, ...]:
    if len(text) < CONVERSATIONAL_MIN_CHARS:
        return (text,)
    units = _sentence_units(text)
    if len(units) < 2:
        return (text,)

    total_chars = sum(len(unit) for unit in units)
    target = CONVERSATIONAL_TARGET_CHARS
    if max_segments > 0:
        target = max(target, math.ceil(total_chars / max_segments))
    segments: list[str] = []
    current = ""
    for unit in units:
        candidate = _join_units(current, unit)
        if (
            current
            and len(candidate) > target
            and (max_segments == 0 or len(segments) < max_segments - 1)
        ):
            segments.append(current.strip())
            current = unit
        else:
            current = candidate
    if current.strip():
        segments.append(current.strip())

    if len(segments) > 1 and len(segments[-1]) < CONVERSATIONAL_MIN_SEGMENT_CHARS:
        segments[-2] = _join_units(segments[-2], segments[-1]).strip()
        segments.pop()
    return tuple(segment for segment in segments if segment) or (text,)


def _sentence_units(text: str) -> list[str]:
    units: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if text.startswith("\n\n", index):
            _append_unit(units, text[start:index])
            index += 2
            start = index
            continue
        if char not in _END_PUNCTUATION:
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] in _END_PUNCTUATION:
            end += 1
        while end < len(text) and text[end] in _CLOSING_PUNCTUATION:
            end += 1
        _append_unit(units, text[start:end])
        start = end
        index = end
    _append_unit(units, text[start:])

    expanded: list[str] = []
    for unit in units:
        if len(unit) <= CONVERSATIONAL_TARGET_CHARS * 2:
            expanded.append(unit)
            continue
        expanded.extend(_split_long_unit(unit))
    return expanded


def _split_long_unit(text: str) -> list[str]:
    units: list[str] = []
    start = 0
    for index, char in enumerate(text):
        if char not in _SOFT_PUNCTUATION:
            continue
        if index + 1 - start < CONVERSATIONAL_TARGET_CHARS:
            continue
        _append_unit(units, text[start : index + 1])
        start = index + 1
    _append_unit(units, text[start:])
    return units or [text]


def _append_unit(target: list[str], value: str) -> None:
    normalized = value.strip()
    if normalized:
        target.append(normalized)


def _join_units(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    separator = " " if left[-1].isascii() and right[0].isascii() else ""
    return f"{left}{separator}{right}"


def _split_to_hard_limit(text: str, limit: int) -> tuple[str, ...]:
    remaining = text.strip()
    if len(remaining) <= limit:
        return (remaining,)
    chunks: list[str] = []
    while len(remaining) > limit:
        cut = _best_hard_cut(remaining, limit)
        chunk = remaining[:cut].strip()
        if not chunk:
            chunk = remaining[:limit]
            cut = limit
        chunks.append(chunk)
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return tuple(chunks)


def _best_hard_cut(text: str, limit: int) -> int:
    floor = max(int(limit * 0.6), 1)
    window = text[: limit + 1]
    candidates = (
        window.rfind("\n\n", floor),
        window.rfind("\n", floor),
        max((window.rfind(char, floor) for char in _END_PUNCTUATION), default=-1),
        window.rfind(" ", floor),
    )
    best = max(candidates)
    return best + 1 if best >= floor else limit


__all__ = [
    "HARD_MESSAGE_CHAR_LIMIT",
    "DeliveryReceipt",
    "ReplyDeliveryPlan",
    "build_reply_delivery_plan",
    "conversational_send_interval",
]
