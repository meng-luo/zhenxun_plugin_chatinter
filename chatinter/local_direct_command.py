from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from .command_index import CommandCandidate
from .models.pydantic_models import CommandSlotSpec
from .route_text import match_command_head, normalize_message_text


@dataclass(frozen=True)
class LocalDirectCommandPlan:
    candidate: CommandCandidate
    raw_slots: dict[str, str] = field(default_factory=dict)
    reason: str = "local_direct_command:direct_match"
    segment: str = ""


@dataclass(frozen=True)
class LocalDirectCommandBatchPlan:
    steps: list[LocalDirectCommandPlan]
    reason: str = "local_direct_command:batch"


_DISCUSSION_TERMS = (
    "聊聊",
    "为什么",
    "怎么看",
    "讨论",
    "分析",
    "比较",
    "评价",
    "隐喻",
    "区别",
    "原理",
    "取舍",
)
_META_QUERY_TERMS = ("有哪些功能", "用法", "怎么用")
_COMPLEX_SLOT_TYPES = {"at", "image"}
_TEXT_SLOT_TYPES = {"", "text", "str", "string", "url", "link", "id", "int", "float"}
_GENERIC_MARKERS = (
    "内容",
    "文本",
    "关键词",
    "关键字",
    "链接",
    "地址",
    "url",
    "id",
    "文案",
    "句子",
    "名称",
)
_EXPLICIT_SCORE_MIN = 100
_ASCII_BOUNDARY_PATTERN = re.compile(r"[0-9A-Za-z_]")
_EXPLICIT_LEAD_INS = (
    "请",
    "请帮我",
    "帮我",
    "帮忙",
    "麻烦",
    "麻烦你",
    "给我",
    "我要",
    "我想",
    "想要",
    "想",
    "来",
    "来一",
    "来个",
    "来一个",
    "来句",
    "来一句",
    "来条",
    "来一条",
    "发",
    "发送",
    "查",
    "查询",
    "查看",
    "搜",
    "搜索",
    "播放",
    "用",
    "执行",
    "运行",
)


def plan_local_direct_command(
    *,
    message_text: str,
    candidates: list[CommandCandidate],
    tool_map: dict[str, Any],
) -> LocalDirectCommandPlan | None:
    del tool_map
    message = _strip_wake_words(message_text)
    if _looks_multi_task(message):
        return None
    candidate = _select_single_candidate(message, candidates)
    if candidate is None:
        return None
    raw_slots = _extract_slots(message, candidate)
    if raw_slots is None:
        return None
    return LocalDirectCommandPlan(
        candidate=candidate,
        raw_slots=raw_slots,
        segment=message,
    )


def plan_local_direct_command_batch(
    *,
    message_text: str,
    candidates: list[CommandCandidate],
    tool_map: dict[str, Any],
) -> LocalDirectCommandBatchPlan | None:
    del tool_map
    segments = _split_segments(_strip_wake_words(message_text))
    if len(segments) < 2:
        return None
    steps: list[LocalDirectCommandPlan] = []
    for segment in segments:
        if _is_discussion_segment(segment):
            return None
        candidate = _select_batch_candidate(segment, candidates)
        if candidate is None:
            return None
        raw_slots = _extract_slots(segment, candidate)
        if raw_slots is None:
            return None
        steps.append(
            LocalDirectCommandPlan(
                candidate=candidate,
                raw_slots=raw_slots,
                segment=segment,
            )
        )
    return LocalDirectCommandBatchPlan(steps=steps) if len(steps) >= 2 else None


def _strip_wake_words(text: str) -> str:
    message = normalize_message_text(text)
    for prefix in ("真寻，", "真寻,", "真寻 "):
        if message.startswith(prefix):
            return message[len(prefix) :].strip()
    return message


def _looks_multi_task(message: str) -> bool:
    return any(
        marker in message for marker in ("；", ";", "然后", "最后", "再", "顺便")
    )


def _split_segments(message: str) -> list[str]:
    text = normalize_message_text(message)
    for marker in ("；", ";"):
        text = text.replace(marker, "；")
    if "；" not in text:
        for marker in (
            "，最后",
            ",最后",
            "最后",
            "，然后",
            ",然后",
            "然后",
            "，再",
            ",再",
            "再",
        ):
            text = text.replace(marker, "；")
    parts = [part.strip(" ，,。") for part in text.split("；") if part.strip()]
    return [
        _strip_segment_prefix(part) for part in parts if _strip_segment_prefix(part)
    ]


def _strip_segment_prefix(segment: str) -> str:
    text = normalize_message_text(segment)
    for prefix in ("先", "然后", "再", "最后", "顺便"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip(" ，,")
    return text


def _select_single_candidate(
    message: str,
    candidates: list[CommandCandidate],
) -> CommandCandidate | None:
    usable = [
        candidate
        for candidate in candidates
        if _candidate_score_ok(candidate)
        and not _is_meta_query_for_usage(message, candidate)
        and not _is_broad_random_mismatch(message, candidate)
    ]
    if not usable:
        return None
    scored = [(candidate, _segment_score(message, candidate)) for candidate in usable]
    scored = [
        (candidate, score)
        for candidate, score in scored
        if score >= _EXPLICIT_SCORE_MIN
    ]
    if not scored:
        return None
    scored.sort(
        key=lambda item: (item[1], *_candidate_rank(item[0])),
        reverse=True,
    )
    candidate = scored[0][0]
    if candidate.score < 80 and not candidate.exact_protected:
        return None
    return candidate


def _select_batch_candidate(
    segment: str,
    candidates: list[CommandCandidate],
) -> CommandCandidate | None:
    scored = [
        (candidate, _segment_score(segment, candidate)) for candidate in candidates
    ]
    scored = [
        (candidate, score)
        for candidate, score in scored
        if score >= _EXPLICIT_SCORE_MIN
        and not _is_meta_query_for_usage(segment, candidate)
        and not _is_broad_random_mismatch(segment, candidate)
    ]
    if not scored:
        return None
    scored.sort(key=lambda item: (item[1], *_candidate_rank(item[0])), reverse=True)
    return scored[0][0]


def _candidate_score_ok(candidate: CommandCandidate) -> bool:
    if candidate.exact_protected:
        return True
    reason = normalize_message_text(candidate.reason)
    if "weak" in reason and candidate.score < 120:
        return False
    return candidate.score >= 80


def _candidate_rank(candidate: CommandCandidate) -> tuple[float, float]:
    schema_quality = 0.0
    if candidate.features is not None:
        schema_quality = float(candidate.features.schema_score or 0.0)
    return (float(candidate.score), schema_quality)


def _is_meta_query_for_usage(message: str, candidate: CommandCandidate) -> bool:
    role = normalize_message_text(str(getattr(candidate.schema, "command_role", "")))
    if role != "usage":
        return False
    return any(word in message for word in _META_QUERY_TERMS)


def _is_broad_random_mismatch(message: str, candidate: CommandCandidate) -> bool:
    if _role(candidate) != "random":
        return False
    if _tail_after_invocation(message, candidate):
        return False
    return not bool(_choice_payload(message, candidate))


def _extract_slots(
    message: str,
    candidate: CommandCandidate,
) -> dict[str, str] | None:
    slots = list(getattr(candidate.schema, "slots", []) or [])
    policy = _payload_policy(candidate)
    raw: dict[str, str] = {"task_text": message, "payload_hint": ""}
    if _has_complex_required_slots(slots):
        return None
    if not slots and policy == "none":
        return raw
    if not slots:
        value = _extract_free_payload(message, candidate)
        raw["payload_hint"] = value
        return raw if value or policy in {"none", "free_tail"} else None
    if len(slots) == 1:
        value, payload_hint = _extract_single_slot_value(message, candidate, slots[0])
        return _slot_payload(raw, slots, value, payload_hint=payload_hint)
    return _slot_values_from_message(raw, slots, message, candidate)


def _slot_payload(
    raw: dict[str, str],
    slots: list[CommandSlotSpec],
    value: str,
    *,
    payload_hint: str = "",
) -> dict[str, str] | None:
    value = normalize_message_text(value)
    if not slots:
        raw["payload_hint"] = payload_hint or value
        return raw
    target = slots[0]
    if target.required and not value:
        return None
    if value:
        raw[target.name] = value
    raw["payload_hint"] = normalize_message_text(payload_hint)
    return raw


def _slot_values_from_message(
    raw: dict[str, str],
    slots: list[CommandSlotSpec],
    message: str,
    candidate: CommandCandidate,
) -> dict[str, str] | None:
    tail = _tail_after_invocation(message, candidate)
    if not tail and _is_random_candidate(candidate):
        tail = _choice_payload(message, candidate)
    if not tail:
        alias_value = _exact_alias_value_for_required_slot(message, candidate, slots)
        if alias_value is not None:
            raw[alias_value[0].name] = alias_value[1]
            return raw
    if not tail:
        if all(not slot.required for slot in slots):
            return raw
        return None

    option_values, positional_tail = _extract_named_option_values(tail, slots)
    positional_slots = [
        slot
        for slot in slots
        if slot.name not in option_values and _slot_accepts_text(slot)
    ]
    parts = positional_tail.split(maxsplit=max(len(positional_slots) - 1, 1))
    positional_index = 0
    for slot in slots:
        if slot.name in option_values:
            value = option_values[slot.name]
        elif _slot_accepts_text(slot):
            value = (
                normalize_message_text(parts[positional_index])
                if positional_index < len(parts)
                else ""
            )
            positional_index += 1
        else:
            value = ""
        if slot.required and not value:
            return None
        if value:
            raw[slot.name] = value
    return raw


def _extract_single_slot_value(
    message: str,
    candidate: CommandCandidate,
    slot: CommandSlotSpec,
) -> tuple[str, str]:
    if _is_random_candidate(candidate):
        choice = _choice_payload(message, candidate)
        if choice:
            return choice, choice
    tail = _tail_after_invocation(message, candidate)
    if tail:
        return tail, ""
    alias_value = _exact_alias_value_for_required_slot(message, candidate, [slot])
    if alias_value is not None:
        return alias_value[1], ""
    quoted = _quoted_text(message)
    if quoted:
        return quoted, quoted
    marked = _marked_payload(message, slot)
    if marked:
        return marked, marked
    transformed = _transform_or_lookup_payload(message, candidate)
    if transformed:
        return transformed, transformed
    identifier = _identifier_payload(message, candidate, slot)
    if identifier:
        return identifier, identifier
    if slot.required:
        return "", ""
    payload = _extract_free_payload(message, candidate)
    return payload, payload


def _extract_free_payload(message: str, candidate: CommandCandidate) -> str:
    return _tail_after_invocation(message, candidate) or _quoted_text(message)


def _has_complex_required_slots(slots: list[CommandSlotSpec]) -> bool:
    return any(
        slot.required and normalize_message_text(str(slot.type)) in _COMPLEX_SLOT_TYPES
        for slot in slots
    )


def _slot_accepts_text(slot: CommandSlotSpec) -> bool:
    return normalize_message_text(str(slot.type)) in _TEXT_SLOT_TYPES


def _extract_named_option_values(
    tail: str,
    slots: list[CommandSlotSpec],
) -> tuple[dict[str, str], str]:
    """Extract explicit option-like slots without plugin-specific branches."""

    text = normalize_message_text(tail)
    values: dict[str, str] = {}
    for slot in slots:
        for alias in _slot_option_aliases(slot):
            pattern = rf"(?<!\S){re.escape(alias)}(?:\s+|=|＝)(\S+)"
            match = re.search(pattern, text)
            if not match:
                continue
            values[slot.name] = normalize_message_text(match.group(1))
            text = normalize_message_text(
                f"{text[: match.start()]} {text[match.end() :]}"
            )
            break
    return values, text


def _slot_option_aliases(slot: CommandSlotSpec) -> list[str]:
    aliases: list[str] = []
    for raw in (slot.name, *list(slot.aliases or [])):
        alias = normalize_message_text(str(raw or ""))
        if alias and (
            alias.startswith("-") or re.fullmatch(r"[A-Za-z][0-9A-Za-z_-]{0,16}", alias)
        ):
            aliases.append(alias)
    return list(dict.fromkeys(aliases))


def _exact_alias_value_for_required_slot(
    message: str,
    candidate: CommandCandidate,
    slots: list[CommandSlotSpec],
) -> tuple[CommandSlotSpec, str] | None:
    required_text_slots = [
        slot for slot in slots if slot.required and _slot_accepts_text(slot)
    ]
    if len(required_text_slots) != 1:
        return None
    text = normalize_message_text(message)
    for phrase in _invocation_phrases(candidate):
        if phrase and phrase == text:
            return required_text_slots[0], phrase
    return None


def _segment_score(segment: str, candidate: CommandCandidate) -> int:
    text = normalize_message_text(segment)
    if not text or _is_discussion_segment(text):
        return 0
    invocation_score = _invocation_score(text, candidate)
    if invocation_score <= 0:
        return 0
    return invocation_score + _shortcut_invocation_bonus(text, candidate)


def _invocation_score(text: str, candidate: CommandCandidate) -> int:
    score = 0
    for index, phrase in enumerate(_invocation_phrases(candidate)):
        if phrase and _explicit_phrase_match(text, phrase):
            score = max(score, 120 if index == 0 else 110)
    return score


def _shortcut_invocation_bonus(text: str, candidate: CommandCandidate) -> int:
    """Prefer schemas that explicitly map user-facing aliases to parser renders."""

    for item in getattr(candidate.schema, "shortcut_renders", []) or []:
        if not isinstance(item, dict):
            continue
        alias = normalize_message_text(str(item.get("alias") or ""))
        render = normalize_message_text(str(item.get("render") or ""))
        if (alias and _explicit_phrase_match(text, alias)) or (
            render and _explicit_phrase_match(text, render)
        ):
            return 18
    return 0


def _candidate_search_text(candidate: CommandCandidate) -> str:
    tool = candidate.tool
    values: list[str] = [
        candidate.schema.command_id,
        candidate.schema.head,
        candidate.plugin_name,
        candidate.schema.description,
        " ".join(candidate.schema.aliases),
        " ".join(getattr(candidate.schema, "retrieval_phrases", []) or []),
    ]
    if tool is not None:
        values.extend(
            [
                getattr(tool, "command_id", ""),
                getattr(tool, "head", ""),
                getattr(tool, "plugin_name", ""),
                getattr(tool, "description", ""),
                getattr(tool, "capability_text", ""),
                " ".join(getattr(tool, "retrieval_phrases", []) or []),
                " ".join(getattr(tool, "task_verbs", []) or []),
                " ".join(getattr(tool, "input_requirements", []) or []),
                " ".join(getattr(tool, "use_cases", []) or []),
                " ".join(getattr(tool, "intent_types", []) or []),
            ]
        )
    return normalize_message_text(
        " ".join(str(value or "") for value in values)
    ).casefold()


def _payload_policy(candidate: CommandCandidate) -> str:
    return normalize_message_text(str(getattr(candidate.schema, "payload_policy", "")))


def _role(candidate: CommandCandidate) -> str:
    return normalize_message_text(str(getattr(candidate.schema, "command_role", "")))


def _is_random_candidate(candidate: CommandCandidate) -> bool:
    return _role(candidate) == "random" or "random" in _intent_types(candidate)


def _intent_types(candidate: CommandCandidate) -> set[str]:
    tool = candidate.tool
    values = list(getattr(tool, "intent_types", []) or []) if tool is not None else []
    return {normalize_message_text(str(item)) for item in values if str(item or "")}


def _invocation_phrases(candidate: CommandCandidate) -> list[str]:
    phrases: list[str] = []

    def add(value: str) -> None:
        text = normalize_message_text(value)
        if text and text not in phrases:
            phrases.append(text)

    add(candidate.schema.head)
    for alias in candidate.schema.aliases:
        add(alias)
    for item in getattr(candidate.schema, "shortcut_renders", []) or []:
        if isinstance(item, dict):
            add(str(item.get("alias") or ""))
            add(str(item.get("render") or ""))
    return sorted(phrases, key=len, reverse=True)


def _tail_after_invocation(message: str, candidate: CommandCandidate) -> str:
    text = normalize_message_text(message)
    for phrase, start, end in _explicit_invocation_matches(text, candidate):
        del start
        if not phrase:
            continue
        tail = text[end:].strip(" ，,。:：")
        if tail:
            return tail
    return ""


def _explicit_invocation_matches(
    text: str,
    candidate: CommandCandidate,
) -> list[tuple[str, int, int]]:
    matches: list[tuple[str, int, int]] = []
    for phrase in _invocation_phrases(candidate):
        match = _find_explicit_phrase(text, phrase)
        if match is not None:
            matches.append((phrase, *match))
    matches.sort(key=lambda item: (item[1], -len(item[0])))
    return matches


def _explicit_phrase_match(text: str, phrase: str) -> bool:
    return _find_explicit_phrase(text, phrase) is not None


def _find_explicit_phrase(text: str, phrase: str) -> tuple[int, int] | None:
    normalized = normalize_message_text(text)
    normalized_phrase = normalize_message_text(phrase)
    if not normalized or not normalized_phrase:
        return None

    text_fold = normalized.casefold()
    phrase_fold = normalized_phrase.casefold()
    if match_command_head(text_fold, phrase_fold):
        return 0, len(normalized_phrase)

    start = 0
    while True:
        index = text_fold.find(phrase_fold, start)
        if index < 0:
            return None
        end = index + len(normalized_phrase)
        if _has_explicit_phrase_boundary(text_fold, phrase_fold, index, end):
            return index, end
        start = index + 1


def _has_explicit_phrase_boundary(
    text: str,
    phrase: str,
    start: int,
    end: int,
) -> bool:
    if _is_ascii_token(phrase):
        right_ok = end >= len(text) or not _ASCII_BOUNDARY_PATTERN.match(text[end])
        return right_ok and _is_explicit_lead_in(text[:start])

    # Chinese command heads and shortcuts are commonly used as sticky commands,
    # e.g. "点歌晴天" or "查看完美天梯排行10".  Single-char heads are too noisy
    # unless they matched the command head at the start above.
    return (
        len(phrase) >= 2
        and any("\u4e00" <= char <= "\u9fff" for char in phrase)
        and _is_explicit_lead_in(text[:start])
    )


def _is_ascii_token(text: str) -> bool:
    return bool(text) and all(
        char.isascii() and (char.isalnum() or char in "_-") for char in text
    )


def _is_explicit_lead_in(prefix: str) -> bool:
    compact = normalize_message_text(prefix).strip(" ，,。:：").replace(" ", "")
    if not compact:
        return True
    return compact in _EXPLICIT_LEAD_INS


def _quoted_text(message: str) -> str:
    match = re.search(r"[“\"]([^”\"]+)[”\"]", message)
    return normalize_message_text(match.group(1)) if match else ""


def _marked_payload(message: str, slot: CommandSlotSpec) -> str:
    markers = [slot.name, *list(slot.aliases or []), *_GENERIC_MARKERS]
    for marker in dict.fromkeys(normalize_message_text(item) for item in markers):
        if not marker:
            continue
        match = re.search(rf"{re.escape(marker)}\s*(?:是|为|:|：)\s*(.+)", message)
        if match:
            return normalize_message_text(match.group(1))
    return ""


def _transform_or_lookup_payload(message: str, candidate: CommandCandidate) -> str:
    if not ({"transform", "query"} & _intent_types(candidate)):
        return ""
    patterns = (
        r"把\s*(.+?)\s*(?:翻译|翻成|转成|转换|译成|变成)",
        r"把\s*([0-9A-Za-z_.:/\-]+).*?(?:解释|展开|查询|查一下|识别|解析)",
        r"(?:翻译|解释|展开|查询|查一下|查下|搜索|搜|提取|识别|解析)\s*([0-9A-Za-z_.:/\-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return normalize_message_text(match.group(1))
    return ""


def _identifier_payload(
    message: str,
    candidate: CommandCandidate,
    slot: CommandSlotSpec,
) -> str:
    if not _slot_suggests_identifier(candidate, slot):
        return ""
    patterns = (
        r"(https?://\S+)",
        r"\b(BV[0-9A-Za-z]+|av\d+|cv\d+)\b",
        r"\b([A-Za-z]{1,8}[0-9][0-9A-Za-z_-]{3,})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            return normalize_message_text(match.group(1))
    return ""


def _slot_suggests_identifier(
    candidate: CommandCandidate,
    slot: CommandSlotSpec,
) -> bool:
    text = normalize_message_text(
        " ".join(
            [
                slot.name,
                slot.type,
                slot.description,
                " ".join(slot.aliases or []),
                _candidate_search_text(candidate),
            ]
        )
    ).casefold()
    return any(term in text for term in ("url", "link", "链接", "地址", "id", "bv"))


def _choice_payload(message: str, candidate: CommandCandidate) -> str:
    quoted = re.findall(r"[“\"]([^”\"]+)[”\"]", message)
    if len(quoted) >= 2:
        return " ".join(normalize_message_text(item) for item in quoted if item)
    match = re.search(r"在(.+?)(?:里面|里|之间).*?(?:选|抽|决定|随机)", message)
    value = normalize_message_text(match.group(1)) if match else ""
    if not value:
        value = _tail_after_invocation(message, candidate)
    value = value.replace("、", " ").replace("/", " ").replace("和", " ")
    value = " ".join(value.split())
    return value if len(value.split()) >= 2 else ""


def _is_discussion_segment(segment: str) -> bool:
    text = normalize_message_text(segment)
    return any(term in text for term in _DISCUSSION_TERMS)


__all__ = [
    "LocalDirectCommandBatchPlan",
    "LocalDirectCommandPlan",
    "plan_local_direct_command",
    "plan_local_direct_command_batch",
]
