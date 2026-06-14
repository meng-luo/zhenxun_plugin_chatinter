from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

from .route_text import normalize_message_text
from .task_ledger import TaskLedger, TaskLedgerEntry

_INTENT_TERMS: dict[str, tuple[str, ...]] = {
    "query": ("查", "查询", "查看", "看看", "搜索", "搜", "找", "解释", "展开"),
    "status": ("状态", "信息", "记录", "排行", "统计", "余额", "列表", "版本"),
    "help": ("帮助", "说明", "用法", "教程", "文档", "入口"),
    "translate": ("翻译", "翻成", "译成", "转成", "中文", "英文", "日语"),
    "transform": ("翻译", "转换", "转成", "解释", "展开", "识别", "解析"),
    "random": ("随机", "抽", "选", "选择", "决定", "roll", "掷", "投"),
    "send": ("发", "发送", "来一", "来句", "来条", "给我"),
    "generate": ("生成", "制作", "做", "画", "来一", "发"),
    "play": ("播放", "点歌", "点播", "听歌", "歌曲"),
    "mutate": ("添加", "新增", "创建", "设置", "绑定", "修改", "删除", "关闭"),
    "execute": ("执行", "调用", "使用", "运行", "打开", "启动"),
}

_VERB_ALIASES: dict[str, tuple[str, ...]] = {
    "查询": _INTENT_TERMS["query"],
    "统计": _INTENT_TERMS["status"],
    "帮助": _INTENT_TERMS["help"],
    "翻译": _INTENT_TERMS["translate"],
    "随机": _INTENT_TERMS["random"],
    "生成": _INTENT_TERMS["generate"],
    "播放": _INTENT_TERMS["play"],
    "添加": ("添加", "新增", "创建", "设置", "绑定"),
    "删除": ("删除", "移除", "取消", "关闭", "退回", "解绑"),
}

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


def plan_local_task_ledger(
    message_text: str,
    *,
    available_capabilities: Iterable[dict[str, Any]],
) -> TaskLedger | None:
    message = _strip_wake_words(message_text)
    segments = _split_task_segments(message)
    if len(segments) < 2:
        return None
    capabilities = [cap for cap in available_capabilities if isinstance(cap, dict)]
    if not capabilities:
        return None

    tasks: list[TaskLedgerEntry] = []
    used: set[str] = set()
    for segment in segments:
        if _is_discussion_segment(segment):
            return None
        capability = _best_capability(segment, capabilities, used=used)
        if capability is None:
            return None
        command_id = _command_id(capability)
        if not command_id:
            return None
        used.add(command_id)
        goal = _clean_segment_goal(segment)
        tasks.append(
            TaskLedgerEntry(
                task_id=f"task_{len(tasks) + 1}",
                goal=goal,
                intent_type=_intent_type(capability),
                requires_real_tool=bool(capability.get("requires_real_tool", True)),
                expected_capabilities=[command_id],
                acceptance_criteria=[f"observation covers {goal}"],
                reason="local_task_ledger:capability_match",
            )
        )

    if len(tasks) < 2:
        return None
    if len({tuple(task.expected_capabilities) for task in tasks}) == 1:
        return None
    return TaskLedger.create(
        original_message=message_text,
        tasks=tasks,
        reason="local_task_ledger:multi_task_capability_match",
    )


def _strip_wake_words(text: str) -> str:
    message = normalize_message_text(text)
    for prefix in ("真寻，", "真寻,", "真寻 "):
        if message.startswith(prefix):
            return message[len(prefix) :].strip()
    return message


def _split_task_segments(message: str) -> list[str]:
    normalized = normalize_message_text(message)
    if not normalized:
        return []
    for marker in ("；", ";"):
        normalized = normalized.replace(marker, "；")
    parts = [part.strip(" ，,。") for part in normalized.split("；") if part.strip()]
    if len(parts) > 1:
        return [
            _clean_segment_goal(part) for part in parts if _clean_segment_goal(part)
        ]

    splitters = (
        "，最后",
        ",最后",
        "最后",
        "，然后",
        ",然后",
        "然后",
        "，再",
        ",再",
        "再",
    )
    text = normalized
    for splitter in splitters:
        text = text.replace(splitter, "；")
    text = text.replace("，先", "；先").replace(",先", "；先")
    parts = [part.strip(" ，,。") for part in text.split("；") if part.strip()]
    return [_clean_segment_goal(part) for part in parts if _clean_segment_goal(part)]


def _clean_segment_goal(segment: str) -> str:
    text = normalize_message_text(segment)
    for prefix in ("先", "然后", "再", "最后", "顺便", "把"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip(" ，,")
    return text


def _best_capability(
    segment: str,
    capabilities: list[dict[str, Any]],
    *,
    used: set[str],
) -> dict[str, Any] | None:
    scored = []
    for capability in capabilities:
        command_id = _command_id(capability)
        if not command_id or command_id in used:
            continue
        score = _score_capability(segment, capability)
        if score > 0:
            scored.append((score, capability))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _score_capability(segment: str, capability: dict[str, Any]) -> int:
    text = normalize_message_text(segment).casefold()
    if _is_discussion_segment(text):
        return 0
    haystack = _capability_search_text(capability)
    score = 0
    for phrase in _capability_invocation_phrases(capability):
        if phrase and phrase.casefold() in text:
            score += 35
    for intent in _capability_intents(capability):
        if any(term.casefold() in text for term in _INTENT_TERMS.get(intent, ())):
            score += 20
    for verb in _capability_task_verbs(capability):
        if any(term.casefold() in text for term in _VERB_ALIASES.get(verb, ())):
            score += 16
    for token in _tokens(text):
        if token and token in haystack:
            score += 8 if len(token) >= 4 else 3
    return score


def _is_discussion_segment(segment: str) -> bool:
    text = normalize_message_text(segment).casefold()
    return any(term.casefold() in text for term in _DISCUSSION_TERMS)


def _capability_search_text(capability: dict[str, Any]) -> str:
    values = [
        capability.get("command_id"),
        capability.get("head"),
        capability.get("plugin"),
        capability.get("description"),
        capability.get("capability_text"),
        capability.get("role"),
        capability.get("output_mode"),
        capability.get("source_of_truth"),
        " ".join(str(item) for item in capability.get("intent_types", []) or []),
        " ".join(str(item) for item in capability.get("task_verbs", []) or []),
        " ".join(str(item) for item in capability.get("use_cases", []) or []),
        " ".join(str(item) for item in capability.get("aliases", []) or []),
        " ".join(str(item) for item in capability.get("retrieval_phrases", []) or []),
    ]
    return normalize_message_text(" ".join(str(value or "") for value in values))


def _capability_invocation_phrases(capability: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for value in (
        capability.get("head"),
        capability.get("plugin"),
        *list(capability.get("aliases", []) or []),
    ):
        text = normalize_message_text(str(value or ""))
        if text and text not in values:
            values.append(text)
    return sorted(values, key=len, reverse=True)


def _capability_intents(capability: dict[str, Any]) -> set[str]:
    return {
        normalize_message_text(str(item or "")).casefold()
        for item in capability.get("intent_types", []) or []
        if normalize_message_text(str(item or ""))
    }


def _capability_task_verbs(capability: dict[str, Any]) -> set[str]:
    return {
        normalize_message_text(str(item or ""))
        for item in capability.get("task_verbs", []) or []
        if normalize_message_text(str(item or ""))
    }


def _tokens(text: str) -> list[str]:
    normalized = normalize_message_text(text).casefold().replace("/", " ")
    result: list[str] = []
    for token in re.findall(r"[0-9a-z_.:/-]+|[\u4e00-\u9fff]{2,}", normalized):
        token = token.strip()
        if not token:
            continue
        result.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{3,}", token):
            for size in range(2, min(len(token), 4) + 1):
                for start in range(0, len(token) - size + 1):
                    result.append(token[start : start + size])
    return list(dict.fromkeys(result))


def _command_id(capability: dict[str, Any]) -> str:
    return normalize_message_text(
        str(capability.get("command_id", "") or capability.get("tool", "") or "")
    )


def _intent_type(capability: dict[str, Any]) -> str:
    intents = capability.get("intent_types")
    if isinstance(intents, list) and intents:
        return normalize_message_text(str(intents[0] or "")) or "unknown"
    return "unknown"


__all__ = ["plan_local_task_ledger"]
