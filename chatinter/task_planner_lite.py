from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .route_text import normalize_message_text


@dataclass(frozen=True, slots=True)
class TaskItem:
    task_id: str
    text: str
    order: int
    dependency: str = ""
    side_effect: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "text": self.text,
            "order": self.order,
            "dependency": self.dependency,
            "side_effect": self.side_effect,
        }


@dataclass(frozen=True, slots=True)
class _Segment:
    text: str
    relation: str = ""


class TaskPlannerLite:
    """Deterministic splitter for obvious multi-command turns.

    It deliberately does not infer tools.  The only contract is turning a
    clearly multi-step message into ordered task texts, so single commands and
    normal chat stay on the existing fast path.
    """

    max_tasks = 8

    @classmethod
    def looks_like_obvious_multi_task(cls, message_text: str) -> bool:
        text = _strip_wake_words(message_text)
        if not text:
            return False
        if any(
            marker in text for marker in ("；", ";", "然后", "最后", "顺便", "接着")
        ):
            return True
        if any(marker in text for marker in ("以及", "同时", "并且")):
            return True
        return bool(_RE_AGAIN_CONNECTOR.search(text))

    @classmethod
    def plan(cls, message_text: str) -> tuple[TaskItem, ...]:
        if not cls.looks_like_obvious_multi_task(message_text):
            return ()
        segments = _split_segments(_strip_wake_words(message_text))
        if len(segments) < 2:
            return ()
        if any(not _is_plannable_segment(segment.text) for segment in segments):
            return ()

        tasks: list[TaskItem] = []
        seen: set[str] = set()
        for segment in segments[: cls.max_tasks]:
            text = _clean_segment(segment.text)
            if not text:
                continue
            normalized = normalize_message_text(text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            task_id = f"task_{len(tasks) + 1}"
            dependency = ""
            if tasks and segment.relation == "sequential":
                dependency = tasks[-1].task_id
            tasks.append(
                TaskItem(
                    task_id=task_id,
                    text=normalized,
                    order=len(tasks) + 1,
                    dependency=dependency,
                    side_effect=_has_side_effect(normalized),
                )
            )
        return tuple(tasks) if len(tasks) >= 2 else ()


def plan_task_items(message_text: str) -> tuple[TaskItem, ...]:
    return TaskPlannerLite.plan(message_text)


def task_items_to_payload(
    tasks: tuple[TaskItem, ...] | list[TaskItem],
) -> dict[str, Any]:
    return {
        "source": "task_planner_lite",
        "tasks": [task.to_payload() for task in tasks],
    }


_SEQ = "\x1eSEQ\x1e"
_PAR = "\x1ePAR\x1e"

_RE_AGAIN_CONNECTOR = re.compile(r"(?:[，,。；;、\s]+再)(?=\S)")
_SEQUENTIAL_CONNECTORS = (
    "然后",
    "接着",
    "最后",
    "顺便",
)
_PARALLEL_CONNECTORS = (
    "以及",
    "同时",
    "并且",
)
_ACTION_TERMS = (
    "查",
    "查询",
    "查看",
    "看看",
    "搜",
    "搜索",
    "帮助",
    "状态",
    "信息",
    "列表",
    "排行",
    "统计",
    "余额",
    "签到",
    "抽",
    "随机",
    "来",
    "发",
    "发送",
    "设置",
    "添加",
    "新增",
    "删除",
    "更新",
    "绑定",
    "解绑",
    "播放",
    "点歌",
    "翻译",
    "解析",
    "识别",
    "生成",
    "画",
    "制作",
    "调用",
    "执行",
    "运行",
    "打开",
    "关闭",
    "测试",
    "修复",
    "修改",
    "分析",
)
_SIDE_EFFECT_TERMS = (
    "签到",
    "设置",
    "添加",
    "新增",
    "创建",
    "修改",
    "删除",
    "关闭",
    "开启",
    "绑定",
    "解绑",
    "购买",
    "兑换",
    "发送",
    "发",
    "禁言",
    "撤回",
    "更新",
)
_DISCUSSION_ONLY_TERMS = (
    "为什么",
    "怎么看",
    "讨论",
    "比较",
    "评价",
    "区别",
    "原理",
    "取舍",
)
_COMMAND_LIKE_RE = re.compile(r"^[#/!！./]?[A-Za-z0-9][A-Za-z0-9_\-]{1,}")


def _strip_wake_words(text: str) -> str:
    message = normalize_message_text(text)
    for prefix in ("真寻，", "真寻,", "真寻 "):
        if message.startswith(prefix):
            return message[len(prefix) :].strip()
    return message


def _split_segments(message: str) -> list[_Segment]:
    text = normalize_message_text(message)
    if not text:
        return []
    text = re.sub(r"[;；]+", _SEQ, text)
    for marker in _SEQUENTIAL_CONNECTORS:
        text = text.replace(marker, _SEQ)
    text = _RE_AGAIN_CONNECTOR.sub(_SEQ, text)
    for marker in _PARALLEL_CONNECTORS:
        text = text.replace(marker, _PAR)

    parts = re.split(f"({re.escape(_SEQ)}|{re.escape(_PAR)})", text)
    segments: list[_Segment] = []
    relation_for_next = ""
    for part in parts:
        if not part:
            continue
        if part == _SEQ:
            relation_for_next = "sequential"
            continue
        if part == _PAR:
            relation_for_next = "parallel"
            continue
        cleaned = _clean_segment(part)
        if cleaned:
            segments.append(_Segment(text=cleaned, relation=relation_for_next))
        relation_for_next = ""
    return segments


def _clean_segment(segment: str) -> str:
    text = normalize_message_text(segment).strip(" ，,。；;、")
    for prefix in ("先", "然后", "接着", "再", "最后", "顺便", "以及", "同时", "并且"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip(" ，,。；;、")
    return text


def _is_plannable_segment(segment: str) -> bool:
    text = _clean_segment(segment)
    if len(text) < 2:
        return False
    if _is_discussion_only(text):
        return False
    return _has_action_term(text) or _looks_command_like(text)


def _is_discussion_only(text: str) -> bool:
    normalized = normalize_message_text(text)
    if not any(term in normalized for term in _DISCUSSION_ONLY_TERMS):
        return False
    return not _has_action_term(normalized)


def _has_action_term(text: str) -> bool:
    return any(term in text for term in _ACTION_TERMS)


def _looks_command_like(text: str) -> bool:
    normalized = normalize_message_text(text)
    return bool(_COMMAND_LIKE_RE.match(normalized)) or "@" in normalized


def _has_side_effect(text: str) -> bool:
    return any(term in text for term in _SIDE_EFFECT_TERMS)


__all__ = [
    "TaskItem",
    "TaskPlannerLite",
    "plan_task_items",
    "task_items_to_payload",
]
