from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import TYPE_CHECKING, Any, Literal

from .persistence import read_json, state_path, utc_now_iso, write_json

if TYPE_CHECKING:
    from .intent_classifier import IntentClassification


def normalize_message_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def contains_any(text: str, needles: tuple[str, ...]) -> bool:
    normalized = normalize_message_text(text)
    return any(item and item in normalized for item in needles)


ChatDialogueKind = Literal[
    "casual_chat",
    "factual_qa",
    "emotional_support",
    "recap",
    "identity_query",
    "memory_update",
    "explain_context",
    "complex_reasoning",
]

ChatDialogueStyle = Literal[
    "concise",
    "detailed",
    "empathetic",
    "step_by_step",
]

DialogueTone = Literal[
    "casual",
    "warm",
    "focused",
    "playful",
    "serious",
    "empathetic",
]

UserEmotion = Literal[
    "neutral",
    "happy",
    "stressed",
    "sad",
    "angry",
    "confused",
]

DialoguePurpose = Literal[
    "chat",
    "answer",
    "support",
    "plan",
    "identity",
    "memory_update",
    "agent_task",
]

ResponseLength = Literal["short", "medium", "long"]
DialogueContinuity = Literal["new_topic", "same_topic", "followup", "unknown"]
GroupReplyPolicy = Literal[
    "brief_react",
    "answer_directly",
    "support",
    "structured",
    "agent",
]
ReplyPosture = Literal[
    "listen",
    "answer",
    "banter",
    "support",
    "clarify",
    "step",
]
GroupAtmosphere = Literal[
    "quiet",
    "active",
    "focused",
    "playful",
    "tense",
    "unknown",
]
AddressMode = Literal[
    "direct",
    "soft",
    "mention_aware",
    "group_general",
]

_DIALOGUE_STATE_PATH = state_path("chat_dialogue_states.json")
_DIALOGUE_STATE_TTL_SECONDS = 14 * 24 * 3600
_DIALOGUE_STATE_MAX = 4096

_EMOTION_HINTS = (
    "难受",
    "累",
    "焦虑",
    "烦",
    "崩溃",
    "委屈",
    "失眠",
    "不开心",
    "想哭",
    "压力",
    "害怕",
)
_FACTUAL_HINTS = (
    "是什么",
    "是啥",
    "为什么",
    "原因",
    "区别",
    "原理",
    "资料",
    "规则",
    "怎么回事",
)
_COMPLEX_HINTS = (
    "分析",
    "方案",
    "设计",
    "实现",
    "步骤",
    "排查",
    "优化",
    "对比",
    "架构",
    "总结",
)
_MEMORY_UPDATE_HINTS = (
    "记住",
    "记一下",
    "以后叫",
    "叫我",
    "我喜欢",
    "我不喜欢",
    "别叫",
)
_CASUAL_HINTS = ("随便聊", "聊两句", "陪我聊", "闲聊", "吐槽")
_HAPPY_HINTS = ("开心", "高兴", "好耶", "舒服", "爽", "赢", "喜欢")
_SAD_HINTS = ("难过", "伤心", "委屈", "想哭", "失落", "不开心")
_ANGRY_HINTS = ("生气", "气死", "烦死", "恼火", "讨厌", "离谱")
_CONFUSED_HINTS = ("不懂", "看不懂", "迷惑", "懵", "不会", "什么意思")
_PLAYFUL_HINTS = ("哈哈", "笑死", "草", "乐", "整活", "好玩", "吐槽")
_FOLLOWUP_HINTS = ("怎么选", "怎么办", "你觉得", "要不要", "该不该")


@dataclass(frozen=True)
class ChatDialoguePlan:
    kind: ChatDialogueKind
    confidence: float = 0.72
    target_hint: str = ""
    need_grounding: bool = False
    need_memory_lookup: bool = False
    need_rewrite_check: bool = False
    style: ChatDialogueStyle = "concise"
    reason: str = "default"

    @property
    def is_complex(self) -> bool:
        return self.kind in {"complex_reasoning", "factual_qa"}


@dataclass(frozen=True)
class DialogueState:
    """Compact turn-level dialogue state for prompt and quality policy.

    This is intentionally lightweight and non-authoritative: it nudges tone and
    memory recall, but it does not decide plugin routing.
    """

    tone: DialogueTone = "casual"
    user_emotion: UserEmotion = "neutral"
    dialogue_purpose: DialoguePurpose = "chat"
    need_followup: bool = False
    response_length: ResponseLength = "short"
    topic_hint: str = ""
    continuity: DialogueContinuity = "unknown"
    group_reply_policy: GroupReplyPolicy = "brief_react"
    reply_posture: ReplyPosture = "listen"
    group_atmosphere: GroupAtmosphere = "unknown"
    address_mode: AddressMode = "direct"
    relevant_people_count: int = 0
    persisted_turns: int = 0
    last_outcome: str = ""
    last_user_message: str = ""
    last_reply_summary: str = ""
    reason: str = "default"

    def to_xml(self) -> str:
        return "\n".join(
            (
                "<dialogue_state>",
                f"tone={self.tone}",
                f"user_emotion={self.user_emotion}",
                f"dialogue_purpose={self.dialogue_purpose}",
                f"need_followup={int(self.need_followup)}",
                f"response_length={self.response_length}",
                f"topic_hint={_xml_escape(self.topic_hint)}",
                f"continuity={self.continuity}",
                f"group_reply_policy={self.group_reply_policy}",
                f"reply_posture={self.reply_posture}",
                f"group_atmosphere={self.group_atmosphere}",
                f"address_mode={self.address_mode}",
                f"relevant_people_count={self.relevant_people_count}",
                f"persisted_turns={self.persisted_turns}",
                f"last_outcome={_xml_escape(self.last_outcome)}",
                f"last_user_message={_xml_escape(self.last_user_message)}",
                f"last_reply_summary={_xml_escape(self.last_reply_summary)}",
                f"reason={self.reason}",
                "</dialogue_state>",
            )
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def plan_chat_dialogue(
    *,
    message_text: str,
    intent: IntentClassification | None = None,
    has_images: bool = False,
    has_reply: bool = False,
) -> ChatDialoguePlan:
    normalized = normalize_message_text(message_text or "")
    chat_subkind = getattr(intent, "chat_subkind", "general_chat") if intent else ""
    target_hint = normalize_message_text(
        getattr(intent, "chat_target_hint", "") if intent else ""
    )

    if chat_subkind == "recap":
        return ChatDialoguePlan(
            kind="recap",
            confidence=0.9,
            need_memory_lookup=True,
            style="concise",
            reason="intent_recap",
        )
    if chat_subkind == "identity_query":
        return ChatDialoguePlan(
            kind="identity_query",
            confidence=0.86,
            target_hint=target_hint,
            need_memory_lookup=True,
            style="concise",
            reason="intent_identity_query",
        )
    if chat_subkind == "memory_confirm":
        return ChatDialoguePlan(
            kind="memory_update",
            confidence=0.84,
            target_hint=target_hint,
            need_memory_lookup=True,
            style="concise",
            reason="intent_memory_confirm",
        )
    if chat_subkind == "explain_context":
        return ChatDialoguePlan(
            kind="explain_context",
            confidence=0.84,
            need_grounding=True,
            need_memory_lookup=True,
            need_rewrite_check=True,
            style="detailed",
            reason="intent_explain_context",
        )

    if not normalized:
        return ChatDialoguePlan(kind="casual_chat", confidence=0.5, reason="empty")
    if contains_any(normalized, _EMOTION_HINTS):
        return ChatDialoguePlan(
            kind="emotional_support",
            confidence=0.78,
            need_memory_lookup=True,
            style="empathetic",
            reason="emotion_hint",
        )
    if contains_any(normalized, _MEMORY_UPDATE_HINTS):
        return ChatDialoguePlan(
            kind="memory_update",
            confidence=0.75,
            need_memory_lookup=True,
            style="concise",
            reason="memory_hint",
        )
    if contains_any(normalized, _CASUAL_HINTS):
        return ChatDialoguePlan(
            kind="casual_chat",
            confidence=0.8,
            style="concise",
            reason="casual_hint",
        )
    if (
        has_images
        or has_reply
        or len(normalized) >= 56
        or contains_any(normalized, _COMPLEX_HINTS)
    ):
        return ChatDialoguePlan(
            kind="complex_reasoning",
            confidence=0.72,
            need_grounding=True,
            need_memory_lookup=True,
            need_rewrite_check=True,
            style="step_by_step",
            reason="complex_hint",
        )
    if contains_any(normalized, _FACTUAL_HINTS):
        return ChatDialoguePlan(
            kind="factual_qa",
            confidence=0.7,
            need_grounding=True,
            need_rewrite_check=True,
            style="detailed",
            reason="factual_hint",
        )

    return ChatDialoguePlan(kind="casual_chat", reason="general_chat")


def build_dialogue_state(
    *,
    message_text: str,
    plan: ChatDialoguePlan | None,
    intent: IntentClassification | None = None,
    scenario: str = "",
    has_images: bool = False,
    has_reply: bool = False,
    previous_state: DialogueState | None = None,
    is_group: bool = False,
    relevant_people_count: int = 0,
    thread_topic: str = "",
) -> DialogueState:
    normalized = normalize_message_text(message_text or "")
    kind = plan.kind if plan is not None else "casual_chat"
    style = plan.style if plan is not None else "concise"
    emotion = _detect_user_emotion(normalized)
    purpose = _dialogue_purpose(kind, scenario=scenario)
    continuity = _dialogue_continuity(
        normalized,
        previous_state=previous_state,
        has_reply=has_reply,
        thread_topic=thread_topic,
    )
    response_length = _response_length(
        kind,
        style=style,
        text=normalized,
        is_group=is_group,
        continuity=continuity,
    )
    tone = _dialogue_tone(
        kind,
        emotion=emotion,
        scenario=scenario,
        text=normalized,
    )
    need_followup = _need_followup(
        normalized,
        kind=kind,
        intent=intent,
        has_images=has_images,
        has_reply=has_reply,
    )
    topic_hint = _topic_hint(
        normalized,
        previous_state=previous_state,
        thread_topic=thread_topic,
    )
    return DialogueState(
        tone=tone,
        user_emotion=emotion,
        dialogue_purpose=purpose,
        need_followup=need_followup,
        response_length=response_length,
        topic_hint=topic_hint,
        continuity=continuity,
        group_reply_policy=_group_reply_policy(
            kind=kind,
            purpose=purpose,
            is_group=is_group,
            emotion=emotion,
            response_length=response_length,
        ),
        reply_posture=_reply_posture(
            kind=kind,
            purpose=purpose,
            emotion=emotion,
            need_followup=need_followup,
            continuity=continuity,
        ),
        group_atmosphere=_group_atmosphere(
            normalized,
            is_group=is_group,
            emotion=emotion,
            relevant_people_count=relevant_people_count,
        ),
        address_mode=_address_mode(
            is_group=is_group,
            has_reply=has_reply,
            relevant_people_count=relevant_people_count,
            purpose=purpose,
        ),
        relevant_people_count=max(int(relevant_people_count or 0), 0),
        persisted_turns=previous_state.persisted_turns
        if previous_state is not None
        else 0,
        last_outcome=previous_state.last_outcome if previous_state is not None else "",
        last_user_message=previous_state.last_user_message
        if previous_state is not None
        else "",
        last_reply_summary=previous_state.last_reply_summary
        if previous_state is not None
        else "",
        reason=(
            f"{getattr(plan, 'reason', 'no_plan')}:{emotion}:{purpose}:" f"{continuity}"
        ),
    )


def load_dialogue_state(
    *,
    session_key: str,
    user_id: str,
    group_id: str | None,
    legacy_session_key: str = "",
) -> DialogueState | None:
    payload = read_json(_DIALOGUE_STATE_PATH, {})
    if not isinstance(payload, dict):
        return None
    key = _dialogue_state_key(
        session_key=session_key,
        user_id=user_id,
        group_id=group_id,
    )
    record = payload.get(key)
    if not isinstance(record, dict) and legacy_session_key:
        legacy_key = _dialogue_state_key(
            session_key=legacy_session_key,
            user_id=user_id,
            group_id=group_id,
        )
        legacy_record = payload.get(legacy_key)
        if isinstance(legacy_record, dict):
            record = legacy_record
            payload[key] = legacy_record
            payload.pop(legacy_key, None)
            write_json(_DIALOGUE_STATE_PATH, payload)
    if not isinstance(record, dict):
        return None
    if (
        time.time() - float(record.get("updated_ts", 0.0) or 0.0)
        > _DIALOGUE_STATE_TTL_SECONDS
    ):
        return None
    state_payload = record.get("state")
    if not isinstance(state_payload, dict):
        return None
    return dialogue_state_from_payload(state_payload)


def persist_dialogue_state(
    *,
    session_key: str,
    user_id: str,
    group_id: str | None,
    message_text: str,
    state: DialogueState,
    outcome: str = "chat_completed",
    reply_text: str = "",
) -> None:
    payload = read_json(_DIALOGUE_STATE_PATH, {})
    if not isinstance(payload, dict):
        payload = {}
    key = _dialogue_state_key(
        session_key=session_key,
        user_id=user_id,
        group_id=group_id,
    )
    now = time.time()
    payload[key] = {
        "updated_at": utc_now_iso(),
        "updated_ts": now,
        "session_key": normalize_message_text(session_key),
        "user_id": normalize_message_text(user_id),
        "group_id": normalize_message_text(group_id or ""),
        "last_message": normalize_message_text(message_text)[:240],
        "state": _state_record_for_persist(
            state,
            message_text=message_text,
            outcome=outcome,
            reply_text=reply_text,
        ),
    }
    _trim_dialogue_state_payload(payload, now=now)
    write_json(_DIALOGUE_STATE_PATH, payload)


def _state_record_for_persist(
    state: DialogueState,
    *,
    message_text: str,
    outcome: str,
    reply_text: str,
) -> dict[str, Any]:
    record = state.to_record()
    record["persisted_turns"] = max(int(state.persisted_turns or 0), 0) + 1
    record["last_outcome"] = normalize_message_text(outcome)
    record["last_user_message"] = normalize_message_text(message_text)[:240]
    record["last_reply_summary"] = normalize_message_text(reply_text)[:240]
    return record


def dialogue_state_from_payload(payload: dict[str, Any]) -> DialogueState | None:
    try:
        return DialogueState(
            tone=_literal_or_default(payload.get("tone"), "casual"),
            user_emotion=_literal_or_default(payload.get("user_emotion"), "neutral"),
            dialogue_purpose=_literal_or_default(
                payload.get("dialogue_purpose"),
                "chat",
            ),
            need_followup=bool(payload.get("need_followup")),
            response_length=_literal_or_default(
                payload.get("response_length"),
                "short",
            ),
            topic_hint=normalize_message_text(str(payload.get("topic_hint", "") or "")),
            continuity=_literal_or_default(payload.get("continuity"), "unknown"),
            group_reply_policy=_literal_or_default(
                payload.get("group_reply_policy"),
                "brief_react",
            ),
            reply_posture=_literal_or_default(payload.get("reply_posture"), "listen"),
            group_atmosphere=_literal_or_default(
                payload.get("group_atmosphere"),
                "unknown",
            ),
            address_mode=_literal_or_default(payload.get("address_mode"), "direct"),
            relevant_people_count=max(
                int(payload.get("relevant_people_count", 0) or 0),
                0,
            ),
            persisted_turns=max(int(payload.get("persisted_turns", 0) or 0), 0),
            last_outcome=normalize_message_text(
                str(payload.get("last_outcome", "") or "")
            ),
            last_user_message=normalize_message_text(
                str(payload.get("last_user_message", "") or "")
            )[:240],
            last_reply_summary=normalize_message_text(
                str(payload.get("last_reply_summary", "") or "")
            )[:240],
            reason=normalize_message_text(str(payload.get("reason", "") or "loaded")),
        )
    except Exception:
        return None


def _detect_user_emotion(text: str) -> UserEmotion:
    if not text:
        return "neutral"
    if contains_any(text, _ANGRY_HINTS):
        return "angry"
    if contains_any(text, _SAD_HINTS):
        return "sad"
    if contains_any(text, _EMOTION_HINTS):
        return "stressed"
    if contains_any(text, _CONFUSED_HINTS):
        return "confused"
    if contains_any(text, _HAPPY_HINTS):
        return "happy"
    return "neutral"


def _dialogue_purpose(kind: str, *, scenario: str) -> DialoguePurpose:
    if scenario == "superuser_agent":
        return "agent_task"
    if kind == "emotional_support":
        return "support"
    if kind == "identity_query":
        return "identity"
    if kind == "memory_update":
        return "memory_update"
    if kind in {"factual_qa", "explain_context", "recap"}:
        return "answer"
    if kind == "complex_reasoning":
        return "plan"
    return "chat"


def _response_length(
    kind: str,
    *,
    style: ChatDialogueStyle,
    text: str,
    is_group: bool = False,
    continuity: DialogueContinuity = "unknown",
) -> ResponseLength:
    if is_group and kind == "casual_chat" and continuity in {"same_topic", "followup"}:
        return "short"
    if style == "step_by_step" or kind == "complex_reasoning" or len(text) >= 80:
        return "long"
    if style in {"detailed", "empathetic"} or kind in {
        "factual_qa",
        "emotional_support",
        "explain_context",
    }:
        return "medium"
    return "short"


def _dialogue_continuity(
    text: str,
    *,
    previous_state: DialogueState | None,
    has_reply: bool,
    thread_topic: str,
) -> DialogueContinuity:
    if has_reply:
        return "followup"
    topic = _topic_hint(text, previous_state=previous_state, thread_topic=thread_topic)
    if not previous_state or not topic:
        return "new_topic" if topic else "unknown"
    if topic and topic == previous_state.topic_hint:
        return "same_topic"
    if (
        topic
        and previous_state.topic_hint
        and (topic in previous_state.topic_hint or previous_state.topic_hint in topic)
    ):
        return "followup"
    return "new_topic"


def _topic_hint(
    text: str,
    *,
    previous_state: DialogueState | None,
    thread_topic: str,
) -> str:
    topic = normalize_message_text(thread_topic)
    if topic:
        return topic[:48]
    tokens = [
        token
        for token in re_split_topic_tokens(text)
        if len(token) >= 2 and token not in {"真寻", "帮我", "一下", "这个", "那个"}
    ]
    if tokens:
        return "_".join(tokens[:3])[:48]
    return previous_state.topic_hint if previous_state is not None else ""


def _group_reply_policy(
    *,
    kind: str,
    purpose: DialoguePurpose,
    is_group: bool,
    emotion: UserEmotion,
    response_length: ResponseLength,
) -> GroupReplyPolicy:
    if purpose == "agent_task":
        return "agent"
    if not is_group:
        return "answer_directly" if response_length != "short" else "brief_react"
    if emotion in {"stressed", "sad", "angry"} or kind == "emotional_support":
        return "support"
    if response_length == "long" or kind in {"complex_reasoning", "factual_qa"}:
        return "structured"
    if purpose in {"answer", "identity", "memory_update"}:
        return "answer_directly"
    return "brief_react"


def _dialogue_tone(
    kind: str,
    *,
    emotion: UserEmotion,
    scenario: str,
    text: str,
) -> DialogueTone:
    if scenario == "superuser_agent":
        return "focused"
    if emotion in {"stressed", "sad", "angry"} or kind == "emotional_support":
        return "empathetic"
    if kind in {"complex_reasoning", "factual_qa", "explain_context"}:
        return "focused"
    if kind in {"identity_query", "memory_update"}:
        return "warm"
    if contains_any(text, _PLAYFUL_HINTS):
        return "playful"
    return "casual"


def _reply_posture(
    *,
    kind: str,
    purpose: DialoguePurpose,
    emotion: UserEmotion,
    need_followup: bool,
    continuity: DialogueContinuity,
) -> ReplyPosture:
    if need_followup:
        return "clarify"
    if emotion in {"stressed", "sad", "angry"} or purpose == "support":
        return "support"
    if kind == "complex_reasoning" or purpose == "plan":
        return "step"
    if purpose in {"answer", "identity", "memory_update"}:
        return "answer"
    if continuity in {"same_topic", "followup"}:
        return "banter"
    return "listen"


def _group_atmosphere(
    text: str,
    *,
    is_group: bool,
    emotion: UserEmotion,
    relevant_people_count: int,
) -> GroupAtmosphere:
    if not is_group:
        return "focused"
    if emotion in {"stressed", "sad", "angry"}:
        return "tense"
    if contains_any(text, _PLAYFUL_HINTS):
        return "playful"
    if relevant_people_count >= 3:
        return "active"
    if contains_any(text, _COMPLEX_HINTS) or contains_any(text, _FACTUAL_HINTS):
        return "focused"
    return "quiet"


def _address_mode(
    *,
    is_group: bool,
    has_reply: bool,
    relevant_people_count: int,
    purpose: DialoguePurpose,
) -> AddressMode:
    if not is_group:
        return "direct"
    if has_reply or relevant_people_count > 1:
        return "mention_aware"
    if purpose in {"chat", "support"}:
        return "soft"
    return "group_general"


def _need_followup(
    text: str,
    *,
    kind: str,
    intent: IntentClassification | None,
    has_images: bool,
    has_reply: bool,
) -> bool:
    intent_kind = normalize_message_text(str(getattr(intent, "kind", "") or ""))
    if intent_kind.endswith("need_arg") or intent_kind == "execute_need_arg":
        return True
    if kind == "identity_query" and not normalize_message_text(
        str(getattr(intent, "chat_target_hint", "") or "")
    ):
        return True
    if kind == "complex_reasoning" and len(text) < 12 and not (has_images or has_reply):
        return True
    if kind == "emotional_support":
        return contains_any(text, _FOLLOWUP_HINTS)
    return False


def re_split_topic_tokens(text: str) -> list[str]:
    import re

    return [
        token
        for token in re.split(r"[\s,，。.!！？?、/|;；：:（）()\[\]【】]+", text)
        if token
    ]


def _dialogue_state_key(
    *,
    session_key: str,
    user_id: str,
    group_id: str | None,
) -> str:
    return "|".join(
        (
            normalize_message_text(group_id or "private"),
            normalize_message_text(session_key),
            normalize_message_text(user_id),
        )
    )


def _trim_dialogue_state_payload(payload: dict[str, Any], *, now: float) -> None:
    for key, record in list(payload.items()):
        if not isinstance(record, dict):
            payload.pop(key, None)
            continue
        if (
            now - float(record.get("updated_ts", 0.0) or 0.0)
            > _DIALOGUE_STATE_TTL_SECONDS
        ):
            payload.pop(key, None)
    if len(payload) <= _DIALOGUE_STATE_MAX:
        return
    stale = sorted(
        payload,
        key=lambda key: float((payload.get(key) or {}).get("updated_ts", 0.0) or 0.0),
    )[: max(len(payload) - _DIALOGUE_STATE_MAX, 64)]
    for key in stale:
        payload.pop(key, None)


def _literal_or_default(value: Any, default: str) -> Any:
    text = normalize_message_text(str(value or ""))
    return text or default


def _xml_escape(value: str) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .strip()
    )


__all__ = [
    "AddressMode",
    "ChatDialogueKind",
    "ChatDialoguePlan",
    "ChatDialogueStyle",
    "DialoguePurpose",
    "DialogueState",
    "DialogueTone",
    "GroupAtmosphere",
    "GroupReplyPolicy",
    "ReplyPosture",
    "ResponseLength",
    "UserEmotion",
    "build_dialogue_state",
    "dialogue_state_from_payload",
    "load_dialogue_state",
    "persist_dialogue_state",
    "plan_chat_dialogue",
]
