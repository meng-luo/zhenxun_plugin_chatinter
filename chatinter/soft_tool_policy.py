"""Soft command exposure policy.

Soft commands are low-context command tools that are easy for a model to call
accidentally: they need no user-provided payload, no media/reply target, and no
required slots.  They stay discoverable, but become executable only when the
turn has an explicit tool intent or the gate selected them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .feedback import (
    get_command_feedback_profile,
    get_contextual_command_feedback_profile,
)
from .route_text import normalize_message_text, strip_invoke_prefix

_POLITE_REQUEST_TERMS = (
    "帮我",
    "给我",
    "请",
    "麻烦",
)
_GENERIC_EXECUTION_TERMS = (
    "执行",
    "调用",
    "使用",
    "打开",
    "启动",
)
_GENERIC_GENERATE_TERMS = (
    "来个",
    "来一",
    "做个",
    "做一",
    "生成",
    "制作",
    "发个",
    "发一",
    "发送",
)
_GENERIC_RANDOM_TERMS = (
    "抽个",
    "抽一",
    "抽张",
    "随机",
    "roll",
    "掷",
    "投",
    "选一个",
    "决定",
)
_GENERIC_QUERY_TERMS = (
    "查一下",
    "查下",
    "查询",
    "查看",
    "看看",
    "看一下",
    "看下",
    "搜一下",
    "搜下",
    "搜",
    "搜索",
    "找一下",
    "找下",
    "找",
    "介绍",
)
_GENERIC_PLAY_TERMS = (
    "播放",
    "点播",
    "点歌",
    "点一首",
    "点首",
    "播一首",
    "播首",
    "放一首",
    "放首",
    "听歌",
)
_GENERIC_STATUS_TERMS = (
    "信息",
    "状态",
    "记录",
    "统计",
    "排行",
    "签到",
    "打卡",
)
_GENERIC_HELP_TERMS = (
    "有哪些",
    "列表",
    "说明",
    "帮助",
    "版本",
    "文档",
    "地址",
)
_EXPLICIT_ACTION_TERMS = (
    *_POLITE_REQUEST_TERMS,
    *_GENERIC_EXECUTION_TERMS,
    *_GENERIC_GENERATE_TERMS,
    *_GENERIC_RANDOM_TERMS,
    *_GENERIC_QUERY_TERMS,
    *_GENERIC_PLAY_TERMS,
    *_GENERIC_STATUS_TERMS,
    *_GENERIC_HELP_TERMS,
)


@dataclass(frozen=True)
class ToolRequestStrength:
    """Generic user-request strength for a recalled capability."""

    score: float
    explicit: bool
    direct_mention: bool
    exact_command: bool
    matched_intents: tuple[str, ...] = ()
    reason: str = "none"


def is_soft_command_candidate(candidate: Any) -> bool:
    """Return whether a recalled command should be explicit-request-only."""

    schema = getattr(candidate, "schema", None)
    snapshot = getattr(candidate, "tool", None)
    return is_soft_command_schema(schema, snapshot=snapshot)


def is_soft_command_schema(schema: Any, *, snapshot: Any | None = None) -> bool:
    """Classify low-context tools without using plugin-specific names."""

    if snapshot is not None and hasattr(snapshot, "soft_tool"):
        return bool(getattr(snapshot, "soft_tool", False))
    if schema is None:
        return False
    if _has_required_user_input(schema, snapshot=snapshot):
        return False
    role = normalize_message_text(str(getattr(schema, "command_role", "") or ""))
    if role in {"catalog"}:
        return False
    return True


def soft_tool_policy_reason(candidate: Any) -> str:
    if not is_soft_command_candidate(candidate):
        return "normal_tool"
    schema = getattr(candidate, "schema", None)
    snapshot = getattr(candidate, "tool", None)
    if snapshot is not None:
        side_effect = normalize_message_text(str(getattr(snapshot, "side_effect", "") or ""))
        output_mode = normalize_message_text(str(getattr(snapshot, "output_mode", "") or ""))
        risk_level = normalize_message_text(str(getattr(snapshot, "risk_level", "") or ""))
        if side_effect or output_mode or risk_level:
            return (
                "explicit_request_only:"
                f"soft_tool:{output_mode or 'unknown'}:{side_effect or 'unknown'}:"
                f"{risk_level or 'unknown'}"
            )
    role = normalize_message_text(str(getattr(schema, "command_role", "") or ""))
    if role:
        return f"explicit_request_only:{role}:low_context"
    return "explicit_request_only:low_context"


def soft_tool_allowed_for_message(
    message_text: str,
    candidate: Any,
    *,
    selected_command_ids: set[str] | frozenset[str] | None = None,
) -> bool:
    """Return whether a soft candidate may be exposed as executable."""

    if not is_soft_command_candidate(candidate):
        return True
    schema = getattr(candidate, "schema", None)
    command_id = normalize_message_text(str(getattr(schema, "command_id", "") or ""))
    if command_id and selected_command_ids and command_id in selected_command_ids:
        return True
    return request_strength_for_candidate(message_text, candidate).explicit


def candidate_exposure_priority(
    message_text: str,
    candidate: Any,
    *,
    selected_command_ids: set[str] | frozenset[str] | None = None,
) -> tuple[int, float, float, float, float, int, str]:
    """Rank executable exposure without plugin-specific rules.

    Ordering implements:
    explicit intent > strong schema > high-confidence recall > soft-tool downrank.
    """

    schema = getattr(candidate, "schema", None)
    snapshot = getattr(candidate, "tool", None)
    command_id = normalize_message_text(str(getattr(schema, "command_id", "") or ""))
    selected = bool(
        command_id and selected_command_ids and command_id in selected_command_ids
    )
    request_strength = request_strength_for_candidate(message_text, candidate)
    explicit = selected or request_strength.explicit
    strong = _schema_strength(schema, snapshot=snapshot)
    score = float(getattr(candidate, "score", 0.0) or 0.0)
    soft = is_soft_command_candidate(candidate)
    risk_bonus = _risk_bonus(snapshot)
    profile = _candidate_feedback_profile(candidate)
    reliability = profile.exposure_score
    if (
        profile.low_reliability
        and not selected
        and not bool(getattr(candidate, "exact_protected", False))
    ):
        reliability -= 24.0
    if profile.high_reliability:
        reliability += 8.0
    if profile.param_failure_rate >= 0.35 and profile.total_count >= 3:
        reliability -= 6.0
    if profile.avg_latency_ms >= 12000 and profile.total_count >= 3:
        reliability -= 2.5
    return (
        1 if explicit else 0,
        request_strength.score,
        strong + risk_bonus,
        reliability,
        score,
        0 if soft else 1,
        command_id,
    )


def sort_exposure_candidates(
    message_text: str,
    candidates: list[Any],
    *,
    selected_command_ids: set[str] | frozenset[str] | None = None,
) -> list[Any]:
    return sorted(
        candidates,
        key=lambda candidate: candidate_exposure_priority(
            message_text,
            candidate,
            selected_command_ids=selected_command_ids,
        ),
        reverse=True,
    )


def has_explicit_soft_tool_request(message_text: str, candidate: Any) -> bool:
    """Check generic explicit request signals for a specific soft command."""

    return request_strength_for_candidate(message_text, candidate).explicit


def request_strength_for_candidate(
    message_text: str,
    candidate: Any,
) -> ToolRequestStrength:
    """Estimate whether the user is asking this capability to act.

    This intentionally uses generic action/intent categories plus candidate
    metadata.  It does not branch on plugin names.
    """

    normalized = normalize_message_text(message_text)
    stripped = normalize_message_text(strip_invoke_prefix(normalized))
    if not stripped:
        return ToolRequestStrength(score=0.0, explicit=False, direct_mention=False, exact_command=False)
    request_text = normalize_message_text(
        " ".join(
            part
            for part in (normalized, stripped)
            if part and part != stripped
        )
        or stripped
    )

    phrases = _candidate_phrases(candidate)
    exact_command = any(_matches_exact_command(stripped, phrase) for phrase in phrases) or bool(
        getattr(candidate, "exact_protected", False)
    )
    snapshot = getattr(candidate, "tool", None)
    intent_types = _candidate_intent_types(candidate)
    matched_intents = _matched_request_intents(
        request_text,
        intent_types,
        task_verbs=_candidate_task_verbs(candidate),
    )
    direct_mention = any(_phrase_mentions_command(request_text, phrase) for phrase in phrases)
    action_aligned_mention = bool(
        direct_mention
        and (
            matched_intents
            or any(
                _phrase_is_action_like(phrase)
                for phrase in phrases
                if phrase in stripped
            )
        )
    )
    score = 0.0
    reasons: list[str] = []
    if exact_command or bool(getattr(candidate, "exact_protected", False)):
        score += 4.0
        reasons.append("exact_command")
    if action_aligned_mention:
        score += 2.0
        reasons.append("capability_mentioned_with_action")
    elif direct_mention:
        score += 0.45
        reasons.append("capability_mentioned_without_action")
    if any(term in request_text for term in _POLITE_REQUEST_TERMS):
        score += 1.0
        reasons.append("request_marker")
    if matched_intents:
        score += 1.35 + 0.35 * min(len(matched_intents), 3)
        reasons.append("intent_match:" + ",".join(matched_intents))
    if _has_execution_phrase(request_text):
        score += 1.25
        reasons.append("execution_phrase")

    recall = float(getattr(candidate, "score", 0.0) or 0.0)
    features = getattr(candidate, "features", None)
    if recall >= 150.0:
        score += 1.0
        reasons.append("high_recall")
    elif recall >= 90.0:
        score += 0.45
        reasons.append("medium_recall")
    if float(getattr(features, "context_score", 0.0) or 0.0) > 0:
        score += 0.5
        reasons.append("context_signal")
    if (
        not matched_intents
        and _has_execution_phrase(request_text)
        and _candidate_is_visible_real_output(candidate)
        and recall >= 70.0
    ):
        score += 0.9
        reasons.append("visible_output_request")

    policy = normalize_message_text(
        str(getattr(snapshot, "execution_policy", "") or "")
    )
    soft = is_soft_command_candidate(candidate)
    generative = bool(getattr(snapshot, "generative", False))
    requires_real = bool(getattr(snapshot, "requires_real_result", True))
    threshold = 2.25
    if policy == "explicit_only" or soft or generative:
        threshold = 2.85
    elif policy == "strong_intent":
        threshold = 2.7
    elif policy == "confirmation_required":
        threshold = 3.2
    if requires_real and action_aligned_mention and matched_intents:
        threshold -= 0.35
    explicit = exact_command or score >= threshold
    return ToolRequestStrength(
        score=round(score, 3),
        explicit=explicit,
        direct_mention=direct_mention,
        exact_command=exact_command,
        matched_intents=matched_intents,
        reason=";".join(reasons) or "no_explicit_request",
    )


def filter_soft_candidates(
    message_text: str,
    candidates: list[Any],
    *,
    selected_command_ids: set[str] | frozenset[str] | None = None,
) -> list[Any]:
    """Remove implicit soft candidates from an executable exposure set."""

    filtered = [
        candidate
        for candidate in candidates
        if soft_tool_allowed_for_message(
            message_text,
            candidate,
            selected_command_ids=selected_command_ids,
        )
    ]
    return sort_exposure_candidates(
        message_text,
        filtered,
        selected_command_ids=selected_command_ids,
    )


def is_low_reliability_candidate(
    candidate: Any,
    *,
    message_text: str = "",
    selected_command_ids: set[str] | frozenset[str] | None = None,
) -> bool:
    schema = getattr(candidate, "schema", None)
    command_id = normalize_message_text(str(getattr(schema, "command_id", "") or ""))
    if command_id and selected_command_ids and command_id in selected_command_ids:
        return False
    if bool(getattr(candidate, "exact_protected", False)):
        return False
    strength = (
        request_strength_for_candidate(message_text, candidate)
        if normalize_message_text(message_text)
        else None
    )
    profile = _candidate_feedback_profile(candidate)
    if strength is not None and strength.explicit:
        features = getattr(candidate, "features", None)
        false_trigger_score = float(
            getattr(features, "false_trigger_score", 0.0) or 0.0
        )
        # Missing target/image/visible-output failures often mean the selected
        # capability was semantically correct but under-specified or poorly
        # observed.  For an explicit current request, keep the command
        # executable unless there is strong false-trigger evidence; reliability
        # can still down-rank it, but should not make it disappear.
        if (
            false_trigger_score > -14.0
            and profile.false_trigger_rate < 0.36
            and profile.false_trigger_count < 3.0
        ):
            return False
    return profile.low_reliability


def is_high_reliability_candidate(candidate: Any) -> bool:
    return _candidate_feedback_profile(candidate).high_reliability


def should_catalog_only_candidate(
    candidate: Any,
    *,
    message_text: str = "",
    selected_command_ids: set[str] | frozenset[str] | None = None,
) -> bool:
    """Return whether feedback should keep a candidate out of first exposure.

    Catalog-only is a routing policy, not deletion: the capability remains
    discoverable via ``retrieve_plugin_commands`` but stops consuming first-turn
    schema budget after enough bad history.
    """

    if is_low_reliability_candidate(
        candidate,
        message_text=message_text,
        selected_command_ids=selected_command_ids,
    ):
        return True
    schema = getattr(candidate, "schema", None)
    command_id = normalize_message_text(str(getattr(schema, "command_id", "") or ""))
    if command_id and selected_command_ids and command_id in selected_command_ids:
        return False
    if bool(getattr(candidate, "exact_protected", False)):
        return False
    context_profile = _candidate_context_feedback_profile(
        candidate,
        message_text=message_text,
    )
    if (
        context_profile.total_count >= 2
        and (
            context_profile.false_trigger_rate >= 0.35
            or context_profile.param_failure_rate >= 0.5
            or context_profile.success_rate < 0.35
        )
    ):
        return True
    if normalize_message_text(message_text):
        strength = request_strength_for_candidate(message_text, candidate)
        if strength.explicit:
            features = getattr(candidate, "features", None)
            false_trigger_score = float(
                getattr(features, "false_trigger_score", 0.0) or 0.0
            )
            if false_trigger_score > -8.0:
                return False
    profile = _candidate_feedback_profile(candidate)
    return (
        profile.total_count >= 5
        and (
            profile.param_failure_rate >= 0.45
            or profile.false_trigger_rate >= 0.22
            or (
                profile.avg_latency_ms >= 18000.0
                and profile.success_rate < 0.68
            )
        )
    )


def _schema_strength(schema: Any, *, snapshot: Any | None = None) -> float:
    if schema is None:
        return 0.0
    strength = 0.0
    slots = _schema_slots(schema)
    if any(bool(getattr(slot, "required", False)) for slot in slots):
        strength += 3.0
    elif slots:
        strength += 1.0
    requires = dict(getattr(schema, "requires", {}) or {})
    if any(bool(requires.get(key)) for key in ("text", "image", "reply", "at")):
        strength += 2.0
    payload_policy = normalize_message_text(str(getattr(schema, "payload_policy", "") or ""))
    if payload_policy not in {"", "none"}:
        strength += 1.5
    source = normalize_message_text(str(getattr(schema, "source", "") or ""))
    if source in {"explicit", "override", "metadata"}:
        strength += 1.0
    confidence = float(getattr(schema, "confidence", 0.0) or 0.0)
    strength += max(0.0, min(confidence, 1.0))
    if snapshot is not None and getattr(snapshot, "input_requirements", None):
        strength += 0.5
    if snapshot is not None:
        schema_quality = float(getattr(snapshot, "schema_quality", 0.5) or 0.5)
        reliability = float(getattr(snapshot, "reliability", 0.5) or 0.5)
        source_of_truth = normalize_message_text(
            str(getattr(snapshot, "source_of_truth", "") or "")
        )
        if schema_quality >= 0.72:
            strength += 0.35
        elif schema_quality < 0.35:
            strength -= 0.45
        if reliability >= 0.72:
            strength += 0.25
        elif reliability < 0.35:
            strength -= 0.35
        if source_of_truth in {"bot_state", "external_service", "local_state"}:
            strength += 0.35
        elif source_of_truth == "model_knowledge":
            strength -= 0.45
    return strength


def _risk_bonus(snapshot: Any | None) -> float:
    if snapshot is None:
        return 0.0
    # Higher risk/capability tools are not "more desirable"; the bonus only keeps
    # concrete external-action tools above vague soft text tools when already
    # recalled for an explicit user request.
    output_mode = normalize_message_text(str(getattr(snapshot, "output_mode", "") or ""))
    side_effect = normalize_message_text(str(getattr(snapshot, "side_effect", "") or ""))
    bonus = 0.0
    if output_mode in {"image", "file", "action"}:
        bonus += 1.0
    elif output_mode == "plugin_output":
        bonus += 0.45
    if side_effect in {"query", "send", "mutate"}:
        bonus += 0.5
    source_of_truth = normalize_message_text(
        str(getattr(snapshot, "source_of_truth", "") or "")
    )
    if source_of_truth in {"bot_state", "external_service", "local_state"}:
        bonus += 0.45
    elif source_of_truth == "model_knowledge":
        bonus -= 0.35
    if bool(getattr(snapshot, "requires_real_result", False)):
        bonus += 0.4
    if bool(getattr(snapshot, "generative", False)):
        bonus += 0.25
    return bonus


def _candidate_is_visible_real_output(candidate: Any) -> bool:
    snapshot = getattr(candidate, "tool", None)
    if snapshot is None:
        return False
    output_mode = normalize_message_text(str(getattr(snapshot, "output_mode", "") or ""))
    side_effect = normalize_message_text(str(getattr(snapshot, "side_effect", "") or ""))
    return (
        output_mode in {"image", "file", "plugin_output", "action"}
        or side_effect in {"query", "send", "mutate"}
        or bool(getattr(snapshot, "requires_real_tool", False))
    )


def _candidate_intent_types(candidate: Any) -> tuple[str, ...]:
    snapshot = getattr(candidate, "tool", None)
    values = list(getattr(snapshot, "intent_types", []) or [])
    if not values:
        role = normalize_message_text(
            str(getattr(getattr(candidate, "schema", None), "command_role", "") or "")
        )
        if role:
            values.append(role)
    result: list[str] = []
    for value in values:
        text = normalize_message_text(str(value or "")).lower()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _candidate_task_verbs(candidate: Any) -> tuple[str, ...]:
    snapshot = getattr(candidate, "tool", None)
    values = list(getattr(snapshot, "task_verbs", []) or [])
    result: list[str] = []
    for value in values:
        text = normalize_message_text(str(value or ""))
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _matched_request_intents(
    text: str,
    intent_types: tuple[str, ...],
    *,
    task_verbs: tuple[str, ...] = (),
) -> tuple[str, ...]:
    groups = {
        "query": _GENERIC_QUERY_TERMS,
        "status": _GENERIC_STATUS_TERMS + _GENERIC_QUERY_TERMS,
        "generate": _GENERIC_GENERATE_TERMS,
        "media": _GENERIC_GENERATE_TERMS,
        "random": _GENERIC_RANDOM_TERMS,
        "help": _GENERIC_HELP_TERMS,
        "transform": ("翻译", "转换", "转成", "识别", "解析"),
        "mutate": ("添加", "新增", "创建", "设置", "修改", "删除", "绑定"),
        "send": _GENERIC_GENERATE_TERMS + ("发给", "通知", "推送"),
        "execute": _GENERIC_EXECUTION_TERMS,
        "play": _GENERIC_PLAY_TERMS,
    }
    verb_groups = {
        "查询": _GENERIC_QUERY_TERMS,
        "统计": _GENERIC_STATUS_TERMS,
        "生成": _GENERIC_GENERATE_TERMS,
        "播放": _GENERIC_PLAY_TERMS,
        "翻译": ("翻译", "转换", "转成", "识别", "解析"),
        "帮助": _GENERIC_HELP_TERMS,
        "添加": ("添加", "新增", "创建", "设置", "绑定"),
        "删除": ("删除", "移除", "取消", "关闭", "退回", "解绑"),
    }
    result: list[str] = []
    for intent in intent_types:
        terms = groups.get(intent, ())
        if terms and any(term in text for term in terms):
            result.append(intent)
    for verb in task_verbs:
        terms = verb_groups.get(verb, ())
        if terms and any(term in text for term in terms):
            result.append(f"verb:{verb}")
    if not result and _has_execution_phrase(text):
        result.append("execute")
    return tuple(dict.fromkeys(result))


def _has_execution_phrase(text: str) -> bool:
    return any(term in text for term in _EXPLICIT_ACTION_TERMS)


def _phrase_is_action_like(phrase: str) -> bool:
    text = normalize_message_text(phrase)
    if not text:
        return False
    return any(term in text for term in _EXPLICIT_ACTION_TERMS)


def _has_required_user_input(schema: Any, *, snapshot: Any | None = None) -> bool:
    requires = dict(getattr(schema, "requires", {}) or {})
    if snapshot is not None:
        requires.update(dict(getattr(snapshot, "requires", {}) or {}))
    hard_requires = ("text", "image", "reply", "at", "private", "to_me")
    if any(bool(requires.get(key)) for key in hard_requires):
        return True

    payload_policy = normalize_message_text(
        str(getattr(schema, "payload_policy", "") or "")
    )
    if payload_policy not in {"", "none"}:
        return True

    if any(bool(getattr(slot, "required", False)) for slot in _schema_slots(schema)):
        return True

    target_requirement = normalize_message_text(
        str(getattr(schema, "target_requirement", "") or "")
    )
    if target_requirement == "required":
        return True
    if bool(getattr(schema, "allow_at", False)):
        return True
    return False


def _schema_slots(schema: Any) -> list[Any]:
    slots = getattr(schema, "slots", []) or []
    return list(slots) if isinstance(slots, list | tuple) else []


def _candidate_phrases(candidate: Any) -> tuple[str, ...]:
    schema = getattr(candidate, "schema", None)
    snapshot = getattr(candidate, "tool", None)
    values: list[Any] = [
        getattr(schema, "head", ""),
        *list(getattr(schema, "aliases", []) or []),
    ]
    if snapshot is not None:
        values.extend(list(getattr(snapshot, "examples", []) or [])[:4])
    phrases: list[str] = []
    for value in values:
        text = normalize_message_text(str(value or ""))
        if text and text not in phrases:
            phrases.append(text)
    return tuple(phrases)


def _candidate_feedback_profile(candidate: Any):
    schema = getattr(candidate, "schema", None)
    snapshot = getattr(candidate, "tool", None)
    return get_command_feedback_profile(
        command_id=str(getattr(schema, "command_id", "") or ""),
        plugin_module=str(getattr(snapshot, "plugin_module", "") or ""),
    )


def _candidate_context_feedback_profile(candidate: Any, *, message_text: str):
    schema = getattr(candidate, "schema", None)
    return get_contextual_command_feedback_profile(
        message_text=message_text,
        command_id=str(getattr(schema, "command_id", "") or ""),
    )


def _matches_exact_command(text: str, phrase: str) -> bool:
    normalized = normalize_message_text(text)
    target = normalize_message_text(phrase)
    if not normalized or not target:
        return False
    return normalized == target


def _phrase_mentions_command(text: str, phrase: str) -> bool:
    normalized = normalize_message_text(text)
    target = normalize_message_text(phrase)
    if not normalized or not target:
        return False
    if target in normalized:
        return True
    return False


__all__ = [
    "filter_soft_candidates",
    "has_explicit_soft_tool_request",
    "is_high_reliability_candidate",
    "is_low_reliability_candidate",
    "is_soft_command_candidate",
    "is_soft_command_schema",
    "request_strength_for_candidate",
    "should_catalog_only_candidate",
    "sort_exposure_candidates",
    "soft_tool_allowed_for_message",
    "soft_tool_policy_reason",
]
