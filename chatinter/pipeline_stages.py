"""
ChatInter - pipeline stages

实现消息处理流程，支持多模态输入（图片识别）。
使用 UniMessage 统一处理消息。
"""

from __future__ import annotations

import time
from typing import cast
import uuid

from nonebot.adapters import Bot, Event
from nonebot_plugin_alconna.uniseg import UniMessage
from nonebot_plugin_uninfo import Uninfo

from zhenxun.configs.config import BotConfig
from zhenxun.services import logger
from zhenxun.services.llm import LLMContentPart, LLMMessage
from zhenxun.utils.message import MessageUtils

from .addressee_resolver import AddresseeResult, resolve_addressee
from .chat_handler import (
    artifacts_from_send_observations,
    messages_summary_from_send_observations,
    normalize_ai_reply_text,
    replace_mention_ids_with_names,
    reroute_to_plugin_with_result,
)
from .chat_runtime import ChatRuntime
from .config import get_config_value
from .context_packer import DialogueContextPack
from .event_context import ChatInterEventContext, build_event_context
from .event_runtime import (
    apply_runtime_plugin_overrides,
    event_adapter_name,
    event_is_private,
    event_type_name,
)
from .execution_observer import (
    EXECUTION_REASON_CANCELLED,
    EXECUTION_REASON_ERROR,
    EXECUTION_REASON_INVALID_COMMAND,
    EXECUTION_REASON_REROUTE_FAILED,
    EXECUTION_REASON_ROUTE_SUCCESS,
    ExecutionObservation,
    record_execution_observation,
    start_execution_observation,
)
from .feedback import FeedbackKind, FeedbackStore
from .feedback_keys import (
    FEEDBACK_REASON_REROUTE_FAILED as _FEEDBACK_REASON_REROUTE_FAILED,
)
from .feedback_keys import (
    FEEDBACK_REASON_ROUTE_SUCCESS as _FEEDBACK_REASON_ROUTE_SUCCESS,
)
from .feedback_keys import (
    FEEDBACK_REASON_TARGET_REQUIRED as _FEEDBACK_REASON_TARGET_REQUIRED,
)
from .intent_classifier import classify_message_intent
from .intervention_router import InterventionDecision, decide_intervention
from .main_request import MainRequestResult, run_chatinter_main_request
from .memory import _chat_memory
from .memory_writer import MemoryWriteContext, MemoryWriter
from .middleware import TurnMiddlewareState
from .models.pydantic_models import CommandToolSnapshot, PluginKnowledgeBase
from .native_executor import NativeToolExecutionResult, NativeValidatedRoute
from .native_route import (
    NativeRouteDecision,
    NativeRouteReport,
)
from .person_registry import (
    PersonProfile,
    get_person_profile,
    resolve_relevant_people,
    upsert_seen_person,
)
from .plugin_registry import (
    PluginRegistry,
    PluginSelectionContext,
    get_user_plugin_knowledge,
)
from .response_quality_judge import schedule_shadow_quality_judge
from .route_execution import (
    build_invalid_route_observation,
    build_reply_image_segments_for_reroute,
    build_route_message_with_explicit_context,
    build_route_observation,
    build_target_modules,
    collect_target_capable_command_heads,
    extract_at_tokens,
    extract_image_tokens,
    extract_reply_sender_id,
    prepare_route_execution_plan,
    select_adapter_policy_for_message,
)
from .route_text import (
    ROUTE_ACTION_WORDS,
    contains_any,
    is_usage_question,
    normalize_message_text,
    should_force_knowledge_refresh,
)
from .skill_store import mark_skills_used, render_skill_hints, search_skills_semantic
from .target_context import (
    append_mention_context_xml,
    build_mention_name_map,
    build_mention_profiles,
    extract_pending_entities,
    needs_target_for_route,
    remember_target_resolution,
)
from .target_resolver import resolve_execution_target, resolve_pre_route_target
from .thread_resolver import ThreadContext, resolve_thread_context
from .thread_store import record_thread_message
from .trace import StageTrace
from .trajectory_eval import schedule_response_quality_eval
from .turn_frame import PipelineStage, TurnFrame
from .turn_metrics import (
    build_turn_metrics_snapshot,
    emit_turn_metrics,
    record_route_observation,
)
from .turn_output import ChannelName, TurnChannelEnvelope, log_turn_channels
from .turn_runtime import TurnBudgetController
from .utils.multimodal import extract_images_from_message
from .utils.unimsg_utils import remove_reply_segment, uni_to_text_with_tags

_KNOWLEDGE_REFRESH_COOLDOWN = 30.0
_last_knowledge_refresh_ts = 0.0
_GROUP_PLUGIN_SELECTOR_RULES = """
<chatinter_main_request>
你正在处理群聊消息。你可以自然聊天，也可以在合适时调用候选插件工具。
候选工具是真实 NoneBot 插件命令；调用后会返回结构化 Observation，
下一轮你可以基于 Observation 继续调用其他工具或给出最终回复。

当前请求会按候选规模选择暴露策略：候选少时直接暴露命令 schema；
候选多时先暴露轻量 capability card 让你选择能力，运行时会在下一步
补入选中命令的完整 schema。候选不足时，可以调用 retrieve_plugin_commands
检索更多能力；它只负责发现和注入 schema，不代表任务已经完成。

如果你决定执行插件能力，请调用对应命令工具获取真实结果；如果只是闲聊、
玩梗、讨论命令概念、候选工具不匹配或信息不足，可以直接自然回复。
如果命令工具当前是空参数/compact schema，调用它只是选择该能力，下一步
再根据完整 schema 填参数执行；不要臆造 schema 外参数。

多任务可以通过连续多个工具调用完成。task_text 只是当前工具调用负责内容
的简短标注；无法明确标注时可以传 null，不需要依赖本地拆句。
工具 Observation 会告诉你执行是否成功、发送了什么摘要、是否可重试以及
是否还有可继续处理的提示。根据这些结果自主决定下一步。
</chatinter_main_request>
""".strip()

_PRIVATE_CHAT_RULES = """
<chatinter_private_chat>
你正在处理私聊普通对话。当前场景不暴露插件工具；除非用户明确要求解释能力，
否则按自然聊天回复，不要声称已经执行插件或系统操作。
</chatinter_private_chat>
""".strip()

_SUPERUSER_AGENT_RULES = """
<chatinter_superuser_agent>
场景：超级用户私聊。你既可以自然聊天，也可以作为 Agent 管理项目和服务器。
闲聊、解释、计划讨论直接回复；只有需要真实检查或操作时才调用工具。
复杂任务会先生成 TaskGraph。每个任务都有 goal、required_tools、
acceptance_criteria、status、observations、artifacts；你应按任务图推进，
不要把未通过 observation/verifier 的任务说成完成。
运行时会做复杂度分流：轻任务走单工具快路径，不一定生成 TaskGraph；
复杂任务才启用 planner-executor-verifier。无论哪种路径，都以真实
Observation 作为完成依据。

工具选择优先级：
- 工具状态：tool_registry_status 查看可用 Agent 工具、分类和风险。
- 任务管理：todo_read / todo_write 维护短 Todo；复杂工程任务先列 Todo，
  逐项推进并用 observation 标记完成。
- 插件开发：plugin_dev_inspect / plugin_dev_scaffold / plugin_dev_write_file。
- 工程隔离：worktree_create / worktree_status / worktree_list / worktree_remove。
- 文件操作：read_file / list_dir / search_files / write_file / append_file /
  replace_in_file。
- 工程闭环协议：engineering_loop_start -> engineering_lsp_read -> semantic_patch_plan
  -> patch_prepare -> patch_apply -> engineering_eval_run -> engineering_eval_gate
  -> engineering_loop_complete。复杂代码任务优先走这条固定协议；不要跳过
  engineering_lsp_read 直接 patch。
- 工程失败恢复协议：engineering_eval_run 失败后先用 engineering_failure_diagnose
  固化诊断，再根据 allowed_next_tools 选择重读代码、二次 semantic_patch/patch、
  或 patch_rollback；不要原样重复失败测试，也不要绕过诊断直接二次 patch。
- 工程修改：patch_prepare 先生成 diff，patch_apply 应用，patch_rollback 回滚，
  patch_show 查看历史。
- 大输出读取：artifact_list / artifact_read 读取被压缩的长日志、diff、工具结果。
- 运行时事件：runtime_event_list / runtime_event_read 查看统一事件流，
  包括 AgentRun、工具进度、Observation、Artifact、Approval、Background Job、TaskGraph。
- AgentRun：agent_run_status 查询状态，agent_run_resume 恢复暂停任务，
  agent_run_cancel 取消暂停/运行中的任务。
- 项目命令：python_exec / python_module / uv_command / git_command。
- 服务器维护：server_status / process_list / server_command。
- 长任务：background_task_start，之后用 background_observation_wait /
  background_observation_list / background_task_status 查看 ObservationEvent，
  必要时 background_task_cancel。
- 工程验收：engineering_eval_plan 建立“读代码 -> 改代码 -> 跑测试 -> 可回滚”
  的 eval，engineering_eval_run 记录测试结果。
- 审计和确认：list_pending_approvals / approve_pending_action /
  reject_pending_action / revoke_pending_approval / audit_log_query。
- 只有无法归类的普通命令才用 shell_command。

安全规则：
所有敏感操作由 data/configs/chatinter_agent_permissions.yaml 的 allow/ask/deny 控制。
ask 会返回 approval_required 和 approval_id；先向用户说明待确认内容，用户确认后调用
approve_pending_action，再用 agent_run_resume 继续原任务；用户拒绝或取消则调用
reject_pending_action。deny 直接说明无法执行。
后台任务启动后当前 AgentRun 会暂停；稍后使用 agent_run_resume 恢复，并先查看
runtime_event_list 或 background_task_status 再继续后续步骤。
不要声称完成未实际执行的操作；工具返回 Observation 后再继续下一步或总结。
final 前会检查 TaskGraph 未完成项；如果被提示 task_graph_incomplete，
继续完成缺失任务，不能用口头总结替代执行证据。
Todo 是过程控制，不替代真实工具结果；Todo completed 需要对应 observation 支持。
涉及代码修改时，优先使用工程闭环协议；轻微单文件修改至少使用
patch_prepare -> patch_apply，便于用户预览和回滚。
当主工作区可能有用户改动，或任务涉及代码写入、多文件修改、插件开发、
长时间验证时，先用 worktree_create 创建隔离工作区；创建后
read/write/search、shell/git、patch、eval、plugin_dev
默认落到该 worktree。不要把隔离 worktree 的改动说成已经合入主工作区；总结时用
worktree_status 给出分支、路径和 diff 摘要。
如果用户明确要求直接改主工作区，可以不创建 worktree，但必须说明风险并遵守 approval。
patch_apply 会校验文件未在 prepare 后被外部修改；冲突时重新 read_file 后再 prepare。
工具结果出现 artifact_id 时，先根据摘要判断；需要原文再调用 artifact_read。
执行工程改动后，必须通过 engineering_eval_gate：gate 未通过时，按返回的
diagnosis / recovery_plan 选择重读代码、二次 semantic patch/patch_prepare，
或 patch_rollback。
不要在 gate blocked 时声称任务完成。
</chatinter_superuser_agent>
""".strip()


async def _persist_message_timeline(
    *,
    main_result: MainRequestResult,
    user_id: str,
    group_id: str | None,
    nickname: str,
    user_message,
    bot_id: str | None,
    event_context: ChatInterEventContext | None = None,
    thread_context: ThreadContext | None = None,
) -> None:
    timeline = [item.to_dict() for item in main_result.timeline]
    if not timeline:
        return
    user_text = uni_to_text_with_tags(user_message)
    response_summary = (
        main_result.output.memory_text or main_result.output.final_text or ""
    ).strip()
    dialog = await _chat_memory.add_timeline(
        user_id=user_id,
        group_id=group_id,
        nickname=nickname,
        user_message=user_message,
        response_summary=response_summary,
        timeline=timeline,
        bot_id=bot_id,
    )
    if event_context is not None and thread_context is not None:
        pending_entities = tuple(
            dict.fromkeys(
                (
                    *thread_context.pending_entities,
                    *extract_pending_entities(user_text),
                )
            )
        )
        await record_thread_message(
            thread_id=thread_context.thread_id,
            group_id=group_id,
            message_id=event_context.event_id,
            dialog_id=int(getattr(dialog, "id", 0) or 0) if dialog else None,
            user_id=user_id,
            participants=thread_context.participants,
            topic_key=thread_context.topic_key,
            source=thread_context.source,
            confidence=thread_context.confidence,
            message_text=user_text,
            pending_entities=pending_entities,
            entity_hints=thread_context.entity_hints,
        )
    await MemoryWriter.write_from_dialog(
        MemoryWriteContext(
            session_id=_chat_memory.get_session_id(user_id, group_id),
            user_id=str(user_id),
            group_id=str(group_id) if group_id else None,
            message_text=user_text,
            response_text=response_summary,
            source_dialog_id=int(getattr(dialog, "id", 0) or 0) if dialog else None,
            thread_id=thread_context.thread_id if thread_context is not None else None,
            topic_key=thread_context.topic_key if thread_context is not None else "",
            participants=thread_context.participants
            if thread_context is not None
            else (),
        )
    )


async def _build_dialogue_context_pack(
    *,
    event_context: ChatInterEventContext,
    mention_profiles: dict[str, dict[str, str]] | None = None,
) -> tuple[
    DialogueContextPack,
    PersonProfile | None,
    AddresseeResult,
    ThreadContext,
    InterventionDecision,
]:
    speaker_profile = await get_person_profile(
        user_id=event_context.user_id,
        group_id=event_context.group_id,
        fallback_name=event_context.nickname,
    )
    await upsert_seen_person(
        user_id=event_context.user_id,
        group_id=event_context.group_id,
        nickname=event_context.nickname,
    )
    addressee = await resolve_addressee(
        event_context=event_context,
        bot_names=(BotConfig.self_nickname or "",),
        mention_profiles=mention_profiles,
        speaker_profile=speaker_profile,
    )
    thread = await resolve_thread_context(
        event_context=event_context,
        addressee=addressee,
    )
    relevant_people = await resolve_relevant_people(
        group_id=event_context.group_id,
        message_text=event_context.message_text_with_tags,
        speaker_profile=speaker_profile,
        bot_id=event_context.bot_id,
        mention_user_ids=tuple(event_context.mentioned_user_ids),
        reply_sender_id=event_context.reply.sender_id
        if event_context.reply is not None
        else None,
        thread_user_ids=thread.participants,
        entity_hints=thread.pending_entities,
    )
    current_pending_entities = extract_pending_entities(
        event_context.message_text_with_tags
    )
    if current_pending_entities:
        thread = ThreadContext(
            thread_id=thread.thread_id,
            source=thread.source,
            confidence=thread.confidence,
            related_user_ids=thread.related_user_ids,
            topic_key=thread.topic_key,
            pending_entities=tuple(
                dict.fromkeys((*thread.pending_entities, *current_pending_entities))
            )[:8],
            entity_hints=tuple(dict.fromkeys((*thread.entity_hints, "identity_query")))[
                :8
            ],
        )
    route_signal = (
        contains_any(event_context.normalized_text, ROUTE_ACTION_WORDS)
        or bool(event_context.mentions)
        or bool(event_context.images)
    )
    intervention = decide_intervention(
        event_context=event_context,
        addressee=addressee,
        route_signal=route_signal,
    )
    pack = DialogueContextPack(
        event_context=event_context,
        speaker_profile=speaker_profile,
        addressee=addressee,
        thread=thread,
        relevant_people=relevant_people,
    )
    return pack, speaker_profile, addressee, thread, intervention


def _finish_trace(
    *,
    trace: StageTrace,
    user_id: str,
    group_id: str | None,
    message_preview: str,
    route_report: NativeRouteReport | None,
    budget_controller: TurnBudgetController | None = None,
) -> None:
    total_seconds = trace.finish()
    emit_turn_metrics(
        build_turn_metrics_snapshot(
            trace=trace,
            total_seconds=total_seconds,
            route_report=route_report,
            budget_controller=budget_controller,
        )
    )
    record_route_observation(
        user_id=user_id,
        group_id=group_id,
        message_preview=message_preview,
        trace_tags=dict(trace.tags),
        route_report=route_report,
    )


def _tag_execution_observation(
    trace: StageTrace,
    observation: ExecutionObservation,
) -> None:
    trace.update_tags(
        exec_action=observation.action,
        exec_success=int(observation.success),
        exec_reason=observation.reason,
        exec_latency_ms=observation.latency_ms,
    )


def _route_report_value(
    route_report: NativeRouteReport | None,
    name: str,
    default: object = 0,
):
    if route_report is None:
        return default
    return getattr(route_report, name, default)


def _route_report_observer_kwargs(
    route_report: NativeRouteReport | None,
) -> dict[str, object]:
    return {
        "candidate_total": _route_report_value(route_report, "candidate_total", 0),
        "tool_candidates": _route_report_value(route_report, "tool_candidates", 0),
    }


_STRUCTURAL_REROUTE_FAILURE_ERRORS = {
    "reroute timeout",
    "unresolved image placeholder",
}
_VISIBLE_OUTPUT_REQUIRED_MODES = {"image", "file"}
_VISIBLE_OUTPUT_REQUIRED_SIDE_EFFECTS = {"query", "send", "mutate"}


def _candidate_tool_snapshot(
    validated: NativeValidatedRoute,
) -> CommandToolSnapshot | None:
    candidate = getattr(validated, "candidate", None)
    snapshot = getattr(candidate, "tool", None) if candidate is not None else None
    return snapshot if isinstance(snapshot, CommandToolSnapshot) else None


def _route_requires_visible_output(validated: NativeValidatedRoute) -> bool:
    snapshot = _candidate_tool_snapshot(validated)
    if snapshot is None:
        return False
    if not bool(getattr(snapshot, "requires_real_tool", True)):
        return False
    role = normalize_message_text(str(getattr(snapshot, "command_role", "") or ""))
    if role in {"helper", "usage", "catalog"}:
        return False
    output_mode = normalize_message_text(
        str(getattr(snapshot, "output_mode", "") or "")
    )
    side_effect = normalize_message_text(
        str(getattr(snapshot, "side_effect", "") or "")
    )
    if output_mode == "plugin_output":
        return True
    return (
        output_mode in _VISIBLE_OUTPUT_REQUIRED_MODES
        or side_effect in _VISIBLE_OUTPUT_REQUIRED_SIDE_EFFECTS
    )


def _reroute_success_from_observation(
    *,
    validated: NativeValidatedRoute,
    reroute_success: bool,
    output_texts: list[str],
    output_artifacts: list[dict[str, object]],
) -> tuple[bool, str, bool]:
    if not reroute_success:
        return False, "", True
    if not _route_requires_visible_output(validated):
        return True, "", False
    if output_texts or output_artifacts:
        return True, "", False
    return False, "plugin_completed_without_visible_output", True


def _execution_failure_reason(
    *,
    reroute_result,
    observation_error: str,
) -> str:
    error = normalize_message_text(
        observation_error or getattr(reroute_result, "error", "") or ""
    )
    if error in _STRUCTURAL_REROUTE_FAILURE_ERRORS:
        return EXECUTION_REASON_REROUTE_FAILED
    if error == "plugin_completed_without_visible_output":
        return EXECUTION_REASON_INVALID_COMMAND
    if getattr(reroute_result, "timed_out", False):
        return EXECUTION_REASON_REROUTE_FAILED
    if error:
        return EXECUTION_REASON_ERROR
    return EXECUTION_REASON_REROUTE_FAILED


def _append_route_notice(context_xml: str, notice: str) -> str:
    text = normalize_message_text(notice)
    if not text:
        return context_xml
    section = f"<route_notice>{text}</route_notice>"
    if section in str(context_xml or ""):
        return context_xml
    return f"{context_xml}\n{section}".strip()


def _append_context_section(context_xml: str, section: str) -> str:
    text = str(section or "").strip()
    if not text:
        return context_xml
    if text in str(context_xml or ""):
        return context_xml
    return f"{context_xml}\n{text}".strip()


async def _inject_skill_hints(frame: TurnFrame) -> None:
    if frame.scenario != "superuser_agent" or not frame.allow_agent_tools:
        return
    try:
        matches = await search_skills_semantic(
            frame.route_message or frame.current_message, limit=3
        )
        hint_xml = render_skill_hints(matches)
        if not hint_xml:
            return
        frame.context_xml = _append_context_section(frame.context_xml, hint_xml)
        frame.update_tags(skill_hints=len(matches))
        mark_skills_used(matches)
    except Exception as exc:
        logger.debug(f"[ChatInter] skill hint injection skipped: {exc}")


async def _execute_native_tool_route(
    *,
    bot: Bot,
    event: Event,
    trace: StageTrace,
    validated: NativeValidatedRoute,
    knowledge_plugins,
    current_message: str,
    user_id: str,
    group_id: str | None,
    session_id: str | None,
    has_reply: bool,
    extra_image_segments: list | None,
    route_report: NativeRouteReport,
    mention_profiles: dict[str, dict[str, str]] | None = None,
) -> NativeToolExecutionResult:
    route_result = validated.route_result
    if route_result is None:
        return NativeToolExecutionResult(
            success=False,
            route_result=None,
            output=build_invalid_route_observation(
                decision=validated.decision,
                task_text=validated.task_frame.effective_text
                if validated.task_frame is not None
                else normalize_message_text(validated.decision.command or ""),
                ambient_message=current_message,
                error="工具调用没有生成有效插件路由。",
                retryable=True,
            ),
            reason="invalid route",
        )

    task_frame = validated.task_frame
    task_message = (
        task_frame.effective_text
        if task_frame is not None and task_frame.effective_text
        else normalize_message_text(route_result.decision.command or "")
    )
    target_resolution = await resolve_execution_target(
        group_id=group_id,
        bot=bot,
        bot_id=getattr(bot, "self_id", None),
        route_result=route_result,
        knowledge_plugins=knowledge_plugins,
        task_message=task_message,
        ambient_message=current_message,
        target_hint=task_frame.target_hint if task_frame is not None else "",
        mention_profiles=mention_profiles,
    )
    if target_resolution.blocked:
        return NativeToolExecutionResult(
            success=False,
            route_result=route_result,
            route_command=route_result.decision.command,
            output=build_route_observation(
                route_result=route_result,
                ok=False,
                route_command=route_result.decision.command,
                task_text=task_message,
                ambient_message=current_message,
                error=target_resolution.prompt,
                missing=["target"],
                retryable=True,
            ),
            display_text=target_resolution.prompt,
            reason=_FEEDBACK_REASON_TARGET_REQUIRED,
        )
    if target_resolution.resolved:
        task_message = target_resolution.message_text
    decision = route_result.decision
    target_modules = build_target_modules(route_result, knowledge_plugins)
    execution_plan = prepare_route_execution_plan(
        route_result=route_result,
        knowledge_plugins=knowledge_plugins,
        current_message=task_message,
        ambient_message=current_message,
        user_id=user_id,
    )
    if execution_plan.need_followup:
        return NativeToolExecutionResult(
            success=False,
            route_result=route_result,
            route_command=execution_plan.command or decision.command,
            output=build_route_observation(
                route_result=route_result,
                ok=False,
                route_command=execution_plan.command or decision.command,
                task_text=task_message,
                ambient_message=current_message,
                error=execution_plan.followup_message or "缺少必要参数或上下文。",
                missing=list(route_result.missing),
                retryable=True,
            ),
            display_text=execution_plan.followup_message or "",
            reason=execution_plan.feedback_reason or "",
        )

    route_command = execution_plan.command or decision.command
    execution_frame = start_execution_observation(
        action="execute",
        plugin_module=decision.plugin_module,
        plugin_name=decision.plugin_name,
        command_id=route_result.command_id,
        command=route_command,
        route_stage=route_result.stage,
        session_id=session_id,
        message_preview=task_message,
        selected_rank=route_result.selected_rank,
        selected_score=route_result.selected_score,
        selected_reason=route_result.selected_reason,
        **_route_report_observer_kwargs(route_report),
    )
    reroute_result = await reroute_to_plugin_with_result(
        bot,
        event,
        route_command,
        target_modules=target_modules,
        extra_image_segments=extra_image_segments,
        trace_id=f"ci-{uuid.uuid4().hex}",
        wait=True,
        timeout=float(get_config_value("NATIVE_REROUTE_TIMEOUT", 10) or 10),
    )
    output_texts = messages_summary_from_send_observations(reroute_result.outputs)
    output_artifacts = artifacts_from_send_observations(
        reroute_result.outputs,
        trace_id=reroute_result.trace_id,
    )
    observed_success, observation_error, retryable = _reroute_success_from_observation(
        validated=validated,
        reroute_success=reroute_result.success,
        output_texts=output_texts,
        output_artifacts=output_artifacts,
    )
    if observed_success:
        observation = execution_frame.finish(
            success=True,
            reason=EXECUTION_REASON_ROUTE_SUCCESS,
        )
        feedback_reason = _FEEDBACK_REASON_ROUTE_SUCCESS
    else:
        failure_reason = _execution_failure_reason(
            reroute_result=reroute_result,
            observation_error=observation_error,
        )
        observation = execution_frame.finish(
            success=False,
            reason=failure_reason,
        )
        feedback_reason = (
            _FEEDBACK_REASON_REROUTE_FAILED
            if failure_reason == EXECUTION_REASON_REROUTE_FAILED
            else failure_reason
        )
    _tag_execution_observation(trace, observation)
    await FeedbackStore.record_plugin_outcome(
        session_id=session_id,
        message_text=task_message,
        route_result=route_result,
        modules=target_modules,
        route_command=route_command,
        success=observed_success,
        reason=feedback_reason,
    )

    error_text = observation_error or reroute_result.error or ""
    payload = build_route_observation(
        route_result=route_result,
        ok=observed_success,
        route_command=route_command,
        messages_sent=output_texts,
        artifacts=output_artifacts,
        task_text=task_message,
        ambient_message=current_message,
        trace_id=reroute_result.trace_id,
        error=error_text,
        retryable=bool(retryable or reroute_result.timed_out or reroute_result.error),
    )
    display_text = (
        "???????????????"
        if observed_success and (output_texts or output_artifacts)
        else "??????"
        if observed_success
        else error_text or "???????"
    )
    return NativeToolExecutionResult(
        success=observed_success,
        route_result=route_result,
        route_command=route_command,
        output=payload,
        display_text=display_text,
        reason=observation.reason,
    )


async def stage_identity(
    *,
    frame: TurnFrame,
    event: Event,
    middleware_state: TurnMiddlewareState,
    middleware,
) -> None:
    await middleware.dispatch("pre_gate", middleware_state)
    FeedbackStore.inspect_user_followup(
        session_id=frame.session_key,
        message_text=frame.raw_message,
    )
    await apply_runtime_plugin_overrides(
        event=event,
        session_key=frame.session_key,
        group_id=frame.group_id,
    )
    frame.stage(PipelineStage.IDENTITY)


async def stage_event_context(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
    session: Uninfo,
    message=None,
    cached_plain_text: str | None = None,
) -> None:
    try:
        event_message = event.get_message()
    except Exception:
        event_message = None
    frame.event_message = event_message

    uni_msg = None
    if message:
        try:
            uni_msg = UniMessage.of(message)
        except Exception:
            pass
    frame.uni_msg = uni_msg

    event_context = build_event_context(
        bot=bot,
        event=event,
        session=session,
        raw_message=frame.raw_message,
        nickname=frame.nickname,
        event_message=event_message,
        uni_msg=uni_msg,
        cached_plain_text=cached_plain_text,
    )
    frame.event_context = event_context
    if event_context.turn_messages:
        frame.update_tags(
            turn_messages=len(event_context.turn_messages),
            turn_priority=event_context.turn_priority,
            pending_updates=len(event_context.pending_human_updates),
        )
    mention_profiles = await build_mention_profiles(
        frame.group_id,
        event_context.message_text_with_tags,
        bot_id=frame.bot_id,
        bot=bot,
    )
    frame.mention_profiles = mention_profiles
    if frame.turn_messages:
        current_message = frame.raw_message.strip()
    elif frame.event_message is not None:
        current_message = uni_to_text_with_tags(frame.event_message)
    elif frame.uni_msg:
        current_msg = remove_reply_segment(frame.uni_msg)
        current_message = uni_to_text_with_tags(current_msg)
    elif cached_plain_text:
        current_message = cached_plain_text.strip()
    else:
        current_message = frame.raw_message.strip()
    frame.current_message = current_message
    frame.stage(PipelineStage.EVENT_CONTEXT)


async def stage_thread_context(
    *,
    frame: TurnFrame,
) -> None:
    if frame.event_context is None:
        raise RuntimeError("missing event context")
    (
        dialogue_context_pack,
        _speaker_profile,
        addressee_result,
        thread_context,
        intervention_decision,
    ) = await _build_dialogue_context_pack(
        event_context=frame.event_context,
        mention_profiles=frame.mention_profiles,
    )
    frame.dialogue_context_pack = dialogue_context_pack
    frame.addressee_result = addressee_result
    frame.thread_context = thread_context
    frame.intervention_decision = intervention_decision
    frame.update_tags(
        addressee_source=addressee_result.source,
        addressee_confidence=f"{addressee_result.confidence:.2f}",
        thread_id=thread_context.thread_id,
        intervention=intervention_decision.action,
        intervention_reason=intervention_decision.reason,
    )
    frame.stage(PipelineStage.THREAD_CONTEXT)


async def stage_memory(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
) -> None:
    profile = ChatRuntime.attach_profile(frame)
    if profile is not None:
        frame.chat_runtime_profile = profile
        frame.dialogue_plan = profile.dialogue_plan
        frame.dialogue_state = profile.dialogue_state
        frame.previous_dialogue_state = profile.previous_state
    dialogue_state = frame.dialogue_state
    memory_message = (
        frame.raw_message if frame.turn_messages else frame.uni_msg or frame.raw_message
    )
    memory_dialogue_state = ChatRuntime.memory_dialogue_state(frame)
    (
        chat_system_prompt,
        context_xml,
        reply_images_data,
        history_messages,
    ) = await _chat_memory.build_full_context(
        frame.user_id,
        frame.group_id,
        frame.nickname,
        memory_message,
        bot,
        frame.bot_id,
        event,
        frame.dialogue_context_pack,
        memory_dialogue_state,
        frame.scenario,
    )
    frame.chat_memory_layered = getattr(
        frame.dialogue_context_pack,
        "layered_memory",
        None,
    )
    frame.dialogue_state = dialogue_state
    frame.set_context(
        system_prompt=chat_system_prompt,
        context_xml=context_xml,
        reply_images_data=reply_images_data,
        history_messages=history_messages,
    )
    frame.stage(PipelineStage.MEMORY)


async def stage_dialogue_state(*, frame: TurnFrame) -> None:
    profile = ChatRuntime.attach_profile(frame)
    if profile is not None:
        frame.chat_runtime_profile = profile
        frame.dialogue_plan = profile.dialogue_plan
        frame.dialogue_state = profile.dialogue_state
        frame.previous_dialogue_state = profile.previous_state
        frame.update_tags(
            dialogue_tone=profile.dialogue_state.tone,
            dialogue_emotion=profile.dialogue_state.user_emotion,
            dialogue_purpose=profile.dialogue_state.dialogue_purpose,
            reply_posture=profile.dialogue_state.reply_posture,
            group_atmosphere=profile.dialogue_state.group_atmosphere,
        )
    frame.stage(PipelineStage.DIALOGUE_STATE)


async def _prepare_current_message_context(
    *,
    frame: TurnFrame,
    middleware_state: TurnMiddlewareState,
    middleware,
    cached_plain_text: str | None = None,
) -> None:
    if frame.turn_messages:
        current_message = frame.raw_message.strip()
    elif frame.event_message is not None:
        current_message = uni_to_text_with_tags(frame.event_message)
    elif frame.uni_msg:
        current_msg = remove_reply_segment(frame.uni_msg)
        current_message = uni_to_text_with_tags(current_msg)
    elif cached_plain_text:
        current_message = cached_plain_text.strip()
    else:
        current_message = frame.raw_message.strip()
    frame.current_message = current_message

    frame.sync_to_middleware(
        middleware_state,
        phase="intent_routing",
    )
    await middleware.dispatch("before_intent", middleware_state)
    frame.apply_prompt_state(middleware_state)
    frame.stage(PipelineStage.INTENT_BUDGET)


async def _prepare_capability_route_context(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
) -> None:
    knowledge_base = frame.knowledge_base
    if knowledge_base is None:
        raise RuntimeError("missing plugin knowledge base")

    command_heads = collect_target_capable_command_heads(knowledge_base)
    event_context = frame.event_context
    reply_sender_id = (
        event_context.reply.sender_id
        if event_context is not None and event_context.reply is not None
        else extract_reply_sender_id(event)
    )
    reply_image_count = len(frame.reply_images_data or [])
    frame.reply_sender_id = reply_sender_id
    frame.reply_image_count = reply_image_count
    frame.has_reply = bool(reply_sender_id) or reply_image_count > 0
    if reply_image_count > 0:
        logger.debug(f"Reply ?????? {reply_image_count} ?????????")
    frame.reply_image_segments_for_reroute = build_reply_image_segments_for_reroute(
        frame.reply_images_data
    )
    pre_route_target_policy = select_adapter_policy_for_message(
        frame.current_message,
        knowledge_base,
    )
    route_message_base = build_route_message_with_explicit_context(
        message_text=frame.current_message,
        user_id=frame.user_id,
        reply_image_count=reply_image_count,
        reply_sender_id=reply_sender_id,
        target_policy=pre_route_target_policy,
    )
    target_resolution = await resolve_pre_route_target(
        group_id=frame.group_id,
        bot=bot,
        original_message=frame.current_message,
        route_message=route_message_base,
        mention_profiles=frame.mention_profiles,
        target_policy=pre_route_target_policy,
        command_heads=command_heads,
    )
    route_message = target_resolution.message_text
    mention_profiles = target_resolution.mention_profiles
    fuzzy_prompt = target_resolution.prompt
    frame.set_tag("target_resolution", target_resolution.status)
    frame.route_message = route_message
    frame.mention_profiles = mention_profiles
    frame.mention_name_map = build_mention_name_map(mention_profiles)
    if frame.mention_name_map or mention_profiles:
        frame.context_xml = append_mention_context_xml(
            frame.context_xml,
            frame.mention_name_map,
            mention_profiles,
        )
        logger.debug(
            "???@????: "
            + ", ".join(
                (
                    f"{mapped_user_id}->{profile.get('display_name')}"
                    + (f"(uid:{profile.get('uid')})" if profile.get("uid") else "")
                )
                for mapped_user_id, profile in mention_profiles.items()
            )
        )
    if fuzzy_prompt:
        frame.set_tag("target_context", "ambiguous")
        frame.context_xml = _append_route_notice(frame.context_xml, fuzzy_prompt)
        logger.debug("???????????????????????" f"{fuzzy_prompt}")

    if needs_target_for_route(
        frame.current_message,
        route_message,
        target_policy=pre_route_target_policy,
    ):
        frame.set_tag("target_context", "required")
        target_notice = (
            pre_route_target_policy.target_missing_message
            or "需要明确目标后才能调用对应插件。"
        )
        frame.context_xml = _append_route_notice(frame.context_xml, target_notice)
        logger.debug(
            "?????????????????????"
            f"{pre_route_target_policy.target_missing_message or '-'}"
        )

    if route_message != frame.current_message:
        logger.debug(
            "ChatInter ????????"
            f"before='{frame.current_message}' -> after='{route_message}'"
        )
    frame.stage(PipelineStage.ROUTE_PREPARE)


async def _select_capability_route(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
    middleware_state: TurnMiddlewareState,
    middleware,
) -> None:
    global _last_knowledge_refresh_ts

    knowledge_base = frame.knowledge_base
    if knowledge_base is None:
        raise RuntimeError("missing plugin knowledge base")

    frame.sync_to_middleware(
        middleware_state,
        phase=PipelineStage.ROUTE_SELECTION.value,
        route_message=frame.route_message,
    )
    frame.stage(PipelineStage.ROUTE_SELECTION)
    await middleware.dispatch("before_route", middleware_state)
    route_message = middleware_state.route_message or frame.route_message
    frame.route_message = route_message
    selection_context = PluginSelectionContext(
        query=route_message,
        session_id=frame.session_key,
        user_id=frame.user_id,
        group_id=frame.group_id,
        is_superuser=frame.is_superuser,
        event_type=event_type_name(event),
        adapter=event_adapter_name(bot),
        is_private=event_is_private(event),
        has_image=bool(extract_image_tokens(route_message)),
        has_at=bool(extract_at_tokens(route_message)),
        has_reply=frame.has_reply,
        addressee_user_id=frame.addressee_result.target_user_id
        if frame.addressee_result
        else None,
        addressee_source=frame.addressee_result.source
        if frame.addressee_result
        else "",
        thread_id=frame.thread_context.thread_id if frame.thread_context else "",
        intervention_action=frame.intervention_decision.action
        if frame.intervention_decision
        else "",
    )
    frame.selection_context = selection_context
    knowledge_base = PluginRegistry.filter_knowledge_base(
        knowledge_base,
        selection_context=selection_context,
    )
    frame.knowledge_base = knowledge_base

    if should_force_knowledge_refresh(route_message, knowledge_base):
        now = time.monotonic()
        if now - _last_knowledge_refresh_ts >= _KNOWLEDGE_REFRESH_COOLDOWN:
            _last_knowledge_refresh_ts = now
            refreshed_knowledge = await get_user_plugin_knowledge(force_refresh=True)
            filtered_knowledge = PluginRegistry.filter_knowledge_base(
                refreshed_knowledge,
                selection_context=selection_context,
            )
            if len(filtered_knowledge.plugins) > len(knowledge_base.plugins):
                knowledge_base = filtered_knowledge
                frame.knowledge_base = knowledge_base
                logger.info(
                    "???????????????????????" f"{len(knowledge_base.plugins)} ???"
                )

    command_tools = (
        PluginRegistry.build_command_tool_snapshots(
            knowledge_base,
            selection_context=selection_context,
        )
        if frame.allow_plugin_tools
        else []
    )
    frame.command_tools = command_tools
    frame.chat_tool_exposure_state = (
        "plugin_tools_exposed" if frame.allow_plugin_tools and command_tools else "none"
    )

    intent_profile = classify_message_intent(route_message, knowledge_base)
    frame.intent_profile = intent_profile
    frame.update_tags(
        intent_kind=intent_profile.kind,
        intent_reason=intent_profile.reason,
    )
    logger.debug(
        "ChatInter intent classify: "
        f"kind={intent_profile.kind} "
        f"reason={intent_profile.reason} "
        f"explicit={intent_profile.explicit_command} "
        f"command={intent_profile.command_head or '-'} "
        f"chat_subkind={getattr(intent_profile, 'chat_subkind', 'general_chat')} "
        f"confidence={intent_profile.confidence:.2f}"
    )
    middleware_state.intent = intent_profile
    middleware_state.route_message = route_message
    middleware_state.metadata = {
        "phase": "after_intent",
        "intent_kind": intent_profile.kind,
        "intent_reason": intent_profile.reason,
    }
    await middleware.dispatch("after_intent", middleware_state)
    frame.apply_prompt_state(middleware_state)
    route_message = middleware_state.route_message or route_message
    frame.route_message = route_message
    frame.stage(PipelineStage.INTENT)

    route_report = NativeRouteReport(helper_mode=is_usage_question(route_message))
    route_report.note_candidate_policy(
        reason="main_request_pending",
        limit=len(command_tools),
    )
    route_report.candidate_total = max(route_report.candidate_total, len(command_tools))
    frame.set_native_route(
        native_decision=NativeRouteDecision(
            action="chat",
            confidence=0.0,
            reason="main_request_pending",
        ),
        route_result=None,
        route_report=route_report,
    )
    frame.update_tags(
        route_reason=route_report.final_reason,
        route_candidates=route_report.candidate_total,
    )


async def stage_capability_hint(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
    middleware_state: TurnMiddlewareState,
    middleware,
    cached_plain_text: str | None = None,
) -> None:
    frame.knowledge_base = (
        await get_user_plugin_knowledge()
        if frame.allow_plugin_tools
        else _empty_plugin_knowledge_base()
    )
    frame.stage(PipelineStage.KNOWLEDGE)
    await _prepare_current_message_context(
        frame=frame,
        middleware_state=middleware_state,
        middleware=middleware,
        cached_plain_text=cached_plain_text,
    )
    if frame.allow_plugin_tools:
        await _prepare_capability_route_context(frame=frame, bot=bot, event=event)
        await _select_capability_route(
            frame=frame,
            bot=bot,
            event=event,
            middleware_state=middleware_state,
            middleware=middleware,
        )
    else:
        await _prepare_lightweight_chat_route(
            frame=frame,
            bot=bot,
            event=event,
            middleware_state=middleware_state,
            middleware=middleware,
        )
    await _inject_skill_hints(frame)
    frame.stage(PipelineStage.CAPABILITY_HINT)


def _empty_plugin_knowledge_base() -> PluginKnowledgeBase:
    return PluginKnowledgeBase(plugins=[], user_role="普通用户")


async def _prepare_lightweight_chat_route(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
    middleware_state: TurnMiddlewareState,
    middleware,
) -> None:
    event_context = frame.event_context
    reply_sender_id = (
        event_context.reply.sender_id
        if event_context is not None and event_context.reply is not None
        else extract_reply_sender_id(event)
    )
    reply_image_count = len(frame.reply_images_data or [])
    frame.reply_sender_id = reply_sender_id
    frame.reply_image_count = reply_image_count
    frame.has_reply = bool(reply_sender_id) or reply_image_count > 0
    frame.reply_image_segments_for_reroute = build_reply_image_segments_for_reroute(
        frame.reply_images_data
    )
    frame.route_message = frame.current_message
    frame.sync_to_middleware(
        middleware_state,
        phase=PipelineStage.ROUTE_SELECTION.value,
        route_message=frame.route_message,
    )
    frame.stage(PipelineStage.ROUTE_SELECTION)
    await middleware.dispatch("before_route", middleware_state)
    route_message = middleware_state.route_message or frame.route_message
    frame.route_message = route_message
    frame.selection_context = PluginSelectionContext(
        query=route_message,
        session_id=frame.session_key,
        user_id=frame.user_id,
        group_id=frame.group_id,
        is_superuser=frame.is_superuser,
        event_type=event_type_name(event),
        adapter=event_adapter_name(bot),
        is_private=event_is_private(event),
        has_image=bool(extract_image_tokens(route_message)),
        has_at=bool(extract_at_tokens(route_message)),
        has_reply=frame.has_reply,
        thread_id=frame.thread_context.thread_id if frame.thread_context else "",
        intervention_action=frame.intervention_decision.action
        if frame.intervention_decision
        else "",
    )
    frame.command_tools = []
    frame.chat_tool_exposure_state = "none"
    intent_profile = classify_message_intent(
        route_message,
        frame.knowledge_base or _empty_plugin_knowledge_base(),
    )
    frame.intent_profile = intent_profile
    frame.update_tags(
        intent_kind=intent_profile.kind,
        intent_reason=intent_profile.reason,
    )
    middleware_state.intent = intent_profile
    middleware_state.route_message = route_message
    middleware_state.metadata = {
        "phase": "after_intent",
        "intent_kind": intent_profile.kind,
        "intent_reason": intent_profile.reason,
    }
    await middleware.dispatch("after_intent", middleware_state)
    frame.apply_prompt_state(middleware_state)
    route_message = middleware_state.route_message or route_message
    frame.route_message = route_message
    frame.stage(PipelineStage.INTENT)
    route_report = NativeRouteReport(helper_mode=is_usage_question(frame.route_message))
    frame.set_native_route(
        native_decision=NativeRouteDecision(
            action="chat",
            confidence=0.0,
            reason="chat_only_no_plugin_tools",
        ),
        route_result=None,
        route_report=route_report,
    )


def _build_system_prompt(frame: TurnFrame) -> str:
    base_prompt = frame.system_prompt
    parts = [normalize_message_text(base_prompt)]
    chat_prompt = ChatRuntime.build_prompt_context(
        frame,
        base_context_xml=frame.enriched_context_xml or frame.context_xml,
    )
    parts.extend(chat_prompt.system_fragments)
    if frame.scenario == "group_plugin_selector":
        parts.append(_GROUP_PLUGIN_SELECTOR_RULES)
    elif frame.scenario == "superuser_agent":
        parts.append(_SUPERUSER_AGENT_RULES)
    else:
        parts.append(_PRIVATE_CHAT_RULES)
    return "\n\n".join(part for part in parts if part)


def _build_agent_messages(frame: TurnFrame) -> list[LLMMessage]:
    system_prompt = _build_system_prompt(frame)
    user_text = frame.route_message or frame.current_message
    turn_xml = _build_turn_queue_context(frame)
    if turn_xml:
        user_text = f"{turn_xml}\n\n{user_text}"
    if frame.context_xml:
        user_text = (
            f"{frame.context_xml}\n\n"
            f"<current_user_message>{user_text}</current_user_message>"
        )
    user_content: str | list[LLMContentPart]
    if frame.image_parts:
        user_content = [LLMContentPart.text_part(user_text), *frame.image_parts]
    else:
        user_content = user_text
    return [
        LLMMessage.system(system_prompt),
        *list(frame.history_messages or []),
        LLMMessage.user(user_content),
    ]


def _should_initial_command_exposure(frame: TurnFrame) -> bool:
    return bool(
        frame.allow_plugin_tools
        and frame.scenario == "group_plugin_selector"
        and frame.command_tools
    )


def _build_turn_queue_context(frame: TurnFrame) -> str:
    sections: list[str] = []
    if frame.turn_messages and len(frame.turn_messages) > 1:
        lines = "\n".join(
            f'<message index="{index}">{normalize_message_text(text)}</message>'
            for index, text in enumerate(frame.turn_messages, 1)
            if normalize_message_text(text)
        )
        if lines:
            sections.append(f"<merged_turn_messages>\n{lines}\n</merged_turn_messages>")
    if frame.pending_human_updates:
        lines = "\n".join(
            f'<update index="{index}">{normalize_message_text(text)}</update>'
            for index, text in enumerate(frame.pending_human_updates, 1)
            if normalize_message_text(text)
        )
        if lines:
            sections.append(
                "<pending_human_updates>\n" f"{lines}\n" "</pending_human_updates>"
            )
    return "\n".join(sections)


async def stage_current_user(
    *,
    frame: TurnFrame,
    message=None,
) -> None:
    source_for_media = (
        frame.event_message or frame.uni_msg or message or frame.raw_message
    )
    image_parts = await extract_images_from_message(source_for_media)
    frame.image_parts = image_parts
    if image_parts:
        logger.debug(f"??????? {len(image_parts)} ???")

    if frame.reply_images_data:
        from .utils.multimodal import _process_image_segment

        for img_seg in frame.reply_images_data:
            image_part = await _process_image_segment(img_seg)
            if image_part:
                image_parts.append(image_part)
        frame.image_parts = image_parts
        if frame.reply_images_data:
            logger.debug(f"?????? {len(frame.reply_images_data)} ???")
    frame.stage(PipelineStage.CURRENT_USER)
    frame.enriched_context_xml = frame.context_xml
    if frame.intent_profile is None:
        raise RuntimeError("missing intent profile for main request")
    profile = ChatRuntime.attach_profile(frame)
    if profile is not None:
        frame.chat_runtime_profile = profile
        frame.dialogue_plan = profile.dialogue_plan
        frame.dialogue_state = profile.dialogue_state
        frame.previous_dialogue_state = profile.previous_state
    dialogue_plan = frame.dialogue_plan
    dialogue_state = frame.dialogue_state
    if dialogue_plan is None or dialogue_state is None:
        raise RuntimeError("missing chat runtime profile")
    chat_prompt = ChatRuntime.build_prompt_context(
        frame,
        base_context_xml=frame.enriched_context_xml,
    )
    frame.enriched_context_xml = chat_prompt.context_xml
    if chat_prompt.tags:
        frame.update_tags(**chat_prompt.tags)
    frame.update_tags(
        chat_kind=dialogue_plan.kind,
        chat_style=dialogue_plan.style,
        chat_reason=dialogue_plan.reason,
        dialogue_tone=dialogue_state.tone,
        dialogue_emotion=dialogue_state.user_emotion,
        dialogue_purpose=dialogue_state.dialogue_purpose,
        reply_posture=dialogue_state.reply_posture,
        group_atmosphere=dialogue_state.group_atmosphere,
    )


async def stage_scratchpad(
    *,
    frame: TurnFrame,
    middleware_state: TurnMiddlewareState,
    middleware,
) -> None:
    if frame.dialogue_plan is None:
        raise RuntimeError("missing dialogue plan for main request")
    frame.context_xml = frame.enriched_context_xml
    frame.sync_to_middleware(
        middleware_state,
        phase=PipelineStage.SCRATCHPAD.value,
    )
    await middleware.dispatch("before_chat", middleware_state)
    frame.apply_prompt_state(middleware_state)
    frame.enriched_context_xml = frame.context_xml
    frame.agent_messages = _build_agent_messages(frame)
    frame.stage(PipelineStage.SCRATCHPAD)


async def stage_agent_run(
    *,
    frame: TurnFrame,
    bot: Bot,
    event: Event,
    middleware_state: TurnMiddlewareState,
    middleware,
) -> None:
    if frame.dialogue_plan is None:
        raise RuntimeError("missing dialogue plan for main request")

    knowledge_base = frame.knowledge_base
    if knowledge_base is None:
        raise RuntimeError("missing plugin knowledge base")
    if not frame.agent_messages:
        frame.agent_messages = _build_agent_messages(frame)
    frame.stage(PipelineStage.AGENT_RUN)

    async def _execute_native_route_callback(
        validated: NativeValidatedRoute,
        report: NativeRouteReport,
    ) -> NativeToolExecutionResult:
        return await _execute_native_tool_route(
            bot=bot,
            event=event,
            trace=frame.trace,
            validated=validated,
            knowledge_plugins=knowledge_base.plugins,
            current_message=frame.route_message or frame.current_message,
            user_id=frame.user_id,
            group_id=frame.group_id,
            session_id=frame.session_key,
            has_reply=frame.has_reply,
            extra_image_segments=frame.reply_image_segments_for_reroute,
            route_report=report,
            mention_profiles=frame.mention_profiles,
        )

    async def _route_completed_callback(main_result) -> None:
        frame.set_native_route(
            native_decision=main_result.decision,
            route_result=main_result.route_result,
            route_report=main_result.report,
        )
        frame.update_tags(
            native_action=main_result.decision.action,
            native_confidence=f"{main_result.decision.confidence:.2f}",
            native_reason=main_result.decision.reason or "",
            native_plugin=main_result.route_result.decision.plugin_module
            if main_result.route_result
            else "",
            native_command=main_result.route_result.decision.command
            if main_result.route_result
            else "",
            route_reason=main_result.report.final_reason,
            route_candidates=main_result.report.candidate_total,
            route_attempts=main_result.report.attempts,
            route_tool_candidates=main_result.report.tool_candidates,
            route_tool_choices=main_result.report.tool_choice_count,
        )
        middleware_state.metadata = {
            "phase": "route_completed",
            "native_action": main_result.decision.action,
            "route_reason": main_result.report.final_reason,
        }
        await middleware.dispatch("after_route", middleware_state)

    async def _reply_hook(reply_text: str) -> str:
        finalized_text = normalize_ai_reply_text(reply_text)
        finalized_text = replace_mention_ids_with_names(
            finalized_text,
            frame.mention_name_map,
        )
        middleware_state.response_text = finalized_text
        await middleware.dispatch("after_chat", middleware_state)
        finalized_text = normalize_ai_reply_text(
            middleware_state.response_text or finalized_text
        )
        return replace_mention_ids_with_names(finalized_text, frame.mention_name_map)

    async def _agent_progress_hook(progress_text: str) -> None:
        # P0-2 最小落地:超级用户长任务的步级进度回显。
        await MessageUtils.build_message(progress_text).send()

    main_result = await run_chatinter_main_request(
        frame.route_message or frame.current_message,
        knowledge_base,
        session_key=frame.session_key,
        budget_controller=frame.budget_controller,
        has_reply=frame.has_reply,
        command_tools=frame.command_tools,
        messages=frame.agent_messages,
        route_executor=_execute_native_route_callback,
        route_completed_hook=_route_completed_callback,
        reply_hook=_reply_hook,
        enable_plugin_tools=frame.allow_plugin_tools,
        initial_command_exposure=_should_initial_command_exposure(frame),
        enable_agent_tools=frame.allow_agent_tools,
        progress_hook=_agent_progress_hook if frame.allow_agent_tools else None,
    )
    frame.main_result = main_result
    envelope = TurnChannelEnvelope()
    frame.update_tags(
        path="main_request",
        outcome=main_result.output.outcome,
    )
    envelope.add(ChannelName.ANALYSIS, main_result.output.analysis)
    if main_result.output.should_send:
        reply_text = main_result.output.final_text
        if not normalize_message_text(reply_text):
            reply_text = "我暂时没想好怎么回答你。"
        quality_enabled = ChatRuntime.isolation_for_frame(frame).allow_quality_judge
        quality = ChatRuntime.judge_final_reply(
            frame=frame,
            final_text=reply_text,
        )
        frame.response_quality_result = quality
        if quality.reason:
            frame.update_tags(
                response_quality=quality.reason,
                response_quality_action=quality.action,
            )
        if not quality.ok:
            if quality.revised_text:
                reply_text = quality.revised_text
            elif quality.instruction:
                envelope.add(ChannelName.ANALYSIS, quality.instruction)
        if quality_enabled:
            schedule_shadow_quality_judge(
                final_text=reply_text,
                original_message=frame.current_message,
                scenario=frame.scenario,
                session_key=frame.session_key,
                trace_id=str(frame.trace.tags.get("message_id", "")),
                quality_result=quality,
            )
            schedule_response_quality_eval(
                quality_result=quality,
                final_text=reply_text,
                original_message=frame.current_message,
                scenario=frame.scenario,
                session_key=frame.session_key,
                trace_id=str(frame.trace.tags.get("message_id", "")),
            )
        envelope.add(ChannelName.FINAL, reply_text)
    frame.final_envelope = envelope


async def stage_persist(frame: TurnFrame) -> None:
    main_result = frame.main_result
    if main_result is None:
        raise RuntimeError("missing main request result")
    envelope = frame.final_envelope or TurnChannelEnvelope()
    if not main_result.output.should_send:
        log_turn_channels(envelope)
        await _persist_message_timeline(
            main_result=main_result,
            user_id=frame.user_id,
            group_id=frame.group_id,
            nickname=frame.nickname,
            user_message=frame.uni_msg or frame.current_message,
            bot_id=frame.bot_id,
            event_context=frame.event_context,
            thread_context=frame.thread_context,
        )
        frame.stage(PipelineStage.PERSIST)
        if frame.post_gate_callback is not None:
            await frame.post_gate_callback(phase="post_gate:main_request")
        _finish_trace(
            trace=frame.trace,
            user_id=frame.user_id,
            group_id=frame.group_id,
            message_preview=frame.current_message,
            route_report=frame.route_report,
            budget_controller=frame.budget_controller,
        )
        frame.turn_finished = True
        return

    log_turn_channels(envelope)
    chat_execution_frame = start_execution_observation(
        action="chat",
        route_stage="main_request",
        session_id=frame.session_key,
        message_preview=frame.current_message,
        **_route_report_observer_kwargs(frame.route_report),
    )
    frame.final_envelope = envelope
    frame.chat_execution_frame = chat_execution_frame
    await _persist_message_timeline(
        main_result=main_result,
        user_id=frame.user_id,
        group_id=frame.group_id,
        nickname=frame.nickname,
        user_message=frame.uni_msg or frame.current_message,
        bot_id=frame.bot_id,
        event_context=frame.event_context,
        thread_context=frame.thread_context,
    )
    frame.stage(PipelineStage.PERSIST)


async def stage_send(frame: TurnFrame) -> None:
    if frame.turn_finished:
        return
    main_result = frame.main_result
    if main_result is None:
        raise RuntimeError("missing main request result")
    envelope = frame.final_envelope
    if envelope is None:
        raise RuntimeError("missing final envelope")
    if not main_result.output.should_send:
        frame.turn_finished = True
        return
    await MessageUtils.build_message(envelope.final).send()
    frame.stage(PipelineStage.SEND)
    if frame.post_gate_callback is not None:
        await frame.post_gate_callback(
            response_text=envelope.final,
            phase="post_gate:main_request",
        )
    if main_result.output.record_chat_feedback:
        FeedbackStore.record_chat(
            session_id=frame.session_key,
            kind=cast(FeedbackKind, main_result.output.feedback_kind),
            message_text=frame.current_message,
            reply_text=envelope.final,
            weight=0.2,
        )
    _persist_plain_chat_dialogue_state(
        frame=frame,
        reply_text=envelope.final,
    )
    _tag_execution_observation(
        frame.trace,
        frame.chat_execution_frame.finish(
            success=True,
            reason=main_result.output.observation_reason,
        )
        if frame.chat_execution_frame is not None
        else record_execution_observation(
            action="chat",
            success=True,
            reason=main_result.output.observation_reason,
            session_id=frame.session_key,
            message_preview=frame.current_message,
        ),
    )
    _finish_trace(
        trace=frame.trace,
        user_id=frame.user_id,
        group_id=frame.group_id,
        message_preview=frame.current_message,
        route_report=frame.route_report,
        budget_controller=frame.budget_controller,
    )
    frame.turn_finished = True


def _persist_plain_chat_dialogue_state(
    *,
    frame: TurnFrame,
    reply_text: str,
) -> None:
    try:
        if ChatRuntime.persist_dialogue_state(frame=frame, reply_text=reply_text):
            frame.update_tags(dialogue_state_persisted="1")
    except Exception:
        return


async def handle_pipeline_cancelled(frame: TurnFrame) -> None:
    frame.update_tags(path="cancelled", outcome="cancelled")
    _tag_execution_observation(
        frame.trace,
        record_execution_observation(
            action="chat",
            success=False,
            reason=EXECUTION_REASON_CANCELLED,
            session_id=frame.session_key,
            message_preview=frame.current_message,
        ),
    )
    group_name = frame.group_id or "private"
    logger.debug(f"ChatInter 处理被取消: user={frame.user_id}, group={group_name}")
    if frame.post_gate_callback is not None:
        await frame.post_gate_callback(phase="post_gate:cancelled")
    _finish_trace(
        trace=frame.trace,
        user_id=frame.user_id,
        group_id=frame.group_id,
        message_preview=frame.current_message,
        route_report=frame.route_report,
        budget_controller=frame.budget_controller,
    )
    frame.turn_finished = True


async def handle_pipeline_error(frame: TurnFrame, error: Exception) -> None:
    frame.update_tags(path="error", outcome="error")
    _tag_execution_observation(
        frame.trace,
        record_execution_observation(
            action="chat",
            success=False,
            reason=EXECUTION_REASON_ERROR,
            session_id=frame.session_key,
            message_preview=frame.current_message,
        ),
    )
    if frame.middleware_state is not None and frame.middleware is not None:
        frame.middleware_state.message_text = frame.current_message
        frame.middleware_state.system_prompt = frame.system_prompt
        frame.middleware_state.context_xml = (
            frame.enriched_context_xml or frame.context_xml
        )
        frame.middleware_state.metadata = {"phase": "error", "error": str(error)}
        await frame.middleware.dispatch("on_error", frame.middleware_state)
    logger.error(f"ChatInter 处理失败: {error}")
    await MessageUtils.build_failure_message().send()
    frame.stage(PipelineStage.ERROR)
    if frame.post_gate_callback is not None:
        await frame.post_gate_callback(phase="post_gate:error")
    _finish_trace(
        trace=frame.trace,
        user_id=frame.user_id,
        group_id=frame.group_id,
        message_preview=frame.current_message,
        route_report=frame.route_report,
        budget_controller=frame.budget_controller,
    )
    frame.turn_finished = True


__all__ = [
    "handle_pipeline_cancelled",
    "handle_pipeline_error",
    "remember_target_resolution",
    "stage_agent_run",
    "stage_capability_hint",
    "stage_current_user",
    "stage_dialogue_state",
    "stage_event_context",
    "stage_identity",
    "stage_memory",
    "stage_persist",
    "stage_scratchpad",
    "stage_send",
    "stage_thread_context",
]
