from __future__ import annotations

from .chat_dialogue_planner import ChatDialoguePlan
from .turn_runtime import TurnBudgetController

_GENERIC_REPLY_HINTS = (
    "我明白",
    "你说得对",
    "确实如此",
    "这个问题很有意思",
    "不好说",
)


def _debug(message: str) -> None:
    try:
        from zhenxun.services import logger

        logger.debug(message)
    except Exception:
        pass


def should_quality_check(plan: ChatDialoguePlan | None, reply_text: str) -> bool:
    if plan is None:
        return False
    text = str(reply_text or "").strip()
    if not text:
        return False
    if plan.need_rewrite_check:
        return True
    if len(text) >= 180 and plan.kind in {"complex_reasoning", "factual_qa"}:
        return True
    if len(text) <= 28 and any(hint in text for hint in _GENERIC_REPLY_HINTS):
        return True
    return False


async def refine_chat_reply(
    *,
    plan: ChatDialoguePlan | None,
    user_message: str,
    reply_text: str,
    context_xml: str = "",
    budget_controller: TurnBudgetController | None = None,
) -> str:
    if not should_quality_check(plan, reply_text):
        return reply_text
    if (
        budget_controller is not None
        and budget_controller.prompt_budget_remaining() < 600
    ):
        return reply_text

    instruction = (
        "你是对话回复质检器。只输出改写后的最终回复，不要解释质检过程。"
        "要求：具体回应用户；不编造；不确定就说明；语气自然；"
        "复杂问题结论优先，必要时分点；普通问题保持简洁。"
    )
    prompt = (
        f"对话类型：{getattr(plan, 'kind', 'chat')}\n"
        f"上下文：{str(context_xml or '')[:1600]}\n"
        f"用户消息：{user_message}\n"
        f"原回复：{reply_text}\n"
        "请在不改变事实的前提下改写得更准确、更具体。"
    )
    try:
        from zhenxun.services import chat

        from .config import build_reasoning_generation_config, get_model_name

        response = await chat(
            prompt,
            instruction=instruction,
            model=get_model_name(),
            config=build_reasoning_generation_config(),
        )
    except Exception as exc:
        _debug(f"chatinter quality guard skipped: {exc}")
        return reply_text

    refined = str(response.text if response else "").strip()
    if not refined:
        return reply_text
    if budget_controller is not None:
        usage = response.usage_info if response and response.usage_info else {}
        try:
            tokens = int(
                usage.get("total_tokens")
                or usage.get("totalTokenCount")
                or usage.get("total_token_count")
                or 0
            )
        except Exception:
            tokens = 0
        budget_controller.record_prompt_use(
            stage="chat_quality",
            estimated_tokens=tokens,
            cache_break=False,
            compacted=False,
        )
    return refined


__all__ = ["refine_chat_reply", "should_quality_check"]
