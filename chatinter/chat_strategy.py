from __future__ import annotations

from .chat_dialogue_planner import ChatDialoguePlan, DialogueState


def build_chat_strategy_prompt(plan: ChatDialoguePlan | None) -> str:
    if plan is None:
        return ""

    common = (
        "\n对话策略：先回应用户真实意图，不要把插件讨论误当执行请求；"
        "不知道就说明不确定，必要时只追问一个关键问题。"
    )
    if plan.kind == "casual_chat":
        return common + "闲聊时自然短答，接住话题即可，避免说教。"
    if plan.kind == "factual_qa":
        return common + "事实问答要具体、区分确定和不确定；没有依据时不要编造。"
    if plan.kind == "emotional_support":
        return common + "情绪场景先用一句话承接感受，再给一个具体可做的小建议。"
    if plan.kind == "recap":
        return common + "回顾对话时只基于给定历史，不要补不存在的内容。"
    if plan.kind == "identity_query":
        return (
            common + "身份/称呼问题只基于 <turn_identity>、<relevant_people>、"
            "<thread> 和长期记忆回答；有唯一高置信候选才说明是谁，"
            "多候选或无候选时直接说明不确定并请用户@确认。"
        )
    if plan.kind == "memory_update":
        return common + "记忆确认要谨慎：可确认已记录/会参考，但不要承诺永久准确。"
    if plan.kind == "explain_context":
        return common + "解释上下文时说明依据来自当前消息、回复链或历史，不要执行命令。"
    if plan.kind == "complex_reasoning":
        return common + "复杂问题结论优先，再分点说明关键步骤和取舍。"
    return common


def build_dialogue_state_prompt(state: DialogueState | None) -> str:
    if state is None:
        return ""
    length_rule = {
        "short": "优先短答，保留一点自然语气。",
        "medium": "中等长度，先结论再补关键依据。",
        "long": "可以展开，但先给结论和结构。",
    }.get(state.response_length, "按用户问题自然控制长度。")
    followup_rule = (
        "如果关键上下文不足，只追问一个最关键问题。"
        if state.need_followup
        else "不要为了显得热情而多问无必要的问题。"
    )
    group_rule = {
        "brief_react": "群聊里优先少说、接话，不抢话题。",
        "answer_directly": "直接回答当前问题，别绕到工具或无关话题。",
        "support": "先接住情绪，再给很小的可执行建议。",
        "structured": "可简短分点，但不要写成长报告。",
        "agent": "按超级用户 Agent 任务语气保持清晰、可验证。",
    }.get(state.group_reply_policy, "按当前场景自然回复。")
    topic_rule = f"当前话题线索={state.topic_hint}。" if state.topic_hint else ""
    previous_rule = ""
    if state.last_user_message and state.continuity in {"same_topic", "followup"}:
        previous_rule = (
            f"上一轮用户大意={state.last_user_message[:80]}；"
            f"上一轮回复大意={state.last_reply_summary[:80]}。"
        )
    return (
        "\n当前对话状态："
        f"语气={state.tone}；用户情绪={state.user_emotion}；"
        f"对话目的={state.dialogue_purpose}；回复长度={state.response_length}；"
        f"连续性={state.continuity}；群聊策略={state.group_reply_policy}。"
        f"回复姿态={state.reply_posture}；群聊氛围={state.group_atmosphere}；"
        f"称呼模式={state.address_mode}。"
        f"{topic_rule}{previous_rule}{length_rule}{followup_rule}{group_rule}"
    )


__all__ = ["build_chat_strategy_prompt", "build_dialogue_state_prompt"]
