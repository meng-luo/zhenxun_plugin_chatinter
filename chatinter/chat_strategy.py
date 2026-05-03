from __future__ import annotations

from .chat_dialogue_planner import ChatDialoguePlan


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


__all__ = ["build_chat_strategy_prompt"]
