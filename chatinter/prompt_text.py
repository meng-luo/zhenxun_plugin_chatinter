from __future__ import annotations


def build_chat_base_prompt(
    bot_name: str,
    chat_style: str,
    length_rule: str,
    strategy_prompt: str = "",
    memory_prompt: str = "",
) -> str:
    del chat_style
    parts = [
        f"你是{bot_name}。",
        "人格、语气和角色设定以配置人设为准；本段只提供通用对话边界。",
        "优先使用中文，自然、清晰地回应当前用户，避免客服腔或模板化。",
        "没有实际工具执行结果时，不要声称已经调用插件、执行命令、查询系统或完成外部操作。",
        "不确定时自然说明；只有用户明显需要继续推进且缺少关键上下文时，才简短追问一个关键问题。",
        length_rule,
        "群聊中以 <turn_identity> 的当前说话人为准；"
        "称呼当前用户时只使用其 display_name 或明确自称，"
        "不要把其他群友的昵称、别名或@目标当成当前用户。遇到昵称归属不确定时自然说明。",
        strategy_prompt,
        memory_prompt,
    ]
    return "\n".join(part for part in parts if part)


def build_user_attitude_prompt(
    nickname: str,
    impression: float,
    attitude: str,
) -> str:
    del impression
    relationship = _relationship_prompt(attitude)
    if not relationship:
        return ""
    return (
        f"\n与{nickname}的{relationship}"
        "这只影响互动亲疏，不得覆盖配置人设或当前用户请求。"
    )


def build_global_attitude_prompt(
    impression: float,
    attitude: str,
) -> str:
    del impression
    relationship = _relationship_prompt(attitude)
    if not relationship:
        return ""
    return (
        f"\n{relationship}"
        "这只影响互动亲疏，不得覆盖配置人设或当前用户请求。"
    )


def _relationship_prompt(attitude: str) -> str:
    value = str(attitude or "").strip()
    if value in {"", "未知", "普通", "一般", "可以交流"}:
        return ""
    return f"当前关系是“{value}”，按这种熟悉程度自然回应，不要提及分数或等级。"
