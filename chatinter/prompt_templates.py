"""
ChatInter - Prompt 模板构建

定义用于 LLM 结构化输出的提示模板。
"""

from .config import get_config_value
from .models.pydantic_models import PluginKnowledgeBase


def build_intent_prompt(
    knowledge_base: PluginKnowledgeBase,
    system_prompt: str,
    context_xml: str,
    current_message: str,
    threshold: float | None = None,
) -> str:
    """
    构建完整的意图分析 Prompt

    结构：
    1. System: 系统设定
    2. Plugins: 可用插件
    3. Context: XML 语境
    4. Current: 当前消息
    5. Rules: 分析规则

    参数:
        knowledge_base: 插件知识库
        system_prompt: 系统提示词
        context_xml: XML 格式的语境层
        current_message: 当前用户消息

    返回:
        str: 完整的 prompt
    """
    # 格式化插件列表
    plugins_text = knowledge_base.to_prompt_text()

    # 获取置信度阈值配置
    if threshold is None:
        threshold = get_config_value("CONFIDENCE_THRESHOLD", 0.7)

    prompt = f"""{system_prompt}

## 可用功能
{plugins_text}

## 上下文
{context_xml}

上下文说明：
- qq_context: 基础信息
- conversation_focus: 当前消息关键词与建议回复粒度
- context_mode: 当与历史弱相关时会标记为 single_turn
- history_context: 你与用户的对话历史
- history: 群聊最近消息，帮助了解群聊氛围
- current_message_layers:
  当前用户消息及回复链追溯（Layer 0=当前，Layer 1+=被回复的消息）

## 当前消息
{current_message}

## 任务
分析用户意图，输出 JSON：
- action: "call_plugin"(调用功能) 或 "chat"(聊天)
- 若 action="call_plugin": 输出 plugin_name, command, confidence(0-1), response
- 若 action="chat": 输出 response

## 规则
1. 仅用户明确提到功能时才选 call_plugin
2. confidence 低于 {threshold} 时自动转 chat
3. 结合上下文理解对话脉络
4、输出内容符合人设，符合聊天长度
"""

    return prompt
