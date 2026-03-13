"""
ChatInter - 意图分析器

使用 LLM 分析用户意图，支持 3 层上下文结构：
1. System: 系统设定
2. Context: XML 格式的语境层
3. Current: 当前用户消息

支持多模态输入（图片识别）。
"""

import asyncio

from zhenxun.services import LLMException, logger
from zhenxun.services.llm import AI, LLMContentPart, LLMMessage as LLMMessageObj

from .config import get_config_value
from .prompt_templates import build_intent_prompt
from .models.pydantic_models import IntentAnalysisResult, PluginKnowledgeBase


async def analyze_intent(
    message: str,
    knowledge_base: PluginKnowledgeBase,
    system_prompt: str,
    context_xml: str,
    user_id: str | None = None,
    nickname: str = "用户",
    group_id: str | None = None,
    image_parts: list[LLMContentPart] | None = None,
) -> IntentAnalysisResult | None:
    """分析用户意图

    上下文结构：
    1. System: 系统提示（角色设定、好感度规则）
    2. Context: XML 语境（<qq_context>, <context_layers>, <history>, <user_state>）
    3. Current: 当前用户消息（支持多模态）

    参数:
        message: 用户当前消息
        knowledge_base: 插件知识库
        system_prompt: 系统提示词
        context_xml: XML 格式的语境层
        user_id: 用户 ID
        nickname: 用户昵称
        group_id: 群组 ID
        image_parts: 图片内容列表（多模态）

    返回:
        IntentAnalysisResult | None: 意图分析结果
    """
    model = get_config_value("INTENT_MODEL", None)
    timeout = get_config_value("INTENT_TIMEOUT", 30)
    threshold = get_config_value("CONFIDENCE_THRESHOLD", 0.7)

    full_prompt = build_intent_prompt(knowledge_base, system_prompt, context_xml, message, threshold)

    try:
        ai = AI()

        user_content: list[LLMContentPart] | str = full_prompt
        if image_parts:
            user_content = [LLMContentPart.text_part(full_prompt), *image_parts]

        result = await asyncio.wait_for(
            ai.generate_structured(
                message=LLMMessageObj.user(user_content),
                response_model=IntentAnalysisResult,
                model=model,
            ),
            timeout=timeout,
        )

        logger.debug(f"意图分析结果：action={result.action}")
        return result

    except asyncio.TimeoutError:
        logger.warning(f"意图分析超时（{timeout}秒）")
        return None
    except LLMException as e:
        logger.error(f"意图分析 LLM 调用失败：{e}")
        return None
    except Exception as e:
        logger.error(f"意图分析发生未知错误：{e}")
        return None


async def analyze_intent_safe(
    message: str,
    knowledge_base: PluginKnowledgeBase,
    system_prompt: str,
    context_xml: str,
    user_id: str | None = None,
    nickname: str = "用户",
    group_id: str | None = None,
    image_parts: list[LLMContentPart] | None = None,
) -> IntentAnalysisResult:
    """安全的意图分析（保证返回结果）

    参数:
        message: 用户当前消息
        knowledge_base: 插件知识库
        system_prompt: 系统提示词
        context_xml: XML 格式的语境层
        user_id: 用户 ID
        nickname: 用户昵称
        group_id: 群组 ID
        image_parts: 图片内容列表（多模态）

    返回:
        IntentAnalysisResult: 意图分析结果
    """
    result = await analyze_intent(
        message, knowledge_base, system_prompt, context_xml, user_id, nickname, group_id, image_parts
    )

    if result is None:
        return IntentAnalysisResult(
            action="unknown",
            plugin_intent=None,
            chat_intent=None,
        )

    return result
