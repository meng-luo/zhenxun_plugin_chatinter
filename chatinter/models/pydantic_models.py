"""
ChatInter - Pydantic 结构化输出模型

定义了用于 LLM 结构化输出的数据模型。
"""

from typing import Literal

from pydantic import BaseModel, Field


class PluginIntent(BaseModel):
    """插件调用意图"""

    plugin_name: str = Field(description="目标插件名称")
    command: str = Field(description="要执行的命令，包含参数")
    confidence: float = Field(description="置信度 0.0-1.0", ge=0.0, le=1.0)
    response: str = Field(description="对用户的回复，告知正在执行操作")


class ChatIntent(BaseModel):
    """普通聊天意图"""

    response: str = Field(description="对用户消息的回复内容")


class IntentAnalysisResult(BaseModel):
    """
    意图分析结果

    action 字段决定下一步操作：
    - call_plugin: 用户意图是调用某个插件功能
    - chat: 用户意图是进行普通对话等其他内容
    """

    action: Literal["call_plugin", "chat"] = Field(
        description="识别出的用户意图类型"
    )
    plugin_intent: PluginIntent | None = Field(
        default=None, description="插件调用意图，当 action='call_plugin' 时有效"
    )
    chat_intent: ChatIntent | None = Field(
        default=None, description="聊天意图，当 action='chat' 时有效"
    )


class PluginInfo(BaseModel):
    """用于意图分析的插件信息"""

    name: str = Field(description="插件名称")
    description: str = Field(description="插件描述")
    commands: list[str] = Field(default_factory=list, description="可用命令列表")
    usage: str | None = Field(default=None, description="用法说明")


class PluginKnowledgeBase(BaseModel):
    """插件知识库，供 LLM 理解可用功能"""

    plugins: list[PluginInfo] = Field(default_factory=list, description="可用插件列表")
    user_role: str = Field(description="用户角色: 普通用户/管理员/超级管理员")

    def to_prompt_text(self) -> str:
        """转换为可用的提示文本"""
        if not self.plugins:
            return "当前没有可用的插件功能。"

        parts = []
        for i, plugin in enumerate(self.plugins, 1):
            cmd_str = ", ".join(plugin.commands) if plugin.commands else "无"
            part = f"{i}. {plugin.name}\n   描述: {plugin.description}\n   命令: {cmd_str}"
            if plugin.usage:
                # 限制用法长度，避免 token 过多
                usage_preview = (
                    plugin.usage[:500] + "..."
                    if len(plugin.usage) > 500
                    else plugin.usage
                )
                part += f"\n   用法: {usage_preview}"
            parts.append(part)

        return "\n\n".join(parts)