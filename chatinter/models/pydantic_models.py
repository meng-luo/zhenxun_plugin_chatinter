"""
ChatInter - Pydantic 数据模型
"""

from typing import Literal

from pydantic import BaseModel, Field


class PluginInfo(BaseModel):
    """用于意图分析的插件信息"""

    class PluginCommandMeta(BaseModel):
        class CommandActorScope:
            SELF_ONLY = "self_only"
            ALLOW_OTHER = "allow_other"

        class CommandTargetRequirement:
            NONE = "none"
            OPTIONAL = "optional"
            REQUIRED = "required"

        command: str = Field(description="命令主干")
        aliases: list[str] = Field(default_factory=list, description="命令别名")
        prefixes: list[str] = Field(default_factory=list, description="命令前缀")
        params: list[str] = Field(default_factory=list, description="参数提示")
        examples: list[str] = Field(default_factory=list, description="示例命令")
        text_min: int | None = Field(default=None, description="文本参数最小数量")
        text_max: int | None = Field(default=None, description="文本参数最大数量")
        image_min: int | None = Field(default=None, description="图片参数最小数量")
        image_max: int | None = Field(default=None, description="图片参数最大数量")
        allow_at: bool | None = Field(
            default=None, description="@是否可作为图片参数输入"
        )
        actor_scope: Literal["self_only", "allow_other"] = Field(
            default=CommandActorScope.ALLOW_OTHER,
            description="执行者范围：self_only=仅本人；allow_other=可作用于他人",
        )
        target_requirement: Literal["none", "optional", "required"] = Field(
            default=CommandTargetRequirement.NONE,
            description="目标参数要求：none/optional/required",
        )
        target_sources: list[Literal["at", "reply", "nickname", "self"]] = Field(
            default_factory=list,
            description="可接受的目标来源",
        )
        requires_reply: bool = Field(
            default=False,
            description="是否需要 reply 上下文",
        )
        requires_private: bool = Field(
            default=False,
            description="是否仅限私聊",
        )
        requires_to_me: bool = Field(
            default=False,
            description="是否需要 to_me / @机器人 上下文",
        )
        allow_sticky_arg: bool = Field(
            default=False,
            description="是否允许命令头和参数粘连（例如：敲葱葱）",
        )
        access_level: Literal["public", "admin", "superuser", "restricted"] = Field(
            default="public",
            description="命令访问级别：public=普通用户可见；admin/superuser/restricted=导入时过滤",
        )

    module: str = Field(description="插件模块名")
    name: str = Field(description="插件名称")
    description: str = Field(description="插件描述")
    commands: list[str] = Field(default_factory=list, description="可用命令列表")
    aliases: list[str] = Field(default_factory=list, description="插件别名")
    command_meta: list[PluginCommandMeta] = Field(
        default_factory=list, description="命令元信息"
    )
    usage: str | None = Field(default=None, description="用法说明")
    admin_level: int | None = Field(default=None, description="插件权限等级要求")
    limit_superuser: bool = Field(
        default=False, description="是否限制超级管理员"
    )


class PluginKnowledgeBase(BaseModel):
    """插件知识库，供 LLM 理解可用功能"""

    plugins: list[PluginInfo] = Field(default_factory=list, description="可用插件列表")
    user_role: str = Field(description="用户角色: 普通用户/管理员/超级管理员")
