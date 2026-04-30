"""
ChatInter - Pydantic 数据模型
"""

from typing import Any, Literal

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
    limit_superuser: bool = Field(default=False, description="是否限制超级管理员")


class PluginKnowledgeBase(BaseModel):
    """插件知识库，供 LLM 理解可用功能"""

    plugins: list[PluginInfo] = Field(default_factory=list, description="可用插件列表")
    user_role: str = Field(description="用户角色: 普通用户/管理员/超级管理员")


class CommandRequirement(BaseModel):
    """插件命令执行前置条件"""

    params: list[str] = Field(default_factory=list, description="文本参数提示")
    text_min: int = Field(default=0, description="文本参数最小数量")
    text_max: int | None = Field(default=None, description="文本参数最大数量")
    image_min: int = Field(default=0, description="图片参数最小数量")
    image_max: int | None = Field(default=None, description="图片参数最大数量")
    allow_at: bool = Field(default=False, description="@是否可作为输入")
    actor_scope: Literal["self_only", "allow_other"] = Field(
        default=PluginInfo.PluginCommandMeta.CommandActorScope.ALLOW_OTHER,
        description="执行者范围",
    )
    target_requirement: Literal["none", "optional", "required"] = Field(
        default=PluginInfo.PluginCommandMeta.CommandTargetRequirement.NONE,
        description="目标参数要求",
    )
    target_sources: list[Literal["at", "reply", "nickname", "self"]] = Field(
        default_factory=list,
        description="可接受的目标来源",
    )
    requires_reply: bool = Field(default=False, description="是否需要回复上下文")
    requires_private: bool = Field(default=False, description="是否仅限私聊")
    requires_to_me: bool = Field(default=False, description="是否需要 @机器人")


class CommandCapability(BaseModel):
    """单条可被 ChatInter 路由的命令能力"""

    command: str = Field(description="命令主干")
    aliases: list[str] = Field(default_factory=list, description="命令别名")
    prefixes: list[str] = Field(default_factory=list, description="命令前缀")
    examples: list[str] = Field(default_factory=list, description="示例命令")
    requirement: CommandRequirement = Field(default_factory=CommandRequirement)
    allow_sticky_arg: bool = Field(default=False, description="是否允许粘连参数")


class PluginCapability(BaseModel):
    """插件级能力描述"""

    module: str = Field(description="插件模块名")
    name: str = Field(description="插件名称")
    description: str = Field(default="", description="插件描述")
    usage: str | None = Field(default=None, description="插件用法")
    commands: list[CommandCapability] = Field(
        default_factory=list,
        description="命令能力列表",
    )
    aliases: list[str] = Field(default_factory=list, description="插件别名")
    tags: list[str] = Field(default_factory=list, description="能力标签")
    public: bool = Field(default=True, description="是否可暴露给普通路由")


class PluginReference(BaseModel):
    """Router / Planner 使用的插件引用卡"""

    module: str = Field(description="插件模块名")
    name: str = Field(description="插件名称")
    does: str = Field(default="", description="插件能力摘要")
    commands: list[str] = Field(default_factory=list, description="命令主干")
    aliases: list[str] = Field(default_factory=list, description="别名")
    examples: list[str] = Field(default_factory=list, description="示例")
    requires: dict[str, bool] = Field(default_factory=dict, description="需求摘要")
    command_schemas: list["PluginCommandSchema"] = Field(
        default_factory=list,
        description="命令级工具 schema，用于自然语言槽位填充和命令渲染",
    )


class CommandSlotSpec(BaseModel):
    """命令参数槽位。"""

    name: str = Field(description="槽位名")
    type: Literal["text", "int", "float", "bool", "at", "image"] = Field(
        default="text",
        description="槽位类型",
    )
    required: bool = Field(default=False, description="是否必填")
    default: Any = Field(default=None, description="默认值")
    aliases: list[str] = Field(default_factory=list, description="自然语言别名")
    description: str = Field(default="", description="槽位说明")


class PluginCommandSchema(BaseModel):
    """单条命令的工具化 schema。"""

    command_id: str = Field(description="稳定命令 ID")
    head: str = Field(description="最终执行命令头")
    aliases: list[str] = Field(default_factory=list, description="自然语言别名")
    description: str = Field(default="", description="命令用途")
    slots: list[CommandSlotSpec] = Field(default_factory=list, description="参数槽位")
    render: str = Field(description="命令渲染模板，例如：塞红包 {amount} {num}")
    requires: dict[str, bool] = Field(default_factory=dict, description="命令级需求")
    command_role: Literal[
        "execute",
        "helper",
        "usage",
        "catalog",
        "template",
        "random",
    ] = Field(default="execute", description="命令在路由中的语义角色")
    payload_policy: Literal[
        "none",
        "text",
        "slots",
        "image_only",
        "text_or_image",
        "free_tail",
    ] = Field(default="none", description="命令对自然语言尾巴的接收策略")
    extra_text_policy: Literal["keep", "discard", "slot_only"] = Field(
        default="keep",
        description="schema 渲染后多余文本的处理策略",
    )
    source: Literal["explicit", "matcher", "metadata", "fallback", "override"] = Field(
        default="fallback",
        description="schema 来源，用于后续质量诊断和路由加权",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="schema 自动生成置信度",
    )
    matcher_key: str | None = Field(
        default=None,
        description="可选 matcher 标识，后续可用于更精确的事件重投扇出控制",
    )


class CapabilityGraphSnapshot(BaseModel):
    """一次插件能力图快照"""

    version: str = Field(default="chatinter.capability_graph.v1")
    plugins: list[PluginCapability] = Field(default_factory=list)
    user_role: str = Field(default="普通用户")
    created_at: float = Field(default=0.0)
