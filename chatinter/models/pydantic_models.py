"""
ChatInter - Pydantic 数据模型
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class SemanticToolContract(BaseModel):
    """Model-visible semantic contract declared by plugin metadata."""

    name: str = Field(description="稳定工具名称")
    description: str = Field(default="", description="能力描述")
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema 参数定义",
    )
    bound_commands: list[str] = Field(
        default_factory=list,
        description="通过 matcher 身份可靠绑定的命令头",
    )
    use_cases: list[str] = Field(default_factory=list, description="适用场景")
    anti_use_cases: list[str] = Field(default_factory=list, description="不适用场景")
    output_mode: Literal["text", "image", "file", "plugin_output", "action"] | None = (
        Field(default=None, description="主要输出形态")
    )
    side_effect: Literal["none", "query", "send", "mutate"] | None = Field(
        default=None,
        description="真实副作用语义",
    )
    risk: Literal["low", "medium", "high"] | None = Field(
        default=None,
        description="能力风险级别",
    )
    source_of_truth: Literal[
        "model_knowledge",
        "plugin_runtime",
        "bot_state",
        "external_service",
        "local_state",
        "user_provided",
        "unknown",
    ] | None = Field(default=None, description="事实权威来源")
    requires_real_tool: bool | None = Field(
        default=None,
        description="是否必须调用真实工具",
    )
    entity_scope: Literal[
        "none",
        "self_bot",
        "actor_user",
        "target_user",
        "group",
        "global",
        "external",
    ] | None = Field(default=None, description="能力作用范围")
    intent_types: list[str] = Field(default_factory=list, description="通用意图类型")
    requires_real_result: bool | None = Field(
        default=None,
        description="是否必须获得真实执行结果",
    )
    execution_policy: Literal[
        "normal",
        "explicit_only",
        "strong_intent",
        "confirmation_required",
    ] | None = Field(default=None, description="执行门槛")
    source: Literal["smart_tools"] = "smart_tools"


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
        choices: dict[str, list[str]] = Field(
            default_factory=dict,
            description="参数枚举约束，key 为参数名，value 为可选值",
        )
        slot_types: dict[str, str] = Field(
            default_factory=dict,
            description="解析器提供的参数类型",
        )
        slot_renderers: dict[str, str] = Field(
            default_factory=dict,
            description="参数值到原生命令片段的渲染模板",
        )
        shortcut_renders: list[dict[str, Any]] = Field(
            default_factory=list,
            description="解析器 shortcut 到真实命令的通用渲染映射",
        )
        description: str = Field(default="", description="命令描述")
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
            description="命令头与首个参数之间是否允许省略分隔符",
        )
        argument_source: Literal[
            "runtime_handler",
            "runtime_parser",
            "discovery",
            "declared",
            "usage",
            "identity_fallback",
            "unknown",
        ] = Field(default="unknown", description="参数契约的最强事实来源")
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
    introduction: str | None = Field(default=None, description="插件能力介绍")
    precautions: list[str] = Field(default_factory=list, description="使用注意事项")
    semantic_tools: list[SemanticToolContract] = Field(
        default_factory=list,
        description="插件声明的显式语义工具契约",
    )
    admin_level: int | None = Field(default=None, description="插件权限等级要求")
    limit_superuser: bool = Field(default=False, description="是否限制超级管理员")
    status: bool = Field(default=True, description="插件全局启用状态")
    block_type: str | None = Field(default=None, description="插件全局禁用场景")
    load_status: bool = Field(default=True, description="插件加载状态")
    block_keys: list[str] = Field(
        default_factory=list,
        description="用于对齐本体群内插件关闭字段的模块键",
    )


class PluginKnowledgeBase(BaseModel):
    """插件知识库，供 LLM 理解可用功能"""

    plugins: list[PluginInfo] = Field(default_factory=list, description="可用插件列表")
    user_role: str = Field(description="用户角色: 普通用户/管理员/超级管理员")


class CommandRequirement(BaseModel):
    """插件命令执行前置条件"""

    params: list[str] = Field(default_factory=list, description="文本参数提示")
    choices: dict[str, list[str]] = Field(
        default_factory=dict,
        description="参数枚举约束",
    )
    slot_types: dict[str, str] = Field(
        default_factory=dict,
        description="解析器提供的参数类型",
    )
    slot_renderers: dict[str, str] = Field(
        default_factory=dict,
        description="参数值到原生命令片段的渲染模板",
    )
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
    argument_source: Literal[
        "runtime_handler",
        "runtime_parser",
        "discovery",
        "declared",
        "usage",
        "identity_fallback",
        "unknown",
    ] = Field(default="unknown", description="参数契约的最强事实来源")


class CommandCapability(BaseModel):
    """单条可被 ChatInter 路由的命令能力"""

    command: str = Field(description="命令主干")
    aliases: list[str] = Field(default_factory=list, description="命令别名")
    prefixes: list[str] = Field(default_factory=list, description="命令前缀")
    description: str = Field(default="", description="命令描述")
    examples: list[str] = Field(default_factory=list, description="示例命令")
    requirement: CommandRequirement = Field(default_factory=CommandRequirement)
    allow_sticky_arg: bool = Field(default=False, description="是否允许粘连参数")
    shortcut_renders: list[dict[str, Any]] = Field(
        default_factory=list,
        description="解析器 shortcut 到真实命令的通用渲染映射",
    )


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
    """Native tools / Planner 使用的插件引用卡"""

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
    choices: list[str] = Field(
        default_factory=list,
        description="可选枚举值；来自命令解析器的 Literal/Union 约束",
    )
    renderer: str = Field(
        default="{value}",
        description="槽位存在时生成原生命令片段的模板",
    )


class PluginCommandSchema(BaseModel):
    """单条命令的工具化 schema。"""

    command_id: str = Field(description="稳定命令 ID")
    head: str = Field(description="最终执行命令头")
    aliases: list[str] = Field(default_factory=list, description="自然语言别名")
    description: str = Field(default="", description="命令用途")
    slots: list[CommandSlotSpec] = Field(default_factory=list, description="参数槽位")
    render: str = Field(description="将参数槽位转换为原生命令文本的模板")
    requires: dict[str, bool] = Field(default_factory=dict, description="命令级需求")
    allow_at: bool | None = Field(default=None, description="@是否可作为目标输入")
    allow_sticky_arg: bool = Field(default=False, description="是否允许粘连参数")
    actor_scope: Literal["self_only", "allow_other"] = Field(
        default="allow_other",
        description="执行者范围",
    )
    target_requirement: Literal["none", "optional", "required"] = Field(
        default="none",
        description="目标参数要求",
    )
    target_sources: list[Literal["at", "reply", "nickname", "self"]] = Field(
        default_factory=list,
        description="可接受的目标来源",
    )
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
    retrieval_phrases: list[str] = Field(
        default_factory=list,
        description="工具检索短语，不直接暴露为可执行命令",
    )
    shortcut_renders: list[dict[str, Any]] = Field(
        default_factory=list,
        description="解析器 shortcut 到真实命令的通用渲染映射",
    )
    argument_source: Literal[
        "runtime_handler",
        "runtime_parser",
        "discovery",
        "declared",
        "usage",
        "identity_fallback",
        "unknown",
    ] = Field(default="unknown", description="参数契约的最强事实来源")


class CommandToolSnapshot(BaseModel):
    """安全过滤后的命令级工具快照。"""

    command_id: str = Field(description="稳定命令 ID")
    plugin_module: str = Field(description="插件模块名")
    plugin_name: str = Field(description="插件名称")
    head: str = Field(description="最终执行命令头")
    aliases: list[str] = Field(default_factory=list, description="自然语言别名")
    description: str = Field(default="", description="命令用途")
    usage: str | None = Field(default=None, description="插件用法")
    examples: list[str] = Field(default_factory=list, description="示例")
    slots: list[CommandSlotSpec] = Field(default_factory=list, description="参数槽位")
    requires: dict[str, bool] = Field(default_factory=dict, description="命令级需求")
    allow_at: bool | None = Field(default=None, description="@是否可作为目标输入")
    allow_sticky_arg: bool = Field(default=False, description="是否允许粘连参数")
    actor_scope: Literal["self_only", "allow_other"] = Field(
        default="allow_other",
        description="执行者范围",
    )
    target_requirement: Literal["none", "optional", "required"] = Field(
        default="none",
        description="目标参数要求",
    )
    target_sources: list[Literal["at", "reply", "nickname", "self"]] = Field(
        default_factory=list,
        description="可接受的目标来源",
    )
    render: str = Field(default="", description="命令渲染模板")
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
    command_role: Literal[
        "execute",
        "helper",
        "usage",
        "catalog",
        "template",
        "random",
    ] = Field(default="execute", description="命令在路由中的语义角色")
    family: str = Field(default="general", description="候选多样化分组")
    retrieval_phrases: list[str] = Field(
        default_factory=list,
        description="用于本地召回/向量召回的自然语言短语",
    )
    capability_text: str = Field(
        default="",
        description="统一、短句化的能力摘要，用于 no-hit 能力检索",
    )
    task_verbs: list[str] = Field(
        default_factory=list,
        description="该命令支持的通用动作词，如 查询/生成/识别",
    )
    input_requirements: list[str] = Field(
        default_factory=list,
        description="输入需求摘要，如 文本/图片/回复/链接/@",
    )
    use_cases: list[str] = Field(
        default_factory=list,
        description="适用场景摘要，由插件元数据/schema 自动生成",
    )
    anti_use_cases: list[str] = Field(
        default_factory=list,
        description="不适用场景摘要，用于降低闲聊误触发",
    )
    output_mode: Literal["text", "image", "file", "plugin_output", "action"] = Field(
        default="plugin_output",
        description="工具主要输出形态",
    )
    side_effect: Literal["none", "query", "send", "mutate"] = Field(
        default="send",
        description="工具副作用级别",
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        default="low",
        description="能力风险级别",
    )
    risk: Literal["low", "medium", "high"] = Field(
        default="low",
        description="通用风险级别别名，供运行时策略统一读取",
    )
    source_of_truth: Literal[
        "model_knowledge",
        "plugin_runtime",
        "bot_state",
        "external_service",
        "local_state",
        "user_provided",
        "unknown",
    ] = Field(
        default="plugin_runtime",
        description="真实结果来源，用于判断能否由模型直接回答",
    )
    requires_real_tool: bool = Field(
        default=True,
        description="是否必须调用真实工具才能宣称完成",
    )
    entity_scope: Literal[
        "none",
        "self_bot",
        "actor_user",
        "target_user",
        "group",
        "global",
        "external",
    ] = Field(
        default="global",
        description="能力作用的实体范围",
    )
    reliability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="能力初始可靠性估计，历史反馈会继续修正",
    )
    schema_quality: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="schema 完整度估计，用于暴露和 obligation 决策",
    )
    soft_tool: bool = Field(
        default=False,
        description="低上下文软工具，只有明确请求时才应执行",
    )
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="插件能力的附加元数据",
    )
    intent_types: list[str] = Field(
        default_factory=list,
        description="通用能力类型，如 query/generate/media/status/random/help",
    )
    requires_real_result: bool = Field(
        default=True,
        description="是否需要真实插件执行结果，避免模型直接代答",
    )
    generative: bool = Field(
        default=False,
        description="是否属于生成/随机/媒体制作类能力",
    )
    execution_policy: Literal[
        "normal",
        "explicit_only",
        "strong_intent",
        "confirmation_required",
    ] = Field(
        default="normal",
        description="运行时执行门槛，由能力属性自动推导",
    )
    source: Literal["explicit", "matcher", "metadata", "fallback", "override"] = Field(
        default="fallback",
        description="schema 来源，用于候选诊断和路由加权",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="schema 自动生成置信度",
    )
    matcher_key: str | None = Field(
        default=None,
        description="可选 matcher 标识",
    )
    source_signature: str = Field(default="", description="工具快照失效签名")


class CommandCandidateFeatures(BaseModel):
    """候选命令进入 LLM 重排前的可解释特征。"""

    lexical_score: float = Field(default=0.0, description="本地词法/别名召回分")
    exact_score: float = Field(default=0.0, description="真实命令头/别名命中分")
    semantic_score: float = Field(default=0.0, description="自然语言短语召回分")
    slot_score: float = Field(default=0.0, description="参数槽位命中分")
    context_score: float = Field(default=0.0, description="图片/@/回复上下文命中分")
    feedback_score: float = Field(default=0.0, description="执行反馈加权分")
    schema_score: float = Field(default=0.0, description="schema 完整度加权分")
    reliability_score: float = Field(default=0.0, description="历史成功可靠性分")
    false_trigger_score: float = Field(default=0.0, description="历史误触发惩罚分")
    param_failure_score: float = Field(default=0.0, description="历史参数失败惩罚分")
    latency_score: float = Field(default=0.0, description="历史平均耗时调节分")
    negative_score: float = Field(default=0.0, description="冲突或不兼容惩罚分")


class CommandCandidateSnapshot(BaseModel):
    """给 LLM 重排/分类使用的候选命令包。"""

    rank: int = Field(description="候选排名")
    score: float = Field(description="最终本地召回分")
    reason: str = Field(default="", description="命中原因")
    exact_protected: bool = Field(default=False, description="是否真实命令头命中")
    plugin_module: str = Field(description="插件模块名")
    plugin_name: str = Field(description="插件名称")
    family: str = Field(default="general", description="候选族群")
    command_id: str = Field(description="稳定命令 ID")
    head: str = Field(description="最终执行命令头")
    aliases: list[str] = Field(default_factory=list, description="自然语言别名")
    description: str = Field(default="", description="命令用途")
    requires: dict[str, bool] = Field(default_factory=dict, description="命令级需求")
    slots: list[CommandSlotSpec] = Field(default_factory=list, description="参数槽位")
    render: str = Field(default="", description="命令渲染模板")
    payload_policy: str = Field(default="none", description="负载策略")
    command_role: str = Field(default="execute", description="命令角色")
    source: str = Field(default="fallback", description="schema 来源")
    confidence: float = Field(default=0.5, description="schema 置信度")
    intent_types: list[str] = Field(default_factory=list, description="通用能力类型")
    requires_real_result: bool = Field(default=True, description="是否需要真实结果")
    generative: bool = Field(default=False, description="是否为生成/随机类能力")
    execution_policy: str = Field(default="normal", description="执行门槛策略")
    source_of_truth: str = Field(default="plugin_runtime", description="真实结果来源")
    requires_real_tool: bool = Field(default=True, description="是否必须真实工具")
    output_mode: str = Field(default="plugin_output", description="主要输出形态")
    entity_scope: str = Field(default="global", description="作用实体范围")
    risk: str = Field(default="low", description="风险级别")
    reliability: float = Field(default=0.5, description="能力可靠性估计")
    schema_quality: float = Field(default=0.5, description="schema 完整度估计")
    soft_tool: bool = Field(default=False, description="是否为低上下文软工具")
    features: CommandCandidateFeatures = Field(
        default_factory=CommandCandidateFeatures,
        description="候选可解释特征",
    )


class CapabilityGraphSnapshot(BaseModel):
    """一次插件能力图快照"""

    version: str = Field(default="chatinter.capability_graph.v1")
    plugins: list[PluginCapability] = Field(default_factory=list)
    user_role: str = Field(default="普通用户")
    created_at: float = Field(default=0.0)
