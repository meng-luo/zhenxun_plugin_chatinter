# ChatInter

**ChatInter** 是一个基于 AI 意图识别的智能对话插件，为 [真寻Bot](https://github.com/zhenxun-org/zhenxun_bot) 提供强大的对话能力。

当用户消息未被其他插件匹配时，ChatInter 使用大语言模型分析用户意图，实现：
- **功能调用意图** → 自动重路由到对应插件
- **普通聊天意图** → 进行自然对话回复

> [!WARNING]
>
> 由于上下文包含了插件的帮助信息，导致消耗的 tokens 会随着插件的数量增加而增加

## ✨ 特性

- 🤖 **AI 意图识别** - 使用 LLM 精准分析用户真实意图
- 🔀 **智能重路由** - 识别插件调用命令时自动转发到对应插件
- 💬 **自然对话** - 支持多轮对话，保持上下文连贯性
- 🖼️ **多模态支持** - 支持图片识别和理解
- 🧠 **聊天记忆** - 持久化存储对话历史，支持语境构建
- 🔧 **Agent 框架** - 支持 Tool Calling 和 MCP 工具集成
- 📚 **知识库 RAG** - 支持知识检索增强生成
- 🛡️ **安全沙箱** - 提供安全的表达式求值和 Shell 命令执行

## 📦 插件结构

```text
chatinter/
├── __init__.py              # 插件入口、matcher 与超级用户命令
├── config.py                # 配置读取与推理配置构建
├── handler.py               # 主处理流程与意图分发
├── chat_handler.py          # 对话回复生成
├── memory.py                # 会话记忆与上下文构建
├── intent_classifier.py     # 意图分类
├── route_engine.py          # 路由引擎
├── route_tool_planner.py    # 工具/命令规划
├── route_policy.py          # 路由策略约束
├── skill_registry.py        # 技能注册与命令归一化
├── plugin_registry.py       # 插件发现与缓存
├── agent_gate.py            # Agent 启用判定
├── agent_runner.py          # Agent 工具调用循环
├── tool_registry.py         # Tool 注册与暴露
├── tool_orchestration.py    # 工具编排
├── subagent_handoff.py      # 子代理交接逻辑
├── knowledge_rag.py         # 知识检索服务
├── retrieval.py             # 历史召回与检索辅助
├── prompt_guard.py          # Prompt 安全护栏
├── lifecycle.py             # 生命周期钩子
├── trace.py                 # 追踪与调试
├── turn_metrics.py          # 路由统计输出
├── models/
│   ├── chat_history.py      # 对话历史表
│   └── pydantic_models.py   # 结构化数据模型
└── utils/
    ├── multimodal.py        # 多模态输入处理
    └── unimsg_utils.py      # UniMessage 工具函数
```

## 🔧 配置项

在机器人配置文件中添加以下配置：

| 配置键 | 说明 | 默认值 | 类型 |
|--------|------|--------|------|
| `ENABLE_FALLBACK` | 是否启用 ChatInter 兜底对话能力 | `True` | bool |
| `ENABLE_AGENT_MODE` | 是否启用 ChatInter Agent（工具调用）模式 | `True` | bool |
| `INTENT_TIMEOUT` | ChatInter 推理超时时间（秒），<=0 时复用 AI.CLIENT_SETTINGS.timeout | `20` | int |
| `AGENT_MAX_TOOL_STEPS` | Agent 工具调用最大迭代步数（复杂请求会自动小幅上调） | `4` | int |
| `AGENT_TOOL_FAILURE_LIMIT` | 单工具连续失败达到阈值后自动熔断 | `2` | int |
| `AGENT_FAILED_ROUND_LIMIT` | 连续失败回合阈值，达到后直接总结 | `2` | int |
| `CHAT_STYLE` | ChatInter 对话风格补充设定，留空使用默认风格 | `""` | str |
| `CUSTOM_PROMPT` | ChatInter 自定义系统提示词补充，会追加到系统提示词末尾 | `""` | str |
| `MCP_ENDPOINTS` | MCP 工具服务地址列表，使用英文逗号分隔 | `""` | str |
| `REASONING_EFFORT` | 强制推理强度，可选 `MEDIUM` 或 `HIGH`，留空表示不强制设置 | `"MEDIUM"` | str |

> 说明：其余路由、上下文和历史召回阈值目前为插件内固定策略参数，不作为外部配置项暴露。

### 配置示例

```yaml
# configs.yml
chatinter:
  ENABLE_FALLBACK: true
  ENABLE_AGENT_MODE: true
  INTENT_TIMEOUT: 20
  AGENT_MAX_TOOL_STEPS: 4
  AGENT_TOOL_FAILURE_LIMIT: 2
  AGENT_FAILED_ROUND_LIMIT: 2
  CHAT_STYLE: "活泼可爱"
  CUSTOM_PROMPT: ""
  MCP_ENDPOINTS: "http://127.0.0.1:9001,http://127.0.0.1:9002"
  REASONING_EFFORT: "MEDIUM"
```

## 📖 使用方法

### 自动加载

插件会自动加载，无需手动操作。当消息满足以下条件时会被处理：

1. 消息 `@` 了机器人
2. 消息未被其他高优先级插件处理
3. 私聊场景下当前消息为纯文本（回复消息允许，图片等非文本内容会被跳过）

### 超级用户命令

```text
重置会话        # 重置当前会话历史
chatinter统计   # 查看最近路由统计
```

## 🏗️ 工作流程

```text
用户消息
  → ChatInter 接管兜底消息
  → 意图分析与路由候选筛选
  → 按策略决定：插件命令 / 帮助 / 普通对话
      ├── 插件命令：重写消息并交还给目标插件执行
      └── 普通对话：构建上下文 → Agent / 普通聊天生成 → 保存记忆 → 返回回复
```

### Agent 能力

ChatInter 当前支持：

- **Tool Calling** - LLM 可调用工具获取实时信息
- **插件查询** - 动态检索可用插件和命令
- **MCP 集成** - 支持外部 MCP 工具服务
- **失败熔断** - 工具连续失败后自动停用并总结
- **子代理分流** - 针对复杂任务自动切换到子代理工作流

### 多模态支持

- 支持识别消息中的图片
- 支持识别回复链中的图片
- 图片会作为多模态输入传递给 LLM

### 待补全机制

当用户发送的消息缺少必要信息时（如需要图片但未提供，或执行命令缺少关键参数），ChatInter 会提示用户补充并等待后续消息。

## 🗄️ 数据库

ChatInter 使用 `ChatInterChatHistory` 模型持久化存储对话历史：

```python
class ChatInterChatHistory(Model):
    id = fields.IntField(pk=True, generated=True, auto_increment=True)
    session_id = fields.CharField(255, index=True)
    user_id = fields.CharField(255)
    group_id = fields.CharField(255, null=True)
    nickname = fields.CharField(255)
    user_message = fields.TextField()
    ai_response = fields.TextField(null=True)
    bot_id = fields.CharField(255, null=True)
    create_time = fields.DatetimeField(auto_now_add=True, index=True)
    reset = fields.BooleanField(default=False, index=True)
```

## 🎨 效果图

![example](docs_image/1.png)

## 🚀 更新日志

### v1.3.0

- 重构插件路由与命令抽取逻辑，增强意图识别稳定性
- 新增插件调用对齐层，统一工具暴露与路由行为
- 优化对话场景下的工具接入策略，减少幻觉与误触发
- 兼容新版本体移除 `GroupInfoUser.nickname` 字段，避免运行时报错

### v1.2.1

- 修复部分场景下 @ 机器人识别问题
- 统一 @ 逻辑处理流程
- 新增 schema_policy.py 用于模型输出格式策略管理

### v1.2.0

- 新增向量化历史记忆召回，智能检索相关对话片段
- 新增上下文关联门控，新话题自动隔离历史上下文
- 新增群聊背景相关度补充，提升群聊响应准确性
- 新增 Agent 框架精细化配置（工具步数、超时、熔断机制）
- 新增复杂问题自动放宽对话长度限制
- 优化路由引擎，减小路由污染
- 修复意图分析失败时的 pydantic 验证错误，优雅降级

### v1.1.0

- 新增 Agent 框架，支持 Tool Calling
- 新增 MCP 工具服务集成
- 新增知识库 RAG 检索
- 新增安全沙箱执行环境
- 新增待补全机制（图片、@目标）
- 优化路由引擎，提升意图识别准确率
- 配置项重构，简化配置流程

### v1.0.0

- 初始版本
- 基础意图识别和重路由
- 多轮对话支持
- 多模态支持

## 📄 许可证

本项目采用 [AGPL-3.0](./LICENSE) 许可证。

## 🙏 致谢

- [绪山真寻 Bot](https://github.com/zhenxun-org/zhenxun_bot)
- [BYM AI 插件](https://github.com/zhenxun-org/zhenxun_bot_plugins/tree/main/plugins/bym_ai)
- Copaan - Agent 框架贡献
- 万能的 AI sama

## 📧 联系方式

如有问题或建议，请提交 Issue。