# ChatInter

ChatInter 是面向 [真寻 Bot](https://github.com/zhenxun-org/zhenxun_bot) 的 AI 对话与插件调度插件。它在消息未被其他插件处理时接管请求，通过统一的混合聊天 Agent 完成对话、原生插件调用、联网搜索和本地表情回复，并为超级用户提供独立的自动化 Agent。

> [!IMPORTANT]
> ChatInter 依赖真寻 Bot 的 AI 服务和默认聊天模型。插件调度、Superuser Agent 及其他工具能力要求模型和 Provider 正确支持 Tool Calling；图片理解还要求模型支持多模态输入。

## 运行场景

ChatInter 将混合聊天与高权限 Agent 隔离，并按消息场景控制工具范围：

| 场景 | 触发方式 | 可用能力 |
| --- | --- | --- |
| 群聊混合聊天 | 在已启用群聊中 `@Bot`，且消息未被其他插件处理 | 对话、记忆、图片理解、公开插件检索与调用；可选联网搜索和本地表情 |
| 普通私聊 | 私聊 Bot，且 Superuser Agent 未开启 | 对话、记忆、图片理解；默认可调用用户有权使用的插件，可通过配置关闭 |
| Superuser Agent | 超级用户私聊执行 `/开启agent` 后发送任务 | 文件与 Shell 工具、网页读取、产物、执行计划、只读子 Agent、主动任务、审批和会话管理 |

混合聊天只会暴露当前用户有权使用的公开命令。管理员、超级用户专用、受限、已禁用或当前场景不可用的插件不会进入普通用户的候选集。

## 主要能力

- 使用统一混合聊天 Agent，在一次工具循环中完成对话、候选检索和插件调用
- 从插件元数据、Alconna 命令和 matcher 构建稀疏检索索引，按需暴露命令 Schema
- 识别文本、图片、回复、`@` 对象和昵称目标，执行前校验参数与场景要求
- 群聊、私聊和 Superuser Agent 使用独立的工具边界、并发门控与上下文策略
- 持久化完整消息时间线，并进行会话记忆、人物信息和相关历史召回
- 合并同一会话内的连续消息，避免多个耗时请求并发覆盖回复
- 根据 Provider 能力调整工具 Schema、工具选择模式与多模态输入
- 支持主模型失败重试和备用模型切换
- 可按群聊范围启用，支持整段或分段回复、触发消息引用和前缀静默跳过
- 可选多搜索 Provider、GScore 原生命令桥接和本地表情语义检索
- 提供路由、插件执行、轨迹和反思统计，以及可复现的离线评测数据
- Superuser Agent 提供三种权限模式、操作审批、网页读取、并行只读调查和跨重启主动任务

## 安装

1. 将仓库中的 `chatinter/` 目录放入真寻 Bot 的插件目录。
2. 安装附加依赖：

   ```bash
   pip install -r chatinter/requirements.txt
   ```

3. 在真寻 Bot 配置中设置 `AI.DEFAULT_MODELS.chat`，或在 ChatInter 的 `AGENTS` 配置中为每类 Agent 指定模型。
4. 启动 Bot。日志出现 `ChatInter 插件已加载` 后，插件会预热插件知识库并在启动后补扫动态 matcher。

如需调用 GScore 插件，将仓库中的 `ChatInterBridge/` 安装到 GScore 插件目录，在 GScore 管理界面设置 `shared_secret`，再配置下方的 `GSCORE_BRIDGE`。两端密钥必须一致；不使用 GScore 时无需安装 Bridge。

## 配置

ChatInter 会注册以下配置项：

| 配置键 | 说明 | 默认值 |
| --- | --- | --- |
| `ENABLED` | 是否启用 ChatInter | `true` |
| `GROUP_ACCESS` | 群聊白名单或黑名单及群号列表 | 黑名单模式，不排除任何群 |
| `AGENTS` | 三类 Agent 的模型、上下文窗口、最大输出和推理强度 | 见下方示例 |
| `FALLBACK_MODELS` | 主模型不可用时依次尝试的备用模型 | `[]` |
| `PERMISSIONS` | Superuser Agent 的权限预设、默认模式和危险操作策略 | 见下方示例 |
| `ACTIVE_TASKS_ENABLED` | 是否允许 Superuser Agent 创建定时、间隔或 Webhook 主动任务 | `true` |
| `PRIVATE_PLUGIN_TOOLS` | 普通私聊是否可调用插件 | `true` |
| `REPLY_TO_TRIGGER_MESSAGE` | 最终回复首段是否引用触发消息 | `false` |
| `REPLY_DELIVERY` | 回复分段、长度、段数和发送间隔 | 见下方示例 |
| `REACTION_IMAGES` | 本地表情库、语义检索、自动描述和群聊发现 | 关闭 |
| `CHAT_HISTORY_LIMIT` | 每个混合聊天会话保留的最近对话数 | `100` |
| `MIXED_CHAT_SKIP_PREFIXES` | 命中后静默跳过的消息前缀 | `[]` |
| `UNIFIED_MAX_TOOL_STEPS` | 混合聊天单回合最大工具循环数 | `4` |
| `WEB_ACCESS_MODE` | 联网能力范围：`off`、`agent` 或 `all` | `agent` |
| `CHAT_WEB_SEARCH_*` | 混合聊天的搜索 Provider、端点和 API Key | 百度协议，未配置 Key |
| `GSCORE_BRIDGE` | GScore Bridge 地址和签名密钥 | 关闭 |

完整配置示例：

```yaml
# data/config.yaml
chatinter:
  ENABLED: true

  GROUP_ACCESS:
    mode: blacklist
    enabled_groups: []
    disabled_groups: []

  AGENTS:
    chat:
      model: DEFAULT_MODELS
      context_window_tokens: 128000
      max_output_tokens: 8192
      reasoning_effort: MEDIUM
    plugin:
      model: DEFAULT_MODELS
      context_window_tokens: 16000
      max_output_tokens: 2048
      reasoning_effort: MEDIUM
    superuser:
      model: DEFAULT_MODELS
      context_window_tokens: 200000
      max_output_tokens: 32000
      reasoning_effort: HIGH

  FALLBACK_MODELS: []

  PERMISSIONS:
    preset: python
    default_mode: ask
    dangerous_policy: ask
    allow: []
    ask: []
    dangerous: []
    deny: []

  ACTIVE_TASKS_ENABLED: true
  PRIVATE_PLUGIN_TOOLS: true
  REPLY_TO_TRIGGER_MESSAGE: false

  REPLY_DELIVERY:
    mode: streaming
    max_chars: 3500
    max_segments: 6
    interval_method: random
    interval: "1.5,3.5"
    log_base: 2.6

  REACTION_IMAGES:
    enabled: false
    directory: data/chatinter/reactions
    import_directory: data/chatinter/reaction_import
    semantic_search: true
    auto_caption: true
    auto_discovery: false

  CHAT_HISTORY_LIMIT: 100
  MIXED_CHAT_SKIP_PREFIXES: []
  UNIFIED_MAX_TOOL_STEPS: 4

  WEB_ACCESS_MODE: agent
  CHAT_WEB_SEARCH_ENABLED: true
  CHAT_WEB_SEARCH_PROVIDER: baidu
  CHAT_WEB_SEARCH_API_URL: DEFAULT
  CHAT_WEB_SEARCH_API_KEY: ""

  GSCORE_BRIDGE:
    enabled: false
    url: ""
    secret: ""
```

### Agent 配置说明

- `model`：模型名称。`DEFAULT_MODELS` 表示读取真寻 Bot 的 `AI.DEFAULT_MODELS.chat`。
- `context_window_tokens`：ChatInter 允许使用的输入窗口上限；若模型声明的窗口更小，以较小值为准。
- `max_output_tokens`：单次模型回复的最大输出 token 数。
- `reasoning_effort`：支持 `DEFAULT`、`NONE`、`MINIMAL`、`LOW`、`MEDIUM`、`HIGH`、`XHIGH`、`MAX`。Provider 不支持的等级可能会被其适配层忽略或降级。
- `FALLBACK_MODELS`：填写模型名称列表。瞬时错误会先有限重试，符合切换条件时再按列表顺序降级。

`chat` 用于群聊和普通私聊的统一混合 Agent，`plugin` 用于内部历史摘要等辅助任务，`superuser` 用于超级用户 Agent。

### 场景与扩展配置

- `GROUP_ACCESS.mode` 为 `blacklist` 时默认启用未列入 `disabled_groups` 的群；为 `whitelist` 时只启用 `enabled_groups`。同一群号同时出现时禁用优先。
- `PRIVATE_PLUGIN_TOOLS=false` 可将普通私聊限制为无插件对话；不会影响 Superuser Agent。
- `REPLY_DELIVERY.mode` 支持 `streaming` 和 `whole`。`streaming` 是生成完成后按句分段发送，并非模型流式输出。
- `WEB_ACCESS_MODE=agent` 只允许 Superuser Agent 联网；设为 `all` 后，混合聊天可使用模型原生搜索，或在配置 API Key 后使用本地搜索回退。
- `CHAT_WEB_SEARCH_PROVIDER` 支持 `baidu`、`tavily`、`bocha`、`brave`、`firecrawl` 和 `exa`；`CHAT_WEB_SEARCH_API_URL=DEFAULT` 使用对应官方端点。
- `REACTION_IMAGES.enabled=true` 后启用本地表情检索。待导入图片放入 `import_directory`；`auto_discovery` 会从群聊重复图片中发现候选，建议确认存储和隐私策略后再开启。
- `GSCORE_BRIDGE.url` 留空时会尝试从真寻 Bot 的 `gsuid_core_host`、`gsuid_core_port` 和 `gsuid_core_https` 推导地址。

### 权限配置说明

- `preset`：内置权限预设，当前支持 `python` 和 `none`。
- `default_mode`：新 Superuser Agent 会话的权限模式，可选 `ask`、`read_only`、`full_access`。
- `dangerous_policy`：危险操作命中后使用 `ask` 请求确认，或使用 `deny` 直接拒绝。
- `allow`、`ask`、`dangerous`、`deny`：附加匹配规则，例如 `Shell(pytest*)` 或 `File(@workspace/**)`。

在 `ask` 和 `read_only` 模式下，默认策略会拒绝凭据目录、`.env`、`.git`、云服务配置等敏感路径，并拦截关机、格式化磁盘等破坏性命令。`full_access` 会绕过这些权限规则，只应在完全信任当前任务和运行环境时临时启用。

## 使用

### 群聊与私聊

- 群聊中 `@Bot` 后直接描述需求，例如“帮我查一下上海明天的天气”。
- 如果插件命令缺少图片、目标用户或必要参数，ChatInter 会返回明确的补充提示。
- 普通私聊直接发送文本、图片或回复消息即可；默认同样可以自然语言调用插件。
- 当前 turn 正在处理时，紧接着发送的短补充消息会并入同一会话队列。

私聊只接受文本、图片和回复消息段；包含其他不支持消息段的请求会被跳过。

### 管理命令

以下命令仅限超级用户，且需要在对应场景中发送：

| 命令 | 作用 |
| --- | --- |
| `重置会话` | 将当前群聊或私聊的 ChatInter 历史标记为已重置 |
| `chatinter统计` | 查看最近的路由、插件执行和反思统计 |
| `重建插件索引` | 强制重新扫描插件并重建知识库索引 |

### Superuser Agent

Superuser Agent 仅在超级用户私聊中可用。先发送 `/开启agent`，之后直接描述任务；发送 `/退出agent` 会退出 Agent 模式但保留会话。Agent 可维护结构化执行计划，复杂只读调查可并行委派两个子任务；当 `ACTIVE_TASKS_ENABLED=true` 时，还可用自然语言创建、查看和管理一次性、Cron、间隔或 Webhook 主动任务。

| 分类 | 命令 |
| --- | --- |
| Agent | `/开启agent`、`/退出agent`、`/agent帮助`、`/状态`、`/中断` |
| 上下文 | `/清除上下文`、`/压缩上下文` |
| 权限 | `/请求批准模式`、`/只读模式`、`/完全访问模式` |
| 会话 | `/新增会话 [名称]`、`/当前会话`、`/列出会话`、`/切换会话 [ID/名称]`、`/重命名会话 ID/名称 新名称` |
| 归档 | `/归档会话 [ID/名称]`、`/列出归档会话`、`/恢复会话 ID/名称`、`/删除会话 ID/名称` |
| 审批 | `/允许`、`/本对话允许`、`/拒绝 [理由]`、`/中断` |

`/允许` 只批准当前待执行操作；`/本对话允许` 会在当前会话内记住同一权限范围。任务执行期间不能切换、清除、归档或删除当前会话，请先使用 `/中断`。主动任务会持久化并可能在 Bot 重启后恢复，其中 Agent 任务按完全访问模式执行；创建前应仔细核对 Agent 给出的审批摘要。

> [!NOTE]
> 仓库包含 MCP 连接与 Provider 适配模块，但当前 Superuser Agent 固定工具集不会自动加载 `MCP_SERVERS`。因此 README 暂不将 MCP 配置列为可用功能。

## 工作流程

```text
消息进入
  -> 已由其他插件处理？是：跳过
  -> 校验全局开关、群聊范围和静默前缀
  -> 解析场景并取得会话执行权
     |-- 群聊/普通私聊：构建身份、记忆、图片和人物上下文
     |                    -> 检索并按需暴露插件/GScore/搜索/表情工具
     |                    -> 统一 Agent 对话或执行工具
     `-- Superuser Agent：恢复会话 -> 计划/工具/审批/主动任务 -> 保存结果
  -> 持久化时间线、轨迹与反馈
  -> 按配置发送整段或分段回复
```

## 项目结构

```text
chatinter/
├── plugin_entry.py              # NoneBot 入口、matcher、管理命令与生命周期
├── config.py                    # 注册配置、模型与权限配置读取
├── handler.py                   # fallback 入口与 TurnFrame 构建
├── scenario_router.py           # 群聊、私聊、Superuser Agent 场景隔离
├── prompt_pipeline.py           # 对话处理流水线
├── pipeline_stages.py           # 身份、上下文、记忆、生成、持久化阶段
├── unified_flow.py              # 群聊与私聊的统一 Agent 工具循环
├── command_index.py             # 插件命令稀疏索引与候选召回
├── capability_registry.py       # 命令与工具能力注册
├── mixed_tool_catalog.py        # 混合聊天工具目录与按需暴露
├── chat_web_search.py           # 多 Provider 搜索回退
├── gscore_adapter.py            # GScore Bridge 客户端、检索与执行
├── reaction_runtime.py          # 本地表情导入、发现与运行时
├── provider_capability.py       # Provider 能力与 Schema 适配
├── provider_failover.py         # 重试、上下文恢复与备用模型切换
├── turn_queue.py                # 同会话消息排队与连续消息合并
├── memory.py                    # 对话历史与记忆入口
├── models/chat_history.py       # ChatInter 历史数据表
├── agents/
│   ├── unified_chat_agent.py    # 群聊与私聊统一混合 Agent
│   └── superuser_entry.py       # Superuser Agent 直接入口
├── superuser_agent/
│   ├── runtime.py               # Agent 工具调用循环
│   ├── store.py                 # 会话与运行状态
│   ├── permission_policy.py     # 权限模式、规则匹配与会话授权
│   ├── runtime_approval.py      # 待审批操作恢复
│   ├── active_tasks.py          # 持久化主动任务与调度器桥接
│   ├── proactive_tasks.py       # 主动任务执行、通知与 Webhook
│   ├── subagent.py              # 并行只读调查
│   └── tools/                   # 文件、Shell、网页、计划、产物和任务工具
└── utils/
    ├── multimodal.py            # 图片与回复链解析
    └── unimsg_utils.py          # UniMessage 转换

ChatInterBridge/                 # 可选 GScore 插件：能力发现、鉴权与执行
```

## 数据与日志

- 对话历史保存在数据库表 `chatinter_chat_history`，包含兼容摘要字段和完整 `timeline`。
- Superuser Agent 的会话、运行快照、审批、主动任务、轨迹和评测数据保存在 `data/chatinter_agent/`。
- 生成或登记的产物保存在 `data/chatinter_artifacts/`。
- 本地表情默认保存在 `data/chatinter/reactions/`，待导入图片默认放在 `data/chatinter/reaction_import/`。
- Superuser Agent 审计日志保存在 `data/log/chatinter_agent_audit.log`。

`重置会话` 对数据库历史执行软重置，不会直接删除记录。

## v1.5.0

- 将群聊和普通私聊合并为统一混合聊天 Agent，支持按需检索并暴露插件能力，私聊插件调用可配置
- 增加群聊白名单/黑名单、静默前缀、触发消息引用和可配置的分段回复投递
- 增加原生或 API 回退的联网搜索，支持百度、Tavily、博查、Brave、Firecrawl 和 Exa
- 增加本地表情库的导入、语义检索、自动描述、群聊发现和回复去重
- 增加 ChatInterBridge，实现 GScore 能力发现、鉴权路由、幂等执行和投递状态追踪
- 扩展 Superuser Agent：网页读取、结构化计划、并行只读子任务及可跨重启的定时/间隔/Webhook 主动任务
- 重构命令与人物检索、上下文压缩和反馈评测链路，并完善 Provider 兼容与运行观测

## v1.4.0

- 将运行时拆分为插件命令、聊天回复和 Superuser 三类 Agent，收紧跨场景工具暴露
- 重构为基于 `TurnFrame` 的 Prompt Pipeline，并加入同会话 turn 队列
- 增强命令索引、两阶段 Schema 暴露、目标解析、原生插件执行与失败反馈
- 增加 Provider 协议适配、模型窗口约束、错误分类和备用模型切换
- 简化 Superuser Agent 工具集，补充权限模式、操作审批、会话归档与运行恢复
- 完善聊天时间线、长期记忆、人物信息、向量召回和多模态上下文

## 效果图

![ChatInter 使用效果](docs_image/1.png)

## 许可证

本项目采用 [AGPL-3.0](./LICENSE) 许可证。

## 致谢

- [绪山真寻 Bot](https://github.com/zhenxun-org/zhenxun_bot)
- [BYM AI 插件](https://github.com/zhenxun-org/zhenxun_bot_plugins/tree/main/plugins/bym_ai)
- Copaan：Agent 框架贡献

如有问题或建议，请提交 Issue。
