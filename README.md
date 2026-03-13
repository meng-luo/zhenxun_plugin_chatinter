# ChatInter

**ChatInter** 是一个基于 AI 意图识别的智能对话插件，为 [真寻Bot](https://github.com/zhenxun-org/zhenxun_bot) 提供强大的对话能力。

当用户消息未被其他插件匹配时，ChatInter 使用大语言模型分析用户意图，实现：
- **功能调用意图** → 自动重路由到对应插件
- **普通聊天意图** → 进行自然对话回复

> [!WARNING]
>
> 由于上下文包含了插件的帮助信息，导致消耗的tokens会随着插件的数量增加而增加

## ✨ 特性

- 🤖 **AI 意图识别** - 使用 LLM 精准分析用户真实意图
- 🔀 **智能重路由** - 识别插件调用命令时自动转发到对应插件
- 💬 **自然对话** - 支持多轮对话，保持上下文连贯性
- 🖼️ **多模态支持** - 支持图片识别和理解
- 🧠 **聊天记忆** - 持久化存储对话历史，支持语境构建
- 🎯 **好感度系统** - 集成签到好感度，让对话更有温度
- 🛡️ **安全防护** - 死循环检测、消息去重、超时保护

## 📦 插件结构

```
chatinter/
├── __init__.py              # 插件入口，定义响应器和元数据
├── config.py                # 配置项定义和获取
├── memory.py                # 聊天记忆管理（ChatMemory 类）
├── intent_analyzer.py       # 意图分析器（LLM 调用）
├── chat_handler.py          # 聊天响应处理
├── handler.py               # 主处理器（handle_fallback）
├── plugin_registry.py       # 插件信息注册表
├── prompt_templates.py      # Prompt 模板构建
├── data_source.py           # 导出模块（整合各子模块）
├── models/
│   ├── __init__.py          # 模型导出
│   ├── chat_history.py      # 数据库模型
│   └── pydantic_models.py   # Pydantic 结构化模型
└── utils/
    ├── __init__.py          # 工具函数导出
    ├── cache.py             # 缓存工具（好感度缓存）
    ├── multimodal.py        # 多模态处理（图片提取）
    └── unimsg_utils.py      # UniMessage 工具函数
```

## 🔧 配置项

在机器人配置文件中添加以下配置：

| 配置键 | 说明 | 默认值 | 类型 |
|--------|------|--------|------|
| `ENABLE_FALLBACK` | 是否启用 ChatInter 功能 | `True` | bool |
| `INTENT_MODEL` | AI 意图识别使用的模型，具体参考 AI 模块 | `None` | str |
| `INTENT_TIMEOUT` | 意图识别超时时间（秒） | `30` | int |
| `CONFIDENCE_THRESHOLD` | 插件调用置信度阈值，低于此值按普通聊天处理 | `0.7` | float |
| `CHAT_STYLE` | 聊天回复风格（为空时使用默认风格） | `""` | str |
| `USE_SIGN_IN_IMPRESSION` | 是否使用签到好感度 | `True` | bool |
| `CONTEXT_PREFIX_SIZE` | 语境前缀消息数（从数据库加载的最近 N 条群消息） | `5` | int |
| `SESSION_CONTEXT_LIMIT` | 单会话上下文上限（对话历史的最大消息数） | `20` | int |
| `MAX_REPLY_LAYERS` | 回复链追溯最大层数（递归获取被回复消息） | `3` | int |

### 配置示例

```python
# configs.yml
chatinter:
  ENABLE_FALLBACK: true
  INTENT_MODEL: "Gemini/gemini-3-flash-preview"
  INTENT_TIMEOUT: 30
  CONFIDENCE_THRESHOLD: 0.7
  CHAT_STYLE: "活泼可爱"
  USE_SIGN_IN_IMPRESSION: true
  CONTEXT_PREFIX_SIZE: 5
  SESSION_CONTEXT_LIMIT: 20
  MAX_REPLY_LAYERS: 3
```

## 📖 使用方法

### 自动加载

插件会自动加载，无需手动操作。当消息满足以下条件时会被处理：

1. 消息 `@` 了机器人
2. 消息未被其他高优先级插件处理

### 超级用户命令

```
重置会话    # 重置当前会话历史（仅超级用户可用）
```

## 🏗️ 工作流程

```
用户消息 → 意图分析 → 判断意图类型
                     ├── 插件调用 → 重路由到对应插件 → 执行命令 → 保存记忆 → 返回结果
                     └── 普通聊天 → 构建上下文 → LLM 对话 → 保存记忆 → 返回回复
```

### 意图识别流程

1. **消息预处理** - 提取文本和图片，构建上下文
2. **LLM 分析** - 调用大语言模型分析用户意图
3. **意图分类** - 判断是插件调用还是普通聊天
4. **执行响应** - 根据意图类型执行相应操作

### 多模态支持

- 支持识别消息中的图片
- 支持识别回复链中的图片
- 图片会作为多模态输入传递给 LLM

## 🗄️ 数据库

ChatInter 使用 `ChatInterChatHistory` 模型持久化存储对话历史：

```python
class ChatInterChatHistory(Model):
    id: int = Field(pk=True, auto_increment=True)
    user_id: str
    group_id: str | None
    nickname: str
    user_message: str
    ai_response: str
    timestamp: datetime
    bot_id: str | None
    session_reset: bool = False
```

## 🎨 效果图

![example](docs_image/1.png)

## 📄 许可证

本项目采用 [AGPL-3.0](./LICENSE) 许可证。

## 🙏 致谢

- [绪山真寻 Bot](https://github.com/zhenxun-org/zhenxun_bot)
- [BYM AI 插件](https://github.com/zhenxun-org/zhenxun_bot_plugins/tree/main/plugins/bym_ai)
- 万能的 AI sama

## 📧 联系方式

如有问题或建议，请提交 Issue。
