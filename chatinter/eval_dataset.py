"""Fixed layered eval dataset for ChatInter trajectory regression.

The dataset is intentionally product-level, not runtime policy.  It gives the
Eval Harness stable cases and thresholds without adding plugin-specific routing
logic to ChatInter itself.
"""

from __future__ import annotations

from typing import Any

DATASET_SCHEMA_VERSION = "chatinter.eval_dataset.v1"
THRESHOLD_SCHEMA_VERSION = "chatinter.eval_thresholds.v1"

DEFAULT_EVAL_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "tool-required-sign-001",
        "layer": "real_tool_required",
        "input_message": "真寻帮我签个到",
        "expectation": "must_call_tool",
        "scenario": "group_plugin_selector",
        "min_tool_calls": 1,
        "tags": ["plugin", "state", "clear_intent"],
    },
    {
        "id": "tool-required-thirdparty-001",
        "layer": "real_tool_required",
        "input_message": "查一下今天有什么适合摸鱼的二次元新闻，来一条摘要",
        "expectation": "must_call_tool",
        "scenario": "group_plugin_selector",
        "min_tool_calls": 1,
        "tags": ["plugin", "third_party", "query", "external_state"],
    },
    {
        "id": "tool-required-profile-001",
        "layer": "real_tool_required",
        "input_message": "看一下我的信息和金币情况",
        "expectation": "must_call_tool",
        "scenario": "group_plugin_selector",
        "min_tool_calls": 1,
        "tags": ["plugin", "bot_state", "self_scope"],
    },
    {
        "id": "tool-required-translate-001",
        "layer": "real_tool_required",
        "input_message": "把 Good morning everyone 翻译成中文",
        "expectation": "must_call_tool",
        "scenario": "group_plugin_selector",
        "min_tool_calls": 1,
        "tags": ["plugin", "transform", "text_slot"],
    },
    {
        "id": "tool-required-random-001",
        "layer": "real_tool_required",
        "input_message": "帮我抽一张塔罗牌看看今天适合做什么",
        "expectation": "must_call_tool",
        "scenario": "group_plugin_selector",
        "min_tool_calls": 1,
        "tags": ["plugin", "random", "third_party"],
    },
    {
        "id": "tool-required-meme-001",
        "layer": "real_tool_required",
        "input_message": "做个番茄的敲表情",
        "expectation": "must_call_tool",
        "scenario": "group_plugin_selector",
        "min_tool_calls": 1,
        "tags": ["plugin", "meme", "target_text"],
    },
    {
        "id": "tool-required-media-001",
        "layer": "real_tool_required",
        "input_message": "把回复的图片做成表情包，文字写今天也要努力",
        "expectation": "must_call_tool",
        "scenario": "group_plugin_selector",
        "min_tool_calls": 1,
        "tags": ["plugin", "meme", "reply_image"],
    },
    {
        "id": "tool-required-query-001",
        "layer": "real_tool_required",
        "input_message": "查一下这首歌的网易云热评，随便来一条",
        "expectation": "must_call_tool",
        "scenario": "group_plugin_selector",
        "min_tool_calls": 1,
        "tags": ["plugin", "third_party", "query"],
    },
    {
        "id": "tool-required-status-001",
        "layer": "real_tool_required",
        "input_message": "看一下今天的小猪状态",
        "expectation": "must_call_tool",
        "scenario": "group_plugin_selector",
        "min_tool_calls": 1,
        "tags": ["plugin", "third_party", "status"],
    },
    {
        "id": "tool-required-thirdparty-002",
        "layer": "real_tool_required",
        "input_message": "搜一下明日方舟今天有什么活动公告，给我一句话总结",
        "expectation": "must_call_tool",
        "scenario": "group_plugin_selector",
        "min_tool_calls": 1,
        "tags": ["plugin", "third_party", "query", "external_state"],
    },
    {
        "id": "tool-required-thirdparty-003",
        "layer": "real_tool_required",
        "input_message": "帮我随机一张二次元图片看看",
        "expectation": "must_call_tool",
        "scenario": "group_plugin_selector",
        "min_tool_calls": 1,
        "tags": ["plugin", "third_party", "image_output"],
    },
    {
        "id": "tool-required-target-001",
        "layer": "real_tool_required",
        "input_message": "敲一下小明的头像",
        "expectation": "must_call_tool",
        "scenario": "group_plugin_selector",
        "min_tool_calls": 1,
        "tags": ["plugin", "meme", "target_text", "entity_scope"],
    },
    {
        "id": "direct-chat-001",
        "layer": "direct_chat",
        "input_message": (
            "真寻，今天写代码写到一半突然卡住了，你觉得我该先休息还是继续啃？"
        ),
        "expectation": "direct_chat",
        "scenario": "group_plugin_selector",
        "tags": ["chat", "advice"],
    },
    {
        "id": "direct-chat-002",
        "layer": "direct_chat",
        "input_message": "我们只是讨论一下签到这个词为什么像打卡，不要真的签到",
        "expectation": "direct_chat",
        "scenario": "group_plugin_selector",
        "tags": ["chat", "negative_tool_intent"],
    },
    {
        "id": "direct-chat-003",
        "layer": "direct_chat",
        "input_message": "解释一下工具调用和普通聊天的边界是什么",
        "expectation": "direct_chat",
        "scenario": "group_plugin_selector",
        "tags": ["chat", "meta"],
    },
    {
        "id": "direct-chat-004",
        "layer": "direct_chat",
        "input_message": (
            "如果一个插件名字叫塔罗，那我们聊塔罗文化时是不是不该触发插件？"
        ),
        "expectation": "direct_chat",
        "scenario": "group_plugin_selector",
        "tags": ["chat", "plugin_mention_only"],
    },
    {
        "id": "direct-chat-005",
        "layer": "direct_chat",
        "input_message": "我说的抽卡只是比喻，今天选择太多了，有点纠结",
        "expectation": "direct_chat",
        "scenario": "group_plugin_selector",
        "tags": ["chat", "metaphor", "negative_tool_intent"],
    },
    {
        "id": "direct-chat-006",
        "layer": "direct_chat",
        "input_message": "你觉得表情包为什么在群里比文字更容易缓和气氛？",
        "expectation": "direct_chat",
        "scenario": "group_plugin_selector",
        "tags": ["chat", "plugin_concept", "negative_tool_intent"],
    },
    {
        "id": "no-tool-001",
        "layer": "no_tool_available",
        "input_message": (
            "帮我调用一个不存在的月球天气占卜插件，如果没有就说明没有可用工具"
        ),
        "expectation": "no_tool_available",
        "scenario": "group_plugin_selector",
        "tags": ["unsupported", "catalog"],
    },
    {
        "id": "no-tool-002",
        "layer": "no_tool_available",
        "input_message": "用插件帮我查询我从未绑定过的虚构系统编号 ZX-UNKNOWN-404",
        "expectation": "no_tool_available",
        "scenario": "group_plugin_selector",
        "tags": ["unsupported", "missing_context"],
    },
    {
        "id": "no-tool-003",
        "layer": "no_tool_available",
        "input_message": "用一个能直接读取我梦境记录的插件分析昨晚梦境，如果没有就别编",
        "expectation": "no_tool_available",
        "scenario": "group_plugin_selector",
        "tags": ["unsupported", "anti_hallucination"],
    },
    {
        "id": "multi-tool-001",
        "layer": "multi_tool",
        "input_message": "真寻帮我签个到，然后看一下我的信息，最后抽一张塔罗牌",
        "expectation": "multi_tool",
        "scenario": "group_plugin_selector",
        "min_tool_calls": 3,
        "tags": ["multi", "state", "random"],
    },
    {
        "id": "multi-tool-002",
        "layer": "multi_tool",
        "input_message": (
            "先把 hello bot 翻译成中文，再来一条网易云热评，最后看一下今天的小猪"
        ),
        "expectation": "multi_tool",
        "scenario": "group_plugin_selector",
        "min_tool_calls": 3,
        "tags": ["multi", "third_party", "text_slot"],
    },
    {
        "id": "multi-tool-003",
        "layer": "multi_tool",
        "input_message": "做个番茄的敲表情，然后抽张塔罗牌，再帮我查一下我的信息",
        "expectation": "multi_tool",
        "scenario": "group_plugin_selector",
        "min_tool_calls": 3,
        "tags": ["multi", "meme", "state"],
    },
    {
        "id": "multi-tool-004",
        "layer": "multi_tool",
        "input_message": "帮我随机一张二次元图，再翻译 good night，最后查一下我的金币",
        "expectation": "multi_tool",
        "scenario": "group_plugin_selector",
        "min_tool_calls": 3,
        "tags": ["multi", "third_party", "image_output", "state"],
    },
    {
        "id": "native-cont-001",
        "layer": "native_continuation",
        "input_message": "签到，然后再帮我抽一张塔罗牌",
        "expectation": "native_continuation",
        "scenario": "group_plugin_selector",
        "min_tool_calls": 1,
        "tags": ["native_first", "continuation"],
    },
    {
        "id": "native-cont-002",
        "layer": "native_continuation",
        "input_message": "我的信息，最后再来一条网易云热评",
        "expectation": "native_continuation",
        "scenario": "group_plugin_selector",
        "min_tool_calls": 1,
        "tags": ["native_first", "continuation"],
    },
    {
        "id": "native-cont-003",
        "layer": "native_continuation",
        "input_message": "抽塔罗，顺便看一下我的信息",
        "expectation": "native_continuation",
        "scenario": "group_plugin_selector",
        "min_tool_calls": 1,
        "tags": ["native_first", "continuation", "state"],
    },
    {
        "id": "superuser-001",
        "layer": "superuser_long_task",
        "input_message": "列出当前项目根目录的关键文件，并总结这个项目大概是做什么的",
        "expectation": "superuser_task",
        "scenario": "superuser_agent",
        "min_tool_calls": 1,
        "allow_paused": True,
        "tags": ["superuser", "file", "read_only"],
    },
    {
        "id": "superuser-002",
        "layer": "superuser_long_task",
        "input_message": (
            "读取 chatinter 的主要入口文件，判断当前 AgentRuntime 的主链路是否清晰"
        ),
        "expectation": "superuser_task",
        "scenario": "superuser_agent",
        "min_tool_calls": 1,
        "allow_paused": True,
        "tags": ["superuser", "code_review", "read_only"],
    },
    {
        "id": "superuser-003",
        "layer": "superuser_long_task",
        "input_message": (
            "创建一个最小 echo 插件草稿，跑一次编译检查，如果失败给出修复建议"
        ),
        "expectation": "superuser_task",
        "scenario": "superuser_agent",
        "min_tool_calls": 3,
        "allow_paused": True,
        "tags": ["superuser", "plugin_dev", "patch", "eval"],
    },
    {
        "id": "superuser-004",
        "layer": "superuser_long_task",
        "input_message": (
            "检查当前工作区是否有未提交改动，如果有只总结风险，不要修改文件"
        ),
        "expectation": "superuser_task",
        "scenario": "superuser_agent",
        "min_tool_calls": 1,
        "allow_paused": True,
        "tags": ["superuser", "git", "read_only", "dirty_lock"],
    },
    {
        "id": "superuser-005",
        "layer": "superuser_long_task",
        "input_message": (
            "走工程闭环检查一个小改动方案：先读代码，再给 patch/eval/rollback 计划，"
            "不要直接改"
        ),
        "expectation": "superuser_task",
        "scenario": "superuser_agent",
        "min_tool_calls": 2,
        "allow_paused": True,
        "tags": ["superuser", "engineering_loop", "plan_only"],
    },
)

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "schema_version": THRESHOLD_SCHEMA_VERSION,
    "global": {
        "min_case_coverage": 0.70,
        "min_pass_rate": 0.82,
        "max_false_trigger_rate": 0.08,
        "max_avg_latency_ms": 18000,
        "max_p95_latency_ms": 45000,
        "max_avg_prompt_tokens": 18000,
        "max_avg_steps": 6.0,
        "max_tool_call_pressure": 4.5,
        "max_over_tooling_rate": 0.18,
    },
    "layers": {
        "real_tool_required": {
            "min_case_coverage": 0.70,
            "min_pass_rate": 0.86,
            "min_hit_rate": 0.86,
            "min_retrieval_hit_rate": 0.80,
            "max_avg_latency_ms": 16000,
            "max_p95_latency_ms": 36000,
        },
        "direct_chat": {
            "min_case_coverage": 0.70,
            "min_pass_rate": 0.90,
            "max_false_trigger_rate": 0.05,
            "max_tool_call_pressure": 0.20,
            "max_avg_latency_ms": 9000,
        },
        "no_tool_available": {
            "min_case_coverage": 0.60,
            "min_pass_rate": 0.80,
            "max_false_trigger_rate": 0.10,
        },
        "multi_tool": {
            "min_case_coverage": 0.60,
            "min_pass_rate": 0.78,
            "min_multi_coverage_rate": 0.75,
            "min_task_coverage_rate": 0.75,
            "max_tool_call_pressure": 6.0,
            "max_avg_latency_ms": 28000,
            "max_avg_steps": 8.0,
        },
        "native_continuation": {
            "min_case_coverage": 0.60,
            "min_pass_rate": 0.75,
            "max_false_trigger_rate": 0.10,
        },
        "superuser_long_task": {
            "min_case_coverage": 0.50,
            "min_pass_rate": 0.72,
            "min_superuser_completion_or_pause_rate": 0.85,
            "max_tool_call_pressure": 8.0,
            "max_avg_latency_ms": 45000,
            "max_p95_latency_ms": 90000,
            "max_avg_prompt_tokens": 26000,
            "max_avg_steps": 12.0,
        },
    },
}

__all__ = [
    "DATASET_SCHEMA_VERSION",
    "DEFAULT_EVAL_CASES",
    "DEFAULT_THRESHOLDS",
    "THRESHOLD_SCHEMA_VERSION",
]
