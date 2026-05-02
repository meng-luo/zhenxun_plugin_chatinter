"""Adapters for high-frequency built-in/simple command plugins."""

from __future__ import annotations

import re
from typing import Any

from ..models.pydantic_models import PluginCommandSchema
from ..route_text import normalize_message_text
from ..slot_extractors import parse_int_token
from . import AdapterScoreHint, PluginCommandAdapter, register_adapter, schema, slot

NUMBER_TEXT = r"\d+|[零〇一二两俩三四五六七八九十百千]+"


def _extract_number(pattern: str, text: str, group: str) -> int | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return parse_int_token(match.group(group))


def _split_options(raw: str) -> str:
    options = normalize_message_text(raw)
    if not options:
        return ""
    options = re.sub(r"[、，,/|]+", " ", options)
    options = re.sub(r"\s*(?:和|或|还是)\s*", " ", options)
    options = normalize_message_text(options)
    if " " not in options and "茶" in options:
        tea_items = re.findall(r"[^茶\s]{1,3}茶", options)
        if len(tea_items) >= 2 and "".join(tea_items) == options:
            return " ".join(tea_items)
    if " " not in options and len(options) >= 4 and len(options) % 2 == 0:
        return " ".join(
            options[index : index + 2] for index in range(0, len(options), 2)
        )
    if " " not in options and len(options) <= 8:
        return " ".join(options)
    return options


def _extract_redbag_slots(message_text: str) -> dict[str, Any]:
    text = normalize_message_text(message_text)
    slots: dict[str, Any] = {}
    amount = _extract_number(
        rf"总额\s*(?P<amount>{NUMBER_TEXT})",
        text,
        "amount",
    )
    num = _extract_number(
        rf"分\s*(?P<num>{NUMBER_TEXT})\s*[份个]",
        text,
        "num",
    )
    if amount is not None:
        slots["amount"] = amount
    if num is not None:
        slots["num"] = num

    patterns = (
        rf"(?P<num>{NUMBER_TEXT})\s*[个份]\s*(?P<amount>{NUMBER_TEXT})\s*金币",
        rf"(?P<num>{NUMBER_TEXT})\s*[个份]\s*红包.*?(?P<amount>{NUMBER_TEXT})\s*金币",
        rf"(?P<amount>{NUMBER_TEXT})\s*金币.*?(?P<num>{NUMBER_TEXT})\s*[个份]\s*红包",
        rf"(?P<amount>{NUMBER_TEXT})\s*金币\s*红包\s*(?P<num>{NUMBER_TEXT})\s*[个份]",
        rf"(?P<amount>{NUMBER_TEXT})\s*金币.*?(?P<num>{NUMBER_TEXT})\s*[个份]",
        rf"红包.*?(?P<amount>{NUMBER_TEXT})\s*金币.*?(?P<num>{NUMBER_TEXT})\s*[个份]",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        parsed_num = parse_int_token(match.group("num"))
        parsed_amount = parse_int_token(match.group("amount"))
        if parsed_num is not None:
            slots["num"] = parsed_num
        if parsed_amount is not None:
            slots["amount"] = parsed_amount

    if "num" not in slots:
        num_only = _extract_number(
            rf"(?P<num>{NUMBER_TEXT})\s*[个份]\s*红包",
            text,
            "num",
        )
        if num_only is not None:
            slots["num"] = num_only

    if "amount" not in slots:
        amount = _extract_number(rf"(?P<amount>{NUMBER_TEXT})\s*金币", text, "amount")
        if amount is not None:
            slots["amount"] = amount
    return slots


def _extract_translate_slots(message_text: str) -> dict[str, Any]:
    text = normalize_message_text(message_text)
    if text in {"帮我翻译一下", "翻译一下"}:
        return {}
    if any(
        token in text for token in ("支持哪些语言", "支持什么语言", "翻译语种", "语种")
    ):
        return {}
    for pattern in (
        r"把\s*(?P<text>.+?)\s*翻(?:译)?成(?:中文|英文|日文|韩文)",
        r"用中文说一下\s*(?P<text>.+)",
        r"翻译一下\s*(?P<text>.+)",
        r"翻译\s*(?P<text>.+)",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = normalize_message_text(match.group("text"))
            if value:
                return {"text": value}
    return {}


def _extract_roll_slots(message_text: str) -> dict[str, Any]:
    text = normalize_message_text(message_text)
    patterns = (
        r"从\s*(?P<options>.+?)\s*(?:里|中|里面)?\s*(?:选|挑|决定)",
        r"帮我从\s*(?P<options>.+?)\s*(?:里|中|里面)?\s*(?:选|挑|决定)",
        r"(?:帮我)?(?:决定|选|挑)\s*(?P<options>.+)",
        r"二选一\s*(?P<options>.+)",
        r"(?P<options>.+?)\s*(?:里|中|里面)?\s*(?:选一个|挑一个|二选一)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        options = _split_options(match.group("options"))
        if options:
            return {"options": options}
    return {}


def _extract_nbnhhsh_slots(message_text: str) -> dict[str, Any]:
    text = normalize_message_text(message_text)
    patterns = (
        r"(?P<text>[0-9A-Za-z_]{2,16})\s*(?:是)?(?:什么|啥|哪个)?缩写",
        r"(?:nbnhhsh|能不能好好说话|解释(?:一下)?缩写|缩写)\s*(?P<text>[0-9A-Za-z_]{2,16})",
        r"(?P<text>[0-9A-Za-z_]{2,16}).{0,4}(?:展开|啥意思|什么意思|是什么意思)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {"text": match.group("text")}
    return {}


def _same_command(command_id: str):
    def extractor(_command_id: str, source: str) -> dict[str, Any]:
        if command_id == "gold_redbag.send":
            return _extract_redbag_slots(source)
        if command_id == "translate.text":
            return _extract_translate_slots(source)
        if command_id == "roll.choose":
            return _extract_roll_slots(source)
        if command_id == "nbnhhsh.expand":
            return _extract_nbnhhsh_slots(source)
        return {}

    return extractor


def _simple_text_extractor(patterns: tuple[str, ...]):
    def extractor(_command_id: str, source: str) -> dict[str, Any]:
        text = normalize_message_text(source)
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            group_name = "target" if "target" in match.groupdict() else "text"
            value = normalize_message_text(match.group(group_name))
            if value:
                return {group_name: value}
        return {}

    return extractor


def _aliases_for(head: str, values: tuple[str, ...]) -> list[str]:
    aliases: list[str] = []
    for value in values:
        text = normalize_message_text(value)
        if text and text != head and text not in aliases:
            aliases.append(text)
    return aliases


def _music_semantic_aliases(
    head: str,
    _module: str,
    _image_required: bool,
) -> list[str]:
    if head not in {"点歌", "搜歌", "播放音乐", "音乐搜索"}:
        return []
    return _aliases_for(
        head,
        (
            "点一首歌",
            "点首歌",
            "点一首",
            "点首",
            "播一首歌",
            "播首歌",
            "来一首歌",
            "来首歌",
            "放一首歌",
            "听歌",
            "搜歌",
            "找歌",
        ),
    )


def _sign_in_semantic_aliases(
    head: str,
    _module: str,
    _image_required: bool,
) -> list[str]:
    if head not in {"签到", "打卡", "补签"}:
        return []
    return _aliases_for(
        head,
        (
            "打卡",
            "签个到",
            "签一下到",
            "今日签到",
            "今天签到",
        ),
    )


def _poetry_semantic_aliases(
    head: str,
    _module: str,
    _image_required: bool,
) -> list[str]:
    if head not in {"念诗", "来首诗", "念首诗", "古诗", "诗词"}:
        return []
    return _aliases_for(
        head,
        (
            "古诗",
            "诗词",
            "来首诗",
            "念首诗",
            "来一首古诗",
            "来首古诗",
        ),
    )


def _builtin_score_hints(
    schema_value: PluginCommandSchema,
    lowered_query: str,
    _stripped_lowered_query: str,
) -> list[AdapterScoreHint]:
    command_id = schema_value.command_id
    hints: list[AdapterScoreHint] = []
    if (
        command_id == "about.info"
        and any(token in lowered_query for token in ("真寻", "小真寻", "bot", "机器人"))
        and any(token in lowered_query for token in ("信息", "介绍", "了解", "项目"))
    ):
        hints.append(AdapterScoreHint(120.0, "about_intent"))
    if command_id == "nbnhhsh.expand" and re.search(
        r"[0-9A-Za-z_]{2,}\s*(?:是)?(?:什么|啥|哪个)?缩写",
        lowered_query,
        re.IGNORECASE,
    ):
        hints.append(AdapterScoreHint(120.0, "abbr_intent"))
    lang_query_tokens = ("支持哪些", "哪些语言", "语种", "支持什么语言")
    if schema_value.requires.get("text") and any(
        token in lowered_query for token in lang_query_tokens
    ):
        hints.append(AdapterScoreHint(-360.0, "text_lang_penalty"))
    return hints


register_adapter(
    PluginCommandAdapter(
        modules=("zhenxun.plugins.sign_in",),
        semantic_aliases=_sign_in_semantic_aliases,
    )
)

register_adapter(
    PluginCommandAdapter(
        modules=("zhenxun.plugins.music",),
        semantic_aliases=_music_semantic_aliases,
    )
)

register_adapter(
    PluginCommandAdapter(
        modules=("zhenxun.plugins.gold_redbag",),
        family="gold_redbag",
        schemas=(
            schema(
                "gold_redbag.send",
                "塞红包",
                aliases=[
                    "金币红包",
                    "发红包",
                    "塞金币红包",
                    "给群里发红包",
                    "发金币红包",
                    "给大家发红包",
                ],
                description="发送金币红包；用于发/塞红包，amount=总金币，num=红包个数",
                slots=[
                    slot(
                        "amount",
                        "int",
                        required=True,
                        aliases=["金额", "金币", "总额"],
                        description="红包总金币数",
                    ),
                    slot(
                        "num",
                        "int",
                        default=5,
                        aliases=["数量", "红包数", "个", "份"],
                        description="红包个数，默认 5",
                    ),
                ],
                render="塞红包 {amount} {num}",
                requires={"text": True},
                payload_policy="slots",
                extra_text_policy="slot_only",
            ),
            schema(
                "gold_redbag.open",
                "开",
                aliases=["抢", "开红包", "抢红包", "我想抢红包", "领红包"],
                description="打开/抢/领取当前群可领取的红包；不发送新红包",
                render="开",
                extra_text_policy="discard",
            ),
            schema(
                "gold_redbag.return",
                "退回红包",
                aliases=["退还红包", "红包退回", "没领完的红包退回", "退回没领完红包"],
                description="退回自己发出且未领取完的红包；不是抢红包",
                render="退回红包",
                extra_text_policy="discard",
            ),
        ),
        slot_extractors={"gold_redbag.send": _same_command("gold_redbag.send")},
    )
)

register_adapter(
    PluginCommandAdapter(
        modules=("zhenxun.plugins.roll",),
        family="choice",
        schemas=(
            schema(
                "roll.choose",
                "roll",
                aliases=[
                    "随机选",
                    "帮我选",
                    "从里面选",
                    "选择困难",
                    "二选一",
                    "选一个",
                    "挑一个",
                    "做个选择",
                    "帮我决定",
                ],
                description="从给定多个候选项中随机选择一个；需要 options",
                slots=[
                    slot(
                        "options",
                        "text",
                        required=True,
                        aliases=["选项", "候选"],
                        description="用空格分隔的候选项",
                    )
                ],
                render="roll {options}",
                requires={"text": True},
                payload_policy="slots",
                extra_text_policy="slot_only",
            ),
            schema(
                "roll.number",
                "roll",
                aliases=[
                    "随机数字",
                    "掷骰子",
                    "roll点",
                    "随机一个数字",
                    "投个随机数字",
                ],
                description="随机生成数字/骰子点数；不需要候选项文本",
                render="roll",
                extra_text_policy="discard",
            ),
        ),
        slot_extractors={"roll.choose": _same_command("roll.choose")},
    )
)

register_adapter(
    PluginCommandAdapter(
        modules=("zhenxun.plugins.poetry",),
        semantic_aliases=_poetry_semantic_aliases,
        schemas=(
            schema(
                "poetry.random",
                "古诗",
                aliases=[
                    "念诗",
                    "来首诗",
                    "念首诗",
                    "给我念一首诗",
                    "来一首古诗",
                    "来首古诗",
                    "诗词",
                ],
                description="随机发送一首古诗词",
                render="念诗",
            ),
        ),
    )
)

register_adapter(
    PluginCommandAdapter(
        modules=("zhenxun.plugins.cover",),
        schemas=(
            schema(
                "cover.bilibili",
                "b封面",
                aliases=["B站封面", "视频封面", "查视频封面"],
                description="获取 B 站视频或直播封面",
                slots=[
                    slot(
                        "target",
                        "text",
                        required=True,
                        aliases=["链接", "BV号", "av号", "直播id"],
                    )
                ],
                render="b封面 {target}",
                requires={"text": True},
                payload_policy="slots",
                extra_text_policy="slot_only",
            ),
        ),
        slot_extractors={
            "cover.bilibili": _simple_text_extractor(
                (r"(?P<target>https?://\S+|BV[0-9A-Za-z]+|av\d+|cv\d+)",)
            )
        },
    )
)

register_adapter(
    PluginCommandAdapter(
        modules=("zhenxun.plugins.translate",),
        family="translate",
        schemas=(
            schema(
                "translate.text",
                "翻译",
                aliases=[
                    "翻译一下",
                    "翻成中文",
                    "翻译成中文",
                    "帮我翻译",
                    "用中文说一下",
                ],
                description="翻译给定文本；需要 text，不用于查看支持语种",
                slots=[slot("text", "text", required=True, aliases=["文本", "内容"])],
                render="翻译 {text}",
                requires={"text": True},
                payload_policy="text",
                extra_text_policy="slot_only",
            ),
            schema(
                "translate.langs",
                "翻译语种",
                aliases=["翻译语种", "支持哪些语言", "翻译支持什么语言"],
                description="查看翻译插件支持的语言列表；不是执行翻译",
                render="翻译语种",
                command_role="helper",
                extra_text_policy="discard",
            ),
        ),
        slot_extractors={"translate.text": _same_command("translate.text")},
        score_hints=_builtin_score_hints,
        prompt_score_hints=lambda schema_value, normalized_query: _builtin_score_hints(
            schema_value, normalized_query, normalized_query
        ),
    )
)

register_adapter(
    PluginCommandAdapter(
        modules=("zhenxun.plugins.luxun",),
        schemas=(
            schema(
                "luxun.say",
                "鲁迅说",
                aliases=["鲁迅风格", "来张鲁迅说", "让鲁迅说"],
                description="生成鲁迅说图片",
                slots=[slot("text", "text", required=True, aliases=["内容", "文本"])],
                render="鲁迅说 {text}",
                requires={"text": True},
                payload_policy="text",
                extra_text_policy="slot_only",
            ),
        ),
        slot_extractors={
            "luxun.say": _simple_text_extractor(
                (
                    r"鲁迅风格(?:写一句|写一段|说)?\s*(?P<text>.+)",
                    r"文字是\s*(?P<text>.+)",
                    r"内容是\s*(?P<text>.+)",
                    r"说\s*(?P<text>.+)",
                )
            )
        },
    )
)

register_adapter(
    PluginCommandAdapter(
        modules=("zhenxun.plugins.nbnhhsh",),
        schemas=(
            schema(
                "nbnhhsh.expand",
                "能不能好好说话",
                aliases=["nbnhhsh", "解释缩写", "缩写是什么意思"],
                description="解释网络缩写",
                slots=[slot("text", "text", required=True, aliases=["缩写", "文本"])],
                render="能不能好好说话 {text}",
                requires={"text": True},
                payload_policy="text",
                extra_text_policy="slot_only",
            ),
        ),
        slot_extractors={"nbnhhsh.expand": _same_command("nbnhhsh.expand")},
        score_hints=_builtin_score_hints,
        prompt_score_hints=lambda schema_value, normalized_query: _builtin_score_hints(
            schema_value, normalized_query, normalized_query
        ),
    )
)

register_adapter(
    PluginCommandAdapter(
        modules=("zhenxun.plugins.quotations",),
        schemas=(
            schema(
                "quotations.hitokoto",
                "语录",
                aliases=["来一句语录", "一言"],
                description="随机发送一句语录",
                render="语录",
            ),
            schema(
                "quotations.acg",
                "二次元",
                aliases=["二次元语录", "来一句二次元语录"],
                description="随机发送一句二次元语录",
                render="二次元",
            ),
        ),
    )
)

register_adapter(
    PluginCommandAdapter(
        modules=("zhenxun.builtin_plugins.about",),
        schemas=(
            schema(
                "about.info",
                "关于",
                aliases=[
                    "about",
                    "真寻信息",
                    "小真寻信息",
                    "小真寻的信息",
                    "了解小真寻",
                    "想了解小真寻",
                    "机器人信息",
                    "bot信息",
                    "项目介绍",
                    "项目说明",
                    "介绍真寻",
                ],
                description="查看真寻项目、版本和帮助入口",
                render="关于",
                command_role="helper",
                extra_text_policy="discard",
            ),
        ),
        score_hints=_builtin_score_hints,
        prompt_score_hints=lambda schema_value, normalized_query: _builtin_score_hints(
            schema_value, normalized_query, normalized_query
        ),
    )
)
