"""Adapters for high-frequency built-in/simple command plugins."""

from __future__ import annotations

from ..route_text import normalize_message_text
from . import PluginCommandAdapter, register_adapter, schema, slot


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
            "放一首",
            "放首",
            "给我放一首",
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


register_adapter(
    PluginCommandAdapter(
        modules=("zhenxun.plugins.sign_in",),
        semantic_aliases=_sign_in_semantic_aliases,
    )
)

register_adapter(
    PluginCommandAdapter(
        modules=("zhenxun.plugins.word_bank",),
        family="word_bank",
        schemas=(
            schema(
                "word_bank.add",
                "添加问答",
                aliases=["添加词条", "问答添加", "加个词条", "新增问答"],
                description="添加词库问答；需要问题和回答文本",
                slots=[slot("text", "text", required=True, aliases=["问题=回答"])],
                render="添加问答 {text}",
                requires={"text": True},
                payload_policy="text",
                extra_text_policy="slot_only",
            ),
        ),
    )
)

register_adapter(
    PluginCommandAdapter(
        modules=("zhenxun.plugins.music",),
        semantic_aliases=_music_semantic_aliases,
        family="music",
        schemas=(
            schema(
                "music.play",
                "点歌",
                aliases=[
                    "点歌",
                    "搜歌",
                    "音乐",
                    "点一首歌",
                    "点首歌",
                    "播一首歌",
                    "播首歌",
                    "来一首歌",
                    "放一首歌",
                    "给我放一首",
                ],
                description="点歌、搜歌、播放歌曲；需要歌曲名",
                slots=[slot("text", "text", required=True, aliases=["歌曲名", "歌名"])],
                render="点歌 {text}",
                requires={"text": True},
                payload_policy="text",
                extra_text_policy="slot_only",
            ),
        ),
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
                    "掷个骰子",
                    "帮我掷骰子",
                    "帮我掷个骰子",
                    "投骰子",
                    "扔骰子",
                    "roll点",
                    "随机一个数字",
                    "投个随机数字",
                ],
                description="随机生成数字/骰子点数；不需要候选项文本",
                render="roll",
                extra_text_policy="discard",
            ),
        ),
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
    )
)

register_adapter(
    PluginCommandAdapter(
        modules=("zhenxun.plugins.parse_bilibili",),
        family="link_parser",
        schemas=(
            schema(
                "parse_bilibili.video",
                "B站解析",
                aliases=["解析B站", "bilibili解析", "解析b站视频", "解析视频"],
                description="解析哔哩哔哩/B站/b23/BV 视频链接",
                slots=[slot("target", "text", required=True, aliases=["链接", "BV号"])],
                render="B站解析 {target}",
                requires={"text": True},
                payload_policy="slots",
                extra_text_policy="slot_only",
            ),
        ),
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
    )
)

register_adapter(
    PluginCommandAdapter(
        modules=("zhenxun.plugins.nbnhhsh",),
        schemas=(
            schema(
                "nbnhhsh.expand",
                "能不能好好说话",
                aliases=[
                    "nbnhhsh",
                    "解释缩写",
                    "缩写是什么意思",
                    "什么意思",
                    "是什么意思",
                    "啥意思",
                    "说清楚",
                ],
                description="解释网络缩写",
                slots=[slot("text", "text", required=True, aliases=["缩写", "文本"])],
                render="能不能好好说话 {text}",
                requires={"text": True},
                payload_policy="text",
                extra_text_policy="slot_only",
            ),
        ),
    )
)

register_adapter(
    PluginCommandAdapter(
        modules=("zhenxun.plugins.what_anime",),
        family="image_recognition",
        schemas=(
            schema(
                "what_anime.search",
                "搜番",
                aliases=["识别番剧", "这是什么番", "识别动漫", "动漫图是什么番"],
                description="根据图片识别动画、番剧、哪部番或动漫来源",
                retrieval_phrases=[
                    "哪部番",
                    "什么番",
                    "截图是哪部番",
                    "动画截图识别",
                ],
                render="搜番",
                requires={"image": True, "reply": True},
                command_role="template",
                payload_policy="image_only",
                extra_text_policy="discard",
            ),
        ),
    )
)

register_adapter(
    PluginCommandAdapter(
        modules=("zhenxun.plugins.what_role",),
        family="image_recognition",
        schemas=(
            schema(
                "what_role.search",
                "识别角色",
                aliases=["角色识别", "这是谁", "识别人物", "图里的角色是谁"],
                description="根据图片识别动漫角色、人物或角色来源",
                retrieval_phrases=["图片里这位是谁", "图里是谁", "角色识别"],
                render="识别角色",
                requires={"image": True, "reply": True},
                command_role="template",
                payload_policy="image_only",
                extra_text_policy="discard",
            ),
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
    )
)
