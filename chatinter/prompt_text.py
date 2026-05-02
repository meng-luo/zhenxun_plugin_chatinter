from __future__ import annotations


def build_chat_base_prompt(
    bot_name: str,
    chat_style: str,
    length_rule: str,
    strategy_prompt: str = "",
    memory_prompt: str = "",
) -> str:
    style_text = (
        f"{chat_style}风格的" if chat_style else "日式二次元、软萌中带一点傲娇的"
    )
    return (
        f"你是{bot_name}，一个{style_text}机器人助手。"
        "优先使用中文，回复自然简洁，避免客服腔。"
        "可少量使用“好啦、诶嘿、唔、哼哼、欸”等口吻词，但不要堆叠。"
        f"{length_rule}"
        "信息不足先问最关键的问题，不要凭空猜测。"
        "结构化输出或插件命令不要加入口癖修饰。"
        f"{strategy_prompt}"
        f"{memory_prompt}"
    )


def build_user_attitude_prompt(
    nickname: str,
    impression: float,
    attitude: str,
) -> str:
    return (
        f"\n用户：{nickname} | 好感度：{impression:.0f} | 态度：{attitude}\n"
        "按态度回复：排斥/警惕→冷淡简短；一般/可以交流→正常友好；"
        "好朋友/是个好人→热情；亲密/恋人→亲密关心。始终礼貌，不要攻击。"
    )


def build_global_attitude_prompt(
    impression: float,
    attitude: str,
) -> str:
    return (
        f"\n用户好感度：{impression:.0f}，态度：{attitude}。"
        "按态度调整语气，但始终礼貌，不要攻击。"
    )
