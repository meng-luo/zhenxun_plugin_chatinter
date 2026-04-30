"""Deterministic slot extraction used after LLM command selection."""

from __future__ import annotations

import re
from typing import Any

from .route_text import normalize_message_text

_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "俩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
NUMBER_TEXT = r"\d+|[零〇一二两俩三四五六七八九十百千]+"


def parse_int_token(value: object) -> int | None:
    text = normalize_message_text(str(value or ""))
    if not text:
        return None
    if text.isdigit():
        return int(text)
    total = 0
    current = 0
    for char in text:
        if char in _CN_DIGITS:
            current = _CN_DIGITS[char]
            continue
        unit = _CN_UNITS.get(char)
        if unit is None:
            return None
        if current == 0:
            current = 1
        total += current * unit
        current = 0
    return total + current if total or current else None


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


def extract_redbag_slots(message_text: str) -> dict[str, Any]:
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


def extract_translate_slots(message_text: str) -> dict[str, Any]:
    text = normalize_message_text(message_text)
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


def extract_roll_slots(message_text: str) -> dict[str, Any]:
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


def extract_meme_slots(command_id: str, message_text: str) -> dict[str, Any]:
    normalized_id = normalize_message_text(command_id)
    text = normalize_message_text(message_text)
    if normalized_id == "memes.search":
        patterns = (
            r"(?:查找|搜索|找|搜|查)(?:一下)?\s*(?P<keyword>.+?)(?:相关)?(?:的)?表情",
            r"表情搜索\s*(?P<keyword>.+)",
        )
    elif normalized_id == "memes.info":
        patterns = (
            r"(?:查|看|了解)(?:一下)?\s*(?P<keyword>.+?)(?:这个)?表情(?:怎么用|详情|用法)?",
            r"表情详情\s*(?P<keyword>.+)",
        )
    else:
        return {}
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        keyword = normalize_message_text(match.group("keyword"))
        keyword = re.sub(r"(?:这个|的|怎么用|用法|详情)$", "", keyword).strip()
        if keyword:
            return {"keyword": keyword}
    return {}


def extract_nbnhhsh_slots(message_text: str) -> dict[str, Any]:
    text = normalize_message_text(message_text)
    patterns = (
        r"(?:nbnhhsh|能不能好好说话|解释(?:一下)?缩写|缩写)\s*(?P<text>[0-9A-Za-z_]{2,16})",
        r"(?P<text>[0-9A-Za-z_]{2,16})\s*(?:是)?(?:什么|啥|哪个)?缩写",
        r"(?P<text>[0-9A-Za-z_]{2,16}).{0,4}(?:展开|啥意思|什么意思|是什么意思)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {"text": match.group("text")}
    return {}


def extract_builtin_slots(
    command_id: str,
    message_text: str,
    arguments_text: str = "",
) -> dict[str, Any]:
    command_id = normalize_message_text(command_id)
    source = normalize_message_text(message_text)
    if not source and arguments_text:
        source = normalize_message_text(arguments_text)
    if command_id == "gold_redbag.send":
        return extract_redbag_slots(source)
    if command_id == "translate.text":
        return extract_translate_slots(source)
    if command_id == "roll.choose":
        return extract_roll_slots(source)
    if command_id in {"memes.search", "memes.info"}:
        return extract_meme_slots(command_id, source)
    if command_id == "nbnhhsh.expand":
        return extract_nbnhhsh_slots(source)
    return {}


__all__ = [
    "NUMBER_TEXT",
    "extract_builtin_slots",
    "extract_meme_slots",
    "extract_nbnhhsh_slots",
    "extract_redbag_slots",
    "extract_roll_slots",
    "extract_translate_slots",
    "parse_int_token",
]
