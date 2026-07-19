"""Pure string similarity helpers for group member target resolution."""

from __future__ import annotations

from dataclasses import dataclass
import re

_ALIAS_CLEAN_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")


@dataclass(frozen=True, slots=True)
class MemberAliasEntry:
    value: str
    kind: str
    source: str = ""


def normalize_member_alias(value: str) -> str:
    return _ALIAS_CLEAN_RE.sub("", str(value or "")).lower().strip()


def build_member_alias_entries(
    *aliases: str,
    kind: str = "full",
    include_suffixes: bool = True,
) -> tuple[MemberAliasEntry, ...]:
    entries: list[MemberAliasEntry] = []
    seen: set[tuple[str, str]] = set()

    def add(value: str, entry_kind: str, source: str) -> None:
        normalized = normalize_member_alias(value)
        if len(normalized) < 2:
            return
        key = (normalized, entry_kind)
        if key in seen:
            return
        seen.add(key)
        entries.append(MemberAliasEntry(normalized, entry_kind, source))

    def add_suffixes(value: str, source: str) -> None:
        normalized = normalize_member_alias(value)
        if not include_suffixes or len(normalized) > 4:
            return
        for size in (2, 3):
            if len(normalized) > size:
                add(normalized[-size:], "suffix", source)

    for raw in aliases:
        source = str(raw or "").strip()
        if not source:
            continue
        add(source, kind, source)
        add_suffixes(source, source)
        for chunk in _CJK_RUN_RE.findall(source):
            add(chunk, "chunk", source)
            add_suffixes(chunk, source)
    return tuple(entries)


def jaro_winkler_similarity(left: str, right: str) -> float:
    left = normalize_member_alias(left)
    right = normalize_member_alias(right)
    jaro = _jaro_similarity(left, right)
    prefix = 0
    for left_ch, right_ch in zip(left[:4], right[:4], strict=False):
        if left_ch != right_ch:
            break
        prefix += 1
    return jaro + prefix * 0.1 * (1.0 - jaro)


def cjk_bigram_dice(left: str, right: str) -> float:
    left_grams = _cjk_bigrams(left)
    right_grams = _cjk_bigrams(right)
    if not left_grams or not right_grams:
        return 0.0
    return 2.0 * len(left_grams & right_grams) / (len(left_grams) + len(right_grams))


def ordered_subsequence_score(query: str, alias: str) -> float:
    query_key = normalize_member_alias(query)
    alias_key = normalize_member_alias(alias)
    if len(query_key) < 3 or not alias_key:
        return 0.0
    position = -1
    for char in query_key:
        position = alias_key.find(char, position + 1)
        if position < 0:
            return 0.0
    coverage = len(query_key) / len(alias_key)
    prefix_bonus = 0.04 if alias_key.startswith(query_key[:2]) else 0.0
    return min(0.91, 0.78 + 0.16 * coverage + prefix_bonus)


def score_member_alias(
    query: str,
    alias: str | MemberAliasEntry,
    kind: str = "full",
) -> float:
    query_key = normalize_member_alias(query)
    if isinstance(alias, MemberAliasEntry):
        alias_key = alias.value
        kind = alias.kind
    else:
        alias_key = normalize_member_alias(alias)

    if len(query_key) < 2 or len(alias_key) < 2:
        return 0.0

    score = 0.0
    exact_match = query_key == alias_key
    if exact_match:
        score = 0.86 if kind == "suffix" else 1.0
    elif _safe_substring_match(query_key, alias_key):
        overlap = min(len(query_key), len(alias_key)) / max(len(query_key), len(alias_key))
        score = 0.80 + overlap * 0.12
    elif not _unsafe_fragment(query_key, alias_key):
        score = max(
            jaro_winkler_similarity(query_key, alias_key)
            if min(len(query_key), len(alias_key)) >= 3
            else 0.0,
            cjk_bigram_dice(query_key, alias_key),
            ordered_subsequence_score(query_key, alias_key),
        )

    if len(query_key) == 2 and score < 0.86 and not exact_match:
        score = 0.0
    if kind == "suffix":
        score = min(score, 0.88)
    return score


def _jaro_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0

    window = max(0, max(len(left), len(right)) // 2 - 1)
    left_matches = [False] * len(left)
    right_matches = [False] * len(right)
    matches = 0
    for left_index, left_char in enumerate(left):
        start = max(0, left_index - window)
        end = min(left_index + window + 1, len(right))
        for right_index in range(start, end):
            if right_matches[right_index] or left_char != right[right_index]:
                continue
            left_matches[left_index] = True
            right_matches[right_index] = True
            matches += 1
            break
    if not matches:
        return 0.0

    transpositions = 0
    right_index = 0
    for left_index, left_char in enumerate(left):
        if not left_matches[left_index]:
            continue
        while not right_matches[right_index]:
            right_index += 1
        if left_char != right[right_index]:
            transpositions += 1
        right_index += 1

    return (
        matches / len(left)
        + matches / len(right)
        + (matches - transpositions / 2.0) / matches
    ) / 3.0


def _cjk_bigrams(value: str) -> set[str]:
    cjk_chars = [char for char in normalize_member_alias(value) if _is_cjk(char)]
    return {"".join(cjk_chars[index : index + 2]) for index in range(len(cjk_chars) - 1)}


def _safe_substring_match(query_key: str, alias_key: str) -> bool:
    if _unsafe_fragment(query_key, alias_key):
        return False
    if query_key not in alias_key and alias_key not in query_key:
        return False
    if _is_ascii(query_key) and _is_ascii(alias_key):
        return query_key.isalpha() and alias_key.isalpha() and len(query_key) >= 3
    return len(query_key) >= 3 or (
        len(query_key) == 2 and len(alias_key) <= 4 and _contains_cjk(query_key)
    )


def _unsafe_fragment(query_key: str, alias_key: str) -> bool:
    return (
        query_key.isdigit()
        or (
            _is_ascii(query_key)
            and _is_ascii(alias_key)
            and query_key != alias_key
            and any(char.isdigit() for char in f"{query_key}{alias_key}")
        )
        or (_is_ascii(query_key) and _contains_cjk(alias_key))
    )


def _is_ascii(value: str) -> bool:
    return bool(value) and all(ord(char) < 128 for char in value)


def _contains_cjk(value: str) -> bool:
    return any(_is_cjk(char) for char in value)


def _is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


__all__ = (
    "MemberAliasEntry",
    "build_member_alias_entries",
    "cjk_bigram_dice",
    "jaro_winkler_similarity",
    "normalize_member_alias",
    "ordered_subsequence_score",
    "score_member_alias",
)
