from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any

from .persistence import read_json, state_path, utc_now_iso, write_json
from .route_text import normalize_message_text

_SKILL_INDEX_PATH = state_path("skills", "index.json")
_SKILL_MARKDOWN_DIR = state_path("skills", "markdown")
_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]{1,8}", re.IGNORECASE)
_MAX_SKILLS = 200
_MAX_MATCHED_SKILLS = 3


@dataclass(frozen=True)
class SkillRecord:
    skill_id: str
    title: str
    pattern: str
    summary: str
    steps: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    cautions: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    markdown: str = ""
    markdown_path: str = ""
    source_trace_ids: tuple[str, ...] = ()
    success_count: int = 1
    confidence: float = 0.5
    created_at: str = ""
    updated_at: str = ""
    last_used_at: str = ""

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Any) -> "SkillRecord | None":
        if not isinstance(payload, dict):
            return None
        skill_id = normalize_message_text(str(payload.get("skill_id", "") or ""))
        title = normalize_message_text(str(payload.get("title", "") or ""))
        pattern = normalize_message_text(str(payload.get("pattern", "") or ""))
        summary = normalize_message_text(str(payload.get("summary", "") or ""))
        if not skill_id or not title or not pattern:
            return None
        return cls(
            skill_id=skill_id,
            title=title,
            pattern=pattern,
            summary=summary,
            steps=_text_tuple(payload.get("steps"), limit=12),
            tools=_text_tuple(payload.get("tools"), limit=24),
            cautions=_text_tuple(payload.get("cautions"), limit=12),
            tags=_text_tuple(payload.get("tags"), limit=16),
            examples=_text_tuple(payload.get("examples"), limit=8),
            markdown=normalize_message_text(str(payload.get("markdown", "") or "")),
            markdown_path=normalize_message_text(
                str(payload.get("markdown_path", "") or "")
            ),
            source_trace_ids=_text_tuple(payload.get("source_trace_ids"), limit=24),
            success_count=max(int(payload.get("success_count", 1) or 1), 1),
            confidence=_clamp01(payload.get("confidence", 0.5)),
            created_at=str(payload.get("created_at", "") or ""),
            updated_at=str(payload.get("updated_at", "") or ""),
            last_used_at=str(payload.get("last_used_at", "") or ""),
        )


@dataclass(frozen=True)
class SkillMatch:
    record: SkillRecord
    score: float
    reason: str = ""

    def to_prompt_xml(self, index: int) -> str:
        tools = ", ".join(self.record.tools[:10])
        steps = "\n".join(
            f"    <step>{_xml_escape(step)}</step>" for step in self.record.steps[:8]
        )
        cautions = "\n".join(
            f"    <caution>{_xml_escape(item)}</caution>"
            for item in self.record.cautions[:6]
        )
        examples = "\n".join(
            f"    <example>{_xml_escape(item)}</example>"
            for item in self.record.examples[:3]
        )
        parts = [
            f'  <skill index="{index}" id="{_xml_escape(self.record.skill_id)}" '
            f'score="{self.score:.3f}">',
            f"    <title>{_xml_escape(self.record.title)}</title>",
            f"    <pattern>{_xml_escape(self.record.pattern)}</pattern>",
            f"    <summary>{_xml_escape(self.record.summary)}</summary>",
            f"    <tools>{_xml_escape(tools)}</tools>" if tools else "",
            "    <steps>" if steps else "",
            steps,
            "    </steps>" if steps else "",
            "    <cautions>" if cautions else "",
            cautions,
            "    </cautions>" if cautions else "",
            "    <examples>" if examples else "",
            examples,
            "    </examples>" if examples else "",
            "  </skill>",
        ]
        return "\n".join(part for part in parts if part)


@dataclass(frozen=True)
class SkillCandidate:
    title: str
    pattern: str
    summary: str
    steps: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    cautions: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    markdown: str = ""
    trace_id: str = ""
    confidence: float = 0.55


def load_skills() -> list[SkillRecord]:
    payload = read_json(_SKILL_INDEX_PATH, {})
    items = payload.get("skills", []) if isinstance(payload, dict) else []
    skills: list[SkillRecord] = []
    if isinstance(items, list):
        for item in items:
            record = SkillRecord.from_payload(item)
            if record is not None:
                skills.append(record)
    return skills


def upsert_skill(candidate: SkillCandidate) -> SkillRecord | None:
    normalized = _normalize_candidate(candidate)
    if normalized is None:
        return None
    skills = load_skills()
    existing_index = _find_existing_skill(skills, normalized)
    now = utc_now_iso()
    if existing_index >= 0:
        old = skills[existing_index]
        merged = _merge_skill(old, normalized, updated_at=now)
        skills[existing_index] = merged
        record = merged
    else:
        record = _candidate_to_record(normalized, created_at=now, updated_at=now)
        skills.append(record)
    skills = _trim_skills(skills)
    write_json(
        _SKILL_INDEX_PATH,
        {
            "schema_version": "chatinter.skill_store.v1",
            "updated_at": now,
            "skills": [skill.to_payload() for skill in skills],
        },
    )
    _write_skill_markdowns(skills)
    return record


def search_skills(query: str, *, limit: int = _MAX_MATCHED_SKILLS) -> list[SkillMatch]:
    normalized_query = normalize_message_text(query)
    query_tokens = _tokens(normalized_query)
    if not query_tokens:
        return []
    matches: list[SkillMatch] = []
    for record in load_skills():
        score, reason = _skill_score(record, query_tokens=query_tokens)
        if score <= 0:
            continue
        matches.append(SkillMatch(record=record, score=score, reason=reason))
    matches.sort(
        key=lambda item: (
            item.score,
            item.record.confidence,
            item.record.success_count,
            item.record.updated_at,
        ),
        reverse=True,
    )
    return matches[: max(1, min(int(limit or _MAX_MATCHED_SKILLS), 8))]







_SKILL_VECTOR_CACHE: dict[str, tuple[str, list[float]]] = {}
_SEMANTIC_BLEND = 0.6


def _skill_embed_text(record: SkillRecord) -> str:
    return normalize_message_text(
        f"{record.title} {record.pattern} {' '.join(record.tags)} "
        f"{' '.join(record.tools)}"
    )


async def search_skills_semantic(
    query: str, *, limit: int = _MAX_MATCHED_SKILLS
) -> list[SkillMatch]:
    """语义技能检索:embedding 余弦 + Jaccard 混合;embedding 不可用则退回 Jaccard。"""
    baseline = search_skills(query, limit=max(limit * 3, limit))
    normalized_query = normalize_message_text(query)
    if not normalized_query:
        return baseline[:limit]
    try:
        from .memory_vector_index import (
            MemoryVectorIndex,
            _cosine_score,
            _has_embedding_models,
        )

        if not _has_embedding_models():
            return baseline[:limit]
        query_vec = await MemoryVectorIndex._embed_query(normalized_query)
        if not query_vec:
            return baseline[:limit]

        candidates = baseline or [
            SkillMatch(record=record, score=0.0, reason="semantic_pool")
            for record in load_skills()
        ]
        max_jaccard = max((m.score for m in candidates), default=0.0) or 1.0
        rescored: list[SkillMatch] = []
        for match in candidates:
            record = match.record
            sig = record.updated_at or ""
            cached = _SKILL_VECTOR_CACHE.get(record.skill_id)
            if cached is None or cached[0] != sig:
                vec = await MemoryVectorIndex._embed_query(_skill_embed_text(record))
                _SKILL_VECTOR_CACHE[record.skill_id] = (sig, vec)
            else:
                vec = cached[1]
            cosine = _cosine_score(query_vec, vec) if vec else 0.0
            blended = (
                _SEMANTIC_BLEND * cosine
                + (1.0 - _SEMANTIC_BLEND) * (match.score / max_jaccard)
            )
            if blended <= 0:
                continue
            rescored.append(
                SkillMatch(
                    record=record,
                    score=blended,
                    reason=f"semantic({cosine:.2f})+{match.reason}",
                )
            )
        if not rescored:
            return baseline[:limit]
        rescored.sort(
            key=lambda item: (
                item.score,
                item.record.confidence,
                item.record.success_count,
                item.record.updated_at,
            ),
            reverse=True,
        )
        return rescored[: max(1, min(int(limit or _MAX_MATCHED_SKILLS), 8))]
    except Exception:
        return baseline[:limit]


def render_skill_hints(matches: list[SkillMatch]) -> str:
    if not matches:
        return ""
    body = "\n".join(
        match.to_prompt_xml(index=index) for index, match in enumerate(matches, 1)
    )
    return (
        "<chatinter_skill_hints>\n"
        "  <instruction>以下是历史成功任务提炼出的可复用技能。"
        "它们是提示，不是强制流程；如果当前任务不匹配，可以忽略。</instruction>\n"
        f"{body}\n"
        "</chatinter_skill_hints>"
    )


def mark_skills_used(matches: list[SkillMatch]) -> None:
    if not matches:
        return
    used_ids = {match.record.skill_id for match in matches}
    skills = load_skills()
    now = utc_now_iso()
    changed = False
    updated: list[SkillRecord] = []
    for skill in skills:
        if skill.skill_id in used_ids:
            updated.append(
                SkillRecord(
                    **{
                        **skill.to_payload(),
                        "last_used_at": now,
                    }
                )
            )
            changed = True
        else:
            updated.append(skill)
    if changed:
        write_json(
            _SKILL_INDEX_PATH,
            {
                "schema_version": "chatinter.skill_store.v1",
                "updated_at": now,
                "skills": [skill.to_payload() for skill in updated],
            },
        )


def _normalize_candidate(candidate: SkillCandidate) -> SkillCandidate | None:
    title = normalize_message_text(candidate.title)
    pattern = normalize_message_text(candidate.pattern)
    summary = normalize_message_text(candidate.summary)
    if not title or not pattern or not summary:
        return None
    tools = _dedupe_tuple(candidate.tools, limit=24)
    steps = _dedupe_tuple(candidate.steps, limit=12)
    if not tools and not steps:
        return None
    return SkillCandidate(
        title=title[:120],
        pattern=pattern[:240],
        summary=summary[:500],
        steps=steps,
        tools=tools,
        cautions=_dedupe_tuple(candidate.cautions, limit=12),
        tags=_dedupe_tuple(candidate.tags, limit=16),
        examples=_dedupe_tuple(candidate.examples, limit=8),
        markdown=normalize_message_text(candidate.markdown)[:4000],
        trace_id=normalize_message_text(candidate.trace_id),
        confidence=_clamp01(candidate.confidence),
    )


def _candidate_to_record(
    candidate: SkillCandidate,
    *,
    created_at: str,
    updated_at: str,
) -> SkillRecord:
    skill_id = _skill_id(candidate)
    markdown = candidate.markdown or _default_markdown(candidate)
    return SkillRecord(
        skill_id=skill_id,
        title=candidate.title,
        pattern=candidate.pattern,
        summary=candidate.summary,
        steps=candidate.steps,
        tools=candidate.tools,
        cautions=candidate.cautions,
        tags=candidate.tags,
        examples=candidate.examples,
        markdown=markdown,
        markdown_path=_relative_markdown_path(skill_id),
        source_trace_ids=(candidate.trace_id,) if candidate.trace_id else (),
        success_count=1,
        confidence=candidate.confidence,
        created_at=created_at,
        updated_at=updated_at,
    )


def _merge_skill(
    old: SkillRecord,
    candidate: SkillCandidate,
    *,
    updated_at: str,
) -> SkillRecord:
    success_count = old.success_count + 1
    confidence = min(max(old.confidence, candidate.confidence) + 0.03, 0.95)
    payload = old.to_payload()
    payload.update(
        {
            "title": candidate.title or old.title,
            "pattern": candidate.pattern or old.pattern,
            "summary": candidate.summary or old.summary,
            "steps": _merge_tuple(old.steps, candidate.steps, limit=12),
            "tools": _merge_tuple(old.tools, candidate.tools, limit=24),
            "cautions": _merge_tuple(old.cautions, candidate.cautions, limit=12),
            "tags": _merge_tuple(old.tags, candidate.tags, limit=16),
            "examples": _merge_tuple(old.examples, candidate.examples, limit=8),
            "markdown": candidate.markdown or old.markdown,
            "markdown_path": old.markdown_path or _relative_markdown_path(old.skill_id),
            "source_trace_ids": _merge_tuple(
                old.source_trace_ids,
                (candidate.trace_id,) if candidate.trace_id else (),
                limit=24,
            ),
            "success_count": success_count,
            "confidence": confidence,
            "updated_at": updated_at,
        }
    )
    return SkillRecord.from_payload(payload) or old


def _find_existing_skill(
    skills: list[SkillRecord],
    candidate: SkillCandidate,
) -> int:
    candidate_tokens = _tokens(
        f"{candidate.title} {candidate.pattern} {' '.join(candidate.tags)}"
    )
    candidate_tools = set(candidate.tools)
    best_index = -1
    best_score = 0.0
    for index, skill in enumerate(skills):
        skill_tokens = _tokens(f"{skill.title} {skill.pattern} {' '.join(skill.tags)}")
        overlap = _jaccard(candidate_tokens, skill_tokens)
        tool_overlap = _jaccard(candidate_tools, set(skill.tools))
        score = overlap * 0.75 + tool_overlap * 0.25
        if score > best_score:
            best_score = score
            best_index = index
    return best_index if best_score >= 0.42 else -1


def _skill_score(
    record: SkillRecord,
    *,
    query_tokens: set[str],
) -> tuple[float, str]:
    skill_tokens = _tokens(
        " ".join(
            [
                record.title,
                record.pattern,
                record.summary,
                " ".join(record.tags),
                " ".join(record.examples),
            ]
        )
    )
    if not skill_tokens:
        return 0.0, ""
    overlap = len(query_tokens & skill_tokens)
    if overlap <= 0:
        return 0.0, ""
    containment = overlap / max(min(len(query_tokens), len(skill_tokens)), 1)
    jaccard = overlap / max(len(query_tokens | skill_tokens), 1)
    score = (
        containment * 0.58
        + jaccard * 0.22
        + min(record.success_count, 10) * 0.015
        + record.confidence * 0.12
    )
    return score, "token_overlap"


def _trim_skills(skills: list[SkillRecord]) -> list[SkillRecord]:
    if len(skills) <= _MAX_SKILLS:
        return skills
    return sorted(
        skills,
        key=lambda skill: (
            skill.success_count,
            skill.confidence,
            skill.last_used_at or skill.updated_at,
        ),
        reverse=True,
    )[:_MAX_SKILLS]


def _default_markdown(candidate: SkillCandidate) -> str:
    lines = [
        f"# {candidate.title}",
        "",
        f"- 任务模式: {candidate.pattern}",
        f"- 摘要: {candidate.summary}",
    ]
    if candidate.tools:
        lines.append("- 常用工具: " + ", ".join(candidate.tools))
    if candidate.steps:
        lines.append("")
        lines.append("## 步骤")
        lines.extend(
            f"{index}. {step}" for index, step in enumerate(candidate.steps, 1)
        )
    if candidate.cautions:
        lines.append("")
        lines.append("## 注意事项")
        lines.extend(f"- {item}" for item in candidate.cautions)
    return "\n".join(lines)


def _skill_id(candidate: SkillCandidate) -> str:
    source = "|".join(
        [
            candidate.title,
            candidate.pattern,
            " ".join(candidate.tools[:8]),
            " ".join(candidate.tags[:8]),
        ]
    )
    return (
        "skill_"
        + hashlib.blake2b(
            source.encode("utf-8", "ignore"),
            digest_size=8,
        ).hexdigest()
    )


def _relative_markdown_path(skill_id: str) -> str:
    safe_id = re.sub(r"[^0-9A-Za-z_.-]+", "_", skill_id).strip("._")
    return f"skills/markdown/{safe_id or 'skill'}.md"


def _skill_markdown_path(skill_id: str) -> Path:
    safe_id = re.sub(r"[^0-9A-Za-z_.-]+", "_", skill_id).strip("._")
    return _SKILL_MARKDOWN_DIR / f"{safe_id or 'skill'}.md"


def _write_skill_markdowns(skills: list[SkillRecord]) -> None:
    _SKILL_MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    live_names: set[str] = set()
    for skill in skills:
        path = _skill_markdown_path(skill.skill_id)
        live_names.add(path.name)
        markdown = skill.markdown or _default_markdown(
            SkillCandidate(
                title=skill.title,
                pattern=skill.pattern,
                summary=skill.summary,
                steps=skill.steps,
                tools=skill.tools,
                cautions=skill.cautions,
                tags=skill.tags,
                examples=skill.examples,
                confidence=skill.confidence,
            )
        )
        path.write_text(markdown, encoding="utf-8")
    for path in _SKILL_MARKDOWN_DIR.glob("*.md"):
        if path.name not in live_names:
            try:
                path.unlink()
            except OSError:
                pass


def _tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _TOKEN_PATTERN.findall(normalize_message_text(text)):
        lowered = token.casefold()
        if not lowered:
            continue
        tokens.add(lowered)
        chars = "".join(char for char in lowered if "\u4e00" <= char <= "\u9fff")
        max_size = min(len(chars), 4)
        for size in range(2, max_size + 1):
            for start in range(0, len(chars) - size + 1):
                tokens.add(chars[start : start + size])
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _text_tuple(value: Any, *, limit: int) -> tuple[str, ...]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list | tuple | set):
        values = [str(item or "") for item in value]
    else:
        values = []
    return _dedupe_tuple(values, limit=limit)


def _dedupe_tuple(value: Any, *, limit: int) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for item in value or ():
        text = normalize_message_text(str(item or ""))
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text[:500])
        if len(result) >= limit:
            break
    return tuple(result)


def _merge_tuple(
    left: tuple[str, ...],
    right: tuple[str, ...],
    *,
    limit: int,
) -> tuple[str, ...]:
    return _dedupe_tuple([*left, *right], limit=limit)


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if math.isnan(number) or math.isinf(number):
        return 0.0
    return max(0.0, min(number, 1.0))


def _xml_escape(text: str) -> str:
    return (
        normalize_message_text(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


__all__ = [
    "SkillCandidate",
    "SkillMatch",
    "SkillRecord",
    "load_skills",
    "mark_skills_used",
    "render_skill_hints",
    "search_skills",
    "search_skills_semantic",
    "upsert_skill",
]
