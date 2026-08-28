"""Structured context packing for the unified chat request."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re
from typing import Literal

from .turn_runtime import estimate_text_tokens

ContextRetention = Literal["head", "tail"]

_SECTION_POLICIES: dict[str, tuple[int, ContextRetention, bool]] = {
    "identity": (0, "head", True),
    "reply_layers": (5, "head", True),
    "current_media": (8, "head", True),
    "event": (10, "head", True),
    "guidance": (80, "head", False),
    "relationship": (90, "head", False),
    "memory": (100, "head", False),
    "chatroom": (110, "tail", False),
}
_OPEN_TAG = re.compile(r"^\s*<([A-Za-z_][\w.:-]*)(?:\s[^>]*)?>\s*$")
_CLOSE_TAG = re.compile(r"^\s*</([A-Za-z_][\w.:-]*)>\s*$")
_INLINE_TAG = re.compile(
    r"^(\s*<([A-Za-z_][\w.:-]*)(?:\s[^>]*)?>)(.*)(</\2>\s*)$",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class ChatContextSection:
    name: str
    lines: tuple[str, ...]
    priority: int = 50
    retention: ContextRetention = "head"
    protected: bool = False

    @classmethod
    def create(
        cls,
        name: str,
        lines: Sequence[str],
    ) -> "ChatContextSection | None":
        materialized = tuple(
            str(line or "") for line in lines if str(line or "").strip()
        )
        if not materialized:
            return None
        priority, retention, protected = _SECTION_POLICIES.get(
            name,
            (50, "head", False),
        )
        return cls(
            name=name,
            lines=materialized,
            priority=priority,
            retention=retention,
            protected=protected,
        )


@dataclass(frozen=True, slots=True)
class ChatContextBundle:
    sections: tuple[ChatContextSection, ...] = ()

    @classmethod
    def from_named_sections(
        cls,
        sections: Sequence[ChatContextSection | tuple[str, Sequence[str]]],
    ) -> "ChatContextBundle":
        materialized: list[ChatContextSection] = []
        for item in sections:
            if isinstance(item, ChatContextSection):
                materialized.append(item)
                continue
            name, lines = item
            materialized.extend(context_sections_from_lines(name, lines))
        return cls(tuple(materialized))

    def with_text(self, name: str, text: str) -> "ChatContextBundle":
        value = str(text or "").strip()
        if not value:
            return self
        additions = context_sections_from_lines(name, value.splitlines())
        if not additions:
            return self
        existing = {(section.name, section.lines) for section in self.sections}
        unique = tuple(
            section
            for section in additions
            if (section.name, section.lines) not in existing
        )
        return ChatContextBundle((*self.sections, *unique)) if unique else self

    def transform_text(
        self,
        transform: Callable[[str], str],
    ) -> "ChatContextBundle":
        transformed: list[ChatContextSection] = []
        for section in self.sections:
            value = str(transform("\n".join(section.lines)) or "").strip()
            if not value:
                continue
            for block in _split_top_level_blocks(value.splitlines()):
                transformed.append(
                    ChatContextSection(
                        name=section.name,
                        lines=block,
                        priority=section.priority,
                        retention=section.retention,
                        protected=section.protected,
                    )
                )
        return ChatContextBundle(tuple(transformed))

    def render_lines(self, token_budget: int | None = None) -> tuple[str, ...]:
        if token_budget is None:
            return _flatten_sections(self.sections)
        budget = max(int(token_budget or 0), 0)
        if budget <= 0 or not self.sections:
            return ()
        full = _flatten_sections(self.sections)
        if _lines_token_cost(full) <= budget:
            return full

        selected: dict[int, tuple[str, ...]] = {}
        indexed = sorted(
            enumerate(self.sections),
            key=lambda item: (item[1].priority, item[0]),
        )
        for position, (index, section) in enumerate(indexed):
            current = _render_selected(self.sections, selected)
            full_trial = _render_selected(
                self.sections,
                {**selected, index: section.lines},
            )
            if _lines_token_cost(full_trial) <= budget:
                selected[index] = section.lines
                continue
            separator_cost = estimate_text_tokens("\n") if current else 0
            section_budget = max(
                budget - _lines_token_cost(current) - separator_cost,
                0,
            )
            if not section.protected:
                remaining_sections = len(indexed) - position
                section_budget //= max(remaining_sections, 1)
            trimmed = trim_context_lines(
                section.lines,
                section_budget,
                retention=section.retention,
            )
            if trimmed:
                selected[index] = trimmed
                while (
                    _lines_token_cost(_render_selected(self.sections, selected))
                    > budget
                    and section_budget > 0
                ):
                    section_budget -= 1
                    trimmed = trim_context_lines(
                        section.lines,
                        section_budget,
                        retention=section.retention,
                    )
                    if trimmed:
                        selected[index] = trimmed
                    else:
                        selected.pop(index, None)
                        break
            if _lines_token_cost(_render_selected(self.sections, selected)) >= budget:
                break
        return _render_selected(self.sections, selected)

    def render(self, token_budget: int | None = None) -> str:
        return "\n".join(self.render_lines(token_budget))


def context_sections_from_lines(
    name: str,
    lines: Sequence[str],
) -> tuple[ChatContextSection, ...]:
    sections: list[ChatContextSection] = []
    for block in _split_top_level_blocks(lines):
        section = ChatContextSection.create(name, block)
        if section is not None:
            sections.append(section)
    return tuple(sections)


def trim_context_lines(
    lines: Sequence[str],
    token_budget: int,
    *,
    retention: ContextRetention = "head",
) -> tuple[str, ...]:
    source = tuple(str(line or "") for line in lines if str(line or "").strip())
    budget = max(int(token_budget or 0), 0)
    if not source or budget <= 0:
        return ()
    if _lines_token_cost(source) <= budget:
        return source

    opening: tuple[str, ...] = ()
    closing: tuple[str, ...] = ()
    body = source
    if len(source) >= 2 and _matching_wrapper(source[0], source[-1]):
        opening = (source[0],)
        closing = (source[-1],)
        body = source[1:-1]
        if _lines_token_cost((*opening, *closing)) > budget:
            return ()

    blocks = (
        _split_top_level_blocks(body)
        if opening
        else tuple((line,) for line in body)
    )
    ordered = list(enumerate(blocks))
    if retention == "tail":
        ordered.reverse()
    selected: dict[int, tuple[str, ...]] = {}
    for index, block in ordered:
        trial_selected = {**selected, index: block}
        trial_body = tuple(
            line
            for key in sorted(trial_selected)
            for line in trial_selected[key]
        )
        trial = (*opening, *trial_body, *closing)
        if _lines_token_cost(trial) <= budget:
            selected[index] = block
            continue
        fixed_body = tuple(
            line for key in sorted(selected) for line in selected[key]
        )
        clipped = _trim_block_to_fit(
            block,
            before=(*opening, *fixed_body),
            after=closing,
            token_budget=budget,
            retention=retention,
        )
        if clipped:
            selected[index] = clipped
        break

    fitted_body = tuple(
        line for key in sorted(selected) for line in selected[key]
    )
    result = (*opening, *fitted_body, *closing)
    return result if _lines_token_cost(result) <= budget else ()


def _trim_block_to_fit(
    block: Sequence[str],
    *,
    before: Sequence[str],
    after: Sequence[str],
    token_budget: int,
    retention: ContextRetention,
) -> tuple[str, ...]:
    materialized = tuple(block)
    if len(materialized) == 1:
        clipped = _clip_line_to_fit(
            materialized[0],
            before=before,
            after=after,
            token_budget=token_budget,
        )
        return (clipped,) if clipped else ()
    if not _matching_wrapper(materialized[0], materialized[-1]):
        return ()

    low = 1
    high = max(token_budget, 1)
    best: tuple[str, ...] = ()
    while low <= high:
        block_budget = (low + high) // 2
        candidate = trim_context_lines(
            materialized,
            block_budget,
            retention=retention,
        )
        combined_fits = (
            _lines_token_cost((*before, *candidate, *after)) <= token_budget
        )
        if candidate and combined_fits:
            best = candidate
            low = block_budget + 1
        else:
            high = block_budget - 1
    return best


def _clip_line_to_fit(
    line: str,
    *,
    before: Sequence[str],
    after: Sequence[str],
    token_budget: int,
) -> str:
    raw = str(line or "")
    inline = _INLINE_TAG.match(raw)
    prefix = inline.group(1) if inline else ""
    content = inline.group(3) if inline else raw
    suffix = inline.group(4) if inline else ""

    def render(retained_chars: int) -> str:
        head_chars = max(int(retained_chars * 0.7), 0)
        tail_chars = max(retained_chars - head_chars, 0)
        omitted = max(len(content) - head_chars - tail_chars, 0)
        head = content[:head_chars].rstrip()
        tail = content[-tail_chars:].lstrip() if tail_chars else ""
        marker = f" ...[{omitted} chars omitted]... "
        return f"{prefix}{head}{marker}{tail}{suffix}"

    low = 0
    high = len(content)
    best = ""
    while low <= high:
        retained = (low + high) // 2
        candidate = render(retained)
        if _lines_token_cost((*before, candidate, *after)) <= token_budget:
            best = candidate
            low = retained + 1
        else:
            high = retained - 1
    return best


def _split_top_level_blocks(lines: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    blocks: list[tuple[str, ...]] = []
    current: list[str] = []
    active_tag = ""
    for raw_line in lines:
        line = str(raw_line or "")
        if not line.strip():
            continue
        if not current:
            opening = _OPEN_TAG.match(line)
            if opening and not _INLINE_TAG.match(line):
                current = [line]
                active_tag = opening.group(1)
            else:
                blocks.append((line,))
            continue
        current.append(line)
        closing = _CLOSE_TAG.match(line)
        if closing and closing.group(1) == active_tag:
            blocks.append(tuple(current))
            current = []
            active_tag = ""
    if current:
        blocks.append(tuple(current))
    return tuple(blocks)


def _matching_wrapper(first: str, last: str) -> bool:
    opening = _OPEN_TAG.match(first)
    closing = _CLOSE_TAG.match(last)
    return bool(opening and closing and opening.group(1) == closing.group(1))


def _lines_token_cost(lines: Sequence[str]) -> int:
    return estimate_text_tokens("\n".join(lines)) if lines else 0


def _flatten_sections(sections: Sequence[ChatContextSection]) -> tuple[str, ...]:
    return tuple(line for section in sections for line in section.lines)


def _render_selected(
    sections: Sequence[ChatContextSection],
    selected: dict[int, tuple[str, ...]],
) -> tuple[str, ...]:
    return tuple(
        line
        for index in range(len(sections))
        for line in selected.get(index, ())
    )


__all__ = [
    "ChatContextBundle",
    "ChatContextSection",
    "context_sections_from_lines",
    "trim_context_lines",
]
