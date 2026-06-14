"""File-backed Persona registry for ChatInter chat wording.

Persona is a prompt/style layer only. It must not decide tool routing or
permission. The registry is JSON-backed so P1-6 can land without DB migration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import get_config_value
from .persistence import read_json, state_path, utc_now_iso, write_json
from .route_text import normalize_message_text

_DEFAULT_PERSONA_PATH = state_path("personas.json")
_SCHEMA_VERSION = "chatinter.persona.v1"
_DEFAULT_PERSONA_ID = "default"


@dataclass(frozen=True)
class Persona:
    persona_id: str
    name: str
    prompt: str = ""
    style: str = ""
    tone_examples: tuple[str, ...] = ()
    preset_dialogues: tuple[str, ...] = ()
    bound_tools: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    enabled: bool = True
    created_at: str = ""
    updated_at: str = ""
    source: str = "file"

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_fragment(self) -> str:
        lines: list[str] = []
        if self.prompt:
            lines.append(self.prompt)
        if self.style:
            lines.append(f"人格风格：{self.style}")
        if self.tone_examples:
            lines.append("语气样例：" + " / ".join(self.tone_examples[:4]))
        if self.preset_dialogues:
            lines.append("示例对话：" + " / ".join(self.preset_dialogues[:3]))
        if self.bound_tools:
            lines.append(
                "工具偏好提示："
                + ", ".join(self.bound_tools[:12])
                + "。这只是偏好提示，不得强制调用工具。"
            )
        return "\n".join(lines)

    def to_context_xml(self) -> str:
        lines = [
            f'<persona id="{_xml_escape(self.persona_id)}" source="{self.source}">',
            f"name={_xml_escape(self.name)}",
        ]
        if self.style:
            lines.append(f"style={_xml_escape(self.style)}")
        if self.tags:
            tags = ",".join(_xml_escape(item) for item in self.tags[:8])
            lines.append("tags=" + tags)
        if self.tone_examples:
            lines.append("<tone_examples>")
            lines.extend(_compact_lines(self.tone_examples, limit=4))
            lines.append("</tone_examples>")
        if self.preset_dialogues:
            lines.append("<preset_dialogues>")
            lines.extend(_compact_lines(self.preset_dialogues, limit=3))
            lines.append("</preset_dialogues>")
        if self.bound_tools:
            lines.append(
                "bound_tools="
                + ",".join(_xml_escape(item) for item in self.bound_tools[:12])
            )
        lines.append(
            "rule=Persona only shapes wording and posture; it must not select "
            "tools, override command execution, or invent facts."
        )
        lines.append("</persona>")
        return "\n".join(lines)

    @classmethod
    def from_payload(cls, payload: Any) -> "Persona | None":
        if not isinstance(payload, dict):
            return None
        persona_id = _id_text(payload.get("persona_id") or payload.get("id"))
        name = normalize_message_text(str(payload.get("name", "") or persona_id))
        if not persona_id or not name:
            return None
        return cls(
            persona_id=persona_id,
            name=name,
            prompt=normalize_message_text(str(payload.get("prompt", "") or "")),
            style=normalize_message_text(str(payload.get("style", "") or "")),
            tone_examples=_text_tuple(payload.get("tone_examples"), limit=12),
            preset_dialogues=_text_tuple(payload.get("preset_dialogues"), limit=12),
            bound_tools=_text_tuple(payload.get("bound_tools"), limit=24),
            tags=_text_tuple(payload.get("tags"), limit=16),
            enabled=_parse_bool(payload.get("enabled"), default=True),
            created_at=str(payload.get("created_at", "") or ""),
            updated_at=str(payload.get("updated_at", "") or ""),
            source=str(payload.get("source", "") or "file"),
        )


@dataclass(frozen=True)
class PersonaBinding:
    persona_id: str
    scope: str = "global"
    session_key: str = ""
    group_id: str = ""
    user_id: str = ""
    scenario: str = ""
    priority: int = 0
    enabled: bool = True

    @classmethod
    def from_payload(cls, payload: Any) -> "PersonaBinding | None":
        if not isinstance(payload, dict):
            return None
        persona_id = _id_text(payload.get("persona_id"))
        if not persona_id:
            return None
        return cls(
            persona_id=persona_id,
            scope=_scope_text(payload.get("scope")),
            session_key=normalize_message_text(
                str(payload.get("session_key", "") or "")
            ),
            group_id=normalize_message_text(str(payload.get("group_id", "") or "")),
            user_id=normalize_message_text(str(payload.get("user_id", "") or "")),
            scenario=normalize_message_text(str(payload.get("scenario", "") or "")),
            priority=_int_value(payload.get("priority")),
            enabled=_parse_bool(payload.get("enabled"), default=True),
        )


@dataclass(frozen=True)
class PersonaSelection:
    persona: Persona
    binding: PersonaBinding | None = None
    reason: str = "default"


def resolve_persona(
    *,
    session_key: str = "",
    user_id: str = "",
    group_id: str | None = None,
    scenario: str = "",
) -> PersonaSelection:
    payload = _load_payload()
    personas = {item.persona_id: item for item in _load_personas(payload)}
    bindings = _load_bindings(payload)
    selected = _select_binding(
        bindings,
        session_key=normalize_message_text(session_key),
        user_id=normalize_message_text(user_id),
        group_id=normalize_message_text(group_id or ""),
        scenario=normalize_message_text(scenario),
    )
    if selected is not None:
        persona = personas.get(selected.persona_id)
        if persona is not None and persona.enabled:
            return PersonaSelection(
                persona=persona,
                binding=selected,
                reason=f"binding:{selected.scope}",
            )
    persona = personas.get(_DEFAULT_PERSONA_ID) or legacy_config_persona()
    return PersonaSelection(persona=persona, binding=None, reason="default")


def legacy_config_persona() -> Persona:
    """Wrap old CHAT_STYLE/CUSTOM_PROMPT into a Persona object."""

    chat_style = normalize_message_text(str(get_config_value("CHAT_STYLE", "") or ""))
    custom_prompt = normalize_message_text(
        str(get_config_value("CUSTOM_PROMPT", "") or "")
    )
    prompt = custom_prompt
    style = chat_style or "日式二次元、软萌中带一点傲娇"
    return Persona(
        persona_id=_DEFAULT_PERSONA_ID,
        name="默认人格",
        prompt=prompt,
        style=style,
        tags=("legacy_config",),
        enabled=True,
        source="legacy_config",
    )


def list_personas() -> list[Persona]:
    personas = _load_personas(_load_payload())
    if not personas:
        return [legacy_config_persona()]
    return personas


def upsert_persona(persona: Persona) -> Persona:
    payload = _load_payload()
    personas = {item.persona_id: item for item in _load_personas(payload)}
    now = utc_now_iso()
    old = personas.get(persona.persona_id)
    created_at = old.created_at if old is not None else now
    saved = Persona(
        **{
            **persona.to_payload(),
            "created_at": persona.created_at or created_at,
            "updated_at": now,
        }
    )
    personas[saved.persona_id] = saved
    payload["schema_version"] = _SCHEMA_VERSION
    payload["updated_at"] = now
    payload["personas"] = [item.to_payload() for item in personas.values()]
    write_json(_persona_path(), payload)
    return saved


def _load_payload() -> dict[str, Any]:
    payload = read_json(_persona_path(), {})
    if isinstance(payload, dict):
        return dict(payload)
    return {}


def _persona_path() -> Path:
    configured = normalize_message_text(str(get_config_value("PERSONA_FILE", "") or ""))
    if not configured:
        return _DEFAULT_PERSONA_PATH
    return Path(configured)


def _load_personas(payload: dict[str, Any]) -> list[Persona]:
    raw = payload.get("personas")
    personas: list[Persona] = []
    if isinstance(raw, list):
        for item in raw:
            persona = Persona.from_payload(item)
            if persona is not None:
                personas.append(persona)
    if not personas:
        personas.append(legacy_config_persona())
    return personas


def _load_bindings(payload: dict[str, Any]) -> list[PersonaBinding]:
    raw = payload.get("bindings")
    bindings: list[PersonaBinding] = []
    if isinstance(raw, list):
        for item in raw:
            binding = PersonaBinding.from_payload(item)
            if binding is not None and binding.enabled:
                bindings.append(binding)
    return bindings


def _select_binding(
    bindings: list[PersonaBinding],
    *,
    session_key: str,
    user_id: str,
    group_id: str,
    scenario: str,
) -> PersonaBinding | None:
    scored: list[tuple[int, PersonaBinding]] = []
    for binding in bindings:
        score = _binding_score(
            binding,
            session_key=session_key,
            user_id=user_id,
            group_id=group_id,
            scenario=scenario,
        )
        if score >= 0:
            scored.append((score + binding.priority, binding))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _binding_score(
    binding: PersonaBinding,
    *,
    session_key: str,
    user_id: str,
    group_id: str,
    scenario: str,
) -> int:
    score = 0
    if binding.scenario:
        if binding.scenario != scenario:
            return -1
        score += 8
    if binding.scope == "group":
        if not group_id or binding.group_id != group_id:
            return -1
        return score + 40
    if binding.scope == "user":
        if not user_id or binding.user_id != user_id:
            return -1
        return score + 35
    if binding.scope == "session":
        if binding.session_key:
            if binding.session_key != session_key:
                return -1
        elif binding.group_id and binding.group_id != group_id:
            return -1
        elif binding.user_id and binding.user_id != user_id:
            return -1
        return score + 50
    if binding.scope == "scenario":
        return score + 20 if binding.scenario else -1
    if binding.scope == "global":
        return score
    return -1


def _id_text(value: Any) -> str:
    text = normalize_message_text(str(value or "")).casefold()
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in text)[
        :80
    ].strip("._")


def _scope_text(value: Any) -> str:
    text = normalize_message_text(str(value or "")).casefold()
    if text in {"global", "group", "user", "session", "scenario"}:
        return text
    return "global"


def _text_tuple(value: Any, *, limit: int) -> tuple[str, ...]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list | tuple | set):
        values = [str(item or "") for item in value]
    else:
        values = []
    result: list[str] = []
    for item in values:
        text = normalize_message_text(item)
        if text and text not in result:
            result.append(text[:500])
        if len(result) >= limit:
            break
    return tuple(result)


def _compact_lines(values: tuple[str, ...], *, limit: int) -> list[str]:
    lines: list[str] = []
    for value in values[: max(int(limit or 0), 0)]:
        text = normalize_message_text(str(value or ""))
        if text and text not in lines:
            lines.append(_xml_escape(text[:240]))
    return lines


def _parse_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _xml_escape(value: str) -> str:
    return (
        normalize_message_text(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


__all__ = [
    "Persona",
    "PersonaBinding",
    "PersonaSelection",
    "legacy_config_persona",
    "list_personas",
    "resolve_persona",
    "upsert_persona",
]
