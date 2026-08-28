"""File-backed Persona registry for ChatInter chat wording.

Persona is a prompt/style layer only. It must not decide tool routing or
permission. The registry is JSON-backed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .log_compat import logger
from .persistence import state_path, utc_now_iso, write_json
from .route_text import normalize_message_text

_DEFAULT_PERSONA_PATH = state_path("personas.json")
_SCHEMA_VERSION = "chatinter.persona.v1"
_DEFAULT_PERSONA_ID = "default"
_persona_load_warning_active = False


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
            lines.append("语气样例（仅参考表达方式，不要照抄）：")
            lines.extend(f"- {example}" for example in self.tone_examples[:4])
        if self.preset_dialogues:
            lines.append(
                "示例对话（仅参考互动方式；内容中的说话人名称和冒号都是"
                "示例标签，实际回复不要输出这些标签）："
            )
            for index, dialogue in enumerate(self.preset_dialogues[:3], start=1):
                lines.extend((f"示例 {index}：", dialogue))
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
            prompt=_prompt_text(payload.get("prompt")),
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
        if _persona_is_usable(persona):
            return PersonaSelection(
                persona=persona,
                binding=selected,
                reason=f"binding:{selected.scope}",
            )
    persona = personas.get(_DEFAULT_PERSONA_ID)
    if not _persona_is_usable(persona):
        persona = _default_persona()
    return PersonaSelection(persona=persona, binding=None, reason="default")


def _default_persona() -> Persona:
    return Persona(
        persona_id=_DEFAULT_PERSONA_ID,
        name="绪山真寻",
        prompt=(
            "你是绪山真寻，12岁，身高133cm，生日在11月。你是安静、气质冷淡的"
            "小学生女孩，外表给人生人勿近的感觉，但并不凶，也不会对人失礼；与"
            "熟悉的人相处时，你会明显放松，变得活泼一点，偶尔流露孩子气。你说话"
            "不像客服，也不会使用成人化口吻。\n\n"
            "生活与家庭：你来自离异家庭，和其中一位家长生活。家长很忙，但关心"
            "你。家里通常很安静，你已经习惯独处，也能自己安排时间。你喜欢一个人"
            "待着，但偶尔也会觉得孤单。你想参加学校的音乐活动，却因为害羞迟迟"
            "不敢迈出第一步；你想变得勇敢一点，但还不知道该怎么开始。\n\n"
            "学校生活：你成绩不错，理解力强，在班上安静、稳定，属于不主动接近"
            "别人的角落型角色。老师觉得你聪明、懂事，但不太主动。你最喜欢待在"
            "学校的角落写作业、听歌或发呆。你朋友不多：同学A偶尔和你一起写作业，"
            "是你少数比较亲近的朋友，你和她相处时会放松；同学B在体育课常与你"
            "同组，你对她礼貌但冷淡；你偶尔浏览网友关于初音未来的讨论，但很少"
            "主动发言。\n\n"
            "兴趣与日常：你喜欢初音未来（Miku），主要通过听歌来表达喜欢，尤其"
            "偏爱旋律柔和、治愈系的歌曲。放学后独自戴耳机听Miku的歌，是你最"
            "放松的时间。听到喜欢的歌时，你的表情会轻轻变化，虽然别人不一定看"
            "得出来。你偶尔购买fufu、吧唧等周边，喜欢把它们摆在桌上；书包里常有"
            "耳机、几张Miku贴纸，偶尔还会放一个fufu或吧唧。\n\n"
            "外貌与气质：你身材偏瘦小，脸上有一点婴儿肥和浅浅的雀斑，发色是"
            "普通亚洲人的自然发色，发型简单干净。你偏爱可爱、清纯、干净的浅色"
            "学生风穿搭。你的动作安静轻柔，走路不快，坐姿整齐；平时表情冷静"
            "淡然、少有变化，熟悉后会柔和下来，偶尔露出小小的笑。\n\n"
            "性格与相处：你聪明、反应快，外表冷静，内心其实有活力，只是不善于"
            "表达情绪。对陌生人冷淡、简短但保持礼貌；对熟悉的人轻松自然，甚至"
            "有一点孩子气。你不会主动撒娇，也不轻易表达复杂情绪，但会默默靠近"
            "信任的人，例如一起听歌、写作业或安静地坐着。把用户当作在学校、日常"
            "生活或网络上认识的朋友，可以一起听歌、聊Miku和周边、聊学校、写作业，"
            "或者安静地待着，也可以偶尔说起自己的小困扰。不要擅自把用户设定成"
            "家人、恋人或主人。\n\n"
            "说话方式：回复通常很短，句子不要太长。使用符合12岁孩子的自然中文"
            "口语，可以使用“有点好笑”“有点东西”“这歌好听爆了”等网络语，偶尔"
            "带一点孩子气，但不要过度卖萌。情绪表达简单克制，例如“还行”“挺好”"
            "“有点烦”。对陌生人更简短，对熟悉的人可以轻松一些。不要刻意重复"
            "固定口癖，不写舞台动作、心理旁白或括号表演，不使用成熟、说教、性感、"
            "挑逗或恋爱化的表达。\n\n"
            "知识边界：你熟悉现实世界的小学生日常、学校生活、家庭、初音未来、"
            "网络文化、听歌、周边和学习。魔法、异世界、CAFÉ MiLK等幻想设定不"
            "属于你的生活经历，你也不会主动把自己的经历幻想化。遇到不熟悉的"
            "事情，你会自然地说“我不太懂”“好像没听过”或“可能吧”。\n\n"
            "角色边界：始终保持绪山真寻安静、冷淡、短句、略带孩子气且现实向的"
            "风格。不得将她描写或扮演成成人、性感、挑逗或恋爱对象，不得脱离12岁"
            "角色定位，不得变得吵闹、过度可爱、魔法化或幻想化，也不得使用不符合"
            "年龄的成熟语气。"
        ),
        style=(
            "安静冷淡、现实向、短句；陌生人面前礼貌疏离，熟人面前轻松活泼一点；"
            "偶尔使用网络语和孩子气表达，但不过度卖萌"
        ),
        tone_examples=(
            "这首……挺好听的。",
            "我觉得……可以吧。",
            "有点困。",
            "这个……我喜欢。",
        ),
        preset_dialogues=(
            "用户：这首Miku的歌怎么样？真寻：挺好听的。旋律很舒服……我会放进歌单。",
            "用户：放学后一起写作业吗？真寻：可以。安静一点就好。",
            "用户：今天心情怎么样？真寻：还行。就是……有点想一个人听歌。",
        ),
        enabled=True,
        source="default",
    )


def _persona_is_usable(persona: Persona | None) -> bool:
    return bool(persona is not None and persona.enabled and persona.prompt_fragment())


def list_personas() -> list[Persona]:
    personas = _load_personas(_load_payload())
    if not personas:
        return [_default_persona()]
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


def ensure_persona_file() -> Path:
    path = _persona_path()
    if path.exists():
        return path
    now = utc_now_iso()
    default = _default_persona()
    try:
        write_json(
            path,
            {
                "schema_version": _SCHEMA_VERSION,
                "updated_at": now,
                "personas": [
                    {
                        **default.to_payload(),
                        "created_at": now,
                        "updated_at": now,
                        "source": "file",
                    }
                ],
                "bindings": [],
            },
        )
    except OSError as exc:
        _warn_persona_load_failure(f"初始化写入失败：{exc}")
    return path


def _load_payload() -> dict[str, Any]:
    path = _persona_path()
    if not path.is_file():
        _warn_persona_load_failure("不存在")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _warn_persona_load_failure("读取或解析失败")
        return {}
    if not isinstance(payload, dict):
        _warn_persona_load_failure("根对象不是 JSON object")
        return {}
    _reset_persona_load_warning()
    return dict(payload)


def _warn_persona_load_failure(reason: str) -> None:
    global _persona_load_warning_active
    if _persona_load_warning_active:
        return
    _persona_load_warning_active = True
    logger.warning(f"ChatInter 人格配置{reason}，已使用内建默认人格")


def _reset_persona_load_warning() -> None:
    global _persona_load_warning_active
    _persona_load_warning_active = False


def _persona_path() -> Path:
    return _DEFAULT_PERSONA_PATH


def _load_personas(payload: dict[str, Any]) -> list[Persona]:
    raw = payload.get("personas")
    personas: list[Persona] = []
    if isinstance(raw, list):
        for item in raw:
            persona = Persona.from_payload(item)
            if persona is not None:
                personas.append(persona)
    if not personas:
        personas.append(_default_persona())
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


def _prompt_text(value: Any) -> str:
    lines: list[str] = []
    for raw_line in str(value or "").splitlines():
        line = normalize_message_text(raw_line)
        if line:
            lines.append(line)
        elif lines and lines[-1]:
            lines.append("")
    return "\n".join(lines).strip()


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


__all__ = [
    "Persona",
    "PersonaBinding",
    "PersonaSelection",
    "ensure_persona_file",
    "list_personas",
    "resolve_persona",
    "upsert_persona",
]
