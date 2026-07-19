from __future__ import annotations

from urllib.parse import quote


def conversation_session_key(session) -> str:
    """Return a bot- and adapter-scoped key for one visible chat scene."""

    scope = getattr(getattr(session, "scope", ""), "value", None) or getattr(
        session, "scope", "unknown"
    )
    self_id = getattr(session, "self_id", "") or "unknown"
    scene_path = getattr(session, "scene_path", "") or legacy_session_key(session)
    return "chat:" + ":".join(
        quote(str(value), safe="") for value in (scope, self_id, scene_path)
    )


def legacy_session_key(session) -> str:
    group = getattr(session, "group", None)
    if group is not None and getattr(group, "id", None) is not None:
        return str(group.id)
    user = getattr(session, "user", None)
    return str(getattr(user, "id", "") or "")


__all__ = ["conversation_session_key", "legacy_session_key"]
