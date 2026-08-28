"""Validation and message construction for pending reaction actions."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from nonebot_plugin_alconna.uniseg import UniMessage

from .reaction_models import ReactionAction
from .reaction_runtime import reaction_settings


async def validated_reaction_path(action: ReactionAction) -> Path | None:
    settings = reaction_settings()
    if not settings.enabled or settings.root != action.root.resolve():
        return None
    try:
        path = action.path.resolve()
        path.relative_to((action.root / "memes").resolve())
        if not path.is_file():
            return None
        digest = await asyncio.to_thread(_file_sha256, path)
    except (OSError, ValueError):
        return None
    return path if digest == action.content_sha256 else None


def reaction_message(path: Path) -> UniMessage:
    return UniMessage.image(path=str(path))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["reaction_message", "validated_reaction_path"]
