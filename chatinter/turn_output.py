"""Output envelope helpers for one ChatInter turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from zhenxun.services import logger

from .route_text import normalize_message_text


class ChannelName(str, Enum):
    ANALYSIS = "analysis"
    COMMENTARY = "commentary"
    FINAL = "final"


@dataclass
class TurnChannelEnvelope:
    analysis: list[str] = field(default_factory=list)
    commentary: list[str] = field(default_factory=list)
    final: str = ""

    def add(self, channel: ChannelName, content: str) -> None:
        raw_text = str(content or "")
        if channel is ChannelName.FINAL:
            text = raw_text.strip()
            if text:
                self.final = text
            return

        text = normalize_message_text(raw_text)
        if not text:
            return
        if channel is ChannelName.ANALYSIS:
            self.analysis.append(text)
        else:
            self.commentary.append(text)


def log_turn_channels(envelope: TurnChannelEnvelope) -> None:
    if envelope.analysis:
        logger.debug("[ChatInter][analysis] " + " | ".join(envelope.analysis))
    if envelope.commentary:
        logger.debug("[ChatInter][commentary] " + " | ".join(envelope.commentary))


__all__ = [
    "ChannelName",
    "TurnChannelEnvelope",
    "log_turn_channels",
]
