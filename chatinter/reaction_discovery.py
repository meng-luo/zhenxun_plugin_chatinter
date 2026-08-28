"""Persistent group-level threshold discovery for reaction images."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable
import json
import os
from pathlib import Path
import time
from typing import Any

from .llm_compat import LLMContentPart
from .log_compat import logger
from .reaction_image import inspect_reaction_image
from .reaction_models import ReactionSettings
from .reaction_semantics import analyze_reaction_bytes
from .reaction_store import ReactionStore

_STATE_VERSION = 1
_WINDOW_SECONDS = 24 * 60 * 60
_INFLIGHT_LEASE_SECONDS = 5 * 60
_MIN_DISCOVERY_CONFIDENCE = 0.85
_MAX_OBSERVATION_BUCKETS = 5_000
_MAX_DECISIONS = 2_000


class ReactionDiscoveryLedger:
    def __init__(
        self,
        root: Path,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = root.resolve()
        self.path = self.root / "chatinter_reaction_discovery.json"
        self._clock = clock
        self._lock = asyncio.Lock()
        self._state: dict[str, Any] | None = None

    async def observe(
        self,
        *,
        settings: ReactionSettings,
        store: ReactionStore,
        group_id: str,
        sender_id: str,
        message_id: str,
        image_parts: list[LLMContentPart],
    ) -> None:
        seen_fingerprints: set[str] = set()
        for part in image_parts:
            content = _part_bytes(part)
            if not content:
                continue
            info = await asyncio.to_thread(inspect_reaction_image, content)
            if info is None or info.visual_fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(info.visual_fingerprint)
            if await store.contains_identity(
                info.content_sha256,
                info.visual_fingerprint,
            ):
                continue
            reserved = await self._register(
                group_id=group_id,
                sender_id=sender_id,
                message_id=message_id,
                fingerprint=info.visual_fingerprint,
            )
            if not reserved:
                continue
            await self._analyze_reserved(
                settings=settings,
                store=store,
                content=content,
                extension=info.extension,
                fingerprint=info.visual_fingerprint,
            )

    async def _register(
        self,
        *,
        group_id: str,
        sender_id: str,
        message_id: str,
        fingerprint: str,
    ) -> bool:
        now = self._clock()
        async with self._lock:
            state = await self._load_locked()
            self._prune_locked(state, now)
            decisions = state.setdefault("decisions", {})
            if isinstance(decisions, dict) and decisions.get(fingerprint) == "rejected":
                return False
            observations = state.setdefault("observations", {})
            if not isinstance(observations, dict):
                observations = {}
                state["observations"] = observations
            key = f"{group_id}:{fingerprint}"
            bucket = observations.get(key)
            if not isinstance(bucket, dict):
                bucket = {
                    "group_id": group_id,
                    "fingerprint": fingerprint,
                    "events": [],
                    "inflight_since": 0.0,
                }
                observations[key] = bucket
            events = bucket.get("events")
            if not isinstance(events, list):
                events = []
                bucket["events"] = events
            event_key = str(message_id or "").strip()
            if event_key and any(
                isinstance(item, dict)
                and str(item.get("message_id") or "") == event_key
                for item in events
            ):
                await self._save_locked()
                return False
            events.append(
                {
                    "sender_id": sender_id,
                    "message_id": event_key,
                    "timestamp": now,
                }
            )
            sender_counts = Counter(
                str(item.get("sender_id") or "")
                for item in events
                if isinstance(item, dict) and str(item.get("sender_id") or "")
            )
            reached = (
                len(sender_counts) >= 4
                or max(sender_counts.values(), default=0) >= 3
            )
            inflight_since = _safe_float(bucket.get("inflight_since"))
            inflight = (
                inflight_since > 0
                and now - inflight_since < _INFLIGHT_LEASE_SECONDS
            )
            if reached and not inflight:
                bucket["inflight_since"] = now
            self._limit_observations_locked(observations)
            await self._save_locked()
            return reached and not inflight

    async def _analyze_reserved(
        self,
        *,
        settings: ReactionSettings,
        store: ReactionStore,
        content: bytes,
        extension: str,
        fingerprint: str,
    ) -> None:
        try:
            analysis = await analyze_reaction_bytes(content, hint=extension)
        except asyncio.CancelledError:
            await self._finish(fingerprint, decision="retry")
            raise
        except Exception as exc:
            logger.debug(f"chatinter reaction discovery analysis failed: {exc}")
            await self._finish(fingerprint, decision="retry")
            return
        if analysis is None:
            await self._finish(fingerprint, decision="retry")
            return
        if not analysis.is_reaction or analysis.confidence < _MIN_DISCOVERY_CONFIDENCE:
            await self._finish(fingerprint, decision="rejected")
            return
        try:
            record = await store.add_collected(
                content,
                extension=extension,
                caption=analysis.caption,
                tags=analysis.tags,
                visible_text=analysis.visible_text,
                reply_intents=analysis.reply_intents,
                usage_scenarios=analysis.usage_scenarios,
                tones=analysis.tones,
                actions=analysis.actions,
                target_relation=analysis.target_relation,
                visual_fingerprint=fingerprint,
            )
        except asyncio.CancelledError:
            await self._finish(fingerprint, decision="retry")
            raise
        except Exception as exc:
            logger.debug(f"chatinter reaction discovery storage failed: {exc}")
            await self._finish(fingerprint, decision="retry")
            return
        if record is None:
            await self._finish(fingerprint, decision="retry")
            return
        await self._finish(fingerprint, decision="accepted")
        logger.info(
            "ChatInter 群聊表情已自动收录："
            f"{record.reaction_id} category={record.category}"
        )

    async def _finish(self, fingerprint: str, *, decision: str) -> None:
        async with self._lock:
            state = await self._load_locked()
            observations = state.setdefault("observations", {})
            if isinstance(observations, dict):
                for key in list(observations):
                    bucket = observations.get(key)
                    if not isinstance(bucket, dict):
                        continue
                    if str(bucket.get("fingerprint") or "") != fingerprint:
                        continue
                    if decision == "retry":
                        bucket["inflight_since"] = 0.0
                    else:
                        observations.pop(key, None)
            if decision in {"accepted", "rejected"}:
                decisions = state.setdefault("decisions", {})
                if not isinstance(decisions, dict):
                    decisions = {}
                    state["decisions"] = decisions
                decisions[fingerprint] = decision
                while len(decisions) > _MAX_DECISIONS:
                    decisions.pop(next(iter(decisions)))
            await self._save_locked()

    async def _load_locked(self) -> dict[str, Any]:
        if self._state is not None:
            return self._state
        try:
            value = await asyncio.to_thread(_read_state, self.path)
        except OSError:
            value = {}
        self._state = value if isinstance(value, dict) else {}
        self._state["version"] = _STATE_VERSION
        self._state.setdefault("observations", {})
        self._state.setdefault("decisions", {})
        return self._state

    async def _save_locked(self) -> None:
        if self._state is None:
            return
        await asyncio.to_thread(_write_state, self.path, self._state)

    def _prune_locked(self, state: dict[str, Any], now: float) -> None:
        observations = state.get("observations")
        if not isinstance(observations, dict):
            state["observations"] = {}
            return
        cutoff = now - _WINDOW_SECONDS
        for key in list(observations):
            bucket = observations.get(key)
            if not isinstance(bucket, dict):
                observations.pop(key, None)
                continue
            events = bucket.get("events")
            if not isinstance(events, list):
                events = []
            bucket["events"] = [
                item
                for item in events
                if isinstance(item, dict)
                and _safe_float(item.get("timestamp")) >= cutoff
            ]
            if not bucket["events"]:
                observations.pop(key, None)
                continue
            inflight_since = _safe_float(bucket.get("inflight_since"))
            if inflight_since and now - inflight_since >= _INFLIGHT_LEASE_SECONDS:
                bucket["inflight_since"] = 0.0

    @staticmethod
    def _limit_observations_locked(observations: dict[str, Any]) -> None:
        while len(observations) > _MAX_OBSERVATION_BUCKETS:
            oldest_key = min(
                observations,
                key=lambda key: _bucket_timestamp(observations.get(key)),
            )
            observations.pop(oldest_key, None)


def _part_bytes(part: LLMContentPart) -> bytes:
    if part.type != "image" or not part.image_source:
        return b""
    import base64

    try:
        return base64.b64decode(part.image_source, validate=True)
    except (TypeError, ValueError):
        return b""


def _bucket_timestamp(value: Any) -> float:
    if not isinstance(value, dict):
        return 0.0
    events = value.get("events")
    if not isinstance(events, list):
        return 0.0
    return min(
        (
            _safe_float(item.get("timestamp"))
            for item in events
            if isinstance(item, dict)
        ),
        default=0.0,
    )


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _read_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["ReactionDiscoveryLedger"]
