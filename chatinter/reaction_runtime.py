"""Lifecycle for mixed-chat reaction images."""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
import time
from typing import Any

from .config import get_reaction_image_settings
from .log_compat import logger
from .reaction_discovery import ReactionDiscoveryLedger
from .reaction_models import ReactionSettings
from .reaction_semantics import analyze_reaction_file_detailed
from .reaction_store import ReactionStore
from .utils.multimodal import extract_chat_images_from_message

_ENRICHMENT_BACKOFF_SECONDS = (60.0, 300.0, 900.0, 1800.0)
_REFRESHABLE_PROVENANCE = frozenset(
    {"startup_import", "auto_discovery"}
)
_STORES: dict[Path, ReactionStore] = {}
_DISCOVERY_LEDGERS: dict[Path, ReactionDiscoveryLedger] = {}
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()
_ENRICHMENT_ROOTS: set[Path] = set()
_ENRICHMENT_RETRY_AFTER: dict[Path, float] = {}
_ENRICHMENT_FAILURE_COUNTS: dict[Path, int] = {}
_ENRICHMENT_RETRY_TASKS: dict[Path, asyncio.Task[Any]] = {}
_INITIALIZATION_TASK: asyncio.Task[Any] | None = None


def reaction_settings() -> ReactionSettings:
    configured = get_reaction_image_settings()
    return ReactionSettings(
        enabled=bool(configured["enabled"]),
        root=Path(str(configured["directory"])).expanduser().resolve(),
        import_root=Path(str(configured["import_directory"])).expanduser().resolve(),
        semantic_search=bool(configured["semantic_search"]),
        auto_caption=bool(configured["auto_caption"]),
        auto_discovery=bool(configured["auto_discovery"]),
    )


def reaction_store(settings: ReactionSettings) -> ReactionStore:
    store = _STORES.get(settings.root)
    if store is None:
        store = ReactionStore(settings.root)
        _STORES[settings.root] = store
    return store


def discovery_ledger(settings: ReactionSettings) -> ReactionDiscoveryLedger:
    ledger = _DISCOVERY_LEDGERS.get(settings.root)
    if ledger is None:
        ledger = ReactionDiscoveryLedger(settings.root)
        _DISCOVERY_LEDGERS[settings.root] = ledger
    return ledger


async def start_reaction_runtime() -> None:
    global _INITIALIZATION_TASK
    settings = reaction_settings()
    if not settings.enabled:
        return
    if _INITIALIZATION_TASK is not None and not _INITIALIZATION_TASK.done():
        return
    _INITIALIZATION_TASK = _schedule(_initialize_reactions(settings))


async def shutdown_reaction_runtime() -> None:
    global _INITIALIZATION_TASK
    tasks = list(_BACKGROUND_TASKS)
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _BACKGROUND_TASKS.clear()
    _ENRICHMENT_ROOTS.clear()
    _ENRICHMENT_RETRY_AFTER.clear()
    _ENRICHMENT_FAILURE_COUNTS.clear()
    _ENRICHMENT_RETRY_TASKS.clear()
    _STORES.clear()
    _DISCOVERY_LEDGERS.clear()
    _INITIALIZATION_TASK = None


async def _initialize_reactions(settings: ReactionSettings) -> None:
    from .reaction_bootstrap import install_default_reaction_pack
    from .reaction_importer import import_reaction_directory

    await install_default_reaction_pack(settings.root)
    current = reaction_settings()
    if not current.enabled or current.root != settings.root:
        return
    store = reaction_store(settings)
    await store.records(force=True)
    await import_reaction_directory(settings, store)
    records = await store.records(force=True)
    status_counts = Counter(record.status for record in records)
    logger.info(
        "ChatInter 表情库已就绪："
        f"total={len(records)} ready={status_counts.get('ready', 0)} "
        f"pending={status_counts.get('pending', 0)} "
        f"rejected={status_counts.get('rejected', 0)} "
        f"error={status_counts.get('error', 0)}"
    )
    schedule_semantic_enrichment(settings, store)


def schedule_semantic_enrichment(
    settings: ReactionSettings,
    store: ReactionStore,
) -> None:
    if not settings.enabled or not settings.auto_caption:
        return
    if settings.root in _ENRICHMENT_ROOTS:
        return
    retry_task = _ENRICHMENT_RETRY_TASKS.get(settings.root)
    if retry_task is not None and not retry_task.done():
        return
    remaining = _ENRICHMENT_RETRY_AFTER.get(settings.root, 0.0) - time.monotonic()
    if remaining > 0:
        _schedule_enrichment_retry(settings, store, remaining)
        return
    _ENRICHMENT_ROOTS.add(settings.root)
    task = _schedule(_enrich_pending(settings, store))
    if task is None:
        _ENRICHMENT_ROOTS.discard(settings.root)


def schedule_reaction_observation(
    *,
    group_id: str,
    sender_id: str,
    message_id: str,
    message: Any,
) -> None:
    settings = reaction_settings()
    if not settings.enabled or not settings.auto_discovery:
        return
    _schedule(
        _observe_group_message(
            settings=settings,
            group_id=str(group_id or ""),
            sender_id=str(sender_id or ""),
            message_id=str(message_id or ""),
            message=message,
        )
    )


async def _observe_group_message(
    *,
    settings: ReactionSettings,
    group_id: str,
    sender_id: str,
    message_id: str,
    message: Any,
) -> None:
    current = reaction_settings()
    if (
        not current.enabled
        or not current.auto_discovery
        or current.root != settings.root
    ):
        return
    extraction = await extract_chat_images_from_message(message)
    if not extraction.image_parts:
        return
    store = reaction_store(settings)
    await discovery_ledger(settings).observe(
        settings=settings,
        store=store,
        group_id=group_id,
        sender_id=sender_id,
        message_id=message_id,
        image_parts=extraction.image_parts,
    )


async def _enrich_pending(
    settings: ReactionSettings,
    store: ReactionStore,
) -> None:
    retry_pending = False
    processed = 0
    consecutive_invalid_responses = 0
    try:
        records = await store.records()
        candidates = tuple(
            record
            for record in records
            if record.status == "pending"
            or (
                record.status == "ready"
                and record.semantic_version < 2
                and record.provenance in _REFRESHABLE_PROVENANCE
            )
        )
        for record in candidates:
            current = reaction_settings()
            if (
                not current.enabled
                or not current.auto_caption
                or current.root != settings.root
            ):
                break
            path = await store.resolve(record)
            if path is None:
                if record.status == "pending":
                    await store.mark_semantic_error(record.content_sha256)
                continue
            outcome = await analyze_reaction_file_detailed(
                path,
                category=record.category,
                category_description=record.category_description,
            )
            if outcome.status == "invalid_image":
                if record.status == "pending":
                    await store.mark_semantic_error(record.content_sha256)
                continue
            if outcome.status in {"no_model", "provider_error"}:
                retry_pending = True
                break
            if outcome.status == "invalid_response":
                retry_pending = True
                consecutive_invalid_responses += 1
                if consecutive_invalid_responses >= 3:
                    retry_pending = True
                    break
                continue
            analysis = outcome.analysis
            if analysis is None:
                consecutive_invalid_responses += 1
                if consecutive_invalid_responses >= 3:
                    retry_pending = True
                    break
                continue
            consecutive_invalid_responses = 0
            _ENRICHMENT_FAILURE_COUNTS.pop(settings.root, None)
            _ENRICHMENT_RETRY_AFTER.pop(settings.root, None)
            await store.update_semantics(
                record.content_sha256,
                caption=analysis.caption,
                tags=analysis.tags,
                visible_text=analysis.visible_text,
                reply_intents=analysis.reply_intents,
                usage_scenarios=analysis.usage_scenarios,
                tones=analysis.tones,
                actions=analysis.actions,
                target_relation=analysis.target_relation,
                accepted=analysis.is_reaction and analysis.confidence >= 0.55,
            )
            processed += 1
            if processed % 25 == 0:
                logger.info(
                    f"ChatInter 表情语义补全进度：processed={processed}"
                )
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        retry_pending = True
        logger.debug(f"chatinter reaction enrichment stopped: {exc}")
    finally:
        _ENRICHMENT_ROOTS.discard(settings.root)
        if retry_pending:
            failures = _ENRICHMENT_FAILURE_COUNTS.get(settings.root, 0) + 1
            _ENRICHMENT_FAILURE_COUNTS[settings.root] = failures
            delay = _ENRICHMENT_BACKOFF_SECONDS[
                min(failures - 1, len(_ENRICHMENT_BACKOFF_SECONDS) - 1)
            ]
            _ENRICHMENT_RETRY_AFTER[settings.root] = time.monotonic() + delay
            _schedule_enrichment_retry(settings, store, delay)
        else:
            _ENRICHMENT_RETRY_AFTER.pop(settings.root, None)
            _ENRICHMENT_FAILURE_COUNTS.pop(settings.root, None)
        try:
            records = await store.records(force=True)
            status_counts = Counter(record.status for record in records)
            logger.info(
                "ChatInter 表情语义补全结束："
                f"processed={processed} total={len(records)} "
                f"ready={status_counts.get('ready', 0)} "
                f"pending={status_counts.get('pending', 0)} "
                f"rejected={status_counts.get('rejected', 0)} "
                f"error={status_counts.get('error', 0)} "
                f"retry_pending={retry_pending}"
            )
        except Exception as exc:
            logger.debug(f"ChatInter 表情语义补全状态读取失败：{exc}")


def _schedule_enrichment_retry(
    settings: ReactionSettings,
    store: ReactionStore,
    delay: float,
) -> None:
    current = _ENRICHMENT_RETRY_TASKS.get(settings.root)
    if current is not None and not current.done():
        return
    task = _schedule(_delayed_enrichment_retry(settings, store, max(delay, 0.0)))
    if task is not None:
        _ENRICHMENT_RETRY_TASKS[settings.root] = task


async def _delayed_enrichment_retry(
    settings: ReactionSettings,
    store: ReactionStore,
    delay: float,
) -> None:
    try:
        await asyncio.sleep(delay)
        _ENRICHMENT_RETRY_AFTER.pop(settings.root, None)
    finally:
        current = asyncio.current_task()
        if _ENRICHMENT_RETRY_TASKS.get(settings.root) is current:
            _ENRICHMENT_RETRY_TASKS.pop(settings.root, None)
    current_settings = reaction_settings()
    if (
        current_settings.enabled
        and current_settings.auto_caption
        and current_settings.root == settings.root
    ):
        schedule_semantic_enrichment(settings, store)


def _schedule(coroutine: Any) -> asyncio.Task[Any] | None:
    try:
        task = asyncio.create_task(coroutine)
    except RuntimeError:
        if hasattr(coroutine, "close"):
            coroutine.close()
        return None
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_finish_background_task)
    return task


def _finish_background_task(task: asyncio.Task[Any]) -> None:
    _BACKGROUND_TASKS.discard(task)
    if task.cancelled():
        return
    try:
        error = task.exception()
    except (asyncio.CancelledError, asyncio.InvalidStateError):
        return
    if error is not None:
        logger.warning(f"ChatInter 表情后台任务失败：{error}")


__all__ = [
    "discovery_ledger",
    "reaction_settings",
    "reaction_store",
    "schedule_reaction_observation",
    "schedule_semantic_enrichment",
    "shutdown_reaction_runtime",
    "start_reaction_runtime",
]
