"""Non-destructive startup importer for local and Astr reaction packs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .log_compat import logger
from .reaction_image import MAX_IMAGE_BYTES, inspect_reaction_image
from .reaction_models import (
    ReactionSettings,
    normalize_semantic_list,
    normalize_tags,
)
from .reaction_semantics import analyze_reaction_bytes
from .reaction_store import ReactionStore

_IMAGE_EXTENSIONS = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"})
_IMPORT_DECISION_VERSION = "startup-import-v1"
_MIN_IMPORT_CONFIDENCE = 0.85
_MAX_METADATA_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ReactionImportReport:
    scanned: int = 0
    imported: int = 0
    reused_metadata: int = 0
    duplicates: int = 0
    rejected: int = 0
    deferred: int = 0
    invalid: int = 0


@dataclass(frozen=True, slots=True)
class _AstrSemantic:
    category: str
    caption: str
    tags: tuple[str, ...]
    visible_text: str
    reply_intents: tuple[str, ...]
    usage_scenarios: tuple[str, ...]
    tones: tuple[str, ...]
    actions: tuple[str, ...]
    target_relation: str
    semantic_version: int
    source_version: str


async def import_reaction_directory(
    settings: ReactionSettings,
    store: ReactionStore,
) -> ReactionImportReport:
    root = settings.import_root
    await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True)
    files = await asyncio.to_thread(_candidate_files, root)
    counters = {
        "scanned": 0,
        "imported": 0,
        "reused_metadata": 0,
        "duplicates": 0,
        "rejected": 0,
        "deferred": 0,
        "invalid": 0,
    }
    pack_cache: dict[Path, tuple[dict[str, Any], dict[str, str], str]] = {}
    for path in files:
        counters["scanned"] += 1
        try:
            if path.stat().st_size > MAX_IMAGE_BYTES:
                counters["invalid"] += 1
                continue
            content = await asyncio.to_thread(path.read_bytes)
        except OSError:
            counters["invalid"] += 1
            continue
        info = await asyncio.to_thread(
            inspect_reaction_image,
            content,
            max_bytes=MAX_IMAGE_BYTES,
        )
        if info is None:
            counters["invalid"] += 1
            continue
        if await store.contains_identity(
            info.content_sha256,
            info.visual_fingerprint,
        ):
            counters["duplicates"] += 1
            continue

        pack_root, relative, path_category = _astr_location(path, root)
        semantic = None
        category = "imported"
        provenance = "startup_import"
        if pack_root is not None:
            metadata, descriptions, version = pack_cache.setdefault(
                pack_root,
                _load_astr_pack(pack_root),
            )
            semantic = _astr_semantic(
                metadata,
                relative=relative,
                digest=info.content_sha256,
                fallback_category=path_category,
                source_version=version,
            )
            category = semantic.category if semantic else path_category or "imported"
            if semantic and not semantic.caption and category in descriptions:
                semantic = _AstrSemantic(
                    category=category,
                    caption=descriptions[category],
                    tags=semantic.tags,
                    visible_text=semantic.visible_text,
                    reply_intents=semantic.reply_intents,
                    usage_scenarios=semantic.usage_scenarios,
                    tones=semantic.tones,
                    actions=semantic.actions,
                    target_relation=semantic.target_relation,
                    semantic_version=semantic.semantic_version,
                    source_version=semantic.source_version,
                )
            provenance = "astr_import"

        if semantic is not None:
            record = await store.add_imported(
                content,
                extension=info.extension,
                category=semantic.category or category,
                caption=semantic.caption,
                tags=semantic.tags,
                visible_text=semantic.visible_text,
                reply_intents=semantic.reply_intents,
                usage_scenarios=semantic.usage_scenarios,
                tones=semantic.tones,
                actions=semantic.actions,
                target_relation=semantic.target_relation,
                semantic_version=semantic.semantic_version,
                visual_fingerprint=info.visual_fingerprint,
                provenance=provenance,
                source_version=semantic.source_version,
            )
            if record is not None:
                counters["imported"] += 1
                counters["reused_metadata"] += 1
            else:
                counters["invalid"] += 1
            continue

        cache_key = f"{_IMPORT_DECISION_VERSION}:{info.visual_fingerprint}"
        if await store.collection_was_rejected(cache_key):
            counters["rejected"] += 1
            continue
        try:
            analysis = await analyze_reaction_bytes(content, hint=info.extension)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(f"chatinter reaction import analysis deferred: {exc}")
            analysis = None
        if analysis is None:
            counters["deferred"] += 1
            continue
        if not analysis.is_reaction or analysis.confidence < _MIN_IMPORT_CONFIDENCE:
            await store.remember_collection_rejected(cache_key)
            counters["rejected"] += 1
            continue
        record = await store.add_imported(
            content,
            extension=info.extension,
            category=category,
            caption=analysis.caption,
            tags=analysis.tags,
            visible_text=analysis.visible_text,
            reply_intents=analysis.reply_intents,
            usage_scenarios=analysis.usage_scenarios,
            tones=analysis.tones,
            actions=analysis.actions,
            target_relation=analysis.target_relation,
            visual_fingerprint=info.visual_fingerprint,
            provenance=provenance,
        )
        if record is not None:
            counters["imported"] += 1
        else:
            counters["invalid"] += 1
        await asyncio.sleep(0)
    report = ReactionImportReport(**counters)
    if report.scanned:
        logger.info(
            "ChatInter 表情导入完成："
            f"扫描 {report.scanned}，导入 {report.imported}，"
            f"重复 {report.duplicates}，拒绝 {report.rejected}，"
            f"待重试 {report.deferred}"
        )
    return report


def _candidate_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in _IMAGE_EXTENSIONS
    )


def _astr_location(path: Path, import_root: Path) -> tuple[Path | None, str, str]:
    resolved = path.resolve()
    current = resolved.parent
    boundary = import_root.resolve()
    while True:
        if current.name.casefold() == "memes":
            try:
                relative = resolved.relative_to(current.parent).as_posix()
                inside = resolved.relative_to(current)
            except ValueError:
                return None, "", ""
            category = inside.parts[0] if len(inside.parts) > 1 else ""
            return current.parent, relative, category[:120]
        if current == boundary or boundary not in current.parents:
            break
        current = current.parent
    return None, "", ""


def _load_astr_pack(pack_root: Path) -> tuple[dict[str, Any], dict[str, str], str]:
    metadata = _read_json(pack_root / "semantic_metadata.json")
    descriptions = {
        str(key).strip(): str(value or "").strip()[:500]
        for key, value in _read_json(pack_root / "memes_data.json").items()
        if str(key).strip() and isinstance(value, str)
    }
    manifest = _read_json(pack_root / "manifest.json")
    categories = manifest.get("categories")
    if isinstance(categories, dict):
        for key, value in categories.items():
            category = str(key or "").strip()
            description = value.get("description") if isinstance(value, dict) else value
            if category and str(description or "").strip():
                descriptions.setdefault(category, str(description).strip()[:500])
    metadata_version = str(metadata.get("schema_version") or "").strip()
    if metadata_version not in {"", "1", "1.0", "2", "2.0"}:
        metadata = {}
    version = str(metadata_version or manifest.get("version") or "")[:120]
    return metadata, descriptions, version


def _astr_semantic(
    payload: dict[str, Any],
    *,
    relative: str,
    digest: str,
    fallback_category: str,
    source_version: str,
) -> _AstrSemantic | None:
    images = payload.get("images")
    if not isinstance(images, dict):
        return None
    for key, value in images.items():
        if not isinstance(value, dict):
            continue
        item_digest = str(value.get("content_sha256") or key or "").casefold()
        item_relative = str(value.get("relative_path") or "").replace("\\", "/")
        if item_digest != digest or item_relative != relative:
            continue
        category = str(value.get("category") or fallback_category or "").strip()[:120]
        review = str(value.get("category_review_status") or "").casefold()
        fit = str(value.get("category_fit") or "").casefold()
        status = str(
            value.get("status") or value.get("caption_status") or ""
        ).casefold()
        if (
            category.casefold() == "needs_review"
            or review in {"unchecked", "needs_review", "manual_rejected"}
            or fit == "conflict"
            or status in {"failed", "error", "pending", "running"}
        ):
            return None
        caption = " ".join(str(value.get("caption") or "").split())[:600]
        tags = normalize_tags(value.get("tags"))
        visible_text = str(value.get("visible_text") or "").strip()[:500]
        reply_intents = normalize_semantic_list(
            value.get("reply_intents"), limit=6, item_limit=80
        )
        usage_scenarios = normalize_semantic_list(
            value.get("usage_scenarios"), limit=5, item_limit=120
        )
        tones = normalize_semantic_list(value.get("tones"), limit=5, item_limit=48)
        actions = normalize_semantic_list(
            value.get("actions"), limit=5, item_limit=48
        )
        target_relation = (
            " ".join(value["target_relation"].split())[:120]
            if isinstance(value.get("target_relation"), str)
            else ""
        )
        if not caption and not tags and not visible_text:
            return None
        return _AstrSemantic(
            category=category or "imported",
            caption=caption,
            tags=tags,
            visible_text=visible_text,
            reply_intents=reply_intents,
            usage_scenarios=usage_scenarios,
            tones=tones,
            actions=actions,
            target_relation=target_relation,
            semantic_version=(
                2
                if reply_intents
                or usage_scenarios
                or tones
                or actions
                or target_relation
                else 0
            ),
            source_version=source_version,
        )
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > _MAX_METADATA_BYTES:
            return {}
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


__all__ = ["ReactionImportReport", "import_reaction_directory"]
