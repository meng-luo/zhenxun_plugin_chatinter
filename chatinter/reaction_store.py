"""Single-pack local storage for ChatInter reaction images."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from .reaction_image import inspect_reaction_image
from .reaction_models import (
    ReactionRecord,
    normalize_semantic_list,
    normalize_tags,
)

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})
_METADATA_VERSION = 2
_SCAN_TTL_SECONDS = 30.0
_MAX_EXISTING_IMAGE_BYTES = 32 * 1024 * 1024
_MAX_COLLECTED_IMAGE_BYTES = 5 * 1024 * 1024
_COLLECTION_CACHE_VERSION = 1
_MAX_COLLECTION_REJECTIONS = 2_000


class ReactionStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.images_root = (self.root / "memes").resolve()
        self.metadata_path = self.root / "chatinter_reaction_metadata.json"
        self.astr_metadata_path = self.root / "semantic_metadata.json"
        self.category_metadata_path = self.root / "memes_data.json"
        self.manifest_path = self.root / "manifest.json"
        self.collection_cache_path = self.root / "chatinter_reaction_decisions.json"
        self._lock = asyncio.Lock()
        self._records: dict[str, ReactionRecord] = {}
        self._collection_rejections: dict[str, int] | None = None
        self._last_scan = 0.0

    async def records(self, *, force: bool = False) -> tuple[ReactionRecord, ...]:
        if (
            not force
            and self._last_scan > 0
            and time.monotonic() - self._last_scan < _SCAN_TTL_SECONDS
        ):
            return tuple(self._records.values())
        async with self._lock:
            if (
                not force
                and self._last_scan > 0
                and time.monotonic() - self._last_scan < _SCAN_TTL_SECONDS
            ):
                return tuple(self._records.values())
            records, changed = await asyncio.to_thread(self._scan_sync)
            self._records = records
            self._last_scan = time.monotonic()
            if changed:
                await asyncio.to_thread(self._write_metadata_sync, records)
            return tuple(records.values())

    async def pending_records(self) -> tuple[ReactionRecord, ...]:
        return tuple(
            record for record in await self.records() if record.status == "pending"
        )

    async def contains_digest(self, content_sha256: str) -> bool:
        return any(
            record.content_sha256 == content_sha256 for record in await self.records()
        )

    async def contains_identity(
        self,
        content_sha256: str,
        visual_fingerprint: str = "",
    ) -> bool:
        digest = str(content_sha256 or "").strip().casefold()
        fingerprint = str(visual_fingerprint or "").strip().casefold()
        return any(
            record.content_sha256 == digest
            or bool(
                fingerprint
                and record.visual_fingerprint
                and record.visual_fingerprint == fingerprint
            )
            for record in await self.records()
        )

    async def collection_was_rejected(self, cache_key: str) -> bool:
        async with self._lock:
            if self._collection_rejections is None:
                payload = await asyncio.to_thread(
                    _read_json_object,
                    self.collection_cache_path,
                )
                raw = payload.get("rejected")
                self._collection_rejections = (
                    {
                        str(key): _safe_int(value)
                        for key, value in raw.items()
                        if str(key)
                    }
                    if isinstance(raw, dict)
                    else {}
                )
            return str(cache_key or "") in self._collection_rejections

    async def remember_collection_rejected(self, cache_key: str) -> None:
        key = str(cache_key or "").strip()[:256]
        if not key:
            return
        async with self._lock:
            if self._collection_rejections is None:
                payload = await asyncio.to_thread(
                    _read_json_object,
                    self.collection_cache_path,
                )
                raw = payload.get("rejected")
                self._collection_rejections = (
                    {
                        str(item_key): _safe_int(value)
                        for item_key, value in raw.items()
                        if str(item_key)
                    }
                    if isinstance(raw, dict)
                    else {}
                )
            self._collection_rejections[key] = int(time.time())
            while len(self._collection_rejections) > _MAX_COLLECTION_REJECTIONS:
                oldest = min(
                    self._collection_rejections,
                    key=self._collection_rejections.__getitem__,
                )
                self._collection_rejections.pop(oldest, None)
            try:
                await asyncio.to_thread(self._write_collection_cache_sync)
            except OSError:
                return

    async def update_semantics(
        self,
        content_sha256: str,
        *,
        caption: str,
        tags: Any,
        visible_text: str,
        accepted: bool,
        reply_intents: Any = (),
        usage_scenarios: Any = (),
        tones: Any = (),
        actions: Any = (),
        target_relation: str = "",
    ) -> ReactionRecord | None:
        async with self._lock:
            if not self._records:
                records, _changed = await asyncio.to_thread(self._scan_sync)
                self._records = records
            record = self._records.get(content_sha256)
            if record is None:
                return None
            updated = replace(
                record,
                caption=" ".join(str(caption or "").split())[:600],
                tags=normalize_tags(tags),
                visible_text=str(visible_text or "").strip()[:500],
                reply_intents=normalize_semantic_list(
                    reply_intents, limit=6, item_limit=80
                ),
                usage_scenarios=normalize_semantic_list(
                    usage_scenarios, limit=5, item_limit=120
                ),
                tones=normalize_semantic_list(tones, limit=5, item_limit=48),
                actions=normalize_semantic_list(actions, limit=5, item_limit=48),
                target_relation=(
                    " ".join(target_relation.split())[:120]
                    if isinstance(target_relation, str)
                    else ""
                ),
                semantic_version=2,
                status="ready" if accepted else "rejected",
            )
            self._records[content_sha256] = updated
            self._last_scan = time.monotonic()
            await asyncio.to_thread(self._write_metadata_sync, self._records)
            return updated

    async def mark_semantic_error(self, content_sha256: str) -> None:
        async with self._lock:
            record = self._records.get(content_sha256)
            if record is None:
                return
            self._records[content_sha256] = replace(record, status="error")
            self._last_scan = time.monotonic()
            await asyncio.to_thread(self._write_metadata_sync, self._records)

    async def add_collected(
        self,
        content: bytes,
        *,
        extension: str,
        caption: str,
        tags: Any,
        visible_text: str,
        reply_intents: Any = (),
        usage_scenarios: Any = (),
        tones: Any = (),
        actions: Any = (),
        target_relation: str = "",
        visual_fingerprint: str = "",
    ) -> ReactionRecord | None:
        return await self.add_imported(
            content,
            extension=extension,
            category="auto_collected",
            caption=caption,
            tags=tags,
            visible_text=visible_text,
            reply_intents=reply_intents,
            usage_scenarios=usage_scenarios,
            tones=tones,
            actions=actions,
            target_relation=target_relation,
            visual_fingerprint=visual_fingerprint,
            provenance="auto_discovery",
            status="ready",
            max_bytes=_MAX_COLLECTED_IMAGE_BYTES,
        )

    async def add_imported(
        self,
        content: bytes,
        *,
        extension: str,
        category: str,
        caption: str,
        tags: Any,
        visible_text: str,
        reply_intents: Any = (),
        usage_scenarios: Any = (),
        tones: Any = (),
        actions: Any = (),
        target_relation: str = "",
        semantic_version: int = 2,
        visual_fingerprint: str = "",
        provenance: str = "import",
        source_version: str = "",
        status: str = "ready",
        max_bytes: int = _MAX_EXISTING_IMAGE_BYTES,
    ) -> ReactionRecord | None:
        info = await asyncio.to_thread(
            inspect_reaction_image,
            content,
            max_bytes=max_bytes,
        )
        if info is None:
            return None
        normalized_extension = info.extension
        digest = info.content_sha256
        fingerprint = str(visual_fingerprint or info.visual_fingerprint).casefold()
        async with self._lock:
            if not self._records:
                records, _changed = await asyncio.to_thread(self._scan_sync)
                self._records = records
            existing = self._records.get(digest)
            if existing is not None:
                return existing
            for record in self._records.values():
                if fingerprint and record.visual_fingerprint == fingerprint:
                    return record
            safe_category = _normalized_category(category)
            relative = Path("memes") / safe_category / f"{digest}{normalized_extension}"
            target = self._safe_path(relative.as_posix())
            if target is None:
                return None
            await asyncio.to_thread(_atomic_write_bytes, target, content)
            stat = await asyncio.to_thread(target.stat)
            record = ReactionRecord(
                content_sha256=digest,
                relative_path=relative.as_posix(),
                category=safe_category,
                category_description="",
                caption=" ".join(str(caption or "").split())[:600],
                tags=normalize_tags(tags),
                visible_text=str(visible_text or "").strip()[:500],
                reply_intents=normalize_semantic_list(
                    reply_intents, limit=6, item_limit=80
                ),
                usage_scenarios=normalize_semantic_list(
                    usage_scenarios, limit=5, item_limit=120
                ),
                tones=normalize_semantic_list(tones, limit=5, item_limit=48),
                actions=normalize_semantic_list(actions, limit=5, item_limit=48),
                target_relation=(
                    " ".join(target_relation.split())[:120]
                    if isinstance(target_relation, str)
                    else ""
                ),
                semantic_version=max(int(semantic_version or 0), 0),
                status=status if status in {"pending", "ready"} else "ready",
                visual_fingerprint=fingerprint,
                provenance=str(provenance or "")[:120],
                source_version=str(source_version or "")[:120],
                size=int(stat.st_size),
                mtime_ns=int(stat.st_mtime_ns),
            )
            self._records[digest] = record
            self._last_scan = time.monotonic()
            await asyncio.to_thread(self._write_metadata_sync, self._records)
            return record

    async def resolve(self, record: ReactionRecord) -> Path | None:
        path = self._safe_path(record.relative_path)
        if path is None or not await asyncio.to_thread(path.is_file):
            return None
        try:
            digest = await asyncio.to_thread(_file_sha256, path)
        except OSError:
            return None
        return path if digest == record.content_sha256 else None

    def _scan_sync(self) -> tuple[dict[str, ReactionRecord], bool]:
        self.images_root.mkdir(parents=True, exist_ok=True)
        metadata = self._load_metadata_sync()
        existing = _metadata_records(metadata)
        by_path = {record.relative_path: record for record in existing.values()}
        category_descriptions = self._load_category_descriptions_sync()
        default_provenance, default_source_version, default_files = (
            self._load_default_pack_source_sync()
        )
        records: dict[str, ReactionRecord] = {}
        changed = False
        for path in sorted(self.images_root.rglob("*")):
            if not path.is_file() or path.suffix.casefold() not in _IMAGE_EXTENSIONS:
                continue
            try:
                stat = path.stat()
                if stat.st_size <= 0 or stat.st_size > _MAX_EXISTING_IMAGE_BYTES:
                    continue
                relative = path.resolve().relative_to(self.root).as_posix()
            except (OSError, ValueError):
                continue
            previous = by_path.get(relative)
            if (
                previous is not None
                and previous.size == stat.st_size
                and previous.mtime_ns == stat.st_mtime_ns
            ):
                if previous.visual_fingerprint:
                    fingerprint = previous.visual_fingerprint
                else:
                    try:
                        info = inspect_reaction_image(
                            path.read_bytes(),
                            max_bytes=_MAX_EXISTING_IMAGE_BYTES,
                        )
                    except OSError:
                        info = None
                    fingerprint = info.visual_fingerprint if info else ""
                category = previous.category or _category_for_path(
                    path,
                    self.images_root,
                )
                category_description = category_descriptions.get(
                    category,
                    previous.category_description,
                )
                record = replace(
                    previous,
                    category=category,
                    category_description=category_description,
                    visual_fingerprint=fingerprint,
                    provenance=(
                        previous.provenance
                        or (default_provenance if relative in default_files else "")
                    ),
                    source_version=(
                        previous.source_version
                        or (default_source_version if relative in default_files else "")
                    ),
                )
                if record != previous:
                    changed = True
            else:
                try:
                    info = inspect_reaction_image(
                        path.read_bytes(),
                        max_bytes=_MAX_EXISTING_IMAGE_BYTES,
                    )
                except OSError:
                    continue
                if info is None:
                    continue
                digest = info.content_sha256
                semantic = existing.get(digest)
                category = _category_for_path(path, self.images_root)
                record = ReactionRecord(
                    content_sha256=digest,
                    relative_path=relative,
                    category=(semantic.category if semantic else category),
                    category_description=(
                        category_descriptions.get(
                            semantic.category if semantic else category,
                            semantic.category_description if semantic else "",
                        )
                    ),
                    caption=semantic.caption if semantic else "",
                    tags=semantic.tags if semantic else (),
                    visible_text=semantic.visible_text if semantic else "",
                    reply_intents=semantic.reply_intents if semantic else (),
                    usage_scenarios=(semantic.usage_scenarios if semantic else ()),
                    tones=semantic.tones if semantic else (),
                    actions=semantic.actions if semantic else (),
                    target_relation=semantic.target_relation if semantic else "",
                    semantic_version=semantic.semantic_version if semantic else 0,
                    status=semantic.status if semantic else "pending",
                    visual_fingerprint=(
                        semantic.visual_fingerprint
                        if semantic and semantic.visual_fingerprint
                        else info.visual_fingerprint
                    ),
                    provenance=(
                        semantic.provenance
                        if semantic and semantic.provenance
                        else (default_provenance if relative in default_files else "")
                    ),
                    source_version=(
                        semantic.source_version
                        if semantic and semantic.source_version
                        else (
                            default_source_version if relative in default_files else ""
                        )
                    ),
                    size=int(stat.st_size),
                    mtime_ns=int(stat.st_mtime_ns),
                )
                changed = True
            records.setdefault(record.content_sha256, record)
        if set(records) != set(existing):
            changed = True
        return records, changed

    def _load_metadata_sync(self) -> dict[str, Any]:
        own = _read_json_object(self.metadata_path)
        astr = _read_json_object(self.astr_metadata_path)
        records = _metadata_records(own)
        for digest, external in _metadata_records(astr).items():
            records[digest] = _merge_external_record(records.get(digest), external)
        return {
            "version": _METADATA_VERSION,
            "images": {
                digest: record.to_metadata() for digest, record in records.items()
            },
        }

    def _load_category_descriptions_sync(self) -> dict[str, str]:
        payload = _read_json_object(self.category_metadata_path)
        descriptions = {
            str(key).strip(): " ".join(str(value or "").split())[:500]
            for key, value in payload.items()
            if str(key).strip() and isinstance(value, str)
        }
        manifest = _read_json_object(self.manifest_path)
        categories = manifest.get("categories")
        if isinstance(categories, dict):
            for key, value in categories.items():
                category = str(key or "").strip()
                description = (
                    value.get("description") if isinstance(value, dict) else value
                )
                normalized = " ".join(str(description or "").split())[:500]
                if category and normalized:
                    descriptions.setdefault(category, normalized)
        return descriptions

    def _load_default_pack_source_sync(
        self,
    ) -> tuple[str, str, frozenset[str]]:
        payload = _read_json_object(self.root / "chatinter_default_pack.json")
        if str(payload.get("id") or "") != "seio-stickers":
            return "", "", frozenset()
        raw_files = payload.get("files")
        files = (
            frozenset(
                path
                for value in raw_files
                if (
                    path := str(
                        value.get("path") if isinstance(value, dict) else value
                    ).replace("\\", "/")
                ).startswith("memes/")
            )
            if isinstance(raw_files, list)
            else frozenset()
        )
        return (
            "seio_default",
            str(payload.get("commit") or "")[:120],
            files,
        )

    def _write_metadata_sync(self, records: dict[str, ReactionRecord]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _METADATA_VERSION,
            "images": {
                digest: record.to_metadata()
                for digest, record in sorted(records.items())
            },
        }
        temporary = self.metadata_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.metadata_path)

    def _write_collection_cache_sync(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _COLLECTION_CACHE_VERSION,
            "rejected": self._collection_rejections or {},
        }
        temporary = self.collection_cache_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.collection_cache_path)

    def _safe_path(self, relative_path: str) -> Path | None:
        value = str(relative_path or "").replace("\\", "/").strip()
        if not value or Path(value).is_absolute():
            return None
        try:
            candidate = (self.root / value).resolve()
            candidate.relative_to(self.images_root)
        except (OSError, ValueError):
            return None
        return candidate


def _metadata_records(payload: dict[str, Any]) -> dict[str, ReactionRecord]:
    raw_images = payload.get("images")
    if not isinstance(raw_images, dict):
        return {}
    result: dict[str, ReactionRecord] = {}
    for key, value in raw_images.items():
        if not isinstance(value, dict):
            continue
        digest = str(value.get("content_sha256") or key or "").strip().casefold()
        relative = str(value.get("relative_path") or "").replace("\\", "/").strip()
        if len(digest) != 64 or not relative:
            continue
        category = str(value.get("category") or "").strip()[:120]
        review_status = (
            str(value.get("category_review_status") or "").strip().casefold()
        )
        category_fit = str(value.get("category_fit") or "").strip().casefold()
        status = str(
            value.get("status") or value.get("caption_status") or ""
        ).casefold()
        status = {
            "done": "ready",
            "failed": "error",
            "running": "pending",
        }.get(status, status)
        if not status:
            status = "ready" if value.get("caption") or value.get("tags") else "pending"
        if (
            category.casefold() == "needs_review"
            or review_status in {"needs_review", "manual_rejected"}
            or category_fit == "conflict"
        ):
            status = "rejected"
        if status not in {"pending", "ready", "rejected", "error"}:
            status = "pending"
        result[digest] = ReactionRecord(
            content_sha256=digest,
            relative_path=relative,
            category=category,
            category_description=str(value.get("category_description") or "").strip()[
                :500
            ],
            caption=str(value.get("caption") or "").strip()[:600],
            tags=normalize_tags(value.get("tags")),
            visible_text=str(value.get("visible_text") or "").strip()[:500],
            reply_intents=normalize_semantic_list(
                value.get("reply_intents"), limit=6, item_limit=80
            ),
            usage_scenarios=normalize_semantic_list(
                value.get("usage_scenarios"), limit=5, item_limit=120
            ),
            tones=normalize_semantic_list(value.get("tones"), limit=5, item_limit=48),
            actions=normalize_semantic_list(
                value.get("actions"), limit=5, item_limit=48
            ),
            target_relation=(
                " ".join(value["target_relation"].split())[:120]
                if isinstance(value.get("target_relation"), str)
                else ""
            ),
            semantic_version=_safe_int(value.get("semantic_version")),
            status=status,
            visual_fingerprint=str(value.get("visual_fingerprint") or "")[:128],
            provenance=str(value.get("provenance") or "")[:120],
            source_version=str(value.get("source_version") or "")[:120],
            size=_safe_int(value.get("size")),
            mtime_ns=_safe_int(value.get("mtime_ns")),
        )
    return result


def _merge_external_record(
    local: ReactionRecord | None,
    external: ReactionRecord,
) -> ReactionRecord:
    if local is None:
        return external
    external_ready = external.status in {"ready", "rejected"} or bool(
        external.caption or external.tags or external.visible_text
    )
    local_v2_owned = local.semantic_version >= 2 and local.provenance in {
        "seio_default",
        "startup_import",
        "auto_discovery",
    }
    semantic = (
        local
        if local_v2_owned
        else external
        if external_ready or local.status != "ready"
        else local
    )
    return replace(
        semantic,
        relative_path=external.relative_path or local.relative_path,
        category=external.category or semantic.category or local.category,
        category_description=(
            external.category_description
            or semantic.category_description
            or local.category_description
        ),
        size=local.size or external.size,
        mtime_ns=local.mtime_ns or external.mtime_ns,
        visual_fingerprint=(external.visual_fingerprint or local.visual_fingerprint),
        provenance=external.provenance or local.provenance,
        source_version=external.source_version or local.source_version,
    )


def _category_for_path(path: Path, images_root: Path) -> str:
    try:
        relative = path.relative_to(images_root)
    except ValueError:
        return ""
    return relative.parts[0][:120] if len(relative.parts) > 1 else ""


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalized_category(value: str) -> str:
    category = "".join(
        character
        for character in str(value or "").strip()[:120]
        if character.isalnum() or character in {"-", "_"}
    )
    return category or "imported"


def _safe_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


__all__ = ["ReactionStore"]
