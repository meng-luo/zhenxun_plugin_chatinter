"""Artifact references and context-safe payload compaction for ChatInter.

Long tool outputs, logs, diffs, and media references should never be pasted
verbatim into the next LLM request.  They are stored as bounded artifacts and
only concise summaries are returned to the model.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import blake2s
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, Literal

from .persistence import write_json
from .route_text import normalize_message_text
from .token_compat import estimate_text_tokens

ArtifactType = Literal["text", "image", "html", "file", "log", "plugin_output"]

_INLINE_TEXT_LIMIT = 240
_SUMMARY_TEXT_LIMIT = 180
_MODEL_STRING_LIMIT = 700
_MODEL_LONG_FIELD_LIMIT = 360
_MODEL_COMPLEX_VALUE_LIMIT = _MODEL_STRING_LIMIT * 2
_MODEL_LIST_ITEMS = 12
_MODEL_DICT_DEPTH = 5
_ARTIFACT_DIR = Path("data") / "chatinter_artifacts"
_MANIFEST_PATH = _ARTIFACT_DIR / "artifacts.json"
_ARTIFACT_RETENTION_SECONDS = 30 * 24 * 60 * 60
_LONG_TEXT_KEYS = {
    "stdout",
    "stderr",
    "content",
    "text",
    "diff",
    "before_content",
    "after_content",
    "log",
    "logs",
    "output",
    "traceback",
    "artifact_content",
    "body",
    "command_output",
    "combined_output",
    "display_content",
    "response_text",
    "result_text",
}
_COMPLEX_VALUE_KEYS = {
    "body",
    "data",
    "details",
    "json",
    "metadata",
    "payload",
    "raw",
    "response",
    "result",
}


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    type: ArtifactType
    summary: str
    size: int = 0
    mime_type: str = ""
    path: str = ""
    inline_text: str = ""
    source: str = ""
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "type": self.type,
            "summary": self.summary,
            "size": self.size,
        }
        if self.mime_type:
            payload["mime_type"] = self.mime_type
        if self.path:
            payload["path"] = self.path
        if self.inline_text:
            payload["inline_text"] = self.inline_text
        if self.source:
            payload["source"] = self.source
        return payload


class ArtifactStore:
    """Bounded memory cache plus durable text artifact files."""

    def __init__(self) -> None:
        self._items: OrderedDict[str, ArtifactRef] = OrderedDict()
        self._loaded = False
        self._protected_ids_provider: Callable[[], set[str]] | None = None

    def set_protected_ids_provider(
        self,
        provider: Callable[[], set[str]] | None,
    ) -> None:
        self._protected_ids_provider = provider

    def store_text(
        self,
        text: str,
        *,
        artifact_type: ArtifactType = "plugin_output",
        trace_id: str = "",
        source: str = "",
        force_file: bool = False,
    ) -> ArtifactRef | None:
        raw = str(text or "")
        if not raw.strip():
            return None
        self._ensure_loaded()
        artifact_id = _artifact_id(
            artifact_type=artifact_type,
            trace_id=trace_id,
            text=raw,
        )
        requires_file = force_file or len(raw) > _INLINE_TEXT_LIMIT
        existing = self._items.get(artifact_id)
        if existing is not None:
            if not requires_file or (existing.path and Path(existing.path).is_file()):
                self._items.move_to_end(artifact_id)
                return existing
            self._items.pop(artifact_id, None)
        inline_text = ""
        path = ""
        if requires_file:
            path = _write_text_artifact(artifact_id, raw)
            if not path:
                return None
        else:
            inline_text = raw
        ref = ArtifactRef(
            artifact_id=artifact_id,
            type=artifact_type,
            summary=summarize_artifact_text(raw),
            size=len(raw),
            path=path,
            inline_text=inline_text,
            source=normalize_message_text(source),
            created_at=time.time(),
        )
        persisted = self._remember(ref)
        if requires_file and not persisted:
            self._items.pop(artifact_id, None)
            Path(path).unlink(missing_ok=True)
            return None
        return ref

    def store_json(
        self,
        value: Any,
        *,
        artifact_type: ArtifactType = "plugin_output",
        trace_id: str = "",
        source: str = "",
    ) -> ArtifactRef | None:
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except Exception:
            text = str(value or "")
        return self.store_text(
            text,
            artifact_type=artifact_type,
            trace_id=trace_id,
            source=source,
            force_file=True,
        )

    def store_file(
        self,
        path: Path | str,
        *,
        artifact_type: ArtifactType = "file",
        trace_id: str = "",
        source: str = "",
        summary: str = "",
        mime_type: str = "",
        move: bool = False,
    ) -> ArtifactRef | None:
        source_path = Path(path)
        try:
            size = source_path.stat().st_size
        except OSError:
            return None
        if size <= 0:
            if move:
                source_path.unlink(missing_ok=True)
            return None

        digest = blake2s(digest_size=8)
        try:
            with source_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(64 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return None
        artifact_id = _artifact_id(
            artifact_type=artifact_type,
            trace_id=trace_id,
            text=digest.hexdigest(),
        )
        self._ensure_loaded()
        existing = self._items.get(artifact_id)
        if existing is not None:
            if move:
                source_path.unlink(missing_ok=True)
            self._items.move_to_end(artifact_id)
            return existing

        suffix = source_path.suffix if len(source_path.suffix) <= 12 else ""
        destination = _ARTIFACT_DIR / f"{artifact_id}{suffix or '.log'}"
        try:
            _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            if source_path.resolve() != destination.resolve():
                if move:
                    source_path.replace(destination)
                else:
                    shutil.copyfile(source_path, destination)
        except OSError:
            return None
        ref = ArtifactRef(
            artifact_id=artifact_id,
            type=artifact_type,
            summary=summarize_artifact_text(summary or source_path.name),
            size=size,
            mime_type=normalize_message_text(mime_type),
            path=str(destination),
            source=normalize_message_text(source),
            created_at=time.time(),
        )
        self._remember(ref)
        return ref

    def store_reference(
        self,
        *,
        artifact_type: ArtifactType,
        summary: str,
        trace_id: str = "",
        source: str = "",
        path: str = "",
        mime_type: str = "",
        size: int = 0,
    ) -> ArtifactRef:
        self._ensure_loaded()
        text = " ".join([summary, path, mime_type, str(size)])
        artifact_id = _artifact_id(
            artifact_type=artifact_type,
            trace_id=trace_id,
            text=text,
        )
        existing = self._items.get(artifact_id)
        if existing is not None:
            self._items.move_to_end(artifact_id)
            return existing
        ref = ArtifactRef(
            artifact_id=artifact_id,
            type=artifact_type,
            summary=summarize_artifact_text(summary),
            size=max(int(size or 0), 0),
            mime_type=normalize_message_text(mime_type),
            path=normalize_message_text(path),
            source=normalize_message_text(source),
            created_at=time.time(),
        )
        self._remember(ref)
        return ref

    def get(self, artifact_id: str) -> ArtifactRef | None:
        self._ensure_loaded()
        artifact_id = normalize_message_text(artifact_id)
        ref = self._items.get(artifact_id)
        if ref is not None:
            self._items.move_to_end(artifact_id)
        return ref

    def list_refs(
        self,
        *,
        limit: int = 20,
        artifact_type: str = "",
        source_contains: str = "",
    ) -> list[ArtifactRef]:
        self._ensure_loaded()
        normalized_type = normalize_message_text(artifact_type)
        source_filter = normalize_message_text(source_contains).lower()
        refs = list(self._items.values())
        if normalized_type:
            refs = [ref for ref in refs if ref.type == normalized_type]
        if source_filter:
            refs = [
                ref
                for ref in refs
                if source_filter in normalize_message_text(ref.source).lower()
            ]
        refs.sort(key=lambda ref: ref.created_at or 0.0, reverse=True)
        return refs[: max(1, min(int(limit or 20), 100))]

    def read_text(
        self,
        artifact_id: str,
        *,
        max_chars: int = 4000,
        offset: int = 0,
    ) -> tuple[ArtifactRef, str] | None:
        ref = self.get(artifact_id)
        if ref is None:
            return None
        start = max(int(offset or 0), 0)
        end = start + max(1, int(max_chars or 4000))
        if ref.inline_text:
            return ref, ref.inline_text[start:end]
        if not ref.path:
            return ref, ""
        text, _has_more = _read_text_window(
            Path(ref.path),
            offset=start,
            max_chars=end - start,
        )
        return ref, text

    def search_text(
        self,
        artifact_id: str,
        query: str,
        *,
        max_matches: int = 6,
        context_chars: int = 240,
        scan_chars: int = 120_000,
        offset: int = 0,
    ) -> tuple[ArtifactRef, dict[str, Any]] | None:
        ref = self.get(artifact_id)
        needle = str(query or "").strip()
        if ref is None or not needle:
            return None
        start = max(int(offset or 0), 0)
        match_limit = max(1, min(int(max_matches or 6), 20))
        context_limit = max(20, min(int(context_chars or 240), 2_000))
        scan_limit = max(1_000, min(int(scan_chars or 120_000), 250_000))
        overlap = min(max(len(needle) - 1, 0), 512)
        read_limit = scan_limit + overlap
        if ref.inline_text:
            content = ref.inline_text[start : start + read_limit + 1]
            has_more = start + len(content) < len(ref.inline_text)
            if len(content) > read_limit:
                content = content[:read_limit]
                has_more = True
        elif ref.path:
            content, has_more = _read_text_window(
                Path(ref.path),
                offset=start,
                max_chars=read_limit,
            )
        else:
            content = ""
            has_more = False

        scan_end = min(len(content), scan_limit)
        pattern = re.compile(re.escape(needle), flags=re.IGNORECASE)
        matches: list[dict[str, Any]] = []
        more_matches = False
        last_match_end = 0
        for match in pattern.finditer(content):
            if match.start() >= scan_end:
                break
            if len(matches) >= match_limit:
                more_matches = True
                break
            excerpt_start = max(match.start() - context_limit, 0)
            excerpt_end = min(match.end() + context_limit, len(content))
            matches.append(
                {
                    "offset": start + match.start(),
                    "match": match.group(0),
                    "excerpt_offset": start + excerpt_start,
                    "excerpt": content[excerpt_start:excerpt_end],
                }
            )
            last_match_end = match.end()

        if more_matches:
            next_offset: int | None = start + max(last_match_end, 1)
        elif has_more or len(content) > scan_end:
            next_offset = start + scan_end
        else:
            next_offset = None
        return ref, {
            "query": needle,
            "matches": matches,
            "offset": start,
            "scanned_chars": scan_end,
            "next_offset": next_offset,
            "truncated": next_offset is not None,
        }

    def cleanup_expired(
        self,
        *,
        now: float | None = None,
        retention_seconds: float = _ARTIFACT_RETENTION_SECONDS,
    ) -> dict[str, int]:
        self._ensure_loaded()
        now_ts = float(now if now is not None else time.time())
        cutoff = now_ts - max(float(retention_seconds or 0), 0.0)
        protected = self._protected_ids()
        expired: list[ArtifactRef] = []
        for ref in self._items.values():
            if ref.artifact_id in protected:
                continue
            timestamp = _artifact_timestamp(ref, fallback=now_ts)
            if timestamp <= cutoff:
                expired.append(ref)

        for ref in expired:
            self._items.pop(ref.artifact_id, None)
        if expired and not self._save_manifest():
            for ref in expired:
                self._items[ref.artifact_id] = ref
            return {
                "artifacts_deleted": 0,
                "artifact_files_deleted": 0,
                "artifact_disk_bytes": _artifact_disk_bytes(),
            }

        files_deleted = 0
        for ref in expired:
            path = _owned_artifact_path(ref.path)
            if path is not None and _delete_file(path):
                files_deleted += 1

        tracked_paths = {
            path
            for ref in self._items.values()
            if (path := _owned_artifact_path(ref.path)) is not None
        }
        if _ARTIFACT_DIR.exists():
            for path in _ARTIFACT_DIR.iterdir():
                if not path.is_file() or path == _MANIFEST_PATH:
                    continue
                resolved = path.resolve()
                if resolved in tracked_paths or path.stem in protected:
                    continue
                try:
                    expired_file = path.stat().st_mtime <= cutoff
                except OSError:
                    continue
                if expired_file and _delete_file(path):
                    files_deleted += 1

        return {
            "artifacts_deleted": len(expired),
            "artifact_files_deleted": files_deleted,
            "artifact_disk_bytes": _artifact_disk_bytes(),
        }

    def _remember(self, ref: ArtifactRef) -> bool:
        self._items.pop(ref.artifact_id, None)
        self._items[ref.artifact_id] = ref
        return self._save_manifest()

    def _protected_ids(self) -> set[str]:
        try:
            return (
                set(self._protected_ids_provider())
                if self._protected_ids_provider is not None
                else set()
            )
        except Exception:
            return set()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not _MANIFEST_PATH.exists():
            return
        try:
            raw = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        for artifact_id, payload in raw.items():
            if not isinstance(payload, dict):
                continue
            try:
                ref = ArtifactRef(
                    artifact_id=str(payload.get("artifact_id") or artifact_id),
                    type=str(payload.get("type") or "plugin_output"),  # type: ignore[arg-type]
                    summary=str(payload.get("summary") or ""),
                    size=int(payload.get("size") or 0),
                    mime_type=str(payload.get("mime_type") or ""),
                    path=str(payload.get("path") or ""),
                    inline_text=str(payload.get("inline_text") or ""),
                    source=str(payload.get("source") or ""),
                    created_at=float(payload.get("created_at") or 0.0),
                )
            except Exception:
                continue
            self._items[ref.artifact_id] = ref

    def _save_manifest(self) -> bool:
        try:
            write_json(
                _MANIFEST_PATH,
                {artifact_id: asdict(ref) for artifact_id, ref in self._items.items()},
            )
        except Exception:
            return False
        return True


_STORE = ArtifactStore()


def get_artifact_store() -> ArtifactStore:
    return _STORE


def _read_text_window(
    path: Path,
    *,
    offset: int,
    max_chars: int,
) -> tuple[str, bool]:
    start = max(int(offset or 0), 0)
    limit = max(int(max_chars or 0), 1)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            remaining = start
            while remaining > 0:
                skipped = handle.read(min(remaining, 64 * 1024))
                if not skipped:
                    return "", False
                remaining -= len(skipped)
            content = handle.read(limit + 1)
    except Exception:
        return "", False
    return content[:limit], len(content) > limit


def compact_tool_result_output(
    output: Any,
    *,
    trace_id: str = "",
    source: str = "tool_result",
    inline_text_limits: dict[str, int] | None = None,
    inline_list_limits: dict[str, int] | None = None,
    inline_text_token_limits: dict[str, int] | None = None,
    inline_list_token_limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Return a model-safe payload and move large values to ArtifactStore."""

    artifacts: list[dict[str, Any]] = []
    text_limits = {
        str(key).lower(): max(int(value), 1)
        for key, value in (inline_text_limits or {}).items()
    }
    list_limits = {
        str(key).lower(): max(int(value), 1)
        for key, value in (inline_list_limits or {}).items()
    }
    text_token_limits = {
        str(key).lower(): max(int(value), 1)
        for key, value in (inline_text_token_limits or {}).items()
    }
    list_token_limits = {
        str(key).lower(): max(int(value), 1)
        for key, value in (inline_list_token_limits or {}).items()
    }
    if not isinstance(output, dict):
        compact_value = _compact_value(
            output,
            trace_id=trace_id,
            source=source,
            key="output",
            artifacts=artifacts,
            depth=0,
            inline_text_limits=text_limits,
            inline_list_limits=list_limits,
            inline_text_token_limits=text_token_limits,
            inline_list_token_limits=list_token_limits,
        )
        return {
            "ok": False,
            "status": "raw_tool_result",
            "content": compact_value,
            "artifacts": artifacts,
        }

    compacted: dict[str, Any] = {}
    existing_artifacts = output.get("artifacts")
    if isinstance(existing_artifacts, list | tuple):
        artifacts.extend(_compact_existing_artifacts(existing_artifacts))
    for key, value in output.items():
        if key == "artifacts":
            continue
        compacted[key] = _compact_value(
            value,
            trace_id=trace_id,
            source=source,
            key=str(key),
            artifacts=artifacts,
            depth=0,
            inline_text_limits=text_limits,
            inline_list_limits=list_limits,
            inline_text_token_limits=text_token_limits,
            inline_list_token_limits=list_token_limits,
        )
    compacted["artifacts"] = _dedupe_artifacts(artifacts)
    return compacted


def summarize_artifact_text(text: str, *, limit: int = _SUMMARY_TEXT_LIMIT) -> str:
    normalized = normalize_message_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 1, 1)].rstrip() + "…"


def _compact_value(
    value: Any,
    *,
    trace_id: str,
    source: str,
    key: str,
    artifacts: list[dict[str, Any]],
    depth: int,
    inline_text_limits: dict[str, int],
    inline_list_limits: dict[str, int],
    inline_text_token_limits: dict[str, int],
    inline_list_token_limits: dict[str, int],
) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _compact_string(
            value,
            trace_id=trace_id,
            source=source,
            key=key,
            artifacts=artifacts,
            inline_limit=inline_text_limits.get(key.lower()),
            inline_token_limit=inline_text_token_limits.get(key.lower()),
        )
    if isinstance(value, dict):
        if depth >= _MODEL_DICT_DEPTH:
            return _store_complex_value(
                value,
                trace_id=trace_id,
                source=source,
                key=key,
                artifacts=artifacts,
            )
        if _should_store_complex_dict(value, key=key):
            return _store_complex_value(
                value,
                trace_id=trace_id,
                source=source,
                key=key,
                artifacts=artifacts,
            )
        return {
            str(item_key): _compact_value(
                item_value,
                trace_id=trace_id,
                source=source,
                key=str(item_key),
                artifacts=artifacts,
                depth=depth + 1,
                inline_text_limits=inline_text_limits,
                inline_list_limits=inline_list_limits,
                inline_text_token_limits=inline_text_token_limits,
                inline_list_token_limits=inline_list_token_limits,
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, list | tuple):
        return _compact_list(
            list(value),
            trace_id=trace_id,
            source=source,
            key=key,
            artifacts=artifacts,
            depth=depth,
            inline_text_limits=inline_text_limits,
            inline_list_limits=inline_list_limits,
            inline_text_token_limits=inline_text_token_limits,
            inline_list_token_limits=inline_list_token_limits,
        )
    return _compact_string(
        str(value),
        trace_id=trace_id,
        source=source,
        key=key,
        artifacts=artifacts,
        inline_limit=inline_text_limits.get(key.lower()),
        inline_token_limit=inline_text_token_limits.get(key.lower()),
    )


def _compact_string(
    value: str,
    *,
    trace_id: str,
    source: str,
    key: str,
    artifacts: list[dict[str, Any]],
    inline_limit: int | None = None,
    inline_token_limit: int | None = None,
) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    lowered = key.lower()
    if inline_limit is not None:
        limit = inline_limit
    elif lowered == "artifact_content" and "artifact_read" in source:
        limit = 4000
    else:
        limit = (
            _MODEL_LONG_FIELD_LIMIT
            if lowered in _LONG_TEXT_KEYS
            else _MODEL_STRING_LIMIT
        )
    if _looks_like_image_reference(raw):
        ref = get_artifact_store().store_reference(
            artifact_type="image",
            summary=f"image reference from {key}",
            trace_id=trace_id,
            source=source,
            path="" if raw.startswith("data:") else raw[:500],
            mime_type=_mime_from_data_url(raw),
            size=len(raw),
        )
        artifacts.append(ref.to_dict())
        return f"[image_artifact:{ref.artifact_id}] {ref.summary}"
    if inline_token_limit is not None:
        if estimate_text_tokens(raw) <= inline_token_limit:
            return raw
    elif len(raw) <= limit:
        return raw
    artifact_type: ArtifactType = (
        "log"
        if lowered in {"stdout", "stderr", "log", "logs", "traceback"}
        else "plugin_output"
    )
    stored_text = (
        raw
        if inline_token_limit is not None
        else raw[limit:]
        if inline_limit is not None
        else raw
    )
    ref = get_artifact_store().store_text(
        stored_text,
        artifact_type=artifact_type,
        trace_id=trace_id,
        source=f"{source}:{key}",
        force_file=True,
    )
    if ref is not None:
        artifacts.append(ref.to_dict())
        if inline_token_limit is not None:
            return _token_bounded_preview(
                raw,
                token_limit=inline_token_limit,
                artifact_id=ref.artifact_id,
            )
        if inline_limit is not None:
            return raw[:limit]
        return f"[artifact:{ref.artifact_id}] {ref.summary}"
    if inline_token_limit is not None:
        return _token_bounded_preview(raw, token_limit=inline_token_limit)
    if inline_limit is not None:
        return raw[:limit]
    return summarize_artifact_text(raw)


def _compact_list(
    values: list[Any],
    *,
    trace_id: str,
    source: str,
    key: str,
    artifacts: list[dict[str, Any]],
    depth: int,
    inline_text_limits: dict[str, int],
    inline_list_limits: dict[str, int],
    inline_text_token_limits: dict[str, int],
    inline_list_token_limits: dict[str, int],
) -> list[Any] | dict[str, Any]:
    if not values:
        return []
    try:
        raw_text = json.dumps(values, ensure_ascii=False, default=str)
    except Exception:
        raw_text = str(values)
    inline_limit = inline_list_limits.get(key.lower())
    inline_token_limit = inline_list_token_limits.get(key.lower())
    item_limit = (
        inline_limit or len(values)
        if inline_token_limit
        else inline_limit or _MODEL_LIST_ITEMS
    )
    compacted: list[Any] = []
    for item in values[:item_limit]:
        item_artifacts: list[dict[str, Any]] = []
        compacted_item = _compact_value(
            item,
            trace_id=trace_id,
            source=source,
            key=key,
            artifacts=item_artifacts,
            depth=depth + 1,
            inline_text_limits=inline_text_limits,
            inline_list_limits=inline_list_limits,
            inline_text_token_limits=inline_text_token_limits,
            inline_list_token_limits=inline_list_token_limits,
        )
        candidate = [*compacted, compacted_item]
        if (
            inline_token_limit is not None
            and estimate_text_tokens(
                json.dumps(candidate, ensure_ascii=False, default=str)
            )
            > inline_token_limit
        ):
            break
        compacted.append(compacted_item)
        artifacts.extend(item_artifacts)
    should_store_full = len(compacted) < len(values) or (
        inline_limit is None
        and inline_token_limit is None
        and len(raw_text) > _MODEL_STRING_LIMIT * 2
    )
    if not should_store_full:
        return compacted
    stored_values = (
        values
        if inline_token_limit is not None
        else values[item_limit:]
        if inline_limit is not None
        else values
    )
    ref = get_artifact_store().store_text(
        json.dumps(stored_values, ensure_ascii=False, default=str),
        artifact_type="plugin_output",
        trace_id=trace_id,
        source=f"{source}:{key}:list_full",
        force_file=True,
    )
    if ref is not None:
        artifacts.append(ref.to_dict())
        return {
            "items": compacted,
            "truncated_items": max(len(values) - len(compacted), 0),
            "artifact_id": ref.artifact_id,
            "summary": ref.summary,
        }
    return compacted


def _token_bounded_preview(
    text: str,
    *,
    token_limit: int,
    artifact_id: str = "",
) -> str:
    raw = str(text or "")
    limit = max(int(token_limit or 0), 1)
    if estimate_text_tokens(raw) <= limit:
        return raw

    def render(retained_chars: int) -> str:
        head_chars = max(int(retained_chars * 0.7), 0)
        tail_chars = max(retained_chars - head_chars, 0)
        omitted = max(len(raw) - head_chars - tail_chars, 0)
        reference = f"; artifact:{artifact_id}" if artifact_id else ""
        marker = f"\n...[{omitted} chars omitted{reference}]...\n"
        head = raw[:head_chars].rstrip()
        tail = raw[-tail_chars:].lstrip() if tail_chars else ""
        return f"{head}{marker}{tail}"

    low = 0
    high = len(raw)
    best = render(0)
    while low <= high:
        retained = (low + high) // 2
        candidate = render(retained)
        if estimate_text_tokens(candidate) <= limit:
            best = candidate
            low = retained + 1
        else:
            high = retained - 1
    if estimate_text_tokens(best) <= limit:
        return best
    return _token_prefix(best, limit)


def _token_prefix(text: str, token_limit: int) -> str:
    low = 0
    high = len(text)
    while low <= high:
        length = (low + high) // 2
        if estimate_text_tokens(text[:length]) <= token_limit:
            low = length + 1
        else:
            high = length - 1
    return text[: max(high, 0)]


def _should_store_complex_dict(value: dict[Any, Any], *, key: str) -> bool:
    lowered = key.lower()
    if lowered not in _COMPLEX_VALUE_KEYS and len(value) <= _MODEL_LIST_ITEMS:
        return False
    try:
        raw_text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        raw_text = str(value)
    return len(value) > _MODEL_LIST_ITEMS or len(raw_text) > _MODEL_COMPLEX_VALUE_LIMIT


def _store_complex_value(
    value: Any,
    *,
    trace_id: str,
    source: str,
    key: str,
    artifacts: list[dict[str, Any]],
) -> str:
    ref = get_artifact_store().store_json(
        value,
        artifact_type="plugin_output",
        trace_id=trace_id,
        source=f"{source}:{key}:nested",
    )
    if ref is not None:
        artifacts.append(ref.to_dict())
        return f"[artifact:{ref.artifact_id}] {ref.summary}"
    return summarize_artifact_text(str(value))


def _compact_existing_artifacts(
    values: list[Any] | tuple[Any, ...],
) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in values:
        if not isinstance(item, dict) or not item.get("artifact_id"):
            continue
        payload = {
            "artifact_id": normalize_message_text(str(item.get("artifact_id", ""))),
            "type": normalize_message_text(str(item.get("type", ""))),
            "summary": summarize_artifact_text(str(item.get("summary", "") or "")),
            "size": _safe_int(item.get("size")),
        }
        for key in ("mime_type", "path", "source", "inline_text"):
            value = str(item.get(key, "") or "")
            if value:
                payload[key] = (
                    summarize_artifact_text(value, limit=240)
                    if key == "inline_text"
                    else normalize_message_text(value)
                )
        compacted.append(payload)
    return compacted


def _dedupe_artifacts(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in values:
        artifact_id = normalize_message_text(str(item.get("artifact_id", "") or ""))
        if not artifact_id or artifact_id in seen:
            continue
        seen.add(artifact_id)
        deduped.append(item)
    return deduped[:24]


def _looks_like_image_reference(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered.startswith("data:image/"):
        return True
    return lowered.startswith(("http://", "https://", "file://")) and any(
        marker in lowered for marker in (".png", ".jpg", ".jpeg", ".gif", ".webp")
    )


def _mime_from_data_url(value: str) -> str:
    if not value.startswith("data:"):
        return ""
    return value.split(";", 1)[0].replace("data:", "")[:80]


def _artifact_id(*, artifact_type: str, trace_id: str, text: str) -> str:
    digest = blake2s(
        "|".join([artifact_type, trace_id, text]).encode("utf-8", errors="ignore"),
        digest_size=8,
    ).hexdigest()
    prefix = normalize_message_text(trace_id)[:12] or "local"
    return f"ci_{artifact_type}_{prefix}_{digest}"


def _write_text_artifact(artifact_id: str, text: str) -> str:
    path = _ARTIFACT_DIR / f"{artifact_id}.txt"
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        temporary.write_text(str(text or ""), encoding="utf-8")
        temporary.replace(path)
        return str(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        return ""


def _artifact_timestamp(ref: ArtifactRef, *, fallback: float) -> float:
    if ref.created_at > 0:
        return ref.created_at
    path = _owned_artifact_path(ref.path)
    if path is not None:
        try:
            return path.stat().st_mtime
        except OSError:
            pass
    return fallback


def _owned_artifact_path(value: str) -> Path | None:
    if not value:
        return None
    try:
        path = Path(value).resolve()
        root = _ARTIFACT_DIR.resolve()
        return path if path.is_relative_to(root) else None
    except (OSError, ValueError):
        return None


def _delete_file(path: Path) -> bool:
    try:
        existed = path.is_file()
        path.unlink(missing_ok=True)
        return existed
    except OSError:
        return False


def _artifact_disk_bytes() -> int:
    if not _ARTIFACT_DIR.exists():
        return 0
    total = 0
    for path in _ARTIFACT_DIR.rglob("*"):
        if not path.is_file():
            continue
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _safe_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "ArtifactRef",
    "ArtifactStore",
    "ArtifactType",
    "compact_tool_result_output",
    "get_artifact_store",
    "summarize_artifact_text",
]
