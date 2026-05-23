"""Artifact references and context-safe payload compaction for ChatInter.

Long tool outputs, logs, diffs, and media references should never be pasted
verbatim into the next LLM request.  They are stored as bounded artifacts and
only concise summaries are returned to the model.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from hashlib import blake2s
import json
from pathlib import Path
import time
from typing import Any, Literal

from .route_text import normalize_message_text
from .runtime_events import emit_runtime_event

ArtifactType = Literal["text", "image", "html", "file", "log", "plugin_output"]

_MAX_MEMORY_ARTIFACTS = 512
_INLINE_TEXT_LIMIT = 240
_SUMMARY_TEXT_LIMIT = 180
_MODEL_STRING_LIMIT = 700
_MODEL_LONG_FIELD_LIMIT = 360
_MODEL_LIST_ITEMS = 12
_MODEL_DICT_DEPTH = 5
_ARTIFACT_DIR = Path("data") / "chatinter_artifacts"
_MANIFEST_PATH = _ARTIFACT_DIR / "artifacts.json"
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
}
_REFERENCE_KEYS = {
    "artifact_id",
    "approval_id",
    "command_id",
    "rendered_command",
    "matched_plugin",
    "plugin_module",
    "task_text",
    "remaining_task_hint",
    "status",
    "ok",
    "returncode",
    "retryable",
    "need_continue",
    "truncated",
    "count",
    "path",
    "cwd",
    "command",
    "args",
    "instruction",
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

    def __init__(self, *, max_items: int = _MAX_MEMORY_ARTIFACTS) -> None:
        self.max_items = max(16, int(max_items or _MAX_MEMORY_ARTIFACTS))
        self._items: OrderedDict[str, ArtifactRef] = OrderedDict()
        self._loaded = False

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
        inline_text = ""
        path = ""
        if force_file or len(raw) > _INLINE_TEXT_LIMIT:
            path = _write_text_artifact(artifact_id, raw)
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
        self._remember(ref)
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
        ref = ArtifactRef(
            artifact_id=_artifact_id(
                artifact_type=artifact_type,
                trace_id=trace_id,
                text=text,
            ),
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
        try:
            text = Path(ref.path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        return ref, text[start:end]

    def _remember(self, ref: ArtifactRef) -> None:
        self._items.pop(ref.artifact_id, None)
        self._items[ref.artifact_id] = ref
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)
        self._save_manifest()
        emit_runtime_event(
            kind="artifact",
            status="created",
            source=ref.source or "artifact_store",
            trace_id=_trace_from_artifact_id(ref.artifact_id),
            summary=ref.summary,
            payload=ref.to_dict(),
            artifacts=[ref.to_dict()],
            related_ids={"artifact_id": ref.artifact_id},
        )

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
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def _save_manifest(self) -> None:
        try:
            _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            _MANIFEST_PATH.write_text(
                json.dumps(
                    {
                        artifact_id: asdict(ref)
                        for artifact_id, ref in self._items.items()
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except Exception:
            return


_STORE = ArtifactStore()


def get_artifact_store() -> ArtifactStore:
    return _STORE


def compact_tool_result_output(
    output: Any,
    *,
    trace_id: str = "",
    source: str = "tool_result",
) -> dict[str, Any]:
    """Return a model-safe payload and move large values to ArtifactStore."""

    artifacts: list[dict[str, Any]] = []
    if not isinstance(output, dict):
        compact_value = _compact_value(
            output,
            trace_id=trace_id,
            source=source,
            key="output",
            artifacts=artifacts,
            depth=0,
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
        return {
            str(item_key): _compact_value(
                item_value,
                trace_id=trace_id,
                source=source,
                key=str(item_key),
                artifacts=artifacts,
                depth=depth + 1,
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
        )
    return _compact_string(
        str(value),
        trace_id=trace_id,
        source=source,
        key=key,
        artifacts=artifacts,
    )


def _compact_string(
    value: str,
    *,
    trace_id: str,
    source: str,
    key: str,
    artifacts: list[dict[str, Any]],
) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    lowered = key.lower()
    if lowered == "artifact_content" and "artifact_read" in source:
        limit = 4000
    else:
        limit = _MODEL_LONG_FIELD_LIMIT if lowered in _LONG_TEXT_KEYS else _MODEL_STRING_LIMIT
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
    if len(raw) <= limit:
        return raw
    artifact_type: ArtifactType = "log" if lowered in {"stdout", "stderr", "log", "logs", "traceback"} else "plugin_output"
    ref = get_artifact_store().store_text(
        raw,
        artifact_type=artifact_type,
        trace_id=trace_id,
        source=f"{source}:{key}",
        force_file=True,
    )
    if ref is not None:
        artifacts.append(ref.to_dict())
        return f"[artifact:{ref.artifact_id}] {ref.summary}"
    return summarize_artifact_text(raw)


def _compact_list(
    values: list[Any],
    *,
    trace_id: str,
    source: str,
    key: str,
    artifacts: list[dict[str, Any]],
    depth: int,
) -> list[Any] | dict[str, Any]:
    if not values:
        return []
    try:
        raw_text = json.dumps(values, ensure_ascii=False, default=str)
    except Exception:
        raw_text = str(values)
    should_store_full = len(values) > _MODEL_LIST_ITEMS or len(raw_text) > _MODEL_STRING_LIMIT * 2
    compacted = [
        _compact_value(
            item,
            trace_id=trace_id,
            source=source,
            key=key,
            artifacts=artifacts,
            depth=depth + 1,
        )
        for item in values[:_MODEL_LIST_ITEMS]
    ]
    if not should_store_full:
        return compacted
    ref = get_artifact_store().store_text(
        raw_text,
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


def _compact_existing_artifacts(values: list[Any] | tuple[Any, ...]) -> list[dict[str, Any]]:
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
                payload[key] = summarize_artifact_text(value, limit=240) if key == "inline_text" else normalize_message_text(value)
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
    try:
        _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = _ARTIFACT_DIR / f"{artifact_id}.txt"
        path.write_text(str(text or ""), encoding="utf-8")
        return str(path)
    except Exception:
        return ""


def _safe_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _trace_from_artifact_id(artifact_id: str) -> str:
    parts = normalize_message_text(artifact_id).split("_")
    return parts[2] if len(parts) >= 4 else ""


__all__ = [
    "ArtifactRef",
    "ArtifactStore",
    "ArtifactType",
    "compact_tool_result_output",
    "get_artifact_store",
    "summarize_artifact_text",
]
