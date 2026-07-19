"""Small durable state helpers for ChatInter Agent runtime data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any

_ROOT = Path("data/chatinter_agent")
_LOCK = threading.RLock()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_path(*parts: str) -> Path:
    path = _ROOT
    for part in parts:
        path = path / str(part).strip().strip("/\\")
    return path


def read_json(path: Path, default: Any) -> Any:
    with _LOCK:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default


def write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        jsonable = to_jsonable(payload)
        content = (
            json.dumps(
                jsonable,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
            if compact
            else json.dumps(jsonable, ensure_ascii=False, indent=2, default=str)
        )
        tmp.write_text(
            content,
            encoding="utf-8",
        )
        tmp.replace(path)


def append_jsonl(path: Path, payload: Any) -> None:
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(to_jsonable(payload), ensure_ascii=False, default=str))
            fp.write("\n")


def write_jsonl(path: Path, rows: Sequence[Any]) -> None:
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fp:
            for row in rows:
                fp.write(json.dumps(to_jsonable(row), ensure_ascii=False, default=str))
                fp.write("\n")
        tmp.replace(path)


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump(mode="json"))
        except Exception:
            try:
                return to_jsonable(value.model_dump())
            except Exception:
                return str(value)
    if is_dataclass(value):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [to_jsonable(item) for item in value]
    return str(value)


__all__ = [
    "append_jsonl",
    "read_json",
    "state_path",
    "to_jsonable",
    "utc_now_iso",
    "write_json",
    "write_jsonl",
]
