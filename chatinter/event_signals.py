"""Backend-agnostic event signal channel.

ChatInter passes control flags across NoneBot hooks through event attributes
and an ``id(event)``-keyed side registry. The side registry supports adapters
whose event models reject extra attributes.

* ``set_event_signal`` writes to the event attribute and side registry.
* ``get_event_signal`` reads the attribute first, then falls back to the side
  registry — so even if ``setattr`` was a no-op the value is still recoverable.

The event object is unhashable (Pydantic v2), so a ``WeakKeyDictionary`` cannot
be used; ``id(event)`` + ``weakref.finalize`` gives the same leak-free lifetime.
"""

from __future__ import annotations

from typing import Any
import weakref

_SIGNAL_STORE: dict[int, dict[str, Any]] = {}
_FINALIZERS: dict[int, weakref.finalize] = {}


_MAX_TRACKED = 4096


def _cleanup(event_id: int) -> None:
    _SIGNAL_STORE.pop(event_id, None)
    _FINALIZERS.pop(event_id, None)


def _ensure_tracked(event: Any, event_id: int) -> dict[str, Any]:
    bucket = _SIGNAL_STORE.get(event_id)
    if bucket is None:
        if len(_SIGNAL_STORE) >= _MAX_TRACKED:

            oldest = next(iter(_SIGNAL_STORE), None)
            if oldest is not None:
                _cleanup(oldest)
        bucket = {}
        _SIGNAL_STORE[event_id] = bucket
        try:
            _FINALIZERS[event_id] = weakref.finalize(event, _cleanup, event_id)
        except TypeError:

            pass
    return bucket


def set_event_signal(event: Any, key: str, value: Any) -> None:
    """Store a cross-hook signal on ``event`` (attribute + side registry)."""
    try:
        setattr(event, key, value)
    except Exception:


        pass
    try:
        _ensure_tracked(event, id(event))[key] = value
    except Exception:
        pass


def get_event_signal(event: Any, key: str, default: Any = None) -> Any:
    """Read a signal: event attribute first, side registry as fallback."""
    sentinel = object()
    value = getattr(event, key, sentinel)
    if value is not sentinel:
        return value
    bucket = _SIGNAL_STORE.get(id(event))
    if bucket is not None and key in bucket:
        return bucket[key]
    return default


def clear_event_signal(event: Any, key: str) -> None:
    try:
        if hasattr(event, key):
            delattr(event, key)
    except Exception:
        pass
    bucket = _SIGNAL_STORE.get(id(event))
    if bucket is not None:
        bucket.pop(key, None)


__all__ = [
    "clear_event_signal",
    "get_event_signal",
    "set_event_signal",
]
