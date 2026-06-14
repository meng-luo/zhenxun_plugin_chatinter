"""Backend-agnostic event signal channel (补强1 / 执行可靠性).

ChatInter passes a handful of control flags across NoneBot hooks by attaching
private attributes to the live ``Event`` object (``_ai_triggered``,
``_chatinter_native_tail_pending`` ...). On OneBot v11 the event model is a
Pydantic model with ``model_config.extra="allow"``, so ``setattr`` persists and
the original mechanism works. But it is the only no-fallback single point in the
plugin-invocation path: on an adapter whose event model forbids extra fields
(``extra="forbid"``) ``setattr`` silently no-ops and every cross-hook signal is
lost — observation collection, tail-task routing and AI-route gating all break.

This module makes the channel backend-agnostic without changing existing
behaviour:

* ``set_event_signal`` writes BOTH to the event attribute (fast path, keeps the
  existing external readers such as ``auth_checker._collect_ai_route_modules``
  working unchanged on OB11) AND to an ``id(event)``-keyed side registry that is
  auto-cleaned by ``weakref.finalize`` when the event is garbage-collected.
* ``get_event_signal`` reads the attribute first, then falls back to the side
  registry — so even if ``setattr`` was a no-op the value is still recoverable.

The event object is unhashable (Pydantic v2), so a ``WeakKeyDictionary`` cannot
be used; ``id(event)`` + ``weakref.finalize`` gives the same leak-free lifetime.
"""

from __future__ import annotations

from typing import Any
import weakref

# id(event) -> {signal_key: value}
_SIGNAL_STORE: dict[int, dict[str, Any]] = {}
_FINALIZERS: dict[int, weakref.finalize] = {}
# Hard cap so a pathological adapter that recycles ids without GC cannot grow
# this unbounded; finalize normally keeps it tiny.
_MAX_TRACKED = 4096


def _cleanup(event_id: int) -> None:
    _SIGNAL_STORE.pop(event_id, None)
    _FINALIZERS.pop(event_id, None)


def _ensure_tracked(event: Any, event_id: int) -> dict[str, Any]:
    bucket = _SIGNAL_STORE.get(event_id)
    if bucket is None:
        if len(_SIGNAL_STORE) >= _MAX_TRACKED:
            # Drop the oldest-inserted bucket; dict preserves insertion order.
            oldest = next(iter(_SIGNAL_STORE), None)
            if oldest is not None:
                _cleanup(oldest)
        bucket = {}
        _SIGNAL_STORE[event_id] = bucket
        try:
            _FINALIZERS[event_id] = weakref.finalize(event, _cleanup, event_id)
        except TypeError:
            # Object does not support weak references; rely on the size cap.
            pass
    return bucket


def set_event_signal(event: Any, key: str, value: Any) -> None:
    """Store a cross-hook signal on ``event`` (attribute + side registry)."""
    try:
        setattr(event, key, value)
    except Exception:
        # extra="forbid" adapters: attribute write is rejected; side registry
        # below still carries the value.
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
