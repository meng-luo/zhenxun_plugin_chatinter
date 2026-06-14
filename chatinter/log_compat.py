"""Logger shim that does not initialize zhenxun services on import."""

from __future__ import annotations

from typing import Any


class _LazyLogger:
    def __getattr__(self, name: str) -> Any:
        try:
            from zhenxun.services.log import logger as real_logger
        except Exception:
            return _noop
        return getattr(real_logger, name)


def _noop(*_: Any, **__: Any) -> None:
    return None


logger = _LazyLogger()

__all__ = ["logger"]
