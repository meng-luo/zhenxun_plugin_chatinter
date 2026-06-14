"""ChatInter package entry.

Pure helper modules under this package are intentionally importable before
NoneBot initialization.  The actual plugin entry is loaded only when the
NoneBot driver exists.
"""

from __future__ import annotations

try:
    from nonebot import get_driver

    get_driver()
except Exception:
    __plugin_meta__ = None
else:
    from .plugin_entry import *  # noqa: F403
