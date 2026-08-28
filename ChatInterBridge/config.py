"""Configuration for the ChatInter bridge."""

from typing import Dict

from gsuid_core.data_store import get_res_path
from gsuid_core.utils.plugins_config.models import GSC, GsStrConfig
from gsuid_core.utils.plugins_config.gs_config import StringConfig

CONFIG_DEFAULT: Dict[str, GSC] = {
    "shared_secret": GsStrConfig(
        title="ChatInter Bridge Secret",
        desc="Shared secret used to sign ChatInter Bridge HTTP requests.",
        data="",
        secret=True,
    ),
}

BRIDGE_CONFIG = StringConfig(
    "ChatInterBridge",
    get_res_path("ChatInterBridge") / "config.json",
    CONFIG_DEFAULT,
)


def get_shared_secret() -> str:
    return str(BRIDGE_CONFIG.get_config("shared_secret").data or "")
