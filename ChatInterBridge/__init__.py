"""ChatInter bridge plugin for GScore command discovery and dispatch."""

from gsuid_core.sv import Plugins

from .metadata_capture import install_metadata_capture

Plugins(name="ChatInterBridge", area="ALL")
install_metadata_capture()

from . import api as api  # noqa: E402

__all__ = ["api"]
