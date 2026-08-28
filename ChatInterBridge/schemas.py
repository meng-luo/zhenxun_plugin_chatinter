"""HTTP request schemas for the ChatInter bridge."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, BaseModel, ConfigDict


class BridgeSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Optional[str] = None
    data: Any = None


class BridgeMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bot_id: str = "Bot"
    bot_self_id: str = ""
    msg_id: str = ""
    user_type: Literal["group", "direct", "channel", "sub_channel"] = "group"
    group_id: Optional[str] = None
    user_id: str
    sender: Dict[str, Any] = Field(default_factory=dict)
    user_pm: int = 6
    content: List[BridgeSegment] = Field(default_factory=list)


class RouteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=8, max_length=200)
    ws_bot_id: str = ""
    message: BridgeMessage


class ExecuteRequest(RouteRequest):
    capability_id: str
    revision: str = Field(min_length=16, max_length=128)
    command_text: str = Field(min_length=1, max_length=100_000)
