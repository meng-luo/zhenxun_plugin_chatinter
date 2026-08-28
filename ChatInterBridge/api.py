"""Authenticated FastAPI endpoints exposed by ChatInterBridge."""

import hmac
import time
import hashlib
from typing import Any, Dict

from fastapi import Header, Depends, Request, HTTPException

from gsuid_core.gss import gss
from gsuid_core.webconsole.app_app import app

from .config import get_shared_secret
from .schemas import RouteRequest, ExecuteRequest
from .service import route_message, execute_capability, get_execution_status, build_capability_manifest

API_PREFIX = "/api/chatinter-bridge/v1"


_MAX_CLOCK_SKEW_SECONDS = 300


async def require_bridge_signature(
    request: Request,
    x_chatinter_timestamp: str | None = Header(default=None),
    x_chatinter_signature: str | None = Header(default=None),
) -> None:
    expected = get_shared_secret()
    if not expected:
        raise HTTPException(status_code=503, detail="ChatInterBridge shared secret is not configured")
    try:
        timestamp = int(x_chatinter_timestamp or "")
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid bridge timestamp") from exc
    if abs(int(time.time()) - timestamp) > _MAX_CLOCK_SKEW_SECONDS:
        raise HTTPException(status_code=401, detail="Bridge timestamp outside allowed clock skew")
    body = await request.body()
    signed = str(timestamp).encode() + b"." + body
    expected_signature = hmac.new(expected.encode(), signed, hashlib.sha256).hexdigest()
    if not x_chatinter_signature or not hmac.compare_digest(x_chatinter_signature, expected_signature):
        raise HTTPException(status_code=401, detail="Invalid bridge signature")


def _ok(data: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": 0, "msg": "ok", "data": data}


@app.get(f"{API_PREFIX}/health")
async def chatinter_bridge_health(
    _auth: None = Depends(require_bridge_signature),
) -> Dict[str, Any]:
    manifest, _ = build_capability_manifest()
    return _ok(
        {
            "ready": True,
            "revision": manifest["revision"],
            "active_connections": len(gss.active_ws),
        }
    )


@app.get(f"{API_PREFIX}/capabilities")
async def chatinter_bridge_capabilities(
    _auth: None = Depends(require_bridge_signature),
) -> Dict[str, Any]:
    manifest, _ = build_capability_manifest()
    return _ok(manifest)


@app.post(f"{API_PREFIX}/route")
async def chatinter_bridge_route(
    request: RouteRequest,
    _auth: None = Depends(require_bridge_signature),
) -> Dict[str, Any]:
    return _ok(await route_message(request))


@app.post(f"{API_PREFIX}/execute")
async def chatinter_bridge_execute(
    request: ExecuteRequest,
    _auth: None = Depends(require_bridge_signature),
) -> Dict[str, Any]:
    return _ok(await execute_capability(request))


@app.get(f"{API_PREFIX}/executions/{{request_id}}")
async def chatinter_bridge_execution_status(
    request_id: str,
    _auth: None = Depends(require_bridge_signature),
) -> Dict[str, Any]:
    execution = get_execution_status(request_id)
    if execution is None:
        raise HTTPException(status_code=404, detail="Bridge execution record not found")
    return _ok(execution)
