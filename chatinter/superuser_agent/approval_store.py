"""Durable approval store for ask-gated superuser agent operations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
import uuid
from typing import Any

from ..persistence import read_json, state_path, write_json

_APPROVAL_TTL_SECONDS = 300.0
_APPROVALS_PATH = state_path("approvals.json")
_PENDING_APPROVALS: dict[str, "PendingApproval"] = {}
_LOADED = False


@dataclass(frozen=True)
class PendingApproval:
    approval_id: str
    user_id: str
    session_key: str
    action: str
    payload: dict[str, Any]
    reason: str = ""
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(
        default_factory=lambda: time.time() + _APPROVAL_TTL_SECONDS
    )

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at

    def to_public_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ttl_seconds"] = max(0, int(self.expires_at - time.time()))
        payload.pop("created_at", None)
        payload.pop("expires_at", None)
        return payload


def create_pending_approval(
    *,
    user_id: str,
    session_key: str,
    action: str,
    payload: dict[str, Any],
    reason: str = "",
) -> PendingApproval:
    _ensure_loaded()
    _purge_expired()
    approval = PendingApproval(
        approval_id=uuid.uuid4().hex[:10],
        user_id=str(user_id or ""),
        session_key=str(session_key or ""),
        action=str(action or ""),
        payload=dict(payload or {}),
        reason=str(reason or ""),
    )
    _PENDING_APPROVALS[approval.approval_id] = approval
    _save_approvals()
    return approval


def consume_pending_approval(
    *,
    approval_id: str,
    user_id: str,
    session_key: str,
) -> PendingApproval | None:
    return _pop_pending_approval(
        approval_id=approval_id,
        user_id=user_id,
        session_key=session_key,
    )


def reject_pending_approval(
    *,
    approval_id: str,
    user_id: str,
    session_key: str,
) -> PendingApproval | None:
    return _pop_pending_approval(
        approval_id=approval_id,
        user_id=user_id,
        session_key=session_key,
    )


def get_pending_approval(
    *,
    approval_id: str,
    user_id: str,
    session_key: str,
) -> PendingApproval | None:
    _ensure_loaded()
    _purge_expired()
    key = str(approval_id or "").strip()
    approval = _PENDING_APPROVALS.get(key)
    if approval is None or approval.expired:
        _PENDING_APPROVALS.pop(key, None)
        _save_approvals()
        return None
    if approval.user_id != str(user_id or ""):
        return None
    if approval.session_key != str(session_key or ""):
        return None
    return approval


def list_pending_approvals(
    *,
    user_id: str,
    session_key: str,
) -> list[PendingApproval]:
    _ensure_loaded()
    _purge_expired()
    return [
        approval
        for approval in _PENDING_APPROVALS.values()
        if approval.user_id == str(user_id or "")
        and approval.session_key == str(session_key or "")
    ]


def _pop_pending_approval(
    *,
    approval_id: str,
    user_id: str,
    session_key: str,
) -> PendingApproval | None:
    _ensure_loaded()
    _purge_expired()
    key = str(approval_id or "").strip()
    approval = _PENDING_APPROVALS.get(key)
    if approval is None:
        return None
    if approval.expired:
        _PENDING_APPROVALS.pop(key, None)
        _save_approvals()
        return None
    if approval.user_id != str(user_id or ""):
        return None
    if approval.session_key != str(session_key or ""):
        return None
    popped = _PENDING_APPROVALS.pop(key, None)
    _save_approvals()
    return popped


def _purge_expired() -> None:
    changed = False
    for approval_id, approval in list(_PENDING_APPROVALS.items()):
        if approval.expired:
            _PENDING_APPROVALS.pop(approval_id, None)
            changed = True
    if changed:
        _save_approvals()


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    raw = read_json(_APPROVALS_PATH, {})
    if not isinstance(raw, dict):
        return
    for approval_id, payload in raw.items():
        approval = _approval_from_payload(approval_id, payload)
        if approval is None or approval.expired:
            continue
        _PENDING_APPROVALS[approval.approval_id] = approval
    _save_approvals()


def _approval_from_payload(
    approval_id: object,
    payload: object,
) -> PendingApproval | None:
    if not isinstance(payload, dict):
        return None
    data = dict(payload)
    data["approval_id"] = str(data.get("approval_id") or approval_id or "")
    if not data["approval_id"]:
        return None
    try:
        return PendingApproval(
            approval_id=str(data["approval_id"]),
            user_id=str(data.get("user_id", "") or ""),
            session_key=str(data.get("session_key", "") or ""),
            action=str(data.get("action", "") or ""),
            payload=dict(data.get("payload") or {}),
            reason=str(data.get("reason", "") or ""),
            created_at=float(data.get("created_at") or time.time()),
            expires_at=float(
                data.get("expires_at") or time.time() + _APPROVAL_TTL_SECONDS
            ),
        )
    except Exception:
        return None


def _save_approvals() -> None:
    write_json(
        _APPROVALS_PATH,
        {
            approval_id: asdict(approval)
            for approval_id, approval in sorted(_PENDING_APPROVALS.items())
            if not approval.expired
        },
    )


__all__ = [
    "PendingApproval",
    "consume_pending_approval",
    "create_pending_approval",
    "get_pending_approval",
    "list_pending_approvals",
    "reject_pending_approval",
]
