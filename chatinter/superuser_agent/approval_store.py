"""Durable approval store for ask-gated superuser agent operations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
import uuid
from typing import Any

from ..persistence import read_json, state_path, write_json
from ..runtime_events import emit_runtime_event
from .audit_log import record_audit_event

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
    matched_pattern: str = ""
    payload_fingerprint: str = ""
    scope: str = "session"
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(
        default_factory=lambda: time.time() + _APPROVAL_TTL_SECONDS
    )
    revoked_at: float = 0.0
    revoked_reason: str = ""

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def revoked(self) -> bool:
        return self.revoked_at > 0

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
    matched_pattern: str = "",
    ttl_seconds: float | None = None,
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
        matched_pattern=str(matched_pattern or ""),
        payload_fingerprint=_payload_fingerprint(payload),
        scope="session",
        expires_at=time.time() + _coerce_ttl(ttl_seconds),
    )
    _PENDING_APPROVALS[approval.approval_id] = approval
    _save_approvals()
    _emit_approval_event(approval, status="waiting", source="approval_created")
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


def revoke_pending_approval(
    *,
    approval_id: str,
    user_id: str,
    session_key: str,
    reason: str = "",
) -> PendingApproval | None:
    approval = _pop_pending_approval(
        approval_id=approval_id,
        user_id=user_id,
        session_key=session_key,
    )
    if approval is None:
        return None
    revoked = PendingApproval(
        approval_id=approval.approval_id,
        user_id=approval.user_id,
        session_key=approval.session_key,
        action=approval.action,
        payload=approval.payload,
        reason=approval.reason,
        matched_pattern=approval.matched_pattern,
        payload_fingerprint=approval.payload_fingerprint,
        scope=approval.scope,
        created_at=approval.created_at,
        expires_at=approval.expires_at,
        revoked_at=time.time(),
        revoked_reason=str(reason or ""),
    )
    record_audit_event(
        event="approval_revoked",
        user_id=revoked.user_id,
        session_key=revoked.session_key,
        action=revoked.action,
        payload={
            "approval_id": revoked.approval_id,
            "reason": revoked.revoked_reason,
        },
        result={"revoked": True},
    )
    _emit_approval_event(revoked, status="cancelled", source="approval_revoked")
    return revoked


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
    if popped is not None:
        _emit_approval_event(popped, status="completed", source="approval_consumed")
    return popped


def _purge_expired() -> None:
    changed = False
    for approval_id, approval in list(_PENDING_APPROVALS.items()):
        if approval.expired:
            _PENDING_APPROVALS.pop(approval_id, None)
            changed = True
            record_audit_event(
                event="approval_expired",
                user_id=approval.user_id,
                session_key=approval.session_key,
                action=approval.action,
                payload={"approval_id": approval.approval_id},
                result={"expired": True},
            )
            _emit_approval_event(approval, status="expired", source="approval_expired")
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
            matched_pattern=str(data.get("matched_pattern", "") or ""),
            payload_fingerprint=str(data.get("payload_fingerprint", "") or "")
            or _payload_fingerprint(dict(data.get("payload") or {})),
            scope=str(data.get("scope", "") or "session"),
            created_at=float(data.get("created_at") or time.time()),
            expires_at=float(
                data.get("expires_at") or time.time() + _APPROVAL_TTL_SECONDS
            ),
            revoked_at=float(data.get("revoked_at") or 0.0),
            revoked_reason=str(data.get("revoked_reason", "") or ""),
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


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    try:
        text = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = str(payload or {})
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _coerce_ttl(value: float | None) -> float:
    try:
        seconds = float(value or _APPROVAL_TTL_SECONDS)
    except (TypeError, ValueError):
        seconds = _APPROVAL_TTL_SECONDS
    return max(30.0, min(seconds, 3600.0))


def _emit_approval_event(
    approval: PendingApproval,
    *,
    status: str,
    source: str,
) -> None:
    emit_runtime_event(
        kind="approval",
        status=status,  # type: ignore[arg-type]
        source=source,
        session_key=approval.session_key,
        user_id=approval.user_id,
        summary=f"{approval.action}:{approval.approval_id}",
        payload=approval.to_public_payload(),
        related_ids={"approval_id": approval.approval_id},
    )


__all__ = [
    "PendingApproval",
    "consume_pending_approval",
    "create_pending_approval",
    "get_pending_approval",
    "list_pending_approvals",
    "reject_pending_approval",
    "revoke_pending_approval",
]
