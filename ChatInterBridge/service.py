"""Pure routing inspection and idempotent GScore queue submission."""

from __future__ import annotations

import json
import time
import asyncio
import hashlib
from copy import deepcopy
from uuid import uuid4
from typing import Any, Dict, Mapping, Iterable, Optional
from dataclasses import dataclass

from gsuid_core.sv import SL, SV
from gsuid_core.bot import Bot, _Bot
from gsuid_core.gss import gss
from gsuid_core.logger import logger
from gsuid_core.models import Event, Message, TaskContext, TraceContext, MessageReceive
from gsuid_core.handler import count_data, msg_process, get_user_pml, _command_start, _sv_authorized
from gsuid_core.trigger import Trigger
from gsuid_core.utils.plugins_config.gs_config import sp_config

from .schemas import RouteRequest, BridgeMessage, ExecuteRequest
from .execution import (
    BridgeBot,
    ClaimCreated,
    ExecutionStore,
    ExecuteResponse,
    ExecutionSnapshot,
    track_execution,
)
from .metadata_capture import trigger_ai_description

_ROUTABLE_TYPES = frozenset({"prefix", "suffix", "keyword", "fullmatch", "command", "file", "regex"})
_EXPOSED_TYPES = _ROUTABLE_TYPES - {"file"}
_IDEMPOTENCY_TTL_SECONDS = 600.0
_IDEMPOTENCY_MAX_ENTRIES = 4096


@dataclass(frozen=True)
class CapabilityBinding:
    capability_id: str
    plugin_name: str
    service: SV
    triggers: tuple[Trigger, ...]
    card: Dict[str, Any]

    @property
    def trigger(self) -> Trigger:
        return self.triggers[0]


@dataclass(frozen=True)
class CapabilityMetadata:
    sort_key: str
    description: str = ""
    retrieval_summary: str = ""
    input_schema: Optional[Dict[str, Any]] = None
    aliases: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    context_tags: tuple[str, ...] = ()
    capability_domain: str = ""


_execution_records = ExecutionStore(
    ttl_seconds=_IDEMPOTENCY_TTL_SECONDS,
    max_entries=_IDEMPOTENCY_MAX_ENTRIES,
)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _handler_identity(trigger: Trigger) -> str:
    return _function_identity(trigger.func)


def _function_identity(func: Any) -> str:
    return f"{getattr(func, '__module__', '')}:{getattr(func, '__qualname__', '')}"


def _original_handler(func: Any) -> Any:
    chain = []
    current = func
    while callable(current) and all(current is not item for item in chain):
        chain.append(current)
        wrapped = getattr(current, "__wrapped__", None)
        if not callable(wrapped):
            break
        current = wrapped
    return current


def _same_handler(left: Any, right: Any) -> bool:
    left_chain = []
    current = left
    while callable(current) and all(current is not item for item in left_chain):
        left_chain.append(current)
        current = getattr(current, "__wrapped__", None)
    right_chain = []
    current = right
    while callable(current) and all(current is not item for item in right_chain):
        right_chain.append(current)
        current = getattr(current, "__wrapped__", None)
    return any(left_item is right_item for left_item in left_chain for right_item in right_chain)


def _acl_fingerprint(values: Iterable[Any]) -> str:
    return _canonical_hash(sorted(str(value) for value in values))


def _trigger_route(trigger: Trigger) -> Dict[str, Any]:
    return {
        "type": trigger.type,
        "keyword": trigger.keyword,
        "prefix": trigger.prefix,
        "block": bool(trigger.block),
        "to_me": bool(trigger.to_me),
    }


def _trigger_sort_key(trigger: Trigger) -> tuple[str, str, str]:
    return (trigger.type, trigger.prefix, trigger.keyword)


def _binding_for(service: SV, triggers: Iterable[Trigger]) -> CapabilityBinding:
    plugin = service.plugins
    ordered_triggers = tuple(sorted(triggers, key=_trigger_sort_key))
    trigger = ordered_triggers[0]
    routes = [_trigger_route(item) for item in ordered_triggers]
    identity = {
        "plugin": plugin.name,
        "service": service.name,
        "handler": _handler_identity(trigger),
        "routes": routes,
    }
    capability_id = f"gscore::v1:{_canonical_hash(identity)[:32]}"
    command = f"{trigger.prefix}{trigger.keyword}"
    plugin_aliases = sorted(str(item) for item in plugin.alias if str(item).strip())
    commands = [f"{item.prefix}{item.keyword}" for item in ordered_triggers]
    keywords = [item.keyword for item in ordered_triggers]
    examples = [_trigger_example(item) for item in ordered_triggers]
    card = {
        "capability_id": capability_id,
        "name": f"{service.name} - {command}",
        "plugin": plugin.name,
        "service": service.name,
        "description": f"GScore {plugin.name} 插件的「{service.name}」命令：{command}",
        "aliases": _unique_strings([*commands, *keywords, *plugin_aliases]),
        "examples": _unique_strings(examples),
        "trigger": {
            "type": trigger.type,
            "keyword": trigger.keyword,
            "prefix": trigger.prefix,
            "command": command,
            "prefixes": list(dict.fromkeys(item.prefix for item in ordered_triggers)),
            "commands": list(dict.fromkeys(commands)),
            "block": bool(trigger.block),
            "to_me": bool(trigger.to_me),
            "argument_mode": "text_suffix",
            "routable": trigger.type in _ROUTABLE_TYPES,
        },
    }
    return CapabilityBinding(capability_id, plugin.name, service, ordered_triggers, card)


def _trigger_example(trigger: Trigger) -> str:
    command = f"{trigger.prefix}{trigger.keyword}"
    if trigger.type == "prefix":
        return f"{command}<参数>"
    if trigger.type == "suffix":
        return f"{trigger.prefix}<参数>{trigger.keyword}"
    if trigger.type == "keyword":
        return f"{trigger.prefix}<文本>{trigger.keyword}"
    if trigger.type == "regex":
        return ""
    return command


def _unique_strings(values: Iterable[Any]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _metadata_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(_unique_strings([value]))
    if isinstance(value, Iterable):
        return tuple(_unique_strings(value))
    return ()


def _tool_function(tool_base: Any) -> Any:
    return getattr(getattr(tool_base, "tool", None), "function", None)


def _tool_schema(tool_base: Any) -> Optional[Dict[str, Any]]:
    function_schema = getattr(getattr(tool_base, "tool", None), "function_schema", None)
    schema = getattr(function_schema, "json_schema", None)
    if not isinstance(schema, dict):
        return None
    try:
        json.dumps(schema, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return None
    return deepcopy(schema)


def _metadata_from_tool(
    tool_base: Any,
    *,
    sort_key: str,
    description: str = "",
) -> CapabilityMetadata:
    resolved_description = (
        description or str(getattr(tool_base, "description", ""))
    ).strip()
    return CapabilityMetadata(
        sort_key=sort_key,
        description=resolved_description,
        retrieval_summary=_retrieval_summary(resolved_description),
        input_schema=_tool_schema(tool_base),
        aliases=_metadata_strings(getattr(tool_base, "aliases", None)),
        examples=_metadata_strings(getattr(tool_base, "examples", None)),
        context_tags=_metadata_strings(getattr(tool_base, "context_tags", None)),
        capability_domain=str(getattr(tool_base, "capability_domain", "") or "").strip(),
    )


def _load_ai_registries() -> Optional[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    try:
        from gsuid_core.ai_core.register import get_registered_tools
        from gsuid_core.ai_core.trigger_bridge import _MCP_TRIGGER_REGISTRY

        registered = get_registered_tools()
        if not isinstance(registered, Mapping) or not isinstance(_MCP_TRIGGER_REGISTRY, Mapping):
            return None
        return registered, _MCP_TRIGGER_REGISTRY
    except Exception:
        return None


def _keyword_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(str(item) for item in value)
    return ()


def _collect_ai_metadata(
    bindings: Mapping[str, CapabilityBinding],
) -> Dict[str, list[CapabilityMetadata]]:
    result: Dict[str, list[CapabilityMetadata]] = {}
    for capability_id, binding in bindings.items():
        description = trigger_ai_description(binding.trigger.func)
        if description:
            result[capability_id] = [
                CapabilityMetadata(
                    sort_key="to_ai:trigger",
                    description=description,
                    retrieval_summary=_retrieval_summary(description),
                )
            ]

    loaded = _load_ai_registries()
    if loaded is None:
        return result
    registered, trigger_registry = loaded
    by_identity: Dict[tuple[str, str], list[CapabilityBinding]] = {}
    for binding in bindings.values():
        key = (binding.plugin_name, _handler_identity(binding.trigger))
        by_identity.setdefault(key, []).append(binding)

    by_trigger = registered.get("by_trigger", {})
    if isinstance(by_trigger, Mapping):
        for tool_name in sorted(trigger_registry):
            entry = trigger_registry[tool_name]
            if not isinstance(entry, Mapping):
                continue
            tool_base = by_trigger.get(tool_name)
            func = entry.get("func")
            plugin_name = str(entry.get("plugin_name", ""))
            handler_id = _function_identity(func)
            if (
                tool_base is None
                or not plugin_name
                or str(getattr(tool_base, "plugin", "")) != plugin_name
                or _function_identity(_tool_function(tool_base)) != handler_id
            ):
                continue
            trigger_type = str(entry.get("trigger_type", ""))
            keywords = set(_keyword_values(entry.get("keyword")))
            service = entry.get("sv")
            for binding in by_identity.get((plugin_name, handler_id), []):
                matching_trigger = any(
                    _same_handler(trigger.func, func) and trigger.type == trigger_type and trigger.keyword in keywords
                    for trigger in binding.triggers
                )
                if binding.service is not service or not matching_trigger:
                    continue
                metadata = _metadata_from_tool(
                    tool_base,
                    sort_key=f"to_ai:{tool_name}",
                    description=str(entry.get("to_ai_doc", "")),
                )
                result.setdefault(binding.capability_id, []).append(metadata)
    return result


def _single_text(metadata: list[CapabilityMetadata], field: str) -> str:
    values = _unique_strings(getattr(item, field) for item in metadata)
    return values[0] if len(values) == 1 else ""


def _single_schema(metadata: list[CapabilityMetadata]) -> Optional[Dict[str, Any]]:
    by_hash: Dict[str, Dict[str, Any]] = {}
    for item in metadata:
        if item.input_schema is None:
            continue
        by_hash[_canonical_hash(item.input_schema)] = item.input_schema
    if len(by_hash) != 1:
        return None
    return deepcopy(next(iter(by_hash.values())))


def _retrieval_summary(description: str) -> str:
    normalized = str(description or "").replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for line in normalized.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines:
                break
            continue
        lines.append(stripped)
    return " ".join(lines)


def _enrich_card(card: Dict[str, Any], metadata: list[CapabilityMetadata]) -> Dict[str, Any]:
    if not metadata:
        return card
    ordered = sorted(metadata, key=lambda item: item.sort_key)
    description = _single_text(ordered, "description")
    retrieval_summary = _single_text(ordered, "retrieval_summary")
    schema = _single_schema(ordered)

    enriched = dict(card)
    if description:
        enriched["description"] = description
    if retrieval_summary:
        enriched["retrieval_summary"] = retrieval_summary
    enriched["aliases"] = _unique_strings(
        [*card.get("aliases", []), *(alias for item in ordered for alias in item.aliases)]
    )
    enriched["examples"] = _unique_strings(
        [*card.get("examples", []), *(example for item in ordered for example in item.examples)]
    )
    enriched["source"] = "to_ai"
    if schema is not None:
        enriched["input_schema"] = schema
    context_tags = _unique_strings(tag for item in ordered for tag in item.context_tags)
    if context_tags:
        enriched["context_tags"] = context_tags
    capability_domain = _single_text(ordered, "capability_domain")
    if capability_domain:
        enriched["capability_domain"] = capability_domain
    return enriched


def build_capability_manifest() -> tuple[Dict[str, Any], Dict[str, CapabilityBinding]]:
    bindings: Dict[str, CapabilityBinding] = {}
    for service_name in sorted(SL.lst):
        service = SL.lst[service_name]
        plugin = getattr(service, "plugins", None)
        if plugin is None or plugin.name == "ChatInterBridge":
            continue
        trigger_groups: Dict[tuple[int, str, bool, bool], list[Trigger]] = {}
        for trigger_type in sorted(service.TL):
            trigger_map = service.TL[trigger_type]
            for trigger_key in sorted(trigger_map):
                trigger = trigger_map[trigger_key]
                group_key = (
                    id(_original_handler(trigger.func)),
                    trigger.type,
                    bool(trigger.block),
                    bool(trigger.to_me),
                )
                trigger_groups.setdefault(group_key, []).append(trigger)
        for group_key in sorted(trigger_groups, key=lambda item: (item[1:], item[0])):
            binding = _binding_for(service, trigger_groups[group_key])
            if binding.capability_id in bindings:
                continue
            bindings[binding.capability_id] = binding
    metadata_by_capability = _collect_ai_metadata(bindings)
    bindings = {
        capability_id: CapabilityBinding(
            binding.capability_id,
            binding.plugin_name,
            binding.service,
            binding.triggers,
            _enrich_card(binding.card, metadata_by_capability.get(capability_id, [])),
        )
        for capability_id, binding in bindings.items()
    }
    revision_rows = [
        {
            "card": binding.card,
            "handler": _handler_identity(binding.trigger),
            "routes": [_trigger_route(trigger) for trigger in binding.triggers],
            "priority": int(binding.service.priority),
            "plugin_enabled": bool(binding.service.plugins.enabled),
            "plugin_pm": int(binding.service.plugins.pm),
            "plugin_area": str(binding.service.plugins.area),
            "plugin_blacklist": _acl_fingerprint(binding.service.plugins.black_list),
            "plugin_whitelist": _acl_fingerprint(binding.service.plugins.white_list),
            "service_enabled": bool(binding.service.enabled),
            "service_pm": int(binding.service.pm),
            "service_area": str(binding.service.area),
            "service_blacklist": _acl_fingerprint(binding.service.black_list),
            "service_whitelist": _acl_fingerprint(binding.service.white_list),
        }
        for binding in bindings.values()
    ]
    capabilities = [bindings[key].card for key in sorted(bindings) if _public_capability(bindings[key])]
    revision_source = {
        "command_start": list(_command_start),
        "capabilities": sorted(revision_rows, key=lambda item: item["card"]["capability_id"]),
    }
    manifest = {
        "revision": _canonical_hash(revision_source),
        "capabilities": capabilities,
    }
    return manifest, bindings


def _public_capability(binding: CapabilityBinding) -> bool:
    plugin = binding.service.plugins
    return bool(
        binding.trigger.type in _EXPOSED_TYPES
        and plugin.enabled
        and binding.service.enabled
        and int(plugin.pm) >= 6
        and int(binding.service.pm) >= 6
    )


def _segments(event: BridgeMessage) -> list[Message]:
    return [Message(type=segment.type, data=segment.data) for segment in event.content]


def _message_receive(
    event: BridgeMessage,
    command_text: Optional[str] = None,
) -> MessageReceive:
    content = _segments(event)
    if command_text is not None:
        content = [segment for segment in content if segment.type != "text"]
        content.insert(0, Message(type="text", data=command_text))
    return MessageReceive(
        bot_id=event.bot_id,
        bot_self_id=event.bot_self_id,
        msg_id=event.msg_id,
        user_type=event.user_type,
        group_id=event.group_id,
        user_id=event.user_id,
        sender=event.sender,
        user_pm=event.user_pm,
        content=content,
    )


def _pure_event(message: BridgeMessage) -> tuple[Event, bool]:
    bot_id = message.bot_id.split(":", 1)[0] if ":" in message.bot_id else message.bot_id
    event = Event(
        bot_id=bot_id,
        bot_self_id=message.bot_self_id,
        msg_id=message.msg_id,
        user_type=message.user_type,
        group_id=message.group_id,
        user_id=message.user_id,
        sender=message.sender,
        user_pm=message.user_pm,
        real_bot_id=message.bot_id,
    )
    event.is_tome = message.user_type == "direct"
    reliable = True
    content = _segments(message)
    for segment in content:
        if segment.type == "text" and segment.data:
            value = str(segment.data).strip()
            event.raw_text += value
            event.text += value
        elif segment.type == "at":
            target = str(segment.data)
            if target == str(event.bot_self_id):
                event.is_tome = True
            else:
                event.at = target
                event.at_list.append(target)
        elif segment.type == "image" and segment.data:
            event.image = str(segment.data)
            event.image_list.append(segment.data)
        elif segment.type == "reply":
            event.reply = segment.data
        elif segment.type == "file" and segment.data:
            file_name, separator, value = str(segment.data).partition("|")
            if not separator:
                reliable = False
                continue
            event.file_name = file_name
            event.file = value
            event.file_type = "url" if value.startswith(("http", "https")) else "base64"
        elif segment.type and segment.type.startswith("meta-"):
            reliable = False
    event.content = content
    return event, reliable


def _apply_command_start(event: Event) -> bool:
    if not _command_start or not event.raw_text:
        return True
    for start in _command_start:
        if event.raw_text.strip().startswith(start):
            event.raw_text = event.raw_text.replace(start, "", 1)
            return True
    return False


def _interaction_consumes(event: Event) -> bool:
    uid = event.user_id or "0"
    temp_gid = event.group_id if event.user_type != "direct" else uid
    temp_gid = temp_gid or "0"
    session_id = f"{event.bot_id or '0'}%%%{temp_gid}%%%{uid}"
    instance = Bot.get_instances().get(session_id)
    if instance is not None and instance.receive_tag:
        return True
    owner = Bot.get_mutiply_map().get(temp_gid)
    multiple = Bot.get_mutiply_instances().get(owner) if owner is not None else None
    return bool(multiple is not None and multiple.mutiply_tag and multiple.session_id == session_id)


def _globally_blocked(event: Event) -> bool:
    blacklist = sp_config.get_config("BlackList").data
    return event.group_id in blacklist or event.user_id in blacklist


def _shielded(event: Event) -> bool:
    if not event.at:
        return False
    return any(event.at.startswith(str(item)) for item in sp_config.get_config("ShieldQQBot").data)


def _matching_trigger(binding: CapabilityBinding, event: Event) -> tuple[Optional[Trigger], bool]:
    matched = []
    had_error = False
    for trigger in binding.triggers:
        try:
            if trigger.check_command(event):
                matched.append(trigger)
        except Exception:
            had_error = True
    if not matched:
        return None, had_error
    matched.sort(
        key=lambda trigger: (
            -len(f"{trigger.prefix}{trigger.keyword}"),
            _trigger_sort_key(trigger),
        )
    )
    return matched[0], had_error


async def route_message(request: RouteRequest) -> Dict[str, Any]:
    revision = ""
    try:
        manifest, bindings = build_capability_manifest()
        revision = manifest["revision"]
        event, reliable = _pure_event(request.message)
        event.WS_BOT_ID = request.ws_bot_id
        event.user_pm = await get_user_pml(_message_receive(request.message))
        if _interaction_consumes(event):
            return _route_response(revision, "interactive")
        if _globally_blocked(event) or _shielded(event):
            return _route_response(revision, "blocked")
        if not _apply_command_start(event):
            return _route_response(revision, "unmatched")
        claimed: list[tuple[CapabilityBinding, Trigger]] = []
        denied: list[tuple[CapabilityBinding, Trigger]] = []
        had_error = not reliable
        for binding in bindings.values():
            if binding.trigger.type not in _ROUTABLE_TYPES:
                continue
            trigger, match_error = _matching_trigger(binding, event)
            had_error = had_error or match_error
            if trigger is None:
                continue
            if _sv_authorized(binding.service, event, event.user_pm):
                claimed.append((binding, trigger))
            else:
                denied.append((binding, trigger))
        if claimed:
            claimed.sort(
                key=lambda item: (
                    not bool(item[1].prefix),
                    item[0].service.priority,
                    item[0].capability_id,
                )
            )
            return _route_response(revision, "claimed", [item[0].capability_id for item in claimed])
        if denied:
            return _route_response(revision, "blocked", [item[0].capability_id for item in denied])
        if had_error:
            return _route_response(revision, "unknown")
        return _route_response(revision, "unmatched")
    except Exception:
        logger.exception("ChatInterBridge route inspection failed")
        return _route_response(revision, "unknown", reason="inspection_failed")


def _route_response(
    revision: str,
    disposition: str,
    matches: Optional[list[str]] = None,
    *,
    reason: str = "",
) -> Dict[str, Any]:
    response: Dict[str, Any] = {
        "disposition": disposition,
        "revision": revision,
        "matches": matches or [],
    }
    if reason:
        response["reason"] = reason
    return response


def _idempotency_fingerprint(request: ExecuteRequest) -> str:
    return _canonical_hash(request.model_dump(mode="json"))


def _claim_request(key: str, fingerprint: str) -> ExecuteResponse | None:
    result = _execution_records.claim(key, fingerprint)
    if isinstance(result, ClaimCreated):
        return None
    return result.response


def _complete_request(key: str, response: ExecuteResponse) -> None:
    state = "rejected" if response["disposition"] == "rejected" else "unknown"
    _execution_records.finish_preparation(key, response, state)


def _active_bot(ws_bot_id: str) -> Optional[_Bot]:
    bot = gss.active_bot.get(ws_bot_id) if ws_bot_id else None
    if bot is None and len(gss.active_bot) == 1:
        bot = next(iter(gss.active_bot.values()))
    if bot is None or bot.bot is None:
        return None
    return bot


async def execute_capability(request: ExecuteRequest) -> ExecuteResponse:
    manifest, bindings = build_capability_manifest()
    revision = manifest["revision"]
    fingerprint = _idempotency_fingerprint(request)
    duplicate = _claim_request(request.request_id, fingerprint)
    if duplicate is not None:
        duplicate.setdefault("revision", revision)
        return duplicate
    try:
        return await _execute_claimed_capability(
            request,
            manifest=manifest,
            bindings=bindings,
        )
    except asyncio.CancelledError:
        _complete_request(
            request.request_id,
            {
                "disposition": "unknown",
                "reason": "submission_cancelled",
                "revision": revision,
                "duplicate": False,
            },
        )
        raise
    except Exception:
        logger.exception("ChatInterBridge execution preparation failed")
        response = {
            "disposition": "unknown",
            "reason": "preparation_failed",
            "revision": revision,
            "duplicate": False,
        }
        _complete_request(request.request_id, response)
        return response


async def _execute_claimed_capability(
    request: ExecuteRequest,
    *,
    manifest: Dict[str, Any],
    bindings: Dict[str, CapabilityBinding],
) -> ExecuteResponse:
    revision = manifest["revision"]
    if request.revision != revision:
        response = {
            "disposition": "rejected",
            "reason": "stale_revision",
            "revision": revision,
            "duplicate": False,
        }
        _complete_request(request.request_id, response)
        return response
    binding = bindings.get(request.capability_id)
    if binding is None or not _public_capability(binding):
        response = {
            "disposition": "rejected",
            "reason": "unknown_capability",
            "revision": revision,
            "duplicate": False,
        }
        _complete_request(request.request_id, response)
        return response
    incoming = _message_receive(request.message, request.command_text)
    incoming.user_pm = await get_user_pml(incoming)
    event = await msg_process(incoming)
    event.WS_BOT_ID = request.ws_bot_id
    if _interaction_consumes(event):
        response = {
            "disposition": "rejected",
            "reason": "interactive_session",
            "revision": revision,
            "duplicate": False,
        }
        _complete_request(request.request_id, response)
        return response
    if _globally_blocked(event) or _shielded(event):
        response = {"disposition": "rejected", "reason": "blocked", "revision": revision, "duplicate": False}
        _complete_request(request.request_id, response)
        return response
    if not _sv_authorized(binding.service, event, event.user_pm):
        response = {
            "disposition": "rejected",
            "reason": "permission_denied",
            "revision": revision,
            "duplicate": False,
        }
        _complete_request(request.request_id, response)
        return response
    trigger, match_error = _matching_trigger(binding, event)
    if trigger is None and match_error:
        logger.warning("ChatInterBridge trigger validation failed")
        response = {
            "disposition": "unknown",
            "reason": "trigger_validation_failed",
            "revision": revision,
            "duplicate": False,
        }
        _complete_request(request.request_id, response)
        return response
    if trigger is None:
        response = {
            "disposition": "rejected",
            "reason": "trigger_mismatch",
            "revision": revision,
            "duplicate": False,
        }
        _complete_request(request.request_id, response)
        return response
    ws = _active_bot(request.ws_bot_id)
    if ws is None:
        response = {
            "disposition": "rejected",
            "reason": "bot_offline",
            "revision": revision,
            "duplicate": False,
        }
        _complete_request(request.request_id, response)
        return response
    coro = None
    tracked_coro = None
    queued = False
    try:
        command_event = deepcopy(event)
        command_event.task_id = str(uuid4())
        command_event = await trigger.get_command(command_event)
        bot = BridgeBot(ws, command_event, request.request_id, _execution_records)
        await count_data(command_event, trigger)
        coro = trigger.func(bot, command_event)
        name = getattr(coro, "__qualname__", str(coro))
        trace = TraceContext(
            trace_id=command_event.task_id,
            short_id=command_event.task_id[:8],
            command=command_event.command or trigger.keyword or "",
            user_id=command_event.user_id,
            group_id=command_event.group_id,
            bot_id=command_event.bot_id,
            session_id=command_event.session_id,
            start_time=time.perf_counter(),
            start_ts=time.time(),
        )
        tracked_coro = track_execution(_execution_records, request.request_id, coro)
        ws.queue.put_nowait(
            TaskContext(
                coro=tracked_coro,
                name=name,
                priority=command_event.user_pm,
                trace_context=trace,
            )
        )
        queued = True
        response = {
            "disposition": "accepted",
            "request_id": request.request_id,
            "task_id": command_event.task_id,
            "revision": revision,
            "duplicate": False,
        }
        _execution_records.mark_accepted(request.request_id, command_event.task_id, response)
        return response
    except asyncio.CancelledError:
        if tracked_coro is not None and not queued:
            tracked_coro.close()
        if coro is not None and not queued:
            coro.close()
        _complete_request(
            request.request_id,
            {
                "disposition": "unknown",
                "reason": "submission_cancelled",
                "revision": revision,
                "duplicate": False,
            },
        )
        raise
    except Exception:
        if tracked_coro is not None and not queued:
            tracked_coro.close()
        if coro is not None and not queued:
            coro.close()
        logger.exception("ChatInterBridge queue submission failed")
        response = {
            "disposition": "unknown",
            "reason": "submission_failed",
            "revision": revision,
            "duplicate": False,
        }
        _complete_request(request.request_id, response)
        return response


def get_execution_status(request_id: str) -> ExecutionSnapshot | None:
    return _execution_records.snapshot(request_id)


def reset_idempotency_for_tests() -> None:
    _execution_records.reset()
