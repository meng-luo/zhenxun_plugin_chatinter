"""Stateless boundary from ChatInter to the host LLM transport."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import copy
from dataclasses import dataclass
from typing import Any

from zhenxun.services.ai.config import get_llm_config
from zhenxun.services.ai.core.exceptions import ConfigurationException
from zhenxun.services.ai.core.messages import ChatRequest, LLMMessage
from zhenxun.services.ai.core.models import CancellationToken, ModelCapabilities
from zhenxun.services.ai.core.options import GenerationConfig
from zhenxun.services.ai.llm.adapters import get_adapter_for_api_type
from zhenxun.services.ai.llm.manager import (
    _get_group_name,
    _resolve_model_group,
    find_model_config,
    get_default_model,
    get_model_instance,
    list_available_models,
    parse_provider_model_string,
    resolve_model_capabilities,
)
from zhenxun.utils.pydantic_compat import model_copy

_PROMPT_CACHE_KEY_API_TYPES = frozenset({"openai", "openai_responses"})


@dataclass(frozen=True, slots=True)
class HostModelCandidate:
    name: str
    capabilities: ModelCapabilities
    api_type: str = "openai"

    def context_window(self, configured_tokens: int) -> int:
        configured = max(int(configured_tokens or 0), 1)
        declared = max(int(self.capabilities.max_input_tokens or 0), 0)
        return min(configured, declared) if declared > 0 else configured


async def resolve_host_model_candidates(
    primary_model: str | None,
    fallback_models: Iterable[str] = (),
    *,
    task: str = "chat",
) -> tuple[HostModelCandidate, ...]:
    sources = [_resolve_primary_source(primary_model, task=task)]
    sources.extend(str(item or "").strip() for item in fallback_models)

    names: list[str] = []
    seen: set[str] = set()
    for index, source in enumerate(sources):
        if not source:
            continue
        expanded = _expand_model_source(source)
        if not expanded and index == 0:
            raise ConfigurationException(
                f"模型路由组 '{source}' 解析失败或为空，请检查配置。"
            )
        for name in expanded:
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            names.append(name)

    if not names:
        raise ConfigurationException("未配置任何可用的 ChatInter 模型")

    candidates: list[HostModelCandidate] = []
    for name in names:
        capabilities = await resolve_model_capabilities(name, task=task)
        api_type = _resolve_candidate_api_type(name)
        if not _api_type_supports_chat(api_type):
            raise ConfigurationException(
                f"API 类型 '{api_type}' 不支持 ChatInter 文本对话。"
            )
        candidates.append(
            HostModelCandidate(
                name=name,
                capabilities=capabilities,
                api_type=api_type,
            )
        )
    return tuple(candidates)


def _resolve_candidate_api_type(model_name: str) -> str:
    provider_name, concrete_model = parse_provider_model_string(model_name)
    if not provider_name or not concrete_model:
        return "openai"
    configured = find_model_config(provider_name, concrete_model)
    if configured is None:
        return "openai"
    provider, model = configured
    return _normalize_api_type(model.api_type or provider.api_type or "openai")


def _normalize_api_type(value: Any) -> str:
    return str(value or "openai").strip().casefold().replace("-", "_") or "openai"


def _api_type_supports_chat(api_type: str) -> bool:
    normalized = _normalize_api_type(api_type)
    if normalized == "smart":
        return True
    try:
        return get_adapter_for_api_type(normalized).text_handler is not None
    except Exception:
        return False


def _request_api_type(
    candidate: HostModelCandidate | str | None,
    model_name: str | None,
) -> str:
    if isinstance(candidate, HostModelCandidate):
        value = candidate.api_type
    else:
        resolved_name = model_name or _resolve_primary_source(None, task="chat")
        value = _resolve_candidate_api_type(resolved_name)
    return _normalize_api_type(value)


def _generation_config_with_prompt_cache_key(
    config: GenerationConfig | None,
    *,
    prompt_cache_key: str | None,
    api_type: str,
) -> GenerationConfig | None:
    if not prompt_cache_key:
        return config

    if api_type not in _PROMPT_CACHE_KEY_API_TYPES:
        if config is None or "prompt_cache_key" not in config.custom_kwargs:
            return config
        request_config = model_copy(config, deep=True)
        custom_kwargs = dict(request_config.custom_kwargs)
        custom_kwargs.pop("prompt_cache_key", None)
        return model_copy(request_config, update={"custom_kwargs": custom_kwargs})

    request_config = (
        model_copy(config, deep=True) if config is not None else GenerationConfig()
    )
    custom_kwargs = dict(request_config.custom_kwargs)
    custom_kwargs["prompt_cache_key"] = prompt_cache_key
    return model_copy(request_config, update={"custom_kwargs": custom_kwargs})


def _generation_config_without_prompt_cache_key(
    config: GenerationConfig | None,
) -> GenerationConfig | None:
    if config is None or "prompt_cache_key" not in config.custom_kwargs:
        return config
    request_config = model_copy(config, deep=True)
    custom_kwargs = dict(request_config.custom_kwargs)
    custom_kwargs.pop("prompt_cache_key", None)
    return model_copy(request_config, update={"custom_kwargs": custom_kwargs})


async def _generation_config_with_responses_replay(
    config: GenerationConfig | None,
    *,
    messages: Sequence[LLMMessage],
    api_type: str,
) -> GenerationConfig | None:
    if api_type != "openai_responses":
        return config
    replay_messages = [
        message
        for message in messages
        if message.role == "assistant"
        and bool(message.tool_calls)
        and isinstance(message.metadata, dict)
        and message.metadata.get("provider_replay_kind", "responses_output")
        == "responses_output"
        and (
            "provider_replay_payload" in message.metadata
            or "reasoning_replay_payload" in message.metadata
            or "reasoning_replay_items" in message.metadata
        )
    ]
    if not replay_messages:
        return config

    from zhenxun.services.ai.llm.adapters.handlers.openai_handlers import (
        ResponsesMessageConverter,
    )

    converter = ResponsesMessageConverter()
    raw_input: list[dict[str, Any]] = []
    for message in messages:
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        replay_kind = metadata.get(
            "provider_replay_kind",
            metadata.get("reasoning_source_wire_type", "responses_output"),
        )
        replay_items = metadata.get(
            "provider_replay_payload",
            metadata.get(
                "reasoning_replay_payload",
                metadata.get("reasoning_replay_items"),
            ),
        )
        if (
            message.role == "assistant"
            and message.tool_calls
            and replay_items is not None
            and replay_kind == "responses_output"
        ):
            if not (
                isinstance(replay_items, list)
                and replay_items
                and all(isinstance(item, dict) for item in replay_items)
            ):
                raise ValueError(
                    "Responses tool replay is missing its original output items"
                )
            expected_ids = {str(call.id or "") for call in message.tool_calls}
            replay_ids = {
                str(item.get("call_id", "") or "")
                for item in replay_items
                if item.get("type") == "function_call"
            }
            if not expected_ids or not expected_ids <= replay_ids:
                raise ValueError(
                    "Responses tool replay does not match its function calls"
                )
            raw_input.extend(copy.deepcopy(replay_items))
            continue
        raw_input.extend(await converter.convert_messages_async([message]))

    request_config = (
        model_copy(config, deep=True) if config is not None else GenerationConfig()
    )
    custom_kwargs = dict(request_config.custom_kwargs)
    custom_kwargs["input"] = raw_input
    return model_copy(request_config, update={"custom_kwargs": custom_kwargs})


def _resolve_primary_source(primary_model: str | None, *, task: str) -> str:
    source = str(primary_model or get_default_model(task) or "").strip()
    if source:
        return source
    available = list_available_models()
    if not available:
        raise ConfigurationException("未配置任何AI模型")
    return str(available[0]["full_name"])


def _expand_model_source(source: str) -> tuple[str, ...]:
    group_name = _get_group_name(source)
    if group_name is None:
        return (source,)
    expanded = tuple(_resolve_model_group(group_name))
    if expanded or group_name in get_llm_config().model_groups:
        return expanded
    return (source,)


class HostLLMClient:
    """Invoke one concrete host model without owning routing or chat state."""

    __slots__ = ()

    async def invoke(
        self,
        *,
        candidate: HostModelCandidate | str | None,
        messages: Sequence[LLMMessage],
        config: GenerationConfig | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        timeout: float | None = None,
        prompt_cache_key: str | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> Any:
        model_name = (
            candidate.name if isinstance(candidate, HostModelCandidate) else candidate
        )
        extra: dict[str, Any] = {"_is_routed_call": True}
        if prompt_cache_key:
            extra["_chatinter_prompt_cache_key"] = prompt_cache_key
        api_type = _request_api_type(candidate, model_name)
        request_config = _generation_config_with_prompt_cache_key(
            config,
            prompt_cache_key=prompt_cache_key,
            api_type=api_type if prompt_cache_key else "",
        )
        request_config = await _generation_config_with_responses_replay(
            request_config,
            messages=messages,
            api_type=api_type,
        )
        instance_config = _generation_config_without_prompt_cache_key(config)
        request = ChatRequest(
            messages=list(messages),
            config=request_config,
            tools=tools,
            tool_choice=tool_choice,
            timeout=timeout,
            extra=extra,
        )
        async with await get_model_instance(
            model_name,
            override_config=instance_config,
            task="chat",
        ) as model:
            return await model.invoke(request, cancellation_token)


__all__ = [
    "HostLLMClient",
    "HostModelCandidate",
    "resolve_host_model_candidates",
]
