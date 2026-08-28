"""Stateless read-only web capability policy for ChatInter requests."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .config import get_web_access_mode
from .route_text import normalize_message_text

if TYPE_CHECKING:
    from .host_llm import HostModelCandidate

WebAccessScope = Literal["chat", "superuser"]
WebSearchToolKind = Literal["native", "client"]

_NATIVE_SEARCH_API_TYPES = frozenset({"openai_responses", "gemini", "mimo"})
_SENSITIVE_QUERY_KEYS = frozenset(
    {
        "accesstoken",
        "apikey",
        "auth",
        "authorization",
        "awsaccesskeyid",
        "credential",
        "clientsecret",
        "googleaccessid",
        "key",
        "oauthtoken",
        "password",
        "secret",
        "sessiontoken",
        "signature",
        "sig",
        "token",
        "xamzcredential",
        "xamzsecuritytoken",
        "xamzsignature",
        "xgoogcredential",
        "xgoogsignature",
    }
)
_MAX_CITATIONS = 5
_MAX_CITATION_TITLE_CHARS = 120


@dataclass(frozen=True, slots=True)
class WebCitation:
    title: str
    url: str


@dataclass(frozen=True, slots=True)
class WebResponseProjection:
    citations: tuple[WebCitation, ...]
    search_used: bool = False


def web_access_enabled(scope: WebAccessScope) -> bool:
    mode = get_web_access_mode()
    return mode == "all" or (mode == "agent" and scope == "superuser")


def candidate_supports_native_web_search(
    candidate: HostModelCandidate,
    *,
    has_client_tools: bool,
) -> bool:
    if "web_search" not in candidate.capabilities.supported_native_tools:
        return False
    api_type = str(candidate.api_type or "").strip().casefold().replace("-", "_")
    if api_type not in _NATIVE_SEARCH_API_TYPES:
        return False
    if api_type == "gemini" and has_client_tools:
        return False
    return True


def candidate_web_search_kind(
    candidate: HostModelCandidate,
    *,
    scope: WebAccessScope,
    has_client_tools: bool,
) -> WebSearchToolKind | None:
    if not web_access_enabled(scope):
        return None
    if candidate_supports_native_web_search(
        candidate,
        has_client_tools=has_client_tools,
    ):
        return "native"
    if scope != "chat":
        return None
    from .chat_web_search import chat_web_search_configured

    return "client" if chat_web_search_configured() else None


def tools_for_web_candidate(
    tools: dict[str, Any] | None,
    *,
    candidate: HostModelCandidate,
    scope: WebAccessScope,
) -> dict[str, Any] | None:
    result = dict(tools or {})
    has_client_tools = any(
        getattr(tool, "execution_side", "client") != "server"
        for tool in result.values()
    )
    search_kind = candidate_web_search_kind(
        candidate,
        scope=scope,
        has_client_tools=has_client_tools,
    )
    if "web_search" not in result and search_kind == "native":
        from zhenxun.services.ai.tools.providers.builtin.native import Native

        result["web_search"] = Native.web_search(
            name="web_search",
            description="搜索公开网页以获取最新或外部信息。",
        )
    elif "web_search" not in result and search_kind == "client":
        from .chat_web_search import build_chat_web_search_tool

        search_tool = build_chat_web_search_tool()
        if search_tool is not None:
            result["web_search"] = search_tool
    return result or None


def native_web_search_exposed(tools: dict[str, Any] | None) -> bool:
    return any(
        getattr(tool, "execution_side", "client") == "server"
        and getattr(tool, "type_id", "") == "web_search"
        for tool in (tools or {}).values()
    )


def client_web_search_exposed(tools: dict[str, Any] | None) -> bool:
    return any(
        getattr(tool, "chatinter_tool_kind", "") == "chat_web_search"
        for tool in (tools or {}).values()
    )


def project_client_web_search_result(
    tool: Any,
    result: Any,
) -> WebResponseProjection:
    if getattr(tool, "chatinter_tool_kind", "") != "chat_web_search":
        return WebResponseProjection(citations=())
    output = getattr(result, "output", None)
    if not isinstance(output, dict):
        return WebResponseProjection(citations=(), search_used=True)
    citations: list[WebCitation] = []
    for item in output.get("citations", ()):
        if not isinstance(item, dict):
            continue
        _append_citation(
            citations,
            title=item.get("title"),
            url=item.get("url"),
        )
    return WebResponseProjection(
        citations=tuple(citations[:_MAX_CITATIONS]),
        search_used=True,
    )


def project_web_response(response: Any) -> WebResponseProjection:
    citations: list[WebCitation] = []
    search_used = False
    raw = getattr(response, "raw_response", None)
    if isinstance(raw, dict):
        for item in raw.get("output", ()):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "web_search_call":
                search_used = True
            if item.get("type") != "message":
                continue
            for content in item.get("content", ()):
                if not isinstance(content, dict):
                    continue
                for annotation in content.get("annotations", ()):
                    if not isinstance(annotation, dict):
                        continue
                    if annotation.get("type") != "url_citation":
                        continue
                    _append_citation(
                        citations,
                        title=annotation.get("title"),
                        url=annotation.get("url"),
                    )

    grounding = getattr(response, "grounding_metadata", None)
    if grounding is not None:
        search_used = search_used or bool(
            getattr(grounding, "web_search_queries", None)
            or _mapping_value(grounding, "web_search_queries")
        )
        attributions = getattr(grounding, "grounding_attributions", None)
        if attributions is None:
            attributions = _mapping_value(grounding, "grounding_attributions")
        for attribution in attributions or ():
            _append_citation(
                citations,
                title=getattr(attribution, "title", None)
                or _mapping_value(attribution, "title"),
                url=getattr(attribution, "uri", None)
                or _mapping_value(attribution, "uri"),
            )
        search_used = search_used or bool(citations)

    return WebResponseProjection(
        citations=tuple(citations[:_MAX_CITATIONS]),
        search_used=search_used,
    )


def append_web_citations(text: str, citations: tuple[WebCitation, ...]) -> str:
    value = str(text or "").strip()
    missing = [citation for citation in citations if citation.url not in value]
    if not missing:
        return value
    sources = ["来源："]
    sources.extend(
        f"{index}. {citation.title} {citation.url}"
        for index, citation in enumerate(missing, start=1)
    )
    block = "\n".join(sources)
    return f"{value}\n\n{block}" if value else block


def _append_citation(
    citations: list[WebCitation],
    *,
    title: Any,
    url: Any,
) -> None:
    safe_url = sanitize_web_citation_url(str(url or ""))
    if not safe_url or any(item.url == safe_url for item in citations):
        return
    safe_title = normalize_message_text(str(title or ""))[:_MAX_CITATION_TITLE_CHARS]
    citations.append(WebCitation(title=safe_title or "网页来源", url=safe_url))


def sanitize_web_citation_url(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError:
        return ""
    if parsed.scheme.casefold() not in {"http", "https"}:
        return ""
    if not parsed.hostname or parsed.username or parsed.password:
        return ""
    query = urlencode(
        [
            (key, item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if re.sub(r"[^a-z0-9]", "", key.casefold()) not in _SENSITIVE_QUERY_KEYS
        ],
        doseq=True,
    )
    host = parsed.hostname.casefold()
    if ":" in host:
        host = f"[{host}]"
    if port is not None:
        host = f"{host}:{port}"
    return urlunsplit((parsed.scheme.casefold(), host, parsed.path, query, ""))


def _mapping_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


__all__ = [
    "WebCitation",
    "WebResponseProjection",
    "append_web_citations",
    "candidate_supports_native_web_search",
    "candidate_web_search_kind",
    "client_web_search_exposed",
    "native_web_search_exposed",
    "project_client_web_search_result",
    "project_web_response",
    "sanitize_web_citation_url",
    "tools_for_web_candidate",
    "web_access_enabled",
]
