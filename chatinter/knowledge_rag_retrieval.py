import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import ClassVar

from zhenxun.services.llm import embed_documents, embed_query, list_embedding_models
from zhenxun.services.log import logger

from .models.pydantic_models import PluginInfo, PluginKnowledgeBase
from .plugin_adapters import get_adapter_target_policy
from .route_text import contains_any, normalize_message_text
from .schema_policy import resolve_command_target_policy

_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+")
_RAG_STOPWORDS = {
    "我",
    "你",
    "他",
    "她",
    "它",
    "我们",
    "你们",
    "他们",
    "她们",
    "这",
    "那",
    "这个",
    "那个",
    "最近",
    "现在",
    "一下",
    "一下下",
    "帮我",
    "请",
    "麻烦",
    "什么",
    "怎么",
    "如何",
    "怎样",
    "可以",
    "能不能",
    "群里",
    "群",
    "消息",
    "聊天",
    "说什么",
    "说了什么",
}
_DEFAULT_TOP_K = 5
_MAX_SNIPPET_LEN = 220
_FALLBACK_DIM = 384
_INDEX_PATH = Path("data/cache/chatinter/plugin_vector_index.json")
_EMBEDDING_COOLDOWN = 120
_GRAPH_TOKEN_MIN_LEN = 2
_VECTOR_WEIGHT = 0.62
_LEXICAL_WEIGHT = 0.20
_GRAPH_WEIGHT = 0.08
_SESSION_PREF_WEIGHT = 0.10
_SESSION_REASON_WEIGHT = 0.08
_SESSION_SLOT_WEIGHT = 0.10
_PREFERRED_MODULE_WEIGHT = 0.08
_QUERY_CACHE_TTL = 90.0
_QUERY_CACHE_MAX_SIZE = 256
_DEFAULT_FETCH_K = 24
_MAX_FETCH_K = 96
_DEFAULT_MIN_SCORE = 0.02
_DEFAULT_MAX_K = 24
_DEFAULT_K_INCREMENT = 4


@dataclass(frozen=True)
class _RetrieveOptions:
    top_k: int
    fetch_k: int
    min_score: float
    max_k: int
    k_increment: int
    rerank: bool
    metadata_filters: dict[str, bool]
    preferred_modules: tuple[str, ...]


@dataclass
class _IndexedDoc:
    plugin: PluginInfo
    signature: str
    vector: list[float]
    vector_type: str
    token_weights: dict[str, float]
    metadata: dict[str, bool]


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _TOKEN_PATTERN.findall(text or ""):
        lower = token.lower()
        if lower in _RAG_STOPWORDS:
            continue
        tokens.append(lower)
        if any("\u4e00" <= char <= "\u9fff" for char in lower):
            chars = [char for char in lower if "\u4e00" <= char <= "\u9fff"]
            if len(lower) <= 3:
                tokens.extend(char for char in chars if char not in _RAG_STOPWORDS)
            if len(lower) >= 2:
                tokens.extend(
                    ngram
                    for ngram in (lower[i : i + 2] for i in range(len(lower) - 1))
                    if ngram not in _RAG_STOPWORDS
                )
            if len(lower) >= 3:
                tokens.extend(
                    ngram
                    for ngram in (lower[i : i + 3] for i in range(len(lower) - 2))
                    if ngram not in _RAG_STOPWORDS
                )
    return tokens


def _doc_text(plugin: PluginInfo) -> str:
    commands = " ".join(plugin.commands or [])
    usage = plugin.usage or ""
    return f"{plugin.name} {plugin.module} {plugin.description} {commands} {usage}"


def _doc_signature(plugin: PluginInfo, text: str) -> str:
    source = f"{plugin.module}|{text}"
    return hashlib.md5(source.encode("utf-8")).hexdigest()


def _build_token_weights(text: str) -> dict[str, float]:
    counter = Counter(_tokenize(text))
    if not counter:
        return {}
    total = float(sum(counter.values()))
    return {token: value / total for token, value in counter.items()}


def _build_doc_metadata(plugin: PluginInfo) -> dict[str, bool]:
    commands = plugin.commands or []
    command_text = " ".join(commands)
    helper_keywords = ("帮助", "详情", "搜索", "说明", "教程", "参数", "列表", "用法")
    has_helper = any(keyword in command_text for keyword in helper_keywords)

    template_keywords = ("表情", "梗图", "meme", "模板", "头像", "图片")
    template_text = (
        f"{plugin.name} {plugin.module} "
        f"{plugin.description} {plugin.usage or ''} {command_text}"
    ).lower()
    is_template_like = any(keyword in template_text for keyword in template_keywords)

    target_capable = False
    image_capable = False
    self_only = False
    adapter_policy = get_adapter_target_policy(
        plugin_module=plugin.module,
        plugin_name=plugin.name,
    )
    for meta in plugin.command_meta:
        policy = resolve_command_target_policy(
            meta,
            adapter_policy=adapter_policy,
        )
        if policy.allow_at or bool(policy.target_sources & {"at", "reply", "nickname"}):
            target_capable = True
        image_min = getattr(meta, "image_min", None)
        image_max = getattr(meta, "image_max", None)
        if (image_min is not None and int(image_min) > 0) or (
            image_max is not None and int(image_max) > 0
        ):
            image_capable = True
        if policy.actor_scope == "self_only":
            self_only = True

    if not image_capable:
        usage_l = str(plugin.usage or "").lower()
        if "[image" in usage_l or "图片" in usage_l or "头像" in usage_l:
            image_capable = True

    return {
        "has_helper": has_helper,
        "is_template_like": is_template_like,
        "target_capable": target_capable,
        "image_capable": image_capable,
        "self_only": self_only,
    }


def _parse_retrieve_options(
    *,
    top_k: int,
    fetch_k: int | None,
    min_score: float | None,
    max_k: int | None,
    k_increment: int,
    rerank: bool,
    metadata_filters: dict[str, bool] | None,
    preferred_modules: list[str] | tuple[str, ...] | None,
) -> _RetrieveOptions:
    resolved_top_k = max(int(top_k or 1), 1)
    resolved_fetch_k = (
        fetch_k if fetch_k is not None else max(resolved_top_k * 4, _DEFAULT_FETCH_K)
    )
    resolved_fetch_k = max(resolved_top_k, min(int(resolved_fetch_k), _MAX_FETCH_K))
    resolved_min_score = (
        _DEFAULT_MIN_SCORE if min_score is None else max(float(min_score), 0.0)
    )
    resolved_max_k = max_k if max_k is not None else max(resolved_top_k, _DEFAULT_MAX_K)
    resolved_max_k = max(resolved_top_k, min(int(resolved_max_k), resolved_fetch_k))
    resolved_k_increment = max(int(k_increment or _DEFAULT_K_INCREMENT), 1)
    resolved_filters: dict[str, bool] = {}
    for key, value in (metadata_filters or {}).items():
        text_key = str(key or "").strip()
        if not text_key:
            continue
        resolved_filters[text_key] = bool(value)
    normalized_preferred: list[str] = []
    for module in preferred_modules or ():
        module_text = normalize_message_text(str(module or ""))
        if module_text and module_text not in normalized_preferred:
            normalized_preferred.append(module_text)
    return _RetrieveOptions(
        top_k=resolved_top_k,
        fetch_k=resolved_fetch_k,
        min_score=resolved_min_score,
        max_k=resolved_max_k,
        k_increment=resolved_k_increment,
        rerank=bool(rerank),
        metadata_filters=resolved_filters,
        preferred_modules=tuple(normalized_preferred),
    )


def _matches_metadata_filters(doc: _IndexedDoc, filters: dict[str, bool]) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if doc.metadata.get(key) is not expected:
            return False
    return True


def _command_relevance_score(plugin: PluginInfo, query_text: str) -> float:
    normalized_query = normalize_message_text(query_text or "").lower()
    if not normalized_query:
        return 0.0
    best = 0.0
    has_template_context = contains_any(
        normalized_query,
        ("表情", "梗图", "meme", "头像", "图片", "做", "来", "生成", "制作"),
    )
    command_candidates: list[str] = []
    command_candidates.extend(plugin.commands or [])
    for meta in plugin.command_meta:
        command_candidates.append(getattr(meta, "command", ""))
        command_candidates.extend(getattr(meta, "aliases", None) or [])
    for raw_command in command_candidates:
        command = normalize_message_text(str(raw_command or "")).lower()
        if not command:
            continue
        if normalized_query.startswith(command):
            best = max(best, 0.20)
            continue
        if len(command) >= 2 and command in normalized_query:
            best = max(best, 0.14)
            continue
        if len(command) == 1 and has_template_context and command in normalized_query:
            best = max(best, 0.10)
    return best


def _normalize_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-12:
        return values
    return [v / norm for v in values]


def _cosine_score(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b, strict=False))


def _fallback_vector(text: str, dim: int = _FALLBACK_DIM) -> list[float]:
    vector = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], byteorder="big", signed=False) % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    return _normalize_vector(vector)


class PluginRAGRetrievalMixin:
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _loaded: ClassVar[bool] = False
    _docs: ClassVar[dict[str, _IndexedDoc]] = {}
    _persisted: ClassVar[dict[str, dict]] = {}
    _module_graph: ClassVar[dict[str, dict[str, float]]] = {}
    _embedding_disabled_until: ClassVar[float] = 0.0
    _embedding_supported: ClassVar[bool | None] = None
    _index_meta: ClassVar[dict[str, str]] = {}
    _cache_version: ClassVar[int] = 0
    _query_cache: ClassVar[dict[str, tuple[float, list[str]]]] = {}

    @classmethod
    def _session_pref_scores(cls, session_id: str | None) -> dict[str, float]:
        return {}

    @classmethod
    def _session_reason_penalty_scores(
        cls,
        session_id: str | None,
    ) -> dict[str, float]:
        return {}

    @classmethod
    def _session_slot_scores(
        cls,
        session_id: str | None,
        query: str,
        context_text: str = "",
    ) -> dict[str, float]:
        return {}

    @classmethod
    def _query_cache_key(
        cls,
        *,
        query: str,
        context_text: str,
        session_id: str | None,
        options: _RetrieveOptions,
    ) -> str:
        filter_part = ",".join(
            f"{key}:{int(value)}"
            for key, value in sorted(options.metadata_filters.items())
        )
        preferred_part = ",".join(options.preferred_modules)
        raw = "|".join(
            (
                str(cls._cache_version),
                normalize_message_text(query),
                normalize_message_text(context_text),
                str(session_id or ""),
                str(options.top_k),
                str(options.fetch_k),
                str(options.min_score),
                str(options.max_k),
                str(options.k_increment),
                str(int(options.rerank)),
                filter_part,
                preferred_part,
            )
        )
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @classmethod
    def _get_cached_modules(cls, cache_key: str) -> list[str] | None:
        cached = cls._query_cache.get(cache_key)
        if not cached:
            return None
        ts, modules = cached
        if (time.monotonic() - ts) > _QUERY_CACHE_TTL:
            cls._query_cache.pop(cache_key, None)
            return None
        return modules[:]

    @classmethod
    def _set_cached_modules(cls, cache_key: str, modules: list[str]) -> None:
        cls._query_cache[cache_key] = (time.monotonic(), modules[:])
        if len(cls._query_cache) <= _QUERY_CACHE_MAX_SIZE:
            return
        stale_keys = sorted(cls._query_cache.items(), key=lambda item: item[1][0])[
            : max(8, _QUERY_CACHE_MAX_SIZE // 8)
        ]
        for key, _ in stale_keys:
            cls._query_cache.pop(key, None)

    @classmethod
    def _clear_query_cache(cls) -> None:
        cls._query_cache.clear()

    @classmethod
    async def _load_persisted_index(cls) -> None:
        if cls._loaded:
            return

        def _read_file() -> tuple[dict[str, dict], dict[str, str]]:
            if not _INDEX_PATH.exists():
                return {}, {}
            try:
                raw = _INDEX_PATH.read_text(encoding="utf-8")
                payload = json.loads(raw)
            except Exception:
                return {}, {}
            if not isinstance(payload, dict):
                return {}, {}
            docs = payload.get("docs", {})
            meta = payload.get("meta", {})
            docs_dict = docs if isinstance(docs, dict) else {}
            meta_dict = (
                {str(k): str(v) for k, v in meta.items()}
                if isinstance(meta, dict)
                else {}
            )
            return docs_dict, meta_dict

        cls._persisted, cls._index_meta = await asyncio.to_thread(_read_file)
        try:
            cls._cache_version = int(cls._index_meta.get("cache_version", "0") or 0)
        except Exception:
            cls._cache_version = 0
        cls._loaded = True

    @classmethod
    async def _save_persisted_index(cls) -> None:
        payload_docs = {}
        for module, doc in cls._docs.items():
            payload_docs[module] = {
                "signature": doc.signature,
                "vector": doc.vector,
                "vector_type": doc.vector_type,
                "token_weights": doc.token_weights,
                "metadata": doc.metadata,
            }
        payload = {
            "version": 4,
            "meta": cls._index_meta,
            "docs": payload_docs,
        }

        def _write_file() -> None:
            _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            _INDEX_PATH.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )

        await asyncio.to_thread(_write_file)

    @classmethod
    async def _embed_documents(cls, texts: list[str]) -> tuple[list[list[float]], str]:
        if not texts:
            return [], "fallback"
        if cls._embedding_supported is None:
            cls._embedding_supported = bool(list_embedding_models())
        if not cls._embedding_supported:
            return [_fallback_vector(text) for text in texts], "fallback"
        now = time.monotonic()
        if now < cls._embedding_disabled_until:
            return [_fallback_vector(text) for text in texts], "fallback"

        try:
            vectors = await embed_documents(texts)
            if not vectors or len(vectors) != len(texts):
                raise RuntimeError("embedding result invalid")
            normalized = [
                _normalize_vector([float(v) for v in vector]) for vector in vectors
            ]
            return normalized, "embedding"
        except Exception as exc:
            cls._embedding_disabled_until = time.monotonic() + _EMBEDDING_COOLDOWN
            logger.debug(f"chatinter rag embedding disabled temporarily: {exc}")
            return [_fallback_vector(text) for text in texts], "fallback"

    @classmethod
    async def _embed_query(cls, text: str) -> list[float]:
        if not text.strip():
            return []
        if cls._embedding_supported is None:
            cls._embedding_supported = bool(list_embedding_models())
        if not cls._embedding_supported:
            return _fallback_vector(text)
        now = time.monotonic()
        if now < cls._embedding_disabled_until:
            return _fallback_vector(text)

        try:
            vector = await embed_query(text)
            return _normalize_vector([float(v) for v in vector])
        except Exception:
            cls._embedding_disabled_until = time.monotonic() + _EMBEDDING_COOLDOWN
            return _fallback_vector(text)

    @classmethod
    async def _sync_index(cls, knowledge: PluginKnowledgeBase) -> None:
        await cls._load_persisted_index()

        changed = False
        current_modules = {plugin.module for plugin in knowledge.plugins}

        for module in list(cls._docs):
            if module not in current_modules:
                cls._docs.pop(module, None)
                changed = True

        pending_plugins: list[PluginInfo] = []
        pending_texts: list[str] = []
        pending_signatures: list[str] = []

        for plugin in knowledge.plugins:
            module = plugin.module
            text = _doc_text(plugin)
            signature = _doc_signature(plugin, text)

            existing = cls._docs.get(module)
            if existing and existing.signature == signature:
                existing.plugin = plugin
                if not existing.token_weights:
                    existing.token_weights = _build_token_weights(text)
                if not existing.metadata:
                    existing.metadata = _build_doc_metadata(plugin)
                continue

            persisted = cls._persisted.get(module)
            if isinstance(persisted, dict) and persisted.get("signature") == signature:
                vector = persisted.get("vector")
                if isinstance(vector, list) and vector:
                    token_weights_raw = persisted.get("token_weights")
                    token_weights = (
                        {
                            str(key): float(value)
                            for key, value in token_weights_raw.items()
                        }
                        if isinstance(token_weights_raw, dict)
                        else _build_token_weights(text)
                    )
                    metadata_raw = persisted.get("metadata")
                    metadata = (
                        {str(key): bool(value) for key, value in metadata_raw.items()}
                        if isinstance(metadata_raw, dict)
                        else _build_doc_metadata(plugin)
                    )
                    cls._docs[module] = _IndexedDoc(
                        plugin=plugin,
                        signature=signature,
                        vector=[float(v) for v in vector],
                        vector_type=str(persisted.get("vector_type", "embedding")),
                        token_weights=token_weights,
                        metadata=metadata,
                    )
                    continue

            pending_plugins.append(plugin)
            pending_texts.append(text)
            pending_signatures.append(signature)

        if pending_plugins:
            vectors, vector_type = await cls._embed_documents(pending_texts)
            for plugin, signature, vector in zip(
                pending_plugins, pending_signatures, vectors, strict=False
            ):
                cls._docs[plugin.module] = _IndexedDoc(
                    plugin=plugin,
                    signature=signature,
                    vector=vector,
                    vector_type=vector_type,
                    token_weights=_build_token_weights(_doc_text(plugin)),
                    metadata=_build_doc_metadata(plugin),
                )
            changed = True

        if changed:
            cls._cache_version += 1
            cls._clear_query_cache()
            cls._index_meta = {
                "cache_version": str(cls._cache_version),
                "updated_at": str(int(time.time())),
                "vector_count": str(len(cls._docs)),
            }
            await cls._save_persisted_index()
            cls._persisted = {
                module: {
                    "signature": doc.signature,
                    "vector": doc.vector,
                    "vector_type": doc.vector_type,
                    "token_weights": doc.token_weights,
                    "metadata": doc.metadata,
                }
                for module, doc in cls._docs.items()
            }
        cls._rebuild_module_graph()

    @classmethod
    def _rebuild_module_graph(cls) -> None:
        token_to_modules: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for module, doc in cls._docs.items():
            for token, weight in doc.token_weights.items():
                if len(token) < _GRAPH_TOKEN_MIN_LEN:
                    continue
                token_to_modules[token].append((module, weight))

        module_graph: dict[str, dict[str, float]] = defaultdict(dict)
        for nodes in token_to_modules.values():
            if len(nodes) < 2:
                continue
            nodes_sorted = sorted(nodes, key=lambda item: item[1], reverse=True)[:24]
            for idx, (left_module, left_weight) in enumerate(nodes_sorted):
                for right_module, right_weight in nodes_sorted[idx + 1 :]:
                    edge = min(left_weight, right_weight)
                    if edge <= 0:
                        continue
                    module_graph[left_module][right_module] = (
                        module_graph[left_module].get(right_module, 0.0) + edge
                    )
                    module_graph[right_module][left_module] = (
                        module_graph[right_module].get(left_module, 0.0) + edge
                    )

        normalized: dict[str, dict[str, float]] = {}
        for module, links in module_graph.items():
            if not links:
                continue
            max_score = max(links.values()) or 1.0
            normalized[module] = {
                target: score / max_score for target, score in links.items()
            }
        cls._module_graph = normalized

    @classmethod
    def _preferred_module_scores(
        cls,
        preferred_modules: tuple[str, ...],
    ) -> dict[str, float]:
        if not preferred_modules:
            return {}
        total = len(preferred_modules)
        if total <= 0:
            return {}
        result: dict[str, float] = {}
        for idx, module in enumerate(preferred_modules):
            if not module:
                continue
            result[module] = max(0.0, 1.0 - (idx / max(total, 1)))
        return result

    @classmethod
    def _select_by_threshold(
        cls,
        ranked_modules: list[tuple[str, float]],
        *,
        options: _RetrieveOptions,
    ) -> list[str]:
        if not ranked_modules:
            return []

        fetch_size = min(options.fetch_k, len(ranked_modules))
        fetched = ranked_modules[:fetch_size]
        if not fetched:
            return []

        probe_k = min(options.top_k, len(fetched))
        max_probe = min(options.max_k, len(fetched))

        while True:
            probe = fetched[:probe_k]
            filtered = [module for module, score in probe if score >= options.min_score]
            if len(filtered) >= options.top_k or probe_k >= max_probe:
                if filtered:
                    return filtered[: options.top_k]
                return [module for module, _ in probe[: options.top_k]]
            probe_k = min(probe_k + options.k_increment, max_probe)

    @classmethod
    async def retrieve(
        cls,
        query: str,
        knowledge: PluginKnowledgeBase,
        top_k: int = _DEFAULT_TOP_K,
        context_text: str = "",
        session_id: str | None = None,
        *,
        fetch_k: int | None = None,
        min_score: float | None = None,
        max_k: int | None = None,
        k_increment: int = _DEFAULT_K_INCREMENT,
        metadata_filters: dict[str, bool] | None = None,
        preferred_modules: list[str] | tuple[str, ...] | None = None,
        rerank: bool = True,
    ) -> list[PluginInfo]:
        if not query.strip():
            return []
        options = _parse_retrieve_options(
            top_k=top_k,
            fetch_k=fetch_k,
            min_score=min_score,
            max_k=max_k,
            k_increment=k_increment,
            rerank=rerank,
            metadata_filters=metadata_filters,
            preferred_modules=preferred_modules,
        )

        async with cls._lock:
            await cls._sync_index(knowledge)
            if not cls._docs:
                return []
            cache_key = cls._query_cache_key(
                query=query,
                context_text=context_text,
                session_id=session_id,
                options=options,
            )
            cached_modules = cls._get_cached_modules(cache_key)
            if cached_modules:
                selected_plugins: list[PluginInfo] = []
                for module in cached_modules:
                    doc = cls._docs.get(module)
                    if doc is None:
                        continue
                    if not _matches_metadata_filters(doc, options.metadata_filters):
                        continue
                    selected_plugins.append(doc.plugin)
                if selected_plugins:
                    return selected_plugins[: options.top_k]
            query_vector = await cls._embed_query(query)
            if not query_vector:
                return []

            token_weights = _build_token_weights(f"{query} {context_text}".strip())
            vector_scores: dict[str, float] = {}
            lexical_scores: dict[str, float] = {}

            for module, doc in cls._docs.items():
                if not _matches_metadata_filters(doc, options.metadata_filters):
                    continue
                vector_scores[module] = _cosine_score(query_vector, doc.vector)
                lexical_scores[module] = sum(
                    token_weights.get(token, 0.0) * doc.token_weights.get(token, 0.0)
                    for token in token_weights
                )
            if not vector_scores:
                return []

            base_rank = sorted(
                (
                    (
                        module,
                        vector_scores[module] * 0.75 + lexical_scores[module] * 0.25,
                    )
                    for module in vector_scores
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            seed_modules = [module for module, score in base_rank[:6] if score > 0]
            graph_scores: dict[str, float] = defaultdict(float)
            for seed in seed_modules:
                seed_score = max(vector_scores.get(seed, 0.0), 0.0)
                for target, edge in cls._module_graph.get(seed, {}).items():
                    graph_scores[target] += seed_score * edge

            if graph_scores:
                max_graph = max(graph_scores.values()) or 1.0
                for module in list(graph_scores):
                    graph_scores[module] = graph_scores[module] / max_graph
            session_scores = cls._session_pref_scores(session_id)
            reason_penalty_scores = cls._session_reason_penalty_scores(session_id)
            slot_scores = cls._session_slot_scores(session_id, query, context_text)
            preferred_scores = cls._preferred_module_scores(options.preferred_modules)
            lexical_weight = _LEXICAL_WEIGHT
            vector_weight = _VECTOR_WEIGHT
            if len(token_weights) <= 3:
                lexical_weight = min(_LEXICAL_WEIGHT + 0.08, 0.35)
                vector_weight = max(_VECTOR_WEIGHT - 0.08, 0.50)

            ranked_modules: list[tuple[str, float]] = []
            for module in vector_scores:
                command_score = _command_relevance_score(
                    cls._docs[module].plugin,
                    query,
                )
                base_score = (
                    vector_weight * vector_scores.get(module, 0.0)
                    + lexical_weight * lexical_scores.get(module, 0.0)
                    + command_score
                )
                if options.rerank:
                    base_score += (
                        _GRAPH_WEIGHT * graph_scores.get(module, 0.0)
                        + _SESSION_PREF_WEIGHT * session_scores.get(module, 0.0)
                        - _SESSION_REASON_WEIGHT
                        * reason_penalty_scores.get(module, 0.0)
                        + _SESSION_SLOT_WEIGHT * slot_scores.get(module, 0.0)
                        + _PREFERRED_MODULE_WEIGHT * preferred_scores.get(module, 0.0)
                    )
                ranked_modules.append((module, base_score))

            ranked_modules.sort(key=lambda item: item[1], reverse=True)
            selected_modules = cls._select_by_threshold(
                ranked_modules,
                options=options,
            )
            selected: list[PluginInfo] = []
            for module in selected_modules:
                doc = cls._docs.get(module)
                if doc is None:
                    continue
                selected.append(doc.plugin)
            if selected:
                cls._set_cached_modules(
                    cache_key,
                    [plugin.module for plugin in selected if plugin.module],
                )
            return selected
