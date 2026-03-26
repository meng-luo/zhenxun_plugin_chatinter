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

_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+")
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
_SESSION_PREF_TTL = 2 * 60 * 60
_SESSION_PREF_KEEP = 48
_SESSION_PREF_PRUNE = 24
_SESSION_PREF_MIN_SCORE = 0.04


@dataclass
class _IndexedDoc:
    plugin: PluginInfo
    signature: str
    vector: list[float]
    vector_type: str
    token_weights: dict[str, float]


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _TOKEN_PATTERN.findall(text or ""):
        lower = token.lower()
        tokens.append(lower)
        if any("\u4e00" <= char <= "\u9fff" for char in lower):
            tokens.extend(char for char in lower if "\u4e00" <= char <= "\u9fff")
            if len(lower) >= 2:
                tokens.extend(lower[i : i + 2] for i in range(len(lower) - 1))
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


class PluginRAGService:
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _loaded: ClassVar[bool] = False
    _docs: ClassVar[dict[str, _IndexedDoc]] = {}
    _persisted: ClassVar[dict[str, dict]] = {}
    _module_graph: ClassVar[dict[str, dict[str, float]]] = {}
    _session_preference: ClassVar[dict[str, dict[str, float]]] = {}
    _session_preference_time: ClassVar[dict[str, float]] = {}
    _embedding_disabled_until: ClassVar[float] = 0.0
    _embedding_supported: ClassVar[bool | None] = None

    @classmethod
    def _prune_session_preference(cls, now: float) -> None:
        expired = [
            session_id
            for session_id, updated_at in cls._session_preference_time.items()
            if now - updated_at >= _SESSION_PREF_TTL
        ]
        for session_id in expired:
            cls._session_preference.pop(session_id, None)
            cls._session_preference_time.pop(session_id, None)

    @classmethod
    async def update_session_feedback(
        cls,
        session_id: str | None,
        modules: set[str] | list[str],
        reward: float = 1.0,
    ) -> None:
        if not session_id:
            return
        normalized_modules = [module for module in modules if module]
        if not normalized_modules:
            return
        now = time.monotonic()
        async with cls._lock:
            cls._prune_session_preference(now)
            pref = cls._session_preference.get(session_id, {})
            if pref:
                for module in list(pref):
                    pref[module] *= 0.94
                    if pref[module] < _SESSION_PREF_MIN_SCORE:
                        pref.pop(module, None)
            for module in normalized_modules:
                pref[module] = pref.get(module, 0.0) + reward
            if len(pref) > _SESSION_PREF_KEEP:
                ranked = sorted(pref.items(), key=lambda item: item[1], reverse=True)[
                    :_SESSION_PREF_PRUNE
                ]
                pref = dict(ranked)
            cls._session_preference[session_id] = pref
            cls._session_preference_time[session_id] = now

    @classmethod
    def _session_pref_scores(cls, session_id: str | None) -> dict[str, float]:
        if not session_id:
            return {}
        now = time.monotonic()
        updated_at = cls._session_preference_time.get(session_id)
        if not updated_at:
            return {}
        if now - updated_at >= _SESSION_PREF_TTL:
            cls._session_preference.pop(session_id, None)
            cls._session_preference_time.pop(session_id, None)
            return {}
        pref = cls._session_preference.get(session_id, {})
        if not pref:
            return {}
        max_score = max(pref.values()) or 1.0
        return {module: score / max_score for module, score in pref.items()}

    @classmethod
    async def _load_persisted_index(cls) -> None:
        if cls._loaded:
            return

        def _read_file() -> dict[str, dict]:
            if not _INDEX_PATH.exists():
                return {}
            try:
                raw = _INDEX_PATH.read_text(encoding="utf-8")
                payload = json.loads(raw)
            except Exception:
                return {}
            if not isinstance(payload, dict):
                return {}
            docs = payload.get("docs", {})
            return docs if isinstance(docs, dict) else {}

        cls._persisted = await asyncio.to_thread(_read_file)
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
            }
        payload = {"version": 2, "docs": payload_docs}

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
                    cls._docs[module] = _IndexedDoc(
                        plugin=plugin,
                        signature=signature,
                        vector=[float(v) for v in vector],
                        vector_type=str(persisted.get("vector_type", "embedding")),
                        token_weights=token_weights,
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
                )
            changed = True

        if changed:
            await cls._save_persisted_index()
            cls._persisted = {
                module: {
                    "signature": doc.signature,
                    "vector": doc.vector,
                    "vector_type": doc.vector_type,
                    "token_weights": doc.token_weights,
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
    async def retrieve(
        cls,
        query: str,
        knowledge: PluginKnowledgeBase,
        top_k: int = _DEFAULT_TOP_K,
        context_text: str = "",
        session_id: str | None = None,
    ) -> list[PluginInfo]:
        if not query.strip():
            return []

        async with cls._lock:
            await cls._sync_index(knowledge)
            if not cls._docs:
                return []
            query_vector = await cls._embed_query(query)
            if not query_vector:
                return []

            token_weights = _build_token_weights(f"{query} {context_text}".strip())
            vector_scores: dict[str, float] = {}
            lexical_scores: dict[str, float] = {}

            for module, doc in cls._docs.items():
                vector_scores[module] = _cosine_score(query_vector, doc.vector)
                lexical_scores[module] = sum(
                    token_weights.get(token, 0.0) * doc.token_weights.get(token, 0.0)
                    for token in token_weights
                )

            base_rank = sorted(
                (
                    (
                        module,
                        vector_scores[module] * 0.75 + lexical_scores[module] * 0.25,
                    )
                    for module in cls._docs
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

            ranked = sorted(
                (
                    (
                        doc.plugin,
                        _VECTOR_WEIGHT * vector_scores.get(module, 0.0)
                        + _LEXICAL_WEIGHT * lexical_scores.get(module, 0.0)
                        + _GRAPH_WEIGHT * graph_scores.get(module, 0.0)
                        + _SESSION_PREF_WEIGHT * session_scores.get(module, 0.0),
                    )
                    for module, doc in cls._docs.items()
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            selected = [plugin for plugin, score in ranked if score > 0][
                : max(top_k, 1)
            ]
            return selected

    @classmethod
    async def build_context(
        cls,
        query: str,
        knowledge: PluginKnowledgeBase,
        top_k: int = _DEFAULT_TOP_K,
        context_text: str = "",
        session_id: str | None = None,
    ) -> str:
        selected = await cls.retrieve(
            query,
            knowledge,
            top_k=top_k,
            context_text=context_text,
            session_id=session_id,
        )
        if not selected:
            return ""

        lines = []
        for index, plugin in enumerate(selected, start=1):
            commands = ", ".join(plugin.commands[:4]) if plugin.commands else "无命令"
            desc = (plugin.description or "").strip()
            usage = (plugin.usage or "").strip()
            snippet = usage or desc
            snippet = snippet.replace("\n", " ").strip()
            if len(snippet) > _MAX_SNIPPET_LEN:
                snippet = f"{snippet[:_MAX_SNIPPET_LEN]}..."
            lines.append(
                f"{index}. [{plugin.module}] {plugin.name}\n"
                f"   commands: {commands}\n"
                f"   note: {snippet}"
            )
        return "\n".join(lines)
