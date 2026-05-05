from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
import re
import sys
from typing import Any, ClassVar, cast

from zhenxun.services.log import logger

from .route_text import normalize_message_text


class AutoMetadataBuilder:
    """自动从运行时插件构建命令元数据。

    链路：
    - matcher/parser 反射提取命令头与参数结构
    - matcher 源码 AST 提取 on_xxx(..., aliases={...}) 别名
    - shortcut(...) / parser.shortcuts / manager.shortcuts 统一提取快捷命令
    - parser dry-run 探针判断是否支持粘连参数
    - meme manager 反射提取模板类命令参数范围
    """

    _module_alias_cache: ClassVar[dict[str, tuple[int, dict[str, list[str]]]]] = {}
    _module_prefix_cache: ClassVar[dict[str, tuple[int, dict[str, list[str]]]]] = {}
    _module_context_cache: ClassVar[
        dict[str, tuple[int, dict[str, dict[str, bool]]]]
    ] = {}
    _module_access_cache: ClassVar[dict[str, tuple[int, dict[str, str]]]] = {}
    _handler_hint_cache: ClassVar[dict[str, tuple[int, dict[str, Any]]]] = {}
    _no_command_log_cache: ClassVar[set[str]] = set()
    _sticky_probe_token: ClassVar[str] = "测试"
    _command_placeholder_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"\s*(?:\[[^\]]+\]|<[^>]+>|\{[^}]+\})\s*"
    )
    _regex_head_pattern: ClassVar[re.Pattern[str]] = re.compile(r"[\[\(\.\*\+\?\|\$\\]")
    _image_type_hints: ClassVar[tuple[str, ...]] = (
        "image",
        "uniimg",
        "picture",
        "img",
        "bytesio",
    )
    _at_type_hints: ClassVar[tuple[str, ...]] = (
        "at",
        "target",
        "member",
        "qq",
    )

    @classmethod
    async def build(
        cls,
        *,
        module_name: str,
        module_obj: object | None,
        loaded_plugin: object | None,
    ) -> list[dict[str, Any]]:
        matcher_commands: list[dict[str, Any]] = []
        if loaded_plugin is not None:
            matcher_commands.extend(
                cls._extract_matcher_command_data(
                    loaded_plugin=loaded_plugin,
                )
            )
        manager_commands: list[dict[str, Any]] = []
        if module_obj is not None:
            manager_commands.extend(
                await cls._extract_manager_command_data(
                    module_name=module_name,
                    module_obj=module_obj,
                )
            )
        extracted = [*matcher_commands, *manager_commands]
        if not extracted:
            if module_name not in cls._no_command_log_cache:
                cls._no_command_log_cache.add(module_name)
                logger.debug(
                    f"ChatInter 自动元数据构建未从插件提取到命令: {module_name}"
                )
        return cls._merge_command_dicts(extracted)

    @classmethod
    def _extract_matcher_command_data(
        cls,
        *,
        loaded_plugin: object,
    ) -> list[dict[str, Any]]:
        alias_map = cls._build_module_alias_map(loaded_plugin)
        prefix_map = cls._build_module_prefix_map(loaded_plugin)
        result: list[dict[str, Any]] = []
        for matcher in cls._iter_plugin_matchers(loaded_plugin):
            module_obj = getattr(matcher, "module", None)
            access_map = (
                cls._load_module_access_map(module_obj)
                if module_obj is not None
                else {}
            )
            context_map = (
                cls._load_module_context_map(module_obj)
                if module_obj is not None
                else {}
            )
            parser = cls._get_matcher_parser(matcher)
            parser_schema = (
                cls._extract_parser_schema(parser)
                if parser is not None
                else cls._default_parser_schema()
            )
            handler_hint = cls._extract_handler_hint(matcher)
            context_hint = cls._extract_matcher_context_hint(matcher)
            parser_shortcut_aliases = (
                cls._extract_parser_shortcut_aliases(parser)
                if parser is not None
                else []
            )
            for payload in cls._extract_rule_command_data(
                matcher=matcher,
                parser_schema=parser_schema,
                handler_hint=handler_hint,
                context_hint=context_hint,
                source_context_map=context_map,
                access_map=access_map,
            ):
                command_head = str(payload.get("command") or "").strip()
                if not command_head:
                    continue
                payload["aliases"] = cls._merge_unique_strings(
                    payload.get("aliases"),
                    alias_map.get(command_head.casefold(), []),
                )
                payload["prefixes"] = cls._merge_unique_strings(
                    payload.get("prefixes"),
                    prefix_map.get(command_head.casefold(), []),
                )
                payload["requires_reply"] = bool(payload.get("requires_reply")) or bool(
                    context_map.get(command_head.casefold(), {}).get("requires_reply")
                )
                payload["requires_private"] = bool(
                    payload.get("requires_private")
                ) or bool(
                    context_map.get(command_head.casefold(), {}).get("requires_private")
                )
                payload["requires_to_me"] = bool(payload.get("requires_to_me")) or bool(
                    context_map.get(command_head.casefold(), {}).get("requires_to_me")
                )
                result.append(payload)

            if parser is None:
                continue
            command_head = cls._extract_parser_command_head(parser)
            if not command_head:
                continue
            access_level = cls._resolve_access_level(
                access_map.get(command_head.casefold()),
                handler_hint.get("requires_superuser"),
            )
            result.append(
                {
                    "command": command_head,
                    "aliases": cls._merge_unique_strings(
                        cls._extract_parser_aliases(parser, command_head),
                        alias_map.get(command_head.casefold(), []),
                        parser_shortcut_aliases,
                    ),
                    "prefixes": cls._merge_unique_strings(
                        parser_schema["prefixes"],
                        prefix_map.get(command_head.casefold(), []),
                    ),
                    "params": parser_schema["params"],
                    "text_min": parser_schema["text_min"],
                    "text_max": parser_schema["text_max"],
                    "image_min": parser_schema["image_min"],
                    "image_max": parser_schema["image_max"],
                    "allow_at": handler_hint["allow_at"]
                    if handler_hint["allow_at"] is not None
                    else parser_schema["allow_at"],
                    "target_sources": handler_hint["target_sources"]
                    or parser_schema["target_sources"],
                    "requires_reply": cls._merge_context_hint(
                        context_hint,
                        context_map.get(command_head.casefold(), {}),
                    )["requires_reply"]
                    or handler_hint["requires_reply"],
                    "requires_private": cls._merge_context_hint(
                        context_hint,
                        context_map.get(command_head.casefold(), {}),
                    )["requires_private"],
                    "requires_to_me": cls._merge_context_hint(
                        context_hint,
                        context_map.get(command_head.casefold(), {}),
                    )["requires_to_me"],
                    "allow_sticky_arg": cls._probe_sticky_arg(
                        parser=parser,
                        command_head=command_head,
                        sample_text=parser_schema["sample_text"],
                    ),
                    "access_level": access_level,
                }
            )
        return result

    @classmethod
    def _iter_plugin_matchers(cls, loaded_plugin: object) -> list[object]:
        plugins: list[object] = [loaded_plugin]
        seen_plugins: set[int] = set()
        seen_matchers: set[int] = set()
        matchers: list[object] = []
        while plugins:
            plugin_obj = plugins.pop()
            if plugin_obj is None or id(plugin_obj) in seen_plugins:
                continue
            seen_plugins.add(id(plugin_obj))
            for matcher in getattr(plugin_obj, "matcher", set()) or set():
                if id(matcher) in seen_matchers:
                    continue
                seen_matchers.add(id(matcher))
                matchers.append(matcher)
            for sub_plugin in getattr(plugin_obj, "sub_plugins", set()) or set():
                if sub_plugin is not None and id(sub_plugin) not in seen_plugins:
                    plugins.append(sub_plugin)
            module_name = str(getattr(plugin_obj, "module_name", "") or "").strip()
            module_obj = getattr(plugin_obj, "module", None)
            if module_obj is None or not module_name:
                continue
            for matcher in cls._iter_dynamic_module_matchers(
                module_name=module_name,
                module_obj=module_obj,
            ):
                if id(matcher) in seen_matchers:
                    continue
                seen_matchers.add(id(matcher))
                matchers.append(matcher)
        return matchers

    @classmethod
    def _iter_dynamic_module_matchers(
        cls,
        *,
        module_name: str,
        module_obj: object,
    ) -> list[object]:
        """补扫 startup 后动态挂到模块级 `matchers` 容器里的 matcher。"""
        result: list[object] = []
        seen_ids: set[int] = set()
        for candidate_module in cls._iter_related_modules(
            module_name=module_name,
            module_obj=module_obj,
        ):
            container = getattr(candidate_module, "matchers", None)
            if not isinstance(container, list | tuple | set | frozenset):
                continue
            for matcher in container:
                if id(matcher) in seen_ids or not cls._looks_like_matcher(matcher):
                    continue
                seen_ids.add(id(matcher))
                result.append(matcher)
        return result

    @staticmethod
    def _looks_like_matcher(matcher: object) -> bool:
        return hasattr(matcher, "rule") and hasattr(matcher, "handlers")

    @staticmethod
    def _default_parser_schema() -> dict[str, Any]:
        return {
            "params": [],
            "prefixes": [],
            "text_min": 0,
            "text_max": None,
            "image_min": 0,
            "image_max": None,
            "allow_at": None,
            "target_sources": [],
            "sample_text": AutoMetadataBuilder._sticky_probe_token,
        }

    @classmethod
    def _extract_rule_command_data(
        cls,
        *,
        matcher: object,
        parser_schema: dict[str, Any],
        handler_hint: dict[str, Any],
        context_hint: dict[str, bool],
        source_context_map: dict[str, dict[str, bool]],
        access_map: dict[str, str],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        rule = getattr(matcher, "rule", None)
        for checker in getattr(rule, "checkers", set()) or set():
            checker_call = getattr(checker, "call", None)
            if checker_call is None:
                continue
            checker_name = type(checker_call).__name__
            if checker_name in {"CommandRule", "ShellCommandRule"}:
                allow_sticky_arg = cls._extract_rule_allow_sticky_arg(checker_call)
                for command_head in cls._iter_command_rule_heads(checker_call):
                    command_key = cls._normalize_command(command_head).casefold()
                    merged_context_hint = cls._merge_context_hint(
                        context_hint,
                        source_context_map.get(command_key, {}),
                    )
                    access_level = cls._resolve_access_level(
                        access_map.get(command_key),
                        handler_hint.get("requires_superuser"),
                    )
                    result.append(
                        cls._build_rule_command_payload(
                            command_head=command_head,
                            parser_schema=parser_schema,
                            handler_hint=handler_hint,
                            context_hint=merged_context_hint,
                            allow_sticky_arg=allow_sticky_arg,
                            access_level=access_level,
                        )
                    )
                continue
            if checker_name == "StartswithRule":
                for command_head in getattr(checker_call, "msg", ()) or ():
                    command_key = cls._normalize_command(command_head).casefold()
                    merged_context_hint = cls._merge_context_hint(
                        context_hint,
                        source_context_map.get(command_key, {}),
                    )
                    access_level = cls._resolve_access_level(
                        access_map.get(command_key),
                        handler_hint.get("requires_superuser"),
                    )
                    result.append(
                        cls._build_rule_command_payload(
                            command_head=command_head,
                            parser_schema=parser_schema,
                            handler_hint=handler_hint,
                            context_hint=merged_context_hint,
                            allow_sticky_arg=True,
                            access_level=access_level,
                        )
                    )
                continue
            if checker_name == "FullmatchRule":
                for command_head in getattr(checker_call, "msg", ()) or ():
                    command_key = cls._normalize_command(command_head).casefold()
                    merged_context_hint = cls._merge_context_hint(
                        context_hint,
                        source_context_map.get(command_key, {}),
                    )
                    access_level = cls._resolve_access_level(
                        access_map.get(command_key),
                        handler_hint.get("requires_superuser"),
                    )
                    result.append(
                        cls._build_rule_command_payload(
                            command_head=command_head,
                            parser_schema=parser_schema,
                            handler_hint=handler_hint,
                            context_hint=merged_context_hint,
                            allow_sticky_arg=False,
                            access_level=access_level,
                        )
                    )
                continue
            if checker_name == "KeywordsRule":
                for command_head in getattr(checker_call, "keywords", ()) or ():
                    command_key = cls._normalize_command(command_head).casefold()
                    merged_context_hint = cls._merge_context_hint(
                        context_hint,
                        source_context_map.get(command_key, {}),
                    )
                    access_level = cls._resolve_access_level(
                        access_map.get(command_key),
                        handler_hint.get("requires_superuser"),
                    )
                    result.append(
                        cls._build_rule_command_payload(
                            command_head=command_head,
                            parser_schema=parser_schema,
                            handler_hint=handler_hint,
                            context_hint=merged_context_hint,
                            allow_sticky_arg=True,
                            access_level=access_level,
                        )
                    )
                continue
            if checker_name == "RegexRule":
                command_head = cls._extract_regex_head(
                    str(getattr(checker_call, "regex", "") or "")
                )
                if command_head:
                    command_key = cls._normalize_command(command_head).casefold()
                    merged_context_hint = cls._merge_context_hint(
                        context_hint,
                        source_context_map.get(command_key, {}),
                    )
                    access_level = cls._resolve_access_level(
                        access_map.get(command_key),
                        handler_hint.get("requires_superuser"),
                    )
                    result.append(
                        cls._build_rule_command_payload(
                            command_head=command_head,
                            parser_schema=parser_schema,
                            handler_hint=handler_hint,
                            context_hint=merged_context_hint,
                            allow_sticky_arg=True,
                            access_level=access_level,
                        )
                    )
        return result

    @classmethod
    def _build_rule_command_payload(
        cls,
        *,
        command_head: object,
        parser_schema: dict[str, Any],
        handler_hint: dict[str, Any],
        context_hint: dict[str, bool],
        allow_sticky_arg: bool | None,
        access_level: str = "public",
    ) -> dict[str, Any]:
        command = cls._normalize_command(str(command_head or ""))
        if not command:
            return {}
        return {
            "command": command,
            "prefixes": parser_schema["prefixes"],
            "params": parser_schema["params"],
            "text_min": parser_schema["text_min"],
            "text_max": parser_schema["text_max"],
            "image_min": parser_schema["image_min"],
            "image_max": parser_schema["image_max"],
            "allow_at": handler_hint["allow_at"]
            if handler_hint["allow_at"] is not None
            else parser_schema["allow_at"],
            "target_sources": handler_hint["target_sources"]
            or parser_schema["target_sources"],
            "requires_reply": handler_hint["requires_reply"]
            or context_hint["requires_reply"],
            "requires_private": context_hint["requires_private"],
            "requires_to_me": context_hint["requires_to_me"],
            "allow_sticky_arg": allow_sticky_arg
            if allow_sticky_arg is not None
            else True,
            "access_level": access_level,
        }

    @classmethod
    def _merge_context_hint(
        cls,
        base: dict[str, bool] | None,
        override: dict[str, bool] | None,
    ) -> dict[str, bool]:
        result = {
            "requires_reply": bool((base or {}).get("requires_reply"))
            or bool((override or {}).get("requires_reply")),
            "requires_to_me": bool((base or {}).get("requires_to_me"))
            or bool((override or {}).get("requires_to_me")),
            "requires_private": bool((base or {}).get("requires_private"))
            or bool((override or {}).get("requires_private")),
        }
        return result

    @classmethod
    def _iter_command_rule_heads(cls, checker_call: object) -> list[str]:
        heads: list[str] = []
        for command in getattr(checker_call, "cmds", ()) or ():
            if isinstance(command, str):
                command_parts = (command,)
            elif isinstance(command, list | tuple):
                command_parts = tuple(str(part or "").strip() for part in command)
            else:
                continue
            command_parts = tuple(part for part in command_parts if part)
            if not command_parts:
                continue
            heads.append(cls._normalize_command(".".join(command_parts)))
            heads.append(cls._normalize_command(" ".join(command_parts)))
        return cls._merge_unique_strings(heads, [])

    @staticmethod
    def _extract_rule_allow_sticky_arg(checker_call: object) -> bool | None:
        checker_name = type(checker_call).__name__
        if checker_name == "CommandRule":
            force_whitespace = getattr(checker_call, "force_whitespace", None)
            if force_whitespace is None or force_whitespace is False:
                return True
            return False
        if checker_name == "ShellCommandRule":
            return True
        return None

    @classmethod
    def _load_module_context_map(cls, module_obj: object) -> dict[str, dict[str, bool]]:
        source_file = inspect.getsourcefile(cast(Any, module_obj))
        if not source_file:
            return {}
        try:
            path = Path(source_file)
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            return {}

        cache_key = str(path)
        cached = cls._module_context_cache.get(cache_key)
        if cached is not None and cached[0] == mtime_ns:
            return cached[1]

        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        context_map: dict[str, dict[str, bool]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            command = cls._extract_command_from_call_node(node)
            if not command:
                continue
            context_hint = cls._extract_call_context_hint(node)
            if not any(context_hint.values()):
                continue
            command_key = command.casefold()
            context_map[command_key] = cls._merge_context_hint(
                context_map.get(command_key),
                context_hint,
            )
        cls._module_context_cache[cache_key] = (mtime_ns, context_map)
        return context_map

    @classmethod
    def _extract_call_context_hint(cls, node: ast.Call) -> dict[str, bool]:
        requires_reply = False
        requires_to_me = False
        requires_private = False
        for keyword in node.keywords or []:
            if keyword.arg != "rule":
                continue
            rule_text = cls._safe_unparse(keyword.value).lower()
            if not rule_text:
                continue
            if "reply" in rule_text:
                requires_reply = True
            if "to_me" in rule_text or "tome" in rule_text:
                requires_to_me = True
            if "ensure_private" in rule_text or "private" in rule_text:
                requires_private = True
        return {
            "requires_reply": requires_reply,
            "requires_to_me": requires_to_me,
            "requires_private": requires_private,
        }

    @staticmethod
    def _safe_unparse(node: ast.AST) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return ""

    @classmethod
    async def _extract_manager_command_data(
        cls,
        *,
        module_name: str,
        module_obj: object,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for manager in cls._iter_candidate_managers(
            module_name=module_name,
            module_obj=module_obj,
        ):
            result.extend(await cls._extract_meme_manager_command_data(manager))
        return result

    @staticmethod
    def _get_matcher_parser(matcher: object) -> object | None:
        command_builder = getattr(matcher, "command", None)
        if not callable(command_builder):
            return None
        try:
            return command_builder()
        except Exception:
            return None

    @classmethod
    def _extract_parser_command_head(cls, parser: object) -> str:
        command_head = cls._normalize_command(str(getattr(parser, "command", "") or ""))
        if not command_head:
            return ""
        if command_head.startswith("re:"):
            return cls._extract_regex_head(command_head[3:]) or ""
        return command_head

    @classmethod
    def _extract_parser_aliases(
        cls,
        parser: object,
        command_head: str,
    ) -> list[str]:
        aliases: list[str] = []
        raw_aliases = getattr(parser, "aliases", None)
        if isinstance(raw_aliases, list | tuple | set | frozenset):
            for alias in raw_aliases:
                alias_text = cls._normalize_command(str(alias or ""))
                if alias_text and alias_text != command_head:
                    aliases.append(alias_text)
        return cls._merge_unique_strings(aliases, [])

    @classmethod
    def _extract_parser_prefixes(cls, parser: object) -> list[str]:
        prefixes: list[str] = []
        seen_ids: set[int] = set()
        candidates: list[object] = [parser]
        while candidates:
            candidate = candidates.pop()
            if candidate is None or id(candidate) in seen_ids:
                continue
            seen_ids.add(id(candidate))

            raw_prefixes = getattr(candidate, "prefixes", None)
            if isinstance(raw_prefixes, str):
                raw_prefixes = [raw_prefixes]
            if isinstance(raw_prefixes, list | tuple | set | frozenset):
                for prefix in raw_prefixes:
                    prefix_text = cls._normalize_command(str(prefix or ""))
                    if prefix_text:
                        prefixes.append(prefix_text)

            for attr_name in ("parser", "meta", "config", "_config", "namespace"):
                nested = getattr(candidate, attr_name, None)
                if nested is not None and not isinstance(nested, str | bytes):
                    candidates.append(nested)
        return cls._merge_unique_strings(prefixes, [])

    @classmethod
    def _extract_parser_shortcut_aliases(cls, parser: object) -> list[str]:
        shortcuts: list[str] = []
        for shortcut_key, shortcut_obj in cls._iter_shortcut_records(parser):
            shortcuts.extend(
                cls._extract_shortcut_labels(
                    shortcut_key=shortcut_key,
                    shortcut_obj=shortcut_obj,
                )
            )
        return cls._merge_unique_strings(shortcuts, [])

    @classmethod
    def _extract_parser_schema(cls, parser: object) -> dict[str, Any]:
        params: list[str] = []
        text_min = 0
        text_max: int | None = 0
        image_min = 0
        image_max: int | None = 0
        allow_at: bool | None = None
        target_sources: list[str] = []
        prefixes = cls._extract_parser_prefixes(parser)
        sample_text = cls._sticky_probe_token

        try:
            args = list(getattr(parser, "args", None) or [])
        except Exception:
            args = []

        for arg in args:
            if bool(getattr(arg, "hidden", False)):
                continue
            arg_name = str(getattr(arg, "name", "") or "").strip()
            if arg_name:
                params.append(arg_name)
            arg_repr = f"{arg_name} {getattr(arg, 'value', None)!r}".lower()
            is_optional = cls._is_optional_arg(arg)
            is_variadic = cls._is_variadic_arg(arg)
            if cls._contains_any(arg_repr, cls._image_type_hints):
                image_min += 0 if is_optional else 1
                if image_max is not None:
                    image_max = None if is_variadic else image_max + 1
                if "reply" not in target_sources:
                    target_sources.append("reply")
                continue
            if cls._contains_any(arg_repr, cls._at_type_hints):
                allow_at = True
                for source in ("at", "reply", "nickname"):
                    if source not in target_sources:
                        target_sources.append(source)
                continue

            text_min += 0 if is_optional else 1
            if text_max is not None:
                text_max = None if is_variadic else text_max + 1
            sample_text = cls._build_sample_text(arg_name, arg_repr)

        if not args:
            text_max = 0
            image_max = 0
        return {
            "params": cls._merge_unique_strings(params, []),
            "text_min": text_min,
            "text_max": text_max,
            "image_min": image_min,
            "image_max": image_max,
            "allow_at": allow_at,
            "target_sources": target_sources,
            "prefixes": prefixes,
            "sample_text": sample_text,
        }

    @staticmethod
    def _is_optional_arg(arg: object) -> bool:
        if bool(getattr(arg, "optional", False)):
            return True
        field = getattr(arg, "field", None)
        if field is None:
            return False
        return getattr(field, "default", inspect._empty) is not inspect._empty

    @staticmethod
    def _is_variadic_arg(arg: object) -> bool:
        if str(getattr(arg, "nargs", "") or "").strip() in {"*", "+"}:
            return True
        value = getattr(arg, "value", None)
        value_text = f"{type(value).__name__} {value!r}".lower()
        return "multivar" in value_text or "variadic" in value_text

    @classmethod
    def _build_sample_text(cls, arg_name: str, arg_repr: str) -> str:
        if "int" in arg_repr or "count" in arg_name.lower() or "id" in arg_name.lower():
            return "1"
        if "float" in arg_repr or "ratio" in arg_name.lower():
            return "1.0"
        return cls._sticky_probe_token

    @classmethod
    def _probe_sticky_arg(
        cls,
        *,
        parser: object,
        command_head: str,
        sample_text: str,
    ) -> bool:
        parse = getattr(parser, "parse", None)
        if not callable(parse) or not command_head or command_head.startswith("re:"):
            return False
        try:
            result = parse(f"{command_head}{sample_text}")
        except Exception:
            return False
        if bool(getattr(result, "matched", False)):
            return True
        header_match = getattr(result, "header_match", None)
        return bool(getattr(header_match, "matched", False))

    @classmethod
    def _extract_handler_hint(cls, matcher: object) -> dict[str, Any]:
        allow_at: bool | None = None
        target_sources: list[str] = []
        requires_reply = False
        requires_superuser = False
        for handler in getattr(matcher, "handlers", []) or []:
            call = getattr(handler, "call", None)
            if call is None:
                continue
            hint = cls._load_handler_hint(call)
            if hint.get("allow_at"):
                allow_at = True
            if hint.get("reply_source") and "reply" not in target_sources:
                target_sources.append("reply")
            if hint.get("reply_source"):
                requires_reply = True
            if hint.get("at_source"):
                for source in ("at", "nickname"):
                    if source not in target_sources:
                        target_sources.append(source)
            if hint.get("self_source") and "self" not in target_sources:
                target_sources.append("self")
            if hint.get("requires_superuser"):
                requires_superuser = True
        return {
            "allow_at": allow_at,
            "target_sources": target_sources,
            "requires_reply": requires_reply,
            "requires_superuser": requires_superuser,
        }

    @classmethod
    def _extract_matcher_context_hint(cls, matcher: object) -> dict[str, bool]:
        requires_reply = False
        requires_to_me = False
        requires_private = False
        rule = getattr(matcher, "rule", None)
        rule_repr = repr(rule).lower() if rule is not None else ""
        before_rules = getattr(rule, "before_rules", None)
        before_repr = repr(before_rules).lower() if before_rules is not None else ""
        combined_repr = " ".join(part for part in (rule_repr, before_repr) if part)
        if "reply" in combined_repr:
            requires_reply = True
        if "tome" in combined_repr or "to_me" in combined_repr:
            requires_to_me = True
        if "ensure_private" in combined_repr or "private" in combined_repr:
            requires_private = True
        return {
            "requires_reply": requires_reply,
            "requires_to_me": requires_to_me,
            "requires_private": requires_private,
        }

    @classmethod
    def _load_handler_hint(cls, call: object) -> dict[str, bool]:
        source_file = inspect.getsourcefile(cast(Any, call))
        qualname = str(getattr(call, "__qualname__", "") or repr(call))
        cache_key = f"{source_file or ''}:{qualname}"
        try:
            mtime_ns = Path(source_file).stat().st_mtime_ns if source_file else 0
        except OSError:
            mtime_ns = 0

        cached = cls._handler_hint_cache.get(cache_key)
        if cached is not None and cached[0] == mtime_ns:
            return cached[1]

        try:
            source = inspect.getsource(cast(Any, call))
        except Exception:
            source = ""
        lowered = source.lower()
        reply_source = any(
            marker in lowered
            for marker in (
                "event.reply",
                ".reply",
                "get_reply",
                "reply_event",
                "reply_source",
                "reply_msg",
                "reply_message",
            )
        )
        # reply_to=True 只是“回复用户消息”的发送方式，不代表命令需要回复上下文。
        reply_source = reply_source and "reply_to" not in lowered
        hint = {
            "allow_at": "at(" in lowered
            or "argot" in lowered
            or "msgtarget" in lowered
            or '"at"' in lowered
            or "'at'" in lowered,
            "reply_source": reply_source,
            "at_source": "at(" in lowered
            or "msgtarget" in lowered
            or '"at"' in lowered
            or "'at'" in lowered,
            "self_source": "自己" in source or "user_id" in lowered,
            "requires_superuser": "dependssuperuser" in lowered
            or "depends(superuser" in lowered
            or "depends(superuser()" in lowered
            or ("is_superuser" in lowered and "depends(" in lowered),
        }
        cls._handler_hint_cache[cache_key] = (mtime_ns, hint)
        return hint

    @classmethod
    def _load_module_access_map(cls, module_obj: object) -> dict[str, str]:
        source_file = inspect.getsourcefile(cast(Any, module_obj))
        if not source_file:
            return {}
        try:
            path = Path(source_file)
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            return {}

        cache_key = str(path)
        cached = cls._module_access_cache.get(cache_key)
        if cached is not None and cached[0] == mtime_ns:
            return cached[1]

        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        symbol_levels: dict[str, str] = {}
        access_map: dict[str, str] = {}

        for node in tree.body:
            if isinstance(node, ast.Assign):
                level = cls._infer_access_level_from_expr(node.value, symbol_levels)
                if level == "public":
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        symbol_levels[target.id.casefold()] = level

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            command_head = cls._extract_command_from_call_node(node)
            if not command_head:
                continue
            level = "public"
            for keyword in node.keywords or []:
                if keyword.arg == "permission":
                    level = cls._merge_access_level(
                        level,
                        cls._infer_access_level_from_expr(keyword.value, symbol_levels),
                    )
                elif keyword.arg == "rule":
                    level = cls._merge_access_level(
                        level,
                        cls._extract_rule_access_level(keyword.value, symbol_levels),
                    )
            if level != "public":
                access_map[command_head.casefold()] = cls._merge_access_level(
                    access_map.get(command_head.casefold(), "public"),
                    level,
                )

        cls._module_access_cache[cache_key] = (mtime_ns, access_map)
        return access_map

    @classmethod
    def _infer_access_level_from_expr(
        cls,
        expr: object,
        symbol_levels: dict[str, str],
    ) -> str:
        if isinstance(expr, ast.Name):
            mapped = symbol_levels.get(expr.id.casefold())
            if mapped:
                return mapped
        if isinstance(expr, ast.AST):
            try:
                text = ast.unparse(expr)
            except Exception:
                text = str(expr or "")
        else:
            text = str(expr or "")
        return cls._infer_access_level_from_text(text, symbol_levels)

    @classmethod
    def _infer_access_level_from_text(
        cls,
        text: str,
        symbol_levels: dict[str, str] | None = None,
    ) -> str:
        normalized = str(text or "").strip().lower()
        if not normalized:
            return "public"
        symbol_levels = symbol_levels or {}
        if normalized in symbol_levels:
            return symbol_levels[normalized]
        if normalized in {"admin", "superuser", "restricted"}:
            return normalized
        if normalized.endswith(".admin") or normalized.endswith("admin()"):
            return "admin"
        if normalized.endswith(".superuser") or normalized.endswith("superuser()"):
            return "superuser"

        has_superuser = "superuser" in normalized or "superuser()" in normalized
        has_admin = (
            "admin_check" in normalized
            or "plugintype.admin" in normalized
            or "plugintype.super_and_admin" in normalized
            or "admin_level" in normalized
            or "depends(admin" in normalized
        )
        if has_superuser and has_admin:
            return "restricted"
        if has_superuser:
            return "superuser"
        if has_admin:
            return "admin"
        return "public"

    @classmethod
    def _extract_rule_access_level(
        cls,
        checker_call: object,
        symbol_levels: dict[str, str] | None = None,
    ) -> str:
        checker_name = type(checker_call).__name__
        try:
            text = ast.unparse(checker_call)  # type: ignore[arg-type]
        except Exception:
            text = repr(checker_call)
        level = cls._infer_access_level_from_text(text, symbol_levels)
        if checker_name == "RegexRule":
            return "public"
        return level

    @staticmethod
    def _merge_access_level(left: str | None, right: str | None) -> str:
        left_level = str(left or "public").strip().lower() or "public"
        right_level = str(right or "public").strip().lower() or "public"
        if left_level == right_level:
            return left_level
        if "restricted" in {left_level, right_level}:
            return "restricted"
        levels = {left_level, right_level} - {"public"}
        if not levels:
            return "public"
        if levels == {"admin"}:
            return "admin"
        if levels == {"superuser"}:
            return "superuser"
        return "restricted"

    @classmethod
    def _resolve_access_level(cls, *levels: object) -> str:
        resolved = "public"
        for level in levels:
            resolved = cls._merge_access_level(
                resolved,
                str(level) if level else "public",
            )
        return resolved

    @classmethod
    async def _extract_meme_manager_command_data(
        cls,
        manager_obj: object,
    ) -> list[dict[str, Any]]:
        memes = cls._load_manager_memes(manager_obj)
        if inspect.isawaitable(memes):
            try:
                memes = await memes
            except Exception:
                memes = None
        if memes is None:
            return []
        if isinstance(memes, dict):
            meme_items = list(memes.values())
        elif isinstance(memes, list | tuple | set):
            meme_items = list(memes)
        else:
            return []

        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for meme in meme_items:
            info = getattr(meme, "info", None)
            params = getattr(info, "params", None) if info is not None else None
            for command in cls._iter_meme_heads(meme, info):
                normalized = cls._normalize_command(command)
                if not normalized:
                    continue
                folded = normalized.casefold()
                if folded in seen:
                    continue
                seen.add(folded)
                result.append(
                    {
                        "command": normalized,
                        "text_min": cls._safe_int(getattr(params, "min_texts", None)),
                        "text_max": cls._safe_int(getattr(params, "max_texts", None)),
                        "image_min": cls._safe_int(getattr(params, "min_images", None)),
                        "image_max": cls._safe_int(getattr(params, "max_images", None)),
                        "allow_at": True,
                        "target_requirement": "optional",
                        "target_sources": ["at", "reply", "nickname", "self"],
                        "allow_sticky_arg": True,
                    }
                )
        return result

    @staticmethod
    def _load_manager_memes(manager_obj: object) -> object | None:
        getter = getattr(manager_obj, "get_memes", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                pass
        for attr_name in ("memes", "meme_dict", "registry", "all_memes"):
            value = getattr(manager_obj, attr_name, None)
            if value is not None:
                return value
        return None

    @staticmethod
    def _iter_meme_heads(meme: object, info: object | None) -> list[str]:
        heads: list[str] = []
        seen: set[str] = set()

        def add_head(value: object) -> None:
            text = str(value or "").strip()
            if not text:
                return
            folded = text.casefold()
            if folded in seen:
                return
            seen.add(folded)
            heads.append(text)

        key = str(getattr(meme, "key", "") or "").strip()
        if key:
            add_head(key)
        if info is None:
            return heads
        for keyword in getattr(info, "keywords", []) or []:
            add_head(keyword)
        for shortcut in getattr(info, "shortcuts", []) or []:
            for attr_name in ("humanized", "pattern", "key"):
                add_head(getattr(shortcut, attr_name, ""))
        return heads

    @classmethod
    def _iter_candidate_managers(
        cls,
        *,
        module_name: str,
        module_obj: object,
    ) -> list[object]:
        result: list[object] = []
        seen_ids: set[int] = set()
        for candidate_module in cls._iter_related_modules(
            module_name=module_name,
            module_obj=module_obj,
        ):
            for attr_name in (
                "meme_manager",
                "manager",
                "MemeManager",
                "MEME_MANAGER",
            ):
                manager = getattr(candidate_module, attr_name, None)
                if manager is None or id(manager) in seen_ids:
                    continue
                seen_ids.add(id(manager))
                result.append(manager)
            for value in (getattr(candidate_module, "__dict__", {}) or {}).values():
                if id(value) in seen_ids:
                    continue
                if callable(getattr(value, "get_memes", None)):
                    seen_ids.add(id(value))
                    result.append(value)
        return result

    @classmethod
    def _iter_related_modules(
        cls,
        *,
        module_name: str,
        module_obj: object,
    ) -> list[object]:
        result: list[object] = [module_obj]
        seen_ids: set[int] = {id(module_obj)}
        if module_name:
            manager_module_name = f"{module_name}.manager"
            if manager_module_name not in sys.modules:
                try:
                    importlib.import_module(manager_module_name)
                except Exception:
                    pass
            prefix = f"{module_name}."
            for related_name, related_module in list(sys.modules.items()):
                if (
                    not related_name
                    or related_module is None
                    or id(related_module) in seen_ids
                    or (
                        related_name != module_name
                        and not related_name.startswith(prefix)
                    )
                ):
                    continue
                seen_ids.add(id(related_module))
                result.append(related_module)
        return result

    @classmethod
    def _build_module_alias_map(
        cls,
        loaded_plugin: object,
    ) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {}
        for matcher in cls._iter_plugin_matchers(loaded_plugin):
            module_obj = getattr(matcher, "module", None)
            if module_obj is None:
                continue
            for command, aliases in cls._load_module_alias_map(module_obj).items():
                merged[command] = cls._merge_unique_strings(
                    merged.get(command),
                    aliases,
                )
        return merged

    @classmethod
    def _build_module_prefix_map(
        cls,
        loaded_plugin: object,
    ) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {}
        for matcher in cls._iter_plugin_matchers(loaded_plugin):
            module_obj = getattr(matcher, "module", None)
            if module_obj is None:
                continue
            for command, prefixes in cls._load_module_prefix_map(module_obj).items():
                merged[command] = cls._merge_unique_strings(
                    merged.get(command),
                    prefixes,
                )
        return merged

    @classmethod
    def _load_module_alias_map(cls, module_obj: object) -> dict[str, list[str]]:
        source_file = inspect.getsourcefile(cast(Any, module_obj))
        if not source_file:
            return {}
        try:
            path = Path(source_file)
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            return {}

        cache_key = str(path)
        cached = cls._module_alias_cache.get(cache_key)
        if cached is not None and cached[0] == mtime_ns:
            return cached[1]

        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        alias_map: dict[str, list[str]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            command = cls._extract_command_from_call_node(node)
            aliases = cls._extract_aliases_from_call_node(node)
            if not command or not aliases:
                shortcut_command, shortcut_aliases = (
                    cls._extract_shortcut_from_call_node(node)
                )
                if not shortcut_command or not shortcut_aliases:
                    continue
                alias_map[shortcut_command.casefold()] = cls._merge_unique_strings(
                    alias_map.get(shortcut_command.casefold()),
                    shortcut_aliases,
                )
                continue
            alias_map[command.casefold()] = cls._merge_unique_strings(
                alias_map.get(command.casefold()),
                aliases,
            )
        cls._module_alias_cache[cache_key] = (mtime_ns, alias_map)
        return alias_map

    @classmethod
    def _load_module_prefix_map(cls, module_obj: object) -> dict[str, list[str]]:
        source_file = inspect.getsourcefile(cast(Any, module_obj))
        if not source_file:
            return {}
        try:
            path = Path(source_file)
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            return {}

        cache_key = str(path)
        cached = cls._module_prefix_cache.get(cache_key)
        if cached is not None and cached[0] == mtime_ns:
            return cached[1]

        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        prefix_map: dict[str, list[str]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            command = cls._extract_command_from_call_node(node)
            prefixes = cls._extract_prefixes_from_call_node(node)
            if not command or not prefixes:
                continue
            prefix_map[command.casefold()] = cls._merge_unique_strings(
                prefix_map.get(command.casefold()),
                prefixes,
            )
        cls._module_prefix_cache[cache_key] = (mtime_ns, prefix_map)
        return prefix_map

    @classmethod
    def _extract_command_from_call_node(cls, node: ast.Call) -> str:
        func_name = cls._get_call_name(node.func)
        if func_name not in {"on_alconna", "on_command", "on_regex"}:
            return ""
        if not node.args:
            return ""
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            if func_name == "on_regex":
                return cls._extract_regex_head(first_arg.value) or ""
            return cls._normalize_command(first_arg.value)
        if (
            isinstance(first_arg, ast.Call)
            and cls._get_call_name(first_arg.func) == "Alconna"
            and first_arg.args
            and isinstance(first_arg.args[0], ast.Constant)
            and isinstance(first_arg.args[0].value, str)
        ):
            return cls._normalize_command(first_arg.args[0].value)
        if (
            isinstance(first_arg, ast.Call)
            and cls._get_call_name(first_arg.func) == "Alconna"
            and len(first_arg.args) >= 2
        ):
            command_arg = first_arg.args[1]
            if isinstance(command_arg, ast.Constant) and isinstance(
                command_arg.value, str
            ):
                prefix_arg = first_arg.args[0]
                if cls._extract_strings_from_node(prefix_arg):
                    return cls._normalize_command(command_arg.value)
        return ""

    @classmethod
    def _extract_prefixes_from_call_node(cls, node: ast.Call) -> list[str]:
        if cls._get_call_name(node.func) != "on_alconna" or not node.args:
            return []
        first_arg = node.args[0]
        if not isinstance(first_arg, ast.Call):
            return []
        if cls._get_call_name(first_arg.func) != "Alconna" or not first_arg.args:
            return []
        raw_prefixes = cls._extract_strings_from_node(first_arg.args[0])
        return raw_prefixes

    @classmethod
    def _extract_strings_from_node(cls, node: ast.AST) -> list[str]:
        result: list[str] = []
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = cls._normalize_command(node.value)
            if text:
                result.append(text)
            return result
        if isinstance(node, ast.List | ast.Tuple | ast.Set):
            for item in node.elts:
                result.extend(cls._extract_strings_from_node(item))
            return cls._merge_unique_strings(result, [])
        return []

    @staticmethod
    def _get_call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    @classmethod
    def _extract_aliases_from_call_node(cls, node: ast.Call) -> list[str]:
        for keyword in node.keywords or []:
            if keyword.arg != "aliases":
                continue
            try:
                raw_aliases = ast.literal_eval(keyword.value)
            except Exception:
                return []
            if isinstance(raw_aliases, str):
                return [cls._normalize_command(raw_aliases)]
            if not isinstance(raw_aliases, list | tuple | set | frozenset):
                return []
            return [
                cls._normalize_command(str(alias or ""))
                for alias in raw_aliases
                if cls._normalize_command(str(alias or ""))
            ]
        return []

    @classmethod
    def _extract_shortcut_from_call_node(cls, node: ast.Call) -> tuple[str, list[str]]:
        if cls._get_call_name(node.func) != "shortcut" or not node.args:
            return "", []

        shortcut_key = ""
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            shortcut_key = cls._coerce_shortcut_alias(first_arg.value)
        if not shortcut_key:
            return "", []

        target_command = ""
        humanized_aliases: list[str] = [shortcut_key]
        for keyword in node.keywords or []:
            if keyword.arg == "command":
                try:
                    raw_command = ast.literal_eval(keyword.value)
                except Exception:
                    raw_command = ""
                target_command = cls._normalize_command(str(raw_command or ""))
                continue
            if keyword.arg == "humanized":
                try:
                    raw_humanized = ast.literal_eval(keyword.value)
                except Exception:
                    raw_humanized = ""
                humanized = cls._coerce_shortcut_alias(str(raw_humanized or ""))
                if humanized:
                    humanized_aliases.append(humanized)
        if not target_command:
            return "", []
        return target_command, cls._merge_unique_strings(humanized_aliases, [])

    @classmethod
    def _normalize_command(cls, command: str) -> str:
        text = str(command or "").strip()
        if not text:
            return ""
        text = cls._command_placeholder_pattern.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return normalize_message_text(text)

    @classmethod
    def _coerce_shortcut_alias(cls, text: object) -> str:
        normalized = cls._normalize_command(str(text or ""))
        if not normalized:
            return ""
        if cls._looks_like_shortcut_alias(normalized):
            return normalized
        regex_head = cls._extract_regex_head(normalized)
        if regex_head and cls._looks_like_shortcut_alias(regex_head):
            return regex_head
        return ""

    @classmethod
    def _looks_like_shortcut_alias(cls, text: str) -> bool:
        normalized = cls._normalize_command(text)
        if not normalized:
            return False
        if normalized.startswith("re:"):
            return False
        if any(char in normalized for char in "\\[]()^$|"):
            return False
        return True

    @classmethod
    def _extract_shortcut_labels(
        cls,
        *,
        shortcut_key: object | None,
        shortcut_obj: object | None,
    ) -> list[str]:
        labels: list[str] = []
        candidates: list[object] = []
        if shortcut_key is not None:
            candidates.append(shortcut_key)
        if shortcut_obj is not None:
            for attr_name in ("humanized", "origin_key", "key", "pattern"):
                candidates.append(getattr(shortcut_obj, attr_name, None))
        for candidate in candidates:
            text = cls._coerce_shortcut_alias(candidate)
            if text:
                labels.append(text)
        return cls._merge_unique_strings(labels, [])

    @classmethod
    def _iter_shortcut_records(cls, owner: object) -> list[tuple[str, object]]:
        records: list[tuple[str, object]] = []
        seen: set[tuple[str, int]] = set()

        def add_record(key: object, value: object) -> None:
            key_text = cls._coerce_shortcut_alias(key)
            if not key_text:
                return
            marker = (key_text.casefold(), id(value))
            if marker in seen:
                return
            seen.add(marker)
            records.append((key_text, value))

        formatter = getattr(owner, "formatter", None)
        data = getattr(formatter, "data", None)
        shortcut_hash = getattr(owner, "_hash", None)
        if isinstance(data, dict) and shortcut_hash in data:
            trace = data.get(shortcut_hash)
            shortcuts = getattr(trace, "shortcuts", None)
            if isinstance(shortcuts, dict):
                for key, value in shortcuts.items():
                    add_record(key, value)

        for attr_name in ("_get_shortcuts", "get_shortcuts"):
            getter = getattr(owner, attr_name, None)
            if not callable(getter):
                continue
            try:
                raw_shortcuts = getter()
            except Exception:
                continue
            if isinstance(raw_shortcuts, dict):
                for key, value in raw_shortcuts.items():
                    add_record(key, value)
            elif isinstance(raw_shortcuts, list | tuple | set | frozenset):
                for item in raw_shortcuts:
                    add_record(item, item)

        raw_shortcuts = getattr(owner, "shortcuts", None)
        if isinstance(raw_shortcuts, dict):
            for key, value in raw_shortcuts.items():
                add_record(key, value)
        elif isinstance(raw_shortcuts, list | tuple | set | frozenset):
            for item in raw_shortcuts:
                add_record(item, item)

        info = getattr(owner, "info", None)
        nested_shortcuts = (
            getattr(info, "shortcuts", None) if info is not None else None
        )
        if isinstance(nested_shortcuts, dict):
            for key, value in nested_shortcuts.items():
                add_record(key, value)
        elif isinstance(nested_shortcuts, list | tuple | set | frozenset):
            for item in nested_shortcuts:
                add_record(item, item)

        return records

    @classmethod
    def _extract_regex_head(cls, pattern: str) -> str | None:
        text = str(pattern or "").strip().lstrip("^")
        if not text or text.startswith("(?:") or text.startswith("(?"):
            return None
        head = cls._regex_head_pattern.split(text, maxsplit=1)[0].strip()
        if not head or any(char in head for char in "{}:"):
            return None
        return cls._normalize_command(head)

    @staticmethod
    def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword and keyword in text for keyword in keywords)

    @staticmethod
    def _safe_int(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(cast(Any, value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _merge_unique_strings(
        left: list[str] | tuple[str, ...] | None,
        right: list[str] | tuple[str, ...] | None,
        *extra: list[str] | tuple[str, ...] | None,
    ) -> list[str]:
        result: list[str] = []
        for collection in (left, right, *extra):
            for value in collection or []:
                text = str(value or "").strip()
                if text and text not in result:
                    result.append(text)
        return result

    @classmethod
    def _merge_command_dicts(
        cls,
        commands: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for payload in commands:
            command = cls._normalize_command(str(payload.get("command") or ""))
            if not command:
                continue
            key = command.casefold()
            current = merged.setdefault(key, {"command": command})
            current["aliases"] = cls._merge_unique_strings(
                current.get("aliases"),
                payload.get("aliases"),
            )
            current["params"] = cls._merge_unique_strings(
                current.get("params"),
                payload.get("params"),
            )
            current["examples"] = cls._merge_unique_strings(
                current.get("examples"),
                payload.get("examples"),
            )
            current["prefixes"] = cls._merge_unique_strings(
                current.get("prefixes"),
                payload.get("prefixes"),
            )
            current["target_sources"] = cls._merge_unique_strings(
                current.get("target_sources"),
                payload.get("target_sources"),
            )
            current["access_level"] = cls._merge_access_level(
                current.get("access_level"), payload.get("access_level")
            )
            current["requires_reply"] = bool(current.get("requires_reply")) or bool(
                payload.get("requires_reply")
            )
            current["requires_private"] = bool(current.get("requires_private")) or bool(
                payload.get("requires_private")
            )
            current["requires_to_me"] = bool(current.get("requires_to_me")) or bool(
                payload.get("requires_to_me")
            )
            for field in (
                "text_min",
                "text_max",
                "image_min",
                "image_max",
                "allow_at",
                "actor_scope",
                "target_requirement",
                "allow_sticky_arg",
                "prefixes",
            ):
                if current.get(field) is None and payload.get(field) is not None:
                    current[field] = payload.get(field)
        return sorted(
            merged.values(),
            key=lambda item: (
                len(str(item.get("command") or "")),
                str(item.get("command") or ""),
            ),
        )


__all__ = ["AutoMetadataBuilder"]
