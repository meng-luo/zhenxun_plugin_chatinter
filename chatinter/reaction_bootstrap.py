"""Safe first-run installation of ChatInter's default reaction pack."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any
import zipfile

import httpx

from .log_compat import logger

DEFAULT_PACK_ID = "seio-stickers"
DEFAULT_PACK_REPOSITORY = "anka-afk/seio-stickers"
DEFAULT_PACK_SOURCE_REF = "main"

_DEFAULT_PACK_COMMIT_API = (
    f"https://api.github.com/repos/{DEFAULT_PACK_REPOSITORY}/commits/"
    f"{DEFAULT_PACK_SOURCE_REF}"
)
_PROVENANCE_SCHEMA_VERSION = 2
_PACK_METADATA_FILES = frozenset(
    {"LICENSE-ASSETS.md", "manifest.json", "memes_data.json", "upstream-manifest.json"}
)
_MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_MAX_ARCHIVE_FILES = 5_000
_MAX_SINGLE_FILE_BYTES = 32 * 1024 * 1024
_IMAGE_EXTENSIONS = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"})


async def install_default_reaction_pack(root: Path) -> bool:
    root = root.expanduser().resolve()
    if await asyncio.to_thread(_library_has_images, root):
        return False
    try:
        source_commit = await _resolve_main_commit()
        installed = await _download_and_install(root, source_commit)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(f"ChatInter 默认表情包首次安装失败，将在下次启动重试：{exc}")
        return False
    if not installed:
        return False
    logger.info(
        f"ChatInter 默认表情包已安装：{DEFAULT_PACK_ID}@{source_commit}"
    )
    return True


async def _resolve_main_commit() -> str:
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        response = await client.get(
            _DEFAULT_PACK_COMMIT_API,
            headers={"Accept": "application/vnd.github+json"},
        )
        response.raise_for_status()
        payload = response.json()
    commit = (
        str(payload.get("sha") or "").strip().casefold()
        if isinstance(payload, dict)
        else ""
    )
    if not _is_commit_sha(commit):
        raise ValueError("默认表情包 main commit 无效")
    return commit


async def _download_and_install(root: Path, source_commit: str) -> bool:
    if not _is_commit_sha(source_commit):
        raise ValueError("默认表情包 source commit 无效")
    await asyncio.to_thread(root.parent.mkdir, parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".chatinter-reactions-", dir=root.parent)
    )
    archive = temporary_root / "pack.zip"
    staging = temporary_root / "staging"
    try:
        await _download_archive(archive, source_commit)
        await asyncio.to_thread(
            _extract_validated,
            archive,
            staging,
            source_commit=source_commit,
        )
        return await asyncio.to_thread(_install_staging, staging, root)
    finally:
        await asyncio.to_thread(shutil.rmtree, temporary_root, True)


async def _download_archive(target: Path, source_commit: str) -> None:
    downloaded = 0
    url = (
        f"https://github.com/{DEFAULT_PACK_REPOSITORY}/archive/"
        f"{source_commit}.zip"
    )
    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            expected = _safe_int(response.headers.get("content-length"))
            if expected and expected > _MAX_ARCHIVE_BYTES:
                raise ValueError("默认表情包压缩文件超过安全限制")
            with target.open("wb") as stream:
                async for chunk in response.aiter_bytes(1024 * 1024):
                    downloaded += len(chunk)
                    if downloaded > _MAX_ARCHIVE_BYTES:
                        raise ValueError("默认表情包压缩文件超过安全限制")
                    stream.write(chunk)


def _extract_validated(
    archive: Path,
    staging: Path,
    *,
    source_commit: str,
) -> None:
    if not _is_commit_sha(source_commit):
        raise ValueError("默认表情包 source commit 无效")
    staging.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        files = [item for item in bundle.infolist() if not item.is_dir()]
        if len(files) > _MAX_ARCHIVE_FILES:
            raise ValueError("默认表情包文件数量超过安全限制")
        total = sum(max(int(item.file_size), 0) for item in files)
        if total > _MAX_UNCOMPRESSED_BYTES:
            raise ValueError("默认表情包解压后超过安全限制")
        entries = _archive_entries(files)
        manifest = _read_archive_json(bundle, entries, "manifest.json")
        categories, pack_version, pack_license = _validate_manifest(manifest)
        category_data = _read_archive_json(bundle, entries, "memes_data.json")
        _validate_category_data(category_data, categories)
        upstream = _read_archive_json(bundle, entries, "upstream-manifest.json")
        stickers = _validate_upstream_manifest(upstream)
        license_item = entries.get("LICENSE-ASSETS.md")
        if license_item is None:
            raise ValueError("默认表情包缺少 LICENSE-ASSETS.md")
        license_text = bundle.read(license_item).decode("utf-8-sig")
        if (
            "仅限非商业使用" not in license_text
            or "自由下载、分享与分发" not in license_text
        ):
            raise ValueError("默认表情包素材许可信息不匹配")

        runtime_entries = [
            (relative, item)
            for relative, item in entries.items()
            if relative.startswith("memes/")
            and PurePosixPath(relative).suffix.casefold() == ".gif"
        ]
        runtime_images = {
            PurePosixPath(relative).name: (relative, item)
            for relative, item in runtime_entries
        }
        if not runtime_entries or len(runtime_entries) != len(stickers):
            raise ValueError("默认表情包运行图片与上游清单数量不一致")
        if len(runtime_images) != len(runtime_entries):
            raise ValueError("默认表情包包含重复图片文件名")

        semantics: dict[str, dict[str, Any]] = {}
        for sticker in stickers:
            upstream_name = PurePosixPath(sticker["path"]).name
            runtime = runtime_images.pop(upstream_name, None)
            if runtime is None:
                raise ValueError(f"默认表情包缺少上游映射：{upstream_name}")
            relative, item = runtime
            path = PurePosixPath(relative)
            if len(path.parts) != 3 or path.parts[1] not in categories:
                raise ValueError(f"默认表情包图片分类无效：{relative}")
            if item.file_size != sticker["size"]:
                raise ValueError(f"默认表情包图片大小不匹配：{relative}")
            content = bundle.read(item)
            digest = hashlib.sha256(content).hexdigest()
            if digest != sticker["sha256"]:
                raise ValueError(f"默认表情包图片摘要不匹配：{relative}")
            _write_staging_file(staging, relative, content)
            semantics[digest] = _semantic_record(
                digest=digest,
                relative=relative,
                category=path.parts[1],
                category_description=categories[path.parts[1]],
                sticker_name=sticker["name"],
                size=item.file_size,
                source_commit=source_commit,
            )
        if runtime_images:
            raise ValueError("默认表情包包含未登记图片")

        for relative in sorted(_PACK_METADATA_FILES):
            item = entries.get(relative)
            if item is None:
                raise ValueError(f"默认表情包缺少 {relative}")
            _write_staging_file(staging, relative, bundle.read(item))
        _write_json(
            staging / "semantic_metadata.json",
            {"version": 2, "images": semantics},
        )
        _write_json(
            staging / "chatinter_default_pack.json",
            _build_provenance(
                staging,
                source_commit=source_commit,
                pack_version=pack_version,
                pack_license=pack_license,
            ),
        )


def _archive_entries(files: list[zipfile.ZipInfo]) -> dict[str, zipfile.ZipInfo]:
    entries: dict[str, zipfile.ZipInfo] = {}
    for item in files:
        relative = _archive_relative(item.filename)
        if not relative:
            continue
        mode = (item.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise ValueError("默认表情包包含符号链接")
        if item.file_size > _MAX_SINGLE_FILE_BYTES:
            raise ValueError("默认表情包包含超大文件")
        if relative in entries:
            raise ValueError(f"默认表情包包含重复路径：{relative}")
        entries[relative] = item
    return entries


def _read_archive_json(
    bundle: zipfile.ZipFile,
    entries: dict[str, zipfile.ZipInfo],
    relative: str,
) -> dict[str, Any]:
    item = entries.get(relative)
    if item is None:
        raise ValueError(f"默认表情包缺少 {relative}")
    try:
        payload = json.loads(bundle.read(item).decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"默认表情包 {relative} 格式无效") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"默认表情包 {relative} 格式无效")
    return payload


def _validate_manifest(
    manifest: dict[str, Any],
) -> tuple[dict[str, str], str, str]:
    source = manifest.get("source")
    pack_version = " ".join(str(manifest.get("version") or "").split())[:120]
    pack_license = " ".join(str(manifest.get("license") or "").split())[:500]
    if (
        _safe_int(manifest.get("schema_version")) < 1
        or str(manifest.get("id") or "") != DEFAULT_PACK_ID
        or not pack_version
        or not pack_license
        or not isinstance(source, dict)
        or str(source.get("repo") or "") != DEFAULT_PACK_REPOSITORY
        or not str(source.get("ref") or "").strip()
    ):
        raise ValueError("默认表情包 manifest 身份或版本无效")
    raw_categories = manifest.get("categories")
    if not isinstance(raw_categories, dict) or not raw_categories:
        raise ValueError("默认表情包分类为空或格式无效")
    categories: dict[str, str] = {}
    for raw_name, raw_value in raw_categories.items():
        name = str(raw_name or "").strip()
        description = (
            raw_value.get("description") if isinstance(raw_value, dict) else ""
        )
        normalized = " ".join(str(description or "").split())
        if not name or not normalized:
            raise ValueError("默认表情包分类描述无效")
        categories[name] = normalized[:500]
    return categories, pack_version, pack_license


def _validate_category_data(
    payload: dict[str, Any], categories: dict[str, str]
) -> None:
    normalized = {
        str(key): " ".join(str(value or "").split())
        for key, value in payload.items()
        if isinstance(value, str)
    }
    if normalized != categories:
        raise ValueError("默认表情包分类适配信息不一致")


def _validate_upstream_manifest(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        str(payload.get("name") or "") != "astrbot-seio-stickers"
        or not str(payload.get("version") or "").strip()
    ):
        raise ValueError("默认表情包上游 manifest 身份不匹配")
    raw_stickers = payload.get("stickers")
    if not isinstance(raw_stickers, list) or not raw_stickers:
        raise ValueError("默认表情包上游图片清单为空或格式无效")
    stickers: list[dict[str, Any]] = []
    names: set[str] = set()
    paths: set[str] = set()
    digests: set[str] = set()
    for raw in raw_stickers:
        if not isinstance(raw, dict):
            raise ValueError("默认表情包上游图片记录无效")
        name = " ".join(str(raw.get("name") or "").split())
        path = str(raw.get("path") or "").replace("\\", "/")
        size = _safe_int(raw.get("size"))
        digest = str(raw.get("sha256") or "").strip().casefold()
        parsed = PurePosixPath(path)
        if (
            not name
            or name in names
            or path in paths
            or digest in digests
            or len(parsed.parts) != 2
            or parsed.parts[0] != "stickers"
            or parsed.suffix.casefold() != ".gif"
            or not size
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("默认表情包上游图片身份、路径或摘要无效")
        names.add(name)
        paths.add(path)
        digests.add(digest)
        stickers.append({"name": name, "path": path, "size": size, "sha256": digest})
    return stickers


def _semantic_record(
    *,
    digest: str,
    relative: str,
    category: str,
    category_description: str,
    sticker_name: str,
    size: int,
    source_commit: str,
) -> dict[str, Any]:
    return {
        "content_sha256": digest,
        "relative_path": relative,
        "category": category,
        "category_description": category_description,
        "caption": sticker_name,
        "tags": [sticker_name],
        "visible_text": "",
        "reply_intents": [],
        "usage_scenarios": [],
        "tones": [],
        "actions": [],
        "target_relation": "",
        "semantic_version": 2,
        "status": "ready",
        "provenance": "seio_default",
        "source_version": source_commit,
        "size": size,
        "mtime_ns": 0,
    }


def _build_provenance(
    staging: Path,
    *,
    source_commit: str,
    pack_version: str,
    pack_license: str,
) -> dict[str, Any]:
    managed: list[dict[str, Any]] = []
    for path in sorted(
        candidate
        for candidate in staging.rglob("*")
        if candidate.is_file()
        and candidate.name
        not in {"chatinter_default_pack.json", "semantic_metadata.json"}
    ):
        managed.append(
            {
                "path": path.relative_to(staging).as_posix(),
                "size": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return {
        "schema_version": _PROVENANCE_SCHEMA_VERSION,
        "id": DEFAULT_PACK_ID,
        "repository": DEFAULT_PACK_REPOSITORY,
        "commit": source_commit,
        "source_ref": DEFAULT_PACK_SOURCE_REF,
        "version": pack_version,
        "license": pack_license,
        "files": managed,
    }


def _archive_relative(filename: str) -> str:
    normalized = str(filename).replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = path.parts
    if (
        path.is_absolute()
        or ".." in parts
        or any(":" in part for part in parts)
        or "\x00" in normalized
    ):
        raise ValueError("默认表情包包含不安全路径")
    if len(parts) < 2:
        return ""
    return PurePosixPath(*parts[1:]).as_posix()


def _write_staging_file(staging: Path, relative: str, content: bytes) -> None:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("默认表情包包含不安全路径")
    target = (staging / Path(*path.parts)).resolve()
    target.relative_to(staging.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def _install_staging(staging: Path, root: Path) -> bool:
    if _library_has_images(root):
        return False
    candidate = staging.parent / "candidate"
    if root.exists():
        shutil.copytree(root, candidate, symlinks=True)
    else:
        candidate.mkdir(parents=True)
    for source in sorted(staging.rglob("*")):
        if not source.is_file():
            continue
        target = candidate / source.relative_to(staging)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _validate_installed_candidate(candidate)
    if _library_has_images(root):
        return False
    _atomic_replace_directory(candidate, root)
    return True


def _validate_installed_candidate(root: Path) -> None:
    payload = _read_json(root / "chatinter_default_pack.json")
    commit = str(payload.get("commit") or "").casefold()
    if (
        str(payload.get("id") or "") != DEFAULT_PACK_ID
        or str(payload.get("repository") or "") != DEFAULT_PACK_REPOSITORY
        or str(payload.get("source_ref") or "") != DEFAULT_PACK_SOURCE_REF
        or not _is_commit_sha(commit)
    ):
        raise ValueError("默认表情包 provenance 无效")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("默认表情包 provenance 文件列表无效")
    image_count = 0
    for value in files:
        if not isinstance(value, dict):
            raise ValueError("默认表情包 provenance 文件记录无效")
        relative = _safe_relative_path(value.get("path"))
        target = _safe_managed_path(root, relative)
        if target is None or not target.is_file():
            raise ValueError(f"默认表情包安装文件缺失：{relative}")
        if target.stat().st_size != _safe_int(value.get("size")):
            raise ValueError(f"默认表情包安装文件大小异常：{relative}")
        if _file_sha256(target) != str(value.get("sha256") or "").casefold():
            raise ValueError(f"默认表情包安装文件摘要异常：{relative}")
        if relative.startswith("memes/") and target.suffix.casefold() == ".gif":
            image_count += 1
    semantics = _read_json(root / "semantic_metadata.json").get("images")
    ready_count = (
        sum(
            1
            for value in semantics.values()
            if isinstance(value, dict)
            and str(value.get("provenance") or "") == "seio_default"
            and str(value.get("source_version") or "").casefold() == commit
            and str(value.get("status") or "") == "ready"
            and _safe_int(value.get("semantic_version")) >= 2
        )
        if isinstance(semantics, dict)
        else 0
    )
    if image_count <= 0 or image_count != ready_count:
        raise ValueError("默认表情包图片与注册 metadata 数量不一致")


def _safe_relative_path(value: Any) -> str:
    normalized = str(value or "").replace("\\", "/").strip()
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or any(":" in part for part in path.parts)
        or "\x00" in normalized
    ):
        return ""
    return path.as_posix()


def _safe_managed_path(root: Path, relative: str) -> Path | None:
    normalized = _safe_relative_path(relative)
    if not normalized:
        return None
    try:
        target = (root / Path(*PurePosixPath(normalized).parts)).resolve()
        target.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return target


def _atomic_replace_directory(candidate: Path, root: Path) -> None:
    backup = candidate.parent / "backup"
    had_root = root.exists()
    if had_root:
        os.replace(root, backup)
    try:
        os.replace(candidate, root)
    except BaseException:
        if had_root and backup.exists() and not root.exists():
            os.replace(backup, root)
        raise
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


def _library_has_images(root: Path) -> bool:
    memes = root / "memes"
    return memes.is_dir() and any(
        path.is_file() and path.suffix.casefold() in _IMAGE_EXTENSIONS
        for path in memes.rglob("*")
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_commit_sha(value: str) -> bool:
    return len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def _safe_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


__all__ = ["DEFAULT_PACK_ID", "install_default_reaction_pack"]
