"""Validation and stable visual identity for reaction images."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO

from PIL import Image

MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
MAX_FINGERPRINT_FRAMES = 8

_FORMAT_EXTENSIONS = {
    "BMP": ".bmp",
    "GIF": ".gif",
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}


@dataclass(frozen=True, slots=True)
class ReactionImageInfo:
    content_sha256: str
    visual_fingerprint: str
    extension: str
    width: int
    height: int
    frame_count: int


def inspect_reaction_image(
    content: bytes,
    *,
    max_bytes: int = MAX_IMAGE_BYTES,
) -> ReactionImageInfo | None:
    if not content or len(content) > max(max_bytes, 0):
        return None
    try:
        with Image.open(BytesIO(content)) as image:
            image_format = str(image.format or "").upper()
            extension = _FORMAT_EXTENSIONS.get(image_format)
            width, height = image.size
            frame_count = max(int(getattr(image, "n_frames", 1) or 1), 1)
            if (
                extension is None
                or width <= 0
                or height <= 0
                or width * height > MAX_IMAGE_PIXELS
            ):
                return None
            fingerprint = _visual_fingerprint(
                image,
                width=width,
                height=height,
                frame_count=frame_count,
            )
    except Exception:
        return None
    return ReactionImageInfo(
        content_sha256=hashlib.sha256(content).hexdigest(),
        visual_fingerprint=fingerprint,
        extension=extension,
        width=width,
        height=height,
        frame_count=frame_count,
    )


def _visual_fingerprint(
    image: Image.Image,
    *,
    width: int,
    height: int,
    frame_count: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"chatinter-reaction-visual-v1\0")
    digest.update(f"{width}x{height}:{frame_count}".encode("ascii"))
    for index in _sample_indexes(frame_count, MAX_FINGERPRINT_FRAMES):
        image.seek(index)
        frame = image.convert("RGBA")
        duration = max(int(image.info.get("duration", 0) or 0), 0)
        digest.update(f"\0{index}:{duration}:".encode("ascii"))
        digest.update(frame.tobytes())
    return digest.hexdigest()


def _sample_indexes(frame_count: int, limit: int) -> tuple[int, ...]:
    count = min(max(frame_count, 1), max(limit, 1))
    if count == 1:
        return (0,)
    return tuple(
        sorted(
            {
                round(index * (frame_count - 1) / (count - 1))
                for index in range(count)
            }
        )
    )


__all__ = [
    "MAX_IMAGE_BYTES",
    "ReactionImageInfo",
    "inspect_reaction_image",
]
