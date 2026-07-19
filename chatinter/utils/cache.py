"""
ChatInter - 缓存工具

包含：
1. 好感度缓存（使用框架 CacheDict）
"""

from datetime import datetime

from zhenxun.services.cache.cache_containers import CacheDict

from .impression_provider import get_impression_provider

_impression_cache: CacheDict[tuple[float, str, float]] = CacheDict(
    "chatinter_impression",
    expire=6 * 60 * 60,
)


async def get_user_impression_with_cache(user_id: str) -> tuple[float, str]:
    """
    获取用户好感度（带缓存）

    参数:
        user_id: 用户 ID

    返回:
        tuple: (impression, attitude)
    """
    now_ts = datetime.now().timestamp()
    impression_cache = _impression_cache.get(user_id)

    if (
        impression_cache is None
        or len(impression_cache) < 3
        or now_ts - impression_cache[2] > _impression_cache.expire
    ):
        impression, attitude = await get_impression_provider().get_user_impression(
            user_id
        )
        _impression_cache[user_id] = (impression, attitude, now_ts)
    else:
        impression = impression_cache[0]
        attitude = impression_cache[1]

    return impression, attitude


def clear_impression_cache():
    """清空好感度缓存"""
    _impression_cache.clear()
