"""
ChatInter - 缓存工具

包含：
1. 好感度缓存（使用框架 CacheDict）
"""

from datetime import datetime

from zhenxun.models.sign_user import SignUser
from zhenxun.services.cache.cache_containers import CacheDict

# 好感度缓存（使用框架 CacheDict，6 小时过期）
_impression_cache: CacheDict[tuple[float, float]] = CacheDict(
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
        or now_ts - impression_cache[1] > _impression_cache.expire
    ):
        sign_user = await SignUser.get_user(user_id)
        impression = float(sign_user.impression)
        _impression_cache[user_id] = (impression, now_ts)
    else:
        impression = impression_cache[0]

    from zhenxun.builtin_plugins.sign_in.config import level2attitude
    from zhenxun.builtin_plugins.sign_in.utils import get_level_and_next_impression

    level, _, _ = get_level_and_next_impression(impression)
    attitude = level2attitude.get(str(level), "未知")

    return impression, attitude


def clear_impression_cache():
    """清空好感度缓存"""
    _impression_cache.clear()
