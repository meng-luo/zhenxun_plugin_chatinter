from tortoise import fields

from zhenxun.services.db_context import Model


class ChatInterChatHistory(Model):
    """ChatInter 聊天历史表

    采用一问一答格式，每条记录包含用户消息和 AI 回复
    参考：bym_ai.models.bym_chat
    """

    id = fields.IntField(pk=True, generated=True, auto_increment=True)
    """自增 id"""
    session_id = fields.CharField(255, index=True)
    """会话标识（私聊：user_id / 群聊：group_id）"""
    user_id = fields.CharField(255)
    """用户 id"""
    group_id = fields.CharField(255, null=True)
    """群组 id（私聊时为 null）"""
    nickname = fields.CharField(255)
    """用户昵称"""
    user_message = fields.TextField()
    """用户消息"""
    ai_response = fields.TextField(null=True)
    """AI 回复内容"""
    bot_id = fields.CharField(255, null=True)
    """Bot ID"""
    create_time = fields.DatetimeField(auto_now_add=True, index=True)
    """创建时间"""
    reset = fields.BooleanField(default=False, index=True)
    """是否被重置（软删除标记）"""

    class Meta:  # pyright: ignore [reportIncompatibleVariableOverride]
        table = "chatinter_chat_history"
        table_description = "ChatInter 聊天历史表"

    @classmethod
    async def get_recent_dialogs(
        cls, session_id: str, limit: int = 5
    ) -> list["ChatInterChatHistory"]:
        """
        获取指定会话最近的 N 轮对话（用于语境前缀）

        参数:
            session_id: 会话标识
            limit: 获取数量（对话轮次）

        返回:
            list[ChatInterChatHistory]: 按时间正序排列的对话列表
        """
        dialogs = (
            await cls.filter(session_id=session_id, reset=False)
            .order_by("-create_time", "-id")
            .limit(limit)
        )
        # 反转为正序（从旧到新）
        return list(reversed(dialogs))

    @classmethod
    async def get_conversation_history(
        cls, session_id: str, limit: int = 20
    ) -> list["ChatInterChatHistory"]:
        """
        获取完整的对话历史（用于构建上下文）

        参数:
            session_id: 会话标识
            limit: 获取数量（对话轮次上限）

        返回:
            list[ChatInterChatHistory]: 按时间正序排列的对话列表
        """
        dialogs = (
            await cls.filter(session_id=session_id, reset=False)
            .order_by("create_time", "id")
            .limit(limit)
        )
        return list(dialogs)

    @classmethod
    async def add_dialog(
        cls,
        session_id: str,
        user_id: str,
        nickname: str,
        user_message: str,
        ai_response: str,
        group_id: str | None = None,
        bot_id: str | None = None,
    ) -> "ChatInterChatHistory":
        """
        添加一轮对话到数据库（一问一答）

        参数:
            session_id: 会话标识
            user_id: 用户 id
            nickname: 用户昵称
            user_message: 用户消息
            ai_response: AI 回复内容
            group_id: 群组 id（可选）
            bot_id: Bot ID（可选）

        返回:
            ChatInterChatHistory: 创建的对话记录
        """
        return await cls.create(
            session_id=session_id,
            user_id=user_id,
            group_id=group_id,
            nickname=nickname,
            user_message=user_message,
            ai_response=ai_response,
            bot_id=bot_id,
        )

    @classmethod
    async def add_user_message(
        cls,
        session_id: str,
        user_id: str,
        nickname: str,
        user_message: str,
        group_id: str | None = None,
        bot_id: str | None = None,
    ) -> "ChatInterChatHistory":
        """
        仅添加用户消息（暂时存储，等待 AI 回复后更新）

        参数:
            session_id: 会话标识
            user_id: 用户 id
            nickname: 用户昵称
            user_message: 用户消息
            group_id: 群组 id（可选）
            bot_id: Bot ID（可选）

        返回:
            ChatInterChatHistory: 创建的消息记录
        """
        return await cls.create(
            session_id=session_id,
            user_id=user_id,
            group_id=group_id,
            nickname=nickname,
            user_message=user_message,
            ai_response=None,
            bot_id=bot_id,
        )

    @classmethod
    async def update_ai_response(
        cls, dialog_id: int, ai_response: str, bot_id: str | None = None
    ) -> bool:
        """
        更新 AI 回复内容

        参数:
            dialog_id: 对话记录 ID
            ai_response: AI 回复内容
            bot_id: Bot ID（可选）

        返回:
            bool: 是否更新成功
        """
        updated = await cls.filter(id=dialog_id).update(
            ai_response=ai_response,
            bot_id=bot_id,
        )
        return updated > 0

    @classmethod
    async def prune_old_dialogs(cls, session_id: str, max_limit: int):
        """
        删除超出上限的旧对话（保留最近的 N 轮）

        参数:
            session_id: 会话标识
            max_limit: 最大保留数量
        """
        total = await cls.filter(session_id=session_id, reset=False).count()
        if total > max_limit:
            to_delete_count = total - max_limit
            to_delete = (
                await cls.filter(session_id=session_id, reset=False)
                .order_by("create_time", "id")
                .limit(to_delete_count)
            )
            await cls.filter(id__in=[dlg.id for dlg in to_delete]).delete()

    @classmethod
    async def reset_session(cls, session_id: str) -> int:
        """
        重置指定会话（软删除，标记 reset=True）

        参数:
            session_id: 会话标识

        返回:
            int: 被重置的对话数量
        """
        updated = await cls.filter(session_id=session_id, reset=False).update(
            reset=True
        )
        return updated or 0

    @classmethod
    async def clear_session(cls, session_id: str):
        """
        清空指定会话的所有对话（硬删除，包括已重置的）

        参数:
            session_id: 会话标识
        """
        await cls.filter(session_id=session_id).delete()


class ChatInterMemory(Model):
    """ChatInter 结构化长期记忆表"""

    id = fields.IntField(pk=True, generated=True, auto_increment=True)
    session_id = fields.CharField(255, index=True)
    user_id = fields.CharField(255, index=True)
    group_id = fields.CharField(255, null=True, index=True)
    memory_type = fields.CharField(64, index=True)
    key = fields.CharField(128, index=True)
    value = fields.TextField()
    confidence = fields.FloatField(default=0.0)
    source_message = fields.TextField(null=True)
    source_dialog_id = fields.IntField(null=True)
    expired = fields.BooleanField(default=False, index=True)
    last_used_time = fields.DatetimeField(null=True)
    create_time = fields.DatetimeField(auto_now_add=True, index=True)
    update_time = fields.DatetimeField(auto_now=True)

    class Meta:  # pyright: ignore [reportIncompatibleVariableOverride]
        table = "chatinter_memory"
        table_description = "ChatInter 结构化长期记忆表"

    @classmethod
    async def upsert_memory(
        cls,
        *,
        session_id: str,
        user_id: str,
        group_id: str | None,
        memory_type: str,
        key: str,
        value: str,
        confidence: float,
        source_dialog_id: int | None = None,
        source_message: str | None = None,
    ) -> "ChatInterMemory":
        existing = await cls.filter(
            session_id=session_id,
            user_id=user_id,
            memory_type=memory_type,
            key=key,
            expired=False,
        ).first()
        if existing is not None:
            if float(existing.confidence or 0.0) <= float(confidence or 0.0):
                existing.value = value
                existing.confidence = float(confidence or 0.0)
                existing.group_id = group_id
                existing.source_dialog_id = source_dialog_id
                existing.source_message = source_message
                await existing.save()
            return existing
        return await cls.create(
            session_id=session_id,
            user_id=user_id,
            group_id=group_id,
            memory_type=memory_type,
            key=key,
            value=value,
            confidence=float(confidence or 0.0),
            source_dialog_id=source_dialog_id,
            source_message=source_message,
        )

    @classmethod
    async def recall_memories(
        cls,
        *,
        session_id: str,
        user_id: str,
        group_id: str | None,
        query: str,
        limit: int = 8,
    ) -> list["ChatInterMemory"]:
        query_text = str(query or "")
        query_tokens = {
            token
            for token in query_text.replace("，", " ").replace("。", " ").split()
            if token
        }
        rows = (
            await cls.filter(user_id=user_id, expired=False)
            .order_by("-confidence", "-update_time", "-id")
            .limit(max(int(limit or 0) * 4, int(limit or 0), 1))
        )
        scoped: list[ChatInterMemory] = []
        for row in rows:
            if row.session_id != session_id and row.group_id not in {group_id, None}:
                continue
            value_text = str(row.value or "")
            key_text = f"{row.memory_type} {row.key} {value_text}"
            score = float(row.confidence or 0.0)
            if query_tokens and any(token in key_text for token in query_tokens):
                score += 0.2
            setattr(row, "_chatinter_recall_score", score)
            scoped.append(row)
        scoped.sort(
            key=lambda item: float(getattr(item, "_chatinter_recall_score", 0.0)),
            reverse=True,
        )
        return scoped[: max(int(limit or 0), 0)]
