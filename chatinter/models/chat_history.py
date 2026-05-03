from tortoise import fields
from tortoise.expressions import F

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
    scope = fields.CharField(32, default="user", index=True)
    thread_id = fields.CharField(64, null=True, index=True)
    topic_key = fields.CharField(255, default="", index=True)
    participants = fields.TextField(default="")
    source_message = fields.TextField(null=True)
    source_dialog_id = fields.IntField(null=True)
    expired = fields.BooleanField(default=False, index=True)
    last_used_time = fields.DatetimeField(null=True)
    recall_count = fields.IntField(default=0)
    create_time = fields.DatetimeField(auto_now_add=True, index=True)
    update_time = fields.DatetimeField(auto_now=True)

    class Meta:  # pyright: ignore [reportIncompatibleVariableOverride]
        table = "chatinter_memory"
        table_description = "ChatInter 结构化长期记忆表"

    @classmethod
    async def _run_script(cls):
        return [
            "ALTER TABLE chatinter_memory ADD COLUMN scope VARCHAR(32) DEFAULT 'user';",
            "ALTER TABLE chatinter_memory ADD COLUMN thread_id VARCHAR(64);",
            "ALTER TABLE chatinter_memory ADD COLUMN topic_key "
            "VARCHAR(255) DEFAULT '';",
            "ALTER TABLE chatinter_memory ADD COLUMN participants TEXT DEFAULT '';",
            "ALTER TABLE chatinter_memory ADD COLUMN recall_count INT DEFAULT 0;",
        ]

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
        scope: str = "user",
        thread_id: str | None = None,
        topic_key: str = "",
        participants: str = "",
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
                existing.scope = scope
                existing.thread_id = thread_id
                existing.topic_key = topic_key
                existing.participants = participants
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
            scope=scope,
            thread_id=thread_id,
            topic_key=topic_key,
            participants=participants,
            source_dialog_id=source_dialog_id,
            source_message=source_message,
        )

    @classmethod
    async def mark_recalled(cls, memory_ids: list[int]) -> None:
        if not memory_ids:
            return
        try:
            await cls.filter(id__in=memory_ids).update(
                recall_count=F("recall_count") + 1
            )
        except Exception:
            return

    @classmethod
    async def recall_memories(
        cls,
        *,
        session_id: str,
        user_id: str,
        group_id: str | None,
        query: str,
        limit: int = 8,
        thread_id: str | None = None,
        topic_key: str = "",
        participants: tuple[str, ...] = (),
        addressee_user_id: str | None = None,
    ) -> list["ChatInterMemory"]:
        query_text = str(query or "")
        query_tokens = {
            token
            for token in query_text.replace("，", " ").replace("。", " ").split()
            if token
        }
        participant_set = {str(item) for item in participants if str(item)}
        structured_limit = max(int(limit or 0) * 8, int(limit or 0), 1)
        rows = (
            await cls.filter(expired=False)
            .order_by("-confidence", "-update_time", "-id")
            .limit(structured_limit)
        )
        scoped: list[ChatInterMemory] = []
        for row in rows:
            row_user_id = str(row.user_id or "")
            row_group_id = str(row.group_id or "") if row.group_id is not None else None
            row_scope = str(row.scope or "")
            row_thread_id = str(row.thread_id or "")
            row_topic_key = str(row.topic_key or "")
            row_participants = {
                item for item in str(row.participants or "").split(",") if item
            }
            if row_user_id != user_id:
                if not (
                    (
                        row_scope in {"group", "thread"}
                        or row.memory_type == "group_digest"
                    )
                    and group_id
                    and row_group_id == group_id
                ):
                    continue
            if row.session_id != session_id and row_group_id not in {group_id, None}:
                continue
            value_text = str(row.value or "")
            key_text = f"{row.memory_type} {row.key} {value_text}"
            score = float(row.confidence or 0.0)
            if row_scope == "thread" or row.memory_type == "group_digest":
                score -= 0.08
            if thread_id and row_thread_id == thread_id:
                score += 0.45
            elif thread_id and row_thread_id and row_thread_id != thread_id:
                score -= 0.28
            if topic_key and row_topic_key and row_topic_key == topic_key:
                score += 0.18
            if participant_set and row_participants:
                overlap = len(participant_set & row_participants)
                if overlap:
                    score += min(overlap, 3) * 0.12
                elif row_scope == "thread":
                    score -= 0.1
            if addressee_user_id and addressee_user_id in row_participants:
                score += 0.18
            if query_tokens and any(token in key_text for token in query_tokens):
                score += 0.2
            if row_user_id == user_id:
                score += 0.08
            setattr(row, "_chatinter_recall_score", score)
            scoped.append(row)
        scoped.sort(
            key=lambda item: float(getattr(item, "_chatinter_recall_score", 0.0)),
            reverse=True,
        )
        selected = scoped[: max(int(limit or 0), 0)]
        await cls.mark_recalled([int(row.id or 0) for row in selected if row.id])
        return selected


class ChatInterPersonProfile(Model):
    """ChatInter 群友身份档案表

    用于把群聊里的昵称、别名和稳定 user_id 对齐，避免把“认识人”
    依赖在一次性 prompt 里。
    """

    id = fields.IntField(pk=True, generated=True, auto_increment=True)
    user_id = fields.CharField(255, index=True)
    group_id = fields.CharField(255, null=True, index=True)
    nickname = fields.CharField(255, default="")
    group_card = fields.CharField(255, default="")
    aliases = fields.TextField(default="")
    alias_weights = fields.TextField(default="")
    alias_sources = fields.TextField(default="")
    known_facts = fields.TextField(default="")
    relationship = fields.CharField(255, default="")
    conflict_state = fields.CharField(64, default="")
    confidence = fields.FloatField(default=0.0)
    last_seen = fields.DatetimeField(auto_now=True)
    create_time = fields.DatetimeField(auto_now_add=True, index=True)
    update_time = fields.DatetimeField(auto_now=True)

    class Meta:  # pyright: ignore [reportIncompatibleVariableOverride]
        table = "chatinter_person_profile"
        table_description = "ChatInter 群友身份档案表"

    @classmethod
    async def _run_script(cls):
        return [
            "ALTER TABLE chatinter_person_profile "
            "ADD COLUMN alias_weights TEXT DEFAULT '';",
            "ALTER TABLE chatinter_person_profile "
            "ADD COLUMN alias_sources TEXT DEFAULT '';",
            "ALTER TABLE chatinter_person_profile "
            "ADD COLUMN conflict_state VARCHAR(64) DEFAULT '';",
        ]


class ChatInterThread(Model):
    """ChatInter 群聊短期话题线程表"""

    id = fields.IntField(pk=True, generated=True, auto_increment=True)
    thread_id = fields.CharField(64, index=True)
    group_id = fields.CharField(255, null=True, index=True)
    participants = fields.TextField(default="")
    topic_key = fields.CharField(255, default="", index=True)
    topic_summary = fields.TextField(default="")
    last_message = fields.TextField(default="")
    source = fields.CharField(64, default="")
    confidence = fields.FloatField(default=0.0)
    archived = fields.BooleanField(default=False, index=True)
    last_active = fields.DatetimeField(auto_now=True, index=True)
    create_time = fields.DatetimeField(auto_now_add=True, index=True)
    update_time = fields.DatetimeField(auto_now=True)

    class Meta:  # pyright: ignore [reportIncompatibleVariableOverride]
        table = "chatinter_thread"
        table_description = "ChatInter 群聊短期话题线程表"


class ChatInterThreadMessage(Model):
    """ChatInter 消息到话题线程的旁路索引表"""

    id = fields.IntField(pk=True, generated=True, auto_increment=True)
    thread_id = fields.CharField(64, index=True)
    group_id = fields.CharField(255, null=True, index=True)
    message_id = fields.CharField(255, null=True, index=True)
    dialog_id = fields.IntField(null=True, index=True)
    user_id = fields.CharField(255, default="", index=True)
    message_preview = fields.TextField(default="")
    create_time = fields.DatetimeField(auto_now_add=True, index=True)

    class Meta:  # pyright: ignore [reportIncompatibleVariableOverride]
        table = "chatinter_thread_message"
        table_description = "ChatInter 消息到话题线程的旁路索引表"
