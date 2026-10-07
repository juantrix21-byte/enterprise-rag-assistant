"""Repositorio para la persistencia del historial conversacional en PostgreSQL."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import ChatMessage, SourceReference
from app.domain.ports import HistoryRepositoryPort
from app.infrastructure.db.models import MessageRecord


class PostgresHistoryRepository(HistoryRepositoryPort):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save_message(self, message: ChatMessage) -> None:
        sources_payload = [s.model_dump() for s in message.sources]
        record = MessageRecord(
            id=message.id,
            session_id=message.session_id,
            role=message.role,
            content=message.content,
            sources_json=sources_payload,
            created_at=message.created_at,
        )
        self.session.add(record)
        await self.session.commit()

    async def get_messages(self, session_id: str, limit: int = 50) -> list[ChatMessage]:
        stmt = (
            select(MessageRecord)
            .where(MessageRecord.session_id == session_id)
            .order_by(MessageRecord.created_at.asc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        messages: list[ChatMessage] = []

        for row in result.scalars().all():
            sources = [SourceReference(**s) for s in (row.sources_json or [])]
            messages.append(
                ChatMessage(
                    id=row.id,
                    session_id=row.session_id,
                    role=row.role,
                    content=row.content,
                    sources=sources,
                    created_at=row.created_at,
                )
            )

        return messages
