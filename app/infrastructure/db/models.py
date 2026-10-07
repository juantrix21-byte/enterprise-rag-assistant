from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship

from app.core.config import get_settings

settings = get_settings()


class Base(DeclarativeBase):
    pass


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class DocumentRecord(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True)
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), unique=True, index=True, nullable=False)
    total_pages = Column(Integer, nullable=False, default=0)
    total_chunks = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    chunks = relationship("ChunkRecord", back_populates="document", cascade="all, delete-orphan")


class ChunkRecord(Base):
    __tablename__ = "document_chunks"

    id = Column(String(64), primary_key=True)
    document_id = Column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename = Column(String(255), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    metadata_json = Column(JSONB, default=dict)
    embedding = Column(Vector(settings.embedding_dim), nullable=False)

    document = relationship("DocumentRecord", back_populates="chunks")


class MessageRecord(Base):
    __tablename__ = "chat_history"

    id = Column(String(36), primary_key=True)
    session_id = Column(String(64), index=True, nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    sources_json = Column(JSONB, default=list)
    created_at = Column(DateTime, default=utc_now, index=True, nullable=False)
