from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class DocumentMetadata(BaseModel):
    id: str
    filename: str
    file_hash: str
    total_pages: int
    total_chunks: int
    created_at: datetime = Field(default_factory=utc_now)


class DocumentChunk(BaseModel):
    id: str
    document_id: str
    filename: str
    chunk_index: int
    page_number: int
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class SourceReference(BaseModel):
    document_name: str
    page_number: int
    chunk_id: str
    snippet: str
    similarity_score: float


class ChatMessage(BaseModel):
    id: str
    session_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    sources: list[SourceReference] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class QueryResult(BaseModel):
    answer: str
    grounded: bool
    sources: list[SourceReference]
    session_id: str
    retrieval_count: int
    latency_ms: float
