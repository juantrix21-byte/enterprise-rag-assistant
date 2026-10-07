"""Esquemas de validación Pydantic para los endpoints de la API."""

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_hash: str
    total_pages: int
    total_chunks: int
    created_at: datetime


class DocumentUploadResponse(BaseModel):
    message: str
    document: DocumentResponse


class AskRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Identificador único de la sesión conversacional",
    )
    question: str = Field(
        ..., min_length=2, max_length=1000, description="Pregunta formulada sobre los documentos"
    )


class SourceReferenceDTO(BaseModel):
    document_name: str
    page_number: int
    chunk_id: str
    snippet: str
    similarity_score: float


class AskResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    grounded: bool
    sources: list[SourceReferenceDTO]
    retrieval_count: int
    latency_ms: float


class MessageDTO(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    sources: list[SourceReferenceDTO]
    created_at: datetime


class HistoryResponse(BaseModel):
    session_id: str
    total_messages: int
    messages: list[MessageDTO]
