"""Puertos (interfaces abstractas) según los principios de Clean Architecture.

La lógica de negocio dependerá exclusivamente de estos puertos, permitiendo cambiar
proveedores de IA (OpenAI, Gemini, Ollama) o de bases de datos sin modificar el dominio.
"""

from typing import Protocol

from app.domain.models import ChatMessage, DocumentChunk, DocumentMetadata


class EmbeddingPort(Protocol):
    """Puerto para la generación de representaciones vectoriales."""

    async def get_embedding(self, text: str) -> list[float]: ...

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]: ...


class LLMPort(Protocol):
    """Puerto para la inferencia con modelos de lenguaje."""

    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str: ...


class VectorStorePort(Protocol):
    """Puerto para el almacenamiento y búsqueda semántica de fragmentos de documentos."""

    async def save_chunks(self, chunks: list[DocumentChunk]) -> None: ...

    async def search_similar(
        self,
        query_embedding: list[float],
        top_k: int,
        min_similarity: float,
    ) -> list[tuple[DocumentChunk, float]]:
        """Retorna una lista de tuplas (chunk, similarity_score)."""
        ...

    async def document_exists(self, file_hash: str) -> bool: ...

    async def save_document_metadata(self, metadata: DocumentMetadata) -> None: ...

    async def get_documents(self) -> list[DocumentMetadata]: ...


class HistoryRepositoryPort(Protocol):
    """Puerto para la persistencia del historial conversacional."""

    async def save_message(self, message: ChatMessage) -> None: ...

    async def get_messages(self, session_id: str, limit: int = 50) -> list[ChatMessage]: ...
