"""Inyección de dependencias para los controladores de FastAPI."""

from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.ingest_service import IngestService
from app.application.rag_service import RAGService
from app.domain.ports import EmbeddingPort, HistoryRepositoryPort, LLMPort, VectorStorePort
from app.infrastructure.db.history_repository import PostgresHistoryRepository
from app.infrastructure.db.session import get_db_session
from app.infrastructure.embeddings.openai_embeddings import OpenAIEmbeddingAdapter
from app.infrastructure.llm.openai_llm import OpenAILLMAdapter
from app.infrastructure.vectorstore.pgvector_store import PgVectorStore


@lru_cache
def get_embedding_provider() -> EmbeddingPort:
    return OpenAIEmbeddingAdapter()


@lru_cache
def get_llm_provider() -> LLMPort:
    return OpenAILLMAdapter()


def get_vector_store(session: AsyncSession = Depends(get_db_session)) -> VectorStorePort:
    return PgVectorStore(session=session)


def get_history_repository(
    session: AsyncSession = Depends(get_db_session),
) -> HistoryRepositoryPort:
    return PostgresHistoryRepository(session=session)


def get_ingest_service(
    vector_store: VectorStorePort = Depends(get_vector_store),
    embedding_provider: EmbeddingPort = Depends(get_embedding_provider),
) -> IngestService:
    return IngestService(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )


def get_rag_service(
    vector_store: VectorStorePort = Depends(get_vector_store),
    embedding_provider: EmbeddingPort = Depends(get_embedding_provider),
    llm_provider: LLMPort = Depends(get_llm_provider),
    history_repo: HistoryRepositoryPort = Depends(get_history_repository),
) -> RAGService:
    return RAGService(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        history_repo=history_repo,
    )
