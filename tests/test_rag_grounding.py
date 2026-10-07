"""Pruebas unitarias para la lógica de RAG, Grounding estricto y prevención de alucinaciones."""

import pytest

from app.application.rag_service import RAGService
from app.domain.models import ChatMessage, DocumentChunk, DocumentMetadata
from app.domain.ports import EmbeddingPort, HistoryRepositoryPort, LLMPort, VectorStorePort


class FakeEmbeddingAdapter(EmbeddingPort):
    async def get_embedding(self, text: str) -> list[float]:
        return [0.1] * 1536

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1536 for _ in texts]


class FakeLLMAdapter(LLMPort):
    def __init__(self, response_text: str = "Respuesta simulada basada en el contexto.") -> None:
        self.response_text = response_text
        self.call_count = 0
        self.last_user_prompt = ""

    async def generate_response(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.0
    ) -> str:
        self.call_count += 1
        self.last_user_prompt = user_prompt
        return self.response_text


class InMemoryVectorStore(VectorStorePort):
    def __init__(self, matches: list[tuple[DocumentChunk, float]] | None = None) -> None:
        self.matches = matches or []

    async def save_chunks(self, chunks: list[DocumentChunk]) -> None:
        pass

    async def search_similar(
        self, query_embedding: list[float], top_k: int, min_similarity: float
    ) -> list[tuple[DocumentChunk, float]]:
        # Filtra por min_similarity
        return [(c, s) for c, s in self.matches if s >= min_similarity][:top_k]

    async def document_exists(self, file_hash: str) -> bool:
        return False

    async def save_document_metadata(self, metadata: DocumentMetadata) -> None:
        pass

    async def get_documents(self) -> list[DocumentMetadata]:
        return []


class InMemoryHistoryRepository(HistoryRepositoryPort):
    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    async def save_message(self, message: ChatMessage) -> None:
        self.messages.append(message)

    async def get_messages(self, session_id: str, limit: int = 50) -> list[ChatMessage]:
        return [m for m in self.messages if m.session_id == session_id][:limit]


@pytest.mark.asyncio
async def test_rag_fallback_when_no_relevant_sources():
    """Verifica que si la similitud no alcanza el umbral, NO se llama al LLM y se responde formalmente."""
    fake_llm = FakeLLMAdapter()
    vstore = InMemoryVectorStore(matches=[])  # Sin coincidencias relevantes
    history = InMemoryHistoryRepository()
    rag_svc = RAGService(
        vector_store=vstore,
        embedding_provider=FakeEmbeddingAdapter(),
        llm_provider=fake_llm,
        history_repo=history,
        min_similarity=0.45,
    )

    result = await rag_svc.answer_question(
        session_id="test-session", question="¿Quién descubrió América?"
    )

    assert result.grounded is False
    assert result.retrieval_count == 0
    assert "prefiero no inventarte una respuesta" in result.answer
    assert fake_llm.call_count == 0  # CRÍTICO: Cero llamadas al LLM = Cero alucinación y Cero costo


@pytest.mark.asyncio
async def test_rag_answers_with_grounding_and_sources():
    """Verifica que con fragmentos relevantes, se llama al LLM y se preservan las citas con número de página."""
    fake_llm = FakeLLMAdapter(
        response_text="Según [Fuente 1], la UNESCO recomienda regular la edad mínima."
    )
    chunk = DocumentChunk(
        id="chunk-1",
        document_id="doc-1",
        filename="guia_unesco.pdf",
        chunk_index=0,
        page_number=14,
        content="La UNESCO propone un límite de edad mínima de 13 años para el uso de herramientas de IA.",
    )
    vstore = InMemoryVectorStore(matches=[(chunk, 0.85)])
    history = InMemoryHistoryRepository()

    rag_svc = RAGService(
        vector_store=vstore,
        embedding_provider=FakeEmbeddingAdapter(),
        llm_provider=fake_llm,
        history_repo=history,
        min_similarity=0.35,
    )

    result = await rag_svc.answer_question(
        session_id="session-edu",
        question="¿Cuál es la edad mínima sugerida por UNESCO?",
    )

    assert result.grounded is True
    assert result.retrieval_count == 1
    assert fake_llm.call_count == 1
    assert len(result.sources) == 1
    assert result.sources[0].page_number == 14
    assert result.sources[0].document_name == "guia_unesco.pdf"
    assert result.sources[0].similarity_score == 0.85
    assert len(history.messages) == 2  # Usuario + Asistente
