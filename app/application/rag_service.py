"""Caso de uso: Consulta mediante RAG con Grounding y Trazabilidad de Fuentes."""

import logging
import time
import uuid

from app.core.config import get_settings
from app.domain.models import ChatMessage, QueryResult, SourceReference
from app.domain.ports import EmbeddingPort, HistoryRepositoryPort, LLMPort, VectorStorePort

logger = logging.getLogger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """Eres un asistente de IA especializado y riguroso para entornos educativos y de investigación (CIE LAB).
Tu objetivo es responder a las preguntas del usuario basándote EXCLUSIVAMENTE en el contexto documental proporcionado.

REGLAS ESTRICTAS DE GROUNDING:
1. Responde únicamente con hechos directamente respaldados por las fuentes citadas.
2. Si el contexto NO contiene información suficiente para responder con certeza, responde exactamente:
   "Busqué en los documentos y no encontré información suficiente para responderte esto con seguridad, y prefiero no inventarte una respuesta."
3. NUNCA inventes autores, fechas, estadísticas ni metodologías no presentes en las fuentes.
4. Cita siempre la fuente correspondiente en el texto utilizando el formato [Fuente 1], [Fuente 2], etc.
5. Mantén un tono académico, claro, profesional y estructurado.
"""


class RAGService:
    def __init__(
        self,
        vector_store: VectorStorePort,
        embedding_provider: EmbeddingPort,
        llm_provider: LLMPort,
        history_repo: HistoryRepositoryPort,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.history_repo = history_repo
        self.top_k = top_k or settings.top_k
        self.min_similarity = min_similarity or settings.min_similarity

    async def answer_question(self, session_id: str, question: str) -> QueryResult:
        start_time = time.perf_counter()
        clean_question = question.strip()

        # 1. Obtener embedding de la consulta
        query_embedding = await self.embedding_provider.get_embedding(clean_question)

        # 2. Recuperación de fragmentos relevantes (Retrieval)
        matches = await self.vector_store.search_similar(
            query_embedding=query_embedding,
            top_k=self.top_k,
            min_similarity=self.min_similarity,
        )

        sources: list[SourceReference] = []
        for chunk, score in matches:
            snippet = chunk.content[:250].replace("\n", " ").strip() + "..."
            sources.append(
                SourceReference(
                    document_name=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_id=chunk.id,
                    snippet=snippet,
                    similarity_score=round(score, 4),
                )
            )

        # 3. Control de Grounding / Fallback determinístico
        # Si no hay fragmentos relevantes que superen el umbral mínimo, no llamamos al LLM
        if not matches:
            fallback_answer = (
                "Busqué en los documentos y no encontré información suficiente para "
                "responderte esto con seguridad, y prefiero no inventarte una respuesta."
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Guardar en historial
            await self._record_history(
                session_id=session_id,
                question=clean_question,
                answer=fallback_answer,
                sources=[],
            )

            return QueryResult(
                answer=fallback_answer,
                grounded=False,
                sources=[],
                session_id=session_id,
                retrieval_count=0,
                latency_ms=round(elapsed_ms, 2),
            )

        # 4. Formateo de contexto con citas numeradas
        context_blocks: list[str] = []
        for idx, (chunk, score) in enumerate(matches, start=1):
            block = (
                f"[Fuente {idx}] Documento: {chunk.filename} (Pág. {chunk.page_number}) "
                f"(Relevancia: {score:.2f}):\n{chunk.content}"
            )
            context_blocks.append(block)

        context_str = "\n\n---\n\n".join(context_blocks)

        # 5. Obtener historial previo para contexto de conversación
        prior_messages = await self.history_repo.get_messages(session_id=session_id, limit=6)
        history_context = ""
        if prior_messages:
            history_lines = [
                f"{'Usuario' if m.role == 'user' else 'Asistente'}: {m.content}"
                for m in prior_messages[-4:]
            ]
            history_context = (
                "HISTORIAL PREVIO DE LA CONVERSACIÓN:\n" + "\n".join(history_lines) + "\n\n"
            )

        user_prompt = (
            f"{history_context}"
            f"CONTEXTO DOCUMENTAL RECUPERADO:\n{context_str}\n\n"
            f"PREGUNTA DEL USUARIO:\n{clean_question}\n\n"
            f"RESPUESTA FUNDAMENTADA:"
        )

        # 6. Generación con LLM
        raw_answer = await self.llm_provider.generate_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Guardar en historial
        await self._record_history(
            session_id=session_id,
            question=clean_question,
            answer=raw_answer,
            sources=sources,
        )

        return QueryResult(
            answer=raw_answer,
            grounded=True,
            sources=sources,
            session_id=session_id,
            retrieval_count=len(sources),
            latency_ms=round(elapsed_ms, 2),
        )

    async def _record_history(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: list[SourceReference],
    ) -> None:
        user_msg = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=question,
            sources=[],
        )
        assistant_msg = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="assistant",
            content=answer,
            sources=sources,
        )
        await self.history_repo.save_message(user_msg)
        await self.history_repo.save_message(assistant_msg)
