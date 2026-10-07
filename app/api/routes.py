"""Rutas y controladores principales de la API."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.dependencies import (
    get_history_repository,
    get_ingest_service,
    get_rag_service,
    get_vector_store,
)
from app.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentResponse,
    DocumentUploadResponse,
    HistoryResponse,
    MessageDTO,
    SourceReferenceDTO,
)
from app.application.ingest_service import IngestService
from app.application.rag_service import RAGService
from app.core.config import get_settings
from app.domain.ports import HistoryRepositoryPort, VectorStorePort

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Cargar y procesar un documento PDF",
    tags=["documents"],
)
async def upload_document(
    file: UploadFile = File(..., description="Archivo PDF para indexar en la base de conocimiento"),
    ingest_service: IngestService = Depends(get_ingest_service),
) -> DocumentUploadResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Solo se admiten archivos en formato PDF (.pdf).",
        )

    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"El tamaño del archivo supera el límite permitido de {settings.max_upload_mb} MB.",
        )

    try:
        metadata = await ingest_service.ingest_pdf(
            file_bytes=content,
            filename=file.filename,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error al procesar el documento PDF: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno durante la ingesta del documento: {e!s}",
        ) from e

    return DocumentUploadResponse(
        message="Documento procesado e indexado exitosamente.",
        document=DocumentResponse(
            id=metadata.id,
            filename=metadata.filename,
            file_hash=metadata.file_hash,
            total_pages=metadata.total_pages,
            total_chunks=metadata.total_chunks,
            created_at=metadata.created_at,
        ),
    )


@router.get(
    "/documents",
    response_model=list[DocumentResponse],
    summary="Listar documentos indexados",
    tags=["documents"],
)
async def list_documents(
    vector_store: VectorStorePort = Depends(get_vector_store),
) -> list[DocumentResponse]:
    docs = await vector_store.get_documents()
    return [
        DocumentResponse(
            id=d.id,
            filename=d.filename,
            file_hash=d.file_hash,
            total_pages=d.total_pages,
            total_chunks=d.total_chunks,
            created_at=d.created_at,
        )
        for d in docs
    ]


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Consultar al asistente RAG con trazabilidad y grounding",
    tags=["rag"],
)
async def ask_question(
    payload: AskRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> AskResponse:
    try:
        result = await rag_service.answer_question(
            session_id=payload.session_id,
            question=payload.question,
        )
    except Exception as e:
        logger.exception("Error al responder la consulta RAG: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al procesar la consulta: {e!s}",
        ) from e

    sources_dto = [
        SourceReferenceDTO(
            document_name=s.document_name,
            page_number=s.page_number,
            chunk_id=s.chunk_id,
            snippet=s.snippet,
            similarity_score=s.similarity_score,
        )
        for s in result.sources
    ]

    return AskResponse(
        session_id=result.session_id,
        question=payload.question,
        answer=result.answer,
        grounded=result.grounded,
        sources=sources_dto,
        retrieval_count=result.retrieval_count,
        latency_ms=result.latency_ms,
    )


@router.get(
    "/history/{session_id}",
    response_model=HistoryResponse,
    summary="Recuperar historial conversacional de una sesión",
    tags=["history"],
)
async def get_history(
    session_id: str,
    history_repo: HistoryRepositoryPort = Depends(get_history_repository),
) -> HistoryResponse:
    messages = await history_repo.get_messages(session_id=session_id)
    dto_list = [
        MessageDTO(
            id=m.id,
            session_id=m.session_id,
            role=m.role,
            content=m.content,
            sources=[
                SourceReferenceDTO(
                    document_name=s.document_name,
                    page_number=s.page_number,
                    chunk_id=s.chunk_id,
                    snippet=s.snippet,
                    similarity_score=s.similarity_score,
                )
                for s in m.sources
            ],
            created_at=m.created_at,
        )
        for m in messages
    ]

    return HistoryResponse(
        session_id=session_id,
        total_messages=len(dto_list),
        messages=dto_list,
    )
