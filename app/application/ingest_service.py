"""Caso de uso: Ingesta y vectorización de documentos PDF."""

import logging

from app.application.pdf_processor import PDFProcessor, calculate_file_hash
from app.domain.models import DocumentMetadata
from app.domain.ports import EmbeddingPort, VectorStorePort

logger = logging.getLogger(__name__)


class IngestService:
    def __init__(
        self,
        vector_store: VectorStorePort,
        embedding_provider: EmbeddingPort,
        pdf_processor: PDFProcessor | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.pdf_processor = pdf_processor or PDFProcessor()

    async def ingest_pdf(
        self,
        file_bytes: bytes,
        filename: str,
        force_reindex: bool = False,
    ) -> DocumentMetadata:
        file_hash = calculate_file_hash(file_bytes)

        # Idempotencia: Verificar si el documento ya fue procesado
        if not force_reindex and await self.vector_store.document_exists(file_hash):
            logger.info(
                "El documento %s con hash %s ya existe en la base vectorial.", filename, file_hash
            )
            # Buscar metadatos existentes
            existing_docs = await self.vector_store.get_documents()
            for doc in existing_docs:
                if doc.file_hash == file_hash:
                    return doc

        # 1. Extracción y chunking
        doc_id, total_pages, chunks = self.pdf_processor.process_pdf(
            file_bytes=file_bytes,
            filename=filename,
        )

        if not chunks:
            raise ValueError(f"El documento '{filename}' no contiene texto extraíble.")

        logger.info(
            "Documento procesado: %s (%d páginas, %d chunks). Generando embeddings...",
            filename,
            total_pages,
            len(chunks),
        )

        # 2. Generación de embeddings por lotes
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedding_provider.get_embeddings(texts)

        for chunk, emb in zip(chunks, embeddings, strict=True):
            chunk.embedding = emb

        # 3. Persistencia en Vector Store
        metadata = DocumentMetadata(
            id=doc_id,
            filename=filename,
            file_hash=file_hash,
            total_pages=total_pages,
            total_chunks=len(chunks),
        )
        await self.vector_store.save_document_metadata(metadata)
        await self.vector_store.save_chunks(chunks)

        logger.info("Ingesta completada exitosamente para %s.", filename)
        return metadata
