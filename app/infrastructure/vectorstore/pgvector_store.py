"""Adaptador de Vector Store usando PostgreSQL con la extensión pgvector."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import DocumentChunk, DocumentMetadata
from app.domain.ports import VectorStorePort
from app.infrastructure.db.models import ChunkRecord, DocumentRecord


class PgVectorStore(VectorStorePort):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def document_exists(self, file_hash: str) -> bool:
        stmt = select(DocumentRecord.id).where(DocumentRecord.file_hash == file_hash)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def save_document_metadata(self, metadata: DocumentMetadata) -> None:
        record = DocumentRecord(
            id=metadata.id,
            filename=metadata.filename,
            file_hash=metadata.file_hash,
            total_pages=metadata.total_pages,
            total_chunks=metadata.total_chunks,
            created_at=metadata.created_at,
        )
        self.session.add(record)
        await self.session.commit()

    async def save_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return

        records = [
            ChunkRecord(
                id=c.id,
                document_id=c.document_id,
                filename=c.filename,
                chunk_index=c.chunk_index,
                page_number=c.page_number,
                content=c.content,
                metadata_json=c.metadata,
                embedding=c.embedding,
            )
            for c in chunks
        ]
        self.session.add_all(records)
        await self.session.commit()

    async def search_similar(
        self,
        query_embedding: list[float],
        top_k: int,
        min_similarity: float,
    ) -> list[tuple[DocumentChunk, float]]:
        # En pgvector, cosine_distance oscila entre 0 (idénticos) y 2 (opuestos)
        # Cosine similarity = 1.0 - cosine_distance
        distance_col = ChunkRecord.embedding.cosine_distance(query_embedding)
        similarity_expr = 1.0 - distance_col

        stmt = (
            select(ChunkRecord, similarity_expr.label("similarity"))
            .where(similarity_expr >= min_similarity)
            .order_by(distance_col.asc())
            .limit(top_k)
        )

        result = await self.session.execute(stmt)
        matches: list[tuple[DocumentChunk, float]] = []

        for row in result.all():
            chunk_rec = row[0]
            sim = float(row[1])
            chunk = DocumentChunk(
                id=chunk_rec.id,
                document_id=chunk_rec.document_id,
                filename=chunk_rec.filename,
                chunk_index=chunk_rec.chunk_index,
                page_number=chunk_rec.page_number,
                content=chunk_rec.content,
                metadata=chunk_rec.metadata_json or {},
            )
            matches.append((chunk, sim))

        return matches

    async def get_documents(self) -> list[DocumentMetadata]:
        stmt = select(DocumentRecord).order_by(DocumentRecord.created_at.desc())
        result = await self.session.execute(stmt)
        docs = []
        for r in result.scalars().all():
            docs.append(
                DocumentMetadata(
                    id=r.id,
                    filename=r.filename,
                    file_hash=r.file_hash,
                    total_pages=r.total_pages,
                    total_chunks=r.total_chunks,
                    created_at=r.created_at,
                )
            )
        return docs
