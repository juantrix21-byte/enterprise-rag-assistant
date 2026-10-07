"""Servicio de extracción de PDF y particionado inteligente (Chunking).

Estrategia de Chunking:
1. Extracción página a página con preservación estricta de metadatos (número de página 1-indexed).
2. Segmentación recursiva por párrafos y oraciones respetando límites naturales de texto.
3. Solape (overlap) entre fragmentos contiguos para mantener continuidad semántica.
"""

import hashlib
import io
import re
import uuid

from pypdf import PdfReader

from app.domain.models import DocumentChunk


def calculate_file_hash(file_bytes: bytes) -> str:
    """Calcula el hash SHA-256 del contenido binario para idempotencia."""
    return hashlib.sha256(file_bytes).hexdigest()


class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_pages(self, file_bytes: bytes) -> list[tuple[int, str]]:
        """Extrae texto de cada página del PDF preservando el número de página."""
        reader = PdfReader(io.BytesIO(file_bytes))
        pages: list[tuple[int, str]] = []

        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            # Normalización básica de espacios en blanco
            text = re.sub(r"[ \t]+", " ", text).strip()
            if text:
                pages.append((page_idx + 1, text))

        return pages

    def split_text_into_chunks(
        self,
        text: str,
        page_number: int,
        document_id: str,
        filename: str,
        start_chunk_index: int,
    ) -> list[DocumentChunk]:
        """Divide el texto de una página en fragmentos con solape sin cortar palabras."""
        if len(text) <= self.chunk_size:
            chunk_id = f"{document_id}_{start_chunk_index}"
            return [
                DocumentChunk(
                    id=chunk_id,
                    document_id=document_id,
                    filename=filename,
                    chunk_index=start_chunk_index,
                    page_number=page_number,
                    content=text,
                    metadata={"char_length": len(text), "page": page_number},
                )
            ]

        # Separar por párrafos dobles o saltos de línea
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
        chunks: list[DocumentChunk] = []
        current_text = ""
        current_index = start_chunk_index

        for para in paragraphs:
            if not current_text:
                current_text = para
            elif len(current_text) + len(para) + 1 <= self.chunk_size:
                current_text += "\n" + para
            else:
                # El fragmento actual está completo
                chunk_id = f"{document_id}_{current_index}"
                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        document_id=document_id,
                        filename=filename,
                        chunk_index=current_index,
                        page_number=page_number,
                        content=current_text,
                        metadata={"char_length": len(current_text), "page": page_number},
                    )
                )
                current_index += 1

                # Mantener solape (overlap) tomando el final del fragmento anterior
                overlap_text = (
                    current_text[-self.chunk_overlap :]
                    if len(current_text) > self.chunk_overlap
                    else ""
                )
                current_text = (overlap_text + " " + para).strip()

        if current_text:
            chunk_id = f"{document_id}_{current_index}"
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    document_id=document_id,
                    filename=filename,
                    chunk_index=current_index,
                    page_number=page_number,
                    content=current_text,
                    metadata={"char_length": len(current_text), "page": page_number},
                )
            )

        return chunks

    def process_pdf(
        self,
        file_bytes: bytes,
        filename: str,
        document_id: str | None = None,
    ) -> tuple[str, int, list[DocumentChunk]]:
        """Extrae páginas y genera la lista total de chunks para el documento."""
        doc_id = document_id or str(uuid.uuid4())
        pages = self.extract_pages(file_bytes)
        total_pages = len(pages)
        all_chunks: list[DocumentChunk] = []

        chunk_idx = 0
        for page_num, text in pages:
            page_chunks = self.split_text_into_chunks(
                text=text,
                page_number=page_num,
                document_id=doc_id,
                filename=filename,
                start_chunk_index=chunk_idx,
            )
            all_chunks.extend(page_chunks)
            chunk_idx += len(page_chunks)

        return doc_id, total_pages, all_chunks
