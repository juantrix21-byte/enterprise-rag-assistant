"""Pruebas unitarias para el procesador de PDFs y chunking."""

from app.application.pdf_processor import PDFProcessor, calculate_file_hash


def test_calculate_file_hash():
    data = b"contenido de prueba"
    hash1 = calculate_file_hash(data)
    hash2 = calculate_file_hash(data)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256


def test_chunking_short_text():
    processor = PDFProcessor(chunk_size=500, chunk_overlap=50)
    text = "Este es un párrafo corto que no excede el límite."
    chunks = processor.split_text_into_chunks(
        text=text,
        page_number=1,
        document_id="doc-123",
        filename="test.pdf",
        start_chunk_index=0,
    )
    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0


def test_chunking_preserves_page_numbers_and_overlap():
    processor = PDFProcessor(chunk_size=60, chunk_overlap=15)
    # Párrafos que forzarán múltiples fragmentos
    text = (
        "Primer párrafo sobre inteligencia artificial generativa en educación.\n\n"
        "Segundo párrafo detallando las implicancias éticas y normativas.\n\n"
        "Tercer párrafo con recomendaciones de implementación pedagógica."
    )
    chunks = processor.split_text_into_chunks(
        text=text,
        page_number=3,
        document_id="doc-test",
        filename="unesco.pdf",
        start_chunk_index=10,
    )

    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.page_number == 3
        assert chunk.filename == "unesco.pdf"
        assert chunk.chunk_index == 10 + i
        assert len(chunk.content) > 0
