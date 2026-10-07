"""Pruebas de integración para los endpoints de la API FastAPI."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_ingest_service, get_rag_service
from app.domain.models import DocumentMetadata, QueryResult, SourceReference
from app.main import app


class MockRAGService:
    async def answer_question(self, session_id: str, question: str) -> QueryResult:
        if "receta" in question.lower():
            return QueryResult(
                answer=(
                    "Busqué en los documentos y no encontré información suficiente para "
                    "responderte esto con seguridad, y prefiero no inventarte una respuesta."
                ),
                grounded=False,
                sources=[],
                session_id=session_id,
                retrieval_count=0,
                latency_ms=15.2,
            )
        return QueryResult(
            answer="EdGPT es un modelo adaptado para educación según [Fuente 1].",
            grounded=True,
            sources=[
                SourceReference(
                    document_name="guia.pdf",
                    page_number=16,
                    chunk_id="chunk-16",
                    snippet="EdGPT entrenado para educación...",
                    similarity_score=0.82,
                )
            ],
            session_id=session_id,
            retrieval_count=1,
            latency_ms=210.5,
        )


class MockIngestService:
    async def ingest_pdf(self, file_bytes: bytes, filename: str) -> DocumentMetadata:
        return DocumentMetadata(
            id="mock-doc-1",
            filename=filename,
            file_hash="mock-hash-123",
            total_pages=5,
            total_chunks=12,
        )


@pytest.fixture
def override_services():
    app.dependency_overrides[get_rag_service] = lambda: MockRAGService()
    app.dependency_overrides[get_ingest_service] = lambda: MockIngestService()
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_ask_endpoint_success(override_services):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"session_id": "test-session", "question": "¿Qué es EdGPT?"}
        res = await client.post("/ask", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["grounded"] is True
        assert "EdGPT" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["page_number"] == 16


@pytest.mark.asyncio
async def test_ask_endpoint_fallback(override_services):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"session_id": "test-session", "question": "¿Cómo hacer una receta de lomo?"}
        res = await client.post("/ask", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["grounded"] is False
        assert len(data["sources"]) == 0


@pytest.mark.asyncio
async def test_upload_non_pdf_fails(override_services):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("datos.txt", b"Texto plano", "text/plain")}
        res = await client.post("/documents", files=files)
        assert res.status_code == 400
        assert "PDF" in res.json()["detail"]


@pytest.mark.asyncio
async def test_upload_pdf_success(override_services):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("manual.pdf", b"%PDF-1.4...", "application/pdf")}
        res = await client.post("/documents", files=files)
        assert res.status_code == 201
        data = res.json()
        assert data["document"]["filename"] == "manual.pdf"
        assert data["document"]["total_pages"] == 5
