"""Punto de entrada principal de la aplicación FastAPI."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.infrastructure.db.session import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("enterprise-rag")


async def _auto_seed_initial_document():
    import asyncio
    from pathlib import Path

    from app.application.ingest_service import IngestService
    from app.core.config import get_settings
    from app.infrastructure.db.session import async_session_maker
    from app.infrastructure.embeddings.openai_embeddings import OpenAIEmbeddingAdapter
    from app.infrastructure.vectorstore.pgvector_store import PgVectorStore

    settings = get_settings()
    if not settings.openai_api_key or "your_openai" in settings.openai_api_key:
        return

    doc_path = Path("docs/unesco.pdf")
    if not doc_path.exists():
        return

    # Breve espera para asegurar que la conexión de BD esté completamente asentada
    await asyncio.sleep(2)

    try:
        async with async_session_maker() as session:
            vstore = PgVectorStore(session)
            docs = await vstore.get_documents()
            if not docs:
                logger.info(
                    "[Bootstrap] Base de datos vacía detectada. Indexando automáticamente %s...",
                    doc_path.name,
                )
                emb = OpenAIEmbeddingAdapter()
                svc = IngestService(vector_store=vstore, embedding_provider=emb)
                with open(doc_path, "rb") as f:
                    content = f.read()
                meta = await svc.ingest_pdf(
                    file_bytes=content, filename="unesco_guia_iagen_educacion.pdf"
                )
                logger.info(
                    "[Bootstrap] Auto-indexación completada: %d páginas, %d chunks.",
                    meta.total_pages,
                    meta.total_chunks,
                )
    except Exception as e:
        logger.warning("[Bootstrap] No se pudo completar el auto-seed inicial: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    logger.info("Iniciando Enterprise RAG Assistant...")
    try:
        await init_db()
        logger.info("Base de datos y extensión pgvector inicializadas con éxito.")
        asyncio.create_task(_auto_seed_initial_document())
    except Exception as e:
        logger.warning(
            "No se pudo conectar a la base de datos al inicio (%s). "
            "Asegúrate de que PostgreSQL/pgvector esté en ejecución.",
            e,
        )
    yield
    logger.info("Deteniendo Enterprise RAG Assistant...")


app = FastAPI(
    title="Enterprise RAG Assistant — CIE LAB",
    description=(
        "Asistente inteligente basado en Retrieval-Augmented Generation (RAG) con "
        "grounding estricto, trazabilidad por página/chunk y arquitectura desacoplada."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS para permitir pruebas cruzadas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Excepción no controlada en %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Ha ocurrido un error interno en el servidor.", "error": str(exc)},
    )


# Registrar rutas de API
app.include_router(router)

# Servir archivos estáticos y Single Page Application
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse("static/index.html")


@app.get("/health", tags=["system"], summary="Verificar salud del servicio")
async def health():
    return {"status": "ok", "service": "enterprise-rag-assistant", "version": "1.0.0"}
