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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando Enterprise RAG Assistant...")
    try:
        await init_db()
        logger.info("Base de datos y extensión pgvector inicializadas con éxito.")
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
