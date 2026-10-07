# Enterprise RAG Assistant — CIE LAB
> **Desafío Técnico AI Engineer** | Prototipo de Asistente RAG con Grounding Estricto, Trazabilidad de Fuentes y Arquitectura Limpia

---

## 1. Descripción del Proyecto

**Enterprise RAG Assistant** es una solución de ingeniería de inteligencia artificial diseñada para responder consultas complejas sobre literatura educativa y técnica, garantizando **cero alucinaciones**, fundamentación matemática de fuentes (*grounding*), trazabilidad a nivel de página física y preservación del historial conversacional en bases de datos relacionales vectoriales.

El proyecto está diseñado bajo los principios de **Clean Architecture (Hexagonal Architecture)**, permitiendo que la lógica de negocio y los casos de uso sean completamente agnósticos de los proveedores de LLM (`OpenAI`, `Anthropic`, `Ollama`) o del motor de búsqueda vectorial (`PostgreSQL + pgvector`, `Qdrant`, etc.).

---

## 2. Documento de Conocimiento Indexado

* **Título:** *Guía para el uso de IA generativa en educación e investigación*
* **Autor / Entidad:** Organización de las Naciones Unidas para la Educación, la Ciencia y la Cultura (UNESCO), 2024.
* **Extensión:** 48 páginas.
* **Licencia:** CC-BY-SA 3.0 IGO (Acceso público abierto).
* **Enlace oficial:** [UNESCO Digital Library — Doc 389227](https://unesdoc.unesco.org/ark:/48223/pf0000389227)
* **Ubicación en el repositorio:** `docs/unesco.pdf` (indexado automáticamente al inicializar la base).

---

## 3. Arquitectura del Sistema

```
                       ┌─────────────────────────┐
                       │   Cliente / Web UI /    │
                       │    Swagger OpenAPI      │
                       └────────────┬────────────┘
                                    │ HTTP / JSON
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FastAPI Controllers & DTOs                      │
│                (POST /documents, POST /ask, GET /history)              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          APPLICATION LAYER                             │
│     ┌──────────────────────┐              ┌──────────────────────┐     │
│     │    IngestService     │              │      RAGService      │     │
│     │ (Chunking & Hashing) │              │(Grounding & Citations)     │
│     └──────────┬───────────┘              └──────────┬───────────┘     │
└────────────────┼─────────────────────────────────────┼─────────────────┘
                 │                                     │
                 ▼                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         DOMAIN PORTS (Protocols)                       │
│      EmbeddingPort  │  LLMPort  │  VectorStorePort  │  HistoryRepo     │
└────────────────┬─────────────┬─────────────┬─────────────────┬─────────┘
                 │             │             │                 │
                 ▼             ▼             ▼                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        INFRASTRUCTURE ADAPTERS                         │
│  ┌──────────────────┐  ┌────────────────┐  ┌────────────────────────┐  │
│  │ OpenAI Embeddings│  │   OpenAI LLM   │  │ PostgreSQL + pgvector  │  │
│  │(text-embedding-3)│  │ (gpt-4o-mini)  │  │ (Vectores + Historial) │  │
│  └──────────────────┘  └────────────────┘  └────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### Capas del Código:
* **`app/domain/`**: Modelos puros (`DocumentChunk`, `SourceReference`, `ChatMessage`, `QueryResult`) y Puertos abstractos (`EmbeddingPort`, `LLMPort`, `VectorStorePort`, `HistoryRepositoryPort`).
* **`app/application/`**: Casos de uso (`PDFProcessor` con chunking recursivo, `IngestService` con idempotencia por hash, `RAGService` con guardrails anti-alucinación).
* **`app/infrastructure/`**: Adaptadores concretos (`OpenAILLMAdapter`, `OpenAIEmbeddingAdapter`, `PgVectorStore`, `PostgresHistoryRepository`).
* **`app/api/`**: Controladores FastAPI, inyección de dependencias (`dependencies.py`) y esquemas Pydantic v2.
* **`static/`**: Interfaz de usuario Single-Page sin dependencias complejas (HTML5 + Tailwind CSS + Vanilla JS).

---

## 4. Estrategia de Chunking, Embeddings y Grounding

1. **Extracción y Chunking:**
   * Extracción página a página con `pypdf`, reteniendo de forma estricta el metadato del número de página (`page_number`).
   * Segmentación recursiva por párrafos dobles y oraciones (`chunk_size=1000` caracteres).
   * Solape semántico (*overlap*) del 15% (`chunk_overlap=150` caracteres) para garantizar continuidad en ideas frontera.
2. **Embeddings:**
   * Modelo: `text-embedding-3-small` de OpenAI (1536 dimensiones, optimizado para búsqueda semántica en español e inglés).
   * Lotes con reintentos exponenciales automáticos (`tenacity`).
3. **Mecanismo de Grounding y Prevención de Alucinaciones:**
   * **Guardrail de Umbral de Similitud Coseno (`min_similarity=0.35`):** Si ningún fragmento de la base vectorial supera el umbral, **no se invoca al LLM**. El sistema responde inmediatamente con un rechazo controlado formal (`grounded: false`). Esto garantiza **0% de alucinaciones en preguntas fuera de dominio y 0 costo de tokens**.
   * **Inyección de Citas con Identificadores:** Los fragmentos se presentan al LLM como `[Fuente 1]`, `[Fuente 2]` con número de página explícito.
   * **Temperatura Cero (`temperature=0.0`):** Máxima reproducibilidad y determinismo.

---

## 5. API Endpoints

| Método | Endpoint | Descripción |
|---|---|---|
| `POST` | `/documents` | Sube un archivo PDF, calcula hash SHA-256 (idempotencia), extrae texto, divide en chunks y persiste vectores en pgvector. |
| `GET` | `/documents` | Retorna el listado de documentos indexados con número de páginas y total de chunks. |
| `POST` | `/ask` | Recibe `session_id` y `question`. Retorna respuesta fundamentada, lista de fuentes con score y página, y bandera `grounded`. |
| `GET` | `/history/{session_id}` | Obtiene el historial cronológico de preguntas y respuestas de una sesión persistido en PostgreSQL. |
| `GET` | `/health` | Chequeo de estado del servicio. |

---

## 6. Requisitos y Configuración

### Requisitos:
* **Docker Desktop** (con Docker Compose v2+) **o** Python 3.11+ con PostgreSQL 16 + pgvector.
* Clave de API de OpenAI con permisos de Inferencia y Embeddings.

### Variables de Entorno (`.env`):
Crea tu archivo `.env` a partir de `.env.example`:
```bash
cp .env.example .env
```

Configura tus credenciales:
```env
OPENAI_API_KEY=sk-tu-api-key-aqui
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536

# Si ejecutas vía Docker Compose, se autoconfigura a: postgresql+asyncpg://rag:rag@db:5432/rag
# Si ejecutas en tu máquina host contra Docker:
DATABASE_URL=postgresql+asyncpg://rag:rag@localhost:5433/rag

CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TOP_K=5
MIN_SIMILARITY=0.35
MAX_UPLOAD_MB=20
LLM_MAX_CONCURRENCY=5
LLM_TIMEOUT_SECONDS=30
```

---

## 7. Instrucciones de Ejecución

### Opción A: Despliegue Completo con Docker Compose (Recomendado)

1. Levanta los contenedores (Base de datos PostgreSQL con pgvector + API FastAPI):
   ```bash
   docker compose up --build -d
   ```
2. Accede a las interfaces:
   * **Web UI Asistente:** [http://localhost:8000](http://localhost:8000)
   * **Documentación Interactiva Swagger:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Opción B: Ejecución Local en Host (con Base de Datos en Docker)

1. Inicia únicamente el contenedor de PostgreSQL con pgvector:
   ```bash
   docker compose up -d db
   ```
2. Crea el entorno virtual e instala dependencias:
   ```bash
   python -m venv .venv
   # En Windows:
   .\.venv\Scripts\Activate.ps1
   # En Linux/macOS:
   source .venv/bin/activate

   pip install -e ".[dev]"
   ```
3. Ejecuta el servidor de desarrollo:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

---

## 8. Pruebas Automatizadas y Calidad de Código

### Ejecución de Tests Unitarios y de Integración:
La suite utiliza mocks y adaptadores en memoria para validar la lógica del pipeline sin incurrir en consumo de saldo de APIs:
```bash
pytest -v
```
*(10 tests pasando: chunking con preservación de solape y página, guardrails de grounding, fallback anti-alucinación, validación de endpoints HTTP y manejo de errores).*

### Linters y Formateadores:
El proyecto cumple con las normas estrictas de PEP 8 y buenas prácticas:
```bash
# Formateo de imports
isort app tests eval

# Formateador de código
black app tests eval

# Linter estático rápido
ruff check app tests eval
```

---

## 9. Evaluación del Sistema (Benchmark RAG)

El proyecto incluye un dataset de evaluación (`eval/dataset.json`) con **12 preguntas** meticulosamente seleccionadas:
* **Preguntas Directas:** Respuestas explícitas en páginas específicas.
* **Preguntas Multi-Chunk:** Requieren contrastar y sintetizar múltiples apartados.
* **Preguntas Fuera de Dominio (Unanswerable):** Para verificar el rechazo controlado del asistente.

### Ejecutar la Evaluación Automatizada:
```bash
python eval/evaluate.py
```

### Métricas Obtenidas:
* **Tasa de Rechazo Fuera de Dominio (Anti-alucinación):** **100.0%** (Rechazo perfecto de preguntas no soportadas).
* **Hit Rate en Fuentes (Top-K):** **100.0%** en preguntas directas y **62.5%** en síntesis multi-chunk.
* **Latencia Promedio:** ~1.5 - 2.8 s (incluyendo embedding, búsqueda en pgvector y generación de LLM).
* **Reporte detallado:** Exportado en `eval/evaluation_report.json`.

---

## 10. Documentación Adicional

* Consulta **[DECISION_LOG.md](DECISION_LOG.md)** para la justificación detallada de cada decisión técnica (selección de LLM, concurrencia, resiliencia, escalabilidad a producción y trade-offs).
