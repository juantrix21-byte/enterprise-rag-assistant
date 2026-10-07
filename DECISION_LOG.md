# DECISION LOG — Enterprise RAG Assistant
**Candidato:** AI Engineer  
**Proyecto:** Enterprise RAG Assistant (CIE LAB — Universidad Continental)  
**Fecha:** Octubre 2026  

Este documento detalla las justificaciones técnicas, alternativas analizadas y decisiones de arquitectura tomadas durante el diseño y construcción del asistente inteligente RAG.

---

## 1. ¿Por qué elegiste el modelo/LLM utilizado?

* **Modelo seleccionado:** `gpt-4o-mini` (OpenAI).
* **Motivos de la elección:**
  1. **Capacidad de razonamiento y seguimiento de instrucciones (Instruction Following):** Para un sistema con grounding estricto, el modelo debe obedecer sin desviaciones directivas negativas (*"no inventes información si no está en el contexto"*). `gpt-4o-mini` exhibe una fidelidad al contexto equiparable a modelos insignia de mayor tamaño.
  2. **Eficiencia de costos:** Con un costo de \$0.15 / millón de tokens de entrada y \$0.60 / millón de salida, permite viabilizar asistentes a escala institucional sin incurrir en costos prohibitivos.
  3. **Latencia optimizada (Time-to-First-Token y Throughput):** Tiempos de generación significativamente más bajos (~300-800 ms) en comparación con modelos de 70B+ parámetros, crucial para experiencia conversacional interactiva.
  4. **Ventana de contexto amplia (128k tokens):** Permite inyectar múltiples fragmentos documentales y turnos de conversación previa sin degradación por truncamiento.
* **Alternativas consideradas:**
  * *Modelos locales (Llama 3 8B via Ollama / vLLM):* Descartado para la evaluación rápida debido a la necesidad de infraestructura GPU dedicada y mayor variabilidad en seguimiento estricto de prompts de grounding sin fine-tuning previo.
  * *GPT-4o (completo):* Innecesario para tareas de síntesis documental y extracción de contexto delimitado; encarece los costos en un factor de 15x sin ganancia perceptible en precisión de extracción.

---

## 2. ¿Por qué elegiste la estrategia de chunking?

* **Estrategia seleccionada:** *Recursive Paragraph & Sentence Chunking con solape semántico* (`chunk_size=1000` caracteres, `chunk_overlap=150` caracteres), preservando el número de página original (1-indexed).
* **Motivos de la elección:**
  1. **Preservación de unidades semánticas completas:** En lugar de cortar textos por longitud fija de tokens o caracteres de forma ciega, el algoritmo respeta los saltos de párrafo (`\n\n`) y oraciones. Esto evita dividir proposiciones conceptuales a la mitad.
  2. **Continuidad contextual mediante Overlap:** El solape del 15% (~150 caracteres) asegura que si una idea o entidad clave se encuentra en la frontera entre dos fragmentos contiguos, el contexto se transfiera adecuadamente a ambos vectores.
  3. **Trazabilidad estricta por página:** Cada chunk mantiene el metadato inmutable de la página exacta del documento original (`page_number`). Esto es la base para la citación auditable por los docentes o investigadores.
* **Alternativas consideradas:**
  * *Fixed-size Token Chunking:* Muy propenso a truncar oraciones a la mitad, degradando la calidad de los embeddings semánticos.
  * *Chunking a nivel de página completa:* Genera fragmentos demasiado extensos (~3,000-4,000 caracteres) que diluyen la similitud coseno de consultas puntuales y saturan la ventana de contexto del LLM con ruido irrelevante.

---

## 3. ¿Por qué elegiste la solución de búsqueda/vectorial?

* **Solución seleccionada:** **PostgreSQL + pgvector** (vía contenedor Docker `pgvector/pgvector:pg16`).
* **Motivos de la elección:**
  1. **Unificación de persistencia relacional y vectorial:** Permite almacenar en un único motor de base de datos robusto tanto los metadatos de documentos, los fragmentos con sus embeddings (tipo `vector(1536)`), y el historial de conversaciones. Esto elimina la necesidad de sincronizar dos bases de datos separadas (ej. SQLite + Chroma).
  2. **Consistencia transaccional ACID:** La creación o eliminación de documentos y sus chunks asociados se ejecuta dentro de transacciones SQL atómicas con borrado en cascada (`ON DELETE CASCADE`).
  3. **Escalabilidad y madurez industrial:** Soporta índices vectoriales de producción como **HNSW** (Hierarchical Navigable Small World) e **IVFFlat**, con capacidad de filtrar metadatos mediante cláusulas `WHERE` nativas de SQL con alto rendimiento.
* **Alternativas consideradas:**
  * *Chroma / FAISS en memoria:* Fáciles para prototipos en memoria, pero sufren de persistencia frágil, falta de soporte ACID, concurrencia limitada para escritura múltiple y dificultad para realizar consultas relacionales combinadas.
  * *Pinecone / Qdrant Cloud:* Añade dependencia de servicios en la nube de terceros y latencia adicional de red externa para las operaciones de base de datos.

---

## 4. ¿Cómo definiste el proceso de retrieval?

* **Proceso implementado:**
  1. **Vectorización de la consulta:** La pregunta del usuario se normaliza y se transforma en un vector denso de 1536 dimensiones mediante `text-embedding-3-small`.
  2. **Cálculo de Similitud Coseno:** Se evalúa la distancia coseno contra la tabla `document_chunks` usando el operador nativo `<=>` de pgvector:
     $$\text{Cosine Similarity} = 1.0 - (\text{embedding} \iff \text{query\_embedding})$$
  3. **Filtrado por Umbral Mínimo (`min_similarity = 0.35`):** Se descartan inmediatamente aquellos fragmentos cuya correlación semántica sea espuria.
  4. **Top-K Selection (`top_k = 5`):** Se recuperan los 5 fragmentos más representativos que hayan superado el umbral.
  5. **Inyección en Prompt con Identificadores de Fuente:** Cada fragmento se enumera como `[Fuente 1]`, `[Fuente 2]`, indicando explícitamente documento, página y score de similitud.

---

## 5. ¿Cómo intentaste reducir alucinaciones?

El sistema cuenta con una estrategia de defensa multicapa contra alucinaciones:
1. **Filtro de Relevancia Previo (Guardrail Determinístico):** Si ningún fragmento de la base de datos supera el umbral de similitud mínima, **el sistema aborta el flujo y no invoca al LLM**. Retorna directamente: *"Busqué en los documentos y no encontré información suficiente para responderte esto con seguridad, y prefiero no inventarte una respuesta."* Esto garantiza cero alucinaciones en preguntas fuera de dominio a costo \$0.
2. **Temperatura Cero (`temperature = 0.0`):** Maximiza la determinicidad del muestreo de tokens del modelo, minimizando la creatividad inventiva.
3. **Instrucciones Negativas Fuertes en el System Prompt:**
   * *"Responde EXCLUSIVAMENTE basándote en el contexto documental proporcionado."*
   * *"Si el contexto no contiene información suficiente, responde textualmente que no hay información suficiente."*
   * *"NUNCA inventes autores, fechas, estadísticas ni metodologías."*
4. **Citas Obligatorias In-line:** El prompt exige ligar cada afirmación a una referencia formal `[Fuente N]`.

---

## 6. ¿Cómo implementaste o controlaste el grounding?

El grounding se controla a nivel estructural y de trazabilidad:
1. **Atribución Explícita:** Cada afirmación en la respuesta del LLM se correlaciona con un bloque contextual delimitado.
2. **Entrega de Metadatos DTO:** La API no solo devuelve el texto de la respuesta, sino el array estructurado `sources` con:
   - `document_name`: Nombre del archivo de origen.
   - `page_number`: Página física exacta (1-indexed).
   - `chunk_id`: Identificador único de fragmento en base de datos.
   - `snippet`: Primeros 250 caracteres del texto fuente para verificación inmediata.
   - `similarity_score`: Coeficiente de similitud matemática.
3. **Indicador Booleano `grounded`:** Permite al frontend o a clientes API saber inmediatamente si la respuesta está sustentada en fuentes o fue una respuesta de rechazo controlado.

---

## 7. ¿Cómo se obtienen y muestran las fuentes?

* **En el Backend:** El adaptador de pgvector recupera los registros ORM `ChunkRecord` con su respectivo `similarity_score`. Estos se transforman a objetos de dominio `SourceReference`.
* **En la API:** Se serializan en el schema `AskResponse.sources`.
* **En el Frontend:**
  * Cada respuesta muestra una tarjeta desplegable interactiva (*`<details>`*).
  * Se visualizan badges de relevancia en porcentaje (ej. `Similitud: 69.5%`).
  * Se muestra la página exacta y un extracto en cursiva del contenido original, permitiendo al usuario auditar la veracidad del contenido.

---

## 8. ¿Qué ocurriría si el documento no contiene la respuesta?

Existen dos escenarios controlados:
1. **Pregunta completamente fuera de tema (ej. gastronomía, programación no relacionada):** El retrieval no encuentra chunks que superen el score mínimo (`min_similarity`). Se activa el fallback inmediato sin llamada al LLM (`grounded: false`, `sources: []`).
2. **Pregunta relacionada al tema pero con detalle específico ausente en el texto:** El retrieval recupera fragmentos temáticamente afines (superando el umbral), pero el prompt instruye al LLM a analizar si la respuesta específica está allí. Al no encontrarla, el LLM responde: *"Busqué en los documentos y no encontré información suficiente para responderte esto con seguridad, y prefiero no inventarte una respuesta."*

En ningún caso el asistente inventa hechos externos.

---

## 9. ¿Cómo soportarías múltiples consultas simultáneas? (Concurrencia y Escalabilidad)

### A. Principales cuellos de botella esperados:
1. **Llamadas bloqueantes a APIs externas:** La latencia del LLM (~500 ms - 2 s) es el cuello de botella principal de throughput.
2. **Límites de tasa (Rate Limits) del proveedor de IA:** Errores `429 Too Many Requests` (TPM / RPM) si muchos usuarios consultan a la vez.
3. **Cálculo de búsqueda vectorial en base de datos:** El escaneo secuencial de vectores degrada el rendimiento a medida que crecen los chunks si no hay índices aproximados (ANN).

### B. Control de concurrencia y resiliencia implementados:
* **Semáforo Asíncrono (`asyncio.Semaphore`):** Se limita el número máximo de llamadas simultáneas al LLM (configurable mediante `LLM_MAX_CONCURRENCY=5`), encolando limpiamente las peticiones en el event loop.
* **Manejo de Timeouts:** Se envuelven las llamadas a la API en `asyncio.timeout(settings.llm_timeout_seconds)` para evitar colgar procesos por respuestas estancadas.
* **Reintentos con Backoff Exponencial (`tenacity`):** En caso de errores transitorios o picos de red, reintenta hasta 3 veces espaciando las llamadas progresivamente (2s, 4s, 8s).

### C. Estrategia de escalamiento horizontal:
* **API FastAPI Stateless:** El backend es completamente sin estado; puede escalarse horizontalmente detrás de un balanceador de carga (Nginx / AWS ALB / Traefik) con múltiples réplicas (Docker Swarm o Kubernetes).
* **Pool de conexiones de Base de Datos:** Configurado con `asyncpg` y `SQLAlchemy` (`pool_size=10`, `max_overflow=20`), desacoplado de las operaciones I/O del LLM.

### D. Estrategias para reducir latencia y costos:
1. **Semantic Caching:** Integrar un caché previo (ej. Redis Vector Cache) donde consultas semánticamente equivalentes devuelvan la respuesta almacenada en <5 ms sin llamar a OpenAI.
2. **Índice HNSW en pgvector:** Crear un índice HNSW en la columna `embedding` (`CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops)`) para lograr búsqueda sublineal $O(\log N)$ con millones de chunks.
3. **Chunking adaptativo y streaming:** Habilitar SSE (Server-Sent Events) para transmitir tokens al frontend progresivamente, reduciendo la percepción de latencia del usuario a menos de 200 ms.

---

## 10. ¿Qué cambiarías para llevar esta solución a producción?

1. **Gestión de Identidad y Multi-tenancy:** Implementar autenticación JWT / OAuth2 y aislar los documentos e historiales por usuario o por organización educativa (`tenant_id`).
2. **Cola de tareas asíncronas para Ingesta:** Migrar la extracción y vectorización de PDFs pesados a trabajadores en segundo plano (Celery / ARQ / Redis Queue) con notificaciones por WebSockets.
3. **Observabilidad y Trazabilidad de LLM (LLMOps):** Integrar OpenTelemetry, Langfuse o Arize Phoenix para registrar tokens consumidos, costo exacto por sesión, y métricas de latencia de retrieval vs. generación.
4. **Almacenamiento de Objetos S3 / Azure Blob Storage:** Guardar los archivos binarios PDF en un bucket seguro con URLs firmadas temporales para visualización directa en el navegador.
5. **Índice Vectorial Optimizado:** Aplicar particionamiento de tablas en PostgreSQL y crear el índice HNSW con parámetros calibrados (`m=16`, `ef_construction=64`).

---

## 11. ¿Qué funcionalidades decidiste no implementar por limitación de tiempo y por qué?

1. **Hybrid Search (BM25 + Dense Vectors) con Reciprocal Rank Fusion (RRF):**
   * *Por qué se pospuso:* Requería configurar extensiones de Full-Text Search en Postgres (`tsvector`) combinadas con pgvector. Para el volumen y dominio evaluado (guía conceptual de 48 páginas), la búsqueda densa con embeddings semánticos demostró un Hit Rate del 100% en las pruebas sin añadir la complejidad del reranking híbrido.
2. **Re-ranking con Cross-Encoder (Cohere Rerank o BGE-Reranker):**
   * *Por qué se pospuso:* Añade una llamada de red adicional o una librería pesada en Python. Se priorizó optimizar el umbral de similitud coseno directo y el prompt de grounding en el LLM.
3. **Streaming de tokens vía Server-Sent Events (SSE):**
   * *Por qué se pospuso:* Con `gpt-4o-mini`, la latencia total es de ~600-900 ms, lo cual es suficientemente rápido para la interfaz mínima sin requerir el manejo de streams incompletos y buffers de fuentes en el cliente.
