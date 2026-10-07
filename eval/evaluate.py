"""Script de evaluación sistemática para el Enterprise RAG Assistant.

Mide:
1. Rejection Rate / Anti-hallucination: Capacidad de rechazar preguntas fuera de contexto.
2. Source Hit Rate: Presencia de la página esperada entre las fuentes recuperadas.
3. Latencia promedio por consulta.
"""

import asyncio
import contextlib
import json
import logging
import sys
import time
from pathlib import Path

from app.application.rag_service import RAGService
from app.infrastructure.db.history_repository import PostgresHistoryRepository
from app.infrastructure.db.session import async_session_maker
from app.infrastructure.embeddings.openai_embeddings import OpenAIEmbeddingAdapter
from app.infrastructure.llm.openai_llm import OpenAILLMAdapter
from app.infrastructure.vectorstore.pgvector_store import PgVectorStore

logging.basicConfig(level=logging.WARNING)

if sys.platform == "win32":
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")


async def run_evaluation(dataset_path: str = "eval/dataset.json") -> dict:
    with open(dataset_path, encoding="utf-8") as f:
        items = json.load(f)

    results = []
    correct_rejections = 0
    total_unanswerable = 0

    retrieval_hits = 0
    total_answerable = 0

    total_latency = 0.0

    print("=" * 80)
    print(f"INICIANDO EVALUACIÓN SISTEMÁTICA RAG ({len(items)} preguntas)")
    print("=" * 80)

    async with async_session_maker() as session:
        vstore = PgVectorStore(session)
        emb = OpenAIEmbeddingAdapter()
        llm = OpenAILLMAdapter()
        hist = PostgresHistoryRepository(session)
        rag_svc = RAGService(
            vector_store=vstore,
            embedding_provider=emb,
            llm_provider=llm,
            history_repo=hist,
        )

        for i, item in enumerate(items, 1):
            q_id = item["id"]
            question = item["question"]
            q_type = item["type"]
            can_answer = item["can_answer"]
            expected_page = item.get("expected_page")

            print(f"\n[{i}/{len(items)}] [{q_type.upper()}] {question}")

            t0 = time.perf_counter()
            res = await rag_svc.answer_question(
                session_id=f"eval-session-{q_id}", question=question
            )
            lat = (time.perf_counter() - t0) * 1000
            total_latency += lat

            retrieved_pages = [s.page_number for s in res.sources]
            is_rejected = (
                (not res.grounded)
                or ("no encontré información suficiente" in res.answer.lower())
                or ("no contiene datos suficientes" in res.answer.lower())
                or ("prefiero no inventarte" in res.answer.lower())
            )

            hit = False
            rejection_ok = False

            if not can_answer:
                total_unanswerable += 1
                if is_rejected:
                    correct_rejections += 1
                    rejection_ok = True
                    print(f"  [OK] Rechazo correcto (Anti-alucinacion OK) | Latencia: {lat:.1f}ms")
                else:
                    print(
                        f"  [ALERTA] Falso Positivo (Respondio sin soporte): {res.answer[:100]}..."
                    )
            else:
                total_answerable += 1
                if expected_page in retrieved_pages:
                    retrieval_hits += 1
                    hit = True
                    print(
                        f"  [OK] Fuente encontrada (Pag. {expected_page} en {retrieved_pages}) | Grounded: {res.grounded}"
                    )
                else:
                    print(
                        f"  [INFO] Pag esperada {expected_page} no estuvo en top-k {retrieved_pages}"
                    )

            results.append(
                {
                    "id": q_id,
                    "type": q_type,
                    "question": question,
                    "can_answer": can_answer,
                    "expected_page": expected_page,
                    "retrieved_pages": retrieved_pages,
                    "grounded": res.grounded,
                    "hit": hit,
                    "correct_rejection": rejection_ok,
                    "answer_snippet": res.answer[:200],
                    "latency_ms": round(lat, 2),
                }
            )

    # Métricas consolidadas
    rejection_rate = (correct_rejections / total_unanswerable * 100) if total_unanswerable else 0.0
    hit_rate = (retrieval_hits / total_answerable * 100) if total_answerable else 0.0
    avg_latency = total_latency / len(items) if items else 0.0

    summary = {
        "total_questions": len(items),
        "total_answerable": total_answerable,
        "total_unanswerable": total_unanswerable,
        "rejection_accuracy_pct": round(rejection_rate, 2),
        "source_hit_rate_pct": round(hit_rate, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "results": results,
    }

    report_path = Path("eval/evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("RESUMEN DE EVALUACIÓN")
    print("=" * 80)
    print(f"Preguntas evaluadas:              {len(items)}")
    print(
        f"Tasa de rechazo controlado:       {rejection_rate:.1f}% ({correct_rejections}/{total_unanswerable})"
    )
    print(
        f"Source Hit Rate (Top-K):          {hit_rate:.1f}% ({retrieval_hits}/{total_answerable})"
    )
    print(f"Latencia promedio:                {avg_latency:.1f} ms")
    print(f"Reporte exportado a:              {report_path.resolve()}")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    asyncio.run(run_evaluation())
