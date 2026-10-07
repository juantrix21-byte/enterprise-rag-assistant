"""Adaptador de Embeddings usando la API de OpenAI."""

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.domain.ports import EmbeddingPort

settings = get_settings()


class OpenAIEmbeddingAdapter(EmbeddingPort):
    def __init__(self, client: AsyncOpenAI | None = None, model: str | None = None) -> None:
        self.client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.embedding_model

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True
    )
    async def get_embedding(self, text: str) -> list[float]:
        cleaned = text.replace("\n", " ").strip()
        response = await self.client.embeddings.create(
            input=[cleaned],
            model=self.model,
        )
        return response.data[0].embedding

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True
    )
    async def get_embeddings(self, texts: list[str], batch_size: int = 50) -> list[list[float]]:
        if not texts:
            return []

        cleaned_texts = [t.replace("\n", " ").strip() for t in texts]
        all_embeddings: list[list[float]] = []

        for i in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[i : i + batch_size]
            response = await self.client.embeddings.create(
                input=batch,
                model=self.model,
            )
            # Ordenar por el índice devuelto por OpenAI para garantizar correspondencia exacta
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([d.embedding for d in sorted_data])

        return all_embeddings
