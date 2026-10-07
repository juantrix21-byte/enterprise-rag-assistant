"""Adaptador de LLM usando la API de OpenAI con control de concurrencia y resiliencia."""

import asyncio

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.domain.ports import LLMPort

settings = get_settings()


class OpenAILLMAdapter(LLMPort):
    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        max_concurrency: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.llm_model
        self.semaphore = asyncio.Semaphore(max_concurrency or settings.llm_max_concurrency)
        self.timeout_seconds = timeout_seconds or settings.llm_timeout_seconds

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        # Control de concurrencia para evitar saturación de rate-limits
        async with self.semaphore:
            async with asyncio.timeout(self.timeout_seconds):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                )
                content = response.choices[0].message.content
                return content.strip() if content else ""
