"""Configuración centralizada leída desde variables de entorno."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5433/rag"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5
    min_similarity: float = 0.35
    max_upload_mb: int = 20

    llm_max_concurrency: int = 5
    llm_timeout_seconds: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
