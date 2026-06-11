# engine/settings.py
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ASKLAKE_", env_file=".env", extra="ignore")

    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    parquet_dir: str | None = None
    observability_backend: str = "noop"  # "noop" | "prometheus"
    auth_config: str | None = None  # path to auth.yaml (token->role map); None -> all public
    api_host: str = "0.0.0.0"
    api_port: int = 8000


def get_settings() -> Settings:
    return Settings()
