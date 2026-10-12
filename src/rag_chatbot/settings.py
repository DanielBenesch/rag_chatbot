from functools import cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPANY_PDF = PROJECT_ROOT / "content" / "company.pdf"
EVAL_DATASET = PROJECT_ROOT / "eval" / "dataset.jsonl"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr
    chat_model: str = "gpt-6-luna"
    embedding_model: str = "text-embedding-3-small"

    chroma_host: str = "localhost"
    chroma_port: int = 8000

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_enabled: bool = True


@cache
def get_settings() -> Settings:
    return Settings()
