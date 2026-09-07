"""全局配置：从 .env / 环境变量读取，未配置时使用合理默认值。"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- LLM ----------
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen2.5:7b"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # ---------- Embedding ----------
    embedding_backend: str = "sentence-transformers"  # sentence-transformers | openai
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_api_model: str = "text-embedding-v3"

    # ---------- 检索 ----------
    top_k: int = 6
    chunk_size: int = 512
    chunk_overlap: int = 64
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"

    # ---------- 路径 ----------
    data_dir: Path = PROJECT_ROOT / "data"
    kb_dir: Path = PROJECT_ROOT / "data" / "documents"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma_db"
    chunks_file: Path = PROJECT_ROOT / "data" / "index" / "chunks.jsonl"
    eval_report_dir: Path = PROJECT_ROOT / "data" / "reports"

    # ---------- 服务 ----------
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()

# 确保目录存在
for _p in (settings.kb_dir, settings.chroma_dir, settings.chunks_file.parent, settings.eval_report_dir):
    _p.mkdir(parents=True, exist_ok=True)
