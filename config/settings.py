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
    # 默认仅监听本机回环地址；如需局域网访问再改为 0.0.0.0。
    # 注意：改为 0.0.0.0 且未配置 RAGKB_API_TOKEN 时，任意内网主机均可调用接口，风险自担。
    host: str = "127.0.0.1"
    port: int = 8000

    # ---------- 安全 ----------
    # 接口鉴权 Token（Bearer Token）。为空 = 演示模式（不校验，仅建议本机使用）；
    # 非空 = 生产模式，所有接口须携带 Authorization: Bearer <token>，否则返回 401。
    api_token: str = ""

    # CORS 允许来源（严格白名单，禁止使用 *）。多个来源用英文逗号分隔。
    # 默认放行本机 Streamlit 前端；生产部署时替换为实际前端域名。
    cors_origins: str = "http://127.0.0.1:8501,http://localhost:8501"

    # 单文件上传大小上限（MB），防止超大文件拖垮内存。
    max_upload_mb: int = 50

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()

# 确保目录存在
for _p in (settings.kb_dir, settings.chroma_dir, settings.chunks_file.parent, settings.eval_report_dir):
    _p.mkdir(parents=True, exist_ok=True)
