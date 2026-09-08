"""FastAPI 主入口 + 启动时自动灌入示例知识库。"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_chat, routes_docs, routes_eval
from app.api.security import get_auth_state
from app.dependencies import get_pipeline, get_vector_store
from config.settings import settings


def _ensure_sample_kb():
    """首次启动且知识库为空时，自动灌入 data/documents 下的示例文档。"""
    pipeline = get_pipeline()
    if get_vector_store().count() == 0 and any(settings.kb_dir.rglob("*")):
        pipeline.ingest_directory(settings.kb_dir)
        print("[init] 示例知识库已自动灌入。")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_sample_kb()
    yield


app = FastAPI(
    title="RagKB · 企业级 RAG 知识库问答系统",
    description="混合检索(向量+BM25+RRF) + 引用溯源 + 流式输出 + 效果评估",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api-docs",  # 内置 Swagger 让出 /docs 给业务路由
)

# CORS 严格白名单（由 settings.cors_origins 配置，禁止使用 *，避免任意网页跨域调用接口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(routes_docs.router)
app.include_router(routes_chat.router)
app.include_router(routes_eval.router)


@app.get("/health", tags=["meta"])
def health():
    from app.dependencies import get_vector_store as _vs

    return {
        "status": "ok",
        "documents": len(get_pipeline().list_documents()),
        "chunks": _vs().count(),
        "embedding_backend": settings.embedding_backend,
        "llm_model": settings.llm_model,
        "auth": get_auth_state(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
