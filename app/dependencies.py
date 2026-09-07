"""依赖注入：全局单例装配，供 FastAPI 各路由复用。

启动顺序（自底向上）：
embedding -> vector_store -> bm25_index -> reranker -> retriever -> llm -> rag_engine
pipeline 依赖 embedding / vector_store / bm25_index，提供入库能力。
"""
from __future__ import annotations

import json
from functools import lru_cache

from app.core.embedding import EmbeddingEngine
from app.core.llm import LLMClient
from app.core.reranker import Reranker
from app.eval.runner import EvalRunner
from app.ingestion.pipeline import IngestionPipeline
from app.rag.qa import RAGEngine
from app.retrieval.bm25 import BM25Index
from app.retrieval.chunker import Chunk
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector_store import VectorStore
from config.settings import settings


@lru_cache
def get_embedding() -> EmbeddingEngine:
    return EmbeddingEngine()


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore(embedding_dim=get_embedding().dim)


@lru_cache
def get_bm25() -> BM25Index:
    index = BM25Index()
    chunks = _load_snapshot()
    index.rebuild(chunks)
    return index


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()


@lru_cache
def get_llm() -> LLMClient:
    return LLMClient()


@lru_cache
def get_retriever() -> HybridRetriever:
    return HybridRetriever(
        vector_store=get_vector_store(),
        bm25_index=get_bm25(),
        embedding=get_embedding(),
        reranker=get_reranker(),
        top_k=settings.top_k,
    )


@lru_cache
def get_rag() -> RAGEngine:
    return RAGEngine(get_retriever(), get_llm())


@lru_cache
def get_pipeline() -> IngestionPipeline:
    return IngestionPipeline(get_embedding(), get_vector_store(), get_bm25())


@lru_cache
def get_eval_runner() -> EvalRunner:
    return EvalRunner(get_rag(), get_llm(), get_embedding())


def _load_snapshot() -> list[Chunk]:
    if not settings.chunks_file.exists():
        return []
    chunks = []
    with open(settings.chunks_file, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chunks.append(Chunk(doc_id=d["doc_id"], text=d["text"], metadata=d.get("metadata", {}), chunk_index=d.get("chunk_index", 0)))
    return chunks
