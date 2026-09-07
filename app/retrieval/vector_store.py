"""向量库封装：ChromaDB 持久化存储 + 余弦相似度检索。

统一正向与反向索引：
- 正向：chunk_id -> chunk（元数据 + 原文），供召回后取回原文；
- 反向：向量集合按 embedding 建近似最近邻索引。
Chroma 把两块天然合并在一个 collection 里，工程上最省心。

Embedding 策略：本模块不内置任何编码器，全部向量由外部 EmbeddingEngine
显式传入（add/query 均走 embeddings / query_embeddings），保证换模型
零污染、不触发 Chroma 默认 MiniLM 模型下载。
"""
from __future__ import annotations

import numpy as np
from chromadb.api.types import EmbeddingFunction

from config.settings import settings


class _RawEmbeddingFunction(EmbeddingFunction):
    """占位编码器：Collection 不允许缺省 embedding_function，但本项目
    向量一律外部传入，绝不允许让 Chroma 内置模型编码文本再入库。"""

    def __call__(self, input):
        raise RuntimeError(
            "向量必须由外部 EmbeddingEngine 显式传入，禁止 Chroma 内置模型编码。"
        )


class VectorStore:
    def __init__(self, cfg=None, embedding_dim: int = 512):
        self.cfg = cfg or settings
        import chromadb

        self.client = chromadb.PersistentClient(path=str(self.cfg.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name="kb_docs",
            embedding_function=_RawEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
        self._dim = embedding_dim

    def upsert_chunks(self, chunks, embeddings):
        """批量写入切块与对应向量。chunks: list[Chunk]，embeddings: np.ndarray"""
        ids = [c.chunk_id for c in chunks]
        if not ids:
            return
        metas = [{**c.metadata, "doc_id": c.doc_id, "chunk_index": c.chunk_index} for c in chunks]
        docs = [c.text for c in chunks]
        self.collection.upsert(ids=ids, embeddings=embeddings.tolist(), documents=docs, metadatas=metas)

    def query(self, query_vector: np.ndarray, top_k: int, where: dict | None = None):
        """返回 list[dict]，字段：id / text / metadata / distance。"""
        res = self.collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for i, cid in enumerate(ids):
            score = 1.0 - float(dists[i])  # cosine distance -> similarity
            out.append(
                {
                    "id": cid,
                    "text": docs[i],
                    "metadata": metas[i] or {},
                    "score": max(0.0, min(1.0, score)),
                }
            )
        return out

    def count(self) -> int:
        return self.collection.count()

    def delete_by_doc(self, doc_id: str) -> int:
        """删除某文档全部切块，返回删除数量。"""
        res = self.collection.get(where={"doc_id": doc_id})
        ids = res["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def reset(self):
        """清空整个知识库（重建索引时使用）。"""
        self.client.delete_collection("kb_docs")
        self.collection = self.client.get_or_create_collection(
            name="kb_docs",
            embedding_function=_RawEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
