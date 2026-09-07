"""混合检索：向量召回 + BM25 召回 -> RRF 融合 -> 可选重排序。

RRF 公式：score(d) = Σ 1/(k + rank_i(d))
- k 常取 60；
- 只依赖排名不依赖原始分数，天然规避两路分数量纲不一致的问题；
- 融合后去重（按 chunk_id），保证喂给 LLM 的上下文不重复。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from app.core.reranker import Reranker
from app.core.embedding import EmbeddingEngine
from app.retrieval.bm25 import BM25Index
from app.retrieval.vector_store import VectorStore


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedding: EmbeddingEngine,
        reranker: Reranker | None = None,
        top_k: int = 6,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding = embedding
        self.reranker = reranker
        self.top_k = top_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or self.top_k
        vec_n = max(top_k * 2, 8)
        bm25_n = max(top_k * 2, 8)

        # 1) 两路召回
        q_vec = self.embedding.encode_query(query)
        vector_hits = self.vector_store.query(q_vec, top_k=vec_n)
        bm25_hits = self.bm25_index.search(query, top_k=bm25_n)

        # 2) RRF 融合
        rrf = defaultdict(float)
        rank_map: dict[str, dict] = {}

        for rank, hit in enumerate(vector_hits):
            cid = hit["id"]
            rrf[cid] += 1.0 / (self.rrf_k + rank + 1)
            rank_map.setdefault(cid, {"id": cid, "text": hit["text"], "metadata": hit["metadata"], "vector_score": hit["score"], "bm25_score": 0.0})

        for rank, hit in enumerate(bm25_hits):
            cid = hit.chunk_id
            rrf[cid] += 1.0 / (self.rrf_k + rank + 1)
            entry = rank_map.setdefault(cid, {"id": cid, "text": hit.text, "metadata": hit.metadata, "vector_score": 0.0, "bm25_score": 0.0})
            entry["bm25_score"] = hit.score

        merged = [(cid, score, rank_map[cid]) for cid, score in rrf.items()]
        merged.sort(key=lambda x: x[1], reverse=True)
        candidates = merged[:top_k]

        results = []
        for cid, rrf_score, entry in candidates:
            entry["rrf_score"] = round(rrf_score, 6)
            entry["combined_score"] = entry["rrf_score"]
            results.append(entry)

        # 3) 可选重排序精排
        if self.reranker and self.reranker.model is not None:
            ranked = self.reranker.rerank(
                query,
                [
                    RankedChunk(
                        doc_id=r["id"],
                        chunk_id=r["id"],
                        text=r["text"],
                        metadata=r["metadata"],
                        score=r["rrf_score"],
                        source="hybrid",
                    )
                    for r in results
                ],
                top_k=top_k,
            )
            results = [
                {"id": r.doc_id, "text": r.text, "metadata": r.metadata, "rrf_score": r.score, "combined_score": r.score}
                for r in ranked
            ]

        return results
