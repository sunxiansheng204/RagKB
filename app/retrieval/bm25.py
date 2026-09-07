"""BM25 关键词检索：与向量检索互补，覆盖精确术语匹配场景。

为什么需要它？
- 向量检索擅长语义相似，但对专有名词、编号、缩写等精确关键词不敏感；
- BM25 是经典词频-逆文档频率模型，命中即高置信，和向量互为补充；
- 两路召回后用 RRF(Reciprocal Rank Fusion) 融合，是 RAG 里的业界标配方案。
"""
from __future__ import annotations

import jieba
from rank_bm25 import BM25Okapi

from app.core.reranker import RankedChunk


def tokenize(text: str) -> list[str]:
    """中文按 jieba 分词，英文数字按词切分，统一小写。"""
    words = []
    for w in jieba.cut(text):
        w = w.strip().lower()
        if w and not w.isspace():
            words.append(w)
    return words


class BM25Index:
    def __init__(self):
        self._chunks: list[RankedChunk] = []
        self._bm25: BM25Okapi | None = None

    def rebuild(self, chunks: list):
        """传入全部切块（Chunk 或 RankedChunk，均需带 chunk_id/text/metadata），重建倒排索引。"""
        self._chunks = list(chunks)
        corpus = [tokenize(c.text) for c in self._chunks]
        if corpus:
            self._bm25 = BM25Okapi(corpus)
        else:
            self._bm25 = None

    @property
    def size(self) -> int:
        return len(self._chunks)

    def search(self, query: str, top_k: int) -> list[RankedChunk]:
        if not self._bm25 or not self._chunks:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order[:top_k]:
            if scores[i] <= 0:
                continue
            c = self._chunks[i]
            out.append(
                RankedChunk(
                    doc_id=c.doc_id,
                    chunk_id=getattr(c, "chunk_id", c.doc_id),
                    text=c.text,
                    metadata=c.metadata,
                    score=float(scores[i]),
                    source="bm25",
                )
            )
        return out
