"""可选的本地重排序器（bge-reranker）。

三级排序管线：BM25 与向量召回 -> RRF 粗融合 -> 重排序器精排。
重排序对 top-K 内的候选做相关度重打分，能显著改善长尾问题，
但需要额外下载约 1GB 的本地模型，默认关闭（use_reranker=false）。
"""
from __future__ import annotations

from dataclasses import dataclass

from config.settings import settings


@dataclass
class RankedChunk:
    """一次召回的结果单元。"""

    doc_id: str
    text: str
    metadata: dict
    score: float = 0.0
    source: str = "manual"  # 召回来源：vector / bm25 / hybrid / rerank
    chunk_id: str = ""  # 全链路唯一键：与向量/快照对齐


class Reranker:
    def __init__(self, cfg=None):
        self.cfg = cfg or settings
        self.model = None
        if self.cfg.use_reranker:
            self._load()

    def _load(self):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(self.cfg.reranker_model, max_length=512)

    def rerank(self, query: str, candidates: list[RankedChunk], top_k: int) -> list[RankedChunk]:
        if not candidates:
            return []
        pairs = [(query, c.text[:500]) for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        out = []
        for chunk, score in ranked[:top_k]:
            chunk.score = float(score)
            chunk.source = "rerank"
            out.append(chunk)
        return out
