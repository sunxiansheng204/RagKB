"""Embedding 引擎：统一封装本地(sentence-transformers)与云端(OpenAI 兼容)两种后端。

设计要点：
- backend 通过配置切换，两种后端暴露同一 encode() 接口，业务层无感知；
- 向量默认做 L2 归一化，保证后续向量库用余弦相似度检索时直接用内积即可；
- 缓存嵌入维度，供向量库建集合时使用。
"""
from __future__ import annotations

import numpy as np

from config.settings import settings


class EmbeddingEngine:
    def __init__(self, cfg=None):
        self.cfg = cfg or settings
        self.backend = self.cfg.embedding_backend
        self.model = None
        self.client = None
        self._dim: int | None = None

        if self.backend == "sentence-transformers":
            self._init_local()
        elif self.backend == "openai":
            self._init_api()
        else:
            raise ValueError(f"未知 embedding_backend: {self.backend}，可选 sentence-transformers / openai")

    def _init_local(self):
        from sentence_transformers import SentenceTransformer  # 延迟导入，加快启动

        self.model = SentenceTransformer(self.cfg.embedding_model, device="cpu")
        self._dim = int(self.model.get_sentence_embedding_dimension())

    def _init_api(self):
        from openai import OpenAI

        self.client = OpenAI(
            base_url=self.cfg.embedding_base_url or None,
            api_key=self.cfg.embedding_api_key or "EMPTY",
        )
        self._dim = self.cfg.embedding_dim or 1024

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        """批量向量化，返回 shape=(n, dim) 的归一化向量矩阵。"""
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        if self.backend == "sentence-transformers":
            emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        else:
            resp = self.client.embeddings.create(model=self.cfg.embedding_api_model, input=texts)
            emb = np.array([d.embedding for d in resp.data], dtype=np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.maximum(norms, 1e-9)
        return np.asarray(emb, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
