"""知识库入库流水线：加载 -> 切分 -> 向量化 -> 双路索引。

维护一份 chunks.jsonl（正向索引快照），每次入库/删除后重建 BM25 倒排索引，
向量走 Chroma 增量 upsert。两个索引通过 chunk_id 对齐。
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from tqdm import tqdm

from app.core.embedding import EmbeddingEngine
from app.retrieval.bm25 import BM25Index, tokenize
from app.retrieval.chunker import Chunk, chunk_document
from app.retrieval.vector_store import VectorStore
from app.ingestion.loader import DocumentLoader
from config.settings import settings


class IngestionPipeline:
    def __init__(self, embedding: EmbeddingEngine, vector_store: VectorStore, bm25_index: BM25Index, cfg=None):
        self.embedding = embedding
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.loader = DocumentLoader()
        self.cfg = cfg or settings

    # ---------- 持久化快照 ----------
    def _save_snapshot(self, chunks: list[Chunk]):
        with open(self.cfg.chunks_file, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps({"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text, "metadata": c.metadata, "chunk_index": c.chunk_index}, ensure_ascii=False) + "\n")

    def _load_snapshot(self) -> list[Chunk]:
        if not self.cfg.chunks_file.exists():
            return []
        chunks = []
        with open(self.cfg.chunks_file, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                chunks.append(Chunk(doc_id=d["doc_id"], text=d["text"], metadata=d.get("metadata", {}), chunk_index=d.get("chunk_index", 0)))
        return chunks

    # ---------- 索引重建 ----------
    def rebuild_indexes(self):
        """从快照重建 BM25 + 向量集（向量集重建由调用方决定是否 reset）。"""
        chunks = self._load_snapshot()
        for c in chunks:
            c._uuid = uuid.uuid4().hex  # type: ignore[attr-defined]
        self.bm25_index.rebuild([Chunk(c.doc_id, c.text, c.metadata, c.chunk_index) for c in chunks])

    # ---------- 单文档入库 ----------
    def ingest_document(self, path: str | Path) -> dict:
        doc = self.loader.load(path)
        chunks = chunk_document(doc.doc_id, doc.text, doc.metadata, self.cfg.chunk_size, self.cfg.chunk_overlap)

        # 向量化（每次处理一个文档，避免内存爆炸）
        texts = [c.text for c in chunks]
        embeddings = None
        if texts:
            embeddings = self.embedding.encode(texts)

        # 写入向量库（显式传向量，Chroma 不再内置编码）
        self.vector_store.upsert_chunks(chunks, embeddings)

        # 追加快照（先把旧的同 doc 记录去掉）
        existing = self._load_snapshot()
        existing = [c for c in existing if c.doc_id != doc.doc_id]
        existing.extend(chunks)
        self._save_snapshot(existing)

        # 重建 BM25
        self.bm25_index.rebuild([Chunk(c.doc_id, c.text, c.metadata, c.chunk_index) for c in existing])

        return {"doc_id": doc.doc_id, "filename": doc.metadata["filename"], "chunks": len(chunks), "chars": doc.metadata["chars"]}

    # ---------- 目录批量入库 ----------
    def ingest_directory(self, dir_path: str | Path) -> list[dict]:
        d = Path(dir_path)
        files = sorted([p for p in d.rglob("*") if p.suffix.lower() in (".md", ".markdown", ".txt", ".pdf", ".docx")])
        results = []
        for p in tqdm(files, desc="ingesting"):
            try:
                results.append(self.ingest_document(p))
            except Exception as e:
                results.append({"doc_id": p.stem, "filename": p.name, "error": str(e)})
        return results

    # ---------- 全量重建 ----------
    def rebuild_all(self):
        """清空向量库，从 kb_dir 全量重建（用于数据一致性修复）。"""
        self.vector_store.reset()
        self._save_snapshot([])
        return self.ingest_directory(self.cfg.kb_dir)

    # ---------- 删除 ----------
    def delete_document(self, doc_id: str) -> int:
        removed = self.vector_store.delete_by_doc(doc_id)
        existing = [c for c in self._load_snapshot() if c.doc_id != doc_id]
        self._save_snapshot(existing)
        self.bm25_index.rebuild([Chunk(c.doc_id, c.text, c.metadata, c.chunk_index) for c in existing])
        return removed

    def list_documents(self) -> list[dict]:
        seen = {}
        for c in self._load_snapshot():
            seen.setdefault(c.doc_id, c.metadata)
        return [{"doc_id": k, **v} for k, v in seen.items()]

    # 兼容测试：tokenize 导出
    tokenize_fn = staticmethod(tokenize)
