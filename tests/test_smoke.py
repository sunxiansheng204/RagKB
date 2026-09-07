"""冒烟测试：不依赖外部模型/网络，仅验证纯逻辑模块。

运行：python -m pytest tests -q
"""
from __future__ import annotations

from app.core.reranker import RankedChunk
from app.rag import prompts
from app.retrieval.bm25 import BM25Index
from app.retrieval.chunker import split_text
from app.rag.citations import extract_citations, build_sources


def test_split_text_overlap():
    text = "第一段落内容。" * 50
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    assert all(len(c) <= 120 for c in chunks)


def test_split_text_empty():
    assert split_text("   ", 100, 10) == []


def test_bm25_search():
    idx = BM25Index()
    idx.rebuild(
        [
            RankedChunk("a", "向量检索擅长语义相似度匹配适合模糊查询", {}),
            RankedChunk("b", "关键词匹配是精确检索的重要手段关键词命中得分高", {}),
            RankedChunk("c", "搜索引擎倒排索引用于加速文档匹配", {}),
            RankedChunk("d", "重排序算法进一步提升结果相关性", {}),
        ]
    )
    hits = idx.search("关键词", top_k=2)
    assert hits and hits[0].doc_id == "b"


def test_bm25_empty():
    idx = BM25Index()
    assert idx.search("x", 5) == []


def test_extract_citations():
    assert extract_citations("结果见[1]与[3][3]") == [1, 3]
    assert extract_citations("没有引用") == []


def test_build_sources():
    results = [{"id": "x1", "text": "hello", "metadata": {"filename": "a.md"}, "combined_score": 0.8}]
    sources = build_sources(results)
    assert sources[0]["index"] == 1
    assert sources[0]["filename"] == "a.md"


def test_build_context():
    results = [{"text": "片段一", "metadata": {"filename": "a.md"}}]
    ctx = prompts.build_context(results)
    assert "[1]" in ctx and "片段一" in ctx
