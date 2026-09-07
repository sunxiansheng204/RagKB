"""引用溯源：把检索命中的 chunk 与模型回答中的 [n] 标注对齐，返回可溯源来源。

溯源是 RAG 生产化的关键能力：
- 让用户一键看到答案依据的原文片段；
- 后续可做"无引用即降权"的幻觉抑制策略。
"""
from __future__ import annotations

import re


def extract_citations(answer: str) -> list[int]:
    """从回答中提取所有 [n] 引用编号，去重保序。"""
    nums = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    return sorted(set(nums))


def build_sources(results: list[dict]) -> list[dict]:
    """把检索结果序列化成对外暴露的来源列表，编号从 1 开始。"""
    sources = []
    for i, r in enumerate(results, start=1):
        metadata = r.get("metadata") or {}
        sources.append(
            {
                "index": i,
                "doc_id": r.get("id") or metadata.get("doc_id", ""),
                "filename": (metadata.get("filename", "") if (isinstance(metadata.get("filename"), str)) else ""),
                "source_path": metadata.get("source", ""),
                "snippet": (r.get("text") or "")[:300],
                "score_combined": round(float(r.get("combined_score", 0)), 6),
            }
        )
    return sources


def cited_sources(answer: str, results: list[dict]) -> list[dict]:
    """返回回答实际引用了的来源子集。"""
    cited = extract_citations(answer)
    all_sources = build_sources(results)
    if not cited:
        return all_sources[:3]
    return [s for s in all_sources if s["index"] in cited]
