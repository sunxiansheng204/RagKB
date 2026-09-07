"""评估数据集生成：从既有知识块中自动构造 (question, reference) 测试样本。

方法：对每个知识块，请 LLM 基于该块内容生成一个可回答的问题；
reference 即该知识块的原文，用于判断检索命中率。
生成后导出为 JSONL，供 runner 复用，避免每次评估都重复生成。
"""
from __future__ import annotations

import json

from pathlib import Path

from app.core.llm import LLMClient

GEN_PROMPT = """基于下面的知识片段，生成 1 个只能从该片段回答出的具体中文问题。
要求：问题独立完整、不带上下文前缀、只输出问题本身（不要输出解释、编号或引号）。

知识片段：
{text}
"""


def generate_dataset(llm: LLMClient, chunks: list, out_path: Path, max_samples: int = 30) -> list[dict]:
    samples = []
    for chunk in chunks[:max_samples]:
        text = chunk.text if hasattr(chunk, "text") else chunk["text"]
        if len(text.strip()) < 30:
            continue
        try:
            q = llm.chat([{"role": "user", "content": GEN_PROMPT.format(text=text[:1200])}], temperature=0.7, max_tokens=64).strip()
            if q and "无法" not in q and len(q) >= 6:
                samples.append({"question": q, "reference": text.strip()})
        except Exception:
            continue
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return samples
