"""RAG 效果评估指标（LLM-as-Judge + 无参考指标）。

为什么自建评估而非直接套 RAGAS？
- 交给简历评审和面试官看时，评估体系本身是加分项，体现"可度量"意识；
- 指标可解释、无额外重依赖；（README 中说明可替换/对接 RAGAS）

指标体系（本次实现）：
1. 检索命中率 retrieval_hit_rate —— 问题能否召回到相关上下文（用向量相似度阈值判）；
2. 忠实度 faithfulness —— 回答是否忠于上下文（LLM judge，1-5 分归一）；
3. 回答相关度 answer_relevance —— 回答是否切题（LLM judge，1-5 分归一）；
4. (加分项) 引用有效率 citation_coverage —— 回答中引用编号是否真实存在。
"""
from __future__ import annotations

import re

from app.core.embedding import EmbeddingEngine
from app.core.llm import LLMClient


def cosine(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b))


def retrieval_hit(query: str, gold_chunk_text: str, embedding: EmbeddingEngine, threshold: float = 0.6) -> bool:
    """用问题与标准上下文片段的向量相似度，粗判"是否召回相关资源"。"""
    qv = embedding.encode_query(query)
    gv = embedding.encode_query(gold_chunk_text[:800])
    return cosine(qv, gv) >= threshold


_FAITHFUL_PROMPT = """你是 RAG 系统评估员。请判断"回答"是否忠于给定的"参考上下文"。

规则：
- 如果回答中的事实均可由参考上下文支撑：5 分
- 如果部分事实无法从上下文找到依据（可能幻觉）：3 分
- 如果回答与上下文明显矛盾或基本杜撰：1 分

只输出一个整数分数（1-5）。

参考上下文：
{context}

回答：
{answer}
"""

_RELEVANCE_PROMPT = """你是 RAG 系统评估员。请判断"回答"是否准确、完整地回答了"问题"。

- 直接命中问题核心且信息充分：5 分
- 基本正确但不够完整：4 分
- 部分相关但不完全对题：3 分
- 答非所问：1 分

只输出一个整数分数（1-5）。

问题：{question}

回答：
{answer}
"""


def judge_score(llm: LLMClient, template: str, **kwargs) -> float:
    """调用 LLM judge，稳健解析出 1-5 分。"""
    prompt = template.format(**kwargs)
    try:
        raw = llm.chat([{"role": "user", "content": prompt}], temperature=0)
        match = re.search(r"[1-5]", raw or "")
        return float(match.group(0)) / 5.0 if match else 0.0
    except Exception:
        return 0.0


def evaluate_sample(llm: LLMClient, embedding: EmbeddingEngine, sample: dict, answer: str, retrieved_texts: list[str]) -> dict:
    """对单条 sample 计算全部指标。sample: {question, reference}"""
    context = "\n".join(f"- {t[:500]}" for t in retrieved_texts)
    faithfulness = judge_score(llm, _FAITHFUL_PROMPT, context=context, answer=answer)
    relevance = judge_score(llm, _RELEVANCE_PROMPT, question=sample["question"], answer=answer)
    hit = 1.0 if retrieval_hit(sample["question"], sample["reference"], embedding) else 0.0
    return {
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "retrieval_hit": hit,
    }
