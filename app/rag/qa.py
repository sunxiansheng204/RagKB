"""问答编排层：检索 -> 组装上下文 -> 生成 -> 溯源，串成完整 RAG 管线。

对外暴露 answer() 与 answer_stream() 两个入口。
answer_stream 把"检索阶段"和"生成阶段"用事件流区分，前端可分别渲染
"正在检索… / 正在生成…"，这也是生产级 RAG 产品常见的交互体验。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections.abc import Generator

from app.core.embedding import EmbeddingEngine
from app.core.llm import LLMClient
from app.rag import prompts
from app.rag.citations import build_sources
from app.retrieval.hybrid import HybridRetriever


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    context_count: int = 0


class RAGEngine:
    def __init__(self, retriever: HybridRetriever, llm: LLMClient):
        self.retriever = retriever
        self.llm = llm

    def answer(self, question: str, history: list[dict] | None = None) -> Answer:
        results = self.retriever.retrieve(question)
        context = prompts.build_context(results)
        messages = prompts.build_messages(question, context, history)
        answer_text = self.llm.chat(messages)
        sources = build_sources(results)
        return Answer(question=question, answer=answer_text, sources=sources, context_count=len(results))

    def answer_stream(self, question: str, history: list[dict] | None = None) -> Generator[dict, None, None]:
        """SSE 事件流：先发 sources 事件，再逐段发 delta 事件，最后发 done。"""
        import asyncio

        # 同步检索（或用线程池；对 demo 足够）
        results = self.retriever.retrieve(question)
        context = prompts.build_context(results)
        messages = prompts.build_messages(question, context, history)
        sources = build_sources(results)

        yield {"type": "sources", "data": sources}

        gen = self.llm.chat(messages, stream=True)
        full = []
        for delta in gen:
            full.append(delta)
            yield {"type": "delta", "data": delta}
        yield {"type": "done", "data": {"answer": "".join(full), "context_count": len(results)}}
