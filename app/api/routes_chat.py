"""问答接口：同步问答 + SSE 流式问答。"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.dependencies import get_rag
from app.rag.qa import RAGEngine

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    history: list[dict] | None = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    context_count: int


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, rag: RAGEngine = Depends(get_rag)):
    if not req.question.strip():
        raise HTTPException(400, "question 不能为空")
    result = rag.answer(req.question, req.history)
    return ChatResponse(**result.__dict__)


@router.post("/stream")
async def chat_stream(req: ChatRequest, rag: RAGEngine = Depends(get_rag)):
    if not req.question.strip():
        raise HTTPException(400, "question 不能为空")

    def event_stream():
        for evt in rag.answer_stream(req.question, req.history):
            yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
