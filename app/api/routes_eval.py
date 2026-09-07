"""评估接口：触发一次全量 RAG 效果评估，返回汇总报告。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.dependencies import get_eval_runner, get_pipeline
from app.eval.runner import EvalRunner
from app.ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post("/run")
def run_eval(
    max_samples: int = Query(20, ge=1, le=100),
    force_gen: bool = Query(False),
    runner: EvalRunner = Depends(get_eval_runner),
    pipeline: IngestionPipeline = Depends(get_pipeline),
):
    chunks = pipeline._load_snapshot()
    if not chunks:
        return {"error": "知识库为空，请先入库文档"}
    report = runner.run(chunks, max_samples=max_samples, force_gen=force_gen)
    return report
