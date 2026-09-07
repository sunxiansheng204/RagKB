"""知识库文档管理接口：上传 / 列表 / 删除 / 全量重建。"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.dependencies import get_pipeline
from app.ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/docs", tags=["docs"])

SAFE_EXTS = {".md", ".markdown", ".txt", ".pdf", ".docx"}


@router.post("/upload")
async def upload_doc(
    file: UploadFile = File(...),
    pipeline: IngestionPipeline = Depends(get_pipeline),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SAFE_EXTS:
        raise HTTPException(400, f"仅支持 {sorted(SAFE_EXTS)} 格式")
    target = pipeline.cfg.kb_dir / _safe_name(file.filename)
    target.write_bytes(await file.read())
    result = pipeline.ingest_document(target)
    return {"message": "入库成功", **result}


@router.get("")
def list_docs(pipeline: IngestionPipeline = Depends(get_pipeline)):
    docs = pipeline.list_documents()
    return {"total": len(docs), "documents": docs}


@router.delete("/{doc_id}")
def delete_doc(doc_id: str, pipeline: IngestionPipeline = Depends(get_pipeline)):
    removed = pipeline.delete_document(doc_id)
    return {"message": f"已移除 {removed} 个切片", "doc_id": doc_id}


@router.post("/rebuild")
def rebuild(pipeline: IngestionPipeline = Depends(get_pipeline)):
    results = pipeline.rebuild_all()
    return {"message": "全量重建完成", "docs": results}


def _safe_name(name: str) -> str:
    name = re.sub(r"[^\w\-\.\u4e00-\u9fff]", "_", name)
    return name or "doc.txt"
