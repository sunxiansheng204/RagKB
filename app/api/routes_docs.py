"""知识库文档管理接口：上传 / 列表 / 删除 / 全量重建。

安全说明：写操作（上传/删除/重建）统一挂载 Bearer Token 鉴权，
生产模式下必须携带 Authorization: Bearer <token>；上传接口另设单文件大小上限。
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.security import verify_token
from app.dependencies import get_pipeline
from app.ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/docs", tags=["docs"])

SAFE_EXTS = {".md", ".markdown", ".txt", ".pdf", ".docx"}
READ_CHUNK = 1024 * 1024  # 1MB，读入时按块累计，避免一次性整文件进内存


@router.post("/upload")
async def upload_doc(
    file: UploadFile = File(...),
    token: None = Depends(verify_token),
    pipeline: IngestionPipeline = Depends(get_pipeline),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SAFE_EXTS:
        raise HTTPException(400, f"仅支持 {sorted(SAFE_EXTS)} 格式")
    max_bytes = pipeline.cfg.max_upload_bytes
    # 预检：客户端若声明了 Content-Length 且超限，直接拒绝
    if file.size is not None and file.size > max_bytes:
        raise HTTPException(413, f"文件超过大小上限 {pipeline.cfg.max_upload_mb} MB")
    # 流式读取并强制大小上限，防止超大文件拖垮内存
    content = bytearray()
    while chunk := await file.read(READ_CHUNK):
        if len(content) + len(chunk) > max_bytes:
            raise HTTPException(413, f"文件超过大小上限 {pipeline.cfg.max_upload_mb} MB")
        content.extend(chunk)
    if not content:
        raise HTTPException(400, "文件为空")
    target = pipeline.cfg.kb_dir / _safe_name(file.filename)
    target.write_bytes(bytes(content))
    result = pipeline.ingest_document(target)
    return {"message": "入库成功", **result}


@router.get("")
def list_docs(
    token: None = Depends(verify_token),
    pipeline: IngestionPipeline = Depends(get_pipeline),
):
    docs = pipeline.list_documents()
    return {"total": len(docs), "documents": docs}


@router.delete("/{doc_id}")
def delete_doc(
    doc_id: str,
    token: None = Depends(verify_token),
    pipeline: IngestionPipeline = Depends(get_pipeline),
):
    removed = pipeline.delete_document(doc_id)
    return {"message": f"已移除 {removed} 个切片", "doc_id": doc_id}


@router.post("/rebuild")
def rebuild(
    token: None = Depends(verify_token),
    pipeline: IngestionPipeline = Depends(get_pipeline),
):
    results = pipeline.rebuild_all()
    return {"message": "全量重建完成", "docs": results}


def _safe_name(name: str) -> str:
    name = re.sub(r"[^\w\-\.\u4e00-\u9fff]", "_", name)
    return name or "doc.txt"
