"""文档入库 / 重建 CLI。

用法：
    python -m scripts.ingest <file 或 dir> [--rebuild]
    python -m scripts.ingest --rebuild      # 清空并按 data/documents 全量重建
"""
from __future__ import annotations

import argparse

from app.dependencies import get_embedding, get_vector_store, get_bm25
from app.ingestion.pipeline import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(description="RagKB 入库 CLI")
    parser.add_argument("path", nargs="?", default=None, help="文件或目录路径；缺省配合 --rebuild 使用")
    parser.add_argument("--rebuild", action="store_true", help="清空全量重建 data/documents")
    args = parser.parse_args()

    pipeline = IngestionPipeline(get_embedding(), get_vector_store(), get_bm25())

    if args.rebuild:
        results = pipeline.rebuild_all()
        ok = [r for r in results if "error" not in r]
        print(f"全量重建完成：{len(ok)} 个文档入库")
        for r in results:
            print(" ", r)
        return

    if not args.path:
        parser.error("请提供 path 或使用 --rebuild")

    from pathlib import Path

    p = Path(args.path)
    if p.is_dir():
        results = pipeline.ingest_directory(p)
        for r in results:
            print(r)
    elif p.is_file():
        print(pipeline.ingest_document(p))
    else:
        parser.error(f"路径不存在: {args.path}")


if __name__ == "__main__":
    main()
