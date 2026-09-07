"""纯命令行问答演示：不依赖 FastAPI，直接在终端体验 RAG 全流程。

用法：
    python -m scripts.demo
    python -m scripts.demo "什么是 RRF 融合？"
"""
from __future__ import annotations

import sys

from app.dependencies import get_rag, get_vector_store


def main():
    rag = get_rag()
    print("=" * 60)
    print("RagKB 命令行问答（Ctrl+C 退出）")
    print(f"知识库切片数: {get_vector_store().count()}")
    print("=" * 60)

    question = " ".join(sys.argv[1:]) or None
    if question:
        _ask(rag, question)
        return

    while True:
        try:
            q = input("\n> 请输入问题：").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                break
            _ask(rag, q)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[错误] {e}")


def _ask(rag, q: str):
    result = rag.answer(q)
    print("\n—— 回答 ——")
    print(result.answer)
    print("\n—— 引用来源 ——")
    for s in result.sources:
        print(f"[{s['index']}] {s['filename']} (score={s['score_combined']})")
    print(f"（共使用 {result.context_count} 个上下文切片）")


if __name__ == "__main__":
    main()
