"""效果评估 CLI：自动构造评估集并输出指标报告。

用法：
    python -m scripts.run_eval --max-samples 20 --force-gen
    python -m scripts.run_eval              # 复用已有评估集
"""
from __future__ import annotations

import argparse

from app.dependencies import get_eval_runner, get_pipeline


def main():
    parser = argparse.ArgumentParser(description="RagKB 效果评估 CLI")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--force-gen", action="store_true", help="强制重新生成评估集")
    args = parser.parse_args()

    runner = get_eval_runner()
    chunks = get_pipeline()._load_snapshot()
    if not chunks:
        print("知识库为空，请先入库（python -m scripts.ingest <path>）")
        return
    report = runner.run(chunks, max_samples=args.max_samples, force_gen=args.force_gen)
    print("评估结果：")
    for k, name in [("faithfulness", "忠实度"), ("answer_relevance", "回答相关度"), ("retrieval_hit", "检索命中率"), ("rag_quality", "综合质量")]:
        print(f"  {name}: {report[k]:.2%}")
    print(f"  平均延迟: {report['avg_latency_s']}s")
    print(f"报告已写入: {report_path()}")


def report_path():
    from config.settings import settings

    return settings.eval_report_dir / "eval_report.md"


if __name__ == "__main__":
    main()
