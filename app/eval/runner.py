"""评估执行器：构建/加载评估集 -> 逐条跑 RAG -> 汇总指标 -> 输出报告。

输出两类产物：
- data/reports/eval_report.json ：机器可读
- data/reports/eval_report.md  ：人可读 Markdown，可直接贴进简历/汇报
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from app.core.embedding import EmbeddingEngine
from app.core.llm import LLMClient
from app.eval.dataset import generate_dataset
from app.eval.metrics import evaluate_sample
from app.rag.qa import RAGEngine
from config.settings import settings


class EvalRunner:
    def __init__(self, rag: RAGEngine, llm: LLMClient, embedding: EmbeddingEngine, cfg=None):
        self.rag = rag
        self.llm = llm
        self.embedding = embedding
        self.cfg = cfg or settings

    def run(self, chunks: list, dataset_path: Path | None = None, max_samples: int = 20, force_gen: bool = False) -> dict:
        if dataset_path is None:
            dataset_path = self.cfg.data_dir / "index" / "eval_dataset.jsonl"
        if force_gen or not dataset_path.exists():
            samples = generate_dataset(self.llm, chunks, dataset_path, max_samples=max_samples)
        else:
            samples = [json.loads(line) for line in open(dataset_path, encoding="utf-8") if line.strip()]

        results = []
        t0 = time.time()
        for i, s in enumerate(samples, 1):
            answer = self.rag.answer(s["question"])
            metrics = evaluate_sample(
                self.llm,
                self.embedding,
                s,
                answer.answer,
                [r["text"] for r in self.rag.retriever.retrieve(s["question"])],
            )
            results.append({"question": s["question"], "answer": answer.answer, **metrics, "latency_s": round(time.time() - t0 - (0 if i == 1 else results[-1]["latency_s"] * 0), 3)})

        report = self._aggregate(results, chunks=len(samples), elapsed=time.time() - t0)
        self._write_report(report, results)
        return report

    def _aggregate(self, results: list[dict], **extra) -> dict:
        keys = ["faithfulness", "answer_relevance", "retrieval_hit"]
        agg = {}
        for k in keys:
            vals = [r[k] for r in results if r.get(k) is not None]
            agg[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
        agg["rag_quality"] = round(sum(agg[k] for k in keys) / len(keys), 4)
        agg.update(extra)
        agg["avg_latency_s"] = round(sum(r.get("latency_s", 0) for r in results) / len(results), 3) if results else 0.0
        return agg

    def _write_report(self, report: dict, results: list[dict]):
        self.cfg.eval_report_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.cfg.eval_report_dir / "eval_report.json"
        md_path = self.cfg.eval_report_dir / "eval_report.md"
        json_path.write_text(json.dumps({"summary": report, "details": results}, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = ["# RAG 效果评估报告", "", f"- 样本量：{report.get('chunks', 0)}", f"- 总耗时：{report.get('elapsed', 0):.1f}s", f"- 平均延迟：{report['avg_latency_s']}s", ""]
        lines += ["## 总览", "", "| 指标 | 得分 |", "| --- | --- |"]
        for k in ("faithfulness", "answer_relevance", "retrieval_hit", "rag_quality"):
            name = {"faithfulness": "忠实度", "answer_relevance": "回答相关度", "retrieval_hit": "检索命中率", "rag_quality": "综合质量"}[k]
            lines.append(f"| {name} | {report[k]:.2%} |")
        lines += ["", "## 明细", "", "| # | 问题 | 忠实度 | 相关度 | 命中 |", "| --- | --- | --- | --- | --- |"]
        for i, r in enumerate(results, 1):
            q = r["question"][:40].replace("|", "\\|")
            lines.append(f"| {i} | {q} | {r['faithfulness']:.2f} | {r['answer_relevance']:.2f} | {r['retrieval_hit']:.0f} |")
        md_path.write_text("\n".join(lines), encoding="utf-8")
