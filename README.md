# RagKB · 企业级 RAG 知识库问答系统

一个**可工程化落地**的检索增强生成（RAG）知识库问答系统，覆盖「文档入库 → 混合检索 → 引用溯源 → 流式生成 → 效果评估」全链路。适合作为 **AI 大模型应用工程师 / LLM 应用开发**方向的面试项目与简历亮点。

## 技术亮点（一句话版）

- **混合检索**：向量检索（ChromaDB + BGE Embedding）+ BM25 关键词检索（jieba 分词），RRF 倒数排名融合，解决专有名词漏召回；
- **引用溯源**：模型回答自动带 `[1][2]` 来源编号，可一键回看原文片段，抑制幻觉、支持审计；
- **流式输出**：SSE 事件流，前端可区分"检索中 / 生成中"两阶段体验；
- **效果评估**：自建 LLM-as-Judge 指标体系（忠实度 / 回答相关度 / 检索命中率），支持从知识块自动生成评估集；
- **双后端可插拔**：Embedding 支持本地 `sentence-transformers` 与 OpenAI 兼容 API；LLM 统一走 OpenAI 兼容协议（Ollama / DeepSeek / 火山方舟 / OpenAI 均可）。

## 架构概览

```
        ┌────────────────────────────── 服务端 FastAPI ──────────────────────────────┐
        │                                                                            │
 用户 ──►  Streamlit 前端 ──►  /chat  /chat/stream /docs/* /eval/*                  │
        │                                                                            │
        │   ┌────────────────────────────────────────────────────────────────┐      │
        │   │  RAG 管线                                                       │      │
        │   │  问题 ─► 混合检索(向量+BM25) ─► RRF 融合 ─► Prompt 组装          │      │
        │   │        ─► LLM 生成(引用标注) ─► 引用溯源 ─► 回答                  │      │
        │   └────────────────────────────────────────────────────────────────┘      │
        │            ▲                    ▲                    ▲                    │
        │   ┌────────┴────────┐  ┌────────┴────────┐  ┌────────┴────────┐           │
        │   │ 向量库 ChromaDB │  │ BM25 倒排索引    │  │ 评估器 Eval      │          │
        │   └─────────────────┘  └─────────────────┘  └─────────────────┘          │
        │            ▲                                                            │
        │   ┌────────┴────────┐                                                   │
        │   │ 入库管线 Ingest │  md/txt/pdf/docx → 切分 → 向量化 → 双路索引         │
        │   └─────────────────┘                                                   │
        └────────────────────────────────────────────────────────────────────────────┘
```

## 快速开始

环境要求：Python 3.10+，推荐 3.10/3.11。

```bash
cd D:\RagKB

# 1. 建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置 LLM（任选其一）
cp .env.example .env
# 默认指向本地 Ollama（qwen2.5:7b）；改用云端时填：
#   LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
#   EMBEDDING_BACKEND=openai（或保持本地 sentence-transformers）

# 3. 启动后端（首次启动自动灌入 data/documents 示例文档并下载 BGE 模型）
python -m app.main
# 或：uvicorn app.main:app --host 0.0.0.0 --port 8000

# 4.（可选）启动前端演示
streamlit run web/app.py

# 5. 命令行快速体验
python -m scripts.demo "什么是 RRF 融合？"
python -m scripts.ingest data/documents          # 手动入库
python -m scripts.run_eval --force-gen           # 效果评估
```

接口速览（`http://127.0.0.1:8000/api-docs` 可在线调试）：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | /chat | 同步问答，返回答案 + 引用来源 |
| POST | /chat/stream | SSE 流式问答 |
| POST | /docs/upload | 上传 md/txt/pdf/docx |
| GET/DELETE | /docs, /docs/{id} | 列出 / 删除文档 |
| POST | /docs/rebuild | 全量重建索引 |
| POST | /eval/run | 运行 RAG 效果评估 |
| GET | /health | 健康检查 |

## 项目结构

```
RagKB/
├── app/
│   ├── main.py              # FastAPI 入口 + 启动自动初始化
│   ├── dependencies.py      # 依赖注入 / 单例装配
│   ├── api/                 # REST 接口（问答 / 文档 / 评估）
│   ├── core/                # Embedding / LLM / Reranker 接入层
│   ├── retrieval/           # 切分 / 向量库 / BM25 / 混合检索(RRF)
│   ├── ingestion/           # 文档加载 / 入库流水线
│   ├── rag/                 # Prompt 工程 / 问答编排 / 引用溯源
│   └── eval/                # 评估集生成 / LLM-as-Judge 指标
├── web/app.py               # Streamlit 前端演示
├── scripts/                 # CLI：入库 / 评估 / 命令行问答
├── tests/                   # 冒烟测试（无外部依赖）
├── config/settings.py       # pydantic-settings 配置
└── data/                    # 知识库文档 / 向量持久化 / 评估报告
```

## 简历项目描述（可直接改写） 

> **企业级 RAG 知识库问答系统（个人项目）**
> - 独立设计并实现一套端到端 RAG 问答系统，支持多格式文档（PDF/Word/Markdown）入库、增量索引与全量重建；
> - 采用「向量检索 + BM25 关键词检索 + RRF 融合」混合检索方案，解决专名漏召回问题；引入 BGE 重排序进一步提升 top-K 精度；
> - 构建引用溯源机制，回答自动标注来源编号并可回看原文，有效抑制幻觉；通过 SSE 实现检索/生成两阶段流式输出，优化交互体验；
> - 自建 LLM-as-Judge 评估体系（忠实度、回答相关度、检索命中率），并从知识块自动生成评估集，支撑检索与提示词的可量化迭代；
> - 技术栈：Python · FastAPI · ChromaDB · sentence-transformers · OpenCLIP/RRF · Streamlit · OpenAI 兼容协议。

## 面试高频考点（项目可深挖）

1. 为什么向量检索 + BM25 混合？RRF 为什么用排名而不是分数？
2. 切分块大小/重叠窗口如何调优？对检索精度的影响是什么？
3. 如何量化"RAG 效果变好了"？评估集从哪来？忠实度与相关度的区别？
4. 引用溯源怎么实现？它如何抑制幻觉？
5. 流式输出怎么实现？SSE 与 WebSocket 的取舍？
6. Embedding 本地与 API 如何优雅切换？维度不一致如何处理？
7. 知识更新：如何只更新受影响切片而不是全量重建？
8. 生产化还差什么？（缓存、可观测、限流、多租户隔离、权限）

## Roadmap（加分方向）

- [ ] 检索结果重排序（`USE_RERANKER=true` 已预留，需下载 bge-reranker-base）
- [ ] 父子块 / 语义切分
- [ ] Query 改写（拆解、纠错、扩写）
- [ ] 对接 RAGAS 评估框架
- [ ] 知识库权限与多租户隔离
- [ ] 缓存（向量缓存 / 回答缓存）与可观测性上报

## 说明

- 本项目为标准软件交付，不携带任何密钥，LLM / Embedding 均通过 `.env` 由使用者自行配置，可安全用于简历项目与开源。
