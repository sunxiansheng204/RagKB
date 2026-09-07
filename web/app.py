"""RagKB 前端演示（Streamlit）：问答 + 引用溯源 + 知识库管理 + 评估。

启动方式（需先启动后端）：
    streamlit run web/app.py
"""
from __future__ import annotations

import requests
import streamlit as st

st.set_page_config(page_title="RagKB · 企业级 RAG 知识库", layout="wide")
st.title("RagKB · 企业级 RAG 知识库问答系统")

API_BASE = st.sidebar.text_input("后端地址", "http://127.0.0.1:8000")
with st.sidebar.expander("快速检查后端"):
    if st.button("查看 /health"):
        try:
            st.json(requests.get(f"{API_BASE}/health", timeout=5).json())
        except Exception as e:
            st.error(f"连接失败: {e}")

tab_chat, tab_docs, tab_eval = st.tabs(["智能问答", "知识库管理", "效果评估"])

# ---------------- 智能问答 ----------------
with tab_chat:
    st.subheader("基于知识库的问答（引用可溯源）")
    if "history" not in st.session_state:
        st.session_state.history = []
    for h in st.session_state.history[-6:]:
        role = "用户" if h["role"] == "user" else "助手"
        st.markdown(f"**{role}**：{h['content']}")
    question = st.text_area("输入问题", placeholder="例如：如何选择 RAG 的切分块大小？")
    if st.button("提问", type="primary"):
        try:
            resp = requests.post(
                f"{API_BASE}/chat",
                json={"question": question, "history": st.session_state.history[-4:]},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            st.markdown(data["answer"])
            with st.expander(f"查看引用来源（{len(data['sources'])} 条）"):
                for s in data["sources"]:
                    st.markdown(f"**[{s['index']}] {s['filename']}**（综合分 {s['score_combined']}）")
                    st.caption(s["snippet"][:200])
            st.session_state.history.append({"role": "user", "content": question})
            st.session_state.history.append({"role": "assistant", "content": data["answer"][:500]})
        except Exception as e:
            st.error(f"调用失败：{e}")

# ---------------- 知识库管理 ----------------
with tab_docs:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("上传文档")
        uploaded = st.file_uploader("选择文件（md/txt/pdf/docx）", type=["md", "txt", "pdf", "docx"])
        if uploaded and st.button("入库", type="primary"):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            r = requests.post(f"{API_BASE}/docs/upload", files=files, timeout=300)
            st.json(r.json())
    with col2:
        st.subheader("现有文档")
        if st.button("刷新列表"):
            docs = requests.get(f"{API_BASE}/docs", timeout=10).json()
            st.session_state.docs = docs
        docs = st.session_state.get("docs", {})
        for d in docs.get("documents", []):
            st.markdown(f"- **{d['filename']}**（{d.get('chars', 0)} 字）")

# ---------------- 效果评估 ----------------
with tab_eval:
    st.subheader("RAG 效果评估（LLM-as-Judge）")
    max_samples = st.slider("样本量", 5, 50, 15)
    if st.button("开始评估", type="primary"):
        with st.spinner("评估中，可能需要几分钟…"):
            r = requests.post(f"{API_BASE}/eval/run", json={"max_samples": max_samples}, timeout=1800)
            r.raise_for_status()
            report = r.json()
        if "error" in report:
            st.error(report["error"])
        else:
            st.success("评估完成")
            for k, name in [("faithfulness", "忠实度"), ("answer_relevance", "回答相关度"), ("retrieval_hit", "检索命中率"), ("rag_quality", "综合质量")]:
                st.metric(name, f"{report[k]:.2%}")
