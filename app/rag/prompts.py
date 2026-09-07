"""Prompt 模板：RAG 系统提示词与引用约束。

设计要点：
- 上下文按 [1][2][3] 编号，要求模型回答时标注引用来源，支持溯源；
- 强制"只依据给定上下文回答，不知道就直说"，抑制幻觉；
- 模板做参数化，方便后续按场景微调或换成更复杂的 few-shot 结构。
"""

SYSTEM_TEMPLATE = """你是一名企业知识库问答助手。请严格遵循以下要求：

1. 只依据下方提供的【参考文档】回答用户问题，严禁编造参考文档中不存在的信息。
2. 回答时在句末用方括号标注信息来源编号，例如 [1][2]，编号必须对应下方给出的来源。
3. 如果参考文档无法回答该问题，请直接回答"根据现有知识库无法回答该问题"，不要强行作答。
4. 回答使用与用户提问相同的语言，结构清晰，适当分点。
5. 如果问题涉及对比，请用表格呈现。

【参考文档】
{context}
"""

USER_TEMPLATE = """用户问题：{question}"""


def build_messages(question: str, context_text: str, history: list[dict] | None = None) -> list[dict]:
    """组装 ChatML 消息序列，history 为 [{"role": "user"/"assistant", "content": ...}]。"""
    messages = [{"role": "system", "content": SYSTEM_TEMPLATE.format(context=context_text)}]
    for item in (history or []):
        role = item.get("role")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": item.get("content", "")})
    messages.append({"role": "user", "content": USER_TEMPLATE.format(question=question)})
    return messages


def build_context(results: list[dict]) -> str:
    """将检索结果渲染为带编号的上下文文本。"""
    blocks = []
    for i, r in enumerate(results, start=1):
        filename = (r.get("metadata") or {}).get("filename", "unknown")
        snippet = r["text"].strip().replace("\n", " ") if r.get("text") else ""
        blocks.append(f"[{i}] 来源文件: {filename}\n{snippet[:2000]}")
    return "\n\n".join(blocks)
