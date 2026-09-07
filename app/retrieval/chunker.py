"""文档切分：递归字符切分 + 重叠窗口。

切分策略对 RAG 效果影响巨大：
- 块过大：上下文被噪声稀释，检索精度下降；
- 块过小：语义不完整，命中上下文碎片化；
- 重叠窗口：弥补切分边界处的语义断裂，是工程里成本最低也最有效的优化之一。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import md5


@dataclass
class Chunk:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_index: int = 0

    @property
    def chunk_id(self) -> str:
        return md5(f"{self.doc_id}:{self.chunk_index}".encode("utf-8")).hexdigest()


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按段落优先、递归降级到字符级的切分，保留重叠窗口。"""
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    # 第一层：按空行拆成段落，再尽量合并进 chunk_size 窗口
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    buffer: list[str] = []
    buff_len = 0

    def flush():
        nonlocal buffer, buff_len
        if buffer:
            joined = "\n\n".join(buffer)
            chunks.append(joined)
            # 保留尾部作为重叠文本，注意不要与下一批重复
            tail = _tail_overlap(joined, overlap)
            buffer = [tail] if tail else []
            buff_len = len(tail) if tail else 0

    for para in paragraphs:
        # 单段太长，直接按字符切
        if len(para) > chunk_size:
            flush()
            for piece in _split_long(para, chunk_size, overlap):
                chunks.append(piece)
            continue
        if buff_len + len(para) + 2 <= chunk_size:
            buffer.append(para)
            buff_len += len(para) + 2
        else:
            # 当前缓冲放不下了：尝试只看这一段的子句能否并入尾巴，否则换新块
            last_sentences = list(_sentence_walk(para, chunk_size))
            if last_sentences and buff_len + len(last_sentences[-1]) + 2 <= chunk_size:
                buffer.append(last_sentences[-1])
                buff_len += len(last_sentences[-1]) + 2
                para = "".join(last_sentences[:-1])
            flush()
            if para.strip():
                buffer.append(para.strip())
                buff_len = len(para.strip())

    flush()
    return [c for c in chunks if c.strip()]


def _tail_overlap(text: str, overlap: int) -> str:
    """取文本末尾 up to overlap 字作为重叠段（按字符）。"""
    if overlap <= 0 or len(text) <= overlap:
        return ""
    return text[-overlap:]


def _split_long(para: str, chunk_size: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    n = len(para)
    step = max(chunk_size - overlap, 1)
    while start < n:
        piece = para[start : start + chunk_size]
        pieces.append(piece)
        start += step
    return pieces


def _sentence_walk(para: str, chunk_size: int) -> list[str]:
    """把段落按句子切分，供防溢出时做缓冲合并依据。"""
    import re

    parts = re.split(r"(?<=[。！？!?])", para)
    return [p for p in parts if p.strip()]


def chunk_document(doc_id: str, text: str, metadata: dict, chunk_size: int, overlap: int) -> list[Chunk]:
    pieces = split_text(text, chunk_size, overlap)
    return [
        Chunk(doc_id=doc_id, text=p, metadata=metadata, chunk_index=i) for i, p in enumerate(pieces)
    ]
