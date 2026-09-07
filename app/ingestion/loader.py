"""文档加载器：支持 Markdown / 纯文本 / PDF / Word，统一转成纯文本+元数据。

loader 是知识库的上游入口，生产环境还需要补：扫描件 OCR、表格结构提取、
HTML 去噪、页眉页脚过滤等。这里给出一个可扩展的注册式实现。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.settings import settings


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict


class DocumentLoader:
    def __init__(self, cfg=None):
        self.cfg = cfg or settings

    def load(self, path: str | Path) -> Document:
        p = Path(path)
        ext = p.suffix.lower()
        if ext in (".md", ".markdown", ".txt"):
            text = p.read_text(encoding="utf-8", errors="ignore")
        elif ext == ".pdf":
            text = self._load_pdf(p)
        elif ext == ".docx":
            text = self._load_docx(p)
        else:
            raise ValueError(f"暂不支持的文件类型: {ext}（支持 md/txt/pdf/docx）")

        metadata = {
            "source": str(p),
            "filename": p.name,
            "ext": ext,
            "size_bytes": p.stat().st_size,
            "chars": len(text),
        }
        return Document(doc_id=p.stem, text=text, metadata=metadata)

    def _load_pdf(self, p: Path) -> str:
        from pypdf import PdfReader

        reader = PdfReader(str(p))
        parts = []
        for i, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            parts.append(f"[第{i + 1}页]\n{t}")
        return "\n\n".join(parts)

    def _load_docx(self, p: Path) -> str:
        from docx import Document as DocxDocument

        doc = DocxDocument(str(p))
        return "\n\n".join(par.text for par in doc.paragraphs if par.text.strip())
