"""LLM 客户端：基于 OpenAI 兼容协议统一封装，支持同步与流式两种调用。

兼容任意 OpenAI 兼容的服务：
- 本地：Ollama / vLLM / LM Studio
- 云端：DeepSeek / 火山方舟 ARK / OpenAI / 通义
"""
from __future__ import annotations

from collections.abc import Generator
from typing import Any

from openai import OpenAI

from config.settings import settings


class LLMClient:
    def __init__(self, cfg=None):
        self.cfg = cfg or settings
        self.client = OpenAI(
            base_url=self.cfg.llm_base_url,
            api_key=self.cfg.llm_api_key or "EMPTY",
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str | Generator[str, None, None]:
        """统一 chat 入口。

        - stream=False: 返回完整文本字符串
        - stream=True : 返回逐段文本生成器
        """
        kwargs: dict[str, Any] = {
            "model": self.cfg.llm_model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature if temperature is not None else self.cfg.llm_temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        elif self.cfg.llm_max_tokens:
            kwargs["max_tokens"] = self.cfg.llm_max_tokens

        resp = self.client.chat.completions.create(**kwargs)

        if not stream:
            return resp.choices[0].message.content or ""

        def gen():
            for chunk in resp:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta

        return gen()
