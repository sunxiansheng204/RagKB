"""接口鉴权依赖：基于 Bearer Token 的轻量鉴权。

安全模型：
- 演示模式（RAGKB_API_TOKEN 为空）：不强制校验，便于本机快速体验；
  但应配合 host=127.0.0.1 与 CORS 白名单使用，禁止直接暴露到公网。
- 生产模式（RAGKB_API_TOKEN 非空）：所有接口（含 /health 之外的读写接口）
  必须携带请求头 `Authorization: Bearer <token>`，否则返回 401。

用法：在需要保护的接口参数中加入
    token: None = Depends(verify_token)
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from config.settings import settings

WWW_AUTH = "Bearer"


def verify_token(request: Request) -> None:
    token = (settings.api_token or "").strip()
    if not token:
        # 演示模式：不校验，交由部署边界（本机回环 + CORS 白名单）控制暴露面
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith(f"{WWW_AUTH} "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少认证信息，请携带 Authorization: Bearer <token>",
            headers={"WWW-Authenticate": WWW_AUTH},
        )
    if auth.split(" ", 1)[1].strip() != token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 无效",
            headers={"WWW-Authenticate": WWW_AUTH},
        )


def get_auth_state() -> dict:
    """供 /health 反映当前鉴权状态，便于部署时自查。"""
    return {"auth_enabled": bool((settings.api_token or "").strip())}


# 便于依赖注入链统一导出
__all__ = ["verify_token", "get_auth_state"]
