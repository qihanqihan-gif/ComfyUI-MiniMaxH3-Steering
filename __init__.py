"""MiniMax-H3 research nodes and frontend helpers for ComfyUI."""

import json

from aiohttp import web  # noqa: F401  (ComfyUI 自带依赖)

from . import prompt_director
from .easycache_safe import (
    NODE_CLASS_MAPPINGS as EASYCACHE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as EASYCACHE_NODE_DISPLAY_NAME_MAPPINGS,
)
from .media_tools import (
    NODE_CLASS_MAPPINGS as MEDIA_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MEDIA_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes import (
    NODE_CLASS_MAPPINGS as CACHE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as CACHE_NODE_DISPLAY_NAME_MAPPINGS,
)
from .profiler import (
    NODE_CLASS_MAPPINGS as PROFILER_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as PROFILER_NODE_DISPLAY_NAME_MAPPINGS,
)
from .prompt_director import (
    NODE_CLASS_MAPPINGS as PROMPT_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as PROMPT_NODE_DISPLAY_NAME_MAPPINGS,
)
from .prompt_modules import (
    NODE_CLASS_MAPPINGS as PROMPTMOD_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as PROMPTMOD_NODE_DISPLAY_NAME_MAPPINGS,
)
from .steering import (
    NODE_CLASS_MAPPINGS as STEERING_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as STEERING_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {
    **CACHE_NODE_CLASS_MAPPINGS,
    **EASYCACHE_NODE_CLASS_MAPPINGS,
    **MEDIA_NODE_CLASS_MAPPINGS,
    **PROFILER_NODE_CLASS_MAPPINGS,
    **PROMPT_NODE_CLASS_MAPPINGS,
    **PROMPTMOD_NODE_CLASS_MAPPINGS,
    **STEERING_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **CACHE_NODE_DISPLAY_NAME_MAPPINGS,
    **EASYCACHE_NODE_DISPLAY_NAME_MAPPINGS,
    **MEDIA_NODE_DISPLAY_NAME_MAPPINGS,
    **PROFILER_NODE_DISPLAY_NAME_MAPPINGS,
    **PROMPT_NODE_DISPLAY_NAME_MAPPINGS,
    **PROMPTMOD_NODE_DISPLAY_NAME_MAPPINGS,
    **STEERING_NODE_DISPLAY_NAME_MAPPINGS,
}

WEB_DIRECTORY = "./web"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]


def _register_api_routes():
    """前端"刷新模型列表"按钮的后端路由（OpenAI 兼容 /models 拉取）。"""
    try:
        from server import PromptServer

        routes = PromptServer.instance.routes
    except Exception:  # noqa: BLE001  (非 ComfyUI 环境导入时忽略)
        return

    @routes.post("/minimaxh3lab/api/models")
    async def _fetch_models(request):
        try:
            body = await request.json()
        except (json.JSONDecodeError, web.HTTPBadRequest):
            return web.json_response({"error": "请求体不是合法 JSON"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": "请求体必须是 JSON 对象"}, status=400)
        base_url = str(body.get("api_base_url", "") or "").strip()
        api_key = str(body.get("api_key", "") or "")
        try:
            timeout_s = min(max(int(body.get("timeout_s", 15)), 5), 60)
        except (TypeError, ValueError):
            timeout_s = 15
        if not base_url:
            return web.json_response({"error": "缺少 api_base_url"}, status=400)
        models = prompt_director._list_models(base_url, api_key, timeout_s=timeout_s)
        if not models:
            return web.json_response(
                {"error": f"无法从 {base_url} 获取模型列表（检查地址/密钥/网络）"}, status=400
            )
        return web.json_response(
            {"source": "OpenAI-compatible /models", "models": [{"id": m} for m in models]}
        )


_register_api_routes()
