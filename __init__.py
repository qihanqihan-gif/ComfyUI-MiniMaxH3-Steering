"""MiniMax-H3 research nodes and frontend helpers for ComfyUI."""

import json

from aiohttp import web  # noqa: F401  (ComfyUI 自带依赖)

from . import cloud_credentials
from .cloud_video import (
    NODE_CLASS_MAPPINGS as CLOUD_VIDEO_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as CLOUD_VIDEO_NODE_DISPLAY_NAME_MAPPINGS,
)
from . import prompt_director
from .easycache_safe import (
    NODE_CLASS_MAPPINGS as EASYCACHE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as EASYCACHE_NODE_DISPLAY_NAME_MAPPINGS,
)
from .h3_compiler import (
    NODE_CLASS_MAPPINGS as COMPILER_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as COMPILER_NODE_DISPLAY_NAME_MAPPINGS,
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
    **CLOUD_VIDEO_NODE_CLASS_MAPPINGS,
    **CACHE_NODE_CLASS_MAPPINGS,
    **EASYCACHE_NODE_CLASS_MAPPINGS,
    **COMPILER_NODE_CLASS_MAPPINGS,
    **MEDIA_NODE_CLASS_MAPPINGS,
    **PROFILER_NODE_CLASS_MAPPINGS,
    **PROMPT_NODE_CLASS_MAPPINGS,
    **PROMPTMOD_NODE_CLASS_MAPPINGS,
    **STEERING_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **CLOUD_VIDEO_NODE_DISPLAY_NAME_MAPPINGS,
    **CACHE_NODE_DISPLAY_NAME_MAPPINGS,
    **EASYCACHE_NODE_DISPLAY_NAME_MAPPINGS,
    **COMPILER_NODE_DISPLAY_NAME_MAPPINGS,
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

    @routes.post("/minimaxh3lab/api/module_files")
    async def _module_files(request):
        """前端「刷新模块列表」按钮：返回 prompt_modules/ 根 + 用户库的 JSON 清单。
        body: {"folder": "nsfw"}（可选）；返回 {"folder", "files", "all_folders"}。"""
        import prompt_modules as _pm
        try:
            body = await request.json()
        except (json.JSONDecodeError, web.HTTPBadRequest):
            body = {}
        if not isinstance(body, dict):
            return web.json_response({"error": "请求体必须是 JSON 对象"}, status=400)
        folder = str(body.get("folder") or "").strip()
        kind = str(body.get("kind") or "")
        # kind=mod 用 5 槽节点文件夹下拉（含正式库子文件夹）；否则用独立加载器下拉
        folder_choices = _pm._folder_choices() if kind == "mod" else _pm._folder_choices_bare()
        all_folders = {}
        for f in folder_choices:
            scan = "" if f == _pm._FOLDER_NONE else f
            choices, _ = _pm._file_title_choices(scan)
            all_folders[f] = choices[1:]
        files = all_folders.get(folder, _pm._folder_json_files(folder))
        try:
            mod_folder = None if (not folder or folder == _pm._FOLDER_NONE) else folder
            mods = [{"id": m["id"], "title_zh": m["title_zh"]} for m in _pm._load_modules(mod_folder)]
        except Exception:  # noqa: BLE001
            mods = []
        return web.json_response(
            {"folder": folder, "files": files, "all_folders": all_folders, "modules": mods}
        )

    @routes.get("/minimaxh3lab/cloud/credential/status")
    async def _cloud_credential_status(request):
        """Return non-secret credential state; never return the saved API Key."""
        provider = str(request.query.get("provider") or "deepseek")
        try:
            credential_id = str(
                request.query.get("credential_id")
                or cloud_credentials.provider_definition(provider)["default_credential_id"]
            )
            status = cloud_credentials.credential_status(provider, credential_id)
        except cloud_credentials.CloudCredentialError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "status": status})

    @routes.post("/minimaxh3lab/cloud/credential/save")
    async def _cloud_credential_save(request):
        try:
            body = await request.json()
        except (json.JSONDecodeError, web.HTTPBadRequest):
            return web.json_response({"ok": False, "error": "请求体不是合法 JSON"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"ok": False, "error": "请求体必须是 JSON 对象"}, status=400)
        try:
            provider = str(body.get("provider") or "deepseek")
            credential_id = str(
                body.get("credential_id")
                or cloud_credentials.provider_definition(provider)["default_credential_id"]
            )
            status = cloud_credentials.save_credential(
                provider,
                credential_id,
                str(body.get("api_key") or ""),
            )
        except cloud_credentials.CloudCredentialError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "status": status})

    @routes.post("/minimaxh3lab/cloud/credential/clear")
    async def _cloud_credential_clear(request):
        try:
            body = await request.json()
        except (json.JSONDecodeError, web.HTTPBadRequest):
            body = {}
        if not isinstance(body, dict):
            return web.json_response({"ok": False, "error": "请求体必须是 JSON 对象"}, status=400)
        try:
            provider = str(body.get("provider") or "deepseek")
            credential_id = str(
                body.get("credential_id")
                or cloud_credentials.provider_definition(provider)["default_credential_id"]
            )
            status = cloud_credentials.clear_credential(
                provider,
                credential_id,
            )
        except cloud_credentials.CloudCredentialError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "status": status})

    @routes.post("/minimaxh3lab/cloud/credential/test")
    async def _cloud_credential_test(request):
        try:
            body = await request.json()
        except (json.JSONDecodeError, web.HTTPBadRequest):
            body = {}
        if not isinstance(body, dict):
            return web.json_response({"ok": False, "error": "请求体必须是 JSON 对象"}, status=400)
        provider = str(body.get("provider") or "deepseek")
        try:
            credential_id = str(
                body.get("credential_id")
                or cloud_credentials.provider_definition(provider)["default_credential_id"]
            )
            key, source = cloud_credentials.resolve_credential(provider, credential_id)
            provider_id = provider.strip().lower()
            model_id = str(body.get("model_id") or "")
            timeout_s = body.get("timeout_s", 30)
            # 2026-08-28：新增常用云端 LLM provider 探针分派（豆包/Claude/GLM/MiniMax/custom）。
            if provider_id == "deepseek":
                result = cloud_credentials.test_deepseek_connection(key, timeout_s)
            elif provider_id == "gemini":
                result = cloud_credentials.test_gemini_connection(
                    key, str(model_id or "gemini-3.1-flash-lite"), timeout_s,
                )
            elif provider_id == "volcengine":
                result = cloud_credentials.test_volcengine_connection(
                    key, model_id, timeout_s,
                )
            elif provider_id == "anthropic":
                result = cloud_credentials.test_anthropic_connection(
                    key, model_id, timeout_s,
                )
            elif provider_id == "zhipu":
                result = cloud_credentials.test_zhipu_connection(
                    key, model_id, timeout_s,
                )
            elif provider_id == "minimax":
                result = cloud_credentials.test_minimax_connection(
                    key, model_id, timeout_s,
                )
            elif provider_id == "custom":
                result = cloud_credentials.test_custom_connection(
                    key,
                    str(body.get("base_url") or ""),
                    model_id,
                    timeout_s,
                )
            else:
                raise cloud_credentials.CloudCredentialError(
                    f"尚未实现 {provider_id or '?'} 的真实连接探针"
                )
        except cloud_credentials.CloudCredentialError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        # source is safe metadata; never include key or provider response bodies.
        return web.json_response({"ok": True, "source": source, **result})

    @routes.post("/minimaxh3lab/cloud/models")
    async def _cloud_models(request):
        """List provider-visible model IDs without exposing the resolved credential."""
        try:
            body = await request.json()
        except (json.JSONDecodeError, web.HTTPBadRequest):
            body = {}
        if not isinstance(body, dict):
            return web.json_response({"ok": False, "error": "请求体必须是 JSON 对象"}, status=400)
        provider = str(body.get("provider") or "deepseek")
        try:
            provider_id = provider.strip().lower()
            base_url = str(body.get("base_url") or "").strip()
            preset_name = str(body.get("preset_name") or "").strip()
            if provider_id == "custom" and preset_name:
                saved = cloud_credentials.get_custom_preset(preset_name)
                credential_id = str(saved["credential_id"])
                base_url = str(saved["base_url"])
            else:
                credential_id = str(
                    body.get("credential_id")
                    or cloud_credentials.provider_definition(provider)["default_credential_id"]
                )
            try:
                timeout_s = min(max(int(body.get("timeout_s", 30)), 5), 60)
            except (TypeError, ValueError):
                timeout_s = 30
            key, source = cloud_credentials.resolve_credential(provider, credential_id)
            models = cloud_credentials.list_cloud_models(
                provider, key, timeout_s, base_url=base_url,
            )
        except cloud_credentials.CloudCredentialError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({
            "ok": True,
            "provider": provider.strip().lower(),
            "source": source,
            "models": models,
        })

    @routes.get("/minimaxh3lab/cloud/presets")
    async def _cloud_presets_list(request):  # noqa: ARG001
        """列出「我的预设」（不含 key，仅 name/base_url/model）。"""
        try:
            presets = cloud_credentials.list_custom_presets()
        except cloud_credentials.CloudCredentialError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, "presets": presets})

    @routes.post("/minimaxh3lab/cloud/presets")
    async def _cloud_presets_save(request):
        try:
            body = await request.json()
        except (json.JSONDecodeError, web.HTTPBadRequest):
            return web.json_response({"ok": False, "error": "请求体不是合法 JSON"}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"ok": False, "error": "请求体必须是 JSON 对象"}, status=400)
        try:
            result = cloud_credentials.save_custom_preset(
                str(body.get("name") or ""),
                str(body.get("base_url") or ""),
                str(body.get("model") or ""),
            )
        except cloud_credentials.CloudCredentialError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response(result)

    @routes.post("/minimaxh3lab/cloud/presets/delete")
    async def _cloud_presets_delete(request):
        try:
            body = await request.json()
        except (json.JSONDecodeError, web.HTTPBadRequest):
            body = {}
        if not isinstance(body, dict):
            return web.json_response({"ok": False, "error": "请求体必须是 JSON 对象"}, status=400)
        try:
            result = cloud_credentials.delete_custom_preset(str(body.get("name") or ""))
        except cloud_credentials.CloudCredentialError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response(result)


_register_api_routes()
