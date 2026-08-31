"""Local credential storage for cloud-only MiniMax H3 nodes.

Security/product boundary:
- the store is created lazily on the first successful save;
- secrets never belong to node inputs, workflow JSON, outputs, or GET responses;
- the first version intentionally uses a documented local plaintext JSON file;
- environment variables override the local file;
- callers must never log returned secret values.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import os
import re
import stat
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


_SCHEMA_VERSION = 1
_MAX_STORE_BYTES = 256 * 1024
_MAX_KEY_CHARS = 4096
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_LOCK = threading.RLock()

_PROVIDERS = {
    "deepseek": {
        "default_credential_id": "deepseek_default",
        "environment_variable": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
    },
    "gemini": {
        "default_credential_id": "gemini_default",
        "environment_variable": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
    },
    # 豆包(火山方舟)：OpenAI 兼容 /v3（不用 /api/coding/v3，那只走 Coding Plan 额度）
    "volcengine": {
        "default_credential_id": "volcengine_default",
        "environment_variable": "ARK_API_KEY",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    },
    # Claude：Anthropic 原生 Messages，x-api-key + anthropic-version
    "anthropic": {
        "default_credential_id": "anthropic_default",
        "environment_variable": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1",
    },
    # GLM(智谱)：OpenAI 兼容 /api/paas/v4
    "zhipu": {
        "default_credential_id": "zhipu_default",
        "environment_variable": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
    },
    # MiniMax 官方 LLM：OpenAI 兼容 /v1
    "minimax": {
        "default_credential_id": "minimax_default",
        "environment_variable": "MINIMAX_API_KEY",
        "base_url": "https://api.minimaxi.com/v1",
    },
    # 自定义 OpenAI 兼容：base_url / model / key 全由用户在节点提供
    "custom": {
        "default_credential_id": "custom_default",
        "environment_variable": "OPENAI_API_KEY",
        "base_url": "",
    },
}

_GEMINI_DEFAULT_PROBE_MODEL = "gemini-3.1-flash-lite"
# Deterministic 32x32 PNG: left half red, right half blue.  The probe asks the
# model to identify both halves, so a successful response validates image input
# as well as authentication and structured output without uploading user media.
_GEMINI_PROBE_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAM0lEQVR4nO3NsQ0AAAiEwNf9d9YRiIUd"
    "1xNqclO5FZ1n7YA4QA6QA+QAOUAOkAPkADkIWZF6Aj8vIm7LAAAAAElFTkSuQmCC"
)


class CloudCredentialError(RuntimeError):
    """Safe, user-facing credential-store failure (never contains a secret)."""


class CredentialStoreUnavailable(CloudCredentialError):
    """The ComfyUI user directory is unavailable (typically a standalone import)."""


def provider_definition(provider: str) -> dict[str, str]:
    provider_id = _normalize_id(provider, "provider")
    definition = _PROVIDERS.get(provider_id)
    if definition is None:
        raise CloudCredentialError(f"不支持的云端供应商：{provider_id}")
    return dict(definition)


def _normalize_id(value: str, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _ID_RE.fullmatch(normalized):
        raise CloudCredentialError(f"{label} 只能包含小写字母、数字、点、下划线和连字符")
    return normalized


def _default_user_dir() -> Path:
    try:
        import folder_paths  # type: ignore

        value = folder_paths.get_user_directory()
    except Exception as exc:  # noqa: BLE001 - ComfyUI is optional in unit tests
        raise CredentialStoreUnavailable("无法定位 ComfyUI user 目录；请确认插件运行在 ComfyUI 中") from exc
    if not value:
        raise CredentialStoreUnavailable("ComfyUI user 目录为空")
    return Path(value)


def credential_file_path(user_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the future store path without creating a directory or a file."""
    root = Path(user_dir) if user_dir is not None else _default_user_dir()
    return root / "default" / "MiniMaxH3-Lab" / "cloud_credentials.json"


def _empty_store() -> dict[str, Any]:
    return {"version": _SCHEMA_VERSION, "credentials": {}}


def _read_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_store()
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise CloudCredentialError("无法读取云端凭据文件状态") from exc
    if size > _MAX_STORE_BYTES:
        raise CloudCredentialError("云端凭据文件异常过大，已拒绝读取")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CloudCredentialError(
            f"云端凭据文件损坏或不可读：{path}；为避免覆盖，未自动修复"
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("credentials"), dict):
        raise CloudCredentialError("云端凭据文件结构无效；为避免覆盖，未自动修复")
    version = data.get("version")
    if version != _SCHEMA_VERSION:
        raise CloudCredentialError(f"不支持的云端凭据文件版本：{version!r}")
    return data


def _tighten_permissions(path: Path, is_directory: bool = False) -> None:
    """Best effort only; Windows ACLs still inherit from the user's directory."""
    try:
        os.chmod(path, stat.S_IRWXU if is_directory else stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def _write_store_atomic(path: Path, data: dict[str, Any]) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    _tighten_permissions(parent, is_directory=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", prefix=".cloud_credentials.",
            suffix=".tmp", dir=parent, delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path = Path(temp_name)
        _tighten_permissions(temp_path)
        os.replace(temp_path, path)
        _tighten_permissions(path)
    except OSError as exc:
        raise CloudCredentialError("保存云端凭据失败；原文件未被主动覆盖") from exc
    finally:
        if temp_name:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass


def _local_entry(provider: str, credential_id: str, *, user_dir=None) -> tuple[dict[str, Any] | None, Path]:
    provider_id = _normalize_id(provider, "provider")
    cred_id = _normalize_id(credential_id, "credential_id")
    path = credential_file_path(user_dir)
    with _LOCK:
        data = _read_store(path)
    entry = data["credentials"].get(cred_id)
    if entry is None:
        return None, path
    if not isinstance(entry, dict) or str(entry.get("provider") or "").lower() != provider_id:
        raise CloudCredentialError(f"凭据 {cred_id} 与供应商 {provider_id} 不匹配")
    return entry, path


def resolve_credential(provider: str, credential_id: str, *, user_dir=None) -> tuple[str, str]:
    """Resolve a secret. The value must stay in process memory and never be logged."""
    definition = provider_definition(provider)
    env_name = definition["environment_variable"]
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return env_value, f"environment:{env_name}"
    try:
        entry, _path = _local_entry(provider, credential_id, user_dir=user_dir)
    except CredentialStoreUnavailable:
        if user_dir is None:
            # A direct module import outside ComfyUI has no user directory. Missing
            # local storage is equivalent to an unconfigured credential.
            return "", "unconfigured"
        raise
    if not entry:
        return "", "unconfigured"
    value = str(entry.get("api_key") or "").strip()
    return (value, f"local_file:{credential_id}") if value else ("", "unconfigured")


def credential_status(provider: str, credential_id: str, *, user_dir=None) -> dict[str, Any]:
    """Return only non-secret metadata. This is safe to send to the frontend."""
    definition = provider_definition(provider)
    provider_id = _normalize_id(provider, "provider")
    cred_id = _normalize_id(credential_id, "credential_id")
    path = credential_file_path(user_dir)
    env_name = definition["environment_variable"]
    env_configured = bool(os.environ.get(env_name, "").strip())
    entry, _ = _local_entry(provider_id, cred_id, user_dir=user_dir)
    local_configured = bool(entry and str(entry.get("api_key") or "").strip())
    source = f"environment:{env_name}" if env_configured else (
        f"local_file:{cred_id}" if local_configured else "unconfigured"
    )
    return {
        "provider": provider_id,
        "credential_id": cred_id,
        "configured": bool(env_configured or local_configured),
        "source": source,
        "environment_variable": env_name,
        "environment_configured": env_configured,
        "local_file_configured": local_configured,
        "storage_path": str(path),
        "storage_format": "local_plaintext_json",
    }


def save_credential(provider: str, credential_id: str, api_key: str, *, user_dir=None) -> dict[str, Any]:
    """Save a credential and return non-secret status. Creates storage lazily."""
    provider_id = _normalize_id(provider, "provider")
    provider_definition(provider_id)
    cred_id = _normalize_id(credential_id, "credential_id")
    key = str(api_key or "").strip()
    if not key:
        raise CloudCredentialError("API Key 不能为空")
    if len(key) > _MAX_KEY_CHARS:
        raise CloudCredentialError("API Key 长度异常，已拒绝保存")
    path = credential_file_path(user_dir)
    with _LOCK:
        data = _read_store(path)
        data["credentials"][cred_id] = {
            "provider": provider_id,
            "api_key": key,
            "updated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        }
        _write_store_atomic(path, data)
    return credential_status(provider_id, cred_id, user_dir=user_dir)


def clear_credential(provider: str, credential_id: str, *, user_dir=None) -> dict[str, Any]:
    """Remove the local entry. An environment variable, if present, remains effective."""
    provider_id = _normalize_id(provider, "provider")
    provider_definition(provider_id)
    cred_id = _normalize_id(credential_id, "credential_id")
    path = credential_file_path(user_dir)
    with _LOCK:
        data = _read_store(path)
        data["credentials"].pop(cred_id, None)
        if data["credentials"]:
            _write_store_atomic(path, data)
        else:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                raise CloudCredentialError("清除本地云端凭据失败") from exc
    return credential_status(provider_id, cred_id, user_dir=user_dir)


def _safe_model_text(value: Any, fallback: str = "") -> str:
    text = str(value or fallback).strip()
    if not text or len(text) > 200 or any(ord(char) < 32 for char in text):
        return str(fallback or "").strip()
    return text


def _safe_model_int(value: Any) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def list_deepseek_models(api_key: str, timeout_s: int = 15) -> list[dict[str, Any]]:
    """Return account-visible DeepSeek model metadata without returning the Key."""
    key = str(api_key or "").strip()
    if not key:
        raise CloudCredentialError("尚未配置 DeepSeek API Key")
    timeout = min(max(int(timeout_s), 5), 60)
    request = urllib.request.Request(
        "https://api.deepseek.com/models",
        headers={"Accept": "application/json", "Authorization": f"Bearer {key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed official host
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise CloudCredentialError(f"DeepSeek 鉴权失败（HTTP {exc.code}）") from exc
        if exc.code == 402:
            raise CloudCredentialError("DeepSeek 账户余额或计费状态异常（HTTP 402）") from exc
        raise CloudCredentialError(f"DeepSeek 模型列表请求失败（HTTP {exc.code}）") from exc
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CloudCredentialError(f"DeepSeek 模型列表请求失败（{type(exc).__name__}）") from exc

    raw_models = payload.get("data", []) if isinstance(payload, dict) else []
    result = []
    seen = set()
    for item in raw_models if isinstance(raw_models, list) else []:
        if not isinstance(item, dict):
            continue
        model_id = _safe_model_text(item.get("id"))
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        result.append({
            "id": model_id,
            "display_name": model_id,
            # Visible does not mean validated for multi-image Prompt IR.  The UI
            # may highlight likely vision models but must keep that distinction.
            "recommended": "vision" in model_id.casefold(),
            "supported_generation_methods": [],
        })
    return result


def _gemini_director_recommended(model_id: str, methods: list[str]) -> bool:
    folded = model_id.casefold()
    excluded = ("embedding", "imagen", "veo", "-image", "-tts", "-live", "omni")
    can_generate = not methods or "generatecontent" in {item.casefold() for item in methods}
    return folded.startswith("gemini-") and can_generate and not any(
        marker in folded for marker in excluded
    )


def list_gemini_models(api_key: str, timeout_s: int = 30) -> list[dict[str, Any]]:
    """Return Gemini ListModels entries safe for a searchable frontend picker."""
    key = str(api_key or "").strip()
    if not key:
        raise CloudCredentialError("尚未配置 Gemini API Key")
    timeout = min(max(int(timeout_s), 10), 60)
    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
        headers={"Accept": "application/json", "x-goog-api-key": key},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed official host
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise CloudCredentialError(f"Gemini 鉴权失败（HTTP {exc.code}）") from exc
        if exc.code == 429:
            raise CloudCredentialError("Gemini 模型列表触发配额限制（HTTP 429）") from exc
        raise CloudCredentialError(f"Gemini 模型列表请求失败（HTTP {exc.code}）") from exc
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CloudCredentialError(f"Gemini 模型列表请求失败（{type(exc).__name__}）") from exc

    raw_models = payload.get("models", []) if isinstance(payload, dict) else []
    result = []
    seen = set()
    for item in raw_models if isinstance(raw_models, list) else []:
        if not isinstance(item, dict):
            continue
        model_id = _safe_model_text(item.get("name")).removeprefix("models/")
        methods = [
            value for value in (
                _safe_model_text(method) for method in item.get("supportedGenerationMethods", [])
            ) if value
        ] if isinstance(item.get("supportedGenerationMethods", []), list) else []
        if not model_id or model_id in seen:
            continue
        # Director calls generateContent. Entries that explicitly do not expose
        # that method are irrelevant and would only create predictable 404/400s.
        if methods and "generatecontent" not in {value.casefold() for value in methods}:
            continue
        seen.add(model_id)
        result.append({
            "id": model_id,
            "display_name": _safe_model_text(item.get("displayName"), model_id),
            "recommended": _gemini_director_recommended(model_id, methods),
            "supported_generation_methods": methods,
            "input_token_limit": _safe_model_int(item.get("inputTokenLimit")),
            "output_token_limit": _safe_model_int(item.get("outputTokenLimit")),
        })
    return result


# 各 OpenAI 兼容 provider 的默认/候选模型（静态列表，供无可发现接口时用）。
# - claude 走原生 Messages，模型由 Anthropic 控制台决定，不在此列。
_STATIC_OPENAI_MODELS: dict[str, list[str]] = {
    "volcengine": ["doubao-seed-evolving", "doubao-seed-2-1-pro-260628", "doubao-seed-2-1-turbo-260628"],
    "anthropic": ["claude-sonnet-4-6"],
    "zhipu": ["glm-5.3-flash", "glm-4.6v", "glm-4.7"],
    "minimax": ["MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M2.5"],
}

_DEFAULT_MODEL_BY_PROVIDER: dict[str, str] = {
    "deepseek": "deepseek-v4-flash-vision-exp",
    "gemini": _GEMINI_DEFAULT_PROBE_MODEL,
    "volcengine": "doubao-seed-evolving",
    "anthropic": "claude-sonnet-4-6",
    "zhipu": "glm-5.3-flash",
    "minimax": "MiniMax-M2.7",
}


def _reject_link_local_target(base_url: str) -> None:
    """拒绝把 key 发往链路本地/多播目标（169.254.0.0/16 云元数据等）；localhost 放行。

    与 prompt_director 的 _reject_link_local_target 一致（SSRF 加固）：custom 预设等
    用户可填 base_url，防 key 被发到内网/云元数据地址。localhost/127.0.0.1 放行。
    """
    from urllib.parse import urlparse

    try:
        import ipaddress
        import socket
    except ImportError:
        return
    host = (urlparse(base_url).hostname or "").strip()
    if not host or host.lower() in ("localhost", "localhost.localdomain"):
        return
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise CloudCredentialError(f"无法解析 API 主机 {host!r}: {exc}") from exc
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if isinstance(ip, ipaddress.IPv6Address):
            mapped = ip.ipv4_mapped
            if mapped is not None:
                ip = mapped
        if ip.is_link_local or ip.is_multicast or ip.is_unspecified:
            raise CloudCredentialError(f"API 目标 {host} 为链路本地/多播地址，已拒绝（云元数据防护）")


def _chat_completions_url(base_url: str) -> str:
    """拼 OpenAI 兼容 /chat/completions；兼容带/不带 /vN 版本段、已含 /chat/completions。

    - 已含 /chat/completions：原样。
    - 以 /vN 结尾（如 /v1、/v3、/v4）视为已是 API 前缀（方舟 /api/v3、智谱 /paas/v4、
      MiniMax /v1 均如此）→ 直接加 /chat/completions。
    - 否则（裸 host，如 api.deepseek.com）→ 加 /v1/chat/completions。
    """
    root = str(base_url or "").strip().rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    if re.search(r"/v[0-9]+$", root):
        return f"{root}/chat/completions"
    return f"{root}/v1/chat/completions"


def _openai_compat_models_url(base_url: str) -> str:
    """Return the conventional OpenAI-compatible model discovery endpoint.

    OpenAI-compatible servers are not perfectly uniform, but ``GET /models``
    beside their configured API prefix covers LM Studio, OpenRouter and most
    proxy/gateway deployments.  Discovery failure remains non-fatal to the
    saved connection: the user can still type a model ID manually.
    """
    root = str(base_url or "").strip().rstrip("/")
    if not root:
        raise CloudCredentialError("OpenAI 兼容连接需要先填写 Base URL")
    if root.endswith("/models"):
        return root
    if root.endswith("/chat/completions"):
        return f"{root[:-len('/chat/completions')]}/models"
    if re.search(r"/v[0-9]+$", root):
        return f"{root}/models"
    return f"{root}/v1/models"


def _is_loopback_url(base_url: str) -> bool:
    try:
        host = (urllib.parse.urlsplit(str(base_url or "")).hostname or "").casefold()
    except ValueError:
        return False
    return host in {"localhost", "127.0.0.1", "::1"}


def list_openai_compat_models(
    api_key: str,
    base_url: str,
    timeout_s: int = 30,
) -> list[dict[str, Any]]:
    """Best-effort ``GET /models`` for a user supplied OpenAI-compatible API."""
    timeout = min(max(int(timeout_s), 5), 60)
    _reject_link_local_target(base_url)
    headers = {"Accept": "application/json"}
    key = str(api_key or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        _openai_compat_models_url(base_url), headers=headers, method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - user-chosen endpoint
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise CloudCredentialError(f"OpenAI 兼容模型列表鉴权失败（HTTP {exc.code}）") from exc
        if exc.code == 404:
            raise CloudCredentialError(
                "该服务没有提供标准 GET /models；可在节点里手动填写模型 ID"
            ) from exc
        raise CloudCredentialError(f"OpenAI 兼容模型列表请求失败（HTTP {exc.code}）") from exc
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CloudCredentialError(
            f"OpenAI 兼容模型列表请求失败（{type(exc).__name__}）"
        ) from exc

    raw_models = payload.get("data", payload.get("models", [])) if isinstance(payload, dict) else []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_models if isinstance(raw_models, list) else []:
        if isinstance(item, str):
            model_id = _safe_model_text(item)
            display_name = model_id
        elif isinstance(item, dict):
            model_id = _safe_model_text(item.get("id") or item.get("name"))
            display_name = _safe_model_text(
                item.get("display_name") or item.get("displayName"), model_id,
            )
        else:
            continue
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        rows.append({
            "id": model_id,
            "display_name": display_name,
            "recommended": False,
            "supported_generation_methods": [],
        })
    rows.sort(key=lambda item: item["id"].casefold())
    return rows


def _static_model_rows(provider: str) -> list[dict[str, Any]]:
    rows = []
    for model_id in _STATIC_OPENAI_MODELS.get(provider, []):
        rows.append({
            "id": model_id,
            "display_name": model_id,
            "recommended": model_id == _DEFAULT_MODEL_BY_PROVIDER.get(provider),
            "supported_generation_methods": [],
        })
    return rows


def list_cloud_models(
    provider: str,
    api_key: str,
    timeout_s: int = 30,
    *,
    base_url: str = "",
) -> list[dict[str, Any]]:
    provider_id = _normalize_id(provider, "provider")
    if provider_id == "deepseek":
        return list_deepseek_models(api_key, timeout_s)
    if provider_id == "gemini":
        return list_gemini_models(api_key, timeout_s)
    # GLM(智谱) 有标准 GET /models；豆包/Claude/MiniMax 无标准发现接口 → 静态列表。
    if provider_id == "zhipu":
        try:
            return list_zhipu_models(api_key, timeout_s)
        except CloudCredentialError:
            # 列表鉴权失败也可先用静态候选（真实探针会重新校验鉴权）。
            return _static_model_rows(provider_id)
    if provider_id in _STATIC_OPENAI_MODELS:
        return _static_model_rows(provider_id)
    if provider_id == "anthropic":
        # Claude 原生 Messages 无对应 GET /models；返回静态候选，真实探针再校验。
        return _static_model_rows("anthropic")
    if provider_id == "custom":
        return list_openai_compat_models(api_key, base_url, timeout_s)
    raise CloudCredentialError(f"尚未实现 {provider_id} 的模型列表适配")


def test_deepseek_connection(api_key: str, timeout_s: int = 15) -> dict[str, Any]:
    """Perform a non-generation authentication probe against DeepSeek /models."""
    models = list_deepseek_models(api_key, timeout_s)
    return {"ok": True, "provider": "deepseek", "model_count": len(models)}


def test_gemini_connection(api_key: str, model_id: str = "",
                           timeout_s: int = 30) -> dict[str, Any]:
    """Validate Gemini auth plus one tiny real image/strict-JSON request.

    Listing models alone is insufficient: a model can be visible to ListModels
    yet fail on ``generateContent`` for the current project or API version.  The
    returned dictionary is deliberately non-secret and excludes provider bodies.
    """
    key = str(api_key or "").strip()
    if not key:
        raise CloudCredentialError("尚未配置 Gemini API Key")
    model = str(model_id or _GEMINI_DEFAULT_PROBE_MODEL).strip()
    if not model or len(model) > 160 or not re.fullmatch(r"[A-Za-z0-9._@-]+", model):
        raise CloudCredentialError("Gemini 模型 ID 格式无效")
    timeout = min(max(int(timeout_s), 10), 60)
    models = list_gemini_models(key, timeout)
    model_count = len(models)
    visible_ids = {str(item.get("id") or "") for item in models}
    if visible_ids and model not in visible_ids:
        raise CloudCredentialError(
            f"Gemini 模型 {model} 未出现在当前项目的 ListModels 中；未继续生成探针"
        )

    schema = {
        "type": "object",
        "properties": {
            "image_received": {"type": "boolean"},
            "left_half": {"type": "string", "enum": ["red"]},
            "right_half": {"type": "string", "enum": ["blue"]},
        },
        "required": ["image_received", "left_half", "right_half"],
        "additionalProperties": False,
    }
    body = {
        "systemInstruction": {
            "parts": [{
                "text": (
                    "Inspect the attached synthetic image. Return only the requested JSON. "
                    "Do not guess from filenames or metadata."
                ),
            }],
        },
        "contents": [{
            "role": "user",
            "parts": [
                {"text": "Identify the color on the left half and the right half of this image."},
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": _GEMINI_PROBE_IMAGE_B64,
                    },
                },
            ],
        }],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 256,
            "thinkingConfig": {"thinkingLevel": "minimal"},
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
        },
    }
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + urllib.parse.quote(model, safe="")
        + ":generateContent"
    )
    generation_request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-goog-api-key": key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(generation_request, timeout=timeout) as response:  # noqa: S310 - fixed official host
            generation_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise CloudCredentialError(f"Gemini 鉴权失败（HTTP {exc.code}）") from exc
        if exc.code == 429:
            raise CloudCredentialError("Gemini 视觉探针触发配额限制（HTTP 429）") from exc
        if exc.code == 404:
            raise CloudCredentialError(f"Gemini 模型 {model} 当前不可调用（HTTP 404）") from exc
        raise CloudCredentialError(f"Gemini 视觉结构化探针失败（HTTP {exc.code}）") from exc
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CloudCredentialError(f"Gemini 视觉结构化探针失败（{type(exc).__name__}）") from exc

    try:
        candidate = generation_payload["candidates"][0]
        parts = candidate["content"]["parts"]
        text = "".join(
            str(part.get("text") or "") for part in parts if isinstance(part, dict)
        )
        parsed = json.loads(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise CloudCredentialError("Gemini 视觉探针未返回可解析的严格 JSON") from exc
    if not (
        parsed.get("image_received") is True
        and str(parsed.get("left_half") or "").casefold() == "red"
        and str(parsed.get("right_half") or "").casefold() == "blue"
    ):
        raise CloudCredentialError("Gemini 已响应，但没有通过红/蓝图像内容校验")
    usage = generation_payload.get("usageMetadata", {})
    usage = usage if isinstance(usage, dict) else {}
    return {
        "ok": True,
        "provider": "gemini",
        "model_count": model_count,
        "probe_model": model,
        "vision_probe": "strict_json_passed",
        "prompt_tokens": usage.get("promptTokenCount"),
        "completion_tokens": usage.get("candidatesTokenCount"),
    }


# ============================================================================
# 新增 4 家 + custom：OpenAI 兼容/原生 Messages 的连接探针。
# GLM(智谱) 是唯一用户确认持有真实 key 的 provider → 做真实视觉 strict-JSON 探针；
# 豆包/MiniMax/Claude/custom 在无 key 时只会因缺 key / 网络失败抛错，绝不冒充通过。
# ============================================================================

# 红/蓝 32x32 PNG（与 Gemini 探针同一张），用于验证视觉输入 + 鉴权 + 严格 JSON。
_OPENAI_PROBE_IMAGE_B64 = _GEMINI_PROBE_IMAGE_B64


def _openai_compat_vision_probe(
    provider: str,
    api_key: str,
    base_url: str,
    model_id: str,
    timeout_s: int = 30,
    *,
    reasoning_effort: str | None = None,
    thinking_config: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    require_json: bool = True,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """OpenAI 兼容 /chat/completions 的图片 + 严格 JSON 探针。

    - 校验鉴权、视觉输入、结构化输出三件事。
    - 返回 dict 不含任何 secret / provider body。
    - 该探针只验证「能否用这个 key + 模型 + 视觉 JSON」，不做内容语义断言。
    """
    key = str(api_key or "").strip()
    allow_unauthenticated_loopback = provider == "custom" and _is_loopback_url(base_url)
    if not key and not allow_unauthenticated_loopback:
        raise CloudCredentialError(f"尚未配置 {provider} API Key")
    model = str(model_id or "").strip()
    if not model:
        raise CloudCredentialError(f"未指定 {provider} 探针模型 ID")
    timeout = min(max(int(timeout_s), 10), 60)
    if not base_url:
        raise CloudCredentialError(f"{provider} 需要提供 base_url（自定义 OpenAI 兼容服务）")

    user_content = [
        {"type": "text", "text": (
            "Identify the color of the left half and the right half of this image. "
            "Return ONLY this exact JSON object: "
            '{"image_received": true, "left_half": "red", "right_half": "blue"}. '
            "Do not add anything else."
        )},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_OPENAI_PROBE_IMAGE_B64}"},
        },
    ]
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": temperature,
        "max_tokens": 256,
    }
    if thinking_config:
        # GLM 等：官方用顶层 thinking:{type:enabled}，不用 reasoning_effort。
        payload["thinking"] = thinking_config
    elif reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if require_json:
        # 用跨服务最稳的 json_object；schema 仅作本地提示，不发给服务端
        # （各家对 json_schema / 非标准字段支持不一，避免 400）。
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if extra_headers:
        headers.update(extra_headers)
    # SSRF 加固：custom/generic 探针会带真实 key 发往用户填的 base_url，先拒链路本地。
    _reject_link_local_target(base_url)
    url = _chat_completions_url(base_url)
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed official host
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise CloudCredentialError(f"{provider} 鉴权失败（HTTP {exc.code}）") from exc
        if exc.code == 429:
            raise CloudCredentialError(f"{provider} 探针触发配额限制（HTTP 429）") from exc
        if exc.code == 404:
            raise CloudCredentialError(f"{provider} 模型 {model} 当前不可调用（HTTP 404）") from exc
        raise CloudCredentialError(f"{provider} 视觉结构化探针失败（HTTP {exc.code}）") from exc
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CloudCredentialError(f"{provider} 视觉结构化探针失败（{type(exc).__name__}）") from exc

    content = ""
    try:
        choice = data["choices"][0]
        message = choice.get("message", {})
        content = str(message.get("content") or "")
    except (KeyError, IndexError, TypeError):
        pass
    if not content.strip():
        # 非流式拿到空 content，通常说明模型把 token 用于 thinking 且未给正文。
        raise CloudCredentialError(f"{provider} 探针返回空 content（模型可能全用于思考）")
    if require_json:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise CloudCredentialError(f"{provider} 探针未返回可解析的严格 JSON") from exc
        if not (
            parsed.get("image_received") is True
            and str(parsed.get("left_half") or "").casefold() == "red"
            and str(parsed.get("right_half") or "").casefold() == "blue"
        ):
            raise CloudCredentialError(f"{provider} 已响应，但没有通过红/蓝图像内容校验")
    usage = data.get("usage", {})
    usage = usage if isinstance(usage, dict) else {}
    return {
        "ok": True,
        "provider": provider,
        "probe_model": model,
        "vision_probe": "strict_json_passed" if require_json else "text_passed",
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def list_zhipu_models(api_key: str, timeout_s: int = 30) -> list[dict[str, Any]]:
    """GLM(智谱) 标准 GET /models 模型列表（open.bigmodel.cn/api/paas/v4/models）。"""
    key = str(api_key or "").strip()
    if not key:
        raise CloudCredentialError("尚未配置 ZHIPU_API_KEY")
    timeout = min(max(int(timeout_s), 10), 60)
    request = urllib.request.Request(
        "https://open.bigmodel.cn/api/paas/v4/models",
        headers={"Accept": "application/json", "Authorization": f"Bearer {key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed official host
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise CloudCredentialError("智谱鉴权失败（HTTP %s）" % exc.code) from exc
        raise CloudCredentialError(f"智谱模型列表请求失败（HTTP {exc.code}）") from exc
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CloudCredentialError(f"智谱模型列表请求失败（{type(exc).__name__}）") from exc
    raw_models = payload.get("data", []) if isinstance(payload, dict) else []
    result = []
    seen = set()
    for item in raw_models if isinstance(raw_models, list) else []:
        if not isinstance(item, dict):
            continue
        model_id = _safe_model_text(item.get("id"))
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        result.append({
            "id": model_id,
            "display_name": _safe_model_text(item.get("name"), model_id),
            "recommended": model_id == _DEFAULT_MODEL_BY_PROVIDER.get("zhipu"),
            "supported_generation_methods": [],
        })
    return result


def test_zhipu_connection(api_key: str, model_id: str = "", timeout_s: int = 30) -> dict[str, Any]:
    """GLM(智谱) 真实视觉 strict-JSON 探针（用户确有真实 key）。

    glm-5.3-flash 思考无法关闭（thinking.type 仅 enabled），故用 reasoning_effort=max
    + response_format json_object，靠「空 content 判定」识别思考占满输出的情形。
    """
    model = str(model_id or _DEFAULT_MODEL_BY_PROVIDER["zhipu"]).strip()
    # GLM 官方视觉示例用顶层 thinking:{type:"enabled"}（唯一允许，不能关闭思考）。
    return _openai_compat_vision_probe(
        "zhipu", api_key, _PROVIDERS["zhipu"]["base_url"], model, timeout_s,
        thinking_config={"type": "enabled"},
    )


def test_volcengine_connection(api_key: str, model_id: str = "", timeout_s: int = 30) -> dict[str, Any]:
    """豆包(火山方舟) 视觉探针。方舟无标准 /models，探针直接校验 + 视觉 JSON。"""
    model = str(model_id or _DEFAULT_MODEL_BY_PROVIDER["volcengine"]).strip()
    return _openai_compat_vision_probe(
        "volcengine", api_key, _PROVIDERS["volcengine"]["base_url"], model, timeout_s,
        reasoning_effort="medium",
    )


def test_minimax_connection(api_key: str, model_id: str = "", timeout_s: int = 30) -> dict[str, Any]:
    """MiniMax 文本 Chat 当前不作为 CloudDirector 视觉连接提供。

    平台具备图像/视频产品不等于文本 Chat wire 支持 OpenAI image_url。没有
    官方或真实探针证据前必须 fail-closed，不能用一个不存在的 M3 名称试撞。
    """
    del api_key, model_id, timeout_s
    raise CloudCredentialError(
        "MiniMax 官方文本 Chat 尚未验证可接收本节点的多图/视频帧输入；"
        "该 CloudDirector 预设已暂停。"
    )


def test_anthropic_connection(api_key: str, model_id: str = "", timeout_s: int = 30) -> dict[str, Any]:
    """Claude 原生 Messages 的真实小图 + JSON 视觉探针。

    不再把“Key 非空/模型名合法”冒充连接成功；只有官方 endpoint 真正返回并
    正确识别内置红蓝图时才标 strict_json_passed。
    """
    key = str(api_key or "").strip()
    if not key:
        raise CloudCredentialError("尚未配置 ANTHROPIC_API_KEY")
    model = str(model_id or _DEFAULT_MODEL_BY_PROVIDER["anthropic"]).strip()
    timeout = min(max(int(timeout_s), 10), 60)
    if not model or len(model) > 160 or not re.fullmatch(r"[A-Za-z0-9._@-]+", model):
        raise CloudCredentialError("Anthropic 模型 ID 格式无效")
    body = {
        "model": model,
        "max_tokens": 256,
        "system": (
            "Inspect the attached synthetic image. Return only the requested JSON object; "
            "do not infer from metadata or add commentary."
        ),
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Identify the left and right half colors. Return exactly: "
                        '{"image_received":true,"left_half":"red","right_half":"blue"}'
                    ),
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _OPENAI_PROBE_IMAGE_B64,
                    },
                },
            ],
        }],
    }
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed official host
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise CloudCredentialError(f"Anthropic 鉴权失败（HTTP {exc.code}）") from exc
        if exc.code == 429:
            raise CloudCredentialError("Anthropic 视觉探针触发配额限制（HTTP 429）") from exc
        if exc.code == 404:
            raise CloudCredentialError(f"Anthropic 模型 {model} 当前不可调用（HTTP 404）") from exc
        raise CloudCredentialError(f"Anthropic 视觉结构化探针失败（HTTP {exc.code}）") from exc
    except (urllib.error.URLError, TimeoutError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CloudCredentialError(f"Anthropic 视觉结构化探针失败（{type(exc).__name__}）") from exc
    try:
        text = "".join(
            str(part.get("text") or "") for part in data["content"]
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
        parsed = json.loads(text)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise CloudCredentialError("Anthropic 视觉探针未返回可解析的严格 JSON") from exc
    if not (
        parsed.get("image_received") is True
        and str(parsed.get("left_half") or "").casefold() == "red"
        and str(parsed.get("right_half") or "").casefold() == "blue"
    ):
        raise CloudCredentialError("Anthropic 已响应，但没有通过红/蓝图像内容校验")
    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    return {
        "ok": True,
        "provider": "anthropic",
        "probe_model": model,
        "vision_probe": "strict_json_passed",
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
    }


def test_custom_connection(api_key: str, base_url: str = "", model_id: str = "",
                           timeout_s: int = 30) -> dict[str, Any]:
    """自定义 OpenAI 兼容服务探针：base_url/model/key 全由用户提供。"""
    model = str(model_id or "").strip()
    return _openai_compat_vision_probe(
        "custom", api_key, base_url, model, timeout_s,
        reasoning_effort=None,
        require_json=True,
    )


# ============================================================================
# 用户自定义预设「我的预设」：持久化 base_url + model + 非秘密 credential_id。
# 每个预设使用独立凭据槽；key 仍只存在 credentials.json/环境变量，不写进
# custom_presets.json、节点或工作流。旧版共享 custom_default 凭据由运行层兼容读取。
# ============================================================================

_PRESETS_SCHEMA_VERSION = 1
_MAX_PRESETS = 100
_PRESETS_FILE_NAME = "custom_presets.json"


def _presets_file_path(user_dir: str | os.PathLike[str] | None = None) -> Path:
    root = Path(user_dir) if user_dir is not None else _default_user_dir()
    return root / "default" / "MiniMaxH3-Lab" / _PRESETS_FILE_NAME


def _read_presets(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": _PRESETS_SCHEMA_VERSION, "presets": {}}
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise CloudCredentialError("无法读取自定义预设文件状态") from exc
    if size > _MAX_STORE_BYTES:
        raise CloudCredentialError("自定义预设文件异常过大，已拒绝读取")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CloudCredentialError(f"自定义预设文件损坏或不可读：{path}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("presets"), dict):
        raise CloudCredentialError("自定义预设文件结构无效")
    if data.get("version") != _PRESETS_SCHEMA_VERSION:
        raise CloudCredentialError(f"不支持的自定义预设文件版本：{data.get('version')!r}")
    return data


def _normalize_preset_id(name: str) -> str:
    """把预设显示名转成稳定的 dict-key id。

    id 直接用显示名（允许中文），仅做安全校验：非空、长度 ≤64、不含路径分隔/
    控制字符，防止穿透存储文件。不用 _normalize_id（它只允许 ASCII 小写，会拒绝中文名）。
    """
    raw = str(name or "").strip()
    if not raw or len(raw) > 64:
        raise CloudCredentialError("自定义预设名称需为 1–64 字符")
    if any(ord(char) < 32 for char in raw) or any(char in raw for char in ("/", "\\", ":", "*", "?", '"', "<", ">", "|")):
        raise CloudCredentialError("自定义预设名称含不合法字符")
    return raw


def _custom_preset_credential_id(preset_id: str) -> str:
    """为每个自定义预设生成稳定、非秘密且符合 credential ID 语法的引用。"""
    digest = hashlib.sha256(str(preset_id).encode("utf-8")).hexdigest()[:24]
    return f"custom_{digest}"


def list_custom_presets(*, user_dir=None) -> list[dict[str, Any]]:
    path = _presets_file_path(user_dir)
    with _LOCK:
        data = _read_presets(path)
    presets = data.get("presets", {})
    result = []
    for pid, item in presets.items():
        if not isinstance(item, dict):
            continue
        result.append({
            "id": pid,
            "name": _safe_model_text(item.get("name"), pid),
            "base_url": _safe_model_text(item.get("base_url")),
            "model": _safe_model_text(item.get("model")),
            "credential_id": _safe_model_text(
                item.get("credential_id"), _custom_preset_credential_id(pid)
            ),
        })
    # 按 name 排序，稳定展示。
    result.sort(key=lambda p: (p["name"], p["id"]))
    return result


def save_custom_preset(name: str, base_url: str, model: str, *, user_dir=None) -> dict[str, Any]:
    name = str(name or "").strip()
    base_url = str(base_url or "").strip()
    model = str(model or "").strip()
    if not name:
        raise CloudCredentialError("自定义预设需要名称")
    if not base_url:
        raise CloudCredentialError("自定义预设需要 base_url")
    if not model:
        raise CloudCredentialError("自定义预设需要模型 ID")
    if not base_url.lower().startswith(("http://", "https://")):
        raise CloudCredentialError("自定义预设 base_url 仅支持 http/https")
    if len(base_url) > 200 or len(model) > 200 or len(name) > 64:
        raise CloudCredentialError("自定义预设字段长度异常，已拒绝保存")
    pid = _normalize_preset_id(name)
    credential_id = _custom_preset_credential_id(pid)
    path = _presets_file_path(user_dir)
    with _LOCK:
        data = _read_presets(path)
        if pid in data["presets"]:
            # 同名覆盖（幂等：重新保存即更新）。
            pass
        if len(data["presets"]) >= _MAX_PRESETS and pid not in data["presets"]:
            raise CloudCredentialError(f"自定义预设数量已达上限 {_MAX_PRESETS}")
        data["presets"][pid] = {
            "name": name,
            "base_url": base_url,
            "model": model,
            "credential_id": credential_id,
            "updated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        }
        _write_store_atomic(path, data)
    return {
        "ok": True, "id": pid, "name": name, "base_url": base_url,
        "model": model, "credential_id": credential_id,
    }


def delete_custom_preset(name: str, *, user_dir=None) -> dict[str, Any]:
    pid = _normalize_preset_id(str(name or "").strip())
    path = _presets_file_path(user_dir)
    with _LOCK:
        data = _read_presets(path)
        if pid not in data["presets"]:
            raise CloudCredentialError(f"自定义预设不存在：{name!r}")
        data["presets"].pop(pid, None)
        if data["presets"]:
            _write_store_atomic(path, data)
        else:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                raise CloudCredentialError("删除自定义预设失败") from exc
    return {"ok": True, "id": pid}


def get_custom_preset(name: str, *, user_dir=None) -> dict[str, Any]:
    pid = _normalize_preset_id(str(name or "").strip())
    path = _presets_file_path(user_dir)
    with _LOCK:
        data = _read_presets(path)
    item = data["presets"].get(pid)
    if item is None or not isinstance(item, dict):
        raise CloudCredentialError(f"自定义预设不存在：{name!r}")
    return {
        "id": pid,
        "name": _safe_model_text(item.get("name"), pid),
        "base_url": _safe_model_text(item.get("base_url")),
        "model": _safe_model_text(item.get("model")),
        "credential_id": _safe_model_text(
            item.get("credential_id"), _custom_preset_credential_id(pid)
        ),
    }
