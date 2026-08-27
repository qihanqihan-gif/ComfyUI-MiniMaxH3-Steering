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


def list_cloud_models(provider: str, api_key: str, timeout_s: int = 30) -> list[dict[str, Any]]:
    provider_id = _normalize_id(provider, "provider")
    if provider_id == "deepseek":
        return list_deepseek_models(api_key, timeout_s)
    if provider_id == "gemini":
        return list_gemini_models(api_key, timeout_s)
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
