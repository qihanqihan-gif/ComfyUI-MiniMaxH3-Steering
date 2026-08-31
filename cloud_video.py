# -*- coding: utf-8 -*-
"""Runtime-only cloud video sources and Gemini native-video transport.

The loader returns an opaque runtime object whose in-memory path is required
for streaming.  Video bytes, paths and remote file URIs are never copied into
STRING reports or workflow JSON.  Gemini File API resources are deleted after
the generation call.

SPDX-License-Identifier: GPL-3.0-only
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import mimetypes
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


CLOUD_VIDEO_TYPE = "MINIMAX_H3_CLOUD_VIDEO"
GEMINI_AUTO_INLINE_BYTES = 8 * 1024 * 1024
GEMINI_INLINE_FILE_MAX_BYTES = 15 * 1024 * 1024
GEMINI_FILE_API_MAX_BYTES = 2 * 1024 * 1024 * 1024
_STREAM_CHUNK_BYTES = 1024 * 1024

_VIDEO_MIME_BY_SUFFIX = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpe": "video/mpeg",
    ".mov": "video/mov",
    ".avi": "video/avi",
    ".flv": "video/x-flv",
    ".mpg": "video/mpg",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".3gp": "video/3gpp",
    ".3gpp": "video/3gpp",
}


class CloudVideoError(RuntimeError):
    """Safe, user-facing native-video transport failure."""


def _folder_paths_module():
    import folder_paths  # ComfyUI runtime dependency; lazy for pure unit tests.

    return folder_paths


def _input_directory() -> str:
    return os.path.realpath(_folder_paths_module().get_input_directory())


def _video_files() -> list[str]:
    root = Path(_input_directory())
    if not root.exists():
        return []
    found: list[str] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.casefold() in _VIDEO_MIME_BY_SUFFIX:
            found.append(path.relative_to(root).as_posix())
    return sorted(found, key=str.casefold)


def _resolve_input_video(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise CloudVideoError("未选择云端原生视频文件")
    folder_paths = _folder_paths_module()
    try:
        resolved = folder_paths.get_annotated_filepath(raw)
    except Exception:  # noqa: BLE001
        resolved = os.path.join(_input_directory(), raw)
    path = os.path.realpath(resolved)
    input_root = _input_directory()
    try:
        inside = os.path.commonpath([path, input_root]) == input_root
    except ValueError:
        inside = False
    if not inside:
        raise CloudVideoError("云端原生视频必须位于 ComfyUI input 目录内")
    if not os.path.isfile(path):
        raise CloudVideoError("所选云端原生视频不存在或不是文件")
    return path


def _mime_for_path(path: str) -> str:
    suffix = Path(path).suffix.casefold()
    mime = _VIDEO_MIME_BY_SUFFIX.get(suffix)
    if not mime:
        guessed, _ = mimetypes.guess_type(path)
        mime = guessed if str(guessed or "").startswith("video/") else ""
    if not mime:
        raise CloudVideoError(f"Gemini 不支持该视频扩展名：{suffix or '?'}")
    return mime


def _stat_fingerprint(path: str, size_bytes: int) -> str:
    stat = os.stat(path)
    raw = f"{size_bytes}:{stat.st_mtime_ns}:{Path(path).suffix.casefold()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _source_from_input_path(path_value: Any) -> dict:
    """Build the opaque runtime source for one verified Comfy input file."""
    try:
        path = os.path.realpath(os.fspath(path_value))
    except TypeError as exc:
        raise CloudVideoError("Comfy VIDEO 当前不是可直接上传的文件型视频") from exc
    input_root = _input_directory()
    try:
        inside = os.path.commonpath([path, input_root]) == input_root
    except ValueError:
        inside = False
    if not inside:
        raise CloudVideoError("云端原生视频必须位于 ComfyUI input 目录内")
    if not os.path.isfile(path):
        raise CloudVideoError("云端原生视频不存在或不是文件")
    size_bytes = int(os.path.getsize(path))
    if size_bytes < 1:
        raise CloudVideoError("云端原生视频是空文件")
    if size_bytes > GEMINI_FILE_API_MAX_BYTES:
        raise CloudVideoError("云端原生视频超过当前保守的 Gemini 2 GiB 单文件上限")
    return {
        "type": CLOUD_VIDEO_TYPE,
        "path": path,
        "size_bytes": size_bytes,
        "mime_type": _mime_for_path(path),
        "source_fingerprint": _stat_fingerprint(path, size_bytes),
    }


def cloud_video_source_from_comfy_video(video: Any) -> dict:
    """Adapt a current Comfy core ``VIDEO`` object without decoding frames.

    Only an untrimmed, file-backed source can be forwarded losslessly.  A
    BytesIO-backed or trimmed source must first be saved as a concrete video;
    otherwise forwarding the original file would silently ignore user edits.
    """
    stream_getter = getattr(video, "get_stream_source", None)
    if not callable(stream_getter):
        raise CloudVideoError("输入不是受支持的 Comfy 原生 VIDEO 对象")
    trim_getter = getattr(video, "get_active_trim_window", None)
    if callable(trim_getter):
        try:
            start_s, duration_s = trim_getter()
            start_s = float(start_s or 0.0)
            duration_s = float(duration_s or 0.0)
        except (TypeError, ValueError) as exc:
            raise CloudVideoError("无法读取 Comfy VIDEO 的裁剪范围") from exc
        if abs(start_s) > 1e-6 or abs(duration_s) > 1e-6:
            raise CloudVideoError(
                "当前零拷贝直连不接受已裁剪的 Comfy VIDEO；请先保存裁剪结果，"
                "或改用“云端原生视频输入”选择成品文件"
            )
    source = stream_getter()
    if not isinstance(source, (str, os.PathLike)):
        raise CloudVideoError(
            "当前零拷贝直连只接受文件型 Comfy VIDEO；内存视频请先保存为 MP4/WebM"
        )
    return _source_from_input_path(source)


def validate_cloud_video_source(source: Any) -> dict:
    """Validate an opaque source again at send time; never trust foreign nodes."""
    if not isinstance(source, dict) or source.get("type") != CLOUD_VIDEO_TYPE:
        raise CloudVideoError("云端原生视频输入不是本插件生成的运行时对象")
    clean = _source_from_input_path(source.get("path"))
    clean.pop("type", None)
    return clean


class MiniMaxH3CloudVideoInput:
    """Select/upload one input video without decoding it into browser previews."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        files = _video_files() or ["（请先上传视频到 ComfyUI input）"]
        return {
            "required": {
                "video": (
                    files,
                    {
                        "video_upload": True,
                        "tooltip": "保留原 MP4/音频供 Gemini 原生视频理解；不生成数百张前端预览",
                    },
                ),
            },
        }

    RETURN_TYPES = (CLOUD_VIDEO_TYPE, "STRING")
    RETURN_NAMES = ("cloud_video", "report")
    FUNCTION = "load"
    CATEGORY = "MiniMax H3 Lab/Media"
    DESCRIPTION = "Runtime-only native video source for the Cloud Director."

    def load(self, video: str) -> tuple[dict, str]:
        path = _resolve_input_video(video)
        size_bytes = int(os.path.getsize(path))
        mime_type = _mime_for_path(path)
        source = {
            "type": CLOUD_VIDEO_TYPE,
            "path": path,
            "size_bytes": size_bytes,
            "mime_type": mime_type,
            "source_fingerprint": _stat_fingerprint(path, size_bytes),
        }
        report = (
            "MiniMax H3 云端原生视频输入\n"
            f"MIME：{mime_type}；体积：{size_bytes / 1024 / 1024:.2f} MiB\n"
            "文件仅作为不透明运行时对象交给云端导演；视频字节、绝对路径和远端 URI "
            "不会写入 STRING 报告、workflow 或 history。"
        )
        return source, report

    @classmethod
    def IS_CHANGED(cls, video: str):
        try:
            path = _resolve_input_video(video)
            stat = os.stat(path)
            return f"{stat.st_mtime_ns}:{stat.st_size}"
        except Exception:  # noqa: BLE001
            return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, video: str):
        try:
            _resolve_input_video(video)
        except CloudVideoError as exc:
            return str(exc)
        return True


def _json_request(url: str, api_key: str, *, method: str = "GET",
                  body: dict | None = None, timeout_s: int = 60,
                  headers: dict | None = None) -> tuple[dict, Any]:
    payload = None
    merged = {"x-goog-api-key": api_key}
    if body is not None:
        payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        merged["Content-Type"] = "application/json"
    if headers:
        merged.update(headers)
    request = urllib.request.Request(url, data=payload, headers=merged, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
            return (json.loads(raw.decode("utf-8")) if raw else {}, response.headers)
    except Exception as exc:  # noqa: BLE001
        raise CloudVideoError(f"Gemini Files API {method} 请求失败（不回显供应商正文）") from exc


def _stream_upload(upload_url: str, path: str, size_bytes: int,
                   mime_type: str, timeout_s: int) -> tuple[dict, str, float]:
    parsed = urllib.parse.urlparse(upload_url)
    host = (parsed.hostname or "").casefold()
    if parsed.scheme != "https" or not (
        host == "googleapis.com" or host.endswith(".googleapis.com")
    ):
        raise CloudVideoError("Gemini 返回了非 Google HTTPS 的上传地址，已拒绝")
    request_path = parsed.path or "/"
    if parsed.query:
        request_path += "?" + parsed.query
    connection = http.client.HTTPSConnection(
        parsed.hostname, parsed.port or 443, timeout=timeout_s,
    )
    digest = hashlib.sha256()
    started = time.perf_counter()
    try:
        connection.putrequest("POST", request_path)
        connection.putheader("Content-Length", str(size_bytes))
        connection.putheader("Content-Type", mime_type)
        connection.putheader("X-Goog-Upload-Offset", "0")
        connection.putheader("X-Goog-Upload-Command", "upload, finalize")
        connection.endheaders()
        sent = 0
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                connection.send(chunk)
                sent += len(chunk)
        if sent != size_bytes:
            raise CloudVideoError("Gemini 视频上传前后文件大小发生变化，请重试")
        response = connection.getresponse()
        raw = response.read()
        if response.status < 200 or response.status >= 300:
            raise CloudVideoError(f"Gemini 视频上传失败（HTTP {response.status}，正文已隐藏）")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CloudVideoError("Gemini 视频上传返回体不是合法 JSON") from exc
        return payload, digest.hexdigest(), (time.perf_counter() - started) * 1000
    finally:
        connection.close()


def _gemini_file_upload(source: dict, api_key: str, timeout_s: int) -> tuple[dict, dict]:
    path = source["path"]
    size_bytes = int(source["size_bytes"])
    mime_type = source["mime_type"]
    display_name = "minimax-h3-" + source["source_fingerprint"][:12] + Path(path).suffix.casefold()
    started = time.perf_counter()
    _metadata, headers = _json_request(
        "https://generativelanguage.googleapis.com/upload/v1beta/files",
        api_key,
        method="POST",
        body={"file": {"display_name": display_name}},
        timeout_s=min(max(30, int(timeout_s)), 300),
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size_bytes),
            "X-Goog-Upload-Header-Content-Type": mime_type,
        },
    )
    upload_url = str(headers.get("X-Goog-Upload-URL") or headers.get("x-goog-upload-url") or "")
    if not upload_url:
        raise CloudVideoError("Gemini Files API 未返回 resumable 上传地址")
    uploaded, content_sha256, upload_ms = _stream_upload(
        upload_url, path, size_bytes, mime_type, timeout_s,
    )
    info = uploaded.get("file") if isinstance(uploaded.get("file"), dict) else uploaded
    name = str(info.get("name") or "")
    uri = str(info.get("uri") or "")
    if not name.startswith("files/") or not uri:
        raise CloudVideoError("Gemini Files API 上传结果缺少安全的 file name/URI")
    wait_started = time.perf_counter()
    try:
        deadline = time.monotonic() + max(30, int(timeout_s))
        state = str(info.get("state") or "PROCESSING").upper()
        while state == "PROCESSING":
            if time.monotonic() >= deadline:
                raise CloudVideoError("Gemini 视频已上传，但等待处理为 ACTIVE 超时")
            time.sleep(2.0)
            info, _ = _json_request(
                "https://generativelanguage.googleapis.com/v1beta/" + urllib.parse.quote(name, safe="/"),
                api_key,
                timeout_s=min(30, max(5, int(timeout_s))),
            )
            state = str(info.get("state") or "").upper()
        if state != "ACTIVE":
            raise CloudVideoError(f"Gemini 视频处理未进入 ACTIVE（state={state or '?'}）")
    except Exception:
        # 上传已经生成远端资源时，即使处理阶段失败也尽量立即清理；
        # 清理失败不覆盖原始、更加有诊断价值的异常。
        delete_gemini_file(name, api_key, timeout_s=30)
        raise
    return info, {
        "route": "file_api",
        "size_bytes": size_bytes,
        "mime_type": mime_type,
        "content_sha256": content_sha256,
        "upload_ms": round(upload_ms),
        "processing_wait_ms": round((time.perf_counter() - wait_started) * 1000),
        "prepare_total_ms": round((time.perf_counter() - started) * 1000),
        "remote_resource_hash": hashlib.sha256(name.encode("utf-8")).hexdigest(),
        "remote_cleanup": "pending",
    }


def delete_gemini_file(name: str, api_key: str, timeout_s: int = 30) -> bool:
    if not str(name or "").startswith("files/"):
        return False
    url = "https://generativelanguage.googleapis.com/v1beta/" + urllib.parse.quote(name, safe="/")
    try:
        _json_request(url, api_key, method="DELETE", timeout_s=min(30, max(5, int(timeout_s))))
        return True
    except CloudVideoError:
        return False


def prepare_gemini_video(source: Any, api_key: str, route: str = "auto",
                         fps: float = 0.0, timeout_s: int = 600,
                         inline_overhead_bytes: int = 0) -> tuple[dict, dict, str]:
    """Return ``(Gemini Part, safe metadata, remote file name for cleanup)``."""
    clean = validate_cloud_video_source(source)
    route_value = str(route or "auto").strip().casefold()
    if route_value not in {"auto", "inline", "file_api"}:
        route_value = "auto"
    overhead_bytes = max(0, int(inline_overhead_bytes or 0))
    encoded_video_estimate = 4 * ((int(clean["size_bytes"]) + 2) // 3)
    inline_total_estimate = encoded_video_estimate + overhead_bytes
    resolved_route = route_value
    route_reason = "forced_by_user"
    if resolved_route == "auto":
        if clean["size_bytes"] > GEMINI_AUTO_INLINE_BYTES:
            resolved_route = "file_api"
            route_reason = "auto_file_raw_video_over_8_mib"
        elif inline_total_estimate > GEMINI_INLINE_FILE_MAX_BYTES:
            resolved_route = "file_api"
            route_reason = "auto_file_combined_inline_budget"
        else:
            resolved_route = "inline"
            route_reason = "auto_inline_within_budget"
    fps_value = min(24.0, max(0.0, float(fps or 0.0)))
    metadata = {
        "fps": fps_value if fps_value > 0 else 1.0,
        "fps_source": "user" if fps_value > 0 else "provider_default",
        "route_reason": route_reason,
        "inline_overhead_bytes": overhead_bytes,
        "inline_total_estimated_bytes": inline_total_estimate,
    }
    if resolved_route == "inline":
        if clean["size_bytes"] > GEMINI_INLINE_FILE_MAX_BYTES:
            raise CloudVideoError(
                "强制 inline 的视频超过 15 MiB 安全上限；请改为 auto 或 file_api"
            )
        started = time.perf_counter()
        digest = hashlib.sha256()
        chunks = []
        with open(clean["path"], "rb") as handle:
            while True:
                chunk = handle.read(_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                chunks.append(chunk)
        encoded = base64.b64encode(b"".join(chunks)).decode("ascii")
        part = {"inlineData": {"mimeType": clean["mime_type"], "data": encoded}}
        if fps_value > 0:
            part["videoMetadata"] = {"fps": fps_value}
        return part, {
            "route": "inline",
            "size_bytes": clean["size_bytes"],
            "encoded_bytes": len(encoded),
            "mime_type": clean["mime_type"],
            "content_sha256": digest.hexdigest(),
            "prepare_total_ms": round((time.perf_counter() - started) * 1000),
            "upload_ms": 0,
            "processing_wait_ms": 0,
            "remote_cleanup": "not_applicable",
            **metadata,
        }, ""

    info, upload_meta = _gemini_file_upload(clean, api_key, timeout_s)
    part = {
        "fileData": {"mimeType": clean["mime_type"], "fileUri": str(info["uri"])},
    }
    if fps_value > 0:
        part["videoMetadata"] = {"fps": fps_value}
    return part, {**upload_meta, **metadata}, str(info["name"])


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3CloudVideoInput": MiniMaxH3CloudVideoInput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3CloudVideoInput": "MiniMax H3 Cloud Native Video Input",
}
