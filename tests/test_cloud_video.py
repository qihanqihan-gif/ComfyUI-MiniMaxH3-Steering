# -*- coding: utf-8 -*-
"""Gemini 原生视频输入与传输纯单元测试（不发起真实网络请求）。"""
import base64
import io
import importlib.util
import os
import sys


LAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod():
    spec = importlib.util.spec_from_file_location(
        "cloud_video", os.path.join(LAB_ROOT, "cloud_video.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cloud_video"] = mod
    spec.loader.exec_module(mod)
    return mod


def _source(mod, path):
    size = path.stat().st_size
    return {
        "type": mod.CLOUD_VIDEO_TYPE,
        "path": str(path),
        "size_bytes": size,
        "mime_type": "video/mp4",
        "source_fingerprint": "placeholder",
    }


def test_inline_video_part_has_bytes_fps_and_no_local_path(monkeypatch, tmp_path):
    mod = _load_mod()
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"fake-mp4-bytes")
    monkeypatch.setattr(mod, "_input_directory", lambda: str(tmp_path))
    part, meta, cleanup = mod.prepare_gemini_video(
        _source(mod, video), "secret", route="inline", fps=2.0, timeout_s=30,
    )
    assert base64.b64decode(part["inlineData"]["data"]) == b"fake-mp4-bytes"
    assert part["inlineData"]["mimeType"] == "video/mp4"
    assert part["videoMetadata"] == {"fps": 2.0}
    assert meta["route"] == "inline"
    assert meta["fps_source"] == "user"
    assert "path" not in meta
    assert cleanup == ""


def test_file_api_route_returns_opaque_part_and_cleanup_name(monkeypatch, tmp_path):
    mod = _load_mod()
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x" * 32)
    monkeypatch.setattr(mod, "_input_directory", lambda: str(tmp_path))
    monkeypatch.setattr(mod, "_gemini_file_upload", lambda source, api_key, timeout_s: (
        {"name": "files/opaque", "uri": "https://files.example.invalid/opaque"},
        {"route": "file_api", "size_bytes": source["size_bytes"], "remote_cleanup": "pending"},
    ))
    part, meta, cleanup = mod.prepare_gemini_video(
        _source(mod, video), "secret", route="file_api", fps=0, timeout_s=30,
    )
    assert part == {"fileData": {
        "mimeType": "video/mp4", "fileUri": "https://files.example.invalid/opaque",
    }}
    assert meta["fps"] == 1.0
    assert meta["fps_source"] == "provider_default"
    assert cleanup == "files/opaque"


def test_auto_route_uses_combined_video_and_reference_image_budget(monkeypatch, tmp_path):
    mod = _load_mod()
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"x" * (6 * 1024 * 1024))
    monkeypatch.setattr(mod, "_input_directory", lambda: str(tmp_path))
    monkeypatch.setattr(mod, "_gemini_file_upload", lambda source, api_key, timeout_s: (
        {"name": "files/combined", "uri": "https://files.example.invalid/combined"},
        {"route": "file_api", "size_bytes": source["size_bytes"], "remote_cleanup": "pending"},
    ))
    _part, meta, cleanup = mod.prepare_gemini_video(
        _source(mod, video), "secret", route="auto", fps=0, timeout_s=30,
        inline_overhead_bytes=8 * 1024 * 1024,
    )
    assert meta["route"] == "file_api"
    assert meta["route_reason"] == "auto_file_combined_inline_budget"
    assert meta["inline_total_estimated_bytes"] > mod.GEMINI_INLINE_FILE_MAX_BYTES
    assert cleanup == "files/combined"


def test_auto_route_keeps_small_combined_payload_inline(monkeypatch, tmp_path):
    mod = _load_mod()
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"small-video")
    monkeypatch.setattr(mod, "_input_directory", lambda: str(tmp_path))
    _part, meta, cleanup = mod.prepare_gemini_video(
        _source(mod, video), "secret", route="auto", fps=0, timeout_s=30,
        inline_overhead_bytes=1024,
    )
    assert meta["route"] == "inline"
    assert meta["route_reason"] == "auto_inline_within_budget"
    assert cleanup == ""


def test_comfy_video_adapter_accepts_only_untrimmed_input_file(monkeypatch, tmp_path):
    mod = _load_mod()
    video_path = tmp_path / "native.mp4"
    video_path.write_bytes(b"native-video")
    monkeypatch.setattr(mod, "_input_directory", lambda: str(tmp_path))

    class FakeVideo:
        @staticmethod
        def get_active_trim_window():
            return (0.0, 0.0)

        @staticmethod
        def get_stream_source():
            return str(video_path)

    source = mod.cloud_video_source_from_comfy_video(FakeVideo())
    assert source["type"] == mod.CLOUD_VIDEO_TYPE
    assert source["path"] == str(video_path)
    assert source["mime_type"] == "video/mp4"


def test_comfy_video_adapter_rejects_trimmed_or_memory_sources(monkeypatch, tmp_path):
    mod = _load_mod()
    monkeypatch.setattr(mod, "_input_directory", lambda: str(tmp_path))

    class TrimmedVideo:
        @staticmethod
        def get_active_trim_window():
            return (1.0, 2.0)

        @staticmethod
        def get_stream_source():
            return str(tmp_path / "unused.mp4")

    class MemoryVideo:
        @staticmethod
        def get_active_trim_window():
            return (0.0, 0.0)

        @staticmethod
        def get_stream_source():
            return io.BytesIO(b"video")

    for candidate, expected in ((TrimmedVideo(), "已裁剪"), (MemoryVideo(), "文件型")):
        try:
            mod.cloud_video_source_from_comfy_video(candidate)
            assert False, "不安全的 Comfy VIDEO 必须在发送前被拒绝"
        except mod.CloudVideoError as exc:
            assert expected in str(exc)


def test_cloud_video_loader_report_does_not_expose_absolute_path(monkeypatch, tmp_path):
    mod = _load_mod()
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(mod, "_input_directory", lambda: str(tmp_path))
    monkeypatch.setattr(mod, "_folder_paths_module", lambda: type("FP", (), {
        "get_annotated_filepath": staticmethod(lambda _value: str(video)),
    }))
    source, report = mod.MiniMaxH3CloudVideoInput().load("sample.mp4")
    assert source["path"] == str(video)
    assert str(tmp_path) not in report
    assert "不会写入 STRING 报告、workflow 或 history" in report
