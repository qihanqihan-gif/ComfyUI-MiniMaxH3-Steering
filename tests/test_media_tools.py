# -*- coding: utf-8 -*-
"""Reference media helper regression tests (zero network, CPU tensors only)."""
import importlib.util
import os
import sys

import torch


LAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod():
    spec = importlib.util.spec_from_file_location("media_tools", os.path.join(LAB_ROOT, "media_tools.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["media_tools"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_official_reference_image_shape_matches_native_formula():
    mod = _load_mod()
    # User's portrait reference from the real workflow: match keeps aspect and
    # reduces to the 864x480 generation pixel area.
    assert mod._official_reference_image_shape(1979, 2977, 864, 480, "match") == (512, 800)
    assert mod._official_reference_image_shape(1979, 2977, 864, 480, "max") == (1984, 2976)


def test_official_reference_video_shape_small_source_is_not_upscaled():
    mod = _load_mod()
    assert mod._official_reference_video_shape(864, 480) == (864, 480)
    wide_w, wide_h = mod._official_reference_video_shape(1920, 1080)
    assert (wide_w, wide_h) == (1344, 768)


def test_cover_crop_without_upscale_centers_small_image():
    mod = _load_mod()
    image = torch.ones((1, 16, 16, 3), dtype=torch.float32)
    prepared, width, height, _ = mod._prepare_geometry(
        image, 64, 64, "cover_crop", False, 0.0, 1,
    )
    assert (width, height) == (64, 64)
    assert torch.all(prepared[:, 24:40, 24:40, :] == 1)
    assert torch.all(prepared[:, :16, :16, :] == 0)


def test_inspector_reports_reference_semantics_and_effective_size(monkeypatch):
    mod = _load_mod()
    monkeypatch.setattr(mod.MiniMaxH3ReferenceInspector, "_system_memory", staticmethod(lambda: (32.0, 64.0)))
    monkeypatch.setattr(mod.MiniMaxH3ReferenceInspector, "_gpu_memory", staticmethod(lambda: (16.0, 24.0)))
    image = torch.zeros((1, 297, 197, 3), dtype=torch.float32)
    result = mod.MiniMaxH3ReferenceInspector().inspect(
        generation_width=864,
        generation_height=480,
        generation_frames=124,
        strict_mode="report_only",
        ref_image_size="match",
        reference_images=image,
    )
    report = result["result"][3]
    assert "不会自动成为第0帧" in report
    assert "官方match后" in report


def test_inspector_new_widget_is_appended_after_strict_mode():
    mod = _load_mod()
    required = list(mod.MiniMaxH3ReferenceInspector.INPUT_TYPES()["required"])
    assert required[-2:] == ["strict_mode", "ref_image_size"]


def test_format_timecode_and_bounds_parsing():
    mod = _load_mod()
    assert mod._format_timecode(0.0) == "00:00.000"
    assert mod._format_timecode(5.0) == "00:05.000"
    assert mod._format_timecode(65.25) == "01:05.250"
    # 毫秒进位
    assert mod._format_timecode(59.9996) == "01:00.000"
    # 边界解析：秒 / 百分比 / 混用 / 空
    assert mod._parse_shot_boundaries("", 10.0) == []
    assert mod._parse_shot_boundaries("0,2.5,5", 10.0) == [0.0, 2.5, 5.0]
    assert mod._parse_shot_boundaries("0%,50%,100%", 10.0) == [0.0, 5.0, 10.0]
    assert mod._parse_shot_boundaries("0,50%", 10.0) == [0.0, 5.0]


def test_build_timeline_manifest_timecodes_and_segments():
    mod = _load_mod()
    manifest = mod._build_timeline_manifest(
        sequence_total=24, fps=24, duration_seconds=1.0,
        sampled_indices=[0, 6, 12, 18, 23],
        shot_boundaries=[0.0, 0.5, 1.0],
        selection_mode="uniform",
    )
    assert manifest["fps"] == 24
    assert manifest["selected_indices"] == [0, 6, 12, 18, 23]
    assert manifest["frames"][0]["timecode"] == "00:00.000"
    assert manifest["frames"][1]["timecode"] == "00:00.250"  # 6/24
    assert manifest["frames"][-1]["timecode"] == "00:00.958"  # 23/24≈0.958
    assert len(manifest["segments"]) == 2
    assert manifest["segments"][0]["end_sec"] == 0.5
    assert manifest["segments"][1]["start_sec"] == 0.5


def test_video_context_build_samples_and_emits_manifest():
    mod = _load_mod()
    frames = torch.zeros((120, 16, 16, 3), dtype=torch.float32)
    selected, manifest_str, report = mod.MiniMaxH3VideoContext().build(
        video_frame_sequence=frames,
        fps=24, duration_seconds=0.0,
        frame_sequence_limit=12, frame_selection_mode="uniform",
        shot_boundaries="0,2.5,5",
    )
    manifest = __import__("json").loads(manifest_str)
    assert selected.shape[0] == 12
    assert manifest["total_frames"] == 120
    assert manifest["selected_count"] == 12
    assert manifest["fps"] == 24
    # duration 由总帧数/fps 推导 = 5.0s
    assert manifest["duration_source"] == "inferred"
    assert manifest["duration_seconds"] == 5.0
    # duration=5.0，shot_boundaries="0,2.5,5"：0 与 5 是首尾被过滤，只剩内部切点 2.5 → 2 段
    assert len(manifest["segments"]) == 2
    assert manifest["segments"][0]["end_sec"] == 2.5
    assert manifest["segments"][1]["start_sec"] == 2.5
    assert manifest["sequence_label"] == "<Video 1>"
    # 第一帧必须含 index 0（uniform 保首尾）
    assert manifest["selected_indices"][0] == 0
    assert manifest["selected_indices"][-1] == 119
    assert "时间线 Context" in report
