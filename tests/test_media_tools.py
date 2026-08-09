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
