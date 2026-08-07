# -*- coding: utf-8 -*-
"""MiniMaxH3PromptModuleLoader 测试（热加载模块合并，零网络）。"""
import importlib.util
import json
import os
import sys

LAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod():
    spec = importlib.util.spec_from_file_location("prompt_modules", os.path.join(LAB_ROOT, "prompt_modules.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prompt_modules"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_module_files_exist():
    """modules/ 应有核心协议 ×2 + 创作策略模块（协议不列为可选模块）。"""
    mod = _load_mod()
    modules = mod._load_modules()
    ids = {m["id"] for m in modules}
    # 创作策略模块（可选）
    for expected in ("anime_identity", "two_person_interaction", "motion_emphasis",
                     "seamless_loop", "ui_screen_text", "cinematic_realism"):
        assert expected in ids, f"缺少策略模块 {expected}"
    # 场景配方模块（官方场景 skill 浓缩）
    for expected in ("scene_3d_animation", "scene_brand_promo", "scene_coop_game_intro",
                     "scene_handdrawn_live", "scene_minimalist_product_ad", "scene_mv_subtitle",
                     "scene_paper_collage", "scene_papercraft_stopmotion"):
        assert expected in ids, f"缺少场景模块 {expected}"
    # 图生视频通用模板（用户常用方向：细微动态/推进/壁纸/天气/视差）
    for expected in ("subtle_still_motion", "slow_push_in", "live_wallpaper",
                     "weather_ambience", "parallax_motion"):
        assert expected in ids, f"缺少图生视频模板 {expected}"
    assert len(modules) == 21, f"可选模块应共 21 个，实际 {len(modules)}"
    # 协议模块不得作为可选模块列出（由导演节点自动加载）
    assert "protocol_base" not in ids and "protocol_ref" not in ids, "协议不得列为可选模块"
    assert "official_three_part" not in ids and "official_six_part" not in ids, "旧规范模块应被过滤"
    for m in modules:
        assert m.get("id") and m.get("title_zh") and m.get("instructions"), "模块字段不完整"
        assert isinstance(m.get("version", 1), int)


def test_load_protocol_base_and_ref():
    """协议自动加载：Ref2VA → 六段式；其余 → 三段式。"""
    mod = _load_mod()
    base = mod.load_protocol("I2VA")
    ref = mod.load_protocol("Ref2VA")
    assert "integrated_multimodal_description" in base
    assert "subject_definitions" in ref
    assert "subject_definitions" not in base
    assert mod.load_protocol("ref2va").startswith("【协议")


def test_load_protocol_unknown_task_falls_back_base():
    mod = _load_mod()
    assert "integrated_multimodal_description" in mod.load_protocol("UNKNOWN_TYPE")


def test_choices_include_modules():
    mod = _load_mod()
    choices = mod._module_choices()
    assert choices[0] == "（无）"
    assert any("动漫角色身份保持" in c for c in choices)
    assert not any("官方三段式" in c for c in choices), "协议模块不应出现在可选列表"
    assert not any("核心协议" in c for c in choices)


def test_load_none():
    mod = _load_mod()
    merged, preview, diag = mod.MiniMaxH3PromptModuleLoader().load()
    assert merged == ""
    assert "未选择模块" in preview


def test_load_single_module():
    mod = _load_mod()
    node = mod.MiniMaxH3PromptModuleLoader()
    merged, preview, diag = node.load(module_1="动漫角色身份保持")
    assert "[anime_identity | 动漫角色身份保持 | v1]" in merged
    assert "五官/发型/服装配色" in merged
    assert "（无）" not in merged


def test_load_custom_instructions():
    mod = _load_mod()
    node = mod.MiniMaxH3PromptModuleLoader()
    merged, preview, diag = node.load(custom_instructions="固定镜头：静止机位。")
    assert "[workflow_custom]" in merged
    assert "固定镜头：静止机位。" in merged
    assert "workflow_custom" in preview


def test_load_scope_all_no_mismatch_note():
    """v0.1 策略模块 scope=全部：任何模式选择都不产生不匹配提示。"""
    mod = _load_mod()
    node = mod.MiniMaxH3PromptModuleLoader()
    merged, _, _ = node.load(scope="Ref2VA", module_1="无缝循环")
    assert "不完全匹配" not in merged


def test_is_changed_nan():
    mod = _load_mod()
    import math
    assert math.isnan(mod.MiniMaxH3PromptModuleLoader.IS_CHANGED()), "必须热加载（每次排队重读）"


def test_merged_within_limit():
    mod = _load_mod()
    node = mod.MiniMaxH3PromptModuleLoader()
    merged, _, diag = node.load(module_1="官方三段式规范", module_2="社区 8 条写作要诀",
                                module_3="对白与屏幕文字保留规则")
    assert len(merged) <= mod._PROMPT_MODULE_MAX_CHARS + 20
    assert "合并字符" in diag
