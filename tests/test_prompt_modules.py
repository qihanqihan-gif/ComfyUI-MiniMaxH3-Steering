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
    # v0.2：原 21 个 + 5 个通用回归模块 + 人物替换实验模块
    assert len(modules) == 27, f"可选模块应共 27 个，实际 {len(modules)}"
    # 协议模块不得作为可选模块列出（由导演节点自动加载）
    assert "protocol_base" not in ids and "protocol_ref" not in ids, "协议不得列为可选模块"
    assert "official_three_part" not in ids and "official_six_part" not in ids, "旧规范模块应被过滤"
    for m in modules:
        assert m.get("id") and m.get("title_zh") and m.get("instructions"), "模块字段不完整"
        assert isinstance(m.get("version", 1), int)
        assert m.get("source"), f"{m['id']} 缺少来源"
        assert m.get("evidence_level") in mod._EVIDENCE_LABELS, f"{m['id']} 证据等级无效"
        assert isinstance(m.get("features"), list) and m["features"], f"{m['id']} 缺少 features"
        assert not m.get("_metadata_issues"), f"{m['id']} 元数据问题：{m['_metadata_issues']}"


def test_new_general_modules_exist():
    mod = _load_mod()
    ids = {m["id"] for m in mod._load_modules()}
    assert {"single_shot_continuity", "dialogue_performance", "product_identity_fidelity",
            "storyboard_reference_map", "natural_camera_follow"} <= ids


def test_load_protocol_base_and_ref():
    """协议自动加载：Ref2VA → 六段式；其余 → 三段式。"""
    mod = _load_mod()
    base = mod.load_protocol("I2VA")
    ref = mod.load_protocol("Ref2VA")
    assert "integrated_multimodal_description" in base
    assert "subject_definitions" in ref
    assert "subject_definitions" not in base
    assert mod.load_protocol("ref2va").startswith("【官方 Ref2VA 协议")
    assert "L2VA" in (json.load(open(os.path.join(LAB_ROOT, "modules", "protocol_base.json"),
                                     encoding="utf-8"))["scope"])


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
    merged, preview, diag, manifest = mod.MiniMaxH3PromptModuleLoader().load()
    assert merged == ""
    assert "未选择模块" in preview
    assert json.loads(manifest)["selected"] == []


def test_load_single_module():
    mod = _load_mod()
    node = mod.MiniMaxH3PromptModuleLoader()
    merged, preview, diag, manifest = node.load(module_1="动漫角色身份保持")
    assert "[anime_identity | 动漫角色身份保持 | v2 | 官方规则对齐]" in merged
    assert "身份锚点" in merged
    assert "（无）" not in merged
    assert json.loads(manifest)["selected"][0]["id"] == "anime_identity"


def test_load_custom_instructions():
    mod = _load_mod()
    node = mod.MiniMaxH3PromptModuleLoader()
    merged, preview, diag, manifest = node.load(custom_instructions="固定镜头：静止机位。")
    assert "[workflow_custom]" in merged
    assert "固定镜头：静止机位。" in merged
    assert "workflow_custom" in preview


def test_load_scope_all_no_mismatch_note():
    """v0.1 策略模块 scope=全部：任何模式选择都不产生不匹配提示。"""
    mod = _load_mod()
    node = mod.MiniMaxH3PromptModuleLoader()
    merged, _, diag, _ = node.load(scope="Ref2VA", module_1="无缝循环")
    assert "作用域不匹配" not in diag


def test_duplicate_selection_is_deduplicated():
    mod = _load_mod()
    merged, _, diag, manifest = mod.MiniMaxH3PromptModuleLoader().load(
        module_1="单镜连续性", module_2="单镜连续性")
    assert merged.count("[single_shot_continuity |") == 1
    assert "重复选择已忽略" in diag
    assert len(json.loads(manifest)["selected"]) == 1


def test_conflict_is_reported():
    mod = _load_mod()
    _, _, diag, manifest = mod.MiniMaxH3PromptModuleLoader().load(
        module_1="单镜连续性", module_2="多参考图分镜映射")
    assert "模块冲突" in diag
    assert json.loads(manifest)["issues"]


def test_is_changed_nan():
    mod = _load_mod()
    import math
    assert math.isnan(mod.MiniMaxH3PromptModuleLoader.IS_CHANGED()), "必须热加载（每次排队重读）"
    assert math.isnan(mod.MiniMaxH3ModuleFolderLoader.IS_CHANGED())


def test_module_file_choices_are_recursive_and_root_qualified(tmp_path, monkeypatch):
    mod = _load_mod()
    built_in = tmp_path / "modules"
    extra = tmp_path / "extra"
    (built_in / "角色").mkdir(parents=True)
    (extra / "镜头" / "推进").mkdir(parents=True)
    payload = {
        "id": "valid", "title_zh": "有效模块", "instructions": "保留身份。",
        "version": 1, "evidence_level": "experimental",
    }
    (built_in / "角色" / "identity.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    payload["id"] = "camera"
    (extra / "镜头" / "推进" / "slow.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    protocol = dict(payload, id="protocol_base")
    (built_in / "protocol_base.json").write_text(json.dumps(protocol, ensure_ascii=False), encoding="utf-8")
    (built_in / "broken.json").write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(mod, "_MODULES_DIR", str(built_in))
    monkeypatch.setattr(mod, "_EXTRA_ROOT", str(extra))

    choices = mod._module_file_choices()
    assert choices[0] == "（未选择）"
    assert "内置/角色/identity.json" in choices
    assert "用户库/镜头/推进/slow.json" in choices
    assert not any("protocol_base.json" in item for item in choices)
    assert not any("broken.json" in item for item in choices)


def test_module_file_loader_loads_only_selected_file(tmp_path, monkeypatch):
    mod = _load_mod()
    built_in = tmp_path / "modules"
    extra = tmp_path / "extra"
    built_in.mkdir()
    extra.mkdir()
    pack = [
        {"id": "one", "title_zh": "模块一", "version": 2,
         "evidence_level": "official_aligned", "instructions": "规则一。"},
        {"id": "two", "title_zh": "模块二", "instructions": "规则二。"},
    ]
    (built_in / "pack.json").write_text(json.dumps(pack, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(mod, "_MODULES_DIR", str(built_in))
    monkeypatch.setattr(mod, "_EXTRA_ROOT", str(extra))

    merged, diag = mod.MiniMaxH3ModuleFolderLoader().load_folder("内置/pack.json")
    assert "[one | 模块一 | v2 | 官方规则对齐]" in merged
    assert "规则一。" in merged and "规则二。" in merged
    assert "有效模块=2" in diag


def test_module_file_loader_combines_five_path_slots_and_deduplicates(tmp_path, monkeypatch):
    mod = _load_mod()
    built_in = tmp_path / "modules"
    extra = tmp_path / "extra"
    built_in.mkdir()
    extra.mkdir()
    first = {"id": "first", "title_zh": "第一模块", "instructions": "第一条规则。"}
    second = {"id": "second", "title_zh": "第二模块", "instructions": "第二条规则。"}
    duplicate = {"id": "first", "title_zh": "重复模块", "instructions": "不应重复注入。"}
    (built_in / "first.json").write_text(json.dumps(first, ensure_ascii=False), encoding="utf-8")
    (extra / "second.json").write_text(
        json.dumps([second, duplicate], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(mod, "_MODULES_DIR", str(built_in))
    monkeypatch.setattr(mod, "_EXTRA_ROOT", str(extra))

    node = mod.MiniMaxH3ModuleFolderLoader()
    merged, diag = node.load_folder(
        module_file="内置/first.json",
        module_file_2="用户库/second.json",
        module_file_3="内置/first.json",
    )
    assert "第一条规则。" in merged and "第二条规则。" in merged
    assert "不应重复注入。" not in merged
    assert merged.count("[first |") == 1
    assert "已选文件=2" in diag and "有效模块=2" in diag
    assert "重复文件选择已忽略" in diag
    assert "重复模块 id 已跳过" in diag


def test_module_file_loader_exposes_five_independent_path_combos():
    mod = _load_mod()
    required = mod.MiniMaxH3ModuleFolderLoader.INPUT_TYPES()["required"]
    assert list(required) == [
        "module_file", "module_file_2", "module_file_3", "module_file_4", "module_file_5"
    ]
    assert all(spec[0][0] == "（未选择）" for spec in required.values())


def test_module_file_loader_empty_and_stale_selection_are_safe(tmp_path, monkeypatch):
    mod = _load_mod()
    built_in = tmp_path / "modules"
    extra = tmp_path / "extra"
    built_in.mkdir()
    extra.mkdir()
    monkeypatch.setattr(mod, "_MODULES_DIR", str(built_in))
    monkeypatch.setattr(mod, "_EXTRA_ROOT", str(extra))

    merged, diag = mod.MiniMaxH3ModuleFolderLoader().load_folder()
    assert merged == ""
    assert "有效模块=0" in diag
    merged, diag = mod.MiniMaxH3ModuleFolderLoader().load_folder("内置/../outside.json")
    assert merged == ""
    assert "不存在或已移出允许目录" in diag


def test_merged_within_limit():
    mod = _load_mod()
    node = mod.MiniMaxH3PromptModuleLoader()
    merged, _, diag, _ = node.load(module_1="社区 8 条写作要诀", module_2="对白表演与口型",
                                   module_3="多参考图分镜映射")
    assert len(merged) <= mod._PROMPT_MODULE_MAX_CHARS + 20
    assert "合并字符" in diag


def test_selectable_modules_do_not_reintroduce_known_bad_syntax():
    mod = _load_mod()
    text = "\n".join(m["instructions"] for m in mod._load_modules())
    for forbidden in ("[LOOP]", "0-4s", "4-8s", "no prompt-only output",
                      "no subject movement", "默认竖屏 9:16", "15 秒 16:9"):
        assert forbidden not in text
