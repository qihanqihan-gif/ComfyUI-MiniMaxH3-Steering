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
    """prompt_modules/builtin/ 应有核心协议 ×2 + 创作策略模块。"""
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
    # 社区/用户模块会持续增加；这里只验证最低基线，不再把扩展库数量写死。
    assert len(modules) >= 26, f"可选模块不应少于 26 个，实际 {len(modules)}"
    assert len(ids) == len(modules), "模块 ID 必须唯一"
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


def test_p0a_modules_use_canonical_contract_and_declared_ownership():
    mod = _load_mod()
    modules = {item["id"]: item for item in mod._load_modules()}
    expected = {
        "character_replacement": (["task.intent", "identity.policy.primary"], "identity.policy.primary"),
        "storyboard_reference_map": (["shot_plan.reference_mapping", "timeline.layout.primary"], "timeline.layout.primary"),
        "single_shot_continuity": (["timeline.layout.primary"], "timeline.layout.primary"),
        "natural_camera_follow": (["camera.motion.primary"], "camera.motion.primary"),
        "slow_push_in": (["camera.motion.primary"], "camera.motion.primary"),
        "subtle_still_motion": (["motion.micro"], ""),
    }
    contract_fields = {"purpose", "adds", "preserves", "forbids", "defaults"}
    for module_id, (writes, exclusive_group) in expected.items():
        item = modules[module_id]
        contract = item["semantic_contract"]
        assert contract_fields <= set(contract), module_id
        assert isinstance(contract["purpose"], str) and contract["purpose"].strip()
        assert isinstance(contract["adds"], list) and contract["adds"]
        assert all(isinstance(add, dict) and add.get("path") and "value" in add for add in contract["adds"])
        assert all(isinstance(contract[field], list) for field in ("preserves", "forbids", "defaults"))
        assert item["ownership"]["writes"] == writes
        assert isinstance(item.get("resolution"), dict)
        assert item["resolution"].get("exclusive_group", "") == exclusive_group
        assert isinstance(item["resolution"].get("priority"), int)
        assert not item.get("_metadata_issues"), f"{module_id}: {item['_metadata_issues']}"


def test_micro_motion_composes_with_exactly_one_primary_camera_rule():
    mod = _load_mod()
    modules = {item["id"]: item for item in mod._load_modules()}
    subtle = modules["subtle_still_motion"]

    for camera_id in ("slow_push_in", "natural_camera_follow"):
        resolved, suppressed, issues, _ = mod._resolve_modules(
            [modules[camera_id], subtle], scope="Ref2VA"
        )
        assert [item["id"] for item in resolved] == [camera_id, "subtle_still_motion"]
        assert suppressed == [] and issues == []

    resolved, suppressed, issues, trace = mod._resolve_modules(
        [modules["slow_push_in"], modules["natural_camera_follow"], subtle], scope="Ref2VA"
    )
    assert [item["id"] for item in resolved] == ["slow_push_in", "subtle_still_motion"]
    assert [item["id"] for item in suppressed] == ["natural_camera_follow"]
    assert "互斥规则已消解" in issues[0]
    assert trace[1]["reason_code"] == "exclusive_group"
    assert trace[1]["winner"] == "slow_push_in"


def test_character_replacement_only_owns_identity_and_composes_with_loop_policy():
    mod = _load_mod()
    modules = {item["id"]: item for item in mod._load_modules()}
    replacement = modules["character_replacement"]
    assert "background.policy.primary" not in replacement["ownership"]["writes"]
    assert {
        "source_video.background", "source_video.camera", "source_video.timeline"
    } <= set(replacement["semantic_contract"]["preserves"])

    resolved, suppressed, issues, _ = mod._resolve_modules(
        [replacement, modules["seamless_loop"]], scope="Ref2VA"
    )
    assert [item["id"] for item in resolved] == ["character_replacement", "seamless_loop"]
    assert suppressed == [] and issues == []


def test_p1_generic_modules_are_canonical_and_do_not_claim_unrelated_domains():
    mod = _load_mod()
    modules = {item["id"]: item for item in mod._load_modules()}
    expected = {
        "cinematic_realism": (
            ["visual.medium.primary", "visual.lighting.primary"], "visual.medium.primary"
        ),
        "negative_guidance": (["constraints.semantic_exclusions"], ""),
        "weather_ambience": (["environment.weather.primary"], "environment.weather.primary"),
    }
    contract_fields = {"purpose", "adds", "preserves", "forbids", "defaults"}
    forbidden_writes = {"camera.motion.primary", "timeline.layout.primary", "identity.policy.primary"}
    for module_id, (writes, exclusive_group) in expected.items():
        item = modules[module_id]
        contract = item["semantic_contract"]
        assert contract_fields <= set(contract), module_id
        assert item["ownership"]["writes"] == writes
        assert set(writes) <= {entry["path"] for entry in contract["adds"]}
        assert forbidden_writes.isdisjoint(writes)
        assert item["resolution"].get("exclusive_group", "") == exclusive_group
        assert not item.get("_metadata_issues"), f"{module_id}: {item['_metadata_issues']}"

    resolved, suppressed, issues, _ = mod._resolve_modules(
        [modules["cinematic_realism"], modules["negative_guidance"], modules["weather_ambience"]],
        scope="Ref2VA",
    )
    assert [item["id"] for item in resolved] == [
        "cinematic_realism", "negative_guidance", "weather_ambience"
    ]
    assert suppressed == [] and issues == []
    assert modules["negative_guidance"]["semantic_contract"]["defaults"][0]["value"] == 3
    assert {"camera.motion.primary", "timeline.layout.primary"} <= set(
        modules["weather_ambience"]["semantic_contract"]["preserves"]
    )


def test_official_scene_recipes_are_canonical_scoped_and_free_of_agent_workflow_text():
    mod = _load_mod()
    modules = {item["id"]: item for item in mod._load_modules()}
    style_recipes = {
        "scene_3d_animation",
        "scene_handdrawn_live",
        "scene_paper_collage",
        "scene_papercraft_stopmotion",
    }
    flexible_style_recipes = {
        "scene_brand_promo",
        "scene_coop_game_intro",
        "scene_minimalist_product_ad",
        "scene_mv_subtitle",
    }
    required_contract_fields = {"purpose", "adds", "preserves", "forbids", "defaults"}
    forbidden_workflow_text = (
        "API", "Agent", "画布", "审批", "下载", "工作流", "选项卡", "模型选择", "供应商连接",
    )

    for module_id in style_recipes | flexible_style_recipes:
        item = modules[module_id]
        contract = item["semantic_contract"]
        groups = set(mod._resolution_metadata(item)["exclusive_group"])
        add_paths = {entry["path"] for entry in contract["adds"]}
        scope_tokens = set(mod._scope_tokens(item["scope"]))

        assert item["version"] == 3
        assert item["category"] == "scene_recipe"
        assert item["evidence_level"] == "official_skill_adaptation"
        assert required_contract_fields <= set(contract)
        assert item["scope"] != "全部"
        assert scope_tokens <= {"T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA"}
        assert scope_tokens
        assert "scene.recipe.primary" in item["ownership"]["writes"]
        assert "scene.recipe.primary" in add_paths
        assert "scene.recipe.primary" in groups
        assert 180 <= len(item["instructions"]) <= 600
        assert not any(token in item["instructions"] for token in forbidden_workflow_text), module_id
        assert not item.get("_metadata_issues"), f"{module_id}: {item['_metadata_issues']}"

        if module_id in style_recipes:
            assert "visual.medium.primary" in item["ownership"]["writes"]
            assert "visual.medium.primary" in add_paths
            assert "visual.medium.primary" in groups
        else:
            assert "visual.medium.primary" not in item["ownership"]["writes"]
            assert groups == {"scene.recipe.primary"}


def test_scene_recipes_are_single_choice_and_visual_media_conflicts_are_resolved():
    mod = _load_mod()
    modules = {item["id"]: item for item in mod._load_modules()}

    resolved, suppressed, issues, trace = mod._resolve_modules(
        [modules["scene_paper_collage"], modules["scene_papercraft_stopmotion"]],
        scope="Ref2VA",
    )
    assert [item["id"] for item in resolved] == ["scene_paper_collage"]
    assert [item["id"] for item in suppressed] == ["scene_papercraft_stopmotion"]
    assert "互斥规则已消解" in issues[0]
    assert trace[1]["reason_code"] == "exclusive_group"
    assert trace[1]["winner"] == "scene_paper_collage"

    resolved, suppressed, issues, trace = mod._resolve_modules(
        [modules["cinematic_realism"], modules["scene_3d_animation"]],
        scope="T2VA",
    )
    assert [item["id"] for item in resolved] == ["scene_3d_animation"]
    assert [item["id"] for item in suppressed] == ["cinematic_realism"]
    assert "visual.medium.primary" in issues[0]
    assert trace[0]["winner"] == "scene_3d_animation"


def test_flexible_scene_recipes_compose_with_cinematic_and_visible_text_rules():
    mod = _load_mod()
    modules = {item["id"]: item for item in mod._load_modules()}

    resolved, suppressed, issues, _ = mod._resolve_modules(
        [modules["scene_brand_promo"], modules["cinematic_realism"]],
        scope="Ref2VA",
    )
    assert [item["id"] for item in resolved] == ["scene_brand_promo", "cinematic_realism"]
    assert suppressed == [] and issues == []

    resolved, suppressed, issues, _ = mod._resolve_modules(
        [modules["scene_mv_subtitle"], modules["ui_screen_text"]],
        scope="I2VA",
    )
    assert [item["id"] for item in resolved] == ["scene_mv_subtitle", "ui_screen_text"]
    assert suppressed == [] and issues == []
    assert "visible_text.verbatim" in modules["scene_mv_subtitle"]["semantic_contract"]["preserves"]


def test_minimalist_product_recipe_requires_reference_capable_scope_and_missing_asset_guard():
    mod = _load_mod()
    modules = {item["id"]: item for item in mod._load_modules()}
    product = modules["scene_minimalist_product_ad"]

    resolved, suppressed, issues, trace = mod._resolve_modules([product], scope="T2VA")
    assert resolved == []
    assert [item["id"] for item in suppressed] == ["scene_minimalist_product_ad"]
    assert "适用 I2VA,FL2VA,L2VA,Ref2VA" in issues[0]
    assert trace[0]["reason_code"] == "scope_mismatch"

    resolved, suppressed, issues, _ = mod._resolve_modules([product], scope="Ref2VA")
    assert [item["id"] for item in resolved] == ["scene_minimalist_product_ad"]
    assert suppressed == [] and issues == []
    assert "product_ad_without_connected_reference" in product["semantic_contract"]["forbids"]
    assert "仅在已连接真实产品参考时使用" in product["instructions"]


def test_p4_atomic_modules_are_canonical_scoped_and_do_not_duplicate_protocol_syntax():
    mod = _load_mod()
    modules = {item["id"]: item for item in mod._load_modules()}
    expected = {
        "anime_identity": (["identity.character.reference_fidelity"], {"identity.character.reference_fidelity"}),
        "product_identity_fidelity": (["identity.product.reference_fidelity"], {"identity.product.reference_fidelity"}),
        "two_person_interaction": (["interaction.blocking.primary"], {"interaction.blocking.primary"}),
        "dialogue_performance": (["audio.dialogue.performance.primary"], {"audio.dialogue.performance.primary"}),
        "ui_screen_text": (["visible_text.policy.primary"], {"visible_text.policy.primary"}),
        "motion_emphasis": (["motion.action.primary"], {"motion.action.primary"}),
        "parallax_motion": (["camera.motion.primary"], {"camera.motion.primary"}),
        "seamless_loop": (["timeline.loop.primary"], {"timeline.loop.primary"}),
        "live_wallpaper": (
            ["presentation.wallpaper.primary", "timeline.loop.primary"],
            {"presentation.wallpaper.primary", "timeline.loop.primary"},
        ),
    }
    contract_fields = {"purpose", "adds", "preserves", "forbids", "defaults"}
    protocol_fragments = ("<Subject", "<Picture", "(S1)", "(S2)", "retention_analysis", "@图片")
    workflow_fragments = ("API", "Agent", "画布", "审批", "下载", "工作流", "选项卡", "模型选择")

    for module_id, (writes, groups) in expected.items():
        item = modules[module_id]
        contract = item["semantic_contract"]
        add_paths = {entry["path"] for entry in contract["adds"]}
        scope_tokens = set(mod._scope_tokens(item["scope"]))

        assert item["version"] == 3
        assert item["scope"] != "全部"
        assert scope_tokens and scope_tokens <= {"T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA"}
        assert contract_fields <= set(contract)
        assert item["ownership"]["writes"] == writes
        assert set(writes) <= add_paths
        assert set(mod._resolution_metadata(item)["exclusive_group"]) == groups
        assert isinstance(item["resolution"]["priority"], int)
        assert 180 <= len(item["instructions"]) <= 600
        assert not any(fragment in item["instructions"] for fragment in protocol_fragments), module_id
        assert not any(fragment in item["instructions"] for fragment in workflow_fragments), module_id
        assert not item.get("_metadata_issues"), f"{module_id}: {item['_metadata_issues']}"


def test_p4_identity_interaction_dialogue_text_and_loop_rules_compose():
    mod = _load_mod()
    modules = {item["id"]: item for item in mod._load_modules()}
    selected = [
        modules["character_replacement"],
        modules["anime_identity"],
        modules["two_person_interaction"],
        modules["dialogue_performance"],
        modules["ui_screen_text"],
        modules["seamless_loop"],
    ]
    resolved, suppressed, issues, _ = mod._resolve_modules(selected, scope="Ref2VA")
    assert [item["id"] for item in resolved] == [item["id"] for item in selected]
    assert suppressed == [] and issues == []

    selected = [
        modules["scene_minimalist_product_ad"],
        modules["product_identity_fidelity"],
        modules["cinematic_realism"],
    ]
    resolved, suppressed, issues, _ = mod._resolve_modules(selected, scope="Ref2VA")
    assert [item["id"] for item in resolved] == [item["id"] for item in selected]
    assert suppressed == [] and issues == []


def test_p4_motion_camera_and_loop_conflicts_are_deterministic():
    mod = _load_mod()
    modules = {item["id"]: item for item in mod._load_modules()}

    resolved, suppressed, issues, trace = mod._resolve_modules(
        [modules["subtle_still_motion"], modules["motion_emphasis"]], scope="I2VA"
    )
    assert [item["id"] for item in resolved] == ["motion_emphasis"]
    assert [item["id"] for item in suppressed] == ["subtle_still_motion"]
    assert trace[0]["reason_code"] == "conflict"
    assert trace[0]["winner"] == "motion_emphasis"

    resolved, suppressed, issues, trace = mod._resolve_modules(
        [modules["slow_push_in"], modules["parallax_motion"]], scope="I2VA"
    )
    assert [item["id"] for item in resolved] == ["slow_push_in"]
    assert [item["id"] for item in suppressed] == ["parallax_motion"]
    assert trace[1]["reason_code"] == "exclusive_group"
    assert trace[1]["winner"] == "slow_push_in"

    resolved, suppressed, issues, trace = mod._resolve_modules(
        [modules["seamless_loop"], modules["live_wallpaper"]], scope="I2VA"
    )
    assert [item["id"] for item in resolved] == ["live_wallpaper"]
    assert [item["id"] for item in suppressed] == ["seamless_loop"]
    assert trace[0]["reason_code"] == "exclusive_group"
    assert trace[0]["winner"] == "live_wallpaper"

    resolved, suppressed, issues, trace = mod._resolve_modules(
        [modules["slow_push_in"], modules["seamless_loop"]], scope="Ref2VA"
    )
    assert [item["id"] for item in resolved] == ["seamless_loop"]
    assert [item["id"] for item in suppressed] == ["slow_push_in"]
    assert trace[0]["reason_code"] == "conflict"
    assert trace[0]["winner"] == "seamless_loop"


def test_p4_reference_dependent_rules_have_scope_and_missing_material_guards():
    mod = _load_mod()
    modules = {item["id"]: item for item in mod._load_modules()}

    for module_id in ("anime_identity", "product_identity_fidelity"):
        item = modules[module_id]
        resolved, suppressed, issues, trace = mod._resolve_modules([item], scope="T2VA")
        assert resolved == []
        assert [entry["id"] for entry in suppressed] == [module_id]
        assert trace[0]["reason_code"] == "scope_mismatch"
        assert any("without_connected" in rule for rule in item["semantic_contract"]["forbids"])

    resolved, suppressed, issues, trace = mod._resolve_modules(
        [modules["parallax_motion"]], scope="FL2VA"
    )
    assert resolved == []
    assert [entry["id"] for entry in suppressed] == ["parallax_motion"]
    assert trace[0]["reason_code"] == "scope_mismatch"
    assert "parallax_without_visible_depth_evidence" in modules["parallax_motion"]["semantic_contract"]["forbids"]


def test_reference_only_community_checklist_is_not_selectable_or_injectable():
    mod = _load_mod()
    path = os.path.join(LAB_ROOT, "prompt_modules", "builtin", "社区 8 条写作要诀.json")
    with open(path, encoding="utf-8") as handle:
        reference = json.load(handle)

    assert reference["id"] == "community_8_tips"
    assert reference["selectable"] is False
    assert reference["status"] == "reference_only"
    assert reference["reference_doc"] == "docs/创作规则作者检查清单.md"
    assert "community_8_tips" not in {item["id"] for item in mod._load_modules()}
    assert "社区 8 条写作要诀" not in mod._module_choices()
    assert mod._selectable_module_ids(path) == []
    assert not mod._file_has_selectable_module(path)
    assert not any(
        "社区 8 条写作要诀" in choice or "community_8_tips" in choice
        for choice in mod._module_file_index()
    )

    modules, issues = mod._modules_from_rule_pack({"modules": [reference]})
    assert modules == []
    assert "参考资料模块不可由外部规则包注入" in issues[0]


def test_authoring_templates_are_valid_and_advanced_contract_is_canonical():
    template_root = os.path.join(LAB_ROOT, "templates")
    template_paths = [
        os.path.join(template_root, "用户简单创作规则模板.example.json"),
        os.path.join(template_root, "用户高级创作规则模板.example.json"),
    ]
    parsed = []
    for template_path in template_paths:
        with open(template_path, encoding="utf-8") as handle:
            parsed.append(json.load(handle))

    advanced = parsed[1]
    contract = advanced["semantic_contract"]
    assert {"purpose", "adds", "preserves", "forbids", "defaults"} <= set(contract)
    assert contract["purpose"]
    assert contract["adds"] == [
        {"path": "camera.motion.primary", "value": "custom_primary_motion"}
    ]
    assert advanced["ownership"]["writes"] == ["camera.motion.primary"]
    assert advanced["resolution"]["exclusive_group"] == "camera.motion.primary"


def test_load_protocol_base_and_ref():
    """协议自动加载：Ref2VA → 六段式；其余 → 三段式。"""
    mod = _load_mod()
    base = mod.load_protocol("I2VA")
    ref = mod.load_protocol("Ref2VA")
    assert "integrated_multimodal_description" in base
    assert "subject_definitions" in ref
    assert "subject_definitions" not in base
    assert mod.load_protocol("ref2va").startswith("【官方 Ref2VA 协议")
    protocol_path = os.path.join(
        LAB_ROOT, "prompt_modules", "builtin", "核心协议：三段式（自动加载）.json"
    )
    with open(protocol_path, encoding="utf-8") as handle:
        assert "L2VA" in json.load(handle)["scope"]


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
    assert "[anime_identity | 动漫角色身份保持 | v3 | 官方规则对齐]" in merged
    assert "身份保持规则" in merged
    assert "（无）" not in merged
    assert json.loads(manifest)["selected"][0]["id"] == "anime_identity"


def test_render_profiles_are_deterministic_and_standard_preserves_existing_text():
    mod = _load_mod()
    module = next(item for item in mod._load_modules() if item["id"] == "anime_identity")

    standard, standard_source = mod._render_module_instructions(module, "standard")
    compact, compact_source = mod._render_module_instructions(module, "compact")
    strong, strong_source = mod._render_module_instructions(module, "strong")

    assert standard == module["instructions"]
    assert standard_source == "instructions"
    assert compact.startswith("目标：")
    assert "identity.character.reference_fidelity" in compact
    assert module["instructions"] not in compact
    assert compact_source == "semantic_contract_summary"
    assert strong.startswith(module["instructions"])
    assert "[Canonical semantic contract 复核]" in strong
    assert module["semantic_contract"]["purpose"] in strong
    assert "新增语义：" in strong
    assert strong_source == "instructions+semantic_contract"
    assert mod._render_module_instructions(module, "未知档位") == (standard, standard_source)


def test_compact_profile_is_actually_shorter_for_every_canonical_module():
    mod = _load_mod()
    for module in mod._load_modules():
        standard = mod._render_module_instructions(module, "standard")[0]
        compact = mod._render_module_instructions(module, "compact")[0]
        strong = mod._render_module_instructions(module, "strong")[0]
        assert len(compact) < len(standard), module["id"]
        assert len(strong) > len(standard), module["id"]


def test_loader_manifest_records_render_profile_and_reproducible_hashes():
    mod = _load_mod()
    node = mod.MiniMaxH3PromptModuleLoader()
    first = node.load(module_1="动漫角色身份保持", render_profile="compact")
    second = node.load(module_1="动漫角色身份保持", render_profile="compact")
    strong = node.load(module_1="动漫角色身份保持", render_profile="strong")

    first_payload = json.loads(first[3])
    second_payload = json.loads(second[3])
    strong_payload = json.loads(strong[3])
    assert first_payload["schema_version"] == "h3_prompt_modules/1.2"
    assert first_payload["render_profile"] == "compact"
    assert first_payload["render_profile_label"].startswith("紧凑")
    assert len(first_payload["module_set_sha256"]) == 64
    assert len(first_payload["rendered_system_sha256"]) == 64
    assert first_payload["rendered_modules"][0]["render_source"] == "semantic_contract_summary"
    assert first_payload["rendered_system_sha256"] == second_payload["rendered_system_sha256"]
    assert first_payload["module_set_sha256"] == second_payload["module_set_sha256"]
    assert first_payload["rendered_system_sha256"] != strong_payload["rendered_system_sha256"]
    assert first_payload["module_set_sha256"] != strong_payload["module_set_sha256"]
    assert "渲染档=紧凑" in first[2]


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
    merged, _, diag, manifest = mod.MiniMaxH3PromptModuleLoader().load(
        module_1="单镜连续性", module_2="多参考图分镜映射")
    payload = json.loads(manifest)
    assert "规则冲突已消解" in diag
    assert "[single_shot_continuity |" in merged
    assert "[storyboard_reference_map |" not in merged
    assert [item["id"] for item in payload["resolved"]] == ["single_shot_continuity"]
    assert payload["suppressed"][0]["id"] == "storyboard_reference_map"
    assert payload["suppressed"][0]["winner"] == "single_shot_continuity"


def test_scope_mismatch_is_suppressed_instead_of_injected():
    mod = _load_mod()
    merged, preview, diag, manifest = mod.MiniMaxH3PromptModuleLoader().load(
        scope="T2VA", module_1="人物替换（Ref2VA 换人复刻）")
    payload = json.loads(manifest)
    assert merged == ""
    assert "规则未应用" in diag and "人物替换" in preview
    assert payload["resolved"] == []
    assert payload["suppressed"][0]["reason_code"] == "scope_mismatch"


def test_exclusive_group_is_resolved_even_without_pairwise_conflict():
    mod = _load_mod()
    first = {
        "id": "first", "title_zh": "第一规则", "instructions": "A", "scope": "全部",
        "resolution": {"exclusive_group": "camera.motion.primary", "priority": 10},
    }
    second = {
        "id": "second", "title_zh": "第二规则", "instructions": "B", "scope": "全部",
        "resolution": {"exclusive_group": "camera.motion.primary", "priority": 10},
    }
    resolved, suppressed, issues, trace = mod._resolve_modules([first, second])
    assert [item["id"] for item in resolved] == ["first"]
    assert [item["id"] for item in suppressed] == ["second"]
    assert "互斥规则已消解" in issues[0]
    assert trace[1]["reason_code"] == "exclusive_group"


def test_resolution_priority_beats_slot_order_and_is_deterministic():
    mod = _load_mod()
    low = {
        "id": "low", "title_zh": "低优先", "instructions": "A", "scope": "全部",
        "resolution": {"exclusive_group": "task.primary", "priority": 10},
    }
    high = {
        "id": "high", "title_zh": "高优先", "instructions": "B", "scope": "全部",
        "resolution": {"exclusive_group": "task.primary", "priority": 90},
    }
    first = mod._resolve_modules([low, high])
    second = mod._resolve_modules([low, high])
    assert [item["id"] for item in first[0]] == ["high"]
    assert first[3] == second[3]
    assert first[3][0]["winner"] == "high"


def test_missing_and_suppressed_requirements_do_not_leak_into_prompt_plan():
    mod = _load_mod()
    dependent = {
        "id": "dependent", "title_zh": "依赖规则", "instructions": "A", "scope": "全部",
        "resolution": {"requires": ["base"], "priority": 100},
    }
    resolved, _, issues, trace = mod._resolve_modules([dependent])
    assert resolved == []
    assert "缺少依赖 base" in issues[0]
    assert trace[0]["reason_code"] == "missing_requirement"


def test_legacy_module_without_resolution_metadata_remains_compatible():
    mod = _load_mod()
    legacy = {"id": "legacy", "title_zh": "旧模块", "instructions": "旧规则。", "scope": "全部"}
    resolved, suppressed, issues, trace = mod._resolve_modules([legacy], "Ref2VA")
    assert resolved == [legacy]
    assert suppressed == [] and issues == []
    assert trace[0]["status"] == "applied"


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

    merged, diag, rule_pack = mod.MiniMaxH3ModuleFolderLoader().load_folder("内置/pack.json")
    assert "[one | 模块一 | v2 | 官方规则对齐]" in merged
    assert "规则一。" in merged and "规则二。" in merged
    assert "有效模块=2" in diag
    assert "不执行作用域/依赖/冲突消解" in diag
    assert [item["id"] for item in rule_pack["modules"]] == ["one", "two"]


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
    merged, diag, rule_pack = node.load_folder(
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
    assert [item["id"] for item in rule_pack["modules"]] == ["first", "second"]


def test_module_file_loader_exposes_five_independent_path_combos():
    mod = _load_mod()
    required = mod.MiniMaxH3ModuleFolderLoader.INPUT_TYPES()["required"]
    assert list(required) == [
        "module_file", "module_file_2", "module_file_3", "module_file_4", "module_file_5"
    ]
    assert all(spec[0][0] == "（未选择）" for spec in required.values())
    assert mod.MiniMaxH3ModuleFolderLoader.RETURN_TYPES == (
        "STRING", "STRING", "H3_PROMPT_RULE_PACK"
    )
    assert mod.MiniMaxH3PromptModuleLoader.INPUT_TYPES()["optional"]["external_rule_pack"][0] == (
        "H3_PROMPT_RULE_PACK"
    )


def test_smart_loader_scans_module_choices_only_once_per_node_definition(monkeypatch):
    mod = _load_mod()
    calls = []

    def fake_choices():
        calls.append(True)
        return ["（无）", "测试规则"]

    monkeypatch.setattr(mod, "_module_choices", fake_choices)
    required = mod.MiniMaxH3PromptModuleLoader.INPUT_TYPES()["required"]
    assert len(calls) == 1
    assert all(required[f"module_{index}"][0] == ["（无）", "测试规则"] for index in range(1, 6))
    assert required["module_1"][0] is not required["module_2"][0]


def test_module_file_loader_empty_and_stale_selection_are_safe(tmp_path, monkeypatch):
    mod = _load_mod()
    built_in = tmp_path / "modules"
    extra = tmp_path / "extra"
    built_in.mkdir()
    extra.mkdir()
    monkeypatch.setattr(mod, "_MODULES_DIR", str(built_in))
    monkeypatch.setattr(mod, "_EXTRA_ROOT", str(extra))

    merged, diag, rule_pack = mod.MiniMaxH3ModuleFolderLoader().load_folder()
    assert merged == ""
    assert "有效模块=0" in diag
    assert rule_pack["modules"] == []
    merged, diag, _ = mod.MiniMaxH3ModuleFolderLoader().load_folder("内置/../outside.json")
    assert merged == ""
    assert "不存在或已移出允许目录" in diag


def test_canonical_prompt_modules_use_chinese_filenames_and_keep_legacy_aliases():
    mod = _load_mod()
    canonical = os.path.join(LAB_ROOT, "prompt_modules", "builtin")
    names = sorted(name for name in os.listdir(canonical) if name.endswith(".json"))
    assert len(names) == 33
    assert "人物替换（Ref2VA 换人复刻）.json" in names
    assert "character_replacement.json" not in names
    choices = mod._module_file_choices()
    assert "内置/人物替换（Ref2VA 换人复刻）.json" in choices
    assert "内置/character_replacement.json" not in choices

    merged, diag, pack = mod.MiniMaxH3ModuleFolderLoader().load_folder(
        "内置/character_replacement.json"
    )
    assert "[character_replacement |" in merged
    assert "有效模块=1" in diag
    assert pack["modules"][0]["id"] == "character_replacement"


def test_file_loader_is_direct_but_rule_pack_can_use_smart_resolver():
    mod = _load_mod()
    direct, direct_diag, rule_pack = mod.MiniMaxH3ModuleFolderLoader().load_folder(
        module_file="内置/缓慢推进（氛围感）.json",
        module_file_2="内置/静态锚定·细微动态.json",
    )
    assert "[slow_push_in |" in direct
    assert "[subtle_still_motion |" in direct
    assert "不执行作用域/依赖/冲突消解" in direct_diag

    resolved, preview, diag, manifest = mod.MiniMaxH3PromptModuleLoader().load(
        external_rule_pack=rule_pack
    )
    payload = json.loads(manifest)
    assert "[slow_push_in |" in resolved
    assert "[subtle_still_motion |" in resolved
    assert "[未应用]" not in preview and "规则冲突已消解" not in diag
    assert [item["id"] for item in payload["resolved"]] == [
        "slow_push_in", "subtle_still_motion"
    ]
    assert payload["suppressed"] == []
    assert payload["resolved"][0]["origin"] == "file_loader"


def test_external_rule_pack_deduplicates_against_builtin_selection():
    mod = _load_mod()
    pack = {
        "schema_version": "h3_prompt_rule_pack/1.0",
        "modules": [{
            "id": "anime_identity",
            "title_zh": "重复动漫规则",
            "instructions": "不应覆盖内置规则。",
        }],
    }
    merged, _, diag, manifest = mod.MiniMaxH3PromptModuleLoader().load(
        module_1="动漫角色身份保持", external_rule_pack=pack
    )
    assert "不应覆盖内置规则。" not in merged
    assert "外部规则与已选规则 id 重复" in diag
    assert len(json.loads(manifest)["selected"]) == 1


def test_canonical_and_legacy_roots_do_not_duplicate_builtin_modules():
    mod = _load_mod()
    roots = mod._module_dirs()
    assert os.path.realpath(roots[0]) == os.path.realpath(
        os.path.join(LAB_ROOT, "prompt_modules", "builtin")
    )
    assert any(
        os.path.realpath(path)
        == os.path.realpath(os.path.join(LAB_ROOT, "prompt_modules", "legacy"))
        for path in roots
    )
    modules = mod._load_modules()
    assert len({item["id"] for item in modules}) == len(modules)
    assert len(modules) >= 26


def test_prompt_module_storage_has_one_top_level_entry_point():
    root = os.path.join(LAB_ROOT, "prompt_modules")
    assert set(name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))) == {
        "builtin", "legacy", "user"
    }
    assert not os.path.exists(os.path.join(LAB_ROOT, "modules"))
    assert not os.path.exists(os.path.join(LAB_ROOT, "extra_module_roots"))


def test_merged_within_limit():
    mod = _load_mod()
    node = mod.MiniMaxH3PromptModuleLoader()
    merged, _, diag, _ = node.load(module_1="写实电影感", module_2="对白表演与口型",
                                   module_3="多参考图分镜映射")
    assert len(merged) <= mod._PROMPT_MODULE_MAX_CHARS + 20
    assert "合并字符" in diag


def test_selectable_modules_do_not_reintroduce_known_bad_syntax():
    mod = _load_mod()
    text = "\n".join(m["instructions"] for m in mod._load_modules())
    for forbidden in ("[LOOP]", "0-4s", "4-8s", "no prompt-only output",
                      "no subject movement", "默认竖屏 9:16", "15 秒 16:9"):
        assert forbidden not in text
