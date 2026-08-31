"""provider_registry 能力矩阵 + contract 层纯函数测试（不联网）。"""
import importlib.util
import os
import sys

LAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod():
    spec = importlib.util.spec_from_file_location(
        "provider_registry", os.path.join(LAB_ROOT, "provider_registry.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["provider_registry"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_profiles_registered_with_evidence_status():
    mod = _load_mod()
    profiles = mod.all_profiles()
    assert set(profiles) == {
        "deepseek_chat_v4_flash_vision",
        "gemini_generate_content_flash_lite",
        "openai_responses_gpt5_documented",
        "claude_messages_native_documented",
        # 2026-08-28：新增常用云端视觉模型（documented）
        "volcengine_doubao_seed_evolving_documented",
        "anthropic_claude_sonnet46_documented",
        "zhipu_glm53_flash_documented",
        "minimax_m27_text_documented",
        "custom_openai_compat_documented",
    }
    # 已实测的两家标 measured；无 key 的 OpenAI/Claude 只能 documented
    assert profiles["deepseek_chat_v4_flash_vision"].evidence is mod.EvidenceStatus.MEASURED
    assert profiles["gemini_generate_content_flash_lite"].evidence is mod.EvidenceStatus.MEASURED
    assert profiles["openai_responses_gpt5_documented"].evidence is mod.EvidenceStatus.DOCUMENTED
    assert profiles["claude_messages_native_documented"].evidence is mod.EvidenceStatus.DOCUMENTED
    # GLM-5.3-Flash 已用真实 key 跑通视觉探针 → measured；其余新增家未实测 → documented。
    assert profiles["zhipu_glm53_flash_documented"].evidence is mod.EvidenceStatus.MEASURED
    for pid in (
        "volcengine_doubao_seed_evolving_documented",
        "anthropic_claude_sonnet46_documented",
        "minimax_m27_text_documented",
        "custom_openai_compat_documented",
    ):
        assert profiles[pid].evidence is mod.EvidenceStatus.DOCUMENTED
    # measured 必须能被 to_dict 序列化且 evidence 为 measured
    assert profiles["zhipu_glm53_flash_documented"].to_dict()["evidence"] == "measured"
    # documented 必须能被 to_dict 序列化且 evidence 有值
    assert profiles["openai_responses_gpt5_documented"].to_dict()["evidence"] == "documented"


def test_implemented_modalities_separates_official_vs_plugin():
    mod = _load_mod()
    gemini = mod.get_profile("gemini_generate_content_flash_lite")
    # Gemini 原生视频已经实现 inline + Files API；其他 provider 仍须抽帧。
    assert "video_native" in gemini.modalities
    assert "video_native" in gemini.implemented_modalities
    assert gemini.must_extract_frames is False
    # OpenAI / Claude 官方也无原生视频内容块
    assert mod.get_profile("openai_responses_gpt5_documented").must_extract_frames is True
    assert mod.get_profile("claude_messages_native_documented").must_extract_frames is True


def test_contract_accepts_gemini_native_video_and_rejects_other_wires():
    mod = _load_mod()
    req = mod.DirectorRequestContract(
        task_type="Ref2VA", num_reference_images=4, num_video_frames=0,
        wants_native_video=True, wants_reasoning="off",
    )
    assert mod.contract_result("gemini_generate_content_flash_lite", req)["ok"] is True
    assert mod.contract_result("deepseek_chat_v4_flash_vision", req)["ok"] is False
    planable = [item["profile_id"] for item in mod.find_planable_profiles(req)]
    assert planable == ["gemini_generate_content_flash_lite"]


def test_contract_rejects_over_image_limit():
    mod = _load_mod()
    req = mod.DirectorRequestContract(
        task_type="Ref2VA", num_reference_images=700, num_video_frames=0,
        wants_reasoning="off",
    )
    res = mod.contract_result("claude_messages_native_documented", req)
    assert res["ok"] is False
    assert any("700" in p and "上限" in p for p in res["problems"])


def test_contract_rejects_unknown_reasoning_level():
    mod = _load_mod()
    req = mod.DirectorRequestContract(
        task_type="Ref2VA", num_reference_images=1, num_video_frames=0, wants_reasoning="ultra",
    )
    res = mod.contract_result("gemini_generate_content_flash_lite", req)
    assert res["ok"] is False
    assert any("未知推理强度" in p for p in res["problems"])


def test_contract_accepts_normal_ref2va_request():
    mod = _load_mod()
    # 4 图 + 48 帧 + off：DeepSeek 与 Gemini 实测可承载；OpenAI/Claude 至少在 contract 层可承载
    req = mod.DirectorRequestContract(
        task_type="Ref2VA", num_reference_images=4, num_video_frames=48, wants_reasoning="off",
    )
    assert mod.contract_result("deepseek_chat_v4_flash_vision", req)["ok"] is True
    assert mod.contract_result("gemini_generate_content_flash_lite", req)["ok"] is True
    assert mod.contract_result("openai_responses_gpt5_documented", req)["ok"] is True
    assert mod.contract_result("claude_messages_native_documented", req)["ok"] is True


# ============================================================================
# 阶段 A-1：(连接预设↔能力 profile) 绑定 + 业务层统一语义解析（以 registry 为准）
# ============================================================================

def test_cloud_preset_binding_resolves_to_registered_profile():
    mod = _load_mod()
    # 所有连接预设都必须能绑定到已注册能力 profile（连接层不得声称无能力声明的能力）。
    assert set(mod.all_cloud_preset_ids()) == {
        "deepseek", "gemini", "doubao", "claude", "glm", "minimax", "custom", "my_presets",
    }
    for pid in mod.all_cloud_preset_ids():
        profile = mod.profile_for_preset(pid)
        assert profile.profile_id in mod.all_profiles()
        binding = mod.preset_binding(pid)
        assert binding["capability_profile_id"] == profile.profile_id
        assert binding["provider"] == profile.provider
        assert binding["evidence"] == profile.evidence.value
    # 未注册 preset 必须抛 KeyError（防止连接层引用幽灵能力）。
    try:
        mod.profile_for_preset("not_a_preset")
        raise AssertionError("profile_for_preset 对未注册 preset 应抛 KeyError")
    except KeyError:
        pass


def test_find_profile_for_model_hits_cloud_and_ignores_local():
    mod = _load_mod()
    assert mod.find_profile_for_model("deepseek-v4-flash-vision-exp").profile_id == "deepseek_chat_v4_flash_vision"
    assert mod.find_profile_for_model("gemini-3.1-flash-lite").profile_id == "gemini_generate_content_flash_lite"
    # 本地/未登记模型不得命中（由调用方保留原启发式）。
    assert mod.find_profile_for_model("qwen3.8-27b@q6_k") is None
    assert mod.find_profile_for_model("gemma4@q6_k") is None
    assert mod.find_profile_for_model("") is None


def test_normalize_reasoning_semantic_aliases_and_defaults():
    mod = _load_mod()
    assert mod.normalize_reasoning_semantic("on") == "high"
    assert mod.normalize_reasoning_semantic("") == "auto"
    assert mod.normalize_reasoning_semantic(None) == "auto"
    assert mod.normalize_reasoning_semantic("off") == "off"
    assert mod.normalize_reasoning_semantic("max") == "max"
    assert mod.normalize_reasoning_semantic("ultra") == "auto"


def test_resolve_reasoning_semantic_respects_can_disable():
    mod = _load_mod()
    ds = mod.get_profile("deepseek_chat_v4_flash_vision")
    gem = mod.get_profile("gemini_generate_content_flash_lite")
    # DeepSeek 能关思考：off 保留 off；auto 透传。
    assert ds.reasoning.can_disable is True
    assert mod.resolve_reasoning_semantic(ds, "off") == "off"
    assert mod.resolve_reasoning_semantic(ds, "auto") == "auto"
    # Gemini 3 系不能关思考：off 抬升到最低非 off 级（low）；auto 透传。
    assert gem.reasoning.can_disable is False
    assert mod.resolve_reasoning_semantic(gem, "off") == "low"
    assert mod.resolve_reasoning_semantic(gem, "auto") == "auto"
    # 显式 level 保持。
    assert mod.resolve_reasoning_semantic(ds, "high") == "high"
    assert mod.resolve_reasoning_semantic(gem, "max") == "max"


def test_capability_profile_to_dict_exposes_binding_fields():
    mod = _load_mod()
    d = mod.get_profile("deepseek_chat_v4_flash_vision").to_dict()
    assert d["evidence"] == "measured"
    assert d["reasoning"]["can_disable"] is True
    assert "modalities" in d and "implemented_modalities" in d
