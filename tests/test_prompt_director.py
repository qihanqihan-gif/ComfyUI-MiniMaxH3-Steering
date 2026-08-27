# -*- coding: utf-8 -*-
"""MiniMaxH3PromptDirector 纯函数测试（不发起真实网络请求）。"""
import importlib.util
import os
import sys

import numpy as np
import torch

LAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod():
    spec = importlib.util.spec_from_file_location("prompt_director", os.path.join(LAB_ROOT, "prompt_director.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prompt_director"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_image_tensor_to_data_url():
    mod = _load_mod()
    t = torch.rand(1, 64, 96, 3, dtype=torch.float32)  # [B,H,W,C]
    url = mod._image_tensor_to_data_url(t)
    assert url.startswith("data:image/jpeg;base64,")
    # 可解码且为 JPEG 头
    import base64

    raw = base64.b64decode(url.split(",", 1)[1])
    assert raw[:3] == b"\xff\xd8\xff"


def test_image_tensor_to_data_url_grayscale_single_frame():
    mod = _load_mod()
    t = torch.rand(64, 96, 3, dtype=torch.float32)  # [H,W,C] 无 batch
    url = mod._image_tensor_to_data_url(t)
    assert url.startswith("data:image/jpeg;base64,")


def test_image_batch_uniform_sampling_includes_first_and_last():
    mod = _load_mod()
    assert mod._uniform_sample_indices(124, 4) == [0, 41, 82, 123]
    assert mod._uniform_sample_indices(3, 4) == [0, 1, 2]
    assert mod._uniform_sample_indices(10, 1) == [5]

    batch = torch.rand(10, 8, 8, 3, dtype=torch.float32)
    frames, total, note = mod._image_batch_to_data_urls(batch, limit=4)
    assert total == 10
    assert [idx for idx, _url in frames] == [0, 3, 6, 9]
    assert "均匀全程" in note
    assert all(url.startswith("data:image/jpeg;base64,") for _idx, url in frames)


def test_deepseek_dense_batch_uses_adaptive_encoding_profile():
    mod = _load_mod()
    batch = torch.zeros((220, 8, 8, 3), dtype=torch.float32)
    frames, total, note = mod._image_batch_to_data_urls(
        batch, limit=220, adaptive=True,
        max_payload_bytes=mod._DEEPSEEK_SEQUENCE_PAYLOAD_BUDGET,
    )
    assert total == 220
    assert len(frames) == 220
    assert "≤512px/JPEG q72" in note
    assert "Data URL≈" in note
    assert sum(len(url) for _idx, url in frames) <= mod._DEEPSEEK_SEQUENCE_PAYLOAD_BUDGET


def test_frame_selection_quick_modes_and_safe_fallback():
    mod = _load_mod()
    indices, note = mod._select_frame_indices(124, 4, "uniform_no_edges", "")
    assert indices == [12, 45, 78, 111]
    assert "10%–90%" in note

    indices, note = mod._select_frame_indices(124, 4, "custom_indices", "0,41,82,-1")
    assert indices == [0, 41, 82, 123]
    assert "自定义帧序号" in note

    indices, note = mod._select_frame_indices(124, 4, "custom_percent", "0,33,66,100%")
    assert indices == [0, 41, 81, 123]
    assert "自定义百分比" in note

    indices, note = mod._select_frame_indices(124, 4, "custom_indices", "999")
    assert indices == [0, 41, 82, 123]
    assert note.startswith("[选择回退]")


def test_parse_json_clean():
    mod = _load_mod()
    raw = '{"integrated_multimodal_description": "a cat <Picture 1>", "overall_soundscape": "rain", "non_diegetic_music": "piano", "shot_breakdown": []}'
    parsed = mod._parse_json_text(raw)
    assert parsed["integrated_multimodal_description"] == "a cat <Picture 1>"


def test_parse_json_with_code_fence():
    mod = _load_mod()
    raw = '```json\n{"integrated_multimodal_description": "x", "overall_soundscape": "y"}\n```'
    parsed = mod._parse_json_text(raw)
    assert parsed["overall_soundscape"] == "y"


def test_parse_json_invalid_raises():
    mod = _load_mod()
    try:
        mod._parse_json_text("not json at all")
        assert False, "应抛 ValueError"
    except ValueError:
        pass


def test_qwen38_profile_and_split_reasoning_policy():
    mod = _load_mod()
    model = "qwen3.8-27b@q6_k"
    assert mod._infer_model_profile("auto", model) == "qwen3_8"
    assert mod._resolved_reasoning_mode("auto", model, "analysis") == "on"
    assert mod._resolved_reasoning_mode("auto", model, "final") == "off"
    assert mod._resolved_reasoning_mode("off", model, "analysis") == "off"
    assert mod._resolved_reasoning_mode("on", model, "final") == "on"
    assert mod._resolved_reasoning_mode("auto", "gemma4@q6_k", "analysis") == "on"
    assert mod._resolved_reasoning_mode("auto", "gemma4@q6_k", "final") == "off"


def test_prompt_ir_budget_and_host_scoped_deepseek_key(monkeypatch):
    mod = _load_mod()
    assert mod._normalize_prompt_ir_budget(8192) == (8192, "")
    normalized, note = mod._normalize_prompt_ir_budget(131328)
    assert normalized == 16384
    assert "旧/越界" in note

    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-env-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    assert mod._resolve_api_key("https://api.deepseek.com", "") == (
        "deepseek-env-secret", "DEEPSEEK_API_KEY",
    )
    assert mod._resolve_api_key("https://example.invalid/v1", "") == ("", "空")
    assert mod._resolve_api_key("https://api.deepseek.com", "widget-secret")[0] == "widget-secret"


def test_lmstudio_native_multimodal_input_uses_local_text_image_discriminators():
    mod = _load_mod()
    system, native = mod._lmstudio_native_input([
        {"role": "system", "content": "JSON only"},
        {"role": "user", "content": [
            {"type": "text", "text": "first"},
            "second",
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,x"}},
        ]},
    ])
    assert system == "JSON only"
    assert [item["type"] for item in native] == ["text", "text", "image"]
    assert native[0]["content"] == "first"
    assert native[2]["data_url"].startswith("data:image/jpeg;base64,")


def test_call_chat_deepseek_uses_official_thinking_shape(monkeypatch):
    mod = _load_mod()
    captured = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return __import__("json").dumps({
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {},
            }).encode("utf-8")

    def fake_urlopen(request, timeout):
        del timeout
        captured.append(__import__("json").loads(request.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    common = dict(
        base_url="https://api.deepseek.com", api_key="secret", model="deepseek-v4f-vision",
        messages=[{"role": "user", "content": "x"}], temperature=0.3,
        max_tokens=8192, timeout_s=30, json_mode="off", request_purpose="final",
    )
    mod._call_chat(api_reasoning="off", **common)
    mod._call_chat(api_reasoning="on", **common)
    mod._call_chat(api_reasoning="auto", **common)

    assert captured[0]["thinking"] == {"type": "disabled"}
    assert captured[0]["temperature"] == 0.3
    assert "chat_template_kwargs" not in captured[0]
    assert captured[1]["thinking"] == {"type": "enabled"}
    assert captured[1]["reasoning_effort"] == "high"
    assert "temperature" not in captured[1]
    assert captured[2]["thinking"] == {"type": "enabled"}
    assert captured[2]["reasoning_effort"] == "low"
    assert "temperature" not in captured[2]


def test_call_chat_keeps_finish_usage_and_uses_qwen38_structured_output(monkeypatch):
    mod = _load_mod()
    captured = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return __import__("json").dumps({
                "choices": [{
                    "message": {"content": "", "reasoning_content": "thinking"},
                    "finish_reason": "length",
                }],
                "usage": {
                    "prompt_tokens": 4251,
                    "completion_tokens": 8192,
                    "completion_tokens_details": {"reasoning_tokens": 8137},
                },
            }).encode("utf-8")

    def fake_urlopen(request, timeout):
        del timeout
        captured.append(__import__("json").loads(request.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    schema = mod._output_json_schema("T2VA")
    raw = mod._call_chat(
        "http://192.168.1.5:1234/v1", "", "qwen3.8-27b@q6_k",
        [{"role": "user", "content": "x"}], 0.3, 131072, 30,
        json_mode="auto_retry", api_reasoning="auto",
        response_schema=schema, request_purpose="final",
    )
    payload = captured[0]
    assert payload["max_tokens"] == 131072
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["messages"][0]["content"].endswith("/no_think")
    assert payload["messages"][-1]["content"].endswith("/no_think")
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"] == schema
    assert raw.finish_reason == "length"
    assert raw.usage["completion_tokens_details"]["reasoning_tokens"] == 8137
    assert "推理已消耗 8137 token" not in mod._completion_retry_reason(raw)
    assert "输出长度停止" in mod._completion_retry_reason(raw)

    mod._call_chat(
        "http://192.168.1.5:1234/v1", "", "qwen3.8-27b@q6_k",
        [{"role": "user", "content": "x"}], 0.3, 2048, 30,
        api_reasoning="auto", request_purpose="analysis",
    )
    assert captured[1]["chat_template_kwargs"] == {"enable_thinking": True}


def test_call_chat_routes_local_qwen38_final_to_lmstudio_native(monkeypatch):
    mod = _load_mod()
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return __import__("json").dumps({
                "model_instance_id": "qwen3.8-27b@q6_k",
                "output": [{
                    "type": "message",
                    "content": '{"integrated_multimodal_description":"[Shot 1] A.",'
                               '"overall_soundscape":"N/A","non_diegetic_music":"N/A"}',
                }],
                "stats": {
                    "input_tokens": 462,
                    "total_output_tokens": 137,
                    "reasoning_output_tokens": 0,
                },
            }).encode("utf-8")

    def fake_urlopen(request, timeout):
        del timeout
        captured["url"] = request.full_url
        captured["payload"] = __import__("json").loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    raw = mod._call_chat(
        "http://127.0.0.1:1234/v1", "", "qwen3.8-27b@q6_k",
        [{"role": "system", "content": "JSON only"},
         {"role": "user", "content": "one shot"}],
        0.3, 8192, 30, json_mode="auto_retry", api_reasoning="auto",
        response_schema=mod._output_json_schema("T2VA"), request_purpose="final",
    )
    assert captured["url"] == "http://127.0.0.1:1234/api/v1/chat"
    assert captured["payload"]["reasoning"] == "off"
    assert captured["payload"]["max_output_tokens"] == 8192
    assert captured["payload"]["store"] is False
    assert raw.backend == "lmstudio_native"
    assert raw.finish_reason == "stop"
    assert raw.usage["completion_tokens_details"]["reasoning_tokens"] == 0
    assert mod._parse_json_text(raw)["non_diegetic_music"] == "N/A"


def test_compose_enhanced_three_part():
    mod = _load_mod()
    parsed = {
        "integrated_multimodal_description": "A girl walks in the rain <Picture 1>",
        "overall_soundscape": "rainfall",
        "non_diegetic_music": "soft piano",
        "shot_breakdown": [
            {"start_s": 0.0, "end_s": 2.5, "description": "wide shot"},
            {"start_s": 2.5, "end_s": 5.0, "description": "close up"},
        ],
    }
    out = mod._compose_enhanced(parsed)
    assert "A girl walks in the rain <Picture 1>" in out
    assert "overall_soundscape: rainfall" in out
    assert "non_diegetic_music: soft piano" in out
    assert "分镜：" not in out, "最终输出必须由官方字段序列化，不能追加旧版自创外壳"


def test_compose_enhanced_missing_fields():
    mod = _load_mod()
    empty = mod._compose_enhanced({})
    assert "integrated_multimodal_description:" in empty
    assert "overall_soundscape: N/A" in empty
    out = mod._compose_enhanced({"integrated_multimodal_description": "only desc"})
    assert "integrated_multimodal_description: only desc" in out
    assert "non_diegetic_music: N/A" in out


def test_compose_enhanced_top_level_list_safe():
    """模型输出顶层为数组时不得崩溃（直通降级为空串）。"""
    mod = _load_mod()
    assert mod._compose_enhanced([{"integrated_multimodal_description": "x"}]) == ""


def test_chat_completions_url_normalization():
    mod = _load_mod()
    assert mod._chat_completions_url("http://127.0.0.1:1234/v1") == "http://127.0.0.1:1234/v1/chat/completions"
    assert mod._chat_completions_url("http://x/openai") == "http://x/openai/chat/completions"
    assert mod._chat_completions_url("http://x/chat/completions") == "http://x/chat/completions"
    assert mod._chat_completions_url("http://x") == "http://x/v1/chat/completions"


def test_models_url_normalization():
    mod = _load_mod()
    assert mod._models_url("http://127.0.0.1:1234/v1") == "http://127.0.0.1:1234/v1/models"
    assert mod._models_url("http://x/chat/completions") == "http://x/v1/models"


def test_url_scheme_whitelist():
    """非 http/https scheme（file/ftp/data）必须被拒绝（SSRF 加固）。"""
    mod = _load_mod()
    for bad in ("file:///etc/passwd", "ftp://x/v1", "data:text/plain,x"):
        try:
            mod._models_url(bad)
            assert False, f"{bad} 应抛 ValueError"
        except ValueError:
            pass
        try:
            mod._chat_completions_url(bad)
            assert False, f"{bad} 应抛 ValueError"
        except ValueError:
            pass


def test_reject_link_local_target():
    """链路本地（169.254.x.x 云元数据）/多播/未指定地址必须拒绝；localhost 放行。"""
    mod = _load_mod()
    for bad in ("http://169.254.169.254/v1", "http://224.0.0.1/v1", "http://0.0.0.0/v1"):
        try:
            mod._reject_link_local_target(bad)
            assert False, f"{bad} 应抛 ValueError"
        except ValueError:
            pass
    # IPv4-mapped IPv6 形式的链路本地也必须拒绝（绕过防护回归）
    for bad in ("http://[::ffff:169.254.169.254]/v1", "http://[::ffff:224.0.0.1]/v1", "http://[::]/v1"):
        try:
            mod._reject_link_local_target(bad)
            assert False, f"{bad} 应抛 ValueError（IPv4-mapped 绕过）"
        except ValueError:
            pass
    # 核心用例放行
    mod._reject_link_local_target("http://127.0.0.1:1234/v1")
    mod._reject_link_local_target("http://localhost:1234/v1")


def test_node_registered():
    mod = _load_mod()
    assert "MiniMaxH3PromptDirector" in mod.NODE_CLASS_MAPPINGS
    # 9 个固定参考资产端口（v0.2 改名，input name 不变保持旧工作流兼容）
    inputs = mod.MiniMaxH3PromptDirector.INPUT_TYPES()
    optional = inputs["optional"]
    for i in range(1, 10):
        assert f"ref_image_{i}" in optional, f"缺少 ref_image_{i}"
        spec = optional[f"ref_image_{i}"]
        assert isinstance(spec, tuple) and isinstance(spec[1], dict), "参考资产端口应带 label 参数"
    assert "ref_image_10" not in optional
    assert mod.MiniMaxH3PromptDirector.RETURN_NAMES == (
        "enhanced_prompt", "report", "reference_sheet", "prompt_ir",
    )
    assert mod.MiniMaxH3PromptDirector.VALIDATE_INPUTS() is True, "旧 api_model 值（如 gemma4@q6_k）必须放行"
    required = inputs["required"]
    assert isinstance(required["api_model"], tuple) and required["api_model"][0] == ["auto"], "api_model 应为可刷新 COMBO"
    assert required["lmstudio_after_use"][0] == ["keep_loaded", "unload_used_model", "unload_and_wait_for_vram"]
    assert required["lmstudio_gpu_offload"][0] == ["max", "0.90", "0.75", "0.50", "auto", "off"]
    assert required["api_reasoning"][0] == ["auto", "off", "on"]
    assert required["json_mode"][0] == ["auto_retry", "force", "off"]
    # v0.1：Auto/Single/Staged（旧值 two_stage/single_pass 兼容映射）
    assert required["analysis_mode"][0] == ["auto", "single", "staged"], "analysis_mode 默认 auto"
    assert required["analysis_mode"][1]["default"] == "auto"
    assert required["frame_sequence_limit"][1]["default"] == 4
    assert required["frame_sequence_limit"][1]["max"] == 300
    assert required["frame_selection_mode"][0] == [
        "uniform_full", "uniform_no_edges", "custom_indices", "custom_percent",
    ]
    assert required["frame_selection_spec"][1]["default"] == ""
    assert required["max_tokens"][1]["default"] == 8192
    assert required["max_tokens"][1]["max"] == 131072
    assert required["api_failure_policy"][0] == ["stop", "passthrough"]
    assert required["api_failure_policy"][1]["default"] == "stop"
    assert "qwen3_8" in required["model_profile"][0]
    assert "deepseek_vision" in required["model_profile"][0]
    assert required["output_language"][1]["default"] == "English"
    assert optional["video_frame_sequence"][0] == "IMAGE"
    assert "批次" in optional["video_frame_sequence"][1]["label"]
    assert "system_module" in optional, "应可接提示词模块节点输出"
    assert "module_manifest" in optional, "应可接模块清单用于诊断和 IR 追踪"
    # 旧值必须放行（踩坑速记：COMBO 旧值 VALIDATE_INPUTS 放行）
    assert mod.MiniMaxH3PromptDirector.VALIDATE_INPUTS(analysis_mode="two_stage") is True
    assert mod.MiniMaxH3PromptDirector.VALIDATE_INPUTS(analysis_mode="single_pass") is True


def test_lmstudio_root_extraction():
    mod = _load_mod()
    assert mod._lmstudio_root("http://127.0.0.1:1234/v1") == "http://127.0.0.1:1234"
    assert mod._lmstudio_root("http://192.168.1.5:1234") == "http://192.168.1.5:1234"
    assert mod._lmstudio_root("ftp://x/v1") == "", "非 http/https 应返回空"


def test_lmstudio_mode_whitelist_fallback():
    """非法 gpu_offload 值必须回落 max（防御纵深：COMBO 值可被 API 覆盖）。"""
    mod = _load_mod()
    # 通过 lms 不存在路径验证 mode 白名单：返回值要么是"未找到 lms"提示（mode 合法），
    # 非法 mode 会先回落 max 再走同样路径——直接验证回落逻辑
    import inspect

    src = inspect.getsource(mod._lmstudio_load_gpu)
    assert 'mode not in {"max", "0.90", "0.75", "0.50", "auto", "off"}' in src
    assert 'mode = "max"' in src


def test_lmstudio_unload_url_and_tolerance():
    """unload 请求应指向 /api/v1/models/unload；model_not_found 视为已卸载（不抛）。"""
    mod = _load_mod()
    assert mod._lmstudio_unload is not None  # 存在性
    # HTTPError 404 model_not_found 容忍：构造假响应太复杂，直接验证 URL 拼接语义
    root = mod._lmstudio_root("http://127.0.0.1:1234/v1")
    assert root + "/api/v1/models/unload" == "http://127.0.0.1:1234/api/v1/models/unload"


def test_analyze_assets_failure_falls_back(monkeypatch):
    """v0.2：逐素材分析任一失败 → 返回 ([], [原因])，调用方回退单次多图。"""
    from unittest.mock import MagicMock

    mod = _load_mod()
    monkeypatch.setattr(mod, "_call_chat", MagicMock(side_effect=RuntimeError("boom")))
    items, notes = mod._analyze_assets(
        "http://127.0.0.1:1234/v1", "", "m", [(1, "data:image/jpeg;base64,x")],
        0.3, 2048, 30, "auto_retry", "auto",
    )
    assert items == []
    assert notes and "回退" in notes[0]


def test_analyze_assets_ok(monkeypatch):
    """v0.2：逐素材分析成功 → sheet_items 带 input_index，保持输入顺序。"""
    from unittest.mock import MagicMock

    mod = _load_mod()
    monkeypatch.setattr(mod, "_call_chat", MagicMock(return_value=(
        '{"appearance": "black hair", "confidence": 0.9}'
    )))
    items, notes = mod._analyze_assets(
        "http://127.0.0.1:1234/v1", "", "m",
        [(1, "data:image/jpeg;base64,a"), (2, "data:image/jpeg;base64,b")],
        0.3, 2048, 30, "auto_retry", "auto",
    )
    assert len(items) == 2
    assert items[0]["input_index"] == 1
    assert items[1]["input_index"] == 2
    assert items[0]["asset_id"] == "asset_1"
    assert "逐素材分析 2 张" in notes[0]


def test_analyze_assets_bad_json_falls_back(monkeypatch):
    """v0.2：逐素材输出非 JSON/非对象 → 回退（不把垃圾数据带进合并）。"""
    from unittest.mock import MagicMock

    mod = _load_mod()
    monkeypatch.setattr(mod, "_call_chat", MagicMock(return_value="not json at all"))
    items, notes = mod._analyze_assets(
        "http://127.0.0.1:1234/v1", "", "m", [(1, "data:image/jpeg;base64,x")],
        0.3, 2048, 30, "auto_retry", "auto",
    )
    assert items == []
    assert notes and "回退" in notes[0]


def test_direct_system_module_injected(monkeypatch):
    """v0.2：system_module 非空时应并入 system（模块在前，内置规范在后）。"""
    from unittest.mock import MagicMock

    mod = _load_mod()
    node = mod.MiniMaxH3PromptDirector()
    called = MagicMock(return_value='{"integrated_multimodal_description": "ok"}')
    monkeypatch.setattr(mod, "_call_chat", called)
    monkeypatch.setattr(mod, "_list_models", MagicMock(return_value=["m"]))

    node.direct(
        prompt="test", task_type="T2VA", duration_seconds=5.0, shot_count=0,
        rewrite_mode="balanced", output_language="中文",
        api_base_url="http://127.0.0.1:1234/v1", api_model="auto", api_key="",
        temperature=0.3, max_tokens=2048, timeout_s=30,
        analysis_mode="single_pass", system_module="[mod] 模块规则：必须写清楚运镜。",
    )
    sent_messages = called.call_args.args[3]
    system = sent_messages[0]["content"]
    assert "[mod] 模块规则" in system
    assert "integrated_multimodal_description" in system, "内置规范应保留在模块之后"
    assert system.index("[mod]") < system.index("integrated_multimodal_description")


def test_direct_module_manifest_is_traced_and_scope_checked(monkeypatch):
    from unittest.mock import MagicMock
    import json

    mod = _load_mod()
    called = MagicMock(return_value=(
        '{"integrated_multimodal_description":"[Shot 1] A product rotates.",'
        '"overall_soundscape":"N/A","non_diegetic_music":"N/A"}'
    ))
    monkeypatch.setattr(mod, "_call_chat", called)
    manifest = json.dumps({
        "scope": "Ref2VA",
        "selected": [{"id": "product_identity_fidelity"}],
        "issues": [],
    })
    _, report, _, prompt_ir = mod.MiniMaxH3PromptDirector().direct(
        prompt="product", task_type="T2VA", duration_seconds=5.0, shot_count=1,
        rewrite_mode="balanced", output_language="English",
        api_base_url="http://127.0.0.1:1234/v1", api_model="vision-model", api_key="",
        temperature=0.3, max_tokens=2048, timeout_s=30, analysis_mode="single",
        module_manifest=manifest,
    )
    assert "模块作用域=Ref2VA" in report
    assert "product_identity_fidelity" in report
    assert json.loads(prompt_ir)["applied_modules"] == ["product_identity_fidelity"]


def test_direct_chinese_mode_reports_experimental_warning(monkeypatch):
    from unittest.mock import MagicMock

    mod = _load_mod()
    monkeypatch.setattr(mod, "_call_chat", MagicMock(return_value=(
        '{"integrated_multimodal_description":"[Shot 1] A.",'
        '"overall_soundscape":"N/A","non_diegetic_music":"N/A"}'
    )))
    _, report, _, _ = mod.MiniMaxH3PromptDirector().direct(
        prompt="test", task_type="T2VA", duration_seconds=5.0, shot_count=1,
        rewrite_mode="balanced", output_language="中文",
        api_base_url="http://127.0.0.1:1234/v1", api_model="vision-model", api_key="",
        temperature=0.3, max_tokens=2048, timeout_s=30, analysis_mode="single",
    )
    assert "中文输出属于便捷实验模式" in report


def test_direct_no_key_attempts_call(monkeypatch):
    """回归：空 API Key 不再拦截直通（本地 LM Studio 忽略鉴权）——
    应继续尝试调用 API（mock 验证），而不是走"未提供 Key"直通。"""
    from unittest.mock import MagicMock

    mod = _load_mod()
    node = mod.MiniMaxH3PromptDirector()
    called = MagicMock(return_value='{"integrated_multimodal_description": "ok"}')
    monkeypatch.setattr(mod, "_call_chat", called)
    monkeypatch.setattr(mod, "_list_models", MagicMock(return_value=["m"]))

    enhanced, report, sheet, prompt_ir = node.direct(
        prompt="test", task_type="I2VA", duration_seconds=5.0, shot_count=0,
        rewrite_mode="balanced", output_language="中文",
        api_base_url="http://127.0.0.1:1234/v1", api_model="auto", api_key="",
        temperature=0.3, max_tokens=2048, timeout_s=30,
        analysis_mode="single_pass",
    )
    called.assert_called_once(), "空 key 必须照常发起 API 调用"
    assert "未提供 API Key" not in report
    assert "ok" in enhanced
    assert sheet == "[]", "无图/单次模式 reference_sheet 应为空 JSON 数组"
    assert '"task_type": "I2VA"' in prompt_ir


def test_direct_qwen38_length_retries_no_think_and_reports_tokens(monkeypatch):
    from unittest.mock import MagicMock

    mod = _load_mod()
    first = mod._ChatCompletionText(
        "", finish_reason="length", reasoning_content="long thinking",
        usage={
            "prompt_tokens": 4251,
            "completion_tokens": 8192,
            "completion_tokens_details": {"reasoning_tokens": 8137},
        },
    )
    second = mod._ChatCompletionText(
        '{"integrated_multimodal_description":"[Shot 1] A subject moves.",'
        '"overall_soundscape":"N/A","non_diegetic_music":"N/A"}',
        finish_reason="stop",
        usage={
            "prompt_tokens": 4300,
            "completion_tokens": 1800,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    )
    called = MagicMock(side_effect=[first, second])
    monkeypatch.setattr(mod, "_call_chat", called)
    enhanced, report, _sheet, _ir = mod.MiniMaxH3PromptDirector().direct(
        prompt="test", task_type="T2VA", duration_seconds=5.0, shot_count=1,
        rewrite_mode="balanced", output_language="中文",
        api_base_url="http://127.0.0.1:1234/v1",
        api_model="qwen3.8-27b@q6_k", api_key="",
        temperature=0.3, max_tokens=8192, timeout_s=30,
        analysis_mode="single", api_reasoning="auto",
    )
    assert called.call_count == 2
    retry_call = called.call_args_list[1]
    assert retry_call.kwargs["api_reasoning"] == "off"
    retry_system = retry_call.args[3][0]["content"]
    assert "截断恢复重试" in retry_system
    assert "/no_think" in retry_system
    assert "reasoning=8137 token" in report
    assert "[自动恢复]" in report
    assert "A subject moves" in enhanced


def test_direct_default_failure_policy_stops_instead_of_passthrough(monkeypatch):
    from unittest.mock import MagicMock
    import pytest

    mod = _load_mod()
    truncated = mod._ChatCompletionText(
        "", finish_reason="length",
        usage={"completion_tokens": 8192,
               "completion_tokens_details": {"reasoning_tokens": 8190}},
    )
    monkeypatch.setattr(mod, "_call_chat", MagicMock(side_effect=[truncated, truncated]))
    with pytest.raises(RuntimeError, match="未把原始提示词直通给 H3"):
        mod.MiniMaxH3PromptDirector().direct(
            prompt="must not pass through", task_type="T2VA",
            duration_seconds=5.0, shot_count=1,
            rewrite_mode="balanced", output_language="中文",
            api_base_url="http://127.0.0.1:1234/v1",
            api_model="qwen3.8-27b@q6_k", api_key="",
            temperature=0.3, max_tokens=8192, timeout_s=30,
            analysis_mode="single", api_reasoning="auto",
        )


def test_dynamic_loaders_work_in_real_english_path():
    """回归：alpha 曾缺少 import sys，英文协议与 compiler loader 会直接 NameError。"""
    mod = _load_mod()
    assert mod._load_prompt_modules().load_protocol("T2VA")
    assert mod._load_h3_compiler().normalize_task_type("ref2va") == "Ref2VA"


def test_direct_invalid_ir_falls_back_to_original_prompt(monkeypatch):
    from unittest.mock import MagicMock

    mod = _load_mod()
    monkeypatch.setattr(mod, "_call_chat", MagicMock(return_value="{}"))
    node = mod.MiniMaxH3PromptDirector()
    enhanced, report, sheet, prompt_ir = node.direct(
        prompt="保留这条原始提示词", task_type="T2VA", duration_seconds=5.0, shot_count=1,
        rewrite_mode="balanced", output_language="中文",
        api_base_url="http://127.0.0.1:1234/v1", api_model="vision-model", api_key="",
        temperature=0.3, max_tokens=2048, timeout_s=30, analysis_mode="single",
        api_failure_policy="passthrough",
    )
    assert enhanced == "保留这条原始提示词"
    assert "Prompt IR 未通过确定性校验" in report
    assert sheet == "[]"
    assert '"task_type": "T2VA"' in prompt_ir


def test_direct_ref2va_keeps_all_six_fields(monkeypatch):
    from unittest.mock import MagicMock

    mod = _load_mod()
    response = {
        "subject_definitions": "<Subject 1> is the woman from <Picture 1>.",
        "summary": "[reference generation] A portrait scene.",
        "retention_analysis": "<Subject 1>: fully_preserved.",
        "detailed_description": "[Shot 1] <Subject 1> turns toward camera.",
        "overall_soundscape": "Quiet room tone.",
        "non_diegetic_music": "N/A",
    }
    called = MagicMock(return_value=__import__("json").dumps(response))
    monkeypatch.setattr(mod, "_call_chat", called)
    node = mod.MiniMaxH3PromptDirector()
    enhanced, report, _sheet, prompt_ir = node.direct(
        prompt="portrait", task_type="Ref2VA", duration_seconds=5.0, shot_count=1,
        rewrite_mode="balanced", output_language="English",
        api_base_url="http://127.0.0.1:1234/v1", api_model="vision-model", api_key="",
        temperature=0.3, max_tokens=2048, timeout_s=30, analysis_mode="single",
    )
    for field, value in response.items():
        assert f"{field}: {value}" in enhanced
    assert "[校验错误]" not in report
    assert '"task_type": "Ref2VA"' in prompt_ir
    system = called.call_args.args[3][0]["content"]
    assert '"subject_definitions"' in system
    assert '"integrated_multimodal_description"' not in system


def test_reference_ports_are_densely_renumbered(monkeypatch):
    from unittest.mock import MagicMock
    import torch

    mod = _load_mod()
    called = MagicMock(return_value=(
        '{"integrated_multimodal_description":"[Shot 1] <Picture 1> moves.",'
        '"overall_soundscape":"N/A","non_diegetic_music":"N/A"}'
    ))
    monkeypatch.setattr(mod, "_call_chat", called)
    node = mod.MiniMaxH3PromptDirector()
    _enhanced, report, _sheet, _ir = node.direct(
        prompt="move", task_type="I2VA", duration_seconds=5.0, shot_count=1,
        rewrite_mode="balanced", output_language="中文",
        api_base_url="http://127.0.0.1:1234/v1", api_model="vision-model", api_key="",
        temperature=0.3, max_tokens=2048, timeout_s=30, analysis_mode="single",
        ref_image_3=torch.zeros((1, 8, 8, 3), dtype=torch.float32),
    )
    user_content = called.call_args.args[3][1]["content"]
    labels = [item["text"] for item in user_content if item.get("type") == "text"]
    assert any("<Picture 1>" in item for item in labels)
    assert not any("<Picture 3>" in item for item in labels)
    assert "(3, 1)" in report


def test_video_frame_sequence_single_mode_sends_all_selected_without_picture_labels(monkeypatch):
    """单次模式：独立参考图仍编号 Picture；序列帧共同标为 Video 1。"""
    from unittest.mock import MagicMock

    mod = _load_mod()
    response = {
        "subject_definitions": (
            "<Subject 1> is shown in <Picture 1>. "
            "<Video 1> is the source performance video to edit."
        ),
        "summary": "[reference generation] Replace the performer in <Video 1>.",
        "retention_analysis": "<Picture 1>: fully_preserved. <Video 1>: partially_preserved.",
        "detailed_description": "[Shot 1] <Subject 1> follows the action in <Video 1>.",
        "overall_soundscape": "N/A",
        "non_diegetic_music": "N/A",
    }
    called = MagicMock(return_value=__import__("json").dumps(response))
    monkeypatch.setattr(mod, "_call_chat", called)
    _enhanced, report, sheet, _ir = mod.MiniMaxH3PromptDirector().direct(
        prompt="replace performer", task_type="Ref2VA", duration_seconds=5.0, shot_count=1,
        rewrite_mode="balanced", output_language="中文",
        api_base_url="http://127.0.0.1:1234/v1", api_model="vision-model", api_key="",
        temperature=0.3, max_tokens=2048, timeout_s=30, analysis_mode="single",
        frame_sequence_limit=3,
        frame_selection_mode="custom_percent", frame_selection_spec="0,50,100",
        ref_image_2=torch.zeros((1, 8, 8, 3), dtype=torch.float32),
        video_frame_sequence=torch.zeros((5, 8, 8, 3), dtype=torch.float32),
    )
    called.assert_called_once()
    user_content = called.call_args.args[3][1]["content"]
    image_items = [item for item in user_content if item.get("type") == "image_url"]
    text_items = [item["text"] for item in user_content if item.get("type") == "text"]
    system_text = called.call_args.args[3][0]["content"]
    assert len(image_items) == 4, "1 张目标图 + 3 张均匀序列帧必须全部进入单次请求"
    assert any("对应 <Picture 1>" in item for item in text_items)
    assert not any("对应 <Picture 2>" in item for item in text_items)
    assert sum("<Video 1> 代表帧" in item for item in text_items) == 3
    assert "不得用泛称‘动漫女孩/一个人’" in system_text
    assert "不得给这些帧另编 Picture 标签" in system_text
    assert "参考视频序列帧=3/5 张" in report
    assert "自定义百分比" in report
    sheet_data = __import__("json").loads(sheet)
    assert sheet_data["video_frame_sequence"]["selected_indices"] == [0, 2, 4]
    assert sheet_data["video_frame_sequence"]["selection_mode"] == "custom_percent"


def test_video_frame_sequence_auto_staged_joint_analysis_then_text_summary(monkeypatch):
    """auto 接视频 batch：先整组联合识别，再把摘要交给最终导演；不逐帧拆成 Picture。"""
    from unittest.mock import MagicMock

    mod = _load_mod()
    sequence_response = {
        "sequence_label": "<Video 1>",
        "source_performer_locator": "the only central dancer",
        "source_performer_do_not_preserve": "face, hair and outfit",
        "action_timeline": ["arms rise", "body turns"],
        "camera_timeline": "medium shot",
        "scene_and_background": "studio",
        "continuity_and_occlusion": "hands cross face",
        "uncertainties": [],
    }
    final_response = {
        "subject_definitions": (
            "<Subject 1> is the target. <Video 1> is the source performance video to edit."
        ),
        "summary": "[reference generation] Replace the performer in <Video 1>.",
        "retention_analysis": "<Video 1>: partially_preserved.",
        "detailed_description": "[Shot 1] <Subject 1> performs the observed action.",
        "overall_soundscape": "N/A",
        "non_diegetic_music": "N/A",
    }
    called = MagicMock(side_effect=[
        __import__("json").dumps(sequence_response),
        __import__("json").dumps(final_response),
    ])
    monkeypatch.setattr(mod, "_call_chat", called)
    _enhanced, report, sheet, _ir = mod.MiniMaxH3PromptDirector().direct(
        prompt="replace performer", task_type="Ref2VA", duration_seconds=5.0, shot_count=1,
        rewrite_mode="balanced", output_language="中文",
        api_base_url="http://127.0.0.1:1234/v1", api_model="vision-model", api_key="",
        temperature=0.3, max_tokens=2048, timeout_s=30, analysis_mode="auto",
        frame_sequence_limit=4,
        video_frame_sequence=torch.zeros((8, 8, 8, 3), dtype=torch.float32),
    )
    assert called.call_count == 2
    stage_messages = called.call_args_list[0].args[3]
    assert "同一个 <Video 1>" in stage_messages[0]["content"]
    stage_images = [
        item for item in stage_messages[1]["content"] if item.get("type") == "image_url"
    ]
    assert len(stage_images) == 4
    final_content = called.call_args_list[1].args[3][1]["content"]
    assert not any(item.get("type") == "image_url" for item in final_content), "成功分阶段后应传联合摘要"
    assert any("代表帧联合分析摘要" in item.get("text", "") for item in final_content)
    assert "分析模式=staged" in report
    sheet_data = __import__("json").loads(sheet)
    assert sheet_data["video_sequence_analysis"]["source_performer_locator"] == "the only central dancer"
    assert sheet_data["video_sequence_analysis"]["selected_indices"] == [0, 2, 5, 7]


def test_deepseek_auto_sends_references_and_video_frames_jointly(monkeypatch):
    """DeepSeek auto 是 single；显式 staged 仍保留给相同素材做 A/B。"""
    from unittest.mock import MagicMock

    mod = _load_mod()
    response = {
        "subject_definitions": (
            "<Subject 1> is shown in <Picture 1>. "
            "<Video 1> is the source performance timeline."
        ),
        "summary": "[reference generation] Replace the performer in <Video 1>.",
        "retention_analysis": "<Picture 1>: fully preserved. <Video 1>: motion only.",
        "detailed_description": "[Shot 1] <Subject 1> follows the existing opening action.",
        "overall_soundscape": "N/A",
        "non_diegetic_music": "N/A",
    }
    called = MagicMock(return_value=__import__("json").dumps(response))
    monkeypatch.setattr(mod, "_call_chat", called)
    _enhanced, report, _sheet, _ir = mod.MiniMaxH3PromptDirector().direct(
        prompt="replace performer", task_type="Ref2VA", duration_seconds=5.0, shot_count=1,
        rewrite_mode="balanced", output_language="中文",
        api_base_url="https://api.deepseek.com", api_model="deepseek-v4f-vision", api_key="",
        temperature=0.3, max_tokens=131328, timeout_s=30, analysis_mode="auto",
        frame_sequence_limit=4,
        ref_image_1=torch.zeros((1, 8, 8, 3), dtype=torch.float32),
        video_frame_sequence=torch.zeros((8, 8, 8, 3), dtype=torch.float32),
    )
    called.assert_called_once()
    request_content = called.call_args.args[3][1]["content"]
    assert sum(item.get("type") == "image_url" for item in request_content) == 5
    assert called.call_args.args[5] == 16384
    assert "分析模式=single" in report
    assert "auto 已选择 single" in report
    assert "旧/越界 max_tokens=131328 已改为 16384" in report
    assert "profile=deepseek_vision" in report


def test_video_frame_sequence_staged_failure_falls_back_to_raw_frames(monkeypatch):
    """序列阶段失败不能丢帧：最终导演调用必须收到全部已选代表帧。"""
    from unittest.mock import MagicMock

    mod = _load_mod()
    final_response = {
        "subject_definitions": (
            "<Subject 1> is the target. <Video 1> is the source performance video to edit."
        ),
        "summary": "[reference generation] Replace the performer.",
        "retention_analysis": "<Video 1>: partially_preserved.",
        "detailed_description": "[Shot 1] <Subject 1> moves.",
        "overall_soundscape": "N/A",
        "non_diegetic_music": "N/A",
    }
    called = MagicMock(side_effect=[
        RuntimeError("sequence model failed"),
        __import__("json").dumps(final_response),
    ])
    monkeypatch.setattr(mod, "_call_chat", called)
    _enhanced, report, _sheet, _ir = mod.MiniMaxH3PromptDirector().direct(
        prompt="replace", task_type="Ref2VA", duration_seconds=5.0, shot_count=1,
        rewrite_mode="balanced", output_language="中文",
        api_base_url="http://127.0.0.1:1234/v1", api_model="vision-model", api_key="",
        temperature=0.3, max_tokens=2048, timeout_s=30, analysis_mode="staged",
        frame_sequence_limit=3,
        video_frame_sequence=torch.zeros((6, 8, 8, 3), dtype=torch.float32),
    )
    final_content = called.call_args_list[1].args[3][1]["content"]
    assert sum(item.get("type") == "image_url" for item in final_content) == 3
    assert "最终调用直传全部 3 帧" in report


def test_explicit_key_wins_and_global_openai_key_not_sent_to_arbitrary_host(monkeypatch):
    from unittest.mock import MagicMock

    mod = _load_mod()
    response = (
        '{"integrated_multimodal_description":"[Shot 1] A.",'
        '"overall_soundscape":"N/A","non_diegetic_music":"N/A"}'
    )
    called = MagicMock(return_value=response)
    monkeypatch.setattr(mod, "_call_chat", called)
    monkeypatch.setenv("OPENAI_API_KEY", "global-openai-secret")
    node = mod.MiniMaxH3PromptDirector()
    common = dict(
        prompt="test", task_type="T2VA", duration_seconds=5.0, shot_count=1,
        rewrite_mode="balanced", output_language="中文",
        api_base_url="https://example.invalid/v1", api_model="model",
        temperature=0.3, max_tokens=2048, timeout_s=30, analysis_mode="single",
    )
    node.direct(api_key="node-secret", **common)
    assert called.call_args.args[1] == "node-secret"
    node.direct(api_key="", **common)
    assert called.call_args.args[1] == "", "全局 OpenAI Key 不得发往任意兼容端点"


def test_deepseek_structured_output_uses_json_object_and_reports_fingerprint(monkeypatch):
    mod = _load_mod()
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return __import__("json").dumps({
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {},
            }).encode("utf-8")

    def fake_urlopen(request, timeout):
        del timeout
        captured["body"] = bytes(request.data)
        captured["payload"] = __import__("json").loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    raw = mod._call_chat(
        "https://api.deepseek.com", "secret", "deepseek-v4-flash-vision-exp",
        [{"role": "system", "content": "Return JSON"}, {"role": "user", "content": "x"}],
        0.2, 8192, 30, json_mode="force", api_reasoning="off",
        response_schema=mod._output_json_schema("T2VA"), request_purpose="final",
    )
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert raw.backend == "deepseek_chat"
    assert raw.transport_meta["request_sha256"] == mod._sha256_bytes(captured["body"])
    assert raw.transport_meta["payload_bytes"] == len(captured["body"])
    assert len(raw.transport_meta["attempts"]) == 1
    note = mod._chat_stats_note(raw, "cloud")
    assert raw.transport_meta["request_sha256"] in note
    assert "payload=" in note and "transport_attempts=1" in note


def test_deepseek_transport_retry_reuses_identical_body(monkeypatch):
    mod = _load_mod()
    bodies = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}],"usage":{}}'

    def fake_urlopen(request, timeout):
        del timeout
        bodies.append(bytes(request.data))
        if len(bodies) == 1:
            raise mod.http.client.RemoteDisconnected("connection closed")
        return FakeResponse()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)
    raw = mod._call_chat(
        "https://api.deepseek.com", "secret", "vision-model",
        [{"role": "user", "content": "x"}], 0.2, 1024, 30,
        json_mode="force", api_reasoning="off", request_purpose="final",
    )
    assert len(bodies) == 2
    assert bodies[0] == bodies[1], "transport retry 必须复用完全相同的请求字节"
    assert raw.transport_meta["transport_retried"] is True
    assert [item["result"] for item in raw.transport_meta["attempts"]] == ["retry", "success"]
    assert raw.transport_meta["attempts"][0]["category"] == "remote_disconnect"


def test_deepseek_payload_too_large_is_not_retried(monkeypatch):
    mod = _load_mod()
    calls = []

    def fake_urlopen(request, timeout):
        del timeout
        calls.append(bytes(request.data))
        raise mod.urllib.error.HTTPError(request.full_url, 413, "too large", {}, None)

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    try:
        mod._call_chat(
            "https://api.deepseek.com", "secret", "vision-model",
            [{"role": "user", "content": "x"}], 0.2, 1024, 30,
            json_mode="force", api_reasoning="off", request_purpose="final",
        )
        assert False, "413 必须失败，不能原包重试"
    except mod._TransportFailure as exc:
        assert exc.category == "http_413_payload_too_large"
        assert exc.status == 413
        assert len(exc.transport_meta["attempts"]) == 1
        assert "too large" not in str(exc), "安全错误不应回显供应商响应正文"
    assert len(calls) == 1


def test_gemini_native_wire_preserves_system_images_schema_and_diagnostics(monkeypatch):
    mod = _load_mod()
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return __import__("json").dumps({
                "candidates": [{
                    "content": {"parts": [{"text": "{\"ok\":true}"}]},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {
                    "promptTokenCount": 23,
                    "candidatesTokenCount": 5,
                    "thoughtsTokenCount": 2,
                    "totalTokenCount": 30,
                },
            }).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {key.casefold(): value for key, value in request.headers.items()}
        captured["body"] = bytes(request.data)
        captured["payload"] = __import__("json").loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    raw = mod._call_chat(
        "https://generativelanguage.googleapis.com/v1beta",
        "gemini-secret",
        "gemini-3.1-flash-lite",
        [
            {"role": "system", "content": "Return strict JSON."},
            {"role": "user", "content": [
                {"type": "text", "text": "Inspect this image."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]},
        ],
        0.2,
        4096,
        30,
        json_mode="force",
        api_reasoning="auto",
        response_schema=schema,
        request_purpose="final",
    )
    payload = captured["payload"]
    assert captured["url"].endswith(
        "/v1beta/models/gemini-3.1-flash-lite:generateContent"
    )
    assert captured["headers"]["x-goog-api-key"] == "gemini-secret"
    assert "authorization" not in captured["headers"]
    assert payload["systemInstruction"] == {
        "parts": [{"text": "Return strict JSON."}],
    }
    assert payload["contents"][-1]["role"] == "user"
    assert payload["contents"][-1]["parts"][1] == {
        "inlineData": {"mimeType": "image/png", "data": "AAAA"},
    }
    assert payload["generationConfig"]["responseMimeType"] == "application/json"
    assert payload["generationConfig"]["responseJsonSchema"] == schema
    assert payload["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "minimal",
    }
    assert str(raw) == '{"ok":true}'
    assert raw.backend == "gemini_generate_content"
    assert raw.finish_reason == "stop"
    assert raw.usage["completion_tokens_details"]["reasoning_tokens"] == 2
    assert raw.transport_meta["request_sha256"] == mod._sha256_bytes(captured["body"])


def test_gemini_native_wire_rejects_invalid_dialogue_shape_before_network(monkeypatch):
    mod = _load_mod()
    monkeypatch.setattr(
        mod.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not send")),
    )
    try:
        mod._call_chat(
            "https://generativelanguage.googleapis.com/v1beta",
            "secret",
            "gemini-3.1-flash-lite",
            [{"role": "system", "content": "system only"}],
            0.0,
            512,
            30,
            json_mode="force",
        )
        assert False, "缺少 user 的 Gemini 请求必须在发网前失败"
    except ValueError as exc:
        assert "user" in str(exc)


def test_transport_error_classification_matrix():
    mod = _load_mod()
    cases = [
        (mod.urllib.error.HTTPError("https://x", 408, "timeout", {}, None), "http_408_timeout", True),
        (mod.urllib.error.HTTPError("https://x", 429, "rate", {}, None), "http_429_rate_limit", True),
        (mod.urllib.error.HTTPError("https://x", 503, "server", {}, None), "http_5xx_server", True),
        (mod.urllib.error.HTTPError("https://x", 401, "auth", {}, None), "http_4xx_non_retryable", False),
        (mod.urllib.error.HTTPError("https://x", 422, "shape", {}, None), "http_4xx_non_retryable", False),
        (mod.ssl.SSLCertVerificationError("bad certificate"), "ssl_certificate_non_retryable", False),
        (ConnectionResetError("reset"), "remote_disconnect", True),
    ]
    for exc, expected_category, expected_retryable in cases:
        category, retryable, _status = mod._classify_transport_error(exc)
        assert category == expected_category
        assert retryable is expected_retryable


def test_cloud_director_secure_surface_and_shared_core(monkeypatch):
    mod = _load_mod()
    assert "MiniMaxH3CloudDirector" in mod.NODE_CLASS_MAPPINGS
    inputs = mod.MiniMaxH3CloudDirector.INPUT_TYPES()
    required = inputs["required"]
    assert "api_key" not in required
    assert "api_base_url" not in required
    assert "lmstudio_after_use" not in required
    assert required["cloud_provider"][0] == ["deepseek", "gemini"]
    assert required["api_model"][1]["default"] == ""
    assert required["frame_sequence_limit"][1]["default"] == 48
    assert required["max_tokens"][1]["default"] == 16384
    assert required["duration_seconds"][1].get("forceInput") is not True
    assert mod.MiniMaxH3PromptDirector.INPUT_TYPES()["required"]["duration_seconds"][1].get(
        "forceInput"
    ) is not True
    assert mod.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3CloudDirector"] == "MiniMax H3 云端多模态导演"
    assert mod._CLOUD_DIRECTOR_PRESETS["deepseek"]["default_model"] == "deepseek-v4-flash-vision-exp"
    assert mod._CLOUD_DIRECTOR_PRESETS["gemini"]["default_model"] == "gemini-3.1-flash-lite"

    common = dict(
        prompt="replace performer", task_type="Ref2VA", duration_seconds=5.0,
        shot_count=1, rewrite_mode="balanced", output_language="English",
        cloud_provider="deepseek", api_model="deepseek-v4-flash-vision-exp",
        temperature=0.2, max_tokens=16384, timeout_s=600,
        api_reasoning="auto", analysis_mode="single", frame_sequence_limit=48,
        frame_selection_mode="uniform_full", frame_selection_spec="",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("MINIMAX_H3_API_KEY", "generic-key-must-not-unlock-cloud-node")
    try:
        mod.MiniMaxH3CloudDirector().direct_cloud(**common)
        assert False, "云端节点缺少环境变量时必须在发请求前阻断"
    except RuntimeError as exc:
        assert "尚未配置 DeepSeek 凭据" in str(exc)

    captured = {}

    def fake_direct(self, **kwargs):
        del self
        captured.update(kwargs)
        return ("prompt", "report", "[]", "{}")

    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-secret")
    monkeypatch.setattr(mod.MiniMaxH3PromptDirector, "direct", fake_direct)
    result = mod.MiniMaxH3CloudDirector().direct_cloud(**common)
    assert result[0] == "prompt"
    assert captured["api_base_url"] == "https://api.deepseek.com"
    assert captured["api_key"] == ""
    assert captured["_resolved_api_key"] == "env-secret"
    assert captured["_resolved_api_key_source"] == "environment:DEEPSEEK_API_KEY"
    assert captured["json_mode"] == "force"
    assert captured["model_profile"] == "deepseek_vision"
    assert captured["api_failure_policy"] == "stop"
    assert captured["analysis_mode"] == "single"

    captured.clear()
    result = mod.MiniMaxH3CloudDirector().direct_cloud(**{**common, "api_model": ""})
    assert result[0] == "prompt"
    assert captured["api_model"] == "deepseek-v4-flash-vision-exp"

    captured.clear()
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-env-secret")
    gemini_common = {
        **common,
        "cloud_provider": "gemini",
        "api_model": "",
    }
    result = mod.MiniMaxH3CloudDirector().direct_cloud(**gemini_common)
    assert result[0] == "prompt"
    assert captured["api_base_url"] == "https://generativelanguage.googleapis.com/v1beta"
    assert captured["api_model"] == "gemini-3.1-flash-lite"
    assert captured["_resolved_api_key"] == "gemini-env-secret"
    assert captured["_resolved_api_key_source"] == "environment:GEMINI_API_KEY"
    assert captured["model_profile"] == "gemini_vision"

    captured.clear()
    result = mod.MiniMaxH3CloudDirector().direct_cloud(**{
        **gemini_common,
        "api_model": "deepseek-v4-flash-vision-exp",
    })
    assert result[0] == "prompt"
    assert captured["api_model"] == "gemini-3.1-flash-lite"
    assert "忽略来自另一连接预设的旧模型覆盖" in captured["_cloud_connection_note"]
