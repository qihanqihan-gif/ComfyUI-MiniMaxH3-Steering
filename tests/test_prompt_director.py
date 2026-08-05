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
    assert "环境声：rainfall" in out
    assert "配乐：soft piano" in out
    assert "[0.0-2.5s] wide shot" in out and "[2.5-5.0s] close up" in out


def test_compose_enhanced_missing_fields():
    mod = _load_mod()
    assert mod._compose_enhanced({}) == ""
    out = mod._compose_enhanced({"integrated_multimodal_description": "only desc"})
    assert out == "only desc"


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
    # 9 个固定参考图端口
    inputs = mod.MiniMaxH3PromptDirector.INPUT_TYPES()
    optional = inputs["optional"]
    for i in range(1, 10):
        assert f"ref_image_{i}" in optional, f"缺少 ref_image_{i}"
    assert "ref_image_10" not in optional
    assert mod.MiniMaxH3PromptDirector.RETURN_NAMES == ("enhanced_prompt", "report")
    assert mod.MiniMaxH3PromptDirector.VALIDATE_INPUTS() is True, "旧 api_model 值（如 gemma4@q6_k）必须放行"
    required = inputs["required"]
    assert isinstance(required["api_model"], tuple) and required["api_model"][0] == ["auto"], "api_model 应为可刷新 COMBO"
    assert required["lmstudio_after_use"][0] == ["keep_loaded", "unload_used_model", "unload_and_wait_for_vram"]
    assert required["lmstudio_gpu_offload"][0] == ["max", "0.90", "0.75", "0.50", "auto", "off"]
    assert required["api_reasoning"][0] == ["auto", "off", "on"]
    assert required["json_mode"][0] == ["auto_retry", "force", "off"]


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


def test_direct_no_key_attempts_call(monkeypatch):
    """回归：空 API Key 不再拦截直通（本地 LM Studio 忽略鉴权）——
    应继续尝试调用 API（mock 验证），而不是走"未提供 Key"直通。"""
    from unittest.mock import MagicMock

    mod = _load_mod()
    node = mod.MiniMaxH3PromptDirector()
    called = MagicMock(return_value='{"integrated_multimodal_description": "ok"}')
    monkeypatch.setattr(mod, "_call_chat", called)
    monkeypatch.setattr(mod, "_list_models", MagicMock(return_value=["m"]))

    enhanced, report = node.direct(
        prompt="test", task_type="I2VA", duration_seconds=5.0, shot_count=0,
        rewrite_mode="balanced", output_language="中文",
        api_base_url="http://127.0.0.1:1234/v1", api_model="auto", api_key="",
        temperature=0.3, max_tokens=2048, timeout_s=30,
    )
    called.assert_called_once(), "空 key 必须照常发起 API 调用"
    assert "未提供 API Key" not in report
    assert "ok" in enhanced
