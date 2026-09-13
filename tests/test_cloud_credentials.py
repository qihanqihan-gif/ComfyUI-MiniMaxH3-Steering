import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_mod():
    spec = importlib.util.spec_from_file_location("minimax_h3_cloud_credentials_test", ROOT / "cloud_credentials.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_status_is_read_only_and_does_not_create_storage(tmp_path, monkeypatch):
    mod = _load_mod()
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    path = mod.credential_file_path(tmp_path)
    status = mod.credential_status("deepseek", "deepseek_default", user_dir=tmp_path)
    assert status["configured"] is False
    assert status["source"] == "unconfigured"
    assert status["storage_path"] == str(path)
    assert not path.exists()
    assert not path.parent.exists(), "查看状态不得提前创建目录"


def test_first_save_creates_plaintext_local_file_but_status_is_redacted(tmp_path, monkeypatch):
    mod = _load_mod()
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    secret = "unit-local-value-should-never-enter-status"
    status = mod.save_credential("deepseek", "deepseek_default", secret, user_dir=tmp_path)
    path = mod.credential_file_path(tmp_path)
    assert path.exists()
    assert status["configured"] is True
    assert status["source"] == "local_file:deepseek_default"
    assert secret not in json.dumps(status, ensure_ascii=False)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["credentials"]["deepseek_default"]["api_key"] == secret
    resolved, source = mod.resolve_credential("deepseek", "deepseek_default", user_dir=tmp_path)
    assert (resolved, source) == (secret, "local_file:deepseek_default")


def test_environment_overrides_local_file_and_is_never_returned_by_status(tmp_path, monkeypatch):
    mod = _load_mod()
    mod.save_credential("deepseek", "deepseek_default", "local-value", user_dir=tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-value")
    resolved, source = mod.resolve_credential("deepseek", "deepseek_default", user_dir=tmp_path)
    assert resolved == "environment-value"
    assert source == "environment:DEEPSEEK_API_KEY"
    status = mod.credential_status("deepseek", "deepseek_default", user_dir=tmp_path)
    serialized = json.dumps(status)
    assert status["environment_configured"] is True
    assert status["local_file_configured"] is True
    assert "environment-value" not in serialized
    assert "local-value" not in serialized


def test_provider_credentials_share_one_store_without_cross_provider_resolution(tmp_path, monkeypatch):
    mod = _load_mod()
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    mod.save_credential("deepseek", "deepseek_default", "deepseek-local", user_dir=tmp_path)
    mod.save_credential("gemini", "gemini_default", "gemini-local", user_dir=tmp_path)
    assert mod.resolve_credential("deepseek", "deepseek_default", user_dir=tmp_path) == (
        "deepseek-local", "local_file:deepseek_default",
    )
    assert mod.resolve_credential("gemini", "gemini_default", user_dir=tmp_path) == (
        "gemini-local", "local_file:gemini_default",
    )
    with pytest.raises(mod.CloudCredentialError, match="不匹配"):
        mod.resolve_credential("gemini", "deepseek_default", user_dir=tmp_path)


def test_clear_removes_last_secret_file_but_not_environment(tmp_path, monkeypatch):
    mod = _load_mod()
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    mod.save_credential("deepseek", "deepseek_default", "local-value", user_dir=tmp_path)
    path = mod.credential_file_path(tmp_path)
    status = mod.clear_credential("deepseek", "deepseek_default", user_dir=tmp_path)
    assert status["configured"] is False
    assert not path.exists()
    assert path.parent.exists(), "清除最后一个 Key 后允许保留空目录"


def test_corrupt_store_is_not_silently_overwritten(tmp_path):
    mod = _load_mod()
    path = mod.credential_file_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(mod.CloudCredentialError, match="损坏|不可读"):
        mod.save_credential("deepseek", "deepseek_default", "replacement", user_dir=tmp_path)
    assert path.read_text(encoding="utf-8") == "{broken"


def test_ids_and_empty_secrets_are_rejected_without_creating_files(tmp_path):
    mod = _load_mod()
    with pytest.raises(mod.CloudCredentialError):
        mod.save_credential("../deepseek", "deepseek_default", "value", user_dir=tmp_path)
    with pytest.raises(mod.CloudCredentialError, match="不能为空"):
        mod.save_credential("deepseek", "deepseek_default", "", user_dir=tmp_path)
    assert not mod.credential_file_path(tmp_path).exists()


def test_deepseek_probe_never_returns_or_embeds_secret(monkeypatch):
    mod = _load_mod()
    secret = "probe-only-value"

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"data":[{"id":"vision-a"},{"id":"vision-b"}]}'

    captured = {}

    def fake_urlopen(request, timeout):
        captured["authorization"] = request.headers.get("Authorization")
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    result = mod.test_deepseek_connection(secret, timeout_s=15)
    assert result == {"ok": True, "provider": "deepseek", "model_count": 2}
    assert captured["authorization"] == f"Bearer {secret}"
    assert secret not in json.dumps(result)


def test_deepseek_model_list_returns_only_safe_visible_ids(monkeypatch):
    mod = _load_mod()
    secret = "deepseek-list-secret"

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "data": [
                    {"id": "deepseek-v4-flash-vision-exp"},
                    {"id": "deepseek-chat"},
                    {"id": "deepseek-chat"},
                ],
            }).encode("utf-8")

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda request, timeout: Response())
    models = mod.list_cloud_models("deepseek", secret, 15)
    assert [item["id"] for item in models] == [
        "deepseek-v4-flash-vision-exp", "deepseek-chat",
    ]
    assert models[0]["recommended"] is True
    assert models[1]["recommended"] is False
    assert secret not in json.dumps(models)


def test_gemini_model_list_filters_non_generate_content_and_marks_director_candidates(monkeypatch):
    mod = _load_mod()
    secret = "gemini-list-secret"

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "models": [
                    {
                        "name": "models/gemini-3.1-flash-lite",
                        "displayName": "Gemini 3.1 Flash-Lite",
                        "supportedGenerationMethods": ["generateContent", "countTokens"],
                        "inputTokenLimit": 1048576,
                        "outputTokenLimit": 65536,
                    },
                    {
                        "name": "models/gemini-3.1-flash-image",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/text-embedding-004",
                        "supportedGenerationMethods": ["embedContent"],
                    },
                ],
            }).encode("utf-8")

    captured = {}

    def fake_urlopen(request, timeout):
        captured["header"] = {key.casefold(): value for key, value in request.headers.items()}
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    models = mod.list_cloud_models("gemini", secret, 30)
    assert [item["id"] for item in models] == [
        "gemini-3.1-flash-lite", "gemini-3.1-flash-image",
    ]
    assert models[0]["display_name"] == "Gemini 3.1 Flash-Lite"
    assert models[0]["recommended"] is True
    assert models[0]["input_token_limit"] == 1048576
    assert models[1]["recommended"] is False
    assert captured["header"]["x-goog-api-key"] == secret
    assert secret not in json.dumps(models)


def test_gemini_probe_validates_list_image_and_strict_json_without_returning_secret(monkeypatch):
    mod = _load_mod()
    secret = "gemini-probe-only-value"
    requests = []
    payloads = [
        {"models": [{"name": "models/gemini-3.1-flash-lite"}]},
        {
            "candidates": [{
                "content": {"parts": [{
                    "text": json.dumps({
                        "image_received": True,
                        "left_half": "red",
                        "right_half": "blue",
                    }),
                }]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 17, "candidatesTokenCount": 8},
        },
    ]

    class Response:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.value).encode("utf-8")

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return Response(payloads[len(requests) - 1])

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    result = mod.test_gemini_connection(secret, "gemini-3.1-flash-lite", timeout_s=30)
    assert result["provider"] == "gemini"
    assert result["probe_model"] == "gemini-3.1-flash-lite"
    assert result["vision_probe"] == "strict_json_passed"
    assert result["prompt_tokens"] == 17
    assert secret not in json.dumps(result)
    assert len(requests) == 2
    headers = {key.casefold(): value for key, value in requests[1][0].headers.items()}
    assert headers["x-goog-api-key"] == secret
    body = json.loads(requests[1][0].data.decode("utf-8"))
    assert body["systemInstruction"]["parts"][0]["text"]
    assert body["contents"][-1]["role"] == "user"
    assert body["contents"][-1]["parts"][1]["inlineData"]["mimeType"] == "image/png"
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["responseJsonSchema"]["required"] == [
        "image_received", "left_half", "right_half",
    ]


def test_frontend_key_widget_is_non_serialized_and_does_not_use_browser_storage():
    source = (ROOT / "web" / "minimax_h3_zh.js").read_text(encoding="utf-8")
    assert "installCloudCredentialButton" in source
    assert 'serializeValue = () => undefined' in source
    assert 'title: "MiniMax H3 云端多模态导演"' in source
    assert "参考图 / 视频如何发送" in source
    assert "installDirectorMediaReferenceButton" in source
    assert "📎 插入 / 检查素材引用" in source
    assert "按实际连线密集编号" in source
    assert "插入全部未引用素材" in source
    assert "DeepSeek 600 帧" in source
    assert "Gemini 原生视频" in source
    assert "管理云端连接" in source
    assert "联合上传（参考图 + 所选视频帧同次发送）" in source
    assert 'gemini: {' in source
    assert 'credentialId: "gemini_default"' in source
    assert "小图 + 严格 JSON 真实视觉探针" in source
    assert "刷新 / 选择云端模型" in source
    assert "/minimaxh3lab/cloud/models" in source
    assert "/minimaxh3lab/cloud/presets" in source
    assert "createCloudPresetManager" in source
    assert "normalizeCloudDirectorWidgetValues" in source
    assert "const CLOUD_WIDGET_SCHEMA_VERSION = 5" in source
    assert '"gemini_video_route", "gemini_video_fps", "reference_fidelity"' in source
    assert 'reference_fidelity: "auto"' in source
    assert "锁定跟随（动作 / 姿势 / 镜头 / 时点）" in source
    assert 'byName.cloud_provider === "my_presets"' in source
    assert "使用已保存连接" in source
    assert "仅临时用于当前节点" in source
    assert 'preset_name: config.presetName || ""' in source
    assert 'payload.vision_probe === "strict_json_passed"' in source
    assert 'base_url: config.baseUrl || ""' in source
    assert "恢复连接预设默认" in source
    cloud_section = source[source.index("const CLOUD_CREDENTIAL_PROVIDERS"):]
    assert "localStorage" not in cloud_section
    assert "sessionStorage" not in cloud_section


# ============================================================================
# 2026-08-28：新增常用云端 LLM provider（豆包/Claude/GLM/MiniMax/custom）
# ============================================================================

def test_new_providers_registered_with_credentials():
    mod = _load_mod()
    for p, env, cred, base in (
        ("volcengine", "ARK_API_KEY", "volcengine_default", "https://ark.cn-beijing.volces.com/api/v3"),
        ("anthropic", "ANTHROPIC_API_KEY", "anthropic_default", "https://api.anthropic.com/v1"),
        ("zhipu", "ZHIPU_API_KEY", "zhipu_default", "https://open.bigmodel.cn/api/paas/v4"),
        ("minimax", "MINIMAX_API_KEY", "minimax_default", "https://api.minimaxi.com/v1"),
        ("custom", "OPENAI_API_KEY", "custom_default", ""),
    ):
        d = mod.provider_definition(p)
        assert d["environment_variable"] == env
        assert d["default_credential_id"] == cred
        assert d["base_url"] == base


def test_static_model_list_for_no_discovery_providers():
    mod = _load_mod()
    # 无标准 GET /models 的 provider 返回静态候选（不联网）。
    vol = mod.list_cloud_models("volcengine", "")
    assert any(m["id"] == "doubao-seed-evolving" for m in vol)
    assert any(m["id"] == "MiniMax-M2.7" for m in mod.list_cloud_models("minimax", ""))
    # custom 会按用户 Base URL 尝试标准 GET /models；未填地址时明确阻断。
    with pytest.raises(mod.CloudCredentialError, match="Base URL"):
        mod.list_cloud_models("custom", "")


def test_custom_openai_model_discovery_uses_base_url_and_optional_auth(monkeypatch):
    mod = _load_mod()
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "data": [
                    {"id": "qwen3.8-27b", "display_name": "Qwen 3.8 27B"},
                    {"id": "vision-b"},
                ],
            }).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {k.casefold(): v for k, v in request.headers.items()}
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    rows = mod.list_cloud_models(
        "custom", "unit-secret", 30, base_url="http://127.0.0.1:1234/v1",
    )
    assert captured["url"] == "http://127.0.0.1:1234/v1/models"
    assert captured["headers"]["authorization"] == "Bearer unit-secret"
    assert [row["id"] for row in rows] == ["qwen3.8-27b", "vision-b"]
    assert "unit-secret" not in json.dumps(rows)


def test_new_provider_probe_functions_exist():
    mod = _load_mod()
    for name in (
        "test_zhipu_connection", "test_volcengine_connection",
        "test_anthropic_connection", "test_minimax_connection", "test_custom_connection",
    ):
        assert callable(getattr(mod, name, None)), name
    # 无 key 的探针必须明确抛错，不冒充通过。
    for name, args in (
        ("test_zhipu_connection", ()),
        ("test_volcengine_connection", ()),
        ("test_minimax_connection", ()),
        ("test_anthropic_connection", ()),
    ):
        with pytest.raises(mod.CloudCredentialError):
            getattr(mod, name)("", *args)


def test_custom_probe_requires_base_url_and_model():
    mod = _load_mod()
    # base_url / model 缺失都抛 CloudCredentialError（需用户填，不联网）。
    with pytest.raises(mod.CloudCredentialError):
        mod.test_custom_connection("k", base_url="", model_id="")
    with pytest.raises(mod.CloudCredentialError):
        mod.test_custom_connection("k", base_url="https://x.com/v1", model_id="")


def test_custom_loopback_probe_can_run_without_api_key(monkeypatch):
    mod = _load_mod()
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"image_received\\":true,\\"left_half\\":\\"red\\",\\"right_half\\":\\"blue\\"}"}}],"usage":{}}'

    def fake_urlopen(request, timeout):
        captured["headers"] = {k.casefold(): v for k, v in request.headers.items()}
        captured["url"] = request.full_url
        return Response()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    result = mod.test_custom_connection(
        "", base_url="http://127.0.0.1:1234/v1", model_id="local-vision",
    )
    assert result["vision_probe"] == "strict_json_passed"
    assert captured["url"].endswith("/v1/chat/completions")
    assert "authorization" not in captured["headers"]


# ============================================================================
# 「我的预设」持久化（2026-08-28）：save/list/get/delete，不含 key
# ============================================================================

def test_custom_preset_save_list_get_delete(tmp_path):
    mod = _load_mod()
    mod.save_custom_preset("我的GLM", "https://open.bigmodel.cn/api/paas/v4", "glm-5.3-flash", user_dir=tmp_path)
    presets = mod.list_custom_presets(user_dir=tmp_path)
    assert any(p["name"] == "我的GLM" and p["model"] == "glm-5.3-flash" for p in presets)
    g = mod.get_custom_preset("我的GLM", user_dir=tmp_path)
    assert g["base_url"] == "https://open.bigmodel.cn/api/paas/v4"
    assert g["model"] == "glm-5.3-flash"
    assert g["credential_id"].startswith("custom_")
    mod.delete_custom_preset("我的GLM", user_dir=tmp_path)
    assert mod.list_custom_presets(user_dir=tmp_path) == []


def test_custom_preset_rejects_invalid_payload(tmp_path):
    mod = _load_mod()
    with pytest.raises(mod.CloudCredentialError):
        mod.save_custom_preset("", "https://x.com/v1", "m", user_dir=tmp_path)  # 空名
    with pytest.raises(mod.CloudCredentialError):
        mod.save_custom_preset("名", "", "m", user_dir=tmp_path)  # 空 base_url
    with pytest.raises(mod.CloudCredentialError):
        mod.save_custom_preset("名", "ftp://x", "m", user_dir=tmp_path)  # 非 http
    with pytest.raises(mod.CloudCredentialError):
        mod.save_custom_preset("名", "https://x.com/v1", "", user_dir=tmp_path)  # 空 model


def test_custom_preset_does_not_store_key(tmp_path):
    mod = _load_mod()
    mod.save_custom_preset("GLM", "https://open.bigmodel.cn/api/paas/v4", "glm-5.3-flash", user_dir=tmp_path)
    # 任何 list/get 都不含 key 字段
    for p in mod.list_custom_presets(user_dir=tmp_path):
        assert "key" not in p and "api_key" not in p
        assert set(p.keys()) <= {"id", "name", "base_url", "model", "credential_id"}


def test_custom_presets_get_stable_separate_credential_slots(tmp_path):
    mod = _load_mod()
    first = mod.save_custom_preset("代理一", "https://one.example/v1", "vision-a", user_dir=tmp_path)
    second = mod.save_custom_preset("代理二", "https://two.example/v1", "vision-b", user_dir=tmp_path)
    assert first["credential_id"] != second["credential_id"]
    updated = mod.save_custom_preset("代理一", "https://new.example/v1", "vision-c", user_dir=tmp_path)
    assert updated["credential_id"] == first["credential_id"]
    stored = json.loads(mod._presets_file_path(tmp_path).read_text(encoding="utf-8"))
    assert "api_key" not in json.dumps(stored)


def test_anthropic_probe_requires_real_visual_json(monkeypatch):
    mod = _load_mod()
    secret = "anthropic-probe-secret"
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "content": [{
                    "type": "text",
                    "text": '{"image_received":true,"left_half":"red","right_half":"blue"}',
                }],
                "usage": {"input_tokens": 19, "output_tokens": 8},
            }).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["headers"] = {key.casefold(): value for key, value in request.headers.items()}
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    result = mod.test_anthropic_connection(secret, "claude-sonnet-4-6", 30)
    assert result["vision_probe"] == "strict_json_passed"
    assert captured["headers"]["x-api-key"] == secret
    assert captured["payload"]["messages"][0]["content"][1]["type"] == "image"
    assert secret not in json.dumps(result)


def test_anthropic_probe_rejects_incorrect_visual_answer(monkeypatch):
    mod = _load_mod()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "content": [{
                    "type": "text",
                    "text": '{"image_received":true,"left_half":"blue","right_half":"red"}',
                }],
            }).encode("utf-8")

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda request, timeout: Response())
    with pytest.raises(mod.CloudCredentialError, match="没有通过"):
        mod.test_anthropic_connection("key", "claude-sonnet-4-6", 30)
