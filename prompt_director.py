# -*- coding: utf-8 -*-
"""MiniMaxH3PromptDirector — H3 顺序接口提示词导演节点（独立实现）。

职责边界（详见《H3顺序接口提示词节点设计方案-2026-08-05.md》）：
- 文字 + 参考图（≤9）：走 OpenAI 兼容多模态 API，生成 H3 三段式提示词
- 参考视频/音频：不传 API，用户直接接官方 ReferenceToVideo 端口（本地编码器消费）
- 固定 9 个 ref_image 物理接口；已连接素材再密集编号为 <Picture 1..N>，规避空端口与 Autogrow 乱序

依赖：仅标准库（urllib/json/base64）+ torch/numpy/PIL（ComfyUI 自带），零第三方包。
API Key 读取顺序：节点 api_key > MINIMAX_H3_API_KEY > 与目标主机严格匹配的专用环境变量 > 空串。
（空 key 不拦截：本地 OpenAI 兼容服务如 LM Studio 通常忽略鉴权。）
"""
import base64
import io
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import torch
from PIL import Image

# v0.1：协议加载（官方 skill 浓缩，按 task_type 自动注入）与确定性校验器
# ComfyUI 包内走相对导入；单测（importlib 直载）走 fallback 加载
_PROMPT_MODULES = None
_H3_COMPILER = None


def _load_prompt_modules():
    global _PROMPT_MODULES
    if _PROMPT_MODULES is None:
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "prompt_modules",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_modules.py"),
        )
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules["prompt_modules"] = _mod
        _spec.loader.exec_module(_mod)
        _PROMPT_MODULES = _mod
    return _PROMPT_MODULES


def _load_h3_compiler():
    global _H3_COMPILER
    if _H3_COMPILER is None:
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "h3_compiler",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "h3_compiler.py"),
        )
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules["h3_compiler"] = _mod
        _spec.loader.exec_module(_mod)
        _H3_COMPILER = _mod
    return _H3_COMPILER

LOGGER = logging.getLogger(__name__)

_ENV_API_KEY = "MINIMAX_H3_API_KEY"

_TASK_RULES = {
    "T2VA": "纯文字生视频：无参考图，直接按提示词创造画面。",
    "I2VA": "首帧图生视频：<Picture 1> 是 0.00 秒首帧；保持身份、构图和场景锚点，再描述连续发展。",
    "FL2VA": "首尾帧生视频：<Picture 1> 是首帧，<Picture 2> 是尾帧；描述两者之间可观察、连续的变化路径。",
    "L2VA": "尾帧图生视频：<Picture 1> 是最终帧；推断合理前态并逐渐收束到该尾帧。",
    "Ref2VA": "参考素材：参考图是角色与场景参考，允许重新构图；用 <Picture i> 引用并明确'作为角色与场景参考'。",
}

_REWRITE_MODES = {
    "strict": "严格遵循用户原始意图，只做结构化和镜头分配，不新增内容。",
    "balanced": "平衡：结构化 + 适度补全细节，保持用户核心意图。",
    "creative": "创意：在用户意图基础上自由发挥，补充生动的镜头语言与细节。",
}

# v0.2：素材角色选项（默认 auto；用户指定优先于自动判断，见规格书）
_ASSET_ROLES = [
    "auto", "人物身份", "物体身份", "场景参考", "风格参考", "首帧", "尾帧", "构图锚点",
]

# v0.2：两阶段视觉分析——阶段 1 逐素材事实抽取（只提取事实，不写视频提示词）
_REF_SHEET_SYSTEM_TEMPLATE = """你是参考素材分析器。逐图提取事实，不创作剧情、不写视频提示词。
对每张图输出 JSON，字段：
{{
  "asset_id": "asset_N",
  "role_hint": "auto/人物身份/物体身份/场景参考/风格参考/首帧/尾帧/构图锚点",
  "appearance": "主体外观事实（发型/脸型/服装/配饰/比例），未知写 unknown",
  "pose_and_view": "姿势/视角/景别事实",
  "scene": "场景/光线/关键地标，无场景内容写 none",
  "conflicts_risk": "与同批其他素材可能的冲突风险，未知写 unknown",
  "confidence": 0.0
}}
只输出 JSON，不要任何额外说明。"""

_H3_SYSTEM_TEMPLATE = """你是 MiniMax H3 视频生成模型的提示词导演。你的任务是把用户意图规划成结构化 Prompt IR；最终字段顺序与格式由本地 Python 编译器负责。

输出必须是合法 JSON，且只能包含下面契约中的字段：
{output_contract}

规则：
1. 总时长约 {duration}s，规划约 {shots} 个镜头；首镜使用 [Shot 1] 且不写时间戳，后续镜头使用 [Shot N] At MM:SS.mmm，时间严格递增且不超过总时长。
2. {task_rule}
3. 改写模式：{rewrite_mode_rule}
4. 字段名固定使用契约中的英文；描述正文使用{language}，用户原始对白、歌词和画面可见文字不得改写或翻译。
5. 不要额外输出 Markdown、解释、字段外文本或重复的分镜数组。"""


def _output_contract(task_type: str) -> str:
    if str(task_type).upper() == "REF2VA":
        return """{
  "subject_definitions": "定义 <Subject N>/<Picture N>/<Video N>/<Audio N>",
  "summary": "以 [任务类型] 开头的目标与素材关系摘要",
  "retention_analysis": "逐标签说明保留、迁移或参考方式",
  "detailed_description": "按播放顺序编排的 [Shot N] 详细描述",
  "overall_soundscape": "环境声、动作声和非语言人声",
  "non_diegetic_music": "仅观众可听的配乐；无配乐写 N/A"
}"""
    return """{
  "integrated_multimodal_description": "按时间线编排的 [Shot N] 画面、动作、镜头、对白与剧情内声音",
  "overall_soundscape": "环境声、动作声和非语言人声",
  "non_diegetic_music": "仅观众可听的配乐；无配乐写 N/A"
}"""


def _image_tensor_to_data_url(image_tensor: torch.Tensor, max_side: int = 1024) -> str:
    """IMAGE 张量 → JPEG data URL（取第一帧，限制最长边，供多模态 API 使用）。"""
    t = image_tensor.detach().cpu()
    while t.ndim > 3:
        t = t[0]
    arr = (t.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    if max(pil.size) > max_side:
        pil.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _chat_completions_url(base_url: str) -> str:
    """把 base_url 规范化为 /chat/completions 端点（兼容裸根/带 /v1//openai）。"""
    url = base_url.rstrip("/")
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"api_base_url 仅支持 http/https: {base_url!r}")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1") or url.endswith("/openai"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


def _models_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"api_base_url 仅支持 http/https: {base_url!r}")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    if url.endswith("/v1") or url.endswith("/openai"):
        return url + "/models"
    return url + "/v1/models"


def _reject_link_local_target(base_url: str) -> None:
    """拒绝链路本地/多播目标（169.254.0.0/16 云元数据等）；localhost/127.0.0.1 允许。

    SSRF 务实加固：本地 LM Studio（127.0.0.1）是核心用例必须放行；封掉
    链路本地与多播即可挡住云元数据探测（内网探测威胁在无鉴权 ComfyUI 前
    不构成新增权限，README 已注明）。
    """
    from urllib.parse import urlparse

    try:
        import ipaddress
        import socket
    except ImportError:
        return
    host = (urlparse(base_url).hostname or "").strip()
    if not host or host.lower() in ("localhost", "localhost.localdomain"):
        return
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"无法解析 API 主机 {host!r}: {exc}") from exc
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if isinstance(ip, ipaddress.IPv6Address):
            mapped = ip.ipv4_mapped  # ::ffff:x.x.x.x 绕过防护（is_ipv4_mapped 仅 3.13+，此处兼容 3.11）
            if mapped is not None:
                ip = mapped
        if ip.is_link_local or ip.is_multicast or ip.is_unspecified:
            raise ValueError(f"API 目标 {host} 为链路本地/多播地址，已拒绝（云元数据防护）")


def _list_models(base_url: str, api_key: str, timeout_s: int = 15) -> list[str]:
    """拉取 OpenAI 兼容 /models 列表（auto 选模型与 report 用）。失败返回空列表。"""
    try:
        _reject_link_local_target(base_url)
        req = urllib.request.Request(_models_url(base_url), headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m["id"] for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("MiniMax H3 PromptDirector: 拉取模型列表失败: %s", exc)
        return []


def _call_chat(base_url, api_key, model, messages, temperature, max_tokens, timeout_s,
               json_mode="auto_retry", api_reasoning="auto") -> str:
    """调用 OpenAI 兼容 chat/completions，返回 assistant 消息文本。

    - json_mode: off=不带 response_format；force=带 json_object；auto_retry=带
      json_object，遇 400/422 降级为无结构重试一次（服务器不支持时）
    - api_reasoning: on/off 通过 chat_template_kwargs.enable_thinking 控制思考
      （Qwen 系 reasoning 模型）；auto=不设置（服务器默认）
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode != "off":
        payload["response_format"] = {"type": "json_object"}
    if api_reasoning == "off" or (
        api_reasoning == "auto"
        and any(token in model.casefold() for token in ("qwen3.6", "qwen3-6", "qwen3_6"))
    ):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    elif api_reasoning == "on":
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    _reject_link_local_target(base_url)
    req = urllib.request.Request(
        _chat_completions_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if json_mode == "auto_retry" and exc.code in (400, 422) and "response_format" in payload:
            # 服务器拒绝结构化输出 → 降级为无结构重试（系统提示词仍要求 JSON）
            payload.pop("response_format", None)
            req = urllib.request.Request(
                _chat_completions_url(base_url),
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        else:
            raise
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"API 响应缺少 choices[0].message.content: {str(data)[:200]}") from exc


def _parse_json_text(raw: str):
    """容错解析 LLM 输出：先整体 json.loads，失败则提取首个 { } 块。"""
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"模型输出不是合法 JSON: {raw[:200]}")


def _lmstudio_root(base_url: str) -> str:
    """从 OpenAI 兼容 base_url 提取 LM Studio REST 根（http://host:port）。"""
    from urllib.parse import urlparse

    p = urlparse(base_url)
    if p.scheme.lower() not in ("http", "https"):
        return ""
    return f"{p.scheme}://{p.netloc}"


def _lmstudio_unload(root: str, model: str, api_key: str, timeout_s: int) -> dict:
    """REST POST /api/v1/models/unload；model_not_found/not loaded 视为已卸载。"""
    _reject_link_local_target(root)  # 与 chat/models 出站同级的 SSRF 防护
    req = urllib.request.Request(
        root.rstrip("/") + "/api/v1/models/unload",
        data=json.dumps({"instance_id": model}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=min(max(2, int(timeout_s)), 30)) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace").casefold()
        if "model_not_found" in body or "not loaded" in body:
            return {}
        raise


def _lmstudio_load_gpu(root: str, model: str, api_key: str, timeout_s: int, mode: str) -> str:
    """显式加载 LM Studio 模型并指定 GPU 放置（lms CLI，稳定跨 REST schema）。

    先卸载已有实例（保证放置确定性），再用 `lms load --gpu <mode>` 加载。
    未找到 lms CLI 时降级为自动放置并返回提示（不阻塞调用）。
    """
    import subprocess

    mode = (mode or "max").strip().casefold()
    if mode not in {"max", "0.90", "0.75", "0.50", "auto", "off"}:
        mode = "max"  # 白名单校验（防御纵深：COMBO 值可被 API 覆盖）
    if mode == "auto":
        return "LM Studio 加载策略：自动（由 LM Studio 决定 GPU/CPU 分配）"
    try:
        _lmstudio_unload(root, model, api_key, min(max(2, int(timeout_s)), 15))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("MiniMax H3 PromptDirector: 预卸载失败（继续）: %s", exc)

    exe = None
    lms_bin = os.path.join(os.path.expanduser("~"), ".lmstudio", "bin",
                           "lms.exe" if os.name == "nt" else "lms")
    if os.path.isfile(lms_bin):
        exe = lms_bin
    elif os.name != "nt":
        import shutil

        exe = shutil.which("lms")
    if not exe:
        return "未找到 lms CLI（~/.lmstudio/bin），跳过显式 GPU 加载（LM Studio 自动决定）"
    try:
        completed = subprocess.run(
            [exe, "load", model, "--gpu", mode, "--identifier", model, "-y"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=max(30, int(timeout_s)), check=False,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
    except Exception as exc:  # noqa: BLE001
        return f"lms CLI 调用失败：{exc}"
    if completed.returncode != 0:
        return f"lms load 失败（GPU={mode}，exit={completed.returncode}）：{(completed.stderr or '').strip()[:200]}"
    return f"LM Studio 已显式加载：GPU={mode}"


def _wait_for_free_vram(target_gb: float, timeout_s: int) -> str:
    """nvidia-smi 轮询空闲显存（unload_and_wait_for_vram 用）。"""
    import subprocess
    import time

    deadline = time.time() + max(5, int(timeout_s))
    while time.time() < deadline:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=False,
            ).stdout
            free = int(out.strip().splitlines()[0])
            if free >= target_gb * 1024:
                return f"空闲显存 {free} MiB >= 目标 {target_gb} GB"
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    return f"等待超时：空闲显存未达 {target_gb} GB"


def _compose_enhanced(parsed, task_type="T2VA", duration_seconds=5.0) -> str:
    """LLM JSON → 由 h3_compiler 确定性序列化的官方 H3 提示词。"""
    if not isinstance(parsed, dict):
        return ""
    return _load_h3_compiler().compile_prompt_ir(
        parsed, task_type=task_type, duration=duration_seconds,
    )["prompt"]


def _analyze_assets(base_url, key, model, images, temperature,
                    max_tokens, timeout_s, json_mode, api_reasoning):
    """v0.2 两阶段视觉分析·阶段 1：逐素材事实抽取（每图一次短调用）。

    返回 (sheet_items, notes)：sheet_items 为每素材 JSON 摘要（保持输入顺序）；
    notes 为过程说明。任何一张图失败 → 返回 ([], [失败原因])（调用方回退 single_pass）。
    """
    notes = []
    sheet_items = []
    per_asset_tokens = max(512, min(int(max_tokens or 2048), 2048))
    for idx, data_url in images:
        user_content = [
            {"type": "text", "text": f"分析这张参考素材 asset_{idx}（角色：auto，请自行判断）："},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        try:
            item = _parse_json_text(_call_chat(
                base_url, key, model,
                [{"role": "system", "content": _REF_SHEET_SYSTEM_TEMPLATE},
                 {"role": "user", "content": user_content}],
                temperature, per_asset_tokens, timeout_s,
                json_mode=json_mode, api_reasoning=api_reasoning,
            ))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("MiniMax H3 PromptDirector: 素材 %d 分析失败——回退单次多图: %s", idx, exc)
            return [], [f"素材 {idx} 分析失败，回退单次多图：{exc}"]
        if not isinstance(item, dict):
            return [], [f"素材 {idx} 分析输出非对象，回退单次多图"]
        item["asset_id"] = f"asset_{idx}"
        item["input_index"] = idx
        sheet_items.append(item)
    notes.append(f"逐素材分析 {len(images)} 张（每张 ≤{per_asset_tokens} token）")
    return sheet_items, notes


class MiniMaxH3PromptDirector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "描述你希望生成的视频内容"}),
                "task_type": (list(_TASK_RULES), {"default": "I2VA"}),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 4.0, "max": 15.0, "step": 0.5}),
                "shot_count": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1}),
                "rewrite_mode": (list(_REWRITE_MODES), {"default": "balanced"}),
                "output_language": (["English", "中文"], {"default": "English"}),
                "api_base_url": ("STRING", {"default": "http://127.0.0.1:1234/v1"}),
                "api_model": (["auto"], {"default": "auto"}),
                "api_key": ("STRING", {"default": "", "password": True}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 128}),
                "timeout_s": ("INT", {"default": 180, "min": 10, "max": 1800, "step": 10}),
                "lmstudio_after_use": (["keep_loaded", "unload_used_model", "unload_and_wait_for_vram"], {"default": "keep_loaded"}),
                "lmstudio_gpu_offload": (["max", "0.90", "0.75", "0.50", "auto", "off"], {"default": "auto"}),
                # 新 widget 一律追加在末尾（ComfyUI 按位置恢复旧工作流值）
                "api_reasoning": (["auto", "off", "on"], {"default": "auto"}),
                "json_mode": (["auto_retry", "force", "off"], {"default": "auto_retry"}),
                # v0.2：视觉分析模式（两阶段=逐素材事实抽取+合并；单次=原多图直传）
                # v0.1：Auto=0-2 图单次、3-9 图分阶段；旧值 two_stage/single_pass 兼容映射
                "analysis_mode": (["auto", "single", "staged"], {"default": "auto"}),
            },
            "optional": {
                **{
                    f"ref_image_{i}": ("IMAGE", {"label": f"导演识图素材 {i}（不传给 H3）"})
                    for i in range(1, 10)
                },
                # v0.2：可接提示词模块节点输出（可选，非空时并入 system）
                "system_module": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                # 模块节点第 4 输出：只用于诊断/IR 追踪，不重复发送给 LLM
                "module_manifest": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "report", "reference_sheet", "prompt_ir")
    FUNCTION = "direct"
    CATEGORY = "MiniMax H3 Lab/Prompt"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # api_model 的 COMBO 值可能来自旧工作流（旧 STRING 值如 'gemma4@q6_k'
        # 不在刷新后的列表里）——放行，执行时原样使用。
        # v0.1：analysis_mode 旧值 two_stage/single_pass 也放行（映射为 staged/single）
        if kwargs.get("analysis_mode") in ("two_stage", "single_pass", "auto", "single", "staged"):
            return True
        return True

    def direct(self, prompt, task_type, duration_seconds, shot_count, rewrite_mode,
               output_language, api_base_url, api_model, api_key, temperature,
               max_tokens, timeout_s, lmstudio_after_use="keep_loaded",
               lmstudio_gpu_offload="auto", api_reasoning="auto", json_mode="auto_retry", **kwargs):
        t0 = time.time()
        lm_notes = []
        module_notes = []
        module_ids = []

        manifest_text = str(kwargs.get("module_manifest") or "").strip()
        if manifest_text:
            try:
                manifest = json.loads(manifest_text)
                if not isinstance(manifest, dict):
                    raise ValueError("顶层不是 JSON 对象")
                module_ids = [
                    str(item.get("id")) for item in manifest.get("selected", [])
                    if isinstance(item, dict) and item.get("id")
                ]
                manifest_scope = str(manifest.get("scope") or "全部")
                if manifest_scope != "全部" and manifest_scope.casefold() != str(task_type).casefold():
                    module_notes.append(
                        f"[模块警告] 模块作用域={manifest_scope}，但导演任务={task_type}；请确认下游官方节点类型。"
                    )
                for item in manifest.get("issues", [])[:5]:
                    module_notes.append(f"[模块诊断] {item}")
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                module_notes.append(f"[模块警告] module_manifest 无法解析：{exc}")

        if output_language != "English":
            module_notes.append(
                "[语言警告] 中文输出属于便捷实验模式；官方 H3 执行协议要求英文主体，建议正式生成使用 English。"
            )

        def passthrough(reason: str, api_called: bool = False,
                        reference_sheet: str = "[]", prompt_ir: str = "{}"):
            """直通统一出口：日志 + report，方便用户在控制台看到跳过原因。"""
            LOGGER.warning("MiniMax H3 PromptDirector: %s——已直通原始提示词", reason)
            call_note = "API 已调用但未得到可编译结果" if api_called else "未调用 API"
            report = f"[直通] {reason}\n已直通原始提示词（{call_note}）。"
            if module_notes:
                report += "\n" + "\n".join(module_notes)
            return (prompt, report,
                    reference_sheet, prompt_ir)
        # 收集参考图：按已连接素材密集编号，确保与官方动态端口的展示顺序一致。
        images = []
        port_map = []
        for i in range(1, 10):
            img = kwargs.get(f"ref_image_{i}")
            if img is not None:
                ordinal = len(images) + 1
                images.append((ordinal, _image_tensor_to_data_url(img)))
                port_map.append((i, ordinal))

        if len(images) > 1 and task_type != "Ref2VA":
            module_notes.append(
                f"[接线警告] 导演接入 {len(images)} 张分析图但任务={task_type}。这些图只供 LLM 识图；"
                "若下游使用 MiniMaxH3ReferenceToVideo，应把任务改为 Ref2VA。"
            )

        # v0.2：素材角色/别名参数已按用户反馈移除（自然语言描述即可）
        # v0.1：analysis_mode 解析（auto = 0-2 图 single、3-9 图 staged；旧值兼容映射）
        mode_raw = str(kwargs.get("analysis_mode") or "auto").strip().lower()
        if mode_raw in ("two_stage", "staged"):
            staged = True
        elif mode_raw in ("single_pass", "single"):
            staged = False
        else:  # auto
            staged = len(images) >= 3

        # 显式节点 Key 优先，避免全局 OPENAI_API_KEY 被发往任意兼容端点。
        from urllib.parse import urlparse
        api_host = (urlparse(api_base_url).hostname or "").casefold()
        key = str(api_key or "").strip() or os.environ.get(_ENV_API_KEY, "")
        if not key and "lingbot" in api_host:
            key = os.environ.get("LINGBOT_API_KEY", "")
        if not key and api_host == "api.openai.com":
            key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            LOGGER.info("MiniMax H3 PromptDirector: 未提供 API Key（按本地服务处理，LM Studio 等忽略鉴权）")

        shots = shot_count if shot_count and shot_count > 0 else max(1, round(duration_seconds / 2.5))
        system = _H3_SYSTEM_TEMPLATE.format(
            duration=duration_seconds,
            shots=shots,
            task_rule=_TASK_RULES[task_type],
            rewrite_mode_rule=_REWRITE_MODES[rewrite_mode],
            language=output_language,
            output_contract=_output_contract(task_type),
        )
        # v0.1：协议自动注入（仅英文输出；Ref2VA→六段式，其余→三段式）+ 创作策略模块
        protocol = ""
        if output_language == "English":
            protocol = _load_prompt_modules().load_protocol(task_type)
        system_module = str(kwargs.get("system_module") or "").strip()
        if protocol and system_module:
            system = protocol + "\n\n" + system_module + "\n\n" + system
        elif protocol:
            system = protocol + "\n\n" + system
        elif system_module:
            system = system_module + "\n\n" + system

        # 模型：auto → 拉取列表，唯一候选自动选；多候选取第一个并提示
        model = api_model.strip()
        models = []
        if not model or model == "auto":
            models = _list_models(api_base_url, key, timeout_s=min(timeout_s, 15))
            if len(models) == 1:
                model = models[0]
            elif not models:
                return passthrough(f"无法从 {api_base_url} 获取模型列表，且 api_model=auto（请确认 API 地址/密钥后点击刷新模型列表）")
            else:
                model = models[0]
                LOGGER.warning("MiniMax H3 PromptDirector: 多个候选模型 %s，取第一个 %s", models, model)

        # LM Studio 显式 GPU 放置必须发生在阶段一识图之前，保证整个请求链使用同一策略。
        lm_root = _lmstudio_root(api_base_url)
        if lmstudio_gpu_offload != "auto" and lm_root:
            lm_notes.append(_lmstudio_load_gpu(lm_root, model, key, timeout_s, lmstudio_gpu_offload))

        # 消息构建：文本 + 参考内容（staged：逐素材分析摘要文本；否则多图直传）
        sheet_items, sheet_notes = [], []
        if staged and images:
            sheet_items, sheet_notes = _analyze_assets(
                api_base_url, key, model, images,
                temperature, max_tokens, timeout_s, json_mode, api_reasoning,
            )
        user_content = [{"type": "text", "text": prompt}]
        if sheet_items:
            for item in sheet_items:
                user_content.append({
                    "type": "text",
                    "text": f"素材 {item.get('input_index')} 分析摘要："
                            + json.dumps({k: v for k, v in item.items()
                                          if k != "input_index"},
                                         ensure_ascii=False),
                })
        else:
            for idx, data_url in images:
                user_content.append({"type": "text", "text": f"参考资产 {idx}（对应 <Picture {idx}>）："})
                user_content.append({"type": "image_url", "image_url": {"url": data_url}})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        try:
            raw = _call_chat(api_base_url, key, model, messages, temperature, max_tokens, timeout_s,
                             json_mode=json_mode, api_reasoning=api_reasoning)
            parsed = _parse_json_text(raw)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("MiniMax H3 PromptDirector: API/解析失败: %s", exc)
            if lmstudio_after_use != "keep_loaded" and lm_root:
                try:
                    _lmstudio_unload(lm_root, model, key, timeout_s)
                except Exception as unload_exc:  # noqa: BLE001
                    LOGGER.warning("MiniMax H3 PromptDirector: 失败清理时卸载异常: %s", unload_exc)
            return passthrough(f"API/解析失败: {exc}", api_called=True)

        # LM Studio 跑完卸载（调用后，非 keep_loaded 时）
        if lmstudio_after_use != "keep_loaded" and lm_root:
            try:
                _lmstudio_unload(lm_root, model, key, timeout_s)
                lm_notes.append(f"LM Studio 已卸载：{model}")
            except Exception as exc:  # noqa: BLE001
                lm_notes.append(f"LM Studio 卸载失败：{exc}")
            if lmstudio_after_use == "unload_and_wait_for_vram":
                lm_notes.append(_wait_for_free_vram(2.0, timeout_s))

        try:
            compiled = _load_h3_compiler().compile_prompt_ir(
                parsed, task_type=task_type, duration=duration_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("MiniMax H3 PromptDirector: IR 编译失败: %s", exc)
            return passthrough(f"IR 编译失败: {exc}", api_called=True)
        enhanced = compiled["prompt"]
        if module_ids:
            compiled["ir"]["applied_modules"] = module_ids
        ir_json = json.dumps(compiled["ir"], ensure_ascii=False, indent=2)
        shots_out = compiled["validation"]["checks"]["shot_count"]
        # v0.2：Reference Sheet 输出（两阶段时含逐素材事实 JSON，否则空）
        if sheet_items:
            sheet_out = json.dumps({"assets": sheet_items}, ensure_ascii=False, indent=2)
        else:
            sheet_out = "[]"
        if compiled["validation"]["errors"]:
            short_errors = "；".join(compiled["validation"]["errors"][:5])
            return passthrough(
                f"Prompt IR 未通过确定性校验：{short_errors}", api_called=True,
                reference_sheet=sheet_out, prompt_ir=ir_json,
            )
        report = "\n".join([
            f"task_type={task_type} 模型={model}",
            f"导演识图素材={len(images)} 张（仅发送给提示词 API；端口→标签 {port_map}）",
            f"分析模式={'staged' if sheet_items else 'single'}",
            f"已应用模块={module_ids or '无'}",
            *sheet_notes,
            f"分镜={shots_out} 段  API 耗时={time.time() - t0:.1f}s",
            *lm_notes,
            *module_notes,
            "提示：本节点的图片输入只供提示词 API 识图，不会给 H3 加条件；图片/视频/音频必须另接官方生成节点。",
        ])
        # v0.1：h3_compiler 接线——确定性校验（errors/warnings 进 report，不阻塞输出）
        try:
            vres = compiled["validation"]
            if vres["errors"]:
                report += "\n[校验错误] " + "；".join(vres["errors"][:5])
            if vres["warnings"]:
                report += "\n[校验警告] " + "；".join(vres["warnings"][:5])
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("MiniMax H3 PromptDirector: 校验器异常（不阻塞）: %s", exc)
        return (enhanced, report, sheet_out, ir_json)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PromptDirector": MiniMaxH3PromptDirector,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PromptDirector": "MiniMax H3 提示词导演 (顺序接口)",
}
