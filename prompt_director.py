# -*- coding: utf-8 -*-
"""MiniMaxH3PromptDirector — H3 顺序接口提示词导演节点（独立实现）。

职责边界（详见《H3顺序接口提示词节点设计方案-2026-08-05.md》）：
- 文字 + 参考图（≤9）：走 OpenAI 兼容多模态 API，生成 H3 三段式提示词
- 参考视频/音频：不传 API，用户直接接官方 ReferenceToVideo 端口（本地编码器消费）
- 固定 9 个 ref_image 顺序接口（1-based，与 <Picture i> 一一对应，规避 Autogrow 乱序）

依赖：仅标准库（urllib/json/base64）+ torch/numpy/PIL（ComfyUI 自带），零第三方包。
API Key 读取顺序：MINIMAX_H3_API_KEY > LINGBOT_API_KEY > OPENAI_API_KEY > 节点 api_key > 空串
（空 key 不拦截：本地 OpenAI 兼容服务如 LM Studio 忽略鉴权，与 LingBot 行为一致）。
"""
import base64
import io
import json
import logging
import os
import time
import urllib.error
import urllib.request

import numpy as np
import torch
from PIL import Image

LOGGER = logging.getLogger(__name__)

_ENV_API_KEY = "MINIMAX_H3_API_KEY"

_TASK_RULES = {
    "T2VA": "纯文字生视频：无参考图，直接按提示词创造画面。",
    "I2VA": "参考图驱动：参考图是素材，可重新构图；用 <Picture i> 引用并描述其角色/场景/元素如何参与。",
    "FL2VA": "首帧锚定：首帧是严格锚定（拉伸），提示词只描述运动与延续，不得要求改变画面构图。",
    "Ref2VA": "参考素材：参考图是角色与场景参考，允许重新构图；用 <Picture i> 引用并明确'作为角色与场景参考'。",
}

_REWRITE_MODES = {
    "strict": "严格遵循用户原始意图，只做结构化和镜头分配，不新增内容。",
    "balanced": "平衡：结构化 + 适度补全细节，保持用户核心意图。",
    "creative": "创意：在用户意图基础上自由发挥，补充生动的镜头语言与细节。",
}

_H3_SYSTEM_TEMPLATE = """你是 MiniMax H3 视频生成模型的提示词导演。你的任务是把用户的意图转成 H3 规范的三段式 JSON 提示词。

输出必须是合法 JSON，结构如下：
{{
  "integrated_multimodal_description": "画面与动作的整体描述，必须使用 <Picture 1>..<Picture 9> 标签引用参考图",
  "overall_soundscape": "环境声/音效描述（diegetic 声音，与画面同步）",
  "non_diegetic_music": "配乐描述（非剧情内音乐，含风格/节奏/情绪）",
  "shot_breakdown": [{{"start_s": 0.0, "end_s": 2.5, "description": "分镜描述"}}]
}}

规则：
1. 总时长约 {duration}s，分 {shots} 个镜头；shot_breakdown 的区间必须连续覆盖全片。
2. {task_rule}
3. 改写模式：{rewrite_mode_rule}
4. 所有文本使用{language}输出。
5. 只输出 JSON，不要输出任何额外说明。"""


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


def _compose_enhanced(parsed) -> str:
    """三段式 JSON → H3 提示词文本（拼接；缺字段/非 dict 时安全降级）。"""
    if not isinstance(parsed, dict):
        return ""
    parts = []
    desc = (parsed.get("integrated_multimodal_description") or "").strip()
    sound = (parsed.get("overall_soundscape") or "").strip()
    music = (parsed.get("non_diegetic_music") or "").strip()
    if desc:
        parts.append(desc)
    if sound:
        parts.append("环境声：" + sound)
    if music:
        parts.append("配乐：" + music)
    shots = parsed.get("shot_breakdown")
    if isinstance(shots, list) and shots:
        lines = []
        for s in shots:
            if isinstance(s, dict) and s.get("description"):
                start = s.get("start_s") or 0.0
                end = s.get("end_s") or 0.0
                lines.append(f"[{start}-{end}s] {s['description']}")
        if lines:
            parts.append("分镜：" + " ".join(lines))
    return "\n".join(parts) if parts else ""


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
                "output_language": (["中文", "English"], {"default": "中文"}),
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
            },
            "optional": {
                **{f"ref_image_{i}": ("IMAGE",) for i in range(1, 10)},
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "report")
    FUNCTION = "direct"
    CATEGORY = "MiniMax H3 Lab/Prompt"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # api_model 的 COMBO 值可能来自旧工作流（旧 STRING 值如 'gemma4@q6_k'
        # 不在刷新后的列表里）——放行，执行时原样使用。
        return True

    def direct(self, prompt, task_type, duration_seconds, shot_count, rewrite_mode,
               output_language, api_base_url, api_model, api_key, temperature,
               max_tokens, timeout_s, lmstudio_after_use="keep_loaded",
               lmstudio_gpu_offload="auto", api_reasoning="auto", json_mode="auto_retry", **kwargs):
        t0 = time.time()
        lm_notes = []

        def passthrough(reason: str):
            """直通统一出口：日志 + report，方便用户在控制台看到跳过原因。"""
            LOGGER.warning("MiniMax H3 PromptDirector: %s——已直通原始提示词", reason)
            return (prompt, f"[直通] {reason}\n已直通原始提示词（未调用 API）。")
        # 收集参考图（固定顺序 1..9，跳过未连接端口）
        images = []
        for i in range(1, 10):
            img = kwargs.get(f"ref_image_{i}")
            if img is not None:
                images.append((i, _image_tensor_to_data_url(img)))

        # API Key：环境变量优先（与 LingBot 一致支持 OPENAI_API_KEY），节点内次之。
        # 空 key 不拦截：本地 OpenAI 兼容服务（LM Studio 等）忽略 Authorization，
        # 远程服务无 key 时由 401 走 API 失败直通分支。
        key = (
            os.environ.get(_ENV_API_KEY)
            or os.environ.get("LINGBOT_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or api_key
            or ""
        )
        if not key:
            LOGGER.info("MiniMax H3 PromptDirector: 未提供 API Key（按本地服务处理，LM Studio 等忽略鉴权）")

        shots = shot_count if shot_count and shot_count > 0 else max(1, round(duration_seconds / 2.5))
        system = _H3_SYSTEM_TEMPLATE.format(
            duration=duration_seconds,
            shots=shots,
            task_rule=_TASK_RULES[task_type],
            rewrite_mode_rule=_REWRITE_MODES[rewrite_mode],
            language=output_language,
        )

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

        # 消息构建：文本 + 多图（content 数组，OpenAI 多模态格式）
        user_content = [{"type": "text", "text": prompt}]
        for idx, data_url in images:
            user_content.append({"type": "text", "text": f"参考图 {idx}（对应 <Picture {idx}>）："})
            user_content.append({"type": "image_url", "image_url": {"url": data_url}})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        # LM Studio 显式 GPU 放置（调用前，非 auto 时）
        lm_root = _lmstudio_root(api_base_url)
        if lmstudio_gpu_offload != "auto" and lm_root:
            lm_notes.append(_lmstudio_load_gpu(lm_root, model, key, timeout_s, lmstudio_gpu_offload))

        try:
            raw = _call_chat(api_base_url, key, model, messages, temperature, max_tokens, timeout_s,
                             json_mode=json_mode, api_reasoning=api_reasoning)
            parsed = _parse_json_text(raw)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("MiniMax H3 PromptDirector: API/解析失败: %s", exc)
            return passthrough(f"API/解析失败: {exc}")

        # LM Studio 跑完卸载（调用后，非 keep_loaded 时）
        if lmstudio_after_use != "keep_loaded" and lm_root:
            try:
                _lmstudio_unload(lm_root, model, key, timeout_s)
                lm_notes.append(f"LM Studio 已卸载：{model}")
            except Exception as exc:  # noqa: BLE001
                lm_notes.append(f"LM Studio 卸载失败：{exc}")
            if lmstudio_after_use == "unload_and_wait_for_vram":
                lm_notes.append(_wait_for_free_vram(2.0, timeout_s))

        enhanced = _compose_enhanced(parsed) or prompt
        shots_out = len(parsed.get("shot_breakdown", [])) if isinstance(parsed, dict) else 0
        report = "\n".join([
            f"task_type={task_type} 模型={model}",
            f"参考图={len(images)} 张（端口 {[i for i, _ in images]}）",
            f"分镜={shots_out} 段  API 耗时={time.time() - t0:.1f}s",
            *lm_notes,
            "提示：参考视频/音频请直接接官方 ReferenceToVideo 的 ref_video/ref_audio 端口（本节点不处理）。",
        ])
        return (enhanced, report)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PromptDirector": MiniMaxH3PromptDirector,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PromptDirector": "MiniMax H3 提示词导演 (顺序接口)",
}
