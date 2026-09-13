# -*- coding: utf-8 -*-
"""MiniMaxH3PromptDirector — H3 顺序接口提示词导演节点（独立实现）。

职责边界（详见《H3顺序接口提示词节点设计方案-2026-08-05.md》）：
- 文字 + 参考图（≤9）+ 一组参考视频抽帧/图像序列：走 OpenAI 兼容多模态 API，生成 H3 提示词
- 参考视频/音频：不传 API，用户直接接官方 ReferenceToVideo 端口（本地编码器消费）
- 固定 9 个 ref_image 物理接口；已连接素材再密集编号为 <Picture 1..N>，规避空端口与 Autogrow 乱序
- video_frame_sequence 接 IMAGE 批次并均匀限量取帧；这些帧共同作为 <Video 1> 时间线证据，不占 Picture 编号

依赖：仅标准库（urllib/json/base64）+ torch/numpy/PIL（ComfyUI 自带），零第三方包。
通用节点 API Key 读取顺序：节点 api_key > 与目标主机严格匹配的专用环境变量 > MINIMAX_H3_API_KEY > 空串。
CloudDirector 单独使用云端凭据层：供应商环境变量 > ComfyUI user 目录本地凭据文件；Key 不进入节点输入。
（空 key 不拦截：本地 OpenAI 兼容服务如 LM Studio 通常忽略鉴权。）
"""
import base64
import datetime
import hashlib
import http.client
import io
import json
import logging
import os
import random
import re
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime

import numpy as np
import torch
from PIL import Image

# v0.1：协议加载（官方 skill 浓缩，按 task_type 自动注入）与确定性校验器
# ComfyUI 包内走相对导入；单测（importlib 直载）走 fallback 加载
_PROMPT_MODULES = None
_H3_COMPILER = None
_CLOUD_CREDENTIALS = None
_CLOUD_VIDEO = None
_PROVIDER_REGISTRY = None


def _load_provider_registry():
    """加载 provider_registry.py（能力矩阵 + 语义解析 + preset↔profile 映射）。

    与其它纯模块一致：ComfyUI 包内走相对导入；单测（importlib 直载本文件）走
    与 probe 相同目录的 fallback 加载。
    """
    global _PROVIDER_REGISTRY
    if _PROVIDER_REGISTRY is None:
        import importlib
        import importlib.util
        if __package__:
            _mod = importlib.import_module(".provider_registry", __package__)
        else:
            _spec = importlib.util.spec_from_file_location(
                "provider_registry",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "provider_registry.py"),
            )
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules["provider_registry"] = _mod
            _spec.loader.exec_module(_mod)
        _PROVIDER_REGISTRY = _mod
    return _PROVIDER_REGISTRY


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


def _load_cloud_credentials():
    global _CLOUD_CREDENTIALS
    if _CLOUD_CREDENTIALS is None:
        import importlib
        import importlib.util
        if __package__:
            _mod = importlib.import_module(".cloud_credentials", __package__)
        else:
            _spec = importlib.util.spec_from_file_location(
                "cloud_credentials",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud_credentials.py"),
            )
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules["cloud_credentials"] = _mod
            _spec.loader.exec_module(_mod)
        _CLOUD_CREDENTIALS = _mod
    return _CLOUD_CREDENTIALS


def _load_cloud_video():
    """Load the runtime-only native video adapter without a hard import cycle."""
    global _CLOUD_VIDEO
    if _CLOUD_VIDEO is None:
        import importlib
        import importlib.util
        if __package__:
            _mod = importlib.import_module(".cloud_video", __package__)
        else:
            _spec = importlib.util.spec_from_file_location(
                "cloud_video",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud_video.py"),
            )
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules["cloud_video"] = _mod
            _spec.loader.exec_module(_mod)
        _CLOUD_VIDEO = _mod
    return _CLOUD_VIDEO

LOGGER = logging.getLogger(__name__)

_ENV_API_KEY = "MINIMAX_H3_API_KEY"
_ENV_DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"
_PROMPT_IR_LEGACY_FALLBACK_TOKENS = 16384
_PROMPT_IR_WIDGET_MAX_TOKENS = 131072
_DEEPSEEK_PROVIDER_MAX_OUTPUT_TOKENS = 393216  # 官方 384K；只做能力说明，不作为默认预算。
_DEEPSEEK_SEQUENCE_PAYLOAD_BUDGET = 28 * 1024 * 1024
_GEMINI_PROVIDER_MAX_OUTPUT_TOKENS = 65536
# Gemini inline image data shares a 20 MB request limit with prompt/system JSON.
# Keep media data URLs below 15 MiB so the request envelope has working room.
_GEMINI_INLINE_MEDIA_BUDGET = 15 * 1024 * 1024

_TASK_RULES = {
    "T2VA": "纯文字生视频：无参考图，直接按提示词创造画面。",
    "I2VA": "首帧图生视频：<Picture 1> 是 0.00 秒首帧；保持身份、构图和场景锚点，再描述连续发展。",
    "FL2VA": "首尾帧生视频：<Picture 1> 是首帧，<Picture 2> 是尾帧；描述两者之间可观察、连续的变化路径。",
    "L2VA": "尾帧图生视频：<Picture 1> 是最终帧；推断合理前态并逐渐收束到该尾帧。",
    "Ref2VA": (
        "参考素材：按用户指定用途分别使用每张 <Picture i>。只用于身份的图片写进对应 "
        "<Subject N> 定义，不得自动扩展成场景、构图或风格参考；只有用户明确要求、且不违反"
        "参考跟随档时才允许重新构图。"
    ),
}

_REWRITE_MODES = {
    "strict": "严格遵循用户原始意图，只做结构化和镜头分配，不新增内容。",
    "balanced": "平衡：结构化 + 适度补全细节，保持用户核心意图。",
    "creative": "创意：在用户意图基础上补充生动的镜头语言与细节，但只作用于未被用户或参考素材锁定的维度。",
    "transcribe": "忠实转译：用户原文按句保留、不改词、不扩写、不润色；只做格式套壳（字段名/时间码/标签/对白 <d> 语法）。创作策略模块的扩写要求在本档忽略，仅保留其结构/负面约束条款。用户以 [[原文块]] 包住的内容逐字进入对应字段。",
}

_REFERENCE_FIDELITY_MODES = ("auto", "locked", "structural", "loose")
_REFERENCE_FIDELITY_RULES = {
    "locked": (
        "锁定跟随：用户未明确要求改变的动作阶段、姿势、手势、位移轨迹、景别、机位、真实切镜、"
        "出现顺序与相对时点都必须来自 <Video 1>；只替换目标身份/外观，不得新增、删除、合并、"
        "重排表演阶段或插入无来源特写。"
    ),
    "structural": (
        "结构跟随：保留 <Video 1> 的主要动作阶段、位移方向、真实切镜与先后顺序；允许补全帧间"
        "过渡和不改变阶段语义的微动作，但不得创造新的表演段、姿势主题或镜头类型。"
    ),
    "loose": (
        "松散参考：可把 <Video 1> 作为动作与镜头灵感重新编排；仍不得声称未观察到的内容来自"
        "参考视频，也不得把源表演者身份迁移给目标主体。"
    ),
}

_MEDIA_LABEL_RE = re.compile(r"<(Picture|Video|Audio)\s+(\d+)>", re.IGNORECASE)

# 2026-08-12：模型兼容层——不同模型对 JSON 输出纪律的遵循度不同，按 profile 注入差异指令
_MODEL_PROFILES = [
    "auto", "gemma", "qwen", "cloud", "qwen3_8", "deepseek_vision", "gemini_vision",
    # 2026-08-28：常用云端视觉模型预设（OpenAI 兼容壳 / Claude 原生）
    "openai_compat_vision", "claude_vision",
]
_FRAME_SELECTION_MODES = [
    "uniform_full", "uniform_no_edges", "custom_indices", "custom_percent",
]
_MODEL_COMPAT = {
    "auto": "输出必须是合法 JSON 对象；不要使用 Markdown 代码块（```）包裹 JSON；不要输出 JSON 以外的任何文字。",
    "gemma": "输出必须是合法 JSON 对象；不要使用 Markdown 代码块（```）包裹 JSON；不要输出任何前言、解释或尾注；键名严格使用英文双引号；无法判断的字段填空字符串，不要省略字段。",
    "qwen": "输出必须是合法 JSON 对象；键名与输出契约完全一致；不要输出 JSON 以外的任何文字。",
    "cloud": "输出必须是合法 JSON 对象；不要使用 Markdown 代码块包裹；严格遵循 system 中的创作策略模块条款与官方 H3 协议；不要输出 JSON 以外的文字。",
    "qwen3_8": (
        "直接给出最小且完整的合法 JSON 对象；不要展示思考过程，不要复述、比较或解释 system、"
        "创作模块、参考摘要和输出契约；每个字段只写一次，严格使用契约键名，完成 JSON 后立即停止。"
    ),
    "deepseek_vision": (
        "先在同一次多模态请求中联合理解全部 <Picture N> 目标参考与 <Video 1> 有序代表帧，"
        "再直接输出一个完整合法 JSON 对象。不得把视频开头重绘为新的开场分镜，不得把源表演者"
        "外观写入目标主体；参考视频的动作/镜头时间线是约束而不是灵感，除非用户明确要求改变；"
        "不要复述视觉分析、思考过程或输出契约，JSON 完成后立即停止。"
    ),
    "gemini_vision": (
        "联合理解全部 <Picture N> 目标参考与同一 <Video 1> 的按时间排列代表帧，再直接输出"
        "一个满足 JSON Schema 的完整对象。连续帧不是多段视频或独立分镜；禁止新增、重绘或"
        "重设计开头，禁止把源表演者外观迁移给目标主体；不要输出思考、分析摘要或 JSON 外文本。"
    ),
    # 2026-08-28：OpenAI 兼容壳视觉（豆包方舟 / GLM / MiniMax / 自定义），连续帧联合理解
    "openai_compat_vision": (
        "先在同一次多模态请求中联合理解全部 <Picture N> 目标参考与 <Video 1> 有序代表帧，"
        "再直接输出一个完整合法 JSON 对象。连续帧不是多段视频或独立分镜；禁止新增、重绘或"
        "重设计开头，禁止把源表演者外观迁移给目标主体；不要复述视觉分析、思考过程或输出契约，"
        "JSON 完成后立即停止。"
    ),
    # 2026-08-28：Claude 原生 Messages 视觉（多图 content blocks）
    "claude_vision": (
        "联合理解全部 <Picture N> 目标参考与同一 <Video 1> 的有序代表帧，再直接输出一个完整"
        "合法 JSON 对象。连续帧不是多段视频或独立分镜；禁止新增、重绘或重设计开头，禁止把"
        "源表演者外观迁移给目标主体；不要输出思考、分析摘要或 JSON 外文本。"
    ),
}


def _resolve_reference_fidelity(value: str, task_type: str, has_video: bool) -> tuple[str, str]:
    mode = str(value or "auto").strip().lower()
    if mode not in _REFERENCE_FIDELITY_MODES:
        mode = "auto"
    if not has_video or str(task_type).upper() != "REF2VA":
        return mode, "本次没有 Ref2VA 的 <Video 1> 时间线；参考视频跟随档不生效。"
    effective = "structural" if mode == "auto" else mode
    rule = _REFERENCE_FIDELITY_RULES[effective]
    return effective, (
        "用户明确要求的变化优先；对用户未要求改变的维度，<Video 1> 是权威时间线证据。"
        + rule
        + " rewrite_mode 只决定该跟随边界内的文字补全自由度，不能覆盖本规则。"
    )


def _build_media_inventory(user_prompt: str, picture_count: int,
                           has_video: bool) -> dict[str, int]:
    """连接素材 + 用户明确写出的外部 H3 标签；不从模块或模型输出推断。"""
    # 没有连接到导演不代表下游 H3 没有素材，因此缺失的类型记为“未知”并跳过
    # 清单校验；一旦导演拿到视频证据，就能确定它不是独立 Audio 资产。
    inventory: dict[str, int] = {}
    if int(picture_count) > 0:
        inventory["picture"] = int(picture_count)
    if has_video:
        inventory["video"] = 1
        inventory["audio"] = 0
    for kind, number in _MEDIA_LABEL_RE.findall(str(user_prompt or "")):
        key = kind.casefold()
        inventory[key] = max(inventory.get(key, 0), int(number))
    return inventory


def _format_media_inventory(inventory: dict[str, int]) -> str:
    parts = []
    for kind in ("picture", "video", "audio"):
        label = kind.capitalize()
        if kind not in inventory:
            parts.append(f"{label}=未声明")
            continue
        count = max(0, int(inventory.get(kind, 0)))
        parts.append(f"{label} 1..{count}" if count else f"{label}=无")
    return "，".join(parts)


def _is_qwen38(model_name: str) -> bool:
    """识别 Qwen3.8 常见模型 ID（如 qwen3.8-27b@q6_k）。"""
    name = str(model_name or "").casefold()
    return any(token in name for token in ("qwen3.8", "qwen3_8", "qwen3-8-"))


def _is_gemma4(model_name: str) -> bool:
    name = str(model_name or "").casefold()
    return any(token in name for token in ("gemma4", "gemma-4", "gemma_4"))


def _is_deepseek_v4(model_name: str) -> bool:
    name = str(model_name or "").casefold()
    return "deepseek-v4" in name or "deepseek_v4" in name


def _api_host(base_url: str) -> str:
    from urllib.parse import urlparse

    return (urlparse(str(base_url or "")).hostname or "").casefold()


def _is_deepseek_api(base_url: str) -> bool:
    return _api_host(base_url) == "api.deepseek.com"


def _is_gemini_api(base_url: str) -> bool:
    return _api_host(base_url) == "generativelanguage.googleapis.com"


# 2026-08-28：新增常用云端视觉 provider 的 host 判断（复用 _api_host）。
def _is_volcengine_api(base_url: str) -> bool:
    host = _api_host(base_url)
    return host == "ark.cn-beijing.volces.com" or host.endswith(".volces.com")


def _is_anthropic_api(base_url: str) -> bool:
    return _api_host(base_url) == "api.anthropic.com"


def _is_zhipu_api(base_url: str) -> bool:
    return _api_host(base_url) == "open.bigmodel.cn"


def _is_minimax_api(base_url: str) -> bool:
    host = _api_host(base_url)
    return host == "api.minimaxi.com" or host == "api.minimax.chat" or host == "api.minimax.io"


def _auto_disables_thinking(model_name: str) -> bool:
    """结构化 Prompt IR 默认不让已知混合思考模型吞掉可见输出预算。"""
    name = str(model_name or "").casefold()
    return _is_qwen38(name) or any(
        token in name for token in ("qwen3.6", "qwen3-6", "qwen3_6")
    )


def _resolved_reasoning_mode(api_reasoning: str, model_name: str,
                             request_purpose: str = "final") -> str:
    """返回业务两态/auto（"on" | "off" | "auto"），作为 `_call_chat` 的思考开关依据。

    契约（保持向后兼容）："off" = 尽可能关闭思考；"on" = 启用思考；"auto" = 让
    wire 自己决定（adapter 再按官方默认处理）。对已在能力矩阵登记的云端模型
    （DeepSeek 等）以 `provider_registry` 的 `ReasoningSpec`（can_disable /
    supported_levels）为**单一来源**；本地模型（Qwen/Gemma/其它 LM Studio）未登记，
    保留原模型名启发式。
    """
    mode = str(api_reasoning or "auto").strip().lower()
    if mode in ("on", "off"):
        return mode
    registry = _load_provider_registry()
    profile = registry.find_profile_for_model(model_name) if registry else None
    if profile is not None:
        semantic = registry.resolve_reasoning_semantic(
            profile, api_reasoning, model_name=model_name, request_purpose=request_purpose,
        )
        # business 语义 → 两态/auto：off 保留 off；auto 保留 auto；其余（low/medium/
        # high/max，表明该 wire 需要启用思考）一律归为 "on"。
        if semantic == "off":
            return "off"
        if semantic == "auto":
            return "auto"
        return "on"
    # 未登记模型的启发式（本地 Qwen/Gemma 等）：能力用于素材理解；最终 Prompt IR
    # 是格式编译任务，默认直接作答。
    if _is_qwen38(model_name) or _is_gemma4(model_name):
        return "on" if request_purpose == "analysis" else "off"
    return "off" if _auto_disables_thinking(model_name) else "auto"


def _infer_model_profile(profile: str, model_name: str) -> str:
    """auto 时按模型名轻量推断：gemma→gemma，qwen→qwen，其余→cloud 纪律。"""
    p = str(profile or "auto").strip().lower()
    if p != "auto":
        return p if p in _MODEL_COMPAT else "auto"
    name = str(model_name or "").casefold()
    if _is_qwen38(name):
        return "qwen3_8"
    if _is_deepseek_v4(name) and "vision" in name:
        return "deepseek_vision"
    if "gemma" in name:
        return "gemma"
    if "qwen" in name:
        return "qwen"
    return "cloud"


def _normalize_prompt_ir_budget(value) -> tuple[int, str]:
    """把旧工作流越界值收敛到项目预算；供应商的理论上限不等于 IR 默认预算。"""
    try:
        requested = int(value)
    except (TypeError, ValueError):
        requested = 8192
    if requested > _PROMPT_IR_WIDGET_MAX_TOKENS:
        return (
            _PROMPT_IR_LEGACY_FALLBACK_TOKENS,
            f"[输出预算规范化] 旧/越界 max_tokens={requested} 已改为 "
            f"{_PROMPT_IR_LEGACY_FALLBACK_TOKENS}；供应商最大输出不直接作为 Prompt IR 预算。",
        )
    normalized = max(256, requested)
    if normalized != requested:
        return normalized, f"[输出预算规范化] max_tokens={requested} 已改为 {normalized}。"
    return normalized, ""


def _resolve_api_key(base_url: str, explicit_key: str) -> tuple[str, str]:
    """节点显式值优先；云端只读取与目标 host 匹配的专用变量，通用变量最后兜底。"""
    explicit = str(explicit_key or "").strip()
    if explicit:
        return explicit, "节点 widget（会进入 workflow/history/输出元数据）"
    host = _api_host(base_url)
    if host == "api.deepseek.com":
        value = os.environ.get(_ENV_DEEPSEEK_API_KEY, "").strip()
        if value:
            return value, _ENV_DEEPSEEK_API_KEY
    if "lingbot" in host:
        value = os.environ.get("LINGBOT_API_KEY", "").strip()
        if value:
            return value, "LINGBOT_API_KEY"
    if host == "api.openai.com":
        value = os.environ.get("OPENAI_API_KEY", "").strip()
        if value:
            return value, "OPENAI_API_KEY"
    value = os.environ.get(_ENV_API_KEY, "").strip()
    if value:
        return value, _ENV_API_KEY
    return "", "空"

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

_VIDEO_SEQUENCE_SYSTEM_TEMPLATE = """你是参考视频连续帧分析器。输入图片全部来自同一个 <Video 1>，并已按原始时间顺序排列。
它们可能是少量代表帧，也可能是数百张高密度连续帧；无论数量多少，都只代表一条视频时间线，不是多段视频、多个任务或一批独立分镜图。
你的任务是联合观察整条时间线，提取可供后续人物替换和动作迁移使用的事实；不要把各帧当作独立 <Picture N>，不要编写最终视频提示词。

只输出下面结构的合法 JSON 对象：
{
  "sequence_label": "<Video 1>",
  "source_performer_locator": "用于从画面中唯一定位待替换表演者的一条最短描述；无人或无法判断写 unknown",
  "source_performer_do_not_preserve": "源表演者可观察到但不应迁移给目标人物的脸、头发、身体身份和服装；无法判断写 unknown",
  "action_timeline": ["把连续相邻帧合并为少数动作阶段，记录姿势、动作方向、手势和节奏变化；只写实际可见事实"],
  "camera_timeline": "景别、机位、推拉摇移和真实切镜点；相邻帧变化不等于切镜，无法判断写 unknown",
  "scene_and_background": "场景、背景、灯光和背景动态事实",
  "continuity_and_occlusion": "遮挡、转身、出入画及可能影响替换的连续性风险",
  "uncertainties": ["仅凭稀疏代表帧无法确定的内容"]
}

纪律：
1. source_performer_locator 只用于选择源视频中的替换对象，不得把源人物外观写成目标人物。
2. 第一张帧只是 <Video 1> 已有开头的证据；禁止重绘、重设计或新增一个“更好的开头”，也禁止把任何后续帧重解释为新的开场。
3. 只在画面出现真实硬切/场景断点时记录切镜；高密度相邻帧的近似、位移和遮挡都是正常连续运动，不得逐帧复述或人为新增分镜。
4. 不推断帧间没有展示的精确动作，不虚构音频、对白或精确时间戳。
5. 多个帧中出现的同一表演者仍是一个人，不得理解为克隆体或多个人。
6. 只输出 JSON，不要 Markdown 或额外说明。"""

_H3_SYSTEM_TEMPLATE = """你是 MiniMax H3 视频生成模型的提示词导演。你的任务是把用户意图规划成结构化 Prompt IR；最终字段顺序与格式由本地 Python 编译器负责。

输出必须是合法 JSON，且只能包含下面契约中的字段：
{output_contract}

规则：
1. 总时长约 {duration}s，规划约 {shots} 个镜头；首镜使用 [Shot 1] 且不写时间戳，后续镜头使用 [Shot N] At MM:SS.mmm，时间严格递增且不超过总时长。
2. {task_rule}
3. 改写模式：{rewrite_mode_rule}
4. 字段名固定使用契约中的英文；描述正文使用{language}，用户原始对白、歌词和画面可见文字不得改写或翻译。
5. 不要额外输出 Markdown、解释、字段外文本或重复的分镜数组。
6. 输出格式纪律：{model_compat}
7. 参考媒体证据纪律：{media_evidence}
8. 参考视频跟随纪律：{reference_fidelity_rule}
9. IR 合规纪律：{ir_discipline}
10. 紧凑输出纪律：不要复述输入、规范、模块或分析过程；不要讨论指令冲突；只保留生成 H3 提示词所需事实。"""


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


def _output_json_schema(task_type: str) -> dict:
    """LM Studio JSON Schema：约束最终 Prompt IR 字段，完整生成时保证可解析。"""
    if str(task_type).upper() == "REF2VA":
        fields = [
            "subject_definitions", "summary", "retention_analysis",
            "detailed_description", "overall_soundscape", "non_diegetic_music",
        ]
    else:
        fields = [
            "integrated_multimodal_description", "overall_soundscape",
            "non_diegetic_music",
        ]
    return {
        "type": "object",
        "properties": {name: {"type": "string"} for name in fields},
        "required": fields,
        "additionalProperties": False,
    }


def _image_tensor_to_data_url(image_tensor: torch.Tensor, max_side: int = 1024,
                              quality: int = 85) -> str:
    """IMAGE 张量 → JPEG data URL（取第一帧，限制最长边，供多模态 API 使用）。"""
    t = image_tensor.detach().cpu()
    while t.ndim > 3:
        t = t[0]
    arr = (t.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    if max(pil.size) > max_side:
        pil.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=max(35, min(95, int(quality))))
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _uniform_sample_indices(frame_count: int, limit: int) -> list[int]:
    """在完整 IMAGE batch 上包含首尾地均匀取样；不重复索引。"""
    count = max(0, int(frame_count))
    cap = max(1, int(limit))
    if count <= cap:
        return list(range(count))
    if cap == 1:
        return [count // 2]
    return [round(i * (count - 1) / (cap - 1)) for i in range(cap)]


def _spread_candidates(candidates: list[int], limit: int) -> list[int]:
    """候选索引超过上限时，在候选列表中再次均匀取样并保持时间顺序。"""
    ordered = sorted(set(int(index) for index in candidates))
    positions = _uniform_sample_indices(len(ordered), limit)
    return [ordered[pos] for pos in positions]


def _selection_values(spec: str) -> list[str]:
    return [
        part for part in str(spec or "").replace("，", ",").replace("；", ",")
        .replace(";", ",").replace(" ", ",").split(",") if part
    ]


def _select_frame_indices(frame_count: int, limit: int, mode: str = "uniform_full",
                          spec: str = "") -> tuple[list[int], str]:
    """按快捷预设/自定义语法选择有序帧索引；无效自定义值安全回退均匀全程。"""
    count = max(0, int(frame_count))
    cap = max(1, int(limit))
    if count < 1:
        return [], "空序列"
    normalized_mode = str(mode or "uniform_full").strip().lower()
    if normalized_mode not in _FRAME_SELECTION_MODES:
        normalized_mode = "uniform_full"

    if normalized_mode == "uniform_full":
        selected = _uniform_sample_indices(count, cap)
        return selected, "均匀全程（包含首尾）"

    if normalized_mode == "uniform_no_edges":
        if count <= cap:
            return list(range(count)), "序列不超过上限，全部发送"
        last = count - 1
        start = round(last * 0.10)
        end = round(last * 0.90)
        selected = [
            round(start + i * (end - start) / (cap - 1))
            for i in range(cap)
        ] if cap > 1 else [round((start + end) / 2)]
        return sorted(set(selected)), "均匀避开首尾（约 10%–90%）"

    values = _selection_values(spec)
    try:
        if not values:
            raise ValueError("未填写选择值")
        if normalized_mode == "custom_indices":
            custom = []
            for value in values:
                index = int(value)
                if index < 0:
                    index = count + index
                if index < 0 or index >= count:
                    raise ValueError(f"索引 {value} 超出 0..{count - 1}")
                custom.append(index)
            selected = _spread_candidates(custom, cap)
            return selected, f"自定义帧序号：{selected}"

        percentages = []
        for value in values:
            number = float(value.rstrip("%"))
            if number < 0.0 or number > 100.0:
                raise ValueError(f"百分比 {value} 超出 0..100")
            percentages.append(number)
        custom = [round((count - 1) * value / 100.0) for value in percentages]
        selected = _spread_candidates(custom, cap)
        return selected, f"自定义百分比：{percentages} → 索引 {selected}"
    except (TypeError, ValueError) as exc:
        selected = _uniform_sample_indices(count, cap)
        return selected, f"[选择回退] {exc}；已改用均匀全程 {selected}"


def _image_batch_to_data_urls(image_tensor: torch.Tensor, limit: int = 4,
                              max_side: int = 1024, selection_mode: str = "uniform_full",
                              selection_spec: str = "", quality: int = 85,
                              max_payload_bytes: int = 0,
                              adaptive: bool = False) -> tuple[list[tuple[int, str]], int, str]:
    """IMAGE 单图/批次 → 限量 JPEG data URLs。

    返回 ``([(原批次索引, data_url), ...], 原批次帧数, 选择说明)``。只把被选帧搬到
    CPU，避免直接连接长视频 batch 时复制整段视频占用额外内存。
    """
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("video_frame_sequence 必须是 ComfyUI IMAGE 张量")
    t = image_tensor.detach()
    if t.ndim == 3:
        t = t.unsqueeze(0)
    if t.ndim != 4:
        raise ValueError(
            f"video_frame_sequence 需要 [B,H,W,C] 或 [H,W,C]，收到 shape={tuple(t.shape)}"
        )
    total = int(t.shape[0])
    if total < 1:
        raise ValueError("video_frame_sequence 是空 IMAGE batch")
    selected, selection_note = _select_frame_indices(
        total, limit, mode=selection_mode, spec=selection_spec,
    )
    encode_presets = [(int(max_side), int(quality))]
    if adaptive:
        if len(selected) > 400:
            # DeepSeek 600-frame boundary mode: preserve the requested count
            # before falling back to uniform frame dropping under the byte cap.
            encode_presets = [(384, 68), (320, 64), (256, 60)]
        elif len(selected) > 200:
            encode_presets = [(512, 72), (384, 68), (320, 64), (256, 60)]
        elif len(selected) > 120:
            encode_presets = [(640, 75), (512, 72), (384, 68)]
        elif len(selected) > 50:
            encode_presets = [(768, 80), (512, 72), (384, 68)]
        else:
            encode_presets = [(int(max_side), int(quality)), (768, 80), (512, 72), (384, 68)]

    encoded = []
    used_side, used_quality = encode_presets[0]
    for used_side, used_quality in encode_presets:
        encoded = [
            (idx, _image_tensor_to_data_url(t[idx], max_side=used_side, quality=used_quality))
            for idx in selected
        ]
        if not max_payload_bytes or sum(len(url) for _, url in encoded) <= max_payload_bytes:
            break

    encoded_bytes = sum(len(url) for _, url in encoded)
    if max_payload_bytes and encoded_bytes > max_payload_bytes and len(encoded) > 1:
        # 极端高熵素材在最低编码档仍可能超预算；均匀减帧，避免整次请求被 413 拒绝。
        keep = max(1, int(len(encoded) * max_payload_bytes / encoded_bytes * 0.95))
        encoded = [encoded[pos] for pos in _uniform_sample_indices(len(encoded), keep)]
        encoded_bytes = sum(len(url) for _, url in encoded)
        selection_note += f"；请求体预算触发均匀减帧至 {len(encoded)} 张"
    selection_note += (
        f"；编码最长边≤{used_side}px/JPEG q{used_quality}，"
        f"序列 Data URL≈{encoded_bytes / (1024 * 1024):.1f} MiB"
    )
    return encoded, total, selection_note


def _summarize_indices(indices: list[int], edge: int = 6) -> str:
    """报告中避免把数百个索引整段塞进 ComfyUI 节点输出。"""
    values = [int(value) for value in indices]
    if len(values) <= edge * 2:
        return str(values)
    return f"{values[:edge]} ... {values[-edge:]}（共 {len(values)} 个）"


def _parse_timeline_manifest(value: str, input_frame_count: int | None = None) -> dict:
    """Parse only the deterministic fields emitted by ``MiniMaxH3VideoContext``.

    The manifest is optional and never trusted as prompt text.  We keep a narrow
    typed projection so arbitrary JSON cannot become a hidden system prompt.
    """
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"时间线 manifest 不是合法 JSON：{exc}") from exc
    if not isinstance(raw, dict) or raw.get("sequence_label") != "<Video 1>":
        raise ValueError("时间线 manifest 必须来自 <Video 1> Video Context 节点")
    try:
        total_frames = max(1, int(raw.get("total_frames")))
        fps = max(0.001, float(raw.get("fps")))
        duration = max(0.0, float(raw.get("duration_seconds")))
    except (TypeError, ValueError) as exc:
        raise ValueError("时间线 manifest 缺少有效 total_frames/fps/duration_seconds") from exc
    frames_raw = raw.get("frames")
    if not isinstance(frames_raw, list):
        raise ValueError("时间线 manifest 缺少 frames 数组")
    if input_frame_count is not None and len(frames_raw) != int(input_frame_count):
        raise ValueError(
            f"时间线 manifest 有 {len(frames_raw)} 帧记录，但当前 IMAGE 批次为 {input_frame_count} 帧；"
            "请把同一个 Video Context 的 selected_frames 与 timeline_manifest 一起连接"
        )
    frames = []
    for item in frames_raw:
        if not isinstance(item, dict):
            raise ValueError("时间线 manifest.frames 含非对象条目")
        try:
            source_index = int(item.get("frame"))
            time_sec = float(item.get("time_sec"))
            position = float(item.get("position"))
        except (TypeError, ValueError) as exc:
            raise ValueError("时间线 manifest.frames 缺少有效 frame/time_sec/position") from exc
        if source_index < 0 or source_index >= total_frames:
            raise ValueError(f"时间线源帧索引 {source_index} 超出 0..{total_frames - 1}")
        frames.append({
            "frame": source_index,
            "time_sec": round(max(0.0, time_sec), 4),
            "timecode": str(item.get("timecode") or ""),
            "position": round(min(1.0, max(0.0, position)), 4),
        })
    segments = []
    for item in raw.get("segments", []):
        if not isinstance(item, dict):
            continue
        try:
            start = max(0.0, float(item.get("start_sec")))
            end = max(start, float(item.get("end_sec")))
            segment = max(1, int(item.get("segment")))
        except (TypeError, ValueError):
            continue
        segments.append({
            "segment": segment,
            "start_sec": round(start, 4),
            "end_sec": round(end, 4),
            "start_timecode": str(item.get("start_timecode") or ""),
            "end_timecode": str(item.get("end_timecode") or ""),
        })
    shot_boundaries = []
    for value in raw.get("shot_boundaries_input", []):
        try:
            boundary = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 < boundary < duration:
            shot_boundaries.append(round(boundary, 4))
    shot_boundaries = sorted(set(shot_boundaries))
    projected = {
        "sequence_label": "<Video 1>",
        "total_frames": total_frames,
        "fps": round(fps, 3),
        "duration_seconds": round(duration, 4),
        "frames": frames,
        "segments": segments,
        "shot_boundaries_input": shot_boundaries,
    }
    # v1 先为多参考视频素材集合预留稳定身份与用途；旧 manifest 没有这两个字段时
    # 仍按单一 <Video 1> 兼容。只投影固定值，避免外部 JSON 注入任意说明文本。
    if raw.get("schema") == "minimax_h3.reference_video_timeline/v1":
        projected["schema"] = raw["schema"]
    asset = raw.get("asset")
    if isinstance(asset, dict) and asset.get("kind") == "reference_video":
        projected["asset"] = {
            "id": "video_1",
            "kind": "reference_video",
            "label": "<Video 1>",
            "roles": ["motion", "camera", "timing"],
        }
    return projected


def _chat_completions_url(base_url: str) -> str:
    """把 base_url 规范化为 /chat/completions 端点（兼容裸根/带 /vN//openai）。

    - 已含 /chat/completions：原样。
    - 以 /vN 结尾（/v1、/v3、/v4，如方舟 /api/v3、智谱 /paas/v4、MiniMax /v1）或 /openai：
      视为已是 API 前缀 → 直接加 /chat/completions。
    - 否则（裸 host）→ 加 /v1/chat/completions。
    """
    url = base_url.rstrip("/")
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"api_base_url 仅支持 http/https: {base_url!r}")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1") or url.endswith("/openai") or re.search(r"/v[0-9]+$", url):
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(bytes(value)).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(str(value or "").encode("utf-8"))


def _json_fingerprint(value) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(canonical)


_SAFE_PROVIDER_ERROR_HINTS = {
    "local_model_not_loaded": (
        "本机 API 已收到请求，但原生/JIT 自动加载未成功；"
        "请刷新模型列表并核对模型 ID，仍失败时再到 LM Studio 检查加载日志"
    ),
    "model_not_found": "服务端找不到该模型；请刷新模型列表并核对模型 ID",
    "context_limit_exceeded": "输入与输出预算超过当前已加载上下文；请减少帧/模块或降低输出令牌",
    "output_budget_rejected": "服务端拒绝当前输出令牌预算；请降低 Prompt IR 最大输出令牌",
    "request_shape_rejected": "服务端拒绝当前请求结构；该模型可能不兼容多图或结构化输出字段",
    "structured_output_rejected": "服务端不接受当前结构化输出参数；节点会在允许时做一次受控降级",
    "authentication_failed": "API 凭据无效或无权调用当前模型",
    "payload_too_large": "请求体过大；请减少视频帧数量或降低图片尺寸",
}


def _safe_provider_error_code(exc) -> str:
    """Map a provider error body to a small safe code; never expose its body.

    Provider bodies may contain prompts or echo request fragments.  We therefore
    only recognize a narrow phrase set and return a canned local-language hint.
    """
    if not isinstance(exc, urllib.error.HTTPError):
        return ""
    try:
        raw = exc.read(8192)
    except Exception:  # noqa: BLE001 - diagnostic only
        return ""
    text = raw.decode("utf-8", errors="ignore").casefold()
    if "no models loaded" in text or "no model loaded" in text:
        return "local_model_not_loaded"
    if "model_not_found" in text or "model not found" in text or "unknown model" in text:
        return "model_not_found"
    if any(token in text for token in ("context length", "context window", "maximum context")):
        return "context_limit_exceeded"
    if any(token in text for token in ("max_tokens", "max output tokens", "output token")):
        return "output_budget_rejected"
    if "invalid discriminator" in text or "invalid content type" in text:
        return "request_shape_rejected"
    if "response_format" in text or "json_schema" in text:
        return "structured_output_rejected"
    if any(token in text for token in ("invalid api key", "unauthorized", "authentication failed")):
        return "authentication_failed"
    if "payload too large" in text or "request entity too large" in text:
        return "payload_too_large"
    return ""


class _TransportFailure(RuntimeError):
    """不携带响应正文/密钥的安全传输错误，同时保留分类元数据。"""

    def __init__(self, category: str, status, meta: dict):
        self.category = str(category or "unknown")
        self.status = status
        self.transport_meta = dict(meta or {})
        status_note = f"，HTTP {status}" if status is not None else ""
        attempts = self.transport_meta.get("attempts") or []
        provider_error = str(attempts[-1].get("provider_error") or "") if attempts else ""
        action = _SAFE_PROVIDER_ERROR_HINTS.get(provider_error, "")
        action_note = f"；建议：{action}" if action else ""
        super().__init__(
            f"API 传输失败：{self.category}{status_note}；"
            f"attempts={len(self.transport_meta.get('attempts') or [])}；"
            f"request_sha256={self.transport_meta.get('request_sha256', '?')}"
            f"{action_note}"
        )


def _classify_transport_error(exc) -> tuple[str, bool, object]:
    """返回 (安全类别, 是否允许原包重试, HTTP 状态)。"""
    if isinstance(exc, urllib.error.HTTPError):
        status = int(exc.code)
        if status == 408:
            return "http_408_timeout", True, status
        if status == 429:
            return "http_429_rate_limit", True, status
        if 500 <= status <= 599:
            return "http_5xx_server", True, status
        if status == 413:
            return "http_413_payload_too_large", False, status
        return "http_4xx_non_retryable", False, status

    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, (socket.timeout, TimeoutError)):
        return "timeout", True, None
    if isinstance(reason, ssl.SSLCertVerificationError) or "certificate verify failed" in str(reason).casefold():
        return "ssl_certificate_non_retryable", False, None
    if isinstance(reason, ssl.SSLError):
        return "ssl_disconnect", True, None
    if isinstance(reason, (
        http.client.RemoteDisconnected,
        http.client.IncompleteRead,
        ConnectionResetError,
        ConnectionAbortedError,
        BrokenPipeError,
    )):
        return "remote_disconnect", True, None

    text = f"{reason.__class__.__name__}: {reason}".casefold()
    if any(token in text for token in (
        "unexpected_eof", "eof occurred", "remote end closed", "connection reset",
        "connection aborted", "incomplete read", "timed out", "timeout",
    )):
        return "remote_disconnect", True, None
    return "non_retryable_transport", False, None


def _retry_after_seconds(exc, attempt_index: int) -> float:
    """429 优先采用 Retry-After；其余指数退避并加入少量抖动。"""
    if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
        value = exc.headers.get("Retry-After") if exc.headers is not None else None
        try:
            if value is not None:
                return min(30.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(value))
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=datetime.timezone.utc)
                remaining = (retry_at - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
                return min(30.0, max(0.0, remaining))
            except (TypeError, ValueError, OverflowError):
                pass
    base = min(8.0, 1.0 * (2 ** max(0, int(attempt_index))))
    return base + random.uniform(0.0, 0.25)


def _post_json_bytes(url: str, body_bytes: bytes, headers: dict, timeout_s: int,
                     max_retries: int = 0) -> tuple[bytes, dict]:
    """发送已固化的 JSON 字节；可重试时始终复用同一 body_bytes。"""
    body = bytes(body_bytes)
    fingerprint = _sha256_bytes(body)
    attempts = []
    retry_budget = max(0, min(1, int(max_retries)))
    for attempt_index in range(retry_budget + 1):
        started = time.perf_counter()
        req = urllib.request.Request(
            url, data=body, headers=dict(headers), method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                response_bytes = resp.read()
                status = getattr(resp, "status", getattr(resp, "code", 200))
            attempts.append({
                "attempt": attempt_index + 1,
                "result": "success",
                "status": int(status or 200),
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            })
            return response_bytes, {
                "request_sha256": fingerprint,
                "payload_bytes": len(body),
                "attempts": attempts,
                "transport_retried": attempt_index > 0,
            }
        except Exception as exc:  # noqa: BLE001
            category, retryable, status = _classify_transport_error(exc)
            provider_error = _safe_provider_error_code(exc)
            should_retry = bool(retryable and attempt_index < retry_budget)
            attempt = {
                "attempt": attempt_index + 1,
                "result": "retry" if should_retry else "failed",
                "category": category,
                "status": status,
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
            if provider_error:
                attempt["provider_error"] = provider_error
            attempts.append(attempt)
            meta = {
                "request_sha256": fingerprint,
                "payload_bytes": len(body),
                "attempts": attempts,
                "transport_retried": any(item.get("result") == "retry" for item in attempts),
            }
            if not should_retry:
                raise _TransportFailure(category, status, meta) from exc
            time.sleep(_retry_after_seconds(exc, attempt_index))

    raise AssertionError("unreachable transport loop")


class _ChatCompletionText(str):
    """兼容 str 的 API 文本，同时保留截断与 token 诊断元数据。"""

    def __new__(cls, content, *, finish_reason="", usage=None, reasoning_content="",
                response_format_fallback=False, backend="openai_compat",
                transport_meta=None, media_meta=None):
        obj = super().__new__(cls, content or "")
        obj.finish_reason = str(finish_reason or "")
        obj.usage = usage if isinstance(usage, dict) else {}
        obj.reasoning_content = str(reasoning_content or "")
        obj.response_format_fallback = bool(response_format_fallback)
        obj.backend = str(backend or "openai_compat")
        obj.transport_meta = dict(transport_meta or {})
        obj.media_meta = dict(media_meta or {})
        return obj


def _chat_stats_note(raw, label: str) -> str:
    """把 LM Studio/OpenAI completion 元数据压成一行可见报告。"""
    finish_reason = str(getattr(raw, "finish_reason", "") or "")
    usage = getattr(raw, "usage", {})
    transport = getattr(raw, "transport_meta", {})
    transport = transport if isinstance(transport, dict) else {}
    if not finish_reason and not usage and not transport:
        return ""
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("completion_tokens_details")
    details = details if isinstance(details, dict) else {}
    prompt_tokens = usage.get("prompt_tokens", "?")
    completion_tokens = usage.get("completion_tokens", "?")
    reasoning_tokens = details.get("reasoning_tokens", "?")
    fallback = "；结构化格式被服务器拒绝后已降级" if getattr(
        raw, "response_format_fallback", False
    ) else ""
    backend = str(getattr(raw, "backend", "") or "")
    backend_note = f"，backend={backend}" if backend else ""
    request_hash = str(transport.get("request_sha256") or "")
    payload_bytes = int(transport.get("payload_bytes") or 0)
    attempts = transport.get("attempts") if isinstance(transport.get("attempts"), list) else []
    network_elapsed_ms = sum(
        int(item.get("elapsed_ms") or 0) for item in attempts if isinstance(item, dict)
    )
    transport_note = ""
    if request_hash:
        transport_note = (
            f"，request_sha256={request_hash}，payload={payload_bytes / 1024 / 1024:.2f} MiB，"
            f"transport_attempts={len(attempts)}"
            f"，network_elapsed={network_elapsed_ms / 1000:.2f}s"
        )
    return (
        f"{label}: finish_reason={finish_reason or '?'}{backend_note}，prompt={prompt_tokens} token，"
        f"completion={completion_tokens} token，reasoning={reasoning_tokens} token"
        f"{fallback}{transport_note}"
    )


def _completion_retry_reason(raw) -> str:
    """检测必须重试的完成状态；普通 str mock/兼容端点仍可继续解析。"""
    finish_reason = str(getattr(raw, "finish_reason", "") or "").casefold()
    if finish_reason in ("length", "max_tokens"):
        return f"API 因输出长度停止（finish_reason={finish_reason}）"
    if not str(raw or "").strip():
        usage = getattr(raw, "usage", {})
        details = usage.get("completion_tokens_details", {}) if isinstance(usage, dict) else {}
        reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
        if reasoning_tokens:
            return f"可见 content 为空，推理已消耗 {reasoning_tokens} token"
        return "API 返回的可见 content 为空"
    return ""


def _compact_retry_messages(messages, model: str, reason: str) -> list[dict]:
    """不携带截断思考内容，给同一请求追加一次直接交付指令。"""
    retry_suffix = (
        "\n\n【截断恢复重试】前一次生成未完整结束：" + reason + "。"
        "本次不要分析、解释、复述规范或讨论冲突；直接输出一个最小且完整的 JSON 对象，"
        "每个必需字段只写一次，闭合所有字符串、数组和大括号后立即停止。"
    )
    if _is_qwen38(model):
        # Qwen 官方同时支持把 /no_think 放在 system/user 消息中；与模板参数双保险。
        retry_suffix += "\n/no_think"
    retried = []
    system_patched = False
    for message in messages:
        item = dict(message)
        if not system_patched and item.get("role") == "system":
            item["content"] = str(item.get("content") or "") + retry_suffix
            system_patched = True
        retried.append(item)
    if not system_patched:
        retried.insert(0, {"role": "system", "content": retry_suffix.lstrip()})
    return retried


def _qwen_no_think_messages(messages) -> list[dict]:
    """给本机模板不响应 enable_thinking=false 的 Qwen3.8 增加双文本控制通道。"""
    patched = []
    system_added = False
    for message in messages:
        item = dict(message)
        if not system_added and item.get("role") == "system":
            content = str(item.get("content") or "")
            if "/no_think" not in content:
                content += "\n/no_think"
            item["content"] = content
            system_added = True
        patched.append(item)
    if not system_added:
        patched.insert(0, {"role": "system", "content": "/no_think"})

    # 这份 Qwen3.8 GGUF/LM Studio 模板实测只放 system 仍可能长思考；
    # 同时把开关放入最后一条 user 消息，符合 Qwen 官方的文本开关语义。
    for index in range(len(patched) - 1, -1, -1):
        if patched[index].get("role") != "user":
            continue
        item = dict(patched[index])
        content = item.get("content")
        if isinstance(content, list):
            parts = [dict(part) if isinstance(part, dict) else part for part in content]
            has_token = any(
                isinstance(part, dict)
                and "/no_think" in str(part.get("text") or "")
                for part in parts
            )
            if not has_token:
                parts.append({"type": "text", "text": "/no_think"})
            item["content"] = parts
        else:
            text = str(content or "")
            if "/no_think" not in text:
                text += "\n/no_think"
            item["content"] = text
        patched[index] = item
        break
    return patched


def _is_local_lmstudio_url(base_url: str) -> bool:
    from urllib.parse import urlparse

    host = (urlparse(str(base_url or "")).hostname or "").casefold()
    return host in ("127.0.0.1", "localhost", "localhost.localdomain", "::1")


def _lmstudio_native_input(messages) -> tuple[str, object]:
    """OpenAI messages → LM Studio /api/v1/chat 的 system_prompt + input。"""
    system_parts = []
    inputs = []
    for message in messages:
        role = str(message.get("role") or "")
        content = message.get("content")
        if role == "system":
            system_parts.append(str(content or ""))
            continue
        if role != "user":
            continue
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    inputs.append({"type": "text", "content": str(part)})
                    continue
                part_type = part.get("type")
                if part_type == "text":
                    inputs.append({"type": "text", "content": str(part.get("text") or "")})
                elif part_type == "image_url":
                    image_url = part.get("image_url")
                    if isinstance(image_url, dict):
                        image_url = image_url.get("url")
                    if image_url:
                        inputs.append({"type": "image", "data_url": str(image_url)})
        else:
            inputs.append({"type": "text", "content": str(content or "")})
    if len(inputs) == 1 and inputs[0].get("type") == "text":
        native_input = inputs[0]["content"]
    else:
        native_input = inputs
    return "\n\n".join(part for part in system_parts if part), native_input


def _call_lmstudio_native(base_url, api_key, model, messages, temperature,
                          max_tokens, timeout_s, reasoning_mode) -> _ChatCompletionText:
    """LM Studio 原生 API；按模型 ID 请求，允许服务端按需/JIT 加载模型。"""
    root = _lmstudio_root(base_url)
    system_prompt, native_input = _lmstudio_native_input(messages)
    payload = {
        "model": model,
        "input": native_input,
        "system_prompt": system_prompt,
        "temperature": min(1.0, max(0.0, float(temperature))),
        "top_p": 0.8 if reasoning_mode == "off" and _is_qwen38(model) else 0.95,
        "top_k": 20,
        "max_output_tokens": int(max_tokens),
        "reasoning": reasoning_mode,
        "store": False,
    }
    _reject_link_local_target(root)
    req = urllib.request.Request(
        root.rstrip("/") + "/api/v1/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    outputs = data.get("output", [])
    content = "\n".join(
        str(item.get("content") or "") for item in outputs
        if isinstance(item, dict) and item.get("type") == "message"
    )
    reasoning_content = "\n".join(
        str(item.get("content") or "") for item in outputs
        if isinstance(item, dict) and item.get("type") == "reasoning"
    )
    stats = data.get("stats", {})
    stats = stats if isinstance(stats, dict) else {}
    completion_tokens = stats.get("total_output_tokens")
    finish_reason = "length" if (
        isinstance(completion_tokens, int) and completion_tokens >= int(max_tokens)
    ) else "stop"
    usage = {
        "prompt_tokens": stats.get("input_tokens"),
        "completion_tokens": completion_tokens,
        "total_tokens": (
            (stats.get("input_tokens") or 0) + (completion_tokens or 0)
        ),
        "completion_tokens_details": {
            "reasoning_tokens": stats.get("reasoning_output_tokens", 0),
        },
    }
    return _ChatCompletionText(
        content, finish_reason=finish_reason, usage=usage,
        reasoning_content=reasoning_content, backend="lmstudio_native",
    )


def _gemini_data_url_part(value) -> dict:
    """OpenAI-style data URL → Gemini inlineData part (user media only)."""
    data_url = value.get("url") if isinstance(value, dict) else value
    raw = str(data_url or "")
    if not raw.startswith("data:") or ";base64," not in raw:
        raise ValueError("Gemini 当前只接受节点内部生成的 base64 data URL 图片")
    header, encoded = raw.split(",", 1)
    mime_type = header[5:].split(";", 1)[0].strip().casefold()
    if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise ValueError(f"Gemini 不支持的图片 MIME：{mime_type or '?'}")
    if not encoded:
        raise ValueError("Gemini 图片 data URL 内容为空")
    return {"inlineData": {"mimeType": mime_type, "data": encoded}}


def _gemini_native_messages(messages) -> tuple[dict | None, list[dict]]:
    """OpenAI messages → Gemini systemInstruction + contents.

    System roles are never inserted into ``contents``.  A real user turn must
    be last; this prevents the INVALID_ARGUMENT shape reproduced in AA-Codex.
    """
    system_parts = []
    contents = []
    for message in messages:
        role = str(message.get("role") or "").strip().casefold()
        content = message.get("content")
        if role == "system":
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append({"text": str(part.get("text") or "")})
                    elif part:
                        raise ValueError("Gemini systemInstruction 只允许文本内容")
            else:
                system_parts.append({"text": str(content or "")})
            continue
        if role not in {"user", "assistant"}:
            raise ValueError(f"Gemini 不支持的消息角色：{role or '?'}")
        parts = []
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    parts.append({"text": str(part)})
                    continue
                part_type = str(part.get("type") or "")
                if part_type == "text":
                    parts.append({"text": str(part.get("text") or "")})
                elif part_type == "image_url":
                    parts.append(_gemini_data_url_part(part.get("image_url")))
                elif part_type == "gemini_video_file":
                    parts.append({"__minimax_h3_native_video__": dict(part)})
                else:
                    raise ValueError(f"Gemini 不支持的内容块类型：{part_type or '?'}")
        else:
            parts.append({"text": str(content or "")})
        native_role = "model" if role == "assistant" else "user"
        if contents and contents[-1].get("role") == native_role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": native_role, "parts": parts})
    if not contents or contents[-1].get("role") != "user":
        raise ValueError("Gemini 请求必须包含有效 user 内容，并以 user 轮结束")
    system_instruction = {"parts": system_parts} if system_parts else None
    return system_instruction, contents


def _gemini_thinking_level(api_reasoning: str, model: str,
                            request_purpose: str) -> str:
    """Map the common UI to supported Gemini 3 thinking levels.

    思考**语义**（off 能否真关 / auto 默认 / level 校验）以 provider_registry 的
    `ReasoningSpec` 为单一来源，不再散落在模型名启发式里；本函数只做「统一语义 →
    Gemini ``thinkingLevel`` 字段」的 wire 层翻译。Gemini 3.7 Flash 与 3.1 Pro
    拒绝 ``minimal``，用 ``low`` 承载"最低开启"语义；其它 Gemini 3 Flash/Lite 模型
    可用 ``minimal`` 产出更紧凑的最终 Prompt IR。
    """
    mode = str(api_reasoning or "auto").strip().casefold()
    if mode == "on":
        return "high"
    registry = _load_provider_registry()
    profile = registry.find_profile_for_model(model) if registry else None
    if registry is not None and profile is not None:
        semantic = registry.resolve_reasoning_semantic(
            profile, api_reasoning, model_name=model, request_purpose=request_purpose,
        )
    else:
        # 能力矩阵不可达时的兜底：保留旧启发式（不中断调用）。
        semantic = mode
    model_name = str(model or "").casefold()
    minimum = "low" if ("3.7" in model_name or "3.1-pro" in model_name) else "minimal"
    if semantic == "auto":
        # auto 不做语义改写（registry 透传），由本 adapter 按官方默认定向。
        return "low" if str(request_purpose or "final") == "analysis" else minimum
    if semantic == "off":
        # registry 只在 can_disable 时保留 off；Gemini 3 字段层无 off，兜底最低开启档。
        return minimum
    if semantic in ("high", "max"):
        return "high"
    if semantic == "medium":
        return "medium"
    # semantic in ("low", 其它非 off 开启级)：都收敛到该 wire 的最低开启字段。
    return minimum


def _reasoning_report_mode(base_url: str, api_reasoning: str, model: str,
                           request_purpose: str) -> str:
    if _is_gemini_api(base_url) and str(model or "").casefold().startswith("gemini-3"):
        return "Gemini:" + _gemini_thinking_level(
            api_reasoning, model, request_purpose,
        )
    return _resolved_reasoning_mode(api_reasoning, model, request_purpose)


def _call_gemini_native(base_url, api_key, model, messages, temperature,
                         max_tokens, timeout_s, json_mode="force",
                         api_reasoning="auto", response_schema=None,
                         request_purpose="final") -> _ChatCompletionText:
    """Call Google's native v1beta generateContent endpoint with images/schema."""
    import urllib.parse

    system_instruction, contents = _gemini_native_messages(messages)
    native_video_meta = {}
    cleanup_name = ""
    native_markers = []
    for content in contents:
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            continue
        for index, part in enumerate(parts):
            if isinstance(part, dict) and "__minimax_h3_native_video__" in part:
                native_markers.append((parts, index, part["__minimax_h3_native_video__"]))
    if len(native_markers) > 1:
        raise ValueError("Gemini 云端导演每次只允许一个原生视频")
    if native_markers:
        parts, index, marker = native_markers[0]
        inline_overhead_bytes = 0
        for content in contents:
            content_parts = content.get("parts") if isinstance(content, dict) else None
            if not isinstance(content_parts, list):
                continue
            for part in content_parts:
                inline_data = part.get("inlineData") if isinstance(part, dict) else None
                if isinstance(inline_data, dict):
                    inline_overhead_bytes += len(str(inline_data.get("data") or ""))
        video_part, native_video_meta, cleanup_name = _load_cloud_video().prepare_gemini_video(
            marker.get("source"), api_key,
            route=str(marker.get("route") or "auto"),
            fps=float(marker.get("fps") or 0.0),
            timeout_s=int(timeout_s),
            inline_overhead_bytes=inline_overhead_bytes,
        )
        parts[index] = video_part
    generation_config = {
        "temperature": min(2.0, max(0.0, float(temperature))),
        "maxOutputTokens": int(max_tokens),
    }
    if str(model or "").casefold().startswith("gemini-3"):
        generation_config["thinkingConfig"] = {
            "thinkingLevel": _gemini_thinking_level(
                api_reasoning, model, request_purpose,
            ),
        }
    if json_mode != "off":
        generation_config["responseMimeType"] = "application/json"
        if isinstance(response_schema, dict) and response_schema:
            generation_config["responseJsonSchema"] = response_schema
    payload = {
        "contents": contents,
        "generationConfig": generation_config,
    }
    if system_instruction:
        payload["systemInstruction"] = system_instruction

    _reject_link_local_target(base_url)
    root = str(base_url or "").rstrip("/")
    if root.endswith("/models"):
        root = root[:-7]
    url = (
        root + "/models/" + urllib.parse.quote(str(model or ""), safe="")
        + ":generateContent"
    )
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    body_bytes = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    response_format_fallback = False
    try:
        try:
            response_bytes, transport_meta = _post_json_bytes(
                url, body_bytes, headers, timeout_s, max_retries=1,
            )
        except _TransportFailure as exc:
            if (
                json_mode == "auto_retry"
                and exc.status in (400, 422)
                and "responseMimeType" in generation_config
            ):
                previous_hash = str(exc.transport_meta.get("request_sha256") or "")
                generation_config.pop("responseMimeType", None)
                generation_config.pop("responseJsonSchema", None)
                response_format_fallback = True
                fallback_body = json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":"),
                ).encode("utf-8")
                response_bytes, transport_meta = _post_json_bytes(
                    url, fallback_body, headers, timeout_s, max_retries=1,
                )
                transport_meta["wire_fallback_from_sha256"] = previous_hash
            else:
                raise
    finally:
        if cleanup_name:
            deleted = _load_cloud_video().delete_gemini_file(
                cleanup_name, api_key, timeout_s=min(30, int(timeout_s)),
            )
            native_video_meta["remote_cleanup"] = "deleted" if deleted else "delete_failed"
    try:
        data = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Gemini 返回体不是合法 UTF-8 JSON；"
            f"request_sha256={transport_meta.get('request_sha256', '?')}"
        ) from exc
    try:
        candidate = data["candidates"][0]
        parts = candidate["content"]["parts"]
        content = "".join(
            str(part.get("text") or "") for part in parts
            if isinstance(part, dict) and not part.get("thought")
        )
        finish_raw = str(candidate.get("finishReason") or "STOP")
    except (KeyError, IndexError, TypeError) as exc:
        block_reason = ""
        if isinstance(data, dict) and isinstance(data.get("promptFeedback"), dict):
            block_reason = str(data["promptFeedback"].get("blockReason") or "")
        suffix = f"（blockReason={block_reason}）" if block_reason else ""
        raise RuntimeError(f"Gemini 响应缺少 candidates[0].content.parts{suffix}") from exc
    usage_raw = data.get("usageMetadata", {})
    usage_raw = usage_raw if isinstance(usage_raw, dict) else {}
    usage = {
        "prompt_tokens": usage_raw.get("promptTokenCount"),
        "completion_tokens": usage_raw.get("candidatesTokenCount"),
        "total_tokens": usage_raw.get("totalTokenCount"),
        "completion_tokens_details": {
            "reasoning_tokens": usage_raw.get("thoughtsTokenCount", 0),
        },
    }
    finish_reason = "length" if finish_raw.casefold() in {
        "max_tokens", "max_output_tokens",
    } else finish_raw.casefold()
    return _ChatCompletionText(
        content,
        finish_reason=finish_reason,
        usage=usage,
        reasoning_content="",
        response_format_fallback=response_format_fallback,
        backend="gemini_generate_content",
        transport_meta=transport_meta,
        media_meta=native_video_meta,
    )


def _claude_thinking_effort(api_reasoning: str, request_purpose: str) -> tuple[str, str]:
    """Claude 4.6/4.7 用 adaptive thinking + output_config.effort。

    返回 (thinking_type, effort)。Claude 4.7 禁止 budget_tokens / 非默认采样参数，
    统一走 thinking:{type:"adaptive"}；effort 按 api_reasoning + request_purpose 映射，
    不传 temperature（4.7 对非默认 temperature 会 400）。
    """
    mode = str(api_reasoning or "auto").strip().casefold()
    # final 是格式编译任务，默认尽量省思考；analysis 识图给中档。
    effort = "low" if request_purpose == "final" else "medium"
    if mode == "on":
        effort = "high"
    elif mode == "off":
        # Claude 可关思考：adaptive 配合 effort=low 已接近最省；off 时不传 effort 让
        # 模型默认（high）。为真正降低思考，这里仍用 adaptive + "low"（3.0 官方语义）。
        effort = "low"
    elif mode == "high":
        effort = "high"
    elif mode == "max":
        effort = "max"
    elif mode == "medium":
        effort = "medium"
    return "adaptive", effort


def _claude_media_source(value) -> dict:
    """OpenAI-style image_url data URL → Claude base64 source block."""
    data_url = value.get("url") if isinstance(value, dict) else value
    raw = str(data_url or "")
    if not raw.startswith("data:") or ";base64," not in raw:
        raise ValueError("Claude 当前只接受节点内部生成的 base64 data URL 图片")
    header, encoded = raw.split(",", 1)
    mime_type = header[5:].split(";", 1)[0].strip().casefold()
    if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise ValueError(f"Claude 不支持的图片 MIME：{mime_type or '?'}")
    if not encoded:
        raise ValueError("Claude 图片 data URL 内容为空")
    return {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": encoded}}


def _claude_messages(messages) -> tuple[str, list[dict]]:
    """OpenAI messages → Claude (system_text, content_blocks[])。"""
    system_text_parts = []
    blocks = []
    for message in messages:
        role = str(message.get("role") or "").strip().casefold()
        content = message.get("content")
        if role == "system":
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_text_parts.append(str(part.get("text") or ""))
                    elif part:
                        raise ValueError("Claude system 只允许文本内容")
            else:
                system_text_parts.append(str(content or ""))
            continue
        if role not in {"user", "assistant"}:
            raise ValueError(f"Claude 不支持的消息角色：{role or '?'}")
        if isinstance(content, str):
            blocks.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue
        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    parts.append({"type": "text", "text": str(part)})
                    continue
                part_type = str(part.get("type") or "")
                if part_type == "text":
                    parts.append({"type": "text", "text": str(part.get("text") or "")})
                elif part_type == "image_url":
                    parts.append(_claude_media_source(part.get("image_url")))
                else:
                    raise ValueError(f"Claude 不支持的内容块类型：{part_type or '?'}")
            blocks.append({"role": role, "content": parts})
            continue
        raise ValueError(f"Claude 不支持的消息 content 类型：{type(content).__name__}")
    return "\n\n".join(system_text_parts), blocks


def _call_claude_messages(base_url, api_key, model, messages, temperature,
                           max_tokens, timeout_s, json_mode="force",
                           api_reasoning="auto", response_schema=None,
                           request_purpose="final") -> _ChatCompletionText:
    """调用 Anthropic 原生 Messages API（/v1/messages）。

    - x-api-key + anthropic-version 认证。
    - 单图/多图 content blocks（base64 source）。
    - thinking:{type:"adaptive"} + output_config:{effort}（4.6/4.7；不传 temperature）。
    - 只要求 JSON 时通过 system 提示 + 本地解析（Claude 原生 output_config.format
      暂未启用，避免跨版本 400）。
    """
    system_text, claude_blocks = _claude_messages(messages)
    thinking_type, effort = _claude_thinking_effort(api_reasoning, request_purpose)
    payload = {
        "model": model,
        "max_tokens": int(max_tokens),
        "messages": claude_blocks,
        "thinking": {"type": thinking_type},
        "output_config": {"effort": effort},
    }
    if system_text:
        payload["system"] = system_text
    if json_mode != "off":
        # 不启用 output_config.format（跨版本兼容风险），靠 system 提示 + 本地 JSON 解析。
        pass
    url = "https://api.anthropic.com/v1/messages"
    if _api_host(base_url) == "api.anthropic.com" or "anthropic" in str(base_url or ""):
        root = str(base_url or "").strip().rstrip("/")
        url = root if root.endswith("/messages") else root + "/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    body_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    try:
        response_bytes, transport_meta = _post_json_bytes(
            url, body_bytes, headers, timeout_s, max_retries=1,
        )
    except _TransportFailure as exc:
        raise
    try:
        data = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Claude 返回体不是合法 UTF-8 JSON；"
            f"request_sha256={transport_meta.get('request_sha256', '?')}"
        ) from exc
    try:
        content_blocks = data["content"]
        content = "".join(
            str(block.get("text") or "") for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        finish_raw = str(data.get("stop_reason") or "end_turn")
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Claude 响应缺少 content：{str(data)[:200]}") from exc
    usage = data.get("usage", {})
    usage = usage if isinstance(usage, dict) else {}
    finish_reason = "length" if finish_raw.casefold() in {"max_tokens", "length"} else finish_raw.casefold()
    return _ChatCompletionText(
        content,
        finish_reason=finish_reason,
        usage={
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "total_tokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
            "completion_tokens_details": {
                "reasoning_tokens": usage.get("output_tokens", 0),
            },
        },
        reasoning_content="",
        backend="claude_messages",
        transport_meta=transport_meta,
    )


def _call_chat(base_url, api_key, model, messages, temperature, max_tokens, timeout_s,
               json_mode="auto_retry", api_reasoning="auto", response_schema=None,
               request_purpose="final") -> str:
    """调用 OpenAI 兼容 chat/completions，返回带诊断属性的 str 子类。

    - json_mode: off=不带 response_format；force=带 json_object；auto_retry=带
      json_object，遇 400/422 降级为无结构重试一次（服务器不支持时）
    - api_reasoning: 本地兼容服务用 chat_template_kwargs；DeepSeek 官方接口改用
      顶层 thinking + reasoning_effort，避免把本地模板参数发给云端
    - response_schema: LM Studio/OpenAI Structured Output JSON Schema；服务器拒绝时
      auto_retry 会去掉 response_format 再试一次
    """
    if _is_gemini_api(base_url):
        return _call_gemini_native(
            base_url, api_key, model, messages, temperature, max_tokens, timeout_s,
            json_mode=json_mode, api_reasoning=api_reasoning,
            response_schema=response_schema, request_purpose=request_purpose,
        )
    if _is_anthropic_api(base_url):
        # Claude 原生 Messages（x-api-key + anthropic-version）；不传 temperature。
        return _call_claude_messages(
            base_url, api_key, model, messages, temperature, max_tokens, timeout_s,
            json_mode=json_mode, api_reasoning=api_reasoning,
            response_schema=response_schema, request_purpose=request_purpose,
        )
    is_deepseek = _is_deepseek_api(base_url)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode != "off":
        # DeepSeek Chat 当前使用 json_object；OpenAI 式 json_schema 会造成偶发 400/422。
        # schema 仍完整写在 system，并由本地 parser/compiler/validator 做双层校验。
        if isinstance(response_schema, dict) and response_schema and not is_deepseek:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "minimax_h3_prompt_ir",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}
    reasoning_mode = _resolved_reasoning_mode(
        api_reasoning, model, request_purpose=request_purpose,
    )
    if _is_local_lmstudio_url(base_url) and json_mode != "force":
        native_reasoning = "off" if reasoning_mode == "off" else "on"
        try:
            return _call_lmstudio_native(
                base_url, api_key, model, messages, temperature,
                max_tokens, timeout_s, reasoning_mode=native_reasoning,
            )
        except Exception as exc:  # noqa: BLE001
            # 旧版/非 LM Studio 本地服务没有原生端点时仍保留兼容路径。
            LOGGER.warning(
                "MiniMax H3 PromptDirector: LM Studio 原生/JIT 调用失败，"
                "回退 OpenAI 兼容接口（reasoning=%s）: %s",
                native_reasoning, exc,
            )
    if is_deepseek:
        if reasoning_mode == "off":
            payload["thinking"] = {"type": "disabled"}
        else:
            payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = "low" if reasoning_mode == "auto" else "high"
            payload.pop("temperature", None)
    elif _is_zhipu_api(base_url):
        # GLM(智谱)：官方 thinking.type 仅支持 enabled，不能关闭思考（视觉示例明确
        # 用 thinking:{type:"enabled"}）。统一注入 enabled，不因 off 关闭、不误发
        # reasoning_effort（GLM 不接受该字段）。
        payload["thinking"] = {"type": "enabled"}
        # 官方推荐 temperature=1 / top_p=0.95；此处保留用户显式设置，不强制覆盖。
    elif _is_local_lmstudio_url(base_url) or _is_qwen38(model):
        # 本地 LM Studio / Qwen3.8：保留 chat_template_kwargs + reasoning_effort=none
        # + /no_think 三通道（本机实测模板变量可能被接受但不生效）。
        if reasoning_mode == "off":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload["reasoning_effort"] = "none"
            if _is_qwen38(model):
                payload["messages"] = _qwen_no_think_messages(messages)
                payload.update({"top_p": 0.8, "top_k": 20, "presence_penalty": 1.5})
        elif reasoning_mode == "on":
            payload["chat_template_kwargs"] = {"enable_thinking": True}
    else:
        # 云端 OpenAI 兼容（豆包方舟 / GLM / MiniMax / 自定义）：统一用顶层
        # reasoning_effort（比 chat_template_kwargs 更通用、各服务接受度更好）。
        # off→none；其它→high（auto 默认）。auto_retry 时遇 400/422 可降级去掉。
        if reasoning_mode == "off":
            payload["reasoning_effort"] = "none"
        elif reasoning_mode in ("on", "auto"):
            payload["reasoning_effort"] = "high"

    _reject_link_local_target(base_url)
    url = _chat_completions_url(base_url)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body_bytes = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    response_format_fallback = False
    try:
        response_bytes, transport_meta = _post_json_bytes(
            url, body_bytes, headers, timeout_s,
            max_retries=1 if is_deepseek else 0,
        )
    except _TransportFailure as exc:
        if exc.status not in (400, 422):
            raise
        # 服务器拒绝某个"可降级"控制字段时，逐项去掉它重发一次，避免把控制参数
        # 静默遗留给服务端默认值（例如模型不支持 reasoning_effort / json_schema）。
        # 阶段 1：去掉 reasoning_effort（保留 JSON 结构）；阶段 2：仅 auto_retry 下
        # 再去掉 response_format 降级为无结构输出（force 保留结构化，不降级）。
        prov_dropped = []
        retried_payload = dict(payload)
        response_format_fallback = False
        original_hash = str(exc.transport_meta.get("request_sha256") or "")
        fallback_error = exc
        fallback_succeeded = False
        for attempt in range(2):
            if "reasoning_effort" in retried_payload:
                retried_payload.pop("reasoning_effort", None)
                prov_dropped.append("reasoning_effort")
            elif json_mode == "auto_retry" and "response_format" in retried_payload:
                retried_payload.pop("response_format", None)
                response_format_fallback = True
                prov_dropped.append("response_format")
            else:
                raise fallback_error
            fallback_body = json.dumps(
                retried_payload, ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")
            try:
                response_bytes, transport_meta = _post_json_bytes(
                    url, fallback_body, headers, timeout_s,
                    max_retries=1 if is_deepseek else 0,
                )
            except _TransportFailure as inner_exc:
                if inner_exc.status not in (400, 422):
                    raise
                fallback_error = inner_exc
                continue
            transport_meta["wire_fallback_from_sha256"] = original_hash
            transport_meta["wire_fallback_dropped"] = list(prov_dropped)
            fallback_succeeded = True
            break
        if not fallback_succeeded:
            raise fallback_error
    try:
        data = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "API 返回体不是合法 UTF-8 JSON；"
            f"request_sha256={transport_meta.get('request_sha256', '?')}"
        ) from exc
    try:
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content")
        if content is None:
            content = ""
        reasoning_content = message.get("reasoning_content", message.get("reasoning", ""))
        return _ChatCompletionText(
            content,
            finish_reason=choice.get("finish_reason", ""),
            usage=data.get("usage", {}),
            reasoning_content=reasoning_content,
            response_format_fallback=response_format_fallback,
            backend="deepseek_chat" if is_deepseek else "openai_compat",
            transport_meta=transport_meta,
        )
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


def _user_media_reference_notes(prompt: str, picture_count: int, has_video: bool) -> list[str]:
    """检查用户提示词里的官方素材标签；只生成提醒，不改写或阻断请求。"""
    text = str(prompt or "")
    try:
        picture_total = max(0, int(picture_count))
    except (TypeError, ValueError):
        picture_total = 0
    picture_refs = sorted({
        int(value)
        for value in re.findall(r"<Picture\s+(\d+)>", text, flags=re.IGNORECASE)
    })
    video_refs = sorted({
        int(value)
        for value in re.findall(r"<Video\s+(\d+)>", text, flags=re.IGNORECASE)
    })

    notes = []
    invalid_pictures = [value for value in picture_refs if value < 1 or value > picture_total]
    if invalid_pictures:
        labels = "、".join(f"<Picture {value}>" for value in invalid_pictures)
        notes.append(
            f"[素材引用提醒] 提示词引用了当前不存在的 {labels}；"
            f"本次实际连接参考图 {picture_total} 张。"
        )
    missing_pictures = [
        value for value in range(1, picture_total + 1) if value not in picture_refs
    ]
    if missing_pictures:
        labels = "、".join(f"<Picture {value}>" for value in missing_pictures)
        notes.append(
            f"[素材引用提醒] 已连接但未显式引用 {labels}；素材仍会发送给提示词模型。"
        )

    invalid_videos = [value for value in video_refs if value != 1 or not has_video]
    if invalid_videos:
        labels = "、".join(f"<Video {value}>" for value in invalid_videos)
        notes.append(
            f"[素材引用提醒] {labels} 与当前视频输入不匹配；"
            "本节点的参考视频统一对应 <Video 1>。"
        )
    if has_video and 1 not in video_refs:
        notes.append(
            "[素材引用提醒] 已连接参考视频但提示词未显式写 <Video 1>；"
            "视频仍会发送给提示词模型。"
        )
    return notes


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
                request_purpose="analysis",
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


def _analyze_video_sequence(base_url, key, model, sequence_frames, sequence_total,
                            temperature, max_tokens, timeout_s, json_mode,
                            api_reasoning):
    """分阶段视觉分析：把同一视频的全部已选代表帧作为一个有序整体分析。

    成功返回 ``(summary_dict, notes)``；失败返回 ``({}, notes)``，调用方会把
    所有已选帧原样放入最终多模态请求，保证不静默丢失视频证据。
    """
    if not sequence_frames:
        return {}, []
    user_content = [{
        "type": "text",
        "text": (
            f"联合分析以下 {len(sequence_frames)} 张 <Video 1> 代表帧。"
            f"原输入序列共 {sequence_total} 帧；帧后的相对位置只表示抽样位置，不是精确时间码。"
        ),
    }]
    for ordinal, (source_index, data_url) in enumerate(sequence_frames, start=1):
        position = 0.0 if sequence_total <= 1 else source_index / (sequence_total - 1)
        user_content.append({
            "type": "text",
            "text": (
                f"<Video 1> 代表帧 {ordinal}/{len(sequence_frames)}："
                f"输入序列索引 {source_index}/{sequence_total - 1}，约位于时间线 {position:.0%}。"
            ),
        })
        user_content.append({"type": "image_url", "image_url": {"url": data_url}})
    sequence_tokens = max(768, min(int(max_tokens or 2048), 3072))
    try:
        summary = _parse_json_text(_call_chat(
            base_url, key, model,
            [{"role": "system", "content": _VIDEO_SEQUENCE_SYSTEM_TEMPLATE},
             {"role": "user", "content": user_content}],
            temperature, sequence_tokens, timeout_s,
            json_mode=json_mode, api_reasoning=api_reasoning,
            request_purpose="analysis",
        ))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("MiniMax H3 PromptDirector: 视频序列分析失败——回退最终调用直传全部选帧: %s", exc)
        return {}, [f"视频序列联合分析失败，最终调用直传全部 {len(sequence_frames)} 帧：{exc}"]
    if not isinstance(summary, dict):
        return {}, [f"视频序列分析输出非对象，最终调用直传全部 {len(sequence_frames)} 帧"]
    summary["sequence_label"] = "<Video 1>"
    summary["input_frame_count"] = sequence_total
    summary["selected_indices"] = [idx for idx, _ in sequence_frames]
    return summary, [
        f"视频序列联合分析 {len(sequence_frames)}/{sequence_total} 张（单次整体识别，≤{sequence_tokens} token）"
    ]


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
                # Qwen3.8 官方把最终响应预算建议开放到 131072；实际可用值仍受
                # LM Studio 加载上下文长度与输入 token 共同约束。
                "max_tokens": ("INT", {"default": 8192, "min": 256, "max": 131072, "step": 1024}),
                "timeout_s": ("INT", {"default": 180, "min": 10, "max": 1800, "step": 10}),
                "lmstudio_after_use": (["keep_loaded", "unload_used_model", "unload_and_wait_for_vram"], {"default": "keep_loaded"}),
                "lmstudio_gpu_offload": (["max", "0.90", "0.75", "0.50", "auto", "off"], {"default": "auto"}),
                # 新 widget 一律追加在末尾（ComfyUI 按位置恢复旧工作流值）
                "api_reasoning": (["auto", "off", "on"], {"default": "auto"}),
                "json_mode": (["auto_retry", "force", "off"], {"default": "auto_retry"}),
                # v0.2：视觉分析模式（两阶段=逐素材事实抽取+合并；单次=原多图直传）
                # v0.1：Auto=0-2 图单次、3-9 图分阶段；旧值 two_stage/single_pass 兼容映射
                "analysis_mode": (["auto", "single", "staged"], {"default": "auto"}),
                # v0.3：模型兼容 profile（auto=按模型名推断；gemma/qwen/cloud 手动指定）
                "model_profile": (_MODEL_PROFILES, {"default": "auto"}),
                # 新 widget 继续追加在末尾：长视频 batch 超限时包含首尾地均匀取样
                "frame_sequence_limit": ("INT", {
                    "default": 4, "min": 1, "max": 300, "step": 1,
                    "tooltip": "仅直连 IMAGE 批次时生效；连接时间线 manifest 后由上游控制",
                }),
                "frame_selection_mode": (_FRAME_SELECTION_MODES, {
                    "default": "uniform_full",
                    "tooltip": "仅直连 IMAGE 批次时生效；连接时间线 manifest 后由上游控制",
                }),
                "frame_selection_spec": (
                    "STRING",
                    {"default": "", "placeholder": "仅直连：0,41,82,-1 或 0,33,66,100"},
                ),
                # 默认阻止 API/JSON/IR 失败时把原提示词静默送进昂贵的视频生成链。
                "api_failure_policy": (["stop", "passthrough"], {"default": "stop"}),
            },
            "optional": {
                # 可选而非必填：旧工作流/旧浏览器不会序列化这个新增字段，
                # 后端签名会确定性回退 auto；新节点仍显示同一个普通 widget。
                "reference_fidelity": (
                    list(_REFERENCE_FIDELITY_MODES), {"default": "auto"},
                ),
                **{
                    f"ref_image_{i}": ("IMAGE", {"label": f"导演识图素材 {i}（不传给 H3）"})
                    for i in range(1, 10)
                },
                # 一个 IMAGE socket 可以接 VHS_LoadVideo / VHS_SelectImages / 图像 batch。
                # 与 ref_image_N 分流：整批共同描述 <Video 1>，不密集编号为 Picture。
                "video_frame_sequence": (
                    "IMAGE",
                    {"label": "参考视频抽帧 / 图像序列（批次；仅供 API，不传给 H3）"},
                ),
                "video_timeline_manifest": (
                    "STRING",
                    {"multiline": True, "default": "", "forceInput": True,
                     "label": "Video Context 时间线 manifest（可选）"},
                ),
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
               lmstudio_gpu_offload="auto", api_reasoning="auto", json_mode="auto_retry",
               model_profile="auto", frame_sequence_limit=4,
               frame_selection_mode="uniform_full", frame_selection_spec="",
               api_failure_policy="stop", reference_fidelity="auto", **kwargs):
        t0 = time.time()
        # CloudDirector can inject a credential resolved by the backend. These
        # private kwargs are never ComfyUI inputs and therefore never serialize.
        resolved_api_key = str(kwargs.pop("_resolved_api_key", "") or "").strip()
        resolved_api_key_source = str(
            kwargs.pop("_resolved_api_key_source", "") or "cloud_credential"
        ).strip()
        cloud_connection_note = str(
            kwargs.pop("_cloud_connection_note", "") or ""
        ).strip()
        native_video_source = kwargs.pop("_cloud_video", None)
        gemini_video_route = str(kwargs.pop("_gemini_video_route", "auto") or "auto")
        try:
            gemini_video_fps = float(kwargs.pop("_gemini_video_fps", 0.0) or 0.0)
        except (TypeError, ValueError):
            gemini_video_fps = 0.0
        lm_notes = []
        module_notes = []
        runtime_notes = []
        if cloud_connection_note:
            runtime_notes.append(cloud_connection_note)
        module_ids = []
        is_deepseek = _is_deepseek_api(api_base_url)
        is_gemini = _is_gemini_api(api_base_url)
        is_volcengine = _is_volcengine_api(api_base_url)
        is_anthropic = _is_anthropic_api(api_base_url)
        is_zhipu = _is_zhipu_api(api_base_url)
        is_minimax = _is_minimax_api(api_base_url)
        if native_video_source is not None and not is_gemini:
            raise RuntimeError(
                "云端原生视频输入目前只适配 Gemini；DeepSeek 边界测试请连接 IMAGE 帧批次"
            )
        max_tokens, budget_note = _normalize_prompt_ir_budget(max_tokens)
        if budget_note:
            runtime_notes.append(budget_note)
        if is_deepseek:
            runtime_notes.append(
                f"[DeepSeek 预算] 供应商输出能力上限为 {_DEEPSEEK_PROVIDER_MAX_OUTPUT_TOKENS} token；"
                f"本次 Prompt IR 独立预算为 {max_tokens} token。"
            )
        if is_gemini:
            if max_tokens > _GEMINI_PROVIDER_MAX_OUTPUT_TOKENS:
                runtime_notes.append(
                    f"[Gemini 预算] max_tokens={max_tokens} 超过当前预设模型输出能力，"
                    f"已收敛到 {_GEMINI_PROVIDER_MAX_OUTPUT_TOKENS}。"
                )
                max_tokens = _GEMINI_PROVIDER_MAX_OUTPUT_TOKENS
            runtime_notes.append(
                f"[Gemini 预算] 本次 Prompt IR={max_tokens} token；inline 媒体项目预算"
                f"≈{_GEMINI_INLINE_MEDIA_BUDGET / 1024 / 1024:.0f} MiB（官方总请求 20MB）。"
            )

        manifest_text = str(kwargs.get("module_manifest") or "").strip()
        if manifest_text:
            try:
                manifest = json.loads(manifest_text)
                if not isinstance(manifest, dict):
                    raise ValueError("顶层不是 JSON 对象")
                # v1.1 manifest 区分“用户选择”与“Resolver 最终应用”；旧 v1.0 继续回退 selected。
                applied_items = manifest.get("resolved")
                if not isinstance(applied_items, list):
                    applied_items = manifest.get("selected", [])
                module_ids = [
                    str(item.get("id")) for item in applied_items
                    if isinstance(item, dict) and item.get("id")
                ]
                render_profile = str(manifest.get("render_profile") or "standard")
                module_set_sha256 = str(manifest.get("module_set_sha256") or "").strip()
                rendered_system_sha256 = str(manifest.get("rendered_system_sha256") or "").strip()
                module_notes.append(
                    "[模块渲染] "
                    f"档位={render_profile}；"
                    f"模块集合={module_set_sha256 or '旧清单未记录'}；"
                    f"规则文本={rendered_system_sha256 or '旧清单未记录'}"
                )
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
            if runtime_notes:
                report += "\n" + "\n".join(runtime_notes)
            if module_notes:
                report += "\n" + "\n".join(module_notes)
            return (prompt, report,
                    reference_sheet, prompt_ir)

        def fail_or_passthrough(reason: str, api_called: bool = False,
                                reference_sheet: str = "[]", prompt_ir: str = "{}"):
            """默认停止下游；legacy passthrough 仅在用户显式选择时保留。"""
            if str(api_failure_policy or "stop").strip().lower() != "passthrough":
                message = (
                    "MiniMax H3 PromptDirector 已停止工作流：" + reason
                    + "。未把原始提示词直通给 H3；如确需旧行为，请把 API 失败策略改为 passthrough。"
                )
                LOGGER.error(message)
                raise RuntimeError(message)
            return passthrough(
                reason, api_called=api_called,
                reference_sheet=reference_sheet, prompt_ir=prompt_ir,
            )
        media_prepare_started = time.perf_counter()
        # 收集参考图：按已连接素材密集编号，确保与官方动态端口的展示顺序一致。
        images = []
        image_inputs = []
        port_map = []
        for i in range(1, 10):
            img = kwargs.get(f"ref_image_{i}")
            if img is not None:
                ordinal = len(image_inputs) + 1
                image_inputs.append((ordinal, img))
                port_map.append((i, ordinal))
        picture_limits = {"T2VA": 0, "I2VA": 1, "FL2VA": 2, "L2VA": 1}
        picture_limit = picture_limits.get(str(task_type or "").upper())
        if picture_limit is not None and len(image_inputs) > picture_limit:
            return fail_or_passthrough(
                f"{task_type} 最多允许 {picture_limit} 个 Picture 锚点，"
                f"但导演接入了 {len(image_inputs)} 张参考图；已在调用提示词 API 前停止。"
                "多参考图/参考视频编辑请显式选择 Ref2VA，并确认下游使用 "
                "MiniMaxH3ReferenceToVideo；不要依赖节点自动更改任务类型"
            )
        if is_gemini and image_inputs:
            # Preserve identity detail when possible, but keep reference images
            # from consuming the entire 20 MB inline request before video frames.
            for max_side, quality in ((1024, 85), (768, 80), (512, 72)):
                images = [
                    (ordinal, _image_tensor_to_data_url(img, max_side=max_side, quality=quality))
                    for ordinal, img in image_inputs
                ]
                if sum(len(url) for _, url in images) <= 6 * 1024 * 1024:
                    break
            runtime_notes.append(
                f"[Gemini 媒体] 参考图 {len(images)} 张 Data URL≈"
                f"{sum(len(url) for _, url in images) / 1024 / 1024:.1f} MiB。"
            )
        else:
            images = [
                (ordinal, _image_tensor_to_data_url(img)) for ordinal, img in image_inputs
            ]

        # 独立收集参考视频抽帧：它们共同对应 <Video 1>，不能占用 <Picture N> 标签。
        sequence_frames = []
        sequence_total = 0
        sequence_input_total = 0
        sequence_selection_note = ""
        timeline_manifest = {}
        sequence_input = kwargs.get("video_frame_sequence")
        timeline_text = str(kwargs.get("video_timeline_manifest") or "").strip()
        if native_video_source is not None and sequence_input is not None:
            raise RuntimeError(
                "Gemini 原生视频与参考视频 IMAGE 帧批次不能同时连接；请断开其中一路后做 A/B"
            )
        if sequence_input is not None:
            try:
                requested_frame_limit = min(600, max(1, int(frame_sequence_limit or 1)))
                if not isinstance(sequence_input, torch.Tensor) or sequence_input.ndim not in (3, 4):
                    raise TypeError("video_frame_sequence 必须是 ComfyUI IMAGE 张量")
                sequence_input_total = 1 if sequence_input.ndim == 3 else int(sequence_input.shape[0])
                timeline_controls_selection = False
                if timeline_text:
                    timeline_manifest = _parse_timeline_manifest(
                        timeline_text, input_frame_count=sequence_input_total,
                    )
                    timeline_controls_selection = True

                if timeline_controls_selection:
                    # Video Context 的 selected_frames 与 manifest 是一个不可拆分的素材包。
                    # 上游已经决定代表帧；导演只允许按传输预算确定性降载，避免第二套人工
                    # 选帧参数让用户无法判断哪一处真正生效。
                    provider_cap = 600 if is_deepseek else 300
                    effective_frame_limit = min(sequence_input_total, provider_cap)
                    effective_selection_mode = "uniform_full"
                    effective_selection_spec = ""
                    if sequence_input_total > provider_cap:
                        runtime_notes.append(
                            f"[传输能力上限] 时间线上游提供 {sequence_input_total} 帧；"
                            f"当前连接最多接收 {provider_cap} 帧，已均匀降载。"
                        )
                else:
                    effective_frame_limit = requested_frame_limit
                    effective_selection_mode = frame_selection_mode
                    effective_selection_spec = frame_selection_spec
                    if requested_frame_limit > 300 and not is_deepseek:
                        effective_frame_limit = 300
                        runtime_notes.append(
                            f"[帧数上限] {requested_frame_limit} 是 DeepSeek 边界实验档；"
                            "当前连接不是官方 DeepSeek，已收敛到 300。"
                        )
                sequence_frames, sequence_total, sequence_selection_note = _image_batch_to_data_urls(
                    sequence_input, limit=effective_frame_limit,
                    selection_mode=effective_selection_mode,
                    selection_spec=effective_selection_spec,
                    max_payload_bytes=(
                        _DEEPSEEK_SEQUENCE_PAYLOAD_BUDGET if is_deepseek else (
                            max(
                                1 * 1024 * 1024,
                                _GEMINI_INLINE_MEDIA_BUDGET
                                - sum(len(url) for _, url in images),
                            ) if is_gemini else 0
                        )
                    ),
                    adaptive=(is_deepseek or is_gemini),
                )
                if timeline_controls_selection:
                    note_tail = sequence_selection_note.partition("；")[2]
                    sequence_selection_note = (
                        f"时间线上游已选 {sequence_input_total} 帧，导演未二次选帧"
                        + (f"；{note_tail}" if note_tail else "")
                    )
                    remapped = []
                    for local_index, data_url in sequence_frames:
                        remapped.append((
                            int(timeline_manifest["frames"][local_index]["frame"]), data_url,
                        ))
                    sequence_frames = remapped
                    sequence_total = int(timeline_manifest["total_frames"])
                    runtime_notes.append(
                        f"[参考素材时间线] 已恢复源视频 fps={timeline_manifest['fps']:g}、"
                        f"时长={timeline_manifest['duration_seconds']:.3f}s、"
                        f"源帧总数={sequence_total}；上游代表帧选择优先，"
                        "导演节点的帧数、选帧方式与自定义选帧参数本次不生效。"
                    )
                if is_deepseek and requested_frame_limit > 300 and not timeline_controls_selection:
                    runtime_notes.append(
                        f"[DeepSeek 600 帧边界实验] 请求上限={requested_frame_limit}，"
                        f"实际编码并准备发送={len(sequence_frames)}；"
                        "最终以报告中的 payload、prompt token 与 transport 耗时为准。"
                    )
                if len(sequence_frames) > 120:
                    runtime_notes.append(
                        f"[高密度时间线提醒] 本次准备发送 {len(sequence_frames)} 帧；"
                        "这属于长多图时序压力档，不应把接口接收成功等同于动作顺序理解稳定。"
                        "正式回归优先使用 24/48 帧并与 staged 做 A/B。"
                    )
                if sequence_selection_note.startswith("[选择回退]"):
                    module_notes.append(f"[序列帧警告] {sequence_selection_note}")
            except (TypeError, ValueError) as exc:
                module_notes.append(f"[序列帧警告] 无法读取参考视频抽帧：{exc}")
        elif timeline_text:
            try:
                timeline_manifest = _parse_timeline_manifest(timeline_text)
                runtime_notes.append(
                    f"[时间线 manifest] 已接入原生视频辅助时间轴：fps={timeline_manifest['fps']:g}，"
                    f"时长={timeline_manifest['duration_seconds']:.3f}s，"
                    f"切镜段={len(timeline_manifest['segments'])}。"
                )
            except ValueError as exc:
                module_notes.append(f"[时间线警告] 已忽略无效 manifest：{exc}")
        media_prepare_ms = round((time.perf_counter() - media_prepare_started) * 1000)
        runtime_notes.append(f"[本地媒体预处理] {media_prepare_ms / 1000:.2f}s。")
        module_notes.extend(_user_media_reference_notes(
            prompt,
            len(images),
            bool(sequence_input is not None or native_video_source is not None),
        ))

        if len(images) > 1 and task_type != "Ref2VA":
            module_notes.append(
                f"[接线警告] 导演接入 {len(images)} 张分析图但任务={task_type}。这些图只供 LLM 识图；"
                "若下游使用 MiniMaxH3ReferenceToVideo，应把任务改为 Ref2VA。"
            )
        if sequence_frames and task_type != "Ref2VA":
            module_notes.append(
                f"[序列帧提醒] 已向提示词 API 发送 {len(sequence_frames)} 张 <Video 1> 时间线帧，"
                f"但导演任务={task_type}；若下游把完整参考视频接 ReferenceToVideo，请改为 Ref2VA。"
            )

        # v0.2：素材角色/别名参数已按用户反馈移除（自然语言描述即可）
        # v0.1：analysis_mode 解析（auto = 0-2 图 single、3-9 图 staged；旧值兼容映射）
        mode_raw = str(kwargs.get("analysis_mode") or "auto").strip().lower()
        if mode_raw in ("two_stage", "staged"):
            staged = True
        elif mode_raw in ("single_pass", "single"):
            staged = False
        else:  # auto
            # DeepSeek 大量连续帧 auto 默认同次联合理解，避免阶段摘要被误当成新开场；
            # 用户仍可显式选择 staged 做同素材 A/B。本地模型沿用原分阶段策略。
            staged = (len(images) >= 3 or bool(sequence_frames)) and not (
                (is_deepseek or is_gemini) and (bool(sequence_frames) or native_video_source is not None)
            )
            if (is_deepseek or is_gemini) and (sequence_frames or native_video_source is not None):
                provider_name = "DeepSeek" if is_deepseek else "Gemini"
                media_name = "<Video 1> 原生视频" if native_video_source is not None else "<Video 1> 连续帧"
                runtime_notes.append(
                    f"[{provider_name} 视频] auto 已选择 single：目标参考图与 {media_name} 同次发送；"
                    "可显式改为 staged 做 A/B。"
                )

        # host 严格匹配的环境变量优先于通用兜底；任何诊断都只写来源，不写 secret。
        if resolved_api_key:
            key, key_source = resolved_api_key, resolved_api_key_source
        else:
            key, key_source = _resolve_api_key(api_base_url, api_key)
        if key_source.startswith("节点 widget"):
            module_notes.append(
                "[密钥警告] 当前 API Key 来自节点输入，可能进入 workflow/history/输出元数据；"
                "DeepSeek 请改用 DEEPSEEK_API_KEY 环境变量，并在外传前扫描/剥离元数据。"
            )
        if not key:
            LOGGER.info("MiniMax H3 PromptDirector: 未提供 API Key（按本地服务处理，LM Studio 等忽略鉴权）")

        # 先解析真实模型，再选择模型 profile；旧实现用 api_model=auto 推断会误落 cloud。
        model = api_model.strip()
        models = []
        if not model or model == "auto":
            models = _list_models(api_base_url, key, timeout_s=min(timeout_s, 15))
            if len(models) == 1:
                model = models[0]
            elif not models:
                return fail_or_passthrough(
                    f"无法从 {api_base_url} 获取模型列表，且 api_model=auto"
                    "（请确认 API 地址/密钥后点击刷新模型列表）"
                )
            else:
                model = models[0]
                LOGGER.warning("MiniMax H3 PromptDirector: 多个候选模型 %s，取第一个 %s", models, model)

        shots = shot_count if shot_count and shot_count > 0 else max(1, round(duration_seconds / 2.5))
        resolved_profile = _infer_model_profile(model_profile, model)
        if str(model_profile or "auto").strip().lower() == "auto" and is_deepseek:
            resolved_profile = "deepseek_vision"
        if str(model_profile or "auto").strip().lower() == "auto" and is_gemini:
            resolved_profile = "gemini_vision"
        if str(model_profile or "auto").strip().lower() == "auto" and is_anthropic:
            resolved_profile = "claude_vision"
        if str(model_profile or "auto").strip().lower() == "auto" and (
            is_volcengine or is_zhipu or is_minimax
        ):
            # 自定义 OpenAI 兼容（custom）host 无法识别，其 profile 由 preset 显式指定。
            resolved_profile = "openai_compat_vision"
        compat = _MODEL_COMPAT[resolved_profile]
        has_video_evidence = bool(sequence_frames or native_video_source is not None)
        effective_fidelity, reference_fidelity_rule = _resolve_reference_fidelity(
            reference_fidelity, task_type, has_video_evidence,
        )
        media_inventory = _build_media_inventory(
            prompt, len(images), has_video_evidence,
        )
        media_inventory_text = _format_media_inventory(media_inventory)
        ir_discipline = (
            "镜头标签必须按出现顺序从 [Shot 1] 连续递增到 [Shot N]，不得从 2 开始或跳号。"
            "只有参考视频发生真实切镜时才开始新的 [Shot N]；同一连续镜头中的动作阶段继续写在"
            "同一个 Shot 内，不得为了分段说明而虚构切镜。人物进出画、动作或姿势改变、运镜、"
            "遮挡、闪白和景深变化本身都不等于切镜；只有前后画面构图发生不连续跳变才算真实切镜。"
            f"本次可用或由用户明确声明的媒体清单：{media_inventory_text}；"
            "不得创建清单外的 <Picture N>/<Video N>/<Audio N>。同一有效媒体标签可以在多个官方"
            "字段中重复引用，重复引用不是新增素材。I2VA 只能使用 <Picture 1>；多参考图任务使用"
            "Ref2VA。任何模块或改写模式与官方 IR 契约冲突时，以官方 IR 契约和本清单为准。"
        )
        if media_inventory.get("audio") == 0:
            ir_discipline += (
                "本次没有独立 <Audio N> 素材：不得声称复用、保留或原样采用源视频的人声、台词、"
                "歌声、音乐或音色；不得从嘴型、画面文字或字幕推断并抄写台词、歌词或音乐。只有"
                "用户在本次文本中明确给出的台词或配乐要求才可采用；否则声音字段只写最少量、由"
                "H3 新生成且与可见动作一致的环境声/动作声，non_diegetic_music 写 N/A。"
            )
        # 防幻觉条款：按实际媒体证据状态声明（Easy 插件 nodes.py:753-764 先例改编）
        evidence_parts = []
        if images:
            evidence_parts.append(
                f"本次请求已附带 {len(images)} 张独立参考图（多模态附件或分析摘要），"
                "它们按顺序对应 <Picture 1..N>。"
            )
        if sequence_frames:
            evidence_parts.append(
                f"本次另附 {len(sequence_frames)} 张按时间顺序排列的参考视频代表帧；"
                "它们共同对应完整的 <Video 1>，不是新的 <Picture N>，不得给这些帧另编 Picture 标签。"
                "仅用它们识别源视频中的动作、姿势顺序、场景、镜头和需要被替换的源表演者；"
                "除非用户明确要求保留，否则不得把源表演者的脸、发型、身体身份或服装写成目标主体外观。"
            )
            if len(sequence_frames) > 50:
                evidence_parts.append(
                    "这是高密度连续取帧：所有帧仍来自同一个 <Video 1>。必须先按时间聚合为少数"
                    "动作阶段与真实切镜点；相邻近似帧不是独立分镜，禁止逐帧复述、禁止新增或"
                    "重绘开头，也禁止把后续任意帧改成新的首镜。"
                )
        if native_video_source is not None:
            evidence_parts.append(
                "本次请求附带一个 Gemini 原生 <Video 1> 文件；它是一条连续视频时间线，"
                "包含原文件中的画面、时间顺序与音频（若源文件具有音轨），不是独立图片集合。"
                "从该视频提取动作、姿势顺序、场景、镜头和待替换源表演者的位置；"
                "可观察声音只用于理解表演节奏，是否能在最终提示词中引用仍以媒体清单为准；"
                "除非用户明确要求，否则不得把源表演者身份或服装写入目标主体。"
            )
        if has_video_evidence:
            if timeline_manifest:
                source_duration = float(timeline_manifest.get("duration_seconds") or 0.0)
                evidence_parts.append(
                    f"时间线 manifest 指明 <Video 1> 源片段时长为 {source_duration:.3f}s，"
                    f"目标成片时长为 {float(duration_seconds):.3f}s；按归一化进度映射并保持阶段顺序，"
                    "不得把目标时长误写成源片段时长。"
                )
                if source_duration > float(duration_seconds) * 1.5:
                    runtime_notes.append(
                        f"[时长映射提醒] <Video 1> 源片段 {source_duration:.3f}s / "
                        f"目标 {float(duration_seconds):.3f}s；动作会被明显压缩，"
                        "逐时点跟随回归应先裁成与目标相同的时长。"
                    )
            else:
                evidence_parts.append(
                    "本次未提供 video_timeline_manifest，无法知道源片段的 fps 与真实时长；"
                    "不得声称源片段等于目标时长，也不得虚构源视频的精确秒点。"
                )
        if task_type == "Ref2VA" and images and (sequence_frames or native_video_source is not None):
            evidence_parts.append(
                "若用户意图是人物替换或视频动作迁移：必须在 subject_definitions 中用可观察到的"
                "发型、脸部/眼睛、体型、服装和 3–5 个身份关键特征，明确把目标 <Subject N> 绑定到"
                "相应 <Picture N>；同时从 <Video 1> 摘要中只采用一条最短 source_performer_locator"
                "定位待替换对象，并立即声明该源人物外观不被保留。不得用泛称‘动漫女孩/一个人’"
                "代替目标外观描述。目标参考图与用户明确文字对身份、服装、身体部件数量、所在侧和"
                "连接关系拥有最高权威；明确的单侧或其他非对称特征必须保持，不得镜像、补齐或"
                "对称化。源视频只迁移动作、姿势、位移、镜头和时序。source_performer_locator 只能"
                "使用人数、画面位置或正在进行的动作（例如‘画面中央唯一的表演者’），不得描述其"
                "发色、脸、服装、翅膀、肢体等外观；随后只写‘源人物外观不保留’，不得在最终 "
                "Prompt IR 的任何字段逐项枚举已丢弃的源人物外观细节。"
            )
        if evidence_parts:
            media_evidence = (
                "".join(evidence_parts)
                + "只使用附件中可直接观察到的细节，或本次请求提供的分析摘要中记载的事实；"
                  "两者之外的内容不得虚构或自信描述。"
            )
        else:
            media_evidence = (
                "本次请求未附带任何参考媒体文件。不要虚构任何图片/视频/音频参考的内容；"
                "如需引用标签请保留，但仅从用户提示词与明确指令推断，绝不臆造主体/外观/动作/场景/声音细节。"
            )
        system = _H3_SYSTEM_TEMPLATE.format(
            duration=duration_seconds,
            shots=shots,
            task_rule=_TASK_RULES[task_type],
            rewrite_mode_rule=_REWRITE_MODES[rewrite_mode],
            language=output_language,
            output_contract=_output_contract(task_type),
            model_compat=compat,
            media_evidence=media_evidence,
            reference_fidelity_rule=reference_fidelity_rule,
            ir_discipline=ir_discipline,
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

        media_manifest_for_hash = {
            "pictures": [
                {"ordinal": idx, "content_sha256": _sha256_text(data_url)}
                for idx, data_url in images
            ],
            "video_sequence": [
                {"source_index": source_index, "content_sha256": _sha256_text(data_url)}
                for source_index, data_url in sequence_frames
            ],
            "video_input_frame_count": sequence_total,
            "frame_selection_mode": str(frame_selection_mode or ""),
            "frame_selection_spec": str(frame_selection_spec or ""),
            "native_video": (
                {
                    "source_fingerprint": str(native_video_source.get("source_fingerprint") or ""),
                    "size_bytes": int(native_video_source.get("size_bytes") or 0),
                    "mime_type": str(native_video_source.get("mime_type") or ""),
                    "requested_route": gemini_video_route,
                    "requested_fps": gemini_video_fps,
                }
                if isinstance(native_video_source, dict) else None
            ),
        }
        input_fingerprints = {
            "prompt_sha256": _sha256_text(prompt),
            "system_sha256": _sha256_text(system),
            "module_sha256": _json_fingerprint({
                "system_module": system_module,
                "module_manifest": manifest_text,
            }),
            "media_manifest_sha256": _json_fingerprint(media_manifest_for_hash),
        }
        runtime_notes.append(
            "[输入指纹] " + "；".join(
                f"{name}={value}" for name, value in input_fingerprints.items()
            )
        )

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
        sequence_sheet = {}
        if staged and sequence_frames:
            sequence_sheet, sequence_notes = _analyze_video_sequence(
                api_base_url, key, model, sequence_frames, sequence_total,
                temperature, max_tokens, timeout_s, json_mode, api_reasoning,
            )
            sheet_notes.extend(sequence_notes)
        runtime_notes.extend(sheet_notes)
        user_content = [{"type": "text", "text": prompt}]
        if sheet_items:
            for item in sheet_items:
                user_content.append({
                    "type": "text",
                    "text": (
                        f"参考资产 {item.get('input_index')}"
                        f"（对应 <Picture {item.get('input_index')}>）分析摘要："
                    )
                            + json.dumps({k: v for k, v in item.items()
                                          if k != "input_index"},
                                         ensure_ascii=False),
                })
        else:
            for idx, data_url in images:
                user_content.append({"type": "text", "text": f"参考资产 {idx}（对应 <Picture {idx}>）："})
                user_content.append({"type": "image_url", "image_url": {"url": data_url}})
        if sequence_sheet:
            user_content.append({
                "type": "text",
                "text": (
                    "<Video 1> 代表帧联合分析摘要（这是从已附媒体中提取的事实，不是最终提示词；"
                    "不得把源表演者外观当成目标主体）："
                    + json.dumps(sequence_sheet, ensure_ascii=False)
                ),
            })
        elif sequence_frames:
            explicit_cut_points = timeline_manifest.get("shot_boundaries_input", [])
            if explicit_cut_points:
                cut_rule = (
                    "上游已明确真实切镜边界为 "
                    + ", ".join(f"{point:.3f}s" for point in explicit_cut_points)
                    + "；最终 [Shot N] 必须严格按这些边界分段，不得增加或遗漏。"
                )
            else:
                cut_rule = (
                    "上游没有提供明确切镜边界；只把前后构图的不连续跳变判断为切镜，"
                    "人物进出画、动作阶段、运镜、遮挡或闪白不得单独新建 Shot。"
                )
            user_content.append({
                "type": "text",
                "text": (
                    f"以下 {len(sequence_frames)} 张图片按输入顺序共同构成 <Video 1> 的代表帧时间线；"
                    "它们是同一段视频的连续时间证据，只用于观察动作、镜头、场景和定位源表演者，"
                    "不是独立的 <Picture N> 参考资产或多段视频。先聚合动作阶段与真实切镜点，"
                    "不要逐帧复述，不得重绘、新增或重设计开头。" + cut_rule
                ),
            })
            timeline_by_source = {
                int(item["frame"]): item for item in timeline_manifest.get("frames", [])
            } if timeline_manifest else {}
            for sequence_ordinal, (source_index, data_url) in enumerate(sequence_frames, start=1):
                position = 0.0 if sequence_total <= 1 else source_index / (sequence_total - 1)
                timeline_item = timeline_by_source.get(int(source_index), {})
                timecode_note = (
                    f"，时间码 {timeline_item.get('timecode')}"
                    if timeline_item.get("timecode") else ""
                )
                user_content.append({
                    "type": "text",
                    "text": (
                        f"<Video 1> 代表帧 {sequence_ordinal}/{len(sequence_frames)}："
                        f"输入序列索引 {source_index}/{sequence_total - 1}，"
                        f"约位于时间线 {position:.0%}{timecode_note}。"
                    ),
                })
                user_content.append({"type": "image_url", "image_url": {"url": data_url}})
        if native_video_source is not None:
            timeline_hint = ""
            if timeline_manifest:
                segment_hint = ", ".join(
                    f"段{item['segment']} {item['start_sec']:.3f}-{item['end_sec']:.3f}s"
                    for item in timeline_manifest.get("segments", [])[:12]
                )
                timeline_hint = (
                    f" 辅助确定性时间轴：fps={timeline_manifest['fps']:g}，"
                    f"时长={timeline_manifest['duration_seconds']:.3f}s"
                    + (f"，切镜段={segment_hint}" if segment_hint else "")
                    + "。"
                )
            user_content.append({
                "type": "text",
                "text": (
                    "以下原生视频文件是同一个 <Video 1>。按完整时间顺序理解动作流、真实切镜、"
                    "背景连续性和音频；不要逐帧复述，不得新增、重绘或重设计开头。"
                    + timeline_hint
                ),
            })
            user_content.append({
                "type": "gemini_video_file",
                "source": native_video_source,
                "route": gemini_video_route,
                "fps": gemini_video_fps,
            })
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        try:
            final_schema = _output_json_schema(task_type)
            raw = _call_chat(
                api_base_url, key, model, messages, temperature, max_tokens, timeout_s,
                json_mode=json_mode, api_reasoning=api_reasoning,
                response_schema=final_schema, request_purpose="final",
            )
            stats_note = _chat_stats_note(raw, "最终生成首次调用")
            if stats_note:
                runtime_notes.append(stats_note)

            retry_reason = _completion_retry_reason(raw)
            if not retry_reason:
                try:
                    parsed = _parse_json_text(raw)
                except ValueError as parse_exc:
                    retry_reason = f"首次 JSON 解析失败：{parse_exc}"

            if retry_reason:
                runtime_notes.append(f"[自动恢复] {retry_reason}；改为直接作答并重试一次")
                retry_messages = _compact_retry_messages(messages, model, retry_reason)
                retry_reasoning = (
                    "off" if "qwen" in model.casefold() or is_deepseek else api_reasoning
                )
                raw = _call_chat(
                    api_base_url, key, model, retry_messages, temperature, max_tokens, timeout_s,
                    json_mode=json_mode, api_reasoning=retry_reasoning,
                    response_schema=final_schema, request_purpose="final",
                )
                stats_note = _chat_stats_note(raw, "最终生成恢复调用")
                if stats_note:
                    runtime_notes.append(stats_note)
                terminal_reason = _completion_retry_reason(raw)
                if terminal_reason:
                    raise RuntimeError(f"恢复调用仍未完整结束：{terminal_reason}")
                parsed = _parse_json_text(raw)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("MiniMax H3 PromptDirector: API/解析失败: %s", exc)
            if lmstudio_after_use != "keep_loaded" and lm_root:
                try:
                    _lmstudio_unload(lm_root, model, key, timeout_s)
                except Exception as unload_exc:  # noqa: BLE001
                    LOGGER.warning("MiniMax H3 PromptDirector: 失败清理时卸载异常: %s", unload_exc)
            return fail_or_passthrough(f"API/解析失败: {exc}", api_called=True)

        native_video_meta = dict(getattr(raw, "media_meta", {}) or {})
        if native_video_meta:
            runtime_notes.append(
                "[Gemini 原生视频] "
                f"route={native_video_meta.get('route', '?')}，"
                f"reason={native_video_meta.get('route_reason', '?')}，"
                f"size={int(native_video_meta.get('size_bytes') or 0) / 1024 / 1024:.2f} MiB，"
                f"inline_estimate={int(native_video_meta.get('inline_total_estimated_bytes') or 0) / 1024 / 1024:.2f} MiB，"
                f"fps={native_video_meta.get('fps', '?')}"
                f"（{native_video_meta.get('fps_source', '?')}），"
                f"prepare={int(native_video_meta.get('prepare_total_ms') or 0) / 1000:.2f}s，"
                f"upload={int(native_video_meta.get('upload_ms') or 0) / 1000:.2f}s，"
                f"processing_wait={int(native_video_meta.get('processing_wait_ms') or 0) / 1000:.2f}s，"
                f"remote_cleanup={native_video_meta.get('remote_cleanup', '?')}，"
                f"content_sha256={native_video_meta.get('content_sha256', '?')}。"
            )

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
                media_inventory=media_inventory,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("MiniMax H3 PromptDirector: IR 编译失败: %s", exc)
            return fail_or_passthrough(f"IR 编译失败: {exc}", api_called=True)
        enhanced = compiled["prompt"]
        if module_ids:
            compiled["ir"]["applied_modules"] = module_ids
        ir_json = json.dumps(compiled["ir"], ensure_ascii=False, indent=2)
        shots_out = compiled["validation"]["checks"]["shot_count"]
        if compiled.get("normalizations"):
            runtime_notes.append(
                "[确定性规范化] " + "；".join(compiled["normalizations"])
            )
        # v0.2：Reference Sheet 输出（两阶段时含逐素材事实 JSON，否则空）
        sheet_payload = {}
        if sheet_items:
            sheet_payload["assets"] = sheet_items
        if sequence_frames:
            sheet_payload["video_frame_sequence"] = {
                "h3_label": "<Video 1>",
                "input_frame_count": sequence_total,
                "director_input_frame_count": sequence_input_total,
                "sent_frame_count": len(sequence_frames),
                "selected_indices": [idx for idx, _ in sequence_frames],
                "selection_owner": "timeline_context" if timeline_manifest else "director",
                "selection_mode": (
                    "timeline_context" if timeline_manifest else frame_selection_mode
                ),
                "selection_spec": "" if timeline_manifest else str(frame_selection_spec or ""),
                "selection_note": sequence_selection_note,
                "note": "API 时间线证据；不占用 Picture 标签，也不会自动传给 H3。",
            }
            if timeline_manifest:
                sheet_payload["video_frame_sequence"]["timeline"] = {
                    "fps": timeline_manifest.get("fps"),
                    "duration_seconds": timeline_manifest.get("duration_seconds"),
                    "segments": timeline_manifest.get("segments", []),
                }
        if native_video_meta:
            sheet_payload["native_video"] = {
                "h3_label": "<Video 1>",
                **native_video_meta,
                "note": "Gemini 原生视频证据；远端 URI 与本地绝对路径不会写入输出。",
            }
        if sequence_sheet:
            sheet_payload["video_sequence_analysis"] = sequence_sheet
        if sheet_payload:
            sheet_out = json.dumps(sheet_payload, ensure_ascii=False, indent=2)
        else:
            sheet_out = "[]"
        if compiled["validation"]["errors"]:
            short_errors = "；".join(compiled["validation"]["errors"][:5])
            return fail_or_passthrough(
                f"Prompt IR 未通过确定性校验：{short_errors}", api_called=True,
                reference_sheet=sheet_out, prompt_ir=ir_json,
            )
        report = "\n".join([
            (
                f"task_type={task_type} 模型={model} profile={resolved_profile} "
                f"最终思考策略={_reasoning_report_mode(api_base_url, api_reasoning, model, 'final')} "
                f"素材分析思考策略={_reasoning_report_mode(api_base_url, api_reasoning, model, 'analysis')}"
            ),
            f"参考视频跟随档={effective_fidelity}（节点值={reference_fidelity}）",
            f"IR 媒体清单={media_inventory_text}",
            f"API Key 来源={key_source}（值不写入报告）",
            f"导演识图素材={len(images)} 张（仅发送给提示词 API；端口→标签 {port_map}）",
            (
                f"参考视频序列帧={len(sequence_frames)}/{sequence_total} 张"
                f"（{sequence_selection_note}；输入索引 "
                f"{_summarize_indices([idx for idx, _ in sequence_frames])}；"
                "共同对应 <Video 1>，不占 Picture 标签）"
                if sequence_frames else "参考视频序列帧=0 张"
            ),
            (
                f"Gemini 原生视频=已发送（route={native_video_meta.get('route')}，"
                f"{int(native_video_meta.get('size_bytes') or 0) / 1024 / 1024:.2f} MiB）"
                if native_video_meta else "Gemini 原生视频=未连接"
            ),
            f"分析模式={'staged' if staged else 'single'}",
            f"已应用模块={module_ids or '无'}",
            *runtime_notes,
            f"分镜={shots_out} 段  节点端到端耗时={time.time() - t0:.1f}s",
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


def _build_cloud_presets() -> dict:
    """构建云端连接预设，并让每个预设绑定 registry 能力 profile。

    连接层（base_url/credential/environment_variable/json_mode/api_failure_policy）
    留在本文件；能力声明（provider/wire/model/evidence/modalities/structured_output/
    reasoning/media_limits）以 provider_registry 的 CapabilityProfile 为**单一来源**。
    `capability_profile_id` 即两者绑定键，任何连接预设都必须能经
    `profile_for_preset()` 解析到已注册能力；否则在加载期直接报错，避免
    「连接层声称支持某能力但能力层无对应 profile」的漂移。

    备注：连接层的 `model_profile`（提示词纪律，见 _MODEL_COMPAT）与能力层的
    `wire`/`model` 是两个不同维度——前者是业务提示词口径，后者是 wire 能力口径，
    不应混用，故连接预设仍保留 model_profile。
    """
    registry = _load_provider_registry()
    presets = {
        "deepseek": {
            "provider": "deepseek",
            "provider_label": "DeepSeek",
            "base_url": "https://api.deepseek.com",
            "credential_id": "deepseek_default",
            "environment_variable": "DEEPSEEK_API_KEY",
            "default_model": "deepseek-v4-flash-vision-exp",
            "model_profile": "deepseek_vision",
            "json_mode": "force",
            "api_failure_policy": "stop",
            # Prompt IR 的业务预算，不等于供应商或模型宣传的理论最大输出。
            "prompt_ir_max_tokens": 16384,
            "is_custom": False,
        },
        "gemini": {
            "provider": "gemini",
            "provider_label": "Gemini",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "credential_id": "gemini_default",
            "environment_variable": "GEMINI_API_KEY",
            # 2026-08-25 AA-Codex strict-schema 30/30; current official model page
            # confirms image/video input and a 65,536-token output limit.  More
            # expensive Flash/Pro models remain explicit A/B overrides, not defaults.
            "default_model": "gemini-3.1-flash-lite",
            "model_profile": "gemini_vision",
            "json_mode": "force",
            "api_failure_policy": "stop",
            "prompt_ir_max_tokens": 65536,
            "is_custom": False,
        },
        # 2026-08-28：常用云端视觉模型预设（OpenAI 兼容壳；Claude 走原生 Messages）。
        # custom 是纯 OpenAI 兼容壳，base_url / model / key 全由用户在节点提供。
        "doubao": {
            "provider": "volcengine",
            "provider_label": "豆包 · 火山方舟",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "credential_id": "volcengine_default",
            "environment_variable": "ARK_API_KEY",
            "default_model": "doubao-seed-evolving",
            "model_profile": "openai_compat_vision",
            "json_mode": "auto_retry",
            "api_failure_policy": "stop",
            "prompt_ir_max_tokens": 16384,
            "is_custom": False,
        },
        "claude": {
            "provider": "anthropic",
            "provider_label": "Claude · Anthropic",
            "base_url": "https://api.anthropic.com/v1",
            "credential_id": "anthropic_default",
            "environment_variable": "ANTHROPIC_API_KEY",
            "default_model": "claude-sonnet-4-6",
            "model_profile": "claude_vision",
            "json_mode": "auto_retry",
            "api_failure_policy": "stop",
            "prompt_ir_max_tokens": 16384,
            "is_custom": False,
        },
        "glm": {
            "provider": "zhipu",
            "provider_label": "GLM · 智谱",
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "credential_id": "zhipu_default",
            "environment_variable": "ZHIPU_API_KEY",
            "default_model": "glm-5.3-flash",
            "model_profile": "openai_compat_vision",
            "json_mode": "auto_retry",
            "api_failure_policy": "stop",
            "prompt_ir_max_tokens": 16384,
            "is_custom": False,
        },
        "minimax": {
            "provider": "minimax",
            "provider_label": "MiniMax 官方 LLM",
            "base_url": "https://api.minimaxi.com/v1",
            "credential_id": "minimax_default",
            "environment_variable": "MINIMAX_API_KEY",
            # 2026-08-28 官方公开文本模型仍为 M2.7/M2.5/M2.x；普通 Chat
            # 未声明可直接接收本节点的多图输入。保留旧 preset id 只用于给已保存
            # workflow 一个明确、fail-closed 的迁移错误，不在新节点下拉中展示。
            "default_model": "MiniMax-M2.7",
            "model_profile": "openai_compat_vision",
            "json_mode": "auto_retry",
            "api_failure_policy": "stop",
            "prompt_ir_max_tokens": 16384,
            "is_custom": False,
            "availability": "hidden_no_vision",
        },
        "custom": {
            "provider": "custom",
            "provider_label": "自定义 OpenAI 兼容",
            "base_url": "",   # 由用户在节点 cloud_base_url 填写
            "credential_id": "custom_default",
            "environment_variable": "OPENAI_API_KEY",
            "default_model": "",   # 由用户在节点 api_model 填写
            "model_profile": "openai_compat_vision",
            "json_mode": "auto_retry",
            "api_failure_policy": "stop",
            "prompt_ir_max_tokens": 16384,
            "is_custom": True,
        },
        "my_presets": {
            "provider": "custom",
            "provider_label": "我的预设",
            "base_url": "",   # 由已保存的「我的预设」决定（my_preset 名）
            "credential_id": "custom_default",
            "environment_variable": "OPENAI_API_KEY",
            "default_model": "",   # 由已保存的预设决定
            "model_profile": "openai_compat_vision",
            "json_mode": "auto_retry",
            "api_failure_policy": "stop",
            "prompt_ir_max_tokens": 16384,
            "is_custom": True,
            "is_my_presets": True,
            # 只为旧 workflow/API 输入保留；新 UI 已把它并入 custom。
            "availability": "hidden_legacy",
        },
    }
    # 引用 registry 能力 profile：校验绑定存在，并把 capability_profile_id 写回。
    for preset_id, item in presets.items():
        profile = registry.profile_for_preset(preset_id)
        item["capability_profile_id"] = profile.profile_id
        # 连接层 default_model 必须与能力层声明一致（custom 例外：model 由用户填，
        # 此时跳过一致性校验，避免连接层自定义模型与能力层空 model 冲突）。
        if item["is_custom"]:
            continue
        if item["default_model"] != profile.model:
            raise ValueError(
                f"cloud preset {preset_id!r} default_model={item['default_model']!r} "
                f"与 registry profile {profile.profile_id!r} model={profile.model!r} 不一致"
            )
    return presets


_CLOUD_DIRECTOR_PRESETS = _build_cloud_presets()
_CLOUD_DIRECTOR_VISIBLE_PRESETS = [
    preset_id for preset_id, preset in _CLOUD_DIRECTOR_PRESETS.items()
    if preset.get("availability", "visible") == "visible"
]


class MiniMaxH3CloudDirector(MiniMaxH3PromptDirector):
    """云端多模态薄入口；供应商差异由受控预设/adapter 解析。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "", "placeholder": "描述视频目标、素材分工与必须保留/替换的内容"},
                ),
                "task_type": (list(_TASK_RULES), {"default": "Ref2VA"}),
                "duration_seconds": (
                    "FLOAT", {"default": 5.0, "min": 4.0, "max": 15.0, "step": 0.5},
                ),
                "shot_count": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1}),
                "rewrite_mode": (list(_REWRITE_MODES), {"default": "balanced"}),
                "output_language": (["English", "中文"], {"default": "English"}),
                # Parameter name stays stable for saved workflow/API compatibility;
                # the UI presents it as a connection preset rather than raw provider knobs.
                "cloud_provider": (_CLOUD_DIRECTOR_VISIBLE_PRESETS, {"default": "deepseek"}),
                "api_model": (
                    "STRING",
                    {"default": "", "placeholder": "留空使用云端预设的模型；custom 预设请填模型 ID"},
                ),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 16384, "min": 256, "max": 131072, "step": 1024}),
                "timeout_s": ("INT", {"default": 600, "min": 30, "max": 1800, "step": 10}),
                "api_reasoning": (["auto", "off", "on"], {"default": "auto"}),
                "analysis_mode": (["auto", "single", "staged"], {"default": "auto"}),
                "frame_sequence_limit": ("INT", {
                    "default": 48, "min": 1, "max": 600, "step": 1,
                    "tooltip": "仅直连 IMAGE 批次时生效；连接时间线 manifest 后由上游控制",
                }),
                "frame_selection_mode": (_FRAME_SELECTION_MODES, {
                    "default": "uniform_full",
                    "tooltip": "仅直连 IMAGE 批次时生效；连接时间线 manifest 后由上游控制",
                }),
                "frame_selection_spec": (
                    "STRING",
                    {"default": "", "placeholder": "仅直连：0,41,82,-1 或 0,33,66,100"},
                ),
                # ComfyUI 按 widgets_values 的位置恢复旧工作流。新增字段必须追加在
                # 末尾，不能插进 api_model/temperature 等既有字段中间。
                "cloud_base_url": (
                    "STRING",
                    {"default": "", "placeholder": "custom：OpenAI 兼容 Base URL（如 https://api.xxx.com/v1）"},
                ),
                "my_preset": (
                    "STRING",
                    {"default": "", "placeholder": "我的预设：由底部管理器选择，或手动填已保存名称"},
                ),
                "gemini_video_route": (
                    ["auto", "inline", "file_api"], {"default": "auto"},
                ),
                "gemini_video_fps": (
                    "FLOAT", {"default": 0.0, "min": 0.0, "max": 24.0, "step": 1.0},
                ),
            },
            "optional": {
                # 兼容没有该字段的旧 workflow/API prompt；函数默认值为 auto。
                "reference_fidelity": (
                    list(_REFERENCE_FIDELITY_MODES), {"default": "auto"},
                ),
                **{
                    f"ref_image_{i}": ("IMAGE", {"label": f"云端导演识图素材 {i}（不传给 H3）"})
                    for i in range(1, 10)
                },
                "video_frame_sequence": (
                    "IMAGE",
                    {"label": "参考视频抽帧 / 图像序列（批次；仅供云端 API，不传给 H3）"},
                ),
                "cloud_video": (
                    "MINIMAX_H3_CLOUD_VIDEO",
                    {"label": "Gemini 原生视频（与视频帧二选一；仅供云端 API）"},
                ),
                "comfy_video": (
                    "VIDEO",
                    {"label": "Comfy 原生 VIDEO（未裁剪文件；与上两路二选一）"},
                ),
                "video_timeline_manifest": (
                    "STRING", {"multiline": True, "default": "", "forceInput": True},
                ),
                "system_module": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "module_manifest": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = MiniMaxH3PromptDirector.RETURN_TYPES
    RETURN_NAMES = MiniMaxH3PromptDirector.RETURN_NAMES
    FUNCTION = "direct_cloud"
    CATEGORY = "MiniMax H3 Lab/Prompt/Cloud"

    def direct_cloud(self, prompt, task_type, duration_seconds, shot_count, rewrite_mode,
                     output_language, cloud_provider, api_model, temperature, max_tokens,
                      timeout_s, api_reasoning="auto", analysis_mode="auto",
                      frame_sequence_limit=48, frame_selection_mode="uniform_full",
                      frame_selection_spec="", cloud_base_url="", my_preset="",
                      gemini_video_route="auto", gemini_video_fps=0.0,
                      reference_fidelity="auto", **kwargs):
        cloud_video_source = kwargs.pop("cloud_video", None)
        comfy_video = kwargs.pop("comfy_video", None)
        if cloud_video_source is not None and comfy_video is not None:
            raise RuntimeError(
                "“云端原生视频”与“Comfy 原生 VIDEO”不能同时连接；请保留其中一路"
            )
        if comfy_video is not None:
            cloud_video_source = _load_cloud_video().cloud_video_source_from_comfy_video(
                comfy_video
            )
        preset_id = str(cloud_provider or "deepseek").strip().lower()
        preset = _CLOUD_DIRECTOR_PRESETS.get(preset_id)
        if preset is None:
            raise ValueError(f"未知云端连接预设：{preset_id!r}")
        if preset.get("availability") == "hidden_no_vision":
            raise RuntimeError(
                "MiniMax 官方文本模型预设尚未证明支持本节点所需的多图/视频帧输入，"
                "已停止请求；请改用已验证的 Gemini、DeepSeek 或 GLM 预设。"
            )
        provider = preset["provider"]
        provider_label = preset["provider_label"]
        # 已保存连接是 custom 的一种状态，不再作为第二个“供应商”。旧工作流
        # 仍可传 my_presets，并在这里走同一条兼容路径。
        my_preset_loaded = None
        cloud_connection_note = ""
        preset_name = str(my_preset or "").strip()
        if preset.get("is_my_presets") or (preset.get("is_custom") and preset_name):
            if not preset_name:
                raise ValueError("我的预设需要填写已保存的预设名（my_preset）")
            creds = _load_cloud_credentials()
            my_preset_loaded = creds.get_custom_preset(preset_name)
            base_url = my_preset_loaded["base_url"]
            cloud_connection_note = (
                f"[OpenAI 兼容已保存连接] 已加载「{my_preset_loaded['name']}」"
                f" base_url={base_url}"
            )
        elif preset.get("is_custom"):
            base_url = str(cloud_base_url or "").strip()
            if not base_url:
                raise ValueError("自定义 OpenAI 兼容预设需要填写 cloud_base_url")
        else:
            base_url = preset["base_url"]
            cloud_connection_note = ""
        credential_id = (
            str(my_preset_loaded.get("credential_id") or preset["credential_id"])
            if my_preset_loaded else preset["credential_id"]
        )
        credentials = _load_cloud_credentials()
        key, key_source = credentials.resolve_credential(provider, credential_id)
        if not key and my_preset_loaded and credential_id != preset["credential_id"]:
            # 兼容旧版“我的预设”共用 custom_default 的本地 Key；用户在新管理器
            # 为该预设保存独立 Key 后自动切换到 per-preset credential。
            key, key_source = credentials.resolve_credential(provider, preset["credential_id"])
            if key:
                cloud_connection_note = (
                    f"[我的预设]「{my_preset_loaded['name']}」仍在使用旧版共享凭据；"
                    "建议在预设管理器中为它保存独立 Key。"
                )
        allow_local_unauthenticated = bool(
            preset.get("is_custom") and _is_local_lmstudio_url(base_url)
        )
        if not key and allow_local_unauthenticated:
            key_source = "本机 loopback 无鉴权兼容模式"
            cloud_connection_note = "；".join(note for note in (
                cloud_connection_note,
                "[OpenAI 兼容] 已识别本机 LM Studio：优先复用原生/JIT 自动加载接口，"
                "失败时回退 OpenAI 兼容接口；日常本地模型仍建议使用“提示词导演（顺序接口）”。",
            ) if note)
        elif not key:
            raise RuntimeError(
                f"MiniMax H3 云端导演尚未配置 {provider_label} 凭据。"
                "请点击节点底部的‘管理云端连接’，保存到本机用户配置，"
                f"或在启动 ComfyUI 前设置 {preset['environment_variable']}。"
            )
        model_override = str(api_model or "").strip()
        if my_preset_loaded:
            # 已保存连接的 model 由预设决定，避免节点残留覆盖它。
            model_override = ""
        elif preset.get("is_custom") and not model_override:
            raise ValueError("自定义 OpenAI 兼容预设需要填写 api_model（模型 ID）")
        other_preset_defaults = {
            str(item.get("default_model") or "")
            for item_id, item in _CLOUD_DIRECTOR_PRESETS.items()
            if item_id != preset_id
        }
        if model_override in other_preset_defaults:
            cloud_connection_note = (
                f"[云端预设] 忽略来自另一连接预设的旧模型覆盖 {model_override!r}；"
                f"已使用 {provider_label} 默认模型。"
            )
            model_override = ""
        # 我的预设：model 由已保存预设指定（覆盖 api_model 留空的情形）。
        if my_preset_loaded:
            model = my_preset_loaded["model"]
            loaded_note = (
                f"[我的预设]「{my_preset_loaded['name']}」"
                f"base_url={my_preset_loaded['base_url']} model={my_preset_loaded['model']}"
            )
            cloud_connection_note = "；".join(
                note for note in (cloud_connection_note, loaded_note) if note
            )
        else:
            model = model_override or preset["default_model"]
        if not model:
            raise ValueError("所选云端连接预设没有默认模型，请填写模型 ID 覆盖")
        try:
            requested_tokens = int(max_tokens)
        except (TypeError, ValueError):
            requested_tokens = 16384
        prompt_ir_cap = max(256, int(preset.get("prompt_ir_max_tokens") or 16384))
        if requested_tokens > prompt_ir_cap:
            cloud_connection_note = "；".join(note for note in (
                cloud_connection_note,
                f"[云端 Prompt IR 预算] 节点值 {requested_tokens} 已收敛到 {prompt_ir_cap}；"
                "模型/供应商理论输出上限不直接作为本节点的默认输出预算。",
            ) if note)
            max_tokens = prompt_ir_cap
        return super().direct(
            prompt=prompt,
            task_type=task_type,
            duration_seconds=duration_seconds,
            shot_count=shot_count,
            rewrite_mode=rewrite_mode,
            output_language=output_language,
            api_base_url=base_url,
            api_model=model,
            api_key="",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            lmstudio_after_use="keep_loaded",
            lmstudio_gpu_offload="auto",
            api_reasoning=api_reasoning,
            json_mode=preset["json_mode"],
            model_profile=preset["model_profile"],
            frame_sequence_limit=frame_sequence_limit,
            frame_selection_mode=frame_selection_mode,
            frame_selection_spec=frame_selection_spec,
            api_failure_policy=preset["api_failure_policy"],
            reference_fidelity=reference_fidelity,
            analysis_mode=analysis_mode,
            _resolved_api_key=key,
            _resolved_api_key_source=key_source,
            _cloud_connection_note=cloud_connection_note,
            _cloud_video=cloud_video_source,
            _gemini_video_route=gemini_video_route,
            _gemini_video_fps=gemini_video_fps,
            **kwargs,
        )


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PromptDirector": MiniMaxH3PromptDirector,
    "MiniMaxH3CloudDirector": MiniMaxH3CloudDirector,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PromptDirector": "MiniMax H3 提示词导演 (顺序接口)",
    "MiniMaxH3CloudDirector": "MiniMax H3 云端多模态导演",
}
