"""MiniMax H3 多 Provider 能力矩阵与 Contract 抽象（Provider Compatibility Core 第一块）。

设计动机（2026-08-28，GPT 网页端 + 本框架共同结论）：
- 不要把「Provider」当能力单位。能力应落在 `Provider × Wire × Model/Profile` 的
  最末端格子（例如 DeepSeek Chat wire 只支持 json_object，而 DeepSeek Responses
  wire 的 text.format 支持 json_schema——同一家、两个 wire、能力不同）。
- 支持状态拆成四档，避免「没账号也宣称支持」：documented / contract_passed /
  measured / recommended。
- 本文件是**声明式、纯数据、可测试**的能力矩阵，不绑定任何网络/API 调用；
  实际 wire 调用仍由 prompt_director 的 adapter 负责。后续 provider 差异应逐步
  收敛到这里（新 provider 只加一行 profile + contract fixture），而不是堆 if/else。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceStatus(str, Enum):
    """某条（provider × wire × model）能力声明的可信度。"""

    DOCUMENTED = "documented"          # 官方文档明确声明，本插件未实测
    CONTRACT_PASSED = "contract_passed"  # mock/fixture 能生成合法请求、解析合法响应
    MEASURED = "measured"              # 真实 API 跑过固定 smoke
    RECOMMENDED = "recommended"        # 重复回归后足以成为默认推荐


# 业务层统一的推理强度语义——adapter 负责翻译到各家真实字段。
REASONING_LEVELS = ("auto", "off", "low", "medium", "high", "max")

# 业务层统一的（media, structured output, reasoning）能力三元组。
_MODALITY_KINDS = ("image", "video_native", "video_frames", "audio")


@dataclass(frozen=True)
class MediaLimits:
    """媒体上传的实用约束。不同 endpoint/path 的约束不同，不能只用单一 max_payload。

    例如 Gemini：media.image.inline.max_payload / media.video.inline.recommended /
    media.file_api.max_file / media.file_api.ttl 各不同；OpenAI Vision 是 1500 图/512MB；
    Claude 标准端点是 600 图/32MB。字段含义见每个注释。
    """

    max_images_per_request: int | None = None      # 单请求最多图片数（None=未声明）
    image_inline_max_bytes: int | None = None      # 图片内联最大字节（None=未声明）
    video_native_supported: bool = False           # 是否支持原生视频（文件）输入
    video_inline_recommended_bytes: int | None = None  # 视频内联推荐上限（大于建议走 file api）
    file_api_max_file_bytes: int | None = None     # File API 单文件上限
    file_api_max_total_bytes: int | None = None    # File API 项目/TTL 总量
    file_api_ttl_hours: int | None = None          # File API 文件保留时长
    default_video_sample_fps: float | None = None  # 官方原生视频默认采 FPS（如 Gemini 1 FPS）
    frames_per_second_token_est: int | None = None  # 官方视频帧 token 估算（如 Gemini 258/帧）

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_images_per_request": self.max_images_per_request,
            "image_inline_max_bytes": self.image_inline_max_bytes,
            "video_native_supported": self.video_native_supported,
            "video_inline_recommended_bytes": self.video_inline_recommended_bytes,
            "file_api_max_file_bytes": self.file_api_max_file_bytes,
            "file_api_max_total_bytes": self.file_api_max_total_bytes,
            "file_api_ttl_hours": self.file_api_ttl_hours,
            "default_video_sample_fps": self.default_video_sample_fps,
            "frames_per_second_token_est": self.frames_per_second_token_est,
        }


@dataclass(frozen=True)
class StructuredOutputSpec:
    """声明式结构化输出能力（业务层只认识这三种 wire，具体字段由 adapter 翻译）。"""

    # wire = json_object | json_schema | native_schema | text_only | none
    wire: str = "json_object"
    # 该 wire 是否保证语义正确（官方文档承认只保证句法时须为 False）
    syntactic_only: bool = True
    # 是否支持严格 schema 模式（strict / required 枚举）
    strict_schema_supported: bool = False
    # 官方已知限制（如 Anthropic strict tool 数上限、schema 编译 180s、字段重排）
    known_quirks: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "wire": self.wire,
            "syntactic_only": self.syntactic_only,
            "strict_schema_supported": self.strict_schema_supported,
            "known_quirks": list(self.known_quirks),
        }


@dataclass(frozen=True)
class ReasoningSpec:
    """声明式推理控制能力。业务层只用 REASONING_LEVELS，adapter 翻译各家字段。"""

    supported_levels: tuple[str, ...]
    can_disable: bool = True                 # 该 wire 是否能真正关闭 thinking
    default_level: str = "auto"              # 官方默认强度
    # 是否随模型代际变化（如 Gemini 2.5 用 budget、3.x 用 level；Claude 4.7 废弃 budget）
    model_generation_dependent: bool = False
    # 官方已知不稳定点（如 Gemini 3 Pro 不能关 thinking、Claude 4.7 拒绝 budget_tokens）
    known_quirks: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported_levels": list(self.supported_levels),
            "can_disable": self.can_disable,
            "default_level": self.default_level,
            "model_generation_dependent": self.model_generation_dependent,
            "known_quirks": list(self.known_quirks),
        }


@dataclass(frozen=True)
class RequestShape:
    """声明式请求形状（发网前自检用，避免把不一致的形状发给供应商）。"""

    # native = 该供应商原生 wire（如 Gemini systemInstruction+contents）；openai = OpenAI 兼容 messages
    messages_shape: str = "openai"
    # system 是否走独立字段（native systemInstruction）还是混进第一条 message
    system_separate: bool = False
    # 图片用什么字段（Gemini inlineData / OpenAI image_url data URL）
    image_part: str = "image_url"
    # 是否支持音频输入
    audio_input: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages_shape": self.messages_shape,
            "system_separate": self.system_separate,
            "image_part": self.image_part,
            "audio_input": self.audio_input,
        }


@dataclass(frozen=True)
class RetryPolicy:
    """声明式重试策略（分类 + 上限）。"""

    retry_categories: tuple[str, ...] = ("timeout", "http_429_rate_limit", "http_5xx_server", "remote_disconnect")
    max_retries: int = 1                     # 昂贵多模态默认只补 1 次
    respect_retry_after: bool = True         # 429 尊重 Retry-After
    must_reuse_identical_body: bool = True   # 重试必须逐字节复用原请求体

    def to_dict(self) -> dict[str, Any]:
        return {
            "retry_categories": list(self.retry_categories),
            "max_retries": self.max_retries,
            "respect_retry_after": self.respect_retry_after,
            "must_reuse_identical_body": self.must_reuse_identical_body,
        }


@dataclass(frozen=True)
class CapabilityProfile:
    """最末端格子：provider × wire × model 的能力声明。"""

    profile_id: str                     # 唯一，如 "deepseek_chat_v4_flash_vision"
    provider: str
    wire: str                           # chat | responses | generate_content | openai_compat | anthropic
    model: str
    label: str
    evidence: EvidenceStatus
    modalities: tuple[str, ...]         # 官方文档支持：image / video_frames / video_native / audio ...
    structured_output: StructuredOutputSpec
    reasoning: ReasoningSpec
    media_limits: MediaLimits
    request_shape: RequestShape
    retry_policy: RetryPolicy
    notes: str = ""
    evidence_source: tuple[str, ...] = ()   # 官方 URL / 实测记录
    # 插件**实际实现**的能力（可能与官方 modalities 不同，如 Gemini 官方支持原生视频但
    # 本插件尚未实现 File API，只能抽帧 inline）。None = 默认等于 modalities。
    implemented_modalities: tuple[str, ...] | None = None

    @property
    def _implemented(self) -> tuple[str, ...]:
        return self.implemented_modalities if self.implemented_modalities is not None else self.modalities

    @property
    def supports_vision_images(self) -> bool:
        return "image" in self._implemented

    @property
    def supports_native_video(self) -> bool:
        return "video_native" in self._implemented and self.media_limits.video_native_supported

    @property
    def must_extract_frames(self) -> bool:
        """该 wire 对视频是否只能抽帧（插件已实现层面）。"""
        return "video_native" not in self._implemented

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "provider": self.provider,
            "wire": self.wire,
            "model": self.model,
            "label": self.label,
            "evidence": self.evidence.value,
            "modalities": list(self.modalities),
            "implemented_modalities": list(self._implemented),
            "structured_output": self.structured_output.to_dict(),
            "reasoning": self.reasoning.to_dict(),
            "media_limits": self.media_limits.to_dict(),
            "request_shape": self.request_shape.to_dict(),
            "retry_policy": self.retry_policy.to_dict(),
            "notes": self.notes,
            "evidence_source": list(self.evidence_source),
        }


# ============================================================================
# 已注册的能力矩阵（初期只登记已知事实，不宣称未实测能力）
# ============================================================================

def _profile_deepseek_chat() -> CapabilityProfile:
    return CapabilityProfile(
        profile_id="deepseek_chat_v4_flash_vision",
        provider="DeepSeek",
        wire="chat",
        model="deepseek-v4-flash-vision-exp",
        label="DeepSeek Chat Vision（官方）",
        # 本插件已实测过这个 wire（codex HANDOFF 记录 json_object + thinking 顶层 + 一次重试）
        evidence=EvidenceStatus.MEASURED,
        modalities=("image", "video_frames"),
        structured_output=StructuredOutputSpec(
            wire="json_object", syntactic_only=True, strict_schema_supported=False,
            known_quirks=("Chat 官方只声明 json_object，未声明 json_schema；JSON mode 偶发空 content",),
        ),
        reasoning=ReasoningSpec(
            supported_levels=("off", "low", "medium", "high", "max", "auto"),
            can_disable=True, default_level="auto", model_generation_dependent=False,
            known_quirks=("官方顶层 thinking:{type} + reasoning_effort；开启时 temperature 无效",),
        ),
        media_limits=MediaLimits(
            # DeepSeek Vision 以图片为核心，H3 用途基本要抽帧；未声明单请求图片上限
            video_native_supported=False,
        ),
        request_shape=RequestShape(messages_shape="openai", image_part="image_url"),
        retry_policy=RetryPolicy(max_retries=1, respect_retry_after=True, must_reuse_identical_body=True),
        evidence_source=(
            "codex HANDOFF 2026-08-26：JSON 用 json_object、thinking 顶层、一次原 body 重试、脱敏",
        ),
    )


def _profile_gemini_generate_content() -> CapabilityProfile:
    return CapabilityProfile(
        profile_id="gemini_generate_content_flash_lite",
        provider="Gemini",
        wire="generate_content",
        model="gemini-3.1-flash-lite",
        label="Gemini GenerateContent（官方原生）",
        evidence=EvidenceStatus.MEASURED,
        # 官方文档支持 image/video_native/video_frames/audio；本插件实现图片/抽帧，
        # 并为原生视频提供小文件 inlineData 与大文件 File API 路径。
        modalities=("image", "video_native", "video_frames", "audio"),
        implemented_modalities=("image", "video_frames", "video_native"),
        structured_output=StructuredOutputSpec(
            wire="native_schema", syntactic_only=True, strict_schema_supported=True,
            known_quirks=("responseMimeType=application/json + responseJsonSchema；深层嵌套 schema 可能被拒",),
        ),
        reasoning=ReasoningSpec(
            # Gemini 2.5 用 budget、3.x 用 level；3 Pro 不能关 thinking
            supported_levels=("off", "low", "medium", "high", "max", "auto"),
            can_disable=False, default_level="high", model_generation_dependent=True,
            known_quirks=(
                "Gemini 3 Pro 无法完全关闭 thinking；2.5 用 thinkingBudget、3.x 用 thinkingLevel；"
                "thinkingBudget:0 在 3.x 会 400",
            ),
        ),
        media_limits=MediaLimits(
            max_images_per_request=3600,
            image_inline_max_bytes=100 * 1024 * 1024,
            video_native_supported=True,
            video_inline_recommended_bytes=20 * 1024 * 1024,
            file_api_max_file_bytes=2 * 1024 * 1024 * 1024,
            file_api_max_total_bytes=20 * 1024 * 1024 * 1024,
            file_api_ttl_hours=48,
            default_video_sample_fps=1.0,
            frames_per_second_token_est=258,
        ),
        request_shape=RequestShape(
            messages_shape="native", system_separate=True, image_part="inline_data", audio_input=True,
        ),
        retry_policy=RetryPolicy(max_retries=1, respect_retry_after=True, must_reuse_identical_body=True),
        evidence_source=(
            "Google File Input / Video Understanding 官方文档；图片 wire 已实测；"
            "原生视频 inline/File API 已完成契约与 mock 回归，真实 Key 验收待用户执行",
        ),
    )


def _profile_openai_responses_documented() -> CapabilityProfile:
    return CapabilityProfile(
        profile_id="openai_responses_gpt5_documented",
        provider="OpenAI",
        wire="responses",
        model="gpt-5.6",
        label="OpenAI GPT-5.6 Responses（documented，未实测）",
        # 只有官方文档声明，插件无账号实测——务必标注 documented，不冒充 measured
        evidence=EvidenceStatus.DOCUMENTED,
        modalities=("image", "video_frames"),   # 官方主模型不接受视频输入，只能抽帧
        structured_output=StructuredOutputSpec(
            wire="json_schema", syntactic_only=True, strict_schema_supported=True,
            known_quirks=("text.format 支持 json_object/json_schema；json_schema 需 strict",),
        ),
        reasoning=ReasoningSpec(
            supported_levels=("off", "low", "medium", "high", "max", "auto"),
            can_disable=True, default_level="medium", model_generation_dependent=True,
            known_quirks=("reasoning.effort 值随模型变化，default=medium",),
        ),
        media_limits=MediaLimits(
            max_images_per_request=1500,
            image_inline_max_bytes=512 * 1024 * 1024,
            video_native_supported=False,   # GPT-5 系主模型不支持视频输入
        ),
        request_shape=RequestShape(messages_shape="responses", image_part="image_url"),
        retry_policy=RetryPolicy(max_retries=1),
        notes="仅凭官方文档登记（documented），无 key 实测；/v1/videos 是生成 endpoint，非视频理解输入。",
        evidence_source=(
            "OpenAI Vision 官方文档（1500 图 / 512MB）；GPT-5.6 官方模型页（Video: 不支持）",
        ),
    )


def _profile_claude_messages_documented() -> CapabilityProfile:
    return CapabilityProfile(
        profile_id="claude_messages_native_documented",
        provider="Anthropic",
        wire="anthropic",
        model="claude-sonnet-4.6",
        label="Claude Messages（documented，未实测）",
        evidence=EvidenceStatus.DOCUMENTED,
        modalities=("image", "video_frames"),   # 官方 Vision 无原生 video content block
        structured_output=StructuredOutputSpec(
            wire="native_schema", syntactic_only=True, strict_schema_supported=True,
            known_quirks=(
                "strict tool 数 / optional 参数 / union 类型有上限；schema 编译 180s 超时；"
                "字段会被重排成 required 在前",
            ),
        ),
        reasoning=ReasoningSpec(
            supported_levels=("off", "low", "medium", "high", "max", "auto"),
            can_disable=True, default_level="high", model_generation_dependent=True,
            known_quirks=(
                "4.6 前 thinking:{type:enabled,budget_tokens}；4.6+ adaptive+output_config.effort；"
                "4.7 废弃 budget_tokens 直接 400",
            ),
        ),
        media_limits=MediaLimits(
            max_images_per_request=600,
            image_inline_max_bytes=32 * 1024 * 1024,
            video_native_supported=False,   # 官方 Messages 无原生 video content block
        ),
        request_shape=RequestShape(messages_shape="anthropic", system_separate=True, image_part="base64"),
        retry_policy=RetryPolicy(max_retries=1),
        notes="仅凭官方文档登记（documented），无 key 实测；native_video 标注 unsupported，fallback=frames。",
        evidence_source=(
            "Anthropic Vision / Messages 官方文档（600 图 / 32MB，无原生 video block）；Extended Thinking 文档",
        ),
    )


# ============================================================================
# 2026-08-28：常用云端视觉模型（documented 档，无 key 实测；GLM 待用户实测后升档）
# ============================================================================

def _profile_volcengine_doubao_documented() -> CapabilityProfile:
    return CapabilityProfile(
        profile_id="volcengine_doubao_seed_evolving_documented",
        provider="Volcengine",
        wire="chat",
        model="doubao-seed-evolving",
        label="豆包 · 火山方舟（documented，未实测）",
        # 官方模型卡片：输入=文本/图片/视频，输出=文本；OpenAI 兼容 /v3
        evidence=EvidenceStatus.DOCUMENTED,
        modalities=("image", "video_frames"),
        structured_output=StructuredOutputSpec(
            wire="json_object", syntactic_only=True, strict_schema_supported=False,
            known_quirks=("方舟 Chat 用 json_object；应避免 json_schema 触发 400",),
        ),
        reasoning=ReasoningSpec(
            supported_levels=("off", "low", "medium", "high", "max", "auto"),
            can_disable=True, default_level="auto", model_generation_dependent=False,
            known_quirks=("thinking.type + reasoning_effort；无标准 GET /models 发现接口",),
        ),
        media_limits=MediaLimits(video_native_supported=False),
        request_shape=RequestShape(messages_shape="openai", image_part="image_url"),
        retry_policy=RetryPolicy(max_retries=1),
        notes="仅凭官方文档登记（documented），无 key 实测；模型列表用静态候选。",
        evidence_source=("火山方舟 Doubao-Seed-Evolving 官方模型卡片 + 第三方程（OpenAI 兼容 /v3）",),
    )


def _profile_anthropic_claude46_documented() -> CapabilityProfile:
    return CapabilityProfile(
        profile_id="anthropic_claude_sonnet46_documented",
        provider="Anthropic",
        wire="anthropic",
        model="claude-sonnet-4-6",
        label="Claude Sonnet 4.6（documented，未实测）",
        # 与既有 claude_messages_native_documented 同 wire，但模型档更贴近常用预设
        evidence=EvidenceStatus.DOCUMENTED,
        modalities=("image", "video_frames"),
        structured_output=StructuredOutputSpec(
            wire="native_schema", syntactic_only=True, strict_schema_supported=True,
            known_quirks=("4.6 后 output_config.format；4.7 禁 budget_tokens / 非默认采样参数",),
        ),
        reasoning=ReasoningSpec(
            supported_levels=("off", "low", "medium", "high", "max", "auto"),
            can_disable=True, default_level="high", model_generation_dependent=True,
            known_quirks=("4.6/4.7 用 thinking:{type:adaptive} + output_config.effort；4.6 前用 enabled+budget_tokens",),
        ),
        media_limits=MediaLimits(
            max_images_per_request=20,   # ComfyUI CLAUDE_MAX_IMAGES=20
            image_inline_max_bytes=32 * 1024 * 1024,
            video_native_supported=False,
        ),
        request_shape=RequestShape(messages_shape="anthropic", system_separate=True, image_part="base64"),
        retry_policy=RetryPolicy(max_retries=1),
        notes="仅凭官方文档登记（documented）；原生 Messages 多图 content blocks；未实测不宣称可用。",
        evidence_source=("Anthropic Migration guide（4.6/4.7 adaptive thinking）+ ComfyUI nodes_anthropic.py",),
    )


def _profile_zhipu_glm53_flash_documented() -> CapabilityProfile:
    return CapabilityProfile(
        profile_id="zhipu_glm53_flash_documented",
        provider="Zhipu",
        wire="chat",
        model="glm-5.3-flash",
        label="GLM-5.3-Flash（measured，真实视觉探针通过）",
        # 2026-08-28：用户提供真实 key，跑通 glm-5.3-flash 视觉 strict-JSON 探针 → 升 measured。
        evidence=EvidenceStatus.MEASURED,
        modalities=("image", "video_frames"),
        structured_output=StructuredOutputSpec(
            wire="json_object", syntactic_only=True, strict_schema_supported=False,
            known_quirks=("智谱 Chat 用 json_object；探针实测 strict JSON 通过；json_schema 兼容性待测",),
        ),
        reasoning=ReasoningSpec(
            supported_levels=("off", "low", "medium", "high", "max", "auto"),
            can_disable=False, default_level="max", model_generation_dependent=False,
            known_quirks=(
                "thinking.type 仅支持 enabled，不能关闭思考；官方视觉示例用 thinking:{type:enabled}，"
                "不接受 reasoning_effort 字段；推荐 temperature=1、top_p=0.95",
            ),
        ),
        media_limits=MediaLimits(video_native_supported=False),
        request_shape=RequestShape(messages_shape="openai", image_part="image_url"),
        retry_policy=RetryPolicy(max_retries=1),
        notes="真实视觉 strict-JSON 探针通过（measured）：glm-5.3-flash 原生多模态、1M 上下文、thinking 不可关闭。",
        evidence_source=(
            "智谱官方 GLM-5.3-Flash 模型文档 + HTTP API 文档；"
            "2026-08-28 真实 key 探针：/paas/v4/chat/completions + thinking:{type:enabled} + 红蓝图 strict JSON OK",
        ),
    )


def _profile_minimax_m27_text_documented() -> CapabilityProfile:
    return CapabilityProfile(
        profile_id="minimax_m27_text_documented",
        provider="MiniMax",
        wire="chat",
        model="MiniMax-M2.7",
        label="MiniMax-M2.7 文本（documented；CloudDirector 隐藏）",
        evidence=EvidenceStatus.DOCUMENTED,
        # 截至 2026-08-28，MiniMax 官方公开 Chat 文档只确认文本模型；图片
        # 理解需另一路 MCP/产品能力，不能把“平台有多模态”推断成该 chat wire
        # 能直接接收 image_url。因此本插件实现层明确不声明 vision。
        modalities=(),
        implemented_modalities=(),
        structured_output=StructuredOutputSpec(
            wire="json_object", syntactic_only=True, strict_schema_supported=False,
            known_quirks=("当前只登记文本 Chat；未验证 image_url 多图输入",),
        ),
        reasoning=ReasoningSpec(
            supported_levels=("auto", "high", "max"),
            can_disable=False, default_level="high", model_generation_dependent=False,
            known_quirks=("M2.x 思考控制与 OpenAI reasoning_effort 并非等价；需真实 wire 探测",),
        ),
        media_limits=MediaLimits(video_native_supported=False),
        request_shape=RequestShape(messages_shape="openai", image_part="image_url"),
        retry_policy=RetryPolicy(max_retries=1),
        notes="官方 OpenAI 兼容 /v1 文本模型；CloudDirector 需要多图，故隐藏并 fail-closed。",
        evidence_source=("MiniMax 开放平台 API Overview / Model Release Notes（2026-08-28 复核）",),
    )


def _profile_custom_openai_compat_documented() -> CapabilityProfile:
    return CapabilityProfile(
        profile_id="custom_openai_compat_documented",
        provider="Custom",
        wire="chat",
        model="",
        label="自定义 OpenAI 兼容（documented，由用户配置）",
        # 纯壳：能力完全取决于用户填的服务端，故只标 documented（不冒充已实现）。
        evidence=EvidenceStatus.DOCUMENTED,
        modalities=("image", "video_frames"),
        structured_output=StructuredOutputSpec(
            wire="json_object", syntactic_only=True, strict_schema_supported=False,
            known_quirks=("默认用 json_object（跨服务最稳）；json_schema 依服务端而定",),
        ),
        reasoning=ReasoningSpec(
            supported_levels=("off", "low", "medium", "high", "max", "auto"),
            can_disable=True, default_level="auto", model_generation_dependent=True,
            known_quirks=("OpenAI 兼容壳，thinking 字段依供应商而定；auto_retry 可降级",),
        ),
        media_limits=MediaLimits(video_native_supported=False),
        request_shape=RequestShape(messages_shape="openai", image_part="image_url"),
        retry_policy=RetryPolicy(max_retries=1),
        notes="纯通用 OpenAI 兼容壳；base_url/model/key 全由用户在节点提供，能力属 documented（未绑定具体服务）。",
        evidence_source=("OpenAI chat/completions 通用规范（自定义 base_url 由用户填写）",),
    )


_CAPABILITY_PROFILES: dict[str, CapabilityProfile] = {}
for _p in (
    _profile_deepseek_chat(),
    _profile_gemini_generate_content(),
    _profile_openai_responses_documented(),
    _profile_claude_messages_documented(),
    _profile_volcengine_doubao_documented(),
    _profile_anthropic_claude46_documented(),
    _profile_zhipu_glm53_flash_documented(),
    _profile_minimax_m27_text_documented(),
    _profile_custom_openai_compat_documented(),
):
    if _p.profile_id in _CAPABILITY_PROFILES:
        raise ValueError(f"dup capability profile_id: {_p.profile_id}")
    _CAPABILITY_PROFILES[_p.profile_id] = _p


def all_profiles() -> dict[str, CapabilityProfile]:
    return dict(_CAPABILITY_PROFILES)


def get_profile(profile_id: str) -> CapabilityProfile:
    profile = _CAPABILITY_PROFILES.get(profile_id)
    if profile is None:
        raise KeyError(f"unknown capability profile: {profile_id}")
    return profile


# ============================================================================
# 云端连接预设 → 能力 profile 的权威映射。
#
# 分层：能力矩阵（CapabilityProfile）是**纯能力声明**（不绑定 endpoint URL / 凭据）；
# 连接预设（prompt_director 的 _CLOUD_DIRECTOR_PRESETS）持有 URL / credential_id /
# environment_variable / json_mode / api_failure_policy 等**连接与策略**字段。
# 本映射把两者绑定：preset_id 只能引用到已在 registry 注册的 profile_id，避免
# 「连接层声称支持某种能力但能力层没有对应 profile」的漂移。prompt_director 通过
# `profile_for_preset()` 拿到能力 profile，从而让能力决策以 registry 为单一来源。
# ============================================================================

# preset_id（连接预设键，如 "deepseek"）→ registry profile_id。
_CLOUD_PRESET_PROFILE_MAP: dict[str, str] = {
    "deepseek": "deepseek_chat_v4_flash_vision",
    "gemini": "gemini_generate_content_flash_lite",
    # 2026-08-28：常用云端视觉模型预设（documented；GLM 待用户实测后升档）
    "doubao": "volcengine_doubao_seed_evolving_documented",
    "claude": "anthropic_claude_sonnet46_documented",
    "glm": "zhipu_glm53_flash_documented",
    "minimax": "minimax_m27_text_documented",
    "custom": "custom_openai_compat_documented",
    # 我的预设（用户保存的 OpenAI 兼容预设集）复用 custom 能力 profile。
    "my_presets": "custom_openai_compat_documented",
}


def all_cloud_preset_ids() -> tuple[str, ...]:
    """当前受支持的云端连接预设 id（顺序稳定，用于节点下拉）。"""
    return tuple(_CLOUD_PRESET_PROFILE_MAP.keys())


def profile_for_preset(preset_id: str) -> CapabilityProfile:
    """给定一个云端连接预设 id，返回其绑定的能力 profile。

    Raises:
        KeyError: 该 preset_id 未注册（连接层引用了一个未知/未登记能力）。
    """
    profile_id = _CLOUD_PRESET_PROFILE_MAP.get(str(preset_id).strip().lower())
    if profile_id is None:
        raise KeyError(f"unknown cloud preset: {preset_id!r}")
    return get_profile(profile_id)


def find_profile_for_model(model_name: str) -> CapabilityProfile | None:
    """按模型名匹配已注册能力 profile（用于 thinking/媒体决策咨询 registry）。

    这是「业务层只写统一语义、能力事实以 registry 为准」的入口之一：当调用方
    只有模型名（例如 `_resolved_reasoning_mode` 只有 `model` 参数）时，用它定位
    到 `CapabilityProfile`，从而拿到 `ReasoningSpec`（能否关、支持哪些 level）。
    匹配不到（本地 Qwen/Gemma/LLM 等未登记模型）返回 None，由调用方保留原逻辑。
    """
    name = str(model_name or "").casefold()
    if not name:
        return None
    # 精确匹配 profile.model 优先；否则按 provider 前缀/包含兜底（避免歧义）。
    for profile in _CAPABILITY_PROFILES.values():
        if str(profile.model).casefold() == name:
            return profile
    candidates = [
        p for p in _CAPABILITY_PROFILES.values()
        if name.startswith(str(p.provider).casefold())
        or str(p.provider).casefold() in name
    ]
    return candidates[0] if len(candidates) == 1 else None


def preset_binding(preset_id: str) -> dict[str, Any]:
    """连接预设↔能力绑定的可序列化摘要（供报告/诊断，不含 URL/凭据）。"""
    profile = profile_for_preset(preset_id)
    return {
        "preset_id": str(preset_id).strip().lower(),
        "capability_profile_id": profile.profile_id,
        "provider": profile.provider,
        "wire": profile.wire,
        "model": profile.model,
        "evidence": profile.evidence.value,
    }


# ============================================================================
# Contract 层：验证「某个 profile 能否承载某个 DirectorRequest」。
# 这些是纯函数（不联网），供 contract test 与后续 adapter 复用。
# ============================================================================

@dataclass(frozen=True)
class DirectorRequestContract:
    """业务层统一请求：只含与 provider 无关的抽象，交由 adapter 翻译。"""

    task_type: str
    num_reference_images: int
    num_video_frames: int
    wants_native_video: bool = False
    needs_structured_output: bool = True
    wants_reasoning: str = "auto"
    audio_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "num_reference_images": self.num_reference_images,
            "num_video_frames": self.num_video_frames,
            "wants_native_video": self.wants_native_video,
            "needs_structured_output": self.needs_structured_output,
            "wants_reasoning": self.wants_reasoning,
            "audio_required": self.audio_required,
        }


def _check_contract(profile: CapabilityProfile, req: DirectorRequestContract) -> list[str]:
    """返回违反契约的列表；空列表=该 profile 能承载该请求。"""
    problems: list[str] = []

    if req.needs_structured_output and profile.structured_output.wire == "none":
        problems.append(f"{profile.profile_id}: 需要结构化输出但 wire=none")

    if req.wants_native_video and not profile.supports_native_video:
        problems.append(f"{profile.profile_id}: 需要原生视频输入但不支持（fallback=抽帧）")

    if req.audio_required and not profile.request_shape.audio_input:
        problems.append(f"{profile.profile_id}: 需要音频输入但不支持")

    if req.num_reference_images > 0 and not profile.supports_vision_images:
        problems.append(f"{profile.profile_id}: 需要多图但该 wire 不支持图片输入")

    limit = profile.media_limits.max_images_per_request
    if limit is not None and req.num_reference_images > limit:
        problems.append(
            f"{profile.profile_id}: 图片 {req.num_reference_images} > 上限 {limit}"
        )

    if profile.must_extract_frames and req.wants_native_video:
        problems.append(f"{profile.profile_id}: 无原生视频，必须抽帧（video_strategy=extracted_frames）")

    # 推理强度：业务层只在 REASONING_LEVELS 里选，adapter 负责翻译。
    if req.wants_reasoning not in REASONING_LEVELS:
        problems.append(f"{profile.profile_id}: 未知推理强度 {req.wants_reasoning!r}")

    if profile.reasoning.supported_levels and req.wants_reasoning not in profile.reasoning.supported_levels:
        problems.append(
            f"{profile.profile_id}: 推理强度 {req.wants_reasoning} 不在 {profile.reasoning.supported_levels}"
        )

    return problems


def find_planable_profiles(req: DirectorRequestContract) -> list[dict[str, Any]]:
    """所有能承载该请求的 profile（按 evidence 排序：recommended > measured > ...）。"""
    order = {
        EvidenceStatus.RECOMMENDED: 0,
        EvidenceStatus.MEASURED: 1,
        EvidenceStatus.CONTRACT_PASSED: 2,
        EvidenceStatus.DOCUMENTED: 3,
    }
    planable: list[CapabilityProfile] = []
    for profile in _CAPABILITY_PROFILES.values():
        if not _check_contract(profile, req):
            planable.append(profile)
    planable.sort(key=lambda p: order[p.evidence])
    return [p.to_dict() for p in planable]


def contract_result(profile_id: str, req: DirectorRequestContract) -> dict[str, Any]:
    profile = get_profile(profile_id)
    problems = _check_contract(profile, req)
    return {
        "profile_id": profile_id,
        "ok": not problems,
        "problems": problems,
    }


# ============================================================================
# 业务层统一语义解析（Thinking / Structured Output 的决策层）。
# 这些函数只回答「该 wire 支持什么、业务层语义该怎么归一化」，不做具体字段翻译；
# 具体 wire 字段（thinking:{type} / thinkingLevel / reasoning_effort …）仍由
# prompt_director 的 adapter 负责，符合「业务层只写统一语义」的定论。
# ============================================================================

# 业务层 UI 目前只暴露 on/off/auto（两态 + auto），但 REASONING_LEVELS 是全量统一语义。
_UI_REASONING_ALIASES = {"on": "high", "enabled": "high", "high": "high"}


def normalize_reasoning_semantic(api_reasoning: str) -> str:
    """把业务层/UI 的输入归一化到全量统一语义（off/low/medium/high/max/auto）。

    - ''/None → auto
    - 'on' → high（"开思考"的业务意图最接近 high，且几乎所有 wire 支持）
    - 'off' 与 REASONING_LEVELS 合法值原样保留
    - 其它未知值返回 'auto'（不抛错，交由后续 contract 校验）
    """
    raw = str(api_reasoning or "auto").strip().casefold()
    if raw == "":
        return "auto"
    if raw in REASONING_LEVELS:
        return raw
    return _UI_REASONING_ALIASES.get(raw, "auto")


def resolve_reasoning_semantic(
    profile: CapabilityProfile,
    api_reasoning: str,
    *,
    model_name: str = "",
    request_purpose: str = "final",
    default_when_auto: str = "auto",
) -> str:
    """基于 profile 的 ReasoningSpec 解析出**该 wire 应使用的统一语义**。

    这是 `_resolved_reasoning_mode` / `_gemini_thinking_level` 等散落逻辑的
    单一来源：业务层传入的 off/low/medium/high/max/auto 先在 REASONING_LEVELS
    归一化，再按 profile 能力做两件约束——(1) 若该 wire 无法真正关闭思考且请求
    希望 off，则抬到该 wire 的最低非 off 可用级别；(2) 显式级别若不在该 wire
    的 supported_levels，clamp 到最近可用级别。auto 不做语义改写（保持调用方
    意图，由 adapter 按官方默认处理），避免隐式改变现有行为的默认导向。
    具体 wire 字段翻译仍由 adapter 负责，不在此处。

    Args:
        profile: 目标能力 profile（provider × wire × model）。
        api_reasoning: 业务层/UI 输入（off/low/…/auto 或 on/空串）。
        model_name: 模型名（当前未用于本函数，保留以对齐调用方签名）。
        request_purpose: "analysis"（识图/素材理解）或 "final"（Prompt IR 编译）。
        default_when_auto: auto 时若调用方希望收敛到具体值，传该默认级；否则 auto 透传。
    """
    semantic = normalize_reasoning_semantic(api_reasoning)
    spec = profile.reasoning

    # auto：不改写语义（保持调用方/UI 的 auto 意图），只让调用方决定默认导向。
    # 若调用方给 default_when_auto 或 profile 声明 default_level，才可能收敛到具体值；
    # 否则原样返回 "auto"，由 adapter 按官方默认处理。这样 `_resolved_reasoning_mode`
    # 等现有调用不会因本函数引入隐式的 analysis/final 语义变化。
    if semantic == "auto":
        return default_when_auto or spec.default_level or "auto"

    # off：该 wire 无法真正关闭时，抬到最低的非 off 可用级别（否则仍给 off）。
    if semantic == "off":
        if spec.can_disable:
            return "off"
        return _lowest_on_level(spec)

    # 显式级别：若不在 supported_levels，取最近可用（min-clamp 到最低）。
    if spec.supported_levels and semantic not in spec.supported_levels:
        return _closest_supported_level(spec, semantic)
    # 未声明 supported_levels 时视为全量支持。
    return semantic


def _level_rank(level: str) -> int:
    order = ("off", "low", "medium", "high", "max")
    try:
        return order.index(level)
    except ValueError:
        # auto 作为中间默认（排在 low 之后、high 之前），永不在此被解析为最终。
        return order.index("medium")


def _lowest_on_level(spec: ReasoningSpec) -> str:
    """最低的"开启思考"级别（排除 off），供无法关闭思考的 wire 抬升用。"""
    ordered = [
        lv for lv in ("off", "low", "medium", "high", "max")
        if lv in spec.supported_levels and lv != "off"
    ]
    return ordered[0] if ordered else "low"


def _closest_supported_level(spec: ReasoningSpec, semantic: str) -> str:
    ordered = [
        lv for lv in ("off", "low", "medium", "high", "max")
        if lv in spec.supported_levels
    ]
    if not ordered:
        return "medium" if spec.supported_levels else semantic
    target = _level_rank(semantic)
    # clamp 到最近支持级别（不跨越 max 之外）。
    return min(ordered, key=lambda lv: abs(_level_rank(lv) - target))


def structured_output_wire(profile: CapabilityProfile) -> str:
    """该 wire 当前应使用的结构化输出 wire（业务层适配入口）。"""
    return profile.structured_output.wire


def reasoning_report(profile: CapabilityProfile, semantic: str) -> str:
    """给 report 的人类可读说明（仅统一语义，不含厂商字段细节）。"""
    return f"{profile.provider}/{profile.wire}: 推理语义={semantic}（{profile.evidence.value}）"
