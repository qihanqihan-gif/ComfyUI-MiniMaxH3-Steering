"""Reference-media preparation and diagnostics for MiniMax H3.

These nodes intentionally operate before ComfyUI's native
``MiniMaxH3ReferenceToVideo`` node.  They do not copy or replace the official
conditioning/VAE implementation.

SPDX-License-Identifier: GPL-3.0-only
"""

from __future__ import annotations

import ctypes
import math
import time
from typing import Any

import torch
import torch.nn.functional as F


H3_MULTIPLE = 32
H3_FPS = 24
H3_AUDIO_LATENT_FPS = 40
H3_HIDDEN_SIZE = 5376


def _mib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def _tensor_mib(value: Any) -> float:
    if not isinstance(value, torch.Tensor):
        return 0.0
    return _mib(value.numel() * value.element_size())


def _align_axis(value: int, multiple: int = H3_MULTIPLE) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _aligned_frame_count_down(frame_count: int) -> int:
    """Largest H3-compatible 17k+5 frame count not above frame_count."""
    if frame_count < 5:
        return frame_count
    return frame_count - ((frame_count - 5) % 17)


def _aligned_frame_count_up(frame_count: int) -> int:
    """Smallest H3-compatible 17k+5 frame count not below frame_count."""
    frame_count = max(5, frame_count)
    remainder = (frame_count - 5) % 17
    return frame_count if remainder == 0 else frame_count + (17 - remainder)


def _video_latent_t(frame_count: int) -> int:
    frame_count = max(5, _aligned_frame_count_down(frame_count))
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def _resize_chunk(chunk: torch.Tensor, width: int, height: int) -> torch.Tensor:
    try:
        return F.interpolate(
            chunk,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    except TypeError:
        # ``antialias`` is unavailable on older torch builds.
        return F.interpolate(chunk, size=(height, width), mode="bicubic", align_corners=False)


def _resize_in_chunks(images: torch.Tensor, width: int, height: int, chunk_size: int) -> torch.Tensor:
    chunks = []
    for start in range(0, images.shape[0], chunk_size):
        chunks.append(_resize_chunk(images[start : start + chunk_size], width, height))
    return torch.cat(chunks, dim=0)


def _sample_frames(images: torch.Tensor, count: int, strategy: str) -> tuple[torch.Tensor, list[int]]:
    total = images.shape[0]
    if count >= total:
        indices = list(range(total))
    elif strategy == "head":
        indices = list(range(count))
    else:
        # Uniform selection always preserves both temporal endpoints.
        sampled = torch.linspace(0, total - 1, count, dtype=torch.float64).round().long()
        indices = sampled.tolist()
    index = torch.tensor(indices, device=images.device, dtype=torch.long)
    return images.index_select(0, index), indices


def _prepare_geometry(
    images: torch.Tensor,
    target_width: int,
    target_height: int,
    resize_mode: str,
    allow_upscale: bool,
    pad_value: float,
    chunk_size: int,
) -> tuple[torch.Tensor, int, int, str]:
    # ComfyUI IMAGE is [frames, height, width, channels].  H3 consumes RGB.
    source = images[..., :3].movedim(-1, 1)
    source_h, source_w = int(source.shape[2]), int(source.shape[3])
    target_width = _align_axis(target_width)
    target_height = _align_axis(target_height)

    if resize_mode == "preserve_area":
        scale = math.sqrt((target_width * target_height) / max(1, source_w * source_h))
        if not allow_upscale:
            scale = min(1.0, scale)
        out_w = _align_axis(max(H3_MULTIPLE, round(source_w * scale)))
        out_h = _align_axis(max(H3_MULTIPLE, round(source_h * scale)))
        prepared = _resize_in_chunks(source, out_w, out_h, chunk_size)
        detail = "完整等比缩放到目标像素面积，不补边、不裁剪"
    elif resize_mode == "stretch":
        out_w, out_h = target_width, target_height
        prepared = _resize_in_chunks(source, out_w, out_h, chunk_size)
        detail = "直接拉伸到生成画布"
    else:
        if resize_mode == "cover_crop":
            scale = max(target_width / source_w, target_height / source_h)
        else:
            scale = min(target_width / source_w, target_height / source_h)
        if not allow_upscale:
            scale = min(1.0, scale)
        scaled_w = max(1, round(source_w * scale))
        scaled_h = max(1, round(source_h * scale))
        scaled = _resize_in_chunks(source, scaled_w, scaled_h, chunk_size)
        out_w, out_h = target_width, target_height

        if resize_mode == "cover_crop":
            left = max(0, (scaled_w - out_w) // 2)
            top = max(0, (scaled_h - out_h) // 2)
            prepared = scaled[:, :, top : top + out_h, left : left + out_w]
            # With upscaling disabled, a small image may not cover the canvas.
            if prepared.shape[2:] != (out_h, out_w):
                canvas = torch.full(
                    (scaled.shape[0], scaled.shape[1], out_h, out_w),
                    float(pad_value),
                    dtype=scaled.dtype,
                    device=scaled.device,
                )
                ph, pw = prepared.shape[2:]
                canvas[:, :, :ph, :pw] = prepared
                prepared = canvas
            detail = "等比覆盖后居中裁剪"
        else:
            canvas = torch.full(
                (scaled.shape[0], scaled.shape[1], out_h, out_w),
                float(pad_value),
                dtype=scaled.dtype,
                device=scaled.device,
            )
            left = (out_w - scaled_w) // 2
            top = (out_h - scaled_h) // 2
            canvas[:, :, top : top + scaled_h, left : left + scaled_w] = scaled
            prepared = canvas
            detail = "完整等比缩放后居中补边，不裁剪"

    return prepared.movedim(1, -1).contiguous(), out_w, out_h, detail


class MiniMaxH3ReferenceMediaPrep:
    """Resize and temporally constrain one image or one video-frame batch."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "images": ("IMAGE",),
                "media_kind": (["reference_image", "reference_video"], {"default": "reference_image"}),
                "target_width": (
                    "INT",
                    {"default": 864, "min": 32, "max": 8192, "step": 32},
                ),
                "target_height": (
                    "INT",
                    {"default": 480, "min": 32, "max": 8192, "step": 32},
                ),
                "resize_mode": (
                    ["contain_pad", "preserve_area", "stretch", "cover_crop"],
                    {"default": "contain_pad"},
                ),
                "allow_upscale": ("BOOLEAN", {"default": False}),
                "max_frames": (
                    "INT",
                    {
                        "default": 124,
                        "min": 0,
                        "max": 3600,
                        "step": 1,
                        "tooltip": "0 keeps all input frames. Video output is then aligned down to 17k+5.",
                    },
                ),
                "frame_selection": (["head", "uniform"], {"default": "head"}),
                "pad_value": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "resize_chunk_size": (
                    "INT",
                    {"default": 16, "min": 1, "max": 256, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "width", "height", "frames", "report")
    FUNCTION = "prepare"
    CATEGORY = "MiniMax H3 Lab/Reference"
    DESCRIPTION = (
        "Prepare one H3 reference image or video batch. The default keeps the complete frame, "
        "adds letterboxing, limits video frames, and aligns videos to the 17k+5 frame grid."
    )

    def prepare(
        self,
        images: torch.Tensor,
        media_kind: str,
        target_width: int,
        target_height: int,
        resize_mode: str,
        allow_upscale: bool,
        max_frames: int,
        frame_selection: str,
        pad_value: float,
        resize_chunk_size: int,
    ) -> tuple[torch.Tensor, int, int, int, str]:
        started_at = time.perf_counter()
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("MiniMax H3 Reference Media Prep expects IMAGE [frames, height, width, channels]")
        if images.shape[0] < 1 or images.shape[-1] < 3:
            raise ValueError("Reference media is empty or has fewer than three color channels")

        source_frames = int(images.shape[0])
        source_h, source_w = int(images.shape[1]), int(images.shape[2])
        selected_indices = [0]
        if media_kind == "reference_image":
            selected = images[:1]
        else:
            requested = source_frames if max_frames == 0 else min(source_frames, max_frames)
            requested = _aligned_frame_count_down(requested)
            if requested < 5:
                raise ValueError("MiniMax H3 reference video needs at least 5 frames")
            selected, selected_indices = _sample_frames(images, requested, frame_selection)

        prepared, out_w, out_h, detail = _prepare_geometry(
            selected,
            target_width,
            target_height,
            resize_mode,
            allow_upscale,
            pad_value,
            max(1, resize_chunk_size),
        )
        output_frames = int(prepared.shape[0])
        temporal = ""
        if media_kind == "reference_video":
            temporal = (
                f"\n时间处理：{source_frames}帧 → {output_frames}帧，"
                f"策略={frame_selection}，范围={selected_indices[0]}..{selected_indices[-1]}，"
                "已对齐17k+5。"
            )
            if frame_selection == "uniform" and output_frames < source_frames:
                temporal += " 均匀抽帧覆盖了完整范围，但官方仍按24 FPS解释，动作速度会相应加快。"
        elif source_frames > 1:
            temporal = f"\n输入批次包含{source_frames}张图；参考图片模式只输出第1张。"

        report = (
            "MiniMax H3 参考素材预处理\n"
            f"类型：{media_kind}\n"
            f"源尺寸：{source_w}×{source_h}；输出：{out_w}×{out_h}\n"
            f"几何策略：{detail}；允许放大={allow_upscale}；批处理={resize_chunk_size}帧"
            f"{temporal}\n"
            f"节点处理时间：{time.perf_counter() - started_at:.2f}秒\n"
            "提示：官方参考节点仍会进行一次尺寸检查，但尺寸已对齐时通常不会再次改变几何。"
        )
        return prepared, out_w, out_h, output_frames, report


class MiniMaxH3ReferenceInspector:
    """Inspect one image batch, one video batch and one audio reference."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "generation_width": (
                    "INT",
                    {"default": 864, "min": 32, "max": 8192, "step": 32},
                ),
                "generation_height": (
                    "INT",
                    {"default": 480, "min": 32, "max": 8192, "step": 32},
                ),
                "generation_frames": (
                    "INT",
                    {"default": 124, "min": 5, "max": 3600, "step": 1},
                ),
                "strict_mode": (["report_only", "raise_on_danger"], {"default": "report_only"}),
            },
            "optional": {
                "reference_images": ("IMAGE",),
                "reference_video": ("IMAGE",),
                "reference_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "AUDIO", "STRING", "H3_PROFILE_CONTEXT")
    RETURN_NAMES = (
        "reference_images",
        "reference_video",
        "reference_audio",
        "report",
        "profile_context",
    )
    FUNCTION = "inspect"
    CATEGORY = "MiniMax H3 Lab/Reference"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Report dimensions, frames, raw tensor memory, estimated H3 reference tokens, and likely "
        "risk before connecting media to the native H3 reference-conditioning node."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        # Refresh available RAM/VRAM and the profiling start marker every run.
        return float("nan")

    @staticmethod
    def _system_memory() -> tuple[float | None, float | None]:
        try:
            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                gib = 1024.0**3
                return status.ullAvailPhys / gib, status.ullTotalPhys / gib
        except Exception:
            pass
        return None, None

    @staticmethod
    def _gpu_memory() -> tuple[float | None, float | None]:
        try:
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                gib = 1024.0**3
                return free / gib, total / gib
        except Exception:
            pass
        return None, None

    def inspect(
        self,
        generation_width: int,
        generation_height: int,
        generation_frames: int,
        strict_mode: str,
        reference_images: torch.Tensor | None = None,
        reference_video: torch.Tensor | None = None,
        reference_audio: dict | None = None,
    ) -> dict[str, Any]:
        lines = ["MiniMax H3 参考素材检查报告"]
        warnings: list[str] = []
        dangers: list[str] = []
        raw_mib = 0.0
        ref_visual_tokens = 0
        ref_audio_tokens = 0

        # Native H3 snaps generation length upward, while reference video is
        # truncated downward after it is capped to that generation length.
        generation_frames_aligned = _aligned_frame_count_up(generation_frames)
        target_video_tokens = (
            _video_latent_t(generation_frames_aligned)
            * max(1, generation_height // H3_MULTIPLE)
            * max(1, generation_width // H3_MULTIPLE)
        )
        duration = generation_frames_aligned / H3_FPS
        target_audio_tokens = 2 * round(duration * H3_AUDIO_LATENT_FPS)
        lines.append(
            f"生成目标：{generation_width}×{generation_height}，{generation_frames_aligned}帧，"
            f"目标视频Token约{target_video_tokens:,}（不含文本），音频Token约{target_audio_tokens:,}。"
        )

        if isinstance(reference_images, torch.Tensor):
            if reference_images.ndim != 4:
                dangers.append(f"参考图片张量维度异常：{tuple(reference_images.shape)}")
            else:
                count, h, w = map(int, reference_images.shape[:3])
                raw_mib += _tensor_mib(reference_images)
                ref_visual_tokens += max(1, h // H3_MULTIPLE) * max(1, w // H3_MULTIPLE)
                lines.append(
                    f"参考图片：输入批次{count}张，{w}×{h}，原始张量{_tensor_mib(reference_images):.1f} MiB；"
                    f"单个参考块约{max(1, h // H3_MULTIPLE) * max(1, w // H3_MULTIPLE):,} Token。"
                )
                if count > 1:
                    warnings.append("一个官方ref_image接口只读取批次第1张；多张图片应分别连接到多个ref_image接口。")
                if h % H3_MULTIPLE or w % H3_MULTIPLE:
                    warnings.append("参考图片宽高不是32倍数，官方节点会再次调整尺寸。")

        if isinstance(reference_video, torch.Tensor):
            if reference_video.ndim != 4:
                dangers.append(f"参考视频张量维度异常：{tuple(reference_video.shape)}")
            else:
                frames, h, w = map(int, reference_video.shape[:3])
                raw_video_mib = _tensor_mib(reference_video)
                raw_mib += raw_video_mib
                effective_frames = min(frames, generation_frames_aligned)
                aligned = _aligned_frame_count_down(effective_frames)
                if aligned >= 5:
                    video_tokens = (
                        _video_latent_t(aligned)
                        * max(1, h // H3_MULTIPLE)
                        * max(1, w // H3_MULTIPLE)
                    )
                    ref_visual_tokens += video_tokens
                else:
                    video_tokens = 0
                lines.append(
                    f"参考视频：{frames}帧，{w}×{h}，原始张量{raw_video_mib:.1f} MiB；"
                    f"H3将使用约{aligned}帧/{video_tokens:,} Token。"
                )
                if frames < 5:
                    dangers.append("参考视频少于5帧，官方节点会直接报错。")
                if frames > generation_frames_aligned:
                    warnings.append(
                        f"参考视频长于生成目标，官方节点还会先截到最多{generation_frames_aligned}帧。"
                    )
                if aligned != effective_frames:
                    warnings.append(f"参考视频有效帧数不是17k+5，官方节点会从尾部截成{aligned}帧。")
                if h % H3_MULTIPLE or w % H3_MULTIPLE:
                    warnings.append("参考视频宽高不是32倍数，官方节点会重新适配画布。")
                if raw_video_mib > 1024:
                    dangers.append("未编码参考视频张量已超过1 GiB，系统内存和VAE编码峰值风险很高。")
                elif raw_video_mib > 256:
                    warnings.append("未编码参考视频张量超过256 MiB，VAE编码期间会产生更大的临时张量。")

        if isinstance(reference_audio, dict) and isinstance(reference_audio.get("waveform"), torch.Tensor):
            waveform = reference_audio["waveform"]
            sample_rate = int(reference_audio.get("sample_rate", 0) or 0)
            samples = int(waveform.shape[-1]) if waveform.ndim >= 1 else 0
            audio_duration = samples / sample_rate if sample_rate > 0 else 0.0
            raw_audio_mib = _tensor_mib(waveform)
            raw_mib += raw_audio_mib
            ref_audio_tokens = 2 * round(audio_duration * H3_AUDIO_LATENT_FPS)
            lines.append(
                f"参考音频：{audio_duration:.2f}秒，{sample_rate} Hz，原始张量{raw_audio_mib:.1f} MiB；"
                f"音频Token约{ref_audio_tokens:,}。"
            )
            if audio_duration > 15:
                warnings.append("参考音频超过15秒；长音频会持续增加条件Token和编码内存。")

        ref_ratio = ref_visual_tokens / max(1, target_video_tokens)
        hidden_equivalent_mib = _mib((ref_visual_tokens + ref_audio_tokens) * H3_HIDDEN_SIZE * 2)
        lines.append(
            f"合计：原始输入张量{raw_mib:.1f} MiB；参考视觉Token约{ref_visual_tokens:,}，"
            f"相当于目标视频Token的{ref_ratio:.2f}倍。"
        )
        lines.append(
            f"参考Token的一份FP16/BF16隐藏状态等效尺寸约{hidden_equivalent_mib:.1f} MiB；"
            "这不是峰值显存，注意力和MLP还会产生额外临时张量。"
        )

        if ref_ratio > 1.5:
            dangers.append("参考视觉Token超过目标视频的1.5倍，会在全部50层中持续增加计算和显存压力。")
        elif ref_ratio > 0.75:
            warnings.append("参考视觉Token已接近目标视频本身，采样速度可能明显下降。")

        ram_free, ram_total = self._system_memory()
        gpu_free, gpu_total = self._gpu_memory()
        if ram_free is not None:
            lines.append(f"执行前系统内存：可用{ram_free:.1f}/{ram_total:.1f} GiB。")
            if ram_free < 12:
                warnings.append("当前可用物理内存低于12 GiB，加载文本编码器或参考视频时容易进入页面文件。")
        if gpu_free is not None:
            lines.append(f"执行前CUDA显存：可用{gpu_free:.1f}/{gpu_total:.1f} GiB。")

        if dangers:
            level = "危险"
        elif warnings:
            level = "注意"
        else:
            level = "安全基线"
        lines.append(f"结论：{level}")
        lines.extend(f"[危险] {item}" for item in dangers)
        lines.extend(f"[注意] {item}" for item in warnings)
        if not dangers and not warnings:
            lines.append("未发现明显的尺寸、帧数或Token预算问题。")

        report = "\n".join(lines)
        print("\n" + report + "\n")
        if strict_mode == "raise_on_danger" and dangers:
            raise RuntimeError(report)
        return {
            "ui": {"text": [report]},
            "result": (
                reference_images,
                reference_video,
                reference_audio,
                report,
                {"started_at": time.perf_counter(), "reference_report": report},
            ),
        }


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ReferenceMediaPrep": MiniMaxH3ReferenceMediaPrep,
    "MiniMaxH3ReferenceInspector": MiniMaxH3ReferenceInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ReferenceMediaPrep": "MiniMax H3 Reference Media Prep",
    "MiniMaxH3ReferenceInspector": "MiniMax H3 Reference Inspector",
}
