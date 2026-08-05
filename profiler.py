"""Workflow-scoped MiniMax H3 sampling profiler.

SPDX-License-Identifier: GPL-3.0-only
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

import comfy.patcher_extension


LOGGER = logging.getLogger(__name__)
PATCH_KEY = "minimax_h3_lab_profiler"


@dataclass
class _ProfileRun:
    expected_steps: int = 0
    wall_started: float = 0.0
    cpu_call_times: list[float] = field(default_factory=list)
    cuda_event_pairs: list[tuple[Any, Any]] = field(default_factory=list)
    call_shapes: list[str] = field(default_factory=list)
    failures: int = 0
    pre_sample_allocated_gib: float = 0.0
    pre_sample_reserved_gib: float = 0.0

    def reset(self, expected_steps: int) -> None:
        self.expected_steps = expected_steps
        self.wall_started = time.perf_counter()
        self.cpu_call_times.clear()
        self.cuda_event_pairs.clear()
        self.call_shapes.clear()
        self.failures = 0
        self.pre_sample_allocated_gib = 0.0
        self.pre_sample_reserved_gib = 0.0


class _H3ProfileScope:
    def __init__(
        self,
        use_cuda_events: bool,
        detailed_log: bool,
        profile_context: dict[str, Any] | None = None,
    ) -> None:
        self.use_cuda_events = bool(use_cuda_events)
        self.detailed_log = bool(detailed_log)
        self.profile_context = profile_context if isinstance(profile_context, dict) else None
        self.run = _ProfileRun()

    @staticmethod
    def _find_sigmas(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor | None:
        candidate = kwargs.get("sigmas")
        if isinstance(candidate, torch.Tensor) and candidate.ndim == 1:
            return candidate
        if len(args) > 3 and isinstance(args[3], torch.Tensor) and args[3].ndim == 1:
            return args[3]
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.ndim == 1 and arg.numel() > 1:
                return arg
        return None

    @staticmethod
    def _shape_summary(value: Any) -> str:
        if isinstance(value, torch.Tensor):
            return str(tuple(value.shape))
        if isinstance(value, (list, tuple)):
            shapes = [str(tuple(item.shape)) for item in value if isinstance(item, torch.Tensor)]
            return " + ".join(shapes) if shapes else type(value).__name__
        return type(value).__name__

    def model_wrapper(self, executor: Callable, *args: Any, **kwargs: Any) -> Any:
        use_events = self.use_cuda_events and torch.cuda.is_available()
        started = time.perf_counter()
        start_event = end_event = None
        if use_events:
            try:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            except Exception:
                start_event = end_event = None
                use_events = False
        try:
            return executor(*args, **kwargs)
        except Exception:
            self.run.failures += 1
            raise
        finally:
            elapsed = time.perf_counter() - started
            self.run.cpu_call_times.append(elapsed)
            self.run.call_shapes.append(self._shape_summary(args[0]) if args else "unknown")
            if use_events and start_event is not None and end_event is not None:
                end_event.record()
                self.run.cuda_event_pairs.append((start_event, end_event))
            if self.detailed_log:
                LOGGER.info(
                    "MiniMax H3 Profiler: model call %d queued in %.3fs, input %s",
                    len(self.run.cpu_call_times),
                    elapsed,
                    self.run.call_shapes[-1],
                )

    def _finish_report(self) -> str:
        wall_seconds = max(0.0, time.perf_counter() - self.run.wall_started)
        gpu_times: list[float] = []
        if self.run.cuda_event_pairs:
            try:
                torch.cuda.synchronize()
                gpu_times = [start.elapsed_time(end) / 1000.0 for start, end in self.run.cuda_event_pairs]
            except Exception as exc:
                LOGGER.warning("MiniMax H3 Profiler could not resolve CUDA events: %s", exc)

        times = gpu_times or self.run.cpu_call_times
        calls = len(times)
        if times:
            total_model = sum(times)
            average = statistics.fmean(times)
            minimum = min(times)
            maximum = max(times)
        else:
            total_model = average = minimum = maximum = 0.0

        peak_allocated = peak_reserved = current_allocated = 0.0
        if torch.cuda.is_available():
            try:
                divisor = 1024.0**3
                peak_allocated = torch.cuda.max_memory_allocated() / divisor
                peak_reserved = torch.cuda.max_memory_reserved() / divisor
                current_allocated = torch.cuda.memory_allocated() / divisor
            except Exception:
                pass

        source = "CUDA events" if gpu_times else "CPU wall submissions"
        shape_text = self.run.call_shapes[0] if self.run.call_shapes else "unknown"
        pre_sampling_seconds = None
        if self.profile_context is not None:
            started_at = self.profile_context.get("started_at")
            if isinstance(started_at, (int, float)):
                pre_sampling_seconds = max(0.0, self.run.wall_started - float(started_at))
        lines = [
            "MiniMax H3 性能分析报告",
            (
                f"检查器结束→采样开始：{pre_sampling_seconds:.2f}秒（主要包含官方参考VAE/文本条件构建）"
                if pre_sampling_seconds is not None
                else "检查器结束→采样开始：未连接 profile_context，无法计时"
            ),
            f"采样墙钟时间：{wall_seconds:.2f}秒",
            f"预计采样步数：{self.run.expected_steps}；进入分析器的模型调用：{calls}",
            (
                f"模型调用计时（{source}）：合计{total_model:.2f}秒，平均{average:.2f}秒，"
                f"最快{minimum:.2f}秒，最慢{maximum:.2f}秒"
            ),
            (
                f"采样开始前CUDA：已分配{self.run.pre_sample_allocated_gib:.2f} GiB，"
                f"已保留{self.run.pre_sample_reserved_gib:.2f} GiB"
            ),
            (
                f"采样阶段CUDA峰值：已分配{peak_allocated:.2f} GiB，峰值保留{peak_reserved:.2f} GiB，"
                f"结束时已分配{current_allocated:.2f} GiB"
            ),
            f"首个模型输入：{shape_text}；异常调用：{self.run.failures}",
            (
                "说明：若串联官方EasyCache，缓存命中的调用可能在进入本分析器前直接返回；"
                "“采样墙钟时间”仍包含这些命中步骤，模型调用数则取决于补丁顺序。"
            ),
        ]
        report = "\n".join(lines)
        LOGGER.info("\n%s", report)
        print("\n" + report + "\n")
        return report

    def sample_wrapper(self, executor: Callable, *args: Any, **kwargs: Any) -> Any:
        sigmas = self._find_sigmas(args, kwargs)
        expected_steps = max(0, int(sigmas.numel()) - 1) if sigmas is not None else 0
        self.run.reset(expected_steps)
        if torch.cuda.is_available():
            try:
                divisor = 1024.0**3
                self.run.pre_sample_allocated_gib = torch.cuda.memory_allocated() / divisor
                self.run.pre_sample_reserved_gib = torch.cuda.memory_reserved() / divisor
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        LOGGER.info(
            "MiniMax H3 Profiler enabled: expected steps %d, CUDA events %s",
            expected_steps,
            self.use_cuda_events,
        )
        try:
            return executor(*args, **kwargs)
        finally:
            self._finish_report()


class MiniMaxH3PerformanceProfiler:
    """Patch an H3 model with run-scoped timing and CUDA-memory telemetry."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model": ("MODEL",),
                "use_cuda_events": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Accurate GPU timing with one synchronization at the end of sampling.",
                    },
                ),
                "detailed_log": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Log every model call as well as the final summary."},
                ),
            },
            "optional": {
                "profile_context": (
                    "H3_PROFILE_CONTEXT",
                    {
                        "tooltip": "Connect Reference Inspector to include native conditioning/VAE time."
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3 Lab/Diagnostics"
    DESCRIPTION = (
        "Measure sampler wall time, diffusion-model calls, per-call GPU time, and CUDA peak memory. "
        "The final report is printed to the ComfyUI console after sampling."
    )

    def apply(
        self,
        model: Any,
        use_cuda_events: bool,
        detailed_log: bool,
        profile_context: dict[str, Any] | None = None,
    ) -> tuple[Any]:
        patched = model.clone()
        diffusion_model = getattr(getattr(patched, "model", None), "diffusion_model", None)
        if diffusion_model is None or diffusion_model.__class__.__name__ != "MiniMaxH3Model":
            actual = diffusion_model.__class__.__name__ if diffusion_model is not None else "unknown"
            raise ValueError(f"MiniMax H3 Performance Profiler only supports MiniMaxH3Model; received {actual}")

        scope = _H3ProfileScope(
            use_cuda_events=use_cuda_events,
            detailed_log=detailed_log,
            profile_context=profile_context,
        )
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            PATCH_KEY,
            scope.model_wrapper,
        )
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            PATCH_KEY,
            scope.sample_wrapper,
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PerformanceProfiler": MiniMaxH3PerformanceProfiler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PerformanceProfiler": "MiniMax H3 Performance Profiler",
}
