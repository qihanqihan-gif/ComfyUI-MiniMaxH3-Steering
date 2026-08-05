"""Experimental MiniMax-H3 cache nodes.

The implementation intentionally uses ComfyUI's public ModelPatcher hooks.  It
does not replace ``comfy/ldm/minimax/model.py`` and contains no native binary.

SPDX-License-Identifier: GPL-3.0-only
"""

from __future__ import annotations

import inspect
import logging
import math
import functools
from dataclasses import dataclass
from typing import Any, Callable

import torch

import comfy.patcher_extension


LOGGER = logging.getLogger(__name__)
PATCH_KEY = "minimax_h3_lab_open_cache"

# Samplers that inject fresh noise / SDE stochasticity are not safe with
# residual reuse: the cached residual is only meaningful on the same
# trajectory.  Anything else (ancestral, SDE, low-step) is blocked.
_BLOCKED_SAMPLERS = frozenset({
    "euler_ancestral", "euler_ancestral_cfg_pp",
    "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu",
    "dpmpp_sde", "dpmpp_sde_gpu", "er_sde",
    "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
    "exp_heun_2_x0_sde", "dpm_2_ancestral", "lcm", "ddpm",
    # sa_solver/sa_solver_pece：随机 Adams Solver（k_diffusion docstring: Stochastic），每步加噪声
    "sa_solver", "sa_solver_pece",
    # seeds_2/seeds_3 每步注入噪声（k_diffusion sampling.py docstring: Stochastic），
    # 同祖先类风险；ddim 由 ComfyUI 映射为 euler(random=True)，函数名是 sample_euler
    # 且按 run 隔离、固定 seed 单 run 确定，故不在名单（条目无意义）。
    "seeds_2", "seeds_3",
})


@dataclass
class _RunStats:
    full_steps: int = 0
    skipped_steps: int = 0
    decisions: int = 0


class _WholeLoopCache:
    """Approximate the complete H3 transformer loop with one cached residual.

    ComfyUI exposes a replacement hook for each H3 ``double_block``.  On a
    full step we record the difference between the input of block 0 and the
    output of the last block.  On a cache hit block 0 applies that difference
    and all remaining blocks become no-ops for that model evaluation.
    """

    def __init__(
        self,
        block_count: int,
        threshold: float,
        start_percent: float,
        end_percent: float,
        max_consecutive_skips: int,
        cache_device: str,
        verbose: bool,
    ) -> None:
        self.block_count = block_count
        self.threshold = float(threshold)
        self.start_percent = float(start_percent)
        self.end_percent = float(end_percent)
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.cache_device = cache_device
        self.verbose = bool(verbose)
        self.total_steps = 0
        self._warning_emitted = False
        self.cfg_scale: float | None = None
        self.sampler_blocked = False
        self._cfg_warned = False
        self._sampler_warned = False
        self.reset()

    def reset(self, total_steps: int | None = None) -> None:
        if total_steps is not None:
            self.total_steps = max(1, int(total_steps))
        self.cached_residual: torch.Tensor | None = None
        self.previous_signature: torch.Tensor | None = None
        self.accumulated_change = 0.0
        self.consecutive_skips = 0
        self.step_index = -1
        self.layout_signature: tuple[Any, ...] | None = None
        self._step_input: torch.Tensor | None = None
        self._skip_active = False
        self.cfg_scale = None
        self.stats = _RunStats()

    def release_tensors(self) -> None:
        self.cached_residual = None
        self.previous_signature = None
        self._step_input = None
        self._skip_active = False

    @staticmethod
    def _target_ranges(mod_segments: Any, token_count: int) -> list[tuple[int, int]]:
        """Return target audio/video ranges, which H3 appends as the last two segments."""
        ranges: list[tuple[int, int]] = []
        if isinstance(mod_segments, (list, tuple)):
            for segment in mod_segments[-2:]:
                if not isinstance(segment, (list, tuple)) or len(segment) < 2:
                    continue
                start = max(0, min(token_count, int(segment[0])))
                end = max(start, min(token_count, int(segment[1])))
                if end > start:
                    ranges.append((start, end))
        return ranges or [(0, token_count)]

    @classmethod
    def _make_signature(cls, img: torch.Tensor, mod_segments: Any) -> torch.Tensor:
        if img.ndim != 2:
            raise ValueError(f"expected packed H3 states [tokens, channels], got {tuple(img.shape)}")

        pieces = []
        max_channels = min(64, img.shape[-1])
        for start, end in cls._target_ranges(mod_segments, img.shape[0]):
            stride = max(1, (end - start) // 128)
            sample = img[start:end:stride, :max_channels]
            if sample.shape[0] > 128:
                sample = sample[:128]
            pieces.append(sample.detach().float().abs().mean(dim=-1))

        # The signature is tiny.  Keeping it on the compute device avoids a
        # PCIe transfer; only the final scalar decision synchronizes.
        return torch.cat(pieces).clone()

    def _progress(self) -> float:
        if self.total_steps <= 1:
            return 0.0
        return min(1.0, max(0.0, self.step_index / (self.total_steps - 1)))

    def _invalidate_for_layout(self, layout_signature: tuple[Any, ...]) -> None:
        self.cached_residual = None
        self.previous_signature = None
        self.accumulated_change = 0.0
        self.consecutive_skips = 0
        self._step_input = None
        self._skip_active = False
        self.layout_signature = layout_signature

    def _relative_change(self, current: torch.Tensor) -> float:
        previous = self.previous_signature
        if previous is None or previous.shape != current.shape:
            return float("inf")
        numerator = (current - previous).abs().mean()
        denominator = previous.abs().mean().clamp_min(1e-6)
        value = float((numerator / denominator).item())
        if not math.isfinite(value):
            return float("inf")
        return value

    def _residual_matches(self, img: torch.Tensor) -> bool:
        residual = self.cached_residual
        return residual is not None and residual.shape == img.shape

    def _apply_residual(self, img: torch.Tensor) -> torch.Tensor:
        residual = self.cached_residual
        if residual is None:
            return img
        if residual.device != img.device or residual.dtype != img.dtype:
            residual = residual.to(device=img.device, dtype=img.dtype, non_blocking=True)
        return img + residual

    def _store_residual(self, residual: torch.Tensor) -> None:
        target = "cpu" if self.cache_device == "cpu" else residual.device
        try:
            self.cached_residual = residual.detach().to(target, non_blocking=True)
        except torch.cuda.OutOfMemoryError:
            LOGGER.warning("MiniMax H3 Open Cache: GPU cache allocation failed; falling back to CPU cache")
            self.cached_residual = residual.detach().to("cpu")

    def _log_decision(self, action: str, reason: str) -> None:
        if self.verbose:
            LOGGER.info(
                "MiniMax H3 Open Cache: step %d/%d %s - %s",
                self.step_index + 1,
                self.total_steps,
                action,
                reason,
            )

    def _begin_step(self, img: torch.Tensor, mod_segments: Any) -> torch.Tensor:
        self.step_index += 1
        self.stats.decisions += 1
        self._skip_active = False

        segments = mod_segments if isinstance(mod_segments, (list, tuple)) else []
        tail_segments = tuple(tuple(int(v) for v in segment[:2]) for segment in segments[-2:])
        layout_signature = (tuple(img.shape), img.dtype, img.device, tail_segments, self.block_count)
        if self.layout_signature != layout_signature:
            self._invalidate_for_layout(layout_signature)

        current_signature = self._make_signature(img, segments)
        relative_change = self._relative_change(current_signature)
        next_accumulated = self.accumulated_change + relative_change
        progress = self._progress()

        can_skip = (
            not self.sampler_blocked
            and (self.cfg_scale is None or math.isclose(self.cfg_scale, 1.0))
            and self._residual_matches(img)
            and self.previous_signature is not None
            and torch.isfinite(current_signature).all().item()
            and self.start_percent <= progress <= self.end_percent
            and next_accumulated < self.threshold
            and self.consecutive_skips < self.max_consecutive_skips
        )

        if self.sampler_blocked and not self._sampler_warned:
            self._sampler_warned = True
            LOGGER.warning(
                "MiniMax H3 Open Cache: sampler is on the blocked list (ancestral/SDE/low-step); "
                "cache disabled for this run to avoid corrupted residual reuse."
            )
        if self.cfg_scale is not None and not math.isclose(self.cfg_scale, 1.0) and not self._cfg_warned:
            self._cfg_warned = True
            LOGGER.warning(
                "MiniMax H3 Open Cache: CFG scale %.2f > 1.0 detected; positive/negative branches "
                "share the same latent, so residual caching is disabled for this run.",
                self.cfg_scale,
            )

        self.previous_signature = current_signature
        if can_skip:
            self.accumulated_change = next_accumulated
            self.consecutive_skips += 1
            self.stats.skipped_steps += 1
            self._skip_active = True
            self._step_input = None
            self._log_decision("SKIP", f"accumulated change {next_accumulated:.5f} < {self.threshold:.5f}")
            return self._apply_residual(img)

        self.accumulated_change = 0.0
        self.consecutive_skips = 0
        self.stats.full_steps += 1
        self._step_input = img.detach().clone()
        if relative_change == float("inf"):
            reason = "initial step or changed layout"
        elif not (self.start_percent <= progress <= self.end_percent):
            reason = f"progress {progress:.1%} outside cache window"
        elif next_accumulated >= self.threshold:
            reason = f"change {next_accumulated:.5f} >= {self.threshold:.5f}"
        else:
            reason = "consecutive-skip safety limit"
        self._log_decision("RUN", reason)
        return img

    def _finish_full_step(self, output: torch.Tensor) -> None:
        if self._step_input is None or self._step_input.shape != output.shape:
            self.cached_residual = None
            return
        # Reuse the saved input buffer for the residual.  For the 864x480
        # baseline this avoids a second temporary allocation of roughly
        # 300+ MiB at the end of every full transformer pass.
        residual = self._step_input
        residual.neg_().add_(output.detach())
        self._store_residual(residual)
        self._step_input = None

    def block_patch(self, block_index: int) -> Callable[[dict, dict], dict]:
        def patch(args: dict, extra: dict) -> dict:
            original_block = extra["original_block"]
            img = args["img"]
            try:
                if block_index == 0:
                    img = self._begin_step(img, args.get("mod_segments", []))
                    if self._skip_active:
                        return {"img": img}
                    run_args = dict(args)
                    run_args["img"] = img
                    return original_block(run_args)

                if self._skip_active:
                    if block_index == self.block_count - 1:
                        self._skip_active = False
                    return {"img": img}

                result = original_block(args)
                if block_index == self.block_count - 1:
                    try:
                        self._finish_full_step(result["img"])
                    except Exception as exc:
                        self.cached_residual = None
                        self._step_input = None
                        if not self._warning_emitted:
                            LOGGER.warning("MiniMax H3 Open Cache could not store its residual: %s", exc)
                            self._warning_emitted = True
                return result
            except Exception as exc:
                # Cache bookkeeping must fail open.  If a real transformer
                # block itself failed, calling it again would be unsafe, so
                # only recover before/no-op paths and re-raise other failures.
                if block_index == 0 and self._step_input is None:
                    if not self._warning_emitted:
                        LOGGER.warning("MiniMax H3 Open Cache disabled for this step: %s", exc)
                        self._warning_emitted = True
                    self._skip_active = False
                    return original_block(args)
                raise

        return patch

    def finish(self) -> None:
        total = self.stats.full_steps + self.stats.skipped_steps
        theoretical = total / max(1, self.stats.full_steps)
        LOGGER.info(
            "MiniMax H3 Open Cache: skipped %d/%d model evaluations (block-compute upper bound %.2fx)",
            self.stats.skipped_steps,
            total,
            theoretical,
        )
        self.release_tensors()


class _SamplingScope:
    def __init__(self, cache: _WholeLoopCache) -> None:
        self.cache = cache

    @staticmethod
    def _find_sigmas(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if kwargs.get("sigmas") is not None:
            return kwargs["sigmas"]
        # OUTER_SAMPLE currently receives noise, latent, sampler, sigmas, ...
        if len(args) > 3 and isinstance(args[3], torch.Tensor) and args[3].ndim == 1:
            return args[3]
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.ndim == 1 and arg.numel() > 1:
                return arg
        return None

    def __call__(self, executor: Callable, *args: Any, **kwargs: Any) -> Any:
        sigmas = self._find_sigmas(args, kwargs)
        total_steps = max(1, int(sigmas.numel()) - 1) if sigmas is not None else 1
        self.cache.reset(total_steps=total_steps)
        # OUTER_SAMPLE receives (noise, latent_image, sampler, sigmas, ...); args[2] is a
        # comfy.samplers.KSAMPLER instance whose sampler function lives on
        # `.sampler_function`.  ComfyUI's registry maps UI names to k_diffusion
        # functions with a "sample_" prefix (e.g. "euler_ancestral" ->
        # "sample_euler_ancestral", samplers.py:1033), so we normalize the prefix
        # away before matching the blocklist.
        sampler_name = ""
        if len(args) > 2:
            sampler_fn = getattr(args[2], "sampler_function", None)
            if sampler_fn is not None:
                raw = str(getattr(sampler_fn, "__name__", "") or "")
                if raw.startswith("sample_"):
                    raw = raw[len("sample_"):]
                if raw == "<lambda>":
                    raw = ""  # 合成名按未知处理
                sampler_name = raw
        if not sampler_name:
            # Unknown or empty sampler name: fail closed instead of risking
            # trajectory pollution on an unidentified sampler.
            self.cache.sampler_blocked = True
            LOGGER.warning(
                "MiniMax H3 Open Cache: sampler name could not be determined; "
                "cache disabled for this run (conservative)."
            )
        elif sampler_name in _BLOCKED_SAMPLERS:
            self.cache.sampler_blocked = True
        else:
            # 名字安全：显式复位（同一 model 先跑风险采样器再跑安全采样器时
            # 不残留上一 run 的禁用状态）。
            self.cache.sampler_blocked = False
            LOGGER.info("MiniMax H3 Open Cache: sampler=%s", sampler_name)
        LOGGER.info(
            "MiniMax H3 Open Cache enabled: threshold %.3f, window %.0f%%-%.0f%%, max skips %d, cache %s",
            self.cache.threshold,
            self.cache.start_percent * 100,
            self.cache.end_percent * 100,
            self.cache.max_consecutive_skips,
            self.cache.cache_device,
        )
        try:
            return executor(*args, **kwargs)
        finally:
            self.cache.finish()


def _capture_cfg(cache: "_WholeLoopCache", args: dict[str, Any]) -> Any:
    """sampler_pre_cfg_function hook：只观测 cond_scale，必须原样透传 conds_out。

    ComfyUI 的 sampling_function 用 `out = fn(args)` 调用本 hook，返回值会
    **覆盖 out**（samplers.py:622-625）；返回 None 会让下游 out[0] 崩溃。
    """
    try:
        cache.cfg_scale = float(args.get("cond_scale", 1.0))
    except (TypeError, ValueError):
        cache.cfg_scale = None
    return args.get("conds_out")


class MiniMaxH3OpenCache:
    """Workflow-scoped, pure-Python cache for the official MiniMax-H3 model."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model": ("MODEL",),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.005,
                        "precision": 3,
                        "tooltip": "Lower is safer and skips less. Start A/B testing at 0.05.",
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.20,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "precision": 2,
                        "tooltip": "Relative point in the sampling schedule where cache decisions begin.",
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 0.80,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "precision": 2,
                        "tooltip": "Relative point where caching stops; early and late steps always run fully.",
                    },
                ),
                "max_consecutive_skips": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 5,
                        "step": 1,
                        "tooltip": "Force a full 50-layer pass after this many consecutive cache hits.",
                    },
                ),
                "cache_device": (
                    ["gpu", "cpu"],
                    {
                        "default": "gpu",
                        "tooltip": "GPU is faster. CPU saves persistent VRAM but uploads the residual on every hit.",
                    },
                ),
                "verbose": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Print every RUN/SKIP decision to the ComfyUI console."},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3 Lab/Experimental"
    DESCRIPTION = (
        "Experimental whole-transformer residual cache for the official MiniMax-H3 implementation. "
        "Uses ModelPatcher hooks and does not modify ComfyUI core files."
    )

    def apply(
        self,
        model: Any,
        threshold: float,
        start_percent: float,
        end_percent: float,
        max_consecutive_skips: int,
        cache_device: str,
        verbose: bool,
    ) -> tuple[Any]:
        if start_percent >= end_percent:
            raise ValueError("MiniMax H3 Open Cache: start_percent must be lower than end_percent")

        patched = model.clone()
        diffusion_model = getattr(getattr(patched, "model", None), "diffusion_model", None)
        if diffusion_model is None or diffusion_model.__class__.__name__ != "MiniMaxH3Model":
            actual = diffusion_model.__class__.__name__ if diffusion_model is not None else "unknown"
            raise ValueError(f"MiniMax H3 Open Cache only supports MiniMaxH3Model; received {actual}")

        blocks = getattr(diffusion_model, "blocks", None)
        if blocks is None or len(blocks) == 0:
            raise RuntimeError("MiniMax H3 Open Cache: model does not expose transformer blocks")

        try:
            forward_source = inspect.getsource(diffusion_model._forward)
        except (OSError, TypeError):
            forward_source = ""
        if "double_block" not in forward_source or "patches_replace" not in forward_source:
            raise RuntimeError(
                "MiniMax H3 Open Cache requires the official per-block patch hooks. "
                "Update ComfyUI to a compatible MiniMax-H3 build first."
            )

        transformer_options = patched.model_options.get("transformer_options", {})
        if transformer_options.get("easycache") is not None:
            raise RuntimeError(
                "MiniMax H3 Open Cache cannot be combined with native EasyCache/LazyCache. "
                "Use exactly one step cache."
            )
        existing = transformer_options.get("patches_replace", {}).get("dit", {})
        collisions = [i for i in range(len(blocks)) if ("double_block", i) in existing]
        loop_collisions = [key for key in existing if isinstance(key, tuple) and key[:1] == ("block_loop",)]
        if collisions or loop_collisions:
            raise RuntimeError(
                "MiniMax H3 Open Cache cannot be combined with another double_block/block_loop "
                f"replacement (block collisions: {collisions[:5]}, loop: {loop_collisions[:3]}). "
                "Remove the other block cache/patch first."
            )

        cache = _WholeLoopCache(
            block_count=len(blocks),
            threshold=threshold,
            start_percent=start_percent,
            end_percent=end_percent,
            max_consecutive_skips=max_consecutive_skips,
            cache_device=cache_device,
            verbose=verbose,
        )
        for block_index in range(len(blocks)):
            patched.set_model_patch_replace(
                cache.block_patch(block_index), "dit", "double_block", block_index
            )
        # Capture the CFG scale on every sampling step.  sampling_function invokes
        # sampler_pre_cfg_function hooks with cond_scale in the args dict; when
        # CFG > 1 the positive/negative branches share one latent, so residual
        # reuse would mix trajectories and the cache disables itself.
        # NOTE: sampling_function does `out = fn(args)` — the hook return value
        # REPLACES `out`. We only observe, so the module-level _capture_cfg
        # passes conds_out through unchanged.
        # WARNING: do NOT define a same-named nested _capture_cfg here — it
        # would shadow the module-level (cache, args) version and the partial
        # below would bind the broken single-arg one (TypeError at runtime).
        # Tests only cover the module-level function; keep this area shadow-free.

        model_options = dict(patched.model_options)
        pre_cfg = list(model_options.get("sampler_pre_cfg_function", []) or [])
        pre_cfg.append(functools.partial(_capture_cfg, cache))
        model_options["sampler_pre_cfg_function"] = pre_cfg
        patched.model_options = model_options
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            PATCH_KEY,
            _SamplingScope(cache),
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3OpenCache": MiniMaxH3OpenCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3OpenCache": "MiniMax H3 Open Cache (Experimental)",
}
