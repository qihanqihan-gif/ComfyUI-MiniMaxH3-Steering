# -*- coding: utf-8 -*-
"""MiniMaxH3EasyCacheSafe — 官方 EasyCache 的安全预设包装节点。

定位：ComfyUI 0.30+ 官方 EasyCache 的保守封装。普通用户先测官方节点；
本节点额外提供：H3 模型校验、双重缓存冲突检测、保守默认参数、
可选残差 offload 与一次性的安全说明报告。

官方 EasyCache 通过 transformer_options["easycache"] + 三个 wrapper 挂载，
并按条件 UUID 隔离缓存（CFG 多分支安全），因此本包装不重复做 CFG 检测。
"""
import logging

import comfy.patcher_extension  # noqa: F401  (ComfyUI 运行时从根目录导入)

LOGGER = logging.getLogger(__name__)
PATCH_KEY = "minimax_h3_lab_easycache_safe"

# 保守 -> 激进 三档预设（阈值 / 起始 / 结束），比官方默认(0.2/0.15/0.95)更安全
PRESETS = {
    "conservative": (0.10, 0.15, 0.90),
    "balanced": (0.15, 0.15, 0.90),
    "aggressive": (0.20, 0.15, 0.85),
}


def _load_official_easycache():
    """延迟导入官方 EasyCache 实现；旧版 ComfyUI 无此模块时返回 None。"""
    try:
        from comfy_extras.nodes_easycache import (
            EasyCacheHolder,
            easycache_calc_cond_batch_wrapper,
            easycache_forward_wrapper,
            easycache_sample_wrapper,
        )
        return (EasyCacheHolder, easycache_sample_wrapper,
                easycache_calc_cond_batch_wrapper, easycache_forward_wrapper)
    except Exception:
        LOGGER.warning("MiniMax H3 EasyCache Safe: comfy_extras.nodes_easycache 不可用，需要 ComfyUI 0.30+")
        return None


class MiniMaxH3EasyCacheSafe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "preset": (list(PRESETS.keys()), {"default": "conservative"}),
                "reuse_threshold": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 3.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
                "offload_cache_diff": ("BOOLEAN", {"default": False}),
                "verbose": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3 Lab/Safety"
    EXPERIMENTAL = True

    def apply(self, model, preset, reuse_threshold, start_percent, end_percent,
              offload_cache_diff, verbose):
        report_lines = []

        # 1) H3 模型校验（与 OpenCache 同源判断）
        model_type = getattr(model, "model", None)
        if not _is_minimax_h3(model_type):
            raise ValueError(
                "MiniMax H3 EasyCache Safe: 仅支持 MiniMax-H3 模型（检测到 %s）。",
                type(model_type).__name__ if model_type is not None else "None",
            )

        # 2) 双重缓存冲突检测：官方 EasyCache / LazyCache / 本 Lab 的 OpenCache
        transformer_options = (model.model_options or {}).get("transformer_options", {})
        if transformer_options.get("easycache") is not None:
            raise ValueError("MiniMax H3 EasyCache Safe: 模型已挂载官方 EasyCache，禁止串联。")
        if transformer_options.get("lazycache") is not None:
            raise ValueError("MiniMax H3 EasyCache Safe: 模型已挂载 LazyCache，禁止串联。")
        if transformer_options.get(PATCH_KEY) is not None:
            raise ValueError("MiniMax H3 EasyCache Safe: 模型已挂载本 Lab 的 OpenCache，禁止串联。")

        # 3) 官方实现可用性
        official = _load_official_easycache()
        if official is None:
            raise ValueError("MiniMax H3 EasyCache Safe: 当前 ComfyUI 缺少官方 EasyCache（需 0.30+），节点不可用。")
        EasyCacheHolder, easycache_sample_wrapper, easycache_calc_cond_batch_wrapper, easycache_forward_wrapper = official

        # 4) 参数生效：preset 填充默认，显式 widget 值优先（用户在界面改动时生效）
        p_threshold, p_start, p_end = PRESETS[preset]
        threshold = reuse_threshold if reuse_threshold != 0.0 else p_threshold
        start = start_percent if start_percent != 0.0 else p_start
        end = end_percent if end_percent != 0.0 else p_end
        if end <= start:
            raise ValueError("MiniMax H3 EasyCache Safe: end_percent 必须大于 start_percent。")

        # 5) 挂载（照官方实现：holder + 三个 wrapper）
        patched = model.clone()
        patched.model_options.setdefault("transformer_options", {})[PATCH_KEY] = True
        patched.model_options["transformer_options"]["easycache"] = EasyCacheHolder(
            threshold, start, end,
            subsample_factor=8,
            offload_cache_diff=bool(offload_cache_diff),
            verbose=bool(verbose),
            output_channels=getattr(patched.model, "latent_format", None) and getattr(
                patched.model.latent_format, "latent_channels", None),
        )
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, PATCH_KEY, easycache_sample_wrapper)
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.CALC_COND_BATCH, PATCH_KEY, easycache_calc_cond_batch_wrapper)
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, PATCH_KEY, easycache_forward_wrapper)

        report_lines.append(
            f"EasyCache Safe 已启用：preset={preset}, threshold={threshold:.2f}, "
            f"start={start:.2f}, end={end:.2f}, offload={bool(offload_cache_diff)}, "
            f"verbose={bool(verbose)}"
        )
        report_lines.append("注意：本节点为官方 EasyCache 的保守封装，同 seed A/B 后再调高 threshold。")
        return (patched, "\n".join(report_lines))


def _is_minimax_h3(model_obj):
    """与 OpenCache 相同的最小 H3 判断：类名含 MiniMaxH3 且带 blocks。"""
    if model_obj is None:
        return False
    type_name = type(model_obj).__name__
    if "MiniMaxH3" not in type_name:
        return False
    blocks = getattr(model_obj, "blocks", None)
    return blocks is not None and len(blocks) > 0


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3EasyCacheSafe": MiniMaxH3EasyCacheSafe,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3EasyCacheSafe": "MiniMax H3 EasyCache Safe (官方缓存保守封装)",
}
