# -*- coding: utf-8 -*-
"""OpenCache 发布保护逻辑的最小单元测试（不触碰任何模型文件）。"""
import importlib.util
import os
import sys

# 路径推导：插件根由测试文件位置计算；ComfyUI 根优先取环境变量
# COMFY_ROOT，未设置时按 custom_nodes/ 标准布局推导（…/ComfyUI/custom_nodes/插件）。
LAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.environ.get("COMFY_ROOT") or os.path.dirname(os.path.dirname(LAB_ROOT))

if COMFY_ROOT and os.path.isdir(os.path.join(COMFY_ROOT, "comfy")) and COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)
if LAB_ROOT not in sys.path:
    sys.path.insert(0, LAB_ROOT)


def _load_nodes():
    spec = importlib.util.spec_from_file_location("lab_open_cache_nodes", LAB_ROOT + "/nodes.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lab_open_cache_nodes"] = mod  # dataclasses/annotations 依赖 sys.modules 注册
    spec.loader.exec_module(mod)
    return mod


def _run_scope(mod, sampler_fn, sigmas):
    import torch

    cache = mod._WholeLoopCache(
        block_count=50, threshold=0.05, start_percent=0.2, end_percent=0.8,
        max_consecutive_skips=1, cache_device="gpu", verbose=False,
    )
    scope = mod._SamplingScope(cache)

    class KSamplerLike:
        pass

    s = KSamplerLike()
    s.sampler_function = sampler_fn
    args = (None, None, s, sigmas)

    def executor(*a, **k):
        return "ok"

    scope(executor, *args)
    return cache.sampler_blocked


def test_blocklist_contains_risky_ui_names():
    mod = _load_nodes()
    blocked = mod._BLOCKED_SAMPLERS
    for name in (
        "euler_ancestral", "dpmpp_2s_ancestral", "dpmpp_2m_sde", "dpmpp_3m_sde",
        "dpmpp_sde", "er_sde", "res_multistep_ancestral", "lcm", "ddpm",
        "sa_solver", "sa_solver_pece", "seeds_2", "seeds_3",
    ):
        assert name in blocked


def test_blocklist_allows_verified_ui_names():
    mod = _load_nodes()
    blocked = mod._BLOCKED_SAMPLERS
    assert "euler" not in blocked
    assert "res_multistep" not in blocked
    assert "heun" not in blocked


def test_blocked_sampler_via_full_scope():
    """真实形态：k_diffusion 函数名带 sample_ 前缀 → 归一化后命中黑名单。"""
    import torch

    mod = _load_nodes()

    def sample_euler_ancestral():
        pass

    assert _run_scope(mod, sample_euler_ancestral, torch.zeros(2)) is True
    # 带 cfg_pp 后缀的祖先采样器
    def sample_euler_ancestral_cfg_pp():
        pass

    assert _run_scope(mod, sample_euler_ancestral_cfg_pp, torch.zeros(2)) is True


def test_safe_sampler_via_full_scope():
    import torch

    mod = _load_nodes()

    def sample_euler():
        pass

    assert _run_scope(mod, sample_euler, torch.zeros(2)) is False


def test_unknown_name_fails_closed():
    """sampler_function 无 __name__（partial/可调用对象）→ 保守禁用。"""
    import torch

    mod = _load_nodes()

    class NoName:
        pass

    assert _run_scope(mod, NoName(), torch.zeros(2)) is True


def test_blocked_then_safe_sampler_resets_flag():
    """同一 cache 实例先跑风险采样器再跑安全采样器：禁用状态不能跨 run 泄漏。"""
    import torch

    mod = _load_nodes()
    cache = mod._WholeLoopCache(
        block_count=50, threshold=0.05, start_percent=0.2, end_percent=0.8,
        max_consecutive_skips=1, cache_device="gpu", verbose=False,
    )
    scope = mod._SamplingScope(cache)

    class KSamplerLike:
        pass

    def run(sampler_fn):
        s = KSamplerLike()
        s.sampler_function = sampler_fn
        args = (None, None, s, torch.zeros(2))

        def executor(*a, **k):
            return "ok"

        scope(executor, *args)

    def sample_euler_ancestral():
        pass

    def sample_euler():
        pass

    run(sample_euler_ancestral)
    assert cache.sampler_blocked is True
    run(sample_euler)
    assert cache.sampler_blocked is False


def test_capture_cfg_passthrough_prevents_none_crash():
    """回归：sampling_function 用 `out = fn(args)` 调 hook，hook 返回 None 会把
    out 覆盖为 None → cfg_function(out[0]) 崩（'NoneType' object is not subscriptable）。
    hook 必须原样透传 conds_out。"""
    mod = _load_nodes()
    cache = mod._WholeLoopCache.__new__(mod._WholeLoopCache)
    cache.cfg_scale = None

    conds_out = (("cond_out",), ("uncond_out",))
    args = {"cond_scale": 1.0, "conds_out": conds_out}
    returned = mod._capture_cfg(cache, args)

    # 模拟 sampling_function 的赋值链
    out = returned
    assert out is conds_out, "hook 必须透传 conds_out，否则 out 被覆盖为 None"
    assert out[0] is not None and out[1] is not None
    assert cache.cfg_scale == 1.0


def test_capture_cfg_reads_scale_and_handles_bad_values():
    mod = _load_nodes()
    cache = mod._WholeLoopCache.__new__(mod._WholeLoopCache)
    cache.cfg_scale = None

    conds_out = (("x",), ("y",))
    assert mod._capture_cfg(cache, {"cond_scale": 3.5, "conds_out": conds_out}) is conds_out
    assert cache.cfg_scale == 3.5

    assert mod._capture_cfg(cache, {"cond_scale": "bad", "conds_out": conds_out}) is conds_out
    assert cache.cfg_scale is None


def test_mounted_hook_single_arg_invocation():
    """回归：apply 以 functools.partial(_capture_cfg, cache) 挂载，ComfyUI 以
    `out = fn(args)` 单参调用。若 apply 内残留单参嵌套定义遮蔽模块级函数，
    partial 会绑定坏版本 → TypeError: takes 1 positional argument but 2 were given。
    本测试模拟真实挂载路径，确保单参调用不抛异常且透传 conds_out。"""
    import functools

    mod = _load_nodes()
    cache = mod._WholeLoopCache.__new__(mod._WholeLoopCache)
    cache.cfg_scale = None

    mounted = functools.partial(mod._capture_cfg, cache)
    conds_out = (("cond",), ("uncond",))
    args = {"cond_scale": 2.0, "conds_out": conds_out}

    out = mounted(args)  # 与 sampling_function 的调用方式一致
    assert out is conds_out
    assert cache.cfg_scale == 2.0
