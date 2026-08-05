# -*- coding: utf-8 -*-
"""MiniMaxH3Steering — H3 文本编码器加载时方向操控节点（独立实现）。

机制（公开技术事实，作者 README 描述同思路）：
对语言层 band（默认 40–49）的 self_attn.o_proj 输出做投影减法：
    h_out = h_out - lam * (h_out · d) * d
方向 d 从本插件 data/ 目录加载（.npy，每层一个 hidden 维向量）。
refusal 与 safety 两个方向独立开关。

方向数据缺失时节点保持直通并输出提示，工作流可先接线；
运行 tools/measure_directions.py 生成 data 后立即生效。
本实现不含任何 gated 仓库代码，仅按公开机制独立编写。
"""
import logging
import os
import weakref

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_DEFAULT_BAND = "40-49"
_DIRECTION_FILES = {
    "refusal": "refusal_dir.npy",
    "safety": "safety_dir.npy",
}


def _parse_band(band: str):
    """解析 "40-49" 形式的层区间（含端点）。"""
    try:
        start_s, _, end_s = band.partition("-")
        start, end = int(start_s.strip()), int(end_s.strip())
    except (ValueError, AttributeError):
        raise ValueError(f"MiniMax H3 Steering: layer_band 格式应为 '40-49'，收到 {band!r}")
    if start < 0 or end < start:
        raise ValueError(f"MiniMax H3 Steering: 非法层区间 {band!r}")
    return start, end


def _load_direction_file(fname: str, device, hidden: int, expected_layers: int | None = None):
    """加载 data/<fname> 方向向量（fname 为 data/ 目录下的 .npy 文件名）。

    支持两种规格：
      - [hidden]            单方向向量，广播到 band 内所有层
      - [n_layers, hidden]  每层独立方向向量（measure_directions.py 输出）
    文件缺失/维度不符/层数不符/非有限/零范数时返回 None（节点直通）。
    expected_layers 传入 band 层数，2D 数据的层数与之不符时忽略（防 d[i-start] 越界）。
    """
    if not isinstance(fname, str):
        # 旧工作流可能残留布尔值（direction_file 曾是 BOOLEAN 位置），直通不崩
        LOGGER.warning("MiniMax H3 Steering: 方向文件参数类型异常 %r，忽略（直通）", fname)
        return None
    fname = os.path.basename(fname)  # 仅允许 data/ 内文件名（防 COMBO 文本路径穿越）
    path = os.path.join(_DATA_DIR, fname)
    if not os.path.isfile(path):
        LOGGER.warning("MiniMax H3 Steering: 缺少方向数据 %s（先运行 tools/measure_directions.py）", path)
        return None
    try:
        arr = np.load(path, allow_pickle=False).astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("MiniMax H3 Steering: 方向文件 %s 读取失败: %s", path, exc)
        return None
    if arr.ndim == 1:
        if arr.shape[0] != hidden:
            LOGGER.warning(
                "MiniMax H3 Steering: 方向 %s 维度 %s 与 hidden %d 不匹配，忽略",
                fname, arr.shape, hidden,
            )
            return None
        t = torch.from_numpy(arr).to(device)
        norm = torch.linalg.vector_norm(t)
        if not torch.isfinite(norm) or norm.item() <= 0.0:
            LOGGER.warning("MiniMax H3 Steering: 方向 %s 非有限/零范数，忽略", fname)
            return None
        return t / norm
    if arr.ndim == 2:
        if arr.shape[1] != hidden:
            LOGGER.warning(
                "MiniMax H3 Steering: 方向 %s 维度 %s 与 hidden %d 不匹配，忽略",
                fname, arr.shape, hidden,
            )
            return None
        if expected_layers is not None and arr.shape[0] != expected_layers:
            LOGGER.warning(
                "MiniMax H3 Steering: 方向 %s 层数 %d 与 band 层数 %d 不匹配，忽略",
                fname, arr.shape[0], expected_layers,
            )
            return None
        t = torch.from_numpy(arr).to(device)
        norms = torch.linalg.vector_norm(t, dim=-1, keepdim=True)
        if not torch.isfinite(norms).all().item() or not (norms > 0).all().item():
            LOGGER.warning("MiniMax H3 Steering: 方向 %s 存在非有限/零范数层，忽略", fname)
            return None
        return t / norms.clamp_min(1e-8)
    LOGGER.warning("MiniMax H3 Steering: 方向 %s 维度异常 %s，忽略", fname, arr.shape)
    return None


def _load_direction(name: str, device, hidden: int, expected_layers: int | None = None):
    """加载预注册方向（refusal/safety），委托 _load_direction_file。"""
    return _load_direction_file(_DIRECTION_FILES[name], device, hidden, expected_layers)


def _available_direction_files() -> list[str]:
    """扫描 data/ 目录下所有 .npy，供方向文件自选下拉使用。"""
    if not os.path.isdir(_DATA_DIR):
        return []
    return sorted(f for f in os.listdir(_DATA_DIR) if f.endswith(".npy"))


_WRAPPED_O_PROJ = weakref.WeakKeyDictionary()  # o_proj -> set of direction keys


def _wrap_o_proj(layer, direction, lam, key: str):
    """包装单层 self_attn.o_proj：输出投影后减去 lam*(h·d)*d。

    兼容性设计：
    - 按 (o_proj, key) 去重：key 为稳定方向身份（"refusal"/"safety"/方向文件名），
      同层同方向只包一次（防多节点/重复排队 lam 叠加——不能用 id(direction)，
      每次排队 np.load 会新建 tensor）；不同 key（refusal+safety）各自包装叠加
    - steered_forward 操控段 try/except 降级：数学失败时原样返回 out（不崩图）并记日志
    """
    o_proj = getattr(getattr(layer, "self_attn", None), "o_proj", None)
    if o_proj is None:
        return False
    wrapped = _WRAPPED_O_PROJ.get(o_proj)
    if wrapped is not None and key in wrapped:
        return False  # 同层同方向已包装（防 lam 叠加）
    original_forward = o_proj.forward

    def steered_forward(x, *args, **kwargs):
        out = original_forward(x, *args, **kwargs)
        try:
            # direction 在 apply 时按权重设备创建（当时可能还是 CPU，ComfyUI 懒加载）；
            # 实际 encode 时权重已被移到 GPU——direction 必须跟随 out 的实际设备，
            # 否则 out(cuda) * direction(cpu) 抛设备不匹配。同设备时 .to() 为 no-op。
            d = direction.to(out.device)
            # out 形状可能为 [batch, seq, hidden] 或 [seq, batch, hidden]：取最后一维
            proj = torch.sum(out * d, dim=-1, keepdim=True)
            return out - lam * proj * d
        except Exception as exc:  # noqa: BLE001  操控失败 → 原样返回（不崩图）
            LOGGER.warning("MiniMax H3 Steering: 方向操控失败已降级直通: %s", exc)
            return out

    o_proj.forward = steered_forward
    if wrapped is None:
        _WRAPPED_O_PROJ[o_proj] = set()
    _WRAPPED_O_PROJ[o_proj].add(key)
    return True


class MiniMaxH3Steering:
    @classmethod
    def INPUT_TYPES(cls):
        auto = "auto (refusal/safety)"
        return {
            "required": {
                "clip": ("CLIP",),
                "steer_refusal": ("BOOLEAN", {"default": True}),
                "steer_safety": ("BOOLEAN", {"default": True}),
                "lam": ("FLOAT", {"default": 3.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "layer_band": ("STRING", {"default": _DEFAULT_BAND, "multiline": False}),
                # 注意：新 widget 必须追加在末尾（ComfyUI 按位置恢复旧工作流 widget 值，
                # 插中间会导致 lam/layer_band 等旧值错位——曾致 lam 收到 "40-49" 崩溃）
                "direction_file": ([auto] + _available_direction_files(), {"default": auto}),
            },
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "report")
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3 Lab/Steering"
    EXPERIMENTAL = True

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # direction_file 的 COMBO 值可能来自旧工作流（残留布尔/任意字符串），
        # 放行避免旧工作流校验失败；apply 内对非法文件名直通处理。
        return True

    def apply(self, clip, direction_file, steer_refusal, steer_safety, lam, layer_band):
        start, end = _parse_band(layer_band)
        auto = "auto (refusal/safety)"

        # 定位语言层（与 Guide 的 tail loader 同一访问路径）
        cond_stage = getattr(clip, "cond_stage_model", None)
        clip_key = getattr(cond_stage, "clip", None) if cond_stage is not None else None
        wrapper = getattr(cond_stage, clip_key, None) if clip_key else None
        base = getattr(wrapper, "transformer", None) if wrapper is not None else None
        model = getattr(base, "model", None) if base is not None else None
        layers = getattr(model, "layers", None) if model is not None else None
        if layers is None or end >= len(layers):
            raise ValueError(
                f"MiniMax H3 Steering: 未找到 50 层语言模型（或 band {layer_band} 越界）。"
                "请用标准 CLIPLoader 加载 H3 文本编码器。"
            )

        # hidden 维度：从 o_proj 输出权重推断
        hidden = None
        probe = getattr(getattr(layers[start], "self_attn", None), "o_proj", None)
        if probe is not None and getattr(probe, "out_features", None):
            hidden = int(probe.out_features)
        if hidden is None:
            raise ValueError("MiniMax H3 Steering: 无法确定 hidden 维度（未找到 self_attn.o_proj）。")

        device = layers[start].self_attn.o_proj.weight.device
        report_lines = []

        # 自选方向文件：跳过 refusal/safety 双开关，直接用该文件 + lam
        if direction_file and direction_file != auto:
            d = _load_direction_file(direction_file, device, hidden, end - start + 1)
            if d is not None:
                count = 0
                for i in range(start, end + 1):
                    layer_d = d[i - start] if d.ndim == 2 else d
                    count += 1 if _wrap_o_proj(layers[i], layer_d, lam, key=os.path.basename(str(direction_file))) else 0
                hint = "" if count > 0 else "（0 层——此前已应用同方向，本次 lam/方向变更未生效）"
                report_lines.append(
                    f"自定义方向 {direction_file}：已应用于层 {start}-{end}（{count} 层），lam={lam}{hint}"
                )
            else:
                report_lines.append(f"自定义方向 {direction_file}：数据缺失或不匹配，未应用（直通）")
            return (clip, "\n".join(report_lines))

        patched_any = False
        if steer_refusal:
            d = _load_direction("refusal", device, hidden, end - start + 1)
            if d is not None:
                count = 0
                for i in range(start, end + 1):
                    layer_d = d[i - start] if d.ndim == 2 else d
                    count += 1 if _wrap_o_proj(layers[i], layer_d, lam, key="refusal") else 0
                patched_any = patched_any or count > 0
                hint = "" if count > 0 else "（0 层——此前已应用同方向，本次 lam/方向变更未生效）"
                report_lines.append(f"refusal 方向：已应用于层 {start}-{end}（{count} 层），lam={lam}{hint}")
            else:
                report_lines.append("refusal 方向：数据缺失或不匹配，未应用（直通）")

        if steer_safety:
            d = _load_direction("safety", device, hidden, end - start + 1)
            if d is not None:
                count = 0
                for i in range(start, end + 1):
                    layer_d = d[i - start] if d.ndim == 2 else d
                    count += 1 if _wrap_o_proj(layers[i], layer_d, lam, key="safety") else 0
                patched_any = patched_any or count > 0
                hint = "" if count > 0 else "（0 层——此前已应用同方向，本次 lam/方向变更未生效）"
                report_lines.append(f"safety 方向：已应用于层 {start}-{end}（{count} 层），lam={lam}{hint}")
            else:
                report_lines.append("safety 方向：数据缺失或不匹配，未应用（直通）")

        if not patched_any:
            report_lines.append(
                "注意：当前为直通模式（无方向数据）。运行 tools/measure_directions.py 生成 data/*.npy 后生效。"
            )
        else:
            report_lines.append("说明：仅修改文本编码器注意力输出投影（加载时），不修改任何权重文件。")

        return (clip, "\n".join(report_lines))


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3Steering": MiniMaxH3Steering,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3Steering": "MiniMax H3 Steering (加载时方向操控, 实验)",
}
