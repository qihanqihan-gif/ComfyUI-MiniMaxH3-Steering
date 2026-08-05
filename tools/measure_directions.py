# -*- coding: utf-8 -*-
"""measure_directions.py — 为 MiniMaxH3Steering 节点生成方向向量（离线、中性描述）。

原理：用两组语义分类提示词分别过 H3 文本编码器，取语言层 band 的
注意力输出投影（self_attn.o_proj）输入激活的均值差，归一化后存 .npy。
节点加载时按 `h -= lam*(h·d)*d` 施加。

用法（在 ComfyUI 的 python 环境执行，例如秋叶包）：
    python tools/measure_directions.py ^
        --encoder "ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" ^
        --prompts-a prompts_a.json --prompts-b prompts_b.json ^
        --out "data/refusal_dir.npy" --band 40-49

prompts_a.json / prompts_b.json：字符串数组，两组语义分类的提示词。
提示词内容由使用者自行定义；本脚本不做任何语义判断。

依赖：torch + numpy + comfy（在 ComfyUI 根目录运行或把根目录加入 PYTHONPATH）。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

COMFY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)

import comfy.sd  # noqa: E402


def _locate_layers(clip, band_start, band_end):
    cond_stage = getattr(clip, "cond_stage_model", None)
    clip_key = getattr(cond_stage, "clip", None)
    wrapper = getattr(cond_stage, clip_key, None)
    base = getattr(wrapper, "transformer", None)
    model = getattr(base, "model", None)
    layers = getattr(model, "layers", None)
    if layers is None or band_end >= len(layers):
        raise RuntimeError("未找到 50 层语言模型或 band 越界；请用标准 CLIPLoader 加载 H3 编码器。")
    return layers


def _encode_prompts(clip, prompts, band_start, band_end, layers):
    """逐条编码提示词，记录 band 内各层 o_proj 输入激活均值（按 token 平均）。"""
    layer_acts = [torch.zeros(0, dtype=torch.float32, device="cpu") for _ in range(band_start, band_end + 1)]
    hooks = []
    for i, layer in enumerate(layers[band_start:band_end + 1]):
        o_proj = getattr(getattr(layer, "self_attn", None), "o_proj", None)
        if o_proj is None:
            raise RuntimeError(f"层 {band_start + i} 缺少 self_attn.o_proj")

        def make_hook(idx):
            def hook_fn(module, args, _out):
                # 记录 o_proj 输出激活（Steering 节点施加于输出空间，测量空间须一致）
                out = _out[0] if isinstance(_out, (tuple, list)) else _out
                mean = out.float().mean(dim=(0, 1))
                layer_acts[idx] = mean.detach().cpu()
            return hook_fn

        hooks.append(o_proj.register_forward_hook(make_hook(i)))

    try:
        per_prompt = []
        for text in prompts:
            tokens = clip.tokenize(text)
            outputs = clip.encode_from_tokens(tokens, return_pooled=False)
            pooled = outputs[0] if isinstance(outputs, tuple) else outputs
            if pooled is not None and hasattr(pooled, "shape"):
                pass  # 触发前向
            # 取隐藏状态（encode_from_tokens 返回值结构随版本变化，尽量兼容）
            if isinstance(outputs, (list, tuple)):
                hidden = outputs[0]
            else:
                hidden = outputs
            if isinstance(hidden, torch.Tensor) and hidden.dim() >= 3:
                pass  # 已触发；激活已由 hook 记录
            per_prompt.append([a.clone() for a in layer_acts])
    finally:
        for h in hooks:
            h.remove()

    stacked = torch.stack([torch.stack(per, dim=0) for per in per_prompt], dim=0)  # [n_prompt, n_layers, hidden]
    return stacked.mean(dim=0)  # [n_layers, hidden]


def main():
    parser = argparse.ArgumentParser(description="生成 H3 文本编码器方向向量（两组分类提示词激活差异）")
    parser.add_argument("--encoder", required=True, help="H3 文本编码器权重路径（safetensors）")
    parser.add_argument("--prompts-a", required=True, help="组 A 提示词 JSON（字符串数组）")
    parser.add_argument("--prompts-b", required=True, help="组 B 提示词 JSON（字符串数组）")
    parser.add_argument("--out", required=True, help="输出 .npy 路径")
    parser.add_argument("--band", default="40-49", help="层区间，默认 40-49")
    parser.add_argument("--device", default="cuda", help="torch 设备，默认 cuda")
    args = parser.parse_args()

    with open(args.prompts_a, encoding="utf-8") as f:
        prompts_a = json.load(f)
    with open(args.prompts_b, encoding="utf-8") as f:
        prompts_b = json.load(f)
    if not isinstance(prompts_a, list) or not isinstance(prompts_b, list) or not prompts_a or not prompts_b:
        raise SystemExit("prompts-a / prompts-b 必须是非空字符串数组 JSON")
    if not all(isinstance(p, str) for p in prompts_a) or not all(isinstance(p, str) for p in prompts_b):
        raise SystemExit("prompts-a / prompts-b 的元素必须全部是字符串")

    start_s, _, end_s = args.band.partition("-")
    band_start, band_end = int(start_s), int(end_s)

    # 先以 CPU 默认设备加载（ComfyUI 内部会把量化 config 元数据张量 .numpy()，
    # 若默认设备已是 cuda 会报 "can't convert cuda tensor to numpy"），
    # 加载完成后整体搬移到目标设备，再设置默认设备供后续张量使用。
    clip = comfy.sd.load_clip([args.encoder], clip_type=comfy.sd.CLIPType.MINIMAX)
    if args.device != "cpu":
        clip.cond_stage_model.to(args.device)
    torch.set_default_device(args.device)
    layers = _locate_layers(clip, band_start, band_end)

    mean_a = _encode_prompts(clip, prompts_a, band_start, band_end, layers)
    mean_b = _encode_prompts(clip, prompts_b, band_start, band_end, layers)
    direction = mean_a - mean_b  # [n_layers, hidden]

    # 逐层归一化；零范数层保留零向量
    norms = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    direction = torch.where(norms > 1e-8, direction / norms.clamp_min(1e-8), torch.zeros_like(direction))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    np.save(args.out, direction.numpy())
    print(f"已生成方向向量：{args.out} shape={tuple(direction.shape)}（层 {args.band}，hidden={direction.shape[1]}）")
    print("节点将按 h -= lam*(h·d)*d 施加；层数与 hidden 维度必须与节点加载的编码器一致。")


if __name__ == "__main__":
    main()
