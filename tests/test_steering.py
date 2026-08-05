# -*- coding: utf-8 -*-
"""MiniMaxH3Steering 最小单元测试（无模型/网络）。"""
import importlib.util
import os
import sys
import tempfile

import numpy as np
import torch

# 路径推导：插件根由测试文件位置计算；ComfyUI 根优先取环境变量
# COMFY_ROOT，未设置时按 custom_nodes/ 标准布局推导（…/ComfyUI/custom_nodes/插件）。
LAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.environ.get("COMFY_ROOT") or os.path.dirname(os.path.dirname(LAB_ROOT))

if COMFY_ROOT and os.path.isdir(os.path.join(COMFY_ROOT, "comfy")) and COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)
if LAB_ROOT not in sys.path:
    sys.path.insert(0, LAB_ROOT)


def _load_mod():
    spec = importlib.util.spec_from_file_location("lab_steering", LAB_ROOT + "/steering.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lab_steering"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parse_band():
    mod = _load_mod()
    assert mod._parse_band("40-49") == (40, 49)
    assert mod._parse_band("0-0") == (0, 0)
    try:
        mod._parse_band("40-30")
        raise AssertionError("应拒绝 end<start")
    except ValueError:
        pass
    try:
        mod._parse_band("abc")
        raise AssertionError("应拒绝非法格式")
    except ValueError:
        pass


def test_wrap_o_proj_device_mismatch_follows_out():
    """回归：direction 创建在 CPU（apply 时权重未上 GPU），encode 时 out 在 cuda——
    steered_forward 必须让 direction 跟随 out 设备（否则设备不匹配崩溃）。"""
    import pytest

    mod = _load_mod()
    if not torch.cuda.is_available():
        pytest.skip("无 CUDA，跳过跨设备回归")

    hidden = 8
    d_cpu = torch.nn.functional.normalize(torch.randn(hidden), dim=-1)

    class FakeOProj(torch.nn.Module):
        def forward(self, x, *args, **kwargs):
            return x

    class FakeLayer:
        def __init__(self):
            self.self_attn = type("A", (), {"o_proj": FakeOProj()})()

    layer = FakeLayer()
    lam = 2.0
    assert mod._wrap_o_proj(layer, d_cpu, lam, key="refusal") is True

    x = torch.randn(2, 4, hidden, device="cuda")
    out = layer.self_attn.o_proj.forward(x)  # 包装后：out 在 cuda、direction 在 cpu
    assert out.device.type == "cuda", "修复后 out 仍在 cuda 上正常返回"
    # 数学校验：与同设备期望一致
    d_gpu = d_cpu.to("cuda")
    raw = x
    proj = torch.sum(raw * d_gpu, dim=-1, keepdim=True)
    expected = raw - lam * proj * d_gpu
    assert torch.allclose(out, expected), "设备跟随后的数学应与同设备一致"


def test_wrap_o_proj_no_double_wrap_same_direction():
    """兼容回归：同层同方向（同 key，即使 tensor 是新对象）重复包装只生效一次（防 lam 叠加）。"""
    mod = _load_mod()
    hidden = 8

    class FakeOProj(torch.nn.Module):
        def forward(self, x, *args, **kwargs):
            return x

    class FakeLayer:
        def __init__(self):
            self.self_attn = type("A", (), {"o_proj": FakeOProj()})()

    layer = FakeLayer()
    d1 = torch.nn.functional.normalize(torch.randn(hidden), dim=-1)
    d1b = torch.nn.functional.normalize(torch.randn(hidden), dim=-1)  # 新对象（模拟重复排队重新加载）
    assert mod._wrap_o_proj(layer, d1, 3.0, key="refusal") is True
    assert mod._wrap_o_proj(layer, d1b, 3.0, key="refusal") is False, "同 key 重复包装应拒绝（不管 tensor 是否新对象）"

    x = torch.randn(2, 4, hidden)
    out = layer.self_attn.o_proj.forward(x)
    proj = torch.sum(x * d1, dim=-1, keepdim=True)
    expected = x - 3.0 * proj * d1
    assert torch.allclose(out, expected), "应只按第一次方向施加一次"


def test_wrap_o_proj_two_directions_both_apply():
    """回归：refusal+safety 两个不同 key 必须各自生效（互不吞并）。"""
    mod = _load_mod()
    hidden = 8

    class FakeOProj(torch.nn.Module):
        def forward(self, x, *args, **kwargs):
            return x

    class FakeLayer:
        def __init__(self):
            self.self_attn = type("A", (), {"o_proj": FakeOProj()})()

    layer = FakeLayer()
    d1 = torch.nn.functional.normalize(torch.randn(hidden), dim=-1)
    d2 = torch.nn.functional.normalize(torch.randn(hidden), dim=-1)
    assert mod._wrap_o_proj(layer, d1, 2.0, key="refusal") is True
    assert mod._wrap_o_proj(layer, d2, 1.5, key="safety") is True, "不同 key 必须允许包装（safety 不能失效）"

    x = torch.randn(2, 4, hidden)
    out = layer.self_attn.o_proj.forward(x)
    # 顺序投影减法：第二次包装基于第一次的输出（d1/d2 不正交，投影在中间值上算）
    p1 = torch.sum(x * d1, dim=-1, keepdim=True)
    y = x - 2.0 * p1 * d1
    p2 = torch.sum(y * d2, dim=-1, keepdim=True)
    expected = y - 1.5 * p2 * d2
    assert torch.allclose(out, expected), "双方向投影减法应叠加（顺序语义）"


def test_wrap_o_proj_bad_direction_degrades():
    """兼容回归：操控段数学失败（坏方向形状）应降级直通（返回原始 out），不崩图。"""
    mod = _load_mod()
    hidden = 8

    class FakeOProj(torch.nn.Module):
        def forward(self, x, *args, **kwargs):
            return x

    class FakeLayer:
        def __init__(self):
            self.self_attn = type("A", (), {"o_proj": FakeOProj()})()

    layer = FakeLayer()
    bad_d = torch.randn(3)  # 形状与 hidden=8 不匹配 → 广播失败
    assert mod._wrap_o_proj(layer, bad_d, 2.0, key="bad") is True
    x = torch.randn(2, 4, hidden)
    out = layer.self_attn.o_proj.forward(x)
    assert torch.allclose(out, x), "操控失败应原样返回（不崩图）"


def test_wrap_o_proj_math():
    """包装后：out' = out - lam*(out·d)*d（投影减法）。"""
    mod = _load_mod()

    hidden = 8
    d = torch.nn.functional.normalize(torch.randn(hidden), dim=0)

    class FakeOProj(torch.nn.Module):
        def forward(self, x, *a, **k):
            return x * 2.0  # 简单线性

    class FakeSelfAttn:
        def __init__(self):
            self.o_proj = FakeOProj()

    class FakeLayer:
        def __init__(self):
            self.self_attn = FakeSelfAttn()

    layer = FakeLayer()
    lam = 3.0
    orig_forward = layer.self_attn.o_proj.forward  # 包装前的原始 forward
    assert mod._wrap_o_proj(layer, d, lam, key="refusal") is True

    x = torch.randn(2, 4, hidden)
    out = layer.self_attn.o_proj.forward(x)  # 包装后
    raw = orig_forward(x)  # 原始输出（= x*2）
    proj = torch.sum(raw * d, dim=-1, keepdim=True)
    expected = raw - lam * proj * d
    assert torch.allclose(out, expected), "投影减法数学错误"


def test_load_direction_missing_returns_none():
    mod = _load_mod()
    with tempfile.TemporaryDirectory() as tmp:
        mod._DATA_DIR = tmp  # 指向空目录
        assert mod._load_direction("refusal", "cpu", 5120) is None


def test_load_direction_wrong_dim_returns_none():
    mod = _load_mod()
    with tempfile.TemporaryDirectory() as tmp:
        np.save(os.path.join(tmp, "refusal_dir.npy"), np.zeros(16))
        mod._DATA_DIR = tmp
        assert mod._load_direction("refusal", "cpu", 5120) is None


def test_load_direction_ok_and_normalized():
    mod = _load_mod()
    with tempfile.TemporaryDirectory() as tmp:
        vec = np.random.randn(8).astype(np.float32)
        np.save(os.path.join(tmp, "refusal_dir.npy"), vec)
        mod._DATA_DIR = tmp
        d = mod._load_direction("refusal", "cpu", 8)
        assert d is not None
        assert d.shape == (8,)
        assert torch.isclose(torch.linalg.vector_norm(d), torch.tensor(1.0), atol=1e-5)


def test_load_direction_2d_per_layer():
    """回归：measure_directions.py 输出 [n_layers, hidden] 2D，节点必须接受并按层施加
    （此前只接受 1D → 2D 文件被忽略 → 节点永远直通，跑批产物不生效）。"""
    mod = _load_mod()
    with tempfile.TemporaryDirectory() as tmp:
        arr = np.random.randn(10, 8).astype(np.float32)
        np.save(os.path.join(tmp, "refusal_dir.npy"), arr)
        mod._DATA_DIR = tmp
        d = mod._load_direction("refusal", "cpu", 8)
        assert d is not None
        assert d.shape == (10, 8), "2D 每层方向必须被接受"
        norms = torch.linalg.vector_norm(d, dim=-1)
        assert torch.allclose(norms, torch.ones(10), atol=1e-5), "每层应独立归一化"


def test_load_direction_2d_wrong_hidden_returns_none():
    mod = _load_mod()
    with tempfile.TemporaryDirectory() as tmp:
        np.save(os.path.join(tmp, "refusal_dir.npy"), np.zeros((10, 16)))
        mod._DATA_DIR = tmp
        assert mod._load_direction("refusal", "cpu", 8) is None


def test_load_direction_2d_layer_mismatch_returns_none():
    """回归：2D 数据的层数与 band 层数不符时必须忽略（防 apply 里 d[i-start] 越界
    IndexError 崩图）——此前只校验 hidden，用户改宽 layer_band 即崩。"""
    mod = _load_mod()
    with tempfile.TemporaryDirectory() as tmp:
        np.save(os.path.join(tmp, "refusal_dir.npy"), np.random.randn(10, 8).astype(np.float32))
        mod._DATA_DIR = tmp
        assert mod._load_direction("refusal", "cpu", 8, expected_layers=10) is not None
        assert mod._load_direction("refusal", "cpu", 8, expected_layers=20) is None, "层数不符应忽略"
        assert mod._load_direction("refusal", "cpu", 8, expected_layers=5) is None


def test_available_direction_files_lists_npy():
    """自选下拉应列出 data/ 下所有 .npy（含用户自定义方向文件）。"""
    mod = _load_mod()
    with tempfile.TemporaryDirectory() as tmp:
        np.save(os.path.join(tmp, "my_style_dir.npy"), np.random.randn(4, 8).astype(np.float32))
        np.save(os.path.join(tmp, "refusal_dir.npy"), np.random.randn(8).astype(np.float32))
        np.save(os.path.join(tmp, "notes.txt"), np.zeros(4))
        mod._DATA_DIR = tmp
        files = mod._available_direction_files()
        assert "my_style_dir.npy" in files, "自定义 npy 必须出现在自选列表"
        assert "refusal_dir.npy" in files
        assert "notes.txt" not in files, "非 npy 不应列出"


def test_load_direction_file_custom():
    """用户自选任意 npy（2D 每层方向 + 层数校验）应能加载并归一化。"""
    mod = _load_mod()
    with tempfile.TemporaryDirectory() as tmp:
        arr = np.random.randn(6, 8).astype(np.float32)
        np.save(os.path.join(tmp, "my_style_dir.npy"), arr)
        mod._DATA_DIR = tmp
        d = mod._load_direction_file("my_style_dir.npy", "cpu", 8, expected_layers=6)
        assert d is not None
        assert d.shape == (6, 8)
        norms = torch.linalg.vector_norm(d, dim=-1)
        assert torch.allclose(norms, torch.ones(6), atol=1e-5)
        assert mod._load_direction_file("my_style_dir.npy", "cpu", 8, expected_layers=3) is None


def test_load_direction_file_bad_dtype_returns_none():
    """非数值 dtype 的 npy 必须直通（None）而非崩溃（astype 在 try 内）。"""
    mod = _load_mod()
    with tempfile.TemporaryDirectory() as tmp:
        np.save(os.path.join(tmp, "bad_dir.npy"), np.array(["a", "b", "c"]))
        mod._DATA_DIR = tmp
        assert mod._load_direction_file("bad_dir.npy", "cpu", 8) is None


def test_load_direction_file_bool_residue_returns_none():
    """旧工作流残留布尔值（direction_file=True）必须直通不崩（TypeError 防护）。"""
    mod = _load_mod()
    with tempfile.TemporaryDirectory() as tmp:
        mod._DATA_DIR = tmp
        assert mod._load_direction_file(True, "cpu", 8) is None
        assert mod._load_direction_file(None, "cpu", 8) is None


def test_load_direction_file_path_traversal_contained():
    """COMBO 文本可编辑：绝对路径/../ 输入必须被 basename 约束在 data/ 内（直通或读同名文件）。"""
    mod = _load_mod()
    with tempfile.TemporaryDirectory() as tmp:
        np.save(os.path.join(tmp, "my_dir.npy"), np.random.randn(8).astype(np.float32))
        mod._DATA_DIR = tmp
        # 绝对路径 + ../ 均回落为纯文件名
        abs_path = os.path.join(tmp, "my_dir.npy")
        assert mod._load_direction_file(abs_path, "cpu", 8) is not None, "绝对路径应回落为 basename"
        assert mod._load_direction_file("../outside.npy", "cpu", 8) is None, "越界文件应直通"


def test_node_registered():
    mod = _load_mod()
    assert "MiniMaxH3Steering" in mod.NODE_CLASS_MAPPINGS
    # 兼容回归：direction_file 必须追加在末尾（ComfyUI 按位置恢复旧工作流值，
    # 插中间会导致 lam 收到 layer_band 的 "40-49" → FLOAT 转换崩溃）
    required = list(mod.MiniMaxH3Steering.INPUT_TYPES()["required"].keys())
    assert required[-1] == "direction_file", f"direction_file 必须在末尾，当前顺序 {required}"
    assert required[:5] == ["clip", "steer_refusal", "steer_safety", "lam", "layer_band"]
    assert mod.MiniMaxH3Steering.VALIDATE_INPUTS() is True, "旧工作流 COMBO 残留值必须放行"
    assert "MiniMax H3 Steering" in mod.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3Steering"]
