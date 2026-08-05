# -*- coding: utf-8 -*-
"""MiniMaxH3EasyCacheSafe 最小单元测试（不触碰模型文件/网络）。"""
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


def _load_mod():
    spec = importlib.util.spec_from_file_location("lab_easycache_safe", LAB_ROOT + "/easycache_safe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lab_easycache_safe"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_presets_values():
    mod = _load_mod()
    assert mod.PRESETS["conservative"] == (0.10, 0.15, 0.90)
    assert mod.PRESETS["balanced"] == (0.15, 0.15, 0.90)
    assert mod.PRESETS["aggressive"] == (0.20, 0.15, 0.85)


def test_is_minimax_h3_positive():
    mod = _load_mod()

    class MiniMaxH3Model:
        blocks = [object()] * 50

    assert mod._is_minimax_h3(MiniMaxH3Model()) is True


def test_is_minimax_h3_negative():
    mod = _load_mod()

    class NotH3:
        blocks = [object()]

    class H3NoBlocks:
        pass

    assert mod._is_minimax_h3(NotH3()) is False
    assert mod._is_minimax_h3(H3NoBlocks()) is False
    assert mod._is_minimax_h3(None) is False


def test_node_registered():
    mod = _load_mod()
    assert "MiniMaxH3EasyCacheSafe" in mod.NODE_CLASS_MAPPINGS
    assert "MiniMax H3 EasyCache Safe" in mod.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3EasyCacheSafe"]


def test_official_easycache_available_in_env():
    """本地 ComfyUI 0.30 必须能导入官方 EasyCache，否则节点运行时不可用。"""
    mod = _load_mod()
    official = mod._load_official_easycache()
    assert official is not None, "comfy_extras.nodes_easycache 在本机 ComfyUI 不可用（需 0.30+）"
    assert len(official) == 4
