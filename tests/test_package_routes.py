# -*- coding: utf-8 -*-
"""Package-level API route regression tests."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_module_files_route_uses_package_relative_prompt_modules_import():
    """ComfyUI loads custom nodes as packages, so route imports must stay relative."""
    tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
    route = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_module_files"
    )

    assert any(
        isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module is None
        and any(alias.name == "prompt_modules" and alias.asname == "_pm" for alias in node.names)
        for node in ast.walk(route)
    )
    assert not any(
        isinstance(node, ast.Import)
        and any(alias.name == "prompt_modules" for alias in node.names)
        for node in ast.walk(route)
    )
