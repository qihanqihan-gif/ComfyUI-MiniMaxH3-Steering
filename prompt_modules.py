# -*- coding: utf-8 -*-
"""MiniMaxH3PromptModuleLoader — 提示词模块节点（LingBot 式热加载，用户自选组合）。

模块文件：插件 `modules/*.json`，每模块 {"id", "title_zh", "version", "scope", "instructions"}。
- instructions：给 LLM 的 system 片段（官方规范/社区经验浓缩，中文规则 + 英文格式要求）
- 输出合并文本可接 MiniMaxH3PromptDirector 的 system_module 输入（非空时并入其 system）
- IS_CHANGED 返回 nan → 每次排队重读模块文件（改 JSON 即生效，无需重启）

依赖：仅标准库。零第三方。
"""
import json
import logging
import os

LOGGER = logging.getLogger(__name__)

_MODULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules")

# 协议模块：由导演节点按 task_type 自动加载（用户不可选），模块节点只列创作策略
_PROTOCOL_IDS = {"protocol_base", "protocol_ref"}
# 旧版规范模块（v0.1 前曾作为可选模块；official 双模板已并入协议，对白/时间码规则已被协议覆盖，保留文件但不再列出）
_LEGACY_PROTOCOL_IDS = {"official_three_part", "official_six_part", "dialogue_verbatim", "shot_timeline"}
_PROTOCOL_FILE = {"protocol_base": "protocol_base.json", "protocol_ref": "protocol_ref.json"}


def load_protocol(task_type: str) -> str:
    """按任务类型加载核心协议（三段式 base / 六段式 ref）。

    Ref2VA → protocol_ref；其余（T2VA/I2VA/FL2VA/L2VA）→ protocol_base。
    找不到文件/损坏时返回空串（不阻塞，导演节点有内置基础规则兜底）。
    """
    key = "protocol_ref" if str(task_type).upper() == "REF2VA" else "protocol_base"
    path = os.path.join(_MODULES_DIR, _PROTOCOL_FILE[key])
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else [data]
        for m in items:
            if isinstance(m, dict) and m.get("id") == key and m.get("instructions"):
                return m["instructions"]
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("MiniMax H3 PromptModuleLoader: 协议 %s 加载失败: %s", key, exc)
    return ""

_PROMPT_MODULE_SCOPES = ["全部", "T2VA", "I2VA", "FL2VA", "Ref2VA"]
_PROMPT_MODULE_NONE = "（无）"
_PROMPT_MODULE_MAX_CHARS = 6000


def _load_modules() -> list[dict]:
    """扫描 modules/*.json 加载模块（dict 或 dict 数组）。损坏文件跳过并告警。"""
    modules: list[dict] = []
    if not os.path.isdir(_MODULES_DIR):
        return modules
    for fname in sorted(os.listdir(_MODULES_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(_MODULES_DIR, fname)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for m in items:
                if not (isinstance(m, dict) and m.get("id") and m.get("instructions")):
                    continue
                if m["id"] in _PROTOCOL_IDS or m["id"] in _LEGACY_PROTOCOL_IDS:
                    continue  # 协议模块由导演节点自动加载，不列为可选模块
                modules.append(m)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("MiniMax H3 PromptModuleLoader: 模块 %s 加载失败: %s", fname, exc)
    return modules


def _module_choices() -> list[str]:
    return [_PROMPT_MODULE_NONE] + [m["title_zh"] for m in _load_modules()]


class MiniMaxH3PromptModuleLoader:
    """提示词模块加载器：scope + 3 个模块槽 + custom，热加载合并为 system 片段。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scope": (_PROMPT_MODULE_SCOPES, {"default": "全部"}),
                "module_1": (_module_choices(), {"default": _PROMPT_MODULE_NONE}),
                "module_2": (_module_choices(), {"default": _PROMPT_MODULE_NONE}),
                "module_3": (_module_choices(), {"default": _PROMPT_MODULE_NONE}),
                "custom_instructions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "可选：只在当前工作流追加规则（如固定镜头/风格/负面约束）",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("system_prompt_module", "module_preview", "module_diagnostics")
    FUNCTION = "load"
    CATEGORY = "MiniMax H3 Lab/Prompt"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # 每次排队重读模块文件（编辑 JSON 即生效）
        return float("nan")

    def load(self, scope="全部", module_1=_PROMPT_MODULE_NONE, module_2=_PROMPT_MODULE_NONE,
             module_3=_PROMPT_MODULE_NONE, custom_instructions=""):
        modules = _load_modules()
        by_title = {m["title_zh"]: m for m in modules}
        selected = []
        for title in (module_1, module_2, module_3):
            if title and title != _PROMPT_MODULE_NONE and title in by_title:
                selected.append(by_title[title])

        sections = []
        for m in selected:
            scope_note = ""
            m_scope = str(m.get("scope") or "")
            # 「全部」= 通配（策略模块默认）；节点 scope 或模块 scope 任一为「全部」即匹配
            if scope != "全部" and m_scope and m_scope != "全部" and scope not in m_scope:
                scope_note = f"（注意：本模块 scope={m_scope}，与当前 {scope} 不完全匹配）"
            sections.append(
                f"[{m['id']} | {m['title_zh']} | v{m.get('version', 1)}]{scope_note}\n{m['instructions']}"
            )
        custom = (custom_instructions or "").strip()
        if custom:
            sections.append("[workflow_custom]\n" + custom)
        merged = "\n\n".join(sections)
        if len(merged) > _PROMPT_MODULE_MAX_CHARS:
            merged = merged[:_PROMPT_MODULE_MAX_CHARS] + "\n（合并文本过长已截断）"

        preview_lines = [
            f"[{i + 1}] {m['title_zh']}（v{m.get('version', 1)}，scope={m.get('scope') or '全部'}）"
            for i, m in enumerate(selected)
        ]
        if custom:
            preview_lines.append("[3+] workflow_custom（当前工作流自定义）")
        diag = f"模块文件数={len(modules)}，已选={len(selected)}，合并字符={len(merged)}"
        return (merged, "\n".join(preview_lines) if preview_lines else "（未选择模块）", diag)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PromptModuleLoader": MiniMaxH3PromptModuleLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PromptModuleLoader": "MiniMax H3 提示词模块 (热加载, 实验)",
}
