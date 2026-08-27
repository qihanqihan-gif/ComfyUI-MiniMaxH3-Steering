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
# 2026-08-13：用户模板库根目录——子文件夹 = 可选的模块库（文件夹隔离）
_EXTRA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extra_module_roots")
_FOLDER_NONE = "内置（默认）"
_MODULE_FILE_NONE = "（未选择）"

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

_PROMPT_MODULE_SCOPES = ["全部", "T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA"]
_VALID_SCOPE_TOKENS = set(_PROMPT_MODULE_SCOPES)
_PROMPT_MODULE_NONE = "（无）"
_PROMPT_MODULE_MAX_CHARS = 6000
_EVIDENCE_LABELS = {
    "official_protocol": "官方协议",
    "official_skill_adaptation": "官方 Skill 改编",
    "official_aligned": "官方规则对齐",
    "community_heuristic": "社区经验",
    "experimental": "实验",
}


def _scope_tokens(value) -> set[str]:
    raw = str(value or "全部")
    return {item.strip() for item in raw.split(",") if item.strip()} or {"全部"}


def _scope_matches(selected_scope: str, module_scope) -> bool:
    selected = str(selected_scope or "全部").strip()
    allowed = _scope_tokens(module_scope)
    return selected == "全部" or "全部" in allowed or selected in allowed


def _metadata_issues(module: dict) -> list[str]:
    issues = []
    if not str(module.get("source") or "").strip():
        issues.append("缺少 source")
    if not isinstance(module.get("features"), list) or not module.get("features"):
        issues.append("features 必须是非空字符串数组")
    evidence = str(module.get("evidence_level") or "").strip()
    if evidence not in _EVIDENCE_LABELS:
        issues.append(f"evidence_level 无效或缺失：{evidence or '空'}")
    unknown_scopes = _scope_tokens(module.get("scope")) - _VALID_SCOPE_TOKENS
    if unknown_scopes:
        issues.append("scope 含未知值：" + ", ".join(sorted(unknown_scopes)))
    conflicts = module.get("conflicts", [])
    if conflicts is not None and not isinstance(conflicts, list):
        issues.append("conflicts 必须是字符串数组")
    elif isinstance(conflicts, list) and any(not isinstance(item, str) for item in conflicts):
        issues.append("conflicts 必须是字符串数组")
    return issues


def _folder_choices() -> list[str]:
    """模块文件夹下拉（5 槽节点用）：内置（默认）+ modules/ 子文件夹 + extra_module_roots/ 子文件夹。
    modules/ 子文件夹优先（如 nsfw）；同名时 extra_module_roots/ 的不再重复列出（v10.2）。"""
    choices = [_FOLDER_NONE]
    seen = {_FOLDER_NONE}
    for root in (_MODULES_DIR, _EXTRA_ROOT):
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if os.path.isdir(os.path.join(root, name)) and name not in seen:
                choices.append(name)
                seen.add(name)
    return choices


def _folder_choices_bare() -> list[str]:
    """文件夹加载器下拉：空（未选择）+ modules/ 直接子文件夹。
    _load_folder_merged 的相对路径基准是 _MODULES_DIR，故只列该根的子文件夹；
    绝对路径可手输（VALIDATE 放行）。"""
    choices = [""]
    if os.path.isdir(_MODULES_DIR):
        for name in sorted(os.listdir(_MODULES_DIR)):
            if os.path.isdir(os.path.join(_MODULES_DIR, name)):
                choices.append(name)
    return choices


# v10：文件下拉联动——记住上次 folder，file_1..5 下拉按它列出 json 文件
_last_folder = ""


def _folder_json_files(folder_path: str) -> list[str]:
    """返回指定文件夹（相对 modules/ 或绝对路径）下的 json 文件名（排序）。
    越界/不存在返回空列表。"""
    folder = str(folder_path or "").strip()
    if not folder:
        # 空 folder = modules/ 根目录平铺 json（默认视图）
        path_abs = os.path.abspath(_MODULES_DIR)
        if not os.path.isdir(path_abs):
            return []
        return sorted(f for f in os.listdir(path_abs) if f.endswith(".json"))
    path = folder if os.path.isabs(folder) else os.path.join(_MODULES_DIR, folder)
    plugin_root = os.path.abspath(os.path.dirname(_MODULES_DIR))
    path_abs = os.path.abspath(path)
    if not (path_abs == plugin_root or path_abs.startswith(plugin_root + os.sep)):
        return []
    if not os.path.isdir(path_abs):
        return []
    return sorted(f for f in os.listdir(path_abs) if f.endswith(".json"))


# file_1..5 下拉显示中文 title_zh（fallback 文件名）；load 时按 title_zh 内容匹配（见 _file_zh_to_name）


def _module_title_zh(path: str) -> str:
    """读模块 json 的 title_zh（对象或数组首元素），失败返回空串。"""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else [data]
        for m in items:
            if isinstance(m, dict) and str(m.get("title_zh") or "").strip():
                return str(m["title_zh"]).strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def _folder_abs(folder: str) -> str:
    """folder（空/相对 modules/ /绝对）→ 绝对路径；越界返回空串。"""
    folder = str(folder or "").strip()
    if not folder:
        return os.path.abspath(_MODULES_DIR)
    path = folder if os.path.isabs(folder) else os.path.join(_MODULES_DIR, folder)
    plugin_root = os.path.abspath(os.path.dirname(_MODULES_DIR))
    path_abs = os.path.abspath(path)
    if not (path_abs == plugin_root or path_abs.startswith(plugin_root + os.sep)):
        return ""
    return path_abs


def _file_title_choices(folder: str) -> tuple[list[str], dict[str, str]]:
    """给定 folder → (['', title_zh1, ...], {title_zh: filename})。损坏 json 的条目用文件名兜底。"""
    files = _folder_json_files(folder)
    base = _folder_abs(folder)
    mapping: dict[str, str] = {}
    out = [""]
    for fn in files:
        zh = _module_title_zh(os.path.join(base, fn)) if base else ""
        label = zh or fn
        mapping[label] = fn
        out.append(label)
    return out, mapping


def _file_choices() -> list[str]:
    """file_1..5 下拉：基于 _last_folder，显示中文 title_zh（fallback 文件名）。"""
    choices, _ = _file_title_choices(_last_folder)
    return choices


def _file_zh_to_name(folder: str, value: str) -> str:
    """file_N 下拉值 → 磁盘文件名：.json 结尾=旧工作流直通；否则扫描 folder 内 json 按 title_zh 内容匹配。
    直接以中文模块名为准，不依赖任何预建映射；匹配不到原样返回（手输文件名等场景）。"""
    value = str(value or "").strip()
    if not value or value.endswith(".json"):
        return value
    base = _folder_abs(folder)
    if not base:
        return value
    for fn in _folder_json_files(folder):
        if _module_title_zh(os.path.join(base, fn)) == value:
            return fn
    return value


def _path_is_within(path: str, root: str) -> bool:
    """真实路径必须位于指定根目录内（同时防 `..` 与目录符号链接越界）。"""
    try:
        return os.path.commonpath((os.path.realpath(path), os.path.realpath(root))) == os.path.realpath(root)
    except (OSError, ValueError):
        return False


def _file_has_selectable_module(path: str) -> bool:
    """仅把至少含一个创作策略模块的合法 JSON 放进下拉。"""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    items = data if isinstance(data, list) else [data]
    return any(
        isinstance(item, dict)
        and item.get("id")
        and item.get("instructions")
        and item["id"] not in _PROTOCOL_IDS
        and item["id"] not in _LEGACY_PROTOCOL_IDS
        for item in items
    )


def _module_file_index() -> dict[str, str]:
    """递归建立“根标签/相对路径.json → 绝对路径”索引。

    使用正斜杠是为了沿用 ComfyUI 模型加载器的路径分组显示；根标签让内置库与
    用户库即使有同名相对路径也不会冲突。
    """
    index: dict[str, str] = {}
    roots = (("内置", _MODULES_DIR), ("用户库", _EXTRA_ROOT))
    for label, root in roots:
        if not os.path.isdir(root):
            continue
        for current, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                name for name in dirnames
                if name not in {".git", "__pycache__"} and not name.startswith(".")
            )
            for filename in sorted(filenames):
                if not filename.lower().endswith(".json"):
                    continue
                path = os.path.join(current, filename)
                if not _path_is_within(path, root) or not _file_has_selectable_module(path):
                    continue
                relative = os.path.relpath(path, root).replace(os.sep, "/")
                index[f"{label}/{relative}"] = path
    return dict(sorted(index.items(), key=lambda item: item[0].casefold()))


def _module_file_choices() -> list[str]:
    """单下拉路径列表；每次获取节点定义时重新扫描目录。"""
    return [_MODULE_FILE_NONE, *_module_file_index().keys()]


def _load_module_files(selections) -> tuple[str, list[str], int, int]:
    """加载多个路径选择，返回（合并文本、问题、有效模块数、已选文件数）。"""
    selected_files: list[str] = []
    issues: list[str] = []
    seen_files: set[str] = set()
    for raw in selections:
        selected = str(raw or "").strip().replace("\\", "/")
        if not selected or selected == _MODULE_FILE_NONE:
            continue
        if selected in seen_files:
            issues.append(f"重复文件选择已忽略：{selected}")
            continue
        seen_files.add(selected)
        selected_files.append(selected)
    if not selected_files:
        return "", issues, 0, 0

    file_index = _module_file_index()
    sections: list[str] = []
    seen_ids: set[str] = set()
    for selected in selected_files:
        path = file_index.get(selected)
        if not path:
            issues.append(f"模块文件不存在或已移出允许目录：{selected}")
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"模块文件加载失败：{selected}（{exc}）")
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not (isinstance(item, dict) and item.get("id") and item.get("instructions")):
                issues.append(f"已跳过字段不完整的条目：{selected}")
                continue
            module_id = str(item["id"])
            if module_id in _PROTOCOL_IDS or module_id in _LEGACY_PROTOCOL_IDS:
                issues.append(f"协议模块禁止由独立加载器注入，已跳过：{module_id}")
                continue
            if module_id in seen_ids:
                issues.append(f"重复模块 id 已跳过：{module_id}")
                continue
            seen_ids.add(module_id)
            title = str(item.get("title_zh") or module_id)
            evidence = str(item.get("evidence_level") or "experimental")
            sections.append(
                f"[{module_id} | {title} | v{item.get('version', 1)} | "
                f"{_EVIDENCE_LABELS.get(evidence, '未分级')}]\n{item['instructions']}"
            )

    merged = "\n\n".join(sections)
    if len(merged) > _PROMPT_MODULE_MAX_CHARS:
        merged = merged[:_PROMPT_MODULE_MAX_CHARS] + "\n（合并文本超过 6000 字符已截断）"
        issues.append("合并文本超过 6000 字符，末尾已截断")
    if not sections:
        issues.append("所选文件中没有可加载的创作策略模块")
    return merged, issues, len(sections), len(selected_files)


def _load_module_file(selection: str) -> tuple[str, list[str], int]:
    """单文件兼容入口；实际合并逻辑由多文件加载器统一处理。"""
    merged, issues, count, _ = _load_module_files((selection,))
    return merged, issues, count


def _module_dirs(folder: str | None = None) -> list[str]:
    """按文件夹选择扫描目录：None/内置 → 插件 modules/；否则先查 modules/<folder>（v10.2），
    再 fallback extra_module_roots/<folder>（兼容旧「示例库」）。路径越界防护两层都做。"""
    folder = str(folder or "").strip()
    if folder and folder != _FOLDER_NONE:
        plugin_root = os.path.abspath(os.path.dirname(_MODULES_DIR))
        # 1) modules/<folder>/（模块库子文件夹，如 nsfw）
        p1 = os.path.abspath(os.path.join(_MODULES_DIR, folder))
        if p1 == plugin_root or p1.startswith(plugin_root + os.sep):
            if os.path.isdir(p1):
                return [p1]
        # 2) extra_module_roots/<folder>/（旧路径，兼容）
        root_abs = os.path.abspath(_EXTRA_ROOT)
        path = os.path.abspath(os.path.join(root_abs, folder))
        if not (path == root_abs or path.startswith(root_abs + os.sep)):
            LOGGER.warning("MiniMax H3 PromptModuleLoader: 非法文件夹 %s（越界）已忽略", folder)
            return []
        return [path] if os.path.isdir(path) else []
    return [_MODULES_DIR]


def _load_modules(folder: str | None = None) -> list[dict]:
    """扫描模块 json（dict 或 dict 数组）。folder=None → 内置 modules/；否则扫描 extra_module_roots/<folder>/。
    协议/旧协议模块始终只从内置根排除（协议不由文件夹提供）。"""
    modules: list[dict] = []
    seen_ids = set()
    for scan_dir in _module_dirs(folder):
        if not os.path.isdir(scan_dir):
            continue
        for fname in sorted(os.listdir(scan_dir)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(scan_dir, fname)
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else [data]
                for m in items:
                    if not (isinstance(m, dict) and m.get("id") and m.get("instructions")):
                        continue
                    if m["id"] in _PROTOCOL_IDS or m["id"] in _LEGACY_PROTOCOL_IDS:
                        continue  # 协议模块由导演节点自动加载，不列为可选模块
                    if m["id"] in seen_ids:
                        LOGGER.warning("MiniMax H3 PromptModuleLoader: 重复模块 id=%s（%s 已跳过）", m["id"], fname)
                        continue
                    seen_ids.add(m["id"])
                    item = dict(m)
                    item["_file"] = fname
                    item["_metadata_issues"] = _metadata_issues(item)
                    modules.append(item)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("MiniMax H3 PromptModuleLoader: 模块 %s 加载失败: %s", fname, exc)
    return modules


def _module_choices() -> list[str]:
    return [_PROMPT_MODULE_NONE] + [m["title_zh"] for m in _load_modules()]


class MiniMaxH3PromptModuleLoader:
    """提示词模块加载器：scope + 5 个模块槽 + 模块文件夹 + custom，热加载合并为 system 片段。"""

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
                # 新 widget 一律追加在末尾（ComfyUI 按位置恢复旧工作流值）
                "module_4": (_module_choices(), {"default": _PROMPT_MODULE_NONE}),
                "module_5": (_module_choices(), {"default": _PROMPT_MODULE_NONE}),
                "module_folder": (_folder_choices(), {"default": _FOLDER_NONE}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("system_prompt_module", "module_preview", "module_diagnostics", "module_manifest")
    FUNCTION = "load"
    CATEGORY = "MiniMax H3 Lab/Prompt"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # 每次排队重读模块文件（编辑 JSON 即生效）
        return float("nan")

    def load(self, scope="全部", module_1=_PROMPT_MODULE_NONE, module_2=_PROMPT_MODULE_NONE,
             module_3=_PROMPT_MODULE_NONE, module_4=_PROMPT_MODULE_NONE, module_5=_PROMPT_MODULE_NONE,
             custom_instructions="", module_folder=_FOLDER_NONE):
        modules = _load_modules(module_folder)
        by_title = {m["title_zh"]: m for m in modules}
        selected = []
        issues = []
        seen_ids = set()
        for title in (module_1, module_2, module_3, module_4, module_5):
            if title and title != _PROMPT_MODULE_NONE and title in by_title:
                module = by_title[title]
                if module["id"] in seen_ids:
                    issues.append(f"重复选择已忽略：{module['title_zh']}")
                    continue
                seen_ids.add(module["id"])
                selected.append(module)
            elif title and title != _PROMPT_MODULE_NONE:
                issues.append(f"找不到模块：{title}")

        selected_ids = {m["id"] for m in selected}
        for m in selected:
            for conflict in m.get("conflicts") or []:
                if conflict in selected_ids:
                    issues.append(f"模块冲突：{m['id']} ↔ {conflict}")
            issues.extend(f"{m['id']}: {item}" for item in m.get("_metadata_issues", []))

        sections = []
        for m in selected:
            scope_note = ""
            m_scope = str(m.get("scope") or "全部")
            if not _scope_matches(scope, m_scope):
                scope_note = f"（注意：本模块 scope={m_scope}，与当前 {scope} 不匹配）"
                issues.append(f"作用域不匹配：{m['title_zh']} 适用 {m_scope}，当前 {scope}")
            evidence = str(m.get("evidence_level") or "experimental")
            sections.append(
                f"[{m['id']} | {m['title_zh']} | v{m.get('version', 1)} | "
                f"{_EVIDENCE_LABELS.get(evidence, '未分级')}]{scope_note}\n{m['instructions']}"
            )
        custom = (custom_instructions or "").strip()
        if custom:
            sections.append("[workflow_custom]\n" + custom)
        merged = "\n\n".join(sections)
        if len(merged) > _PROMPT_MODULE_MAX_CHARS:
            merged = merged[:_PROMPT_MODULE_MAX_CHARS] + "\n（合并文本过长已截断）"
            issues.append("合并文本超过 6000 字符，末尾已截断")

        preview_lines = [
            f"[{i + 1}] {m['title_zh']}（{_EVIDENCE_LABELS.get(m.get('evidence_level'), '未分级')}，"
            f"scope={m.get('scope') or '全部'}）"
            for i, m in enumerate(selected)
        ]
        if custom:
            preview_lines.append("[自定义] workflow_custom（当前工作流规则，未做证据分级）")
        diag_lines = [f"模块文件夹={module_folder}，可选模块={len(modules)}，已选={len(selected)}，合并字符={len(merged)}"]
        if not modules and module_folder != _FOLDER_NONE:
            diag_lines.append("该文件夹无模块（下拉不随文件夹联动，可在模块槽手输该文件夹内模块的 title_zh）")
        diag_lines.extend(f"[提示] {item}" for item in dict.fromkeys(issues))
        manifest = {
            "schema_version": "h3_prompt_modules/1.0",
            "scope": scope,
            "selected": [
                {
                    "id": m["id"], "title_zh": m["title_zh"],
                    "version": m.get("version", 1), "scope": m.get("scope", "全部"),
                    "evidence_level": m.get("evidence_level", "experimental"),
                    "source": m.get("source", ""), "source_url": m.get("source_url", ""),
                    "features": m.get("features", []),
                }
                for m in selected
            ],
            "custom_instructions": custom,
            "issues": list(dict.fromkeys(issues)),
        }
        return (
            merged,
            "\n".join(preview_lines) if preview_lines else "（未选择模块）",
            "\n".join(diag_lines),
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )


def _load_folder_merged(folder_path: str, selected_files=None) -> tuple[str, list[str]]:
    """扫描文件夹 json 并合并 instructions；selected_files 非空时只加载指定文件名子集（纯文件名，防路径注入）。
    返回 (merged, issues)。损坏文件跳过并记入 issues。
    """
    folder = str(folder_path or "").strip()
    if not folder:
        # 空 folder = modules/ 根目录（与下拉空选项一致，v10.1c）
        path_abs = os.path.abspath(_MODULES_DIR)
    else:
        path = folder if os.path.isabs(folder) else os.path.join(_MODULES_DIR, folder)
        # 越界防护：相对路径解析后必须仍在插件根内（对齐 _module_dirs）
        plugin_root = os.path.abspath(os.path.dirname(_MODULES_DIR))
        path_abs = os.path.abspath(path)
        if not (path_abs == plugin_root or path_abs.startswith(plugin_root + os.sep)):
            return "", [f"文件夹越界已拒绝：{folder}"]
    if not os.path.isdir(path_abs):
        return "", [f"文件夹不存在：{path_abs}"]
    path = path_abs
    selected = None
    if selected_files:
        selected = set()
        for name in selected_files:
            name = str(name or "").strip()
            if not name or "/" in name or "\\" in name:
                continue
            selected.add(name)
    sections = []
    issues = []
    seen_ids = set()
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".json"):
            continue
        if selected is not None and fname not in selected:
            continue
        try:
            with open(os.path.join(path, fname), encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for m in items:
                if not (isinstance(m, dict) and m.get("id") and m.get("instructions")):
                    continue
                if m["id"] in _PROTOCOL_IDS or m["id"] in _LEGACY_PROTOCOL_IDS:
                    issues.append(f"协议模块禁止从文件夹加载器注入，已跳过：{m['id']}（{fname}）")
                    continue
                if m["id"] in seen_ids:
                    issues.append(f"重复模块 id 已跳过：{m['id']}（{fname}）")
                    continue
                seen_ids.add(m["id"])
                evidence = str(m.get("evidence_level") or "experimental")
                sections.append(
                    f"[{m['id']} | {m['title_zh']} | v{m.get('version', 1)} | "
                    f"{_EVIDENCE_LABELS.get(evidence, '未分级')}]\n{m['instructions']}"
                )
        except Exception as exc:  # noqa: BLE001
            issues.append(f"{fname} 加载失败: {exc}")
    if not sections:
        issues.append("该文件夹无有效模块 json（或所选文件不匹配）")
    merged = "\n\n".join(sections)
    if len(merged) > _PROMPT_MODULE_MAX_CHARS:
        merged = merged[:_PROMPT_MODULE_MAX_CHARS] + "\n（合并文本超过 6000 字符已截断）"
        issues.append("合并文本超过 6000 字符，末尾已截断")
    return merged, issues


class MiniMaxH3ModuleFolderLoader:
    """最多组合 5 个递归路径模块文件，交互与 ComfyUI 原生模型加载器一致。"""

    @classmethod
    def INPUT_TYPES(cls):
        choices = _module_file_choices()
        return {
            "required": {
                # 第一槽保留 module_file 名称，兼容单槽版本保存的工作流。
                "module_file": (choices, {"default": _MODULE_FILE_NONE}),
                "module_file_2": (choices, {"default": _MODULE_FILE_NONE}),
                "module_file_3": (choices, {"default": _MODULE_FILE_NONE}),
                "module_file_4": (choices, {"default": _MODULE_FILE_NONE}),
                "module_file_5": (choices, {"default": _MODULE_FILE_NONE}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # 允许旧工作流中的已删除值进入执行阶段并给出可读诊断，而非排队前报 COMBO 错误。
        return True

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("system_prompt_module", "diagnostics")
    FUNCTION = "load_folder"
    CATEGORY = "MiniMax H3 Lab/Prompt"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # 每次排队重读文件夹（编辑 JSON 即生效）
        return float("nan")

    def load_folder(self, module_file=_MODULE_FILE_NONE, module_file_2=_MODULE_FILE_NONE,
                    module_file_3=_MODULE_FILE_NONE, module_file_4=_MODULE_FILE_NONE,
                    module_file_5=_MODULE_FILE_NONE, **_legacy_inputs):
        selections = (module_file, module_file_2, module_file_3, module_file_4, module_file_5)
        merged, issues, count, selected_count = _load_module_files(selections)
        diag = f"已选文件={selected_count}，有效模块={count}，合并字符={len(merged)}"
        if issues:
            diag += "；" + "；".join(issues[:3])
        return merged, diag


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PromptModuleLoader": MiniMaxH3PromptModuleLoader,
    "MiniMaxH3ModuleFolderLoader": MiniMaxH3ModuleFolderLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PromptModuleLoader": "MiniMax H3 提示词模块 (热加载, 实验)",
    "MiniMaxH3ModuleFolderLoader": "MiniMax H3 模块文件夹加载器 (独立, 热加载)",
}
