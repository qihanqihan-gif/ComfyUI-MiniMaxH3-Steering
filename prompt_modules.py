# -*- coding: utf-8 -*-
"""MiniMaxH3PromptModuleLoader — 提示词模块节点（LingBot 式热加载，用户自选组合）。

模块文件统一位于 `prompt_modules/`：`builtin/` 正式库、`legacy/` 兼容库、`user/` 用户库。
旧模块只需 {"id", "title_zh", "instructions"}；
新模块可追加 semantic_contract / ownership / resolution 元数据。
- instructions：给 LLM 的 system 片段（官方规范/社区经验浓缩，中文规则 + 英文格式要求）
- resolution：由本文件的确定性 Resolver 在调用 LLM 前消解作用域、依赖、冲突与互斥组
- 输出合并文本可接 MiniMaxH3PromptDirector 的 system_module 输入（非空时并入其 system）
- IS_CHANGED 返回 nan → 每次排队重读模块文件（改 JSON 即生效，无需重启）

依赖：仅标准库。零第三方。
"""
import hashlib
import json
import logging
import os

LOGGER = logging.getLogger(__name__)

_PLUGIN_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROMPT_MODULES_ROOT = os.path.join(_PLUGIN_ROOT, "prompt_modules")
_CANONICAL_MODULES_DIR = os.path.join(_PROMPT_MODULES_ROOT, "builtin")
_LEGACY_MODULES_DIR = os.path.join(_PROMPT_MODULES_ROOT, "legacy")
_EXTRA_ROOT = os.path.join(_PROMPT_MODULES_ROOT, "user")

# 旧版曾把三种职责平铺在插件根目录。新安装只暴露 prompt_modules/ 一个入口；
# 覆盖升级后若旧目录仍存在，继续只读扫描，避免用户自建规则或旧工作流静默失效。
_OLD_CANONICAL_MODULES_DIR = _PROMPT_MODULES_ROOT
_OLD_LEGACY_MODULES_DIR = os.path.join(_PLUGIN_ROOT, "modules")
_OLD_EXTRA_ROOT = os.path.join(_PLUGIN_ROOT, "extra_module_roots")
_PROMPT_MODULES_DIR = (
    _CANONICAL_MODULES_DIR
    if os.path.isdir(_CANONICAL_MODULES_DIR)
    else _OLD_CANONICAL_MODULES_DIR
)
_MODULES_DIR = _PROMPT_MODULES_DIR
_FOLDER_NONE = "内置（默认）"
_MODULE_FILE_NONE = "（未选择）"

# 协议模块：由导演节点按 task_type 自动加载（用户不可选），模块节点只列创作策略
_PROTOCOL_IDS = {"protocol_base", "protocol_ref"}
# 旧版规范模块（v0.1 前曾作为可选模块；official 双模板已并入协议，对白/时间码规则已被协议覆盖，保留文件但不再列出）
_LEGACY_PROTOCOL_IDS = {"official_three_part", "official_six_part", "dialogue_verbatim", "shot_timeline"}
# 已降级为作者参考资料的旧规则。ID 级保护同时覆盖只读 legacy 目录，避免旧副本重新出现在下拉中。
_REFERENCE_ONLY_IDS = {"community_8_tips"}
_PROTOCOL_FILES = {
    "protocol_base": ("核心协议：三段式（自动加载）.json", "protocol_base.json"),
    "protocol_ref": ("核心协议：六段式 Ref2VA（自动加载）.json", "protocol_ref.json"),
}


def load_protocol(task_type: str) -> str:
    """按任务类型加载核心协议（三段式 base / 六段式 ref）。

    Ref2VA → protocol_ref；其余（T2VA/I2VA/FL2VA/L2VA）→ protocol_base。
    找不到文件/损坏时返回空串（不阻塞，导演节点有内置基础规则兜底）。
    """
    key = "protocol_ref" if str(task_type).upper() == "REF2VA" else "protocol_base"
    roots = [_MODULES_DIR]
    if os.path.realpath(_MODULES_DIR) != os.path.realpath(_LEGACY_MODULES_DIR):
        roots.append(_LEGACY_MODULES_DIR)
    last_error = None
    for root in roots:
        for filename in _PROTOCOL_FILES[key]:
            path = os.path.join(root, filename)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else [data]
                for m in items:
                    if isinstance(m, dict) and m.get("id") == key and m.get("instructions"):
                        return m["instructions"]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
    if last_error is not None:
        LOGGER.warning("MiniMax H3 PromptModuleLoader: 协议 %s 加载失败: %s", key, last_error)
    return ""

_PROMPT_MODULE_SCOPES = ["全部", "T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA"]
_VALID_SCOPE_TOKENS = set(_PROMPT_MODULE_SCOPES)
_PROMPT_MODULE_NONE = "（无）"
_PROMPT_MODULE_MAX_CHARS = 6000
_RENDER_PROFILES = ["standard", "compact", "strong"]
_RENDER_PROFILE_LABELS = {
    "standard": "标准（当前正文）",
    "compact": "紧凑（契约摘要，实验）",
    "strong": "强化（正文 + 契约复核，实验）",
}
_EVIDENCE_LABELS = {
    "official_protocol": "官方协议",
    "official_skill_adaptation": "官方 Skill 改编",
    "official_aligned": "官方规则对齐",
    "community_heuristic": "社区经验",
    "experimental": "实验",
}


def _using_canonical_module_dir() -> bool:
    return os.path.realpath(_MODULES_DIR) == os.path.realpath(_PROMPT_MODULES_DIR)


def _existing_unique_roots(*roots: str) -> list[str]:
    """保序返回存在且真实路径不重复的目录。"""
    result: list[str] = []
    seen: set[str] = set()
    for root in roots:
        real = os.path.realpath(root)
        if real in seen or not os.path.isdir(root):
            continue
        seen.add(real)
        result.append(root)
    return result


def _legacy_module_roots() -> list[str]:
    return _existing_unique_roots(_LEGACY_MODULES_DIR, _OLD_LEGACY_MODULES_DIR)


def _extra_module_roots() -> list[str]:
    return _existing_unique_roots(_EXTRA_ROOT, _OLD_EXTRA_ROOT)


def _builtin_module_roots() -> list[str]:
    """正式目录优先；兼容目录只补充唯一规则，重复 id 由加载器去重。"""
    roots = [_MODULES_DIR]
    if _using_canonical_module_dir():
        roots.extend(_legacy_module_roots())
    return roots


def _scope_tokens(value) -> set[str]:
    raw = str(value or "全部")
    return {item.strip() for item in raw.split(",") if item.strip()} or {"全部"}


def _scope_matches(selected_scope: str, module_scope) -> bool:
    selected = str(selected_scope or "全部").strip()
    allowed = _scope_tokens(module_scope)
    return selected == "全部" or "全部" in allowed or selected in allowed


def _string_list(value) -> list[str]:
    """把可选字符串/字符串数组规范成去重列表；非法值返回空列表。"""
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        return []
    result = []
    seen = set()
    for item in values:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _normalize_render_profile(value) -> str:
    """把旧值、中文显示值和非法输入收敛到稳定英文档位。"""
    text = str(value or "standard").strip()
    aliases = {
        "标准": "standard",
        "紧凑": "compact",
        "强化": "strong",
        **{label: key for key, label in _RENDER_PROFILE_LABELS.items()},
    }
    normalized = aliases.get(text, text.casefold())
    return normalized if normalized in _RENDER_PROFILES else "standard"


def _contract_value_text(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return str(value).strip()


def _contract_item_text(item) -> str:
    if isinstance(item, dict):
        path = str(item.get("path") or "").strip()
        value = _contract_value_text(item.get("value"))
        if path and value:
            return f"{path}={value}"
        return path or value
    return _contract_value_text(item)


def _render_contract_summary(module: dict, compact=False) -> str:
    """把 canonical contract 确定性渲染成可比较文本。

    这里不猜测供应商能力，也不翻译语义路径；稳定路径和值保留原样，方便不同模型、
    provider 和 wire 使用完全相同的实验输入。
    """
    contract = module.get("semantic_contract")
    if not isinstance(contract, dict):
        return ""
    lines = []
    purpose = str(contract.get("purpose") or "").strip()
    if purpose:
        lines.append("目标：" + purpose)
    if compact:
        ownership = module.get("ownership")
        ownership = ownership if isinstance(ownership, dict) else {}
        writes = _string_list(ownership.get("writes"))
        if writes:
            lines.append("主责：" + "；".join(writes))
        for field, heading in (("preserves", "关键保留"), ("forbids", "关键禁止")):
            raw_items = contract.get(field)
            if not isinstance(raw_items, list):
                continue
            # compact 是低显式度实验档：保留契约目的、主所有权和每类前两个高优先边界。
            items = [text for text in (_contract_item_text(item) for item in raw_items[:2]) if text]
            if items:
                lines.append(f"{heading}：" + "；".join(items))
        return "\n".join(lines)
    headings = (
        ("adds", "新增语义"),
        ("preserves", "必须保留"),
        ("forbids", "禁止"),
        ("defaults", "默认值"),
    )
    for field, heading in headings:
        raw_items = contract.get(field)
        if not isinstance(raw_items, list):
            continue
        items = [text for text in (_contract_item_text(item) for item in raw_items) if text]
        if items:
            lines.append(f"{heading}：" + "；".join(items))
    return "\n".join(lines)


def _render_module_instructions(module: dict, render_profile="standard") -> tuple[str, str]:
    """返回（渲染文本，来源）。standard 保证与既有工作流完全一致。"""
    profile = _normalize_render_profile(render_profile)
    instructions = str(module.get("instructions") or "").strip()
    if profile == "standard":
        return instructions, "instructions"
    contract_text = _render_contract_summary(module)
    if not contract_text:
        return instructions, "fallback_instructions"
    if profile == "compact":
        return _render_contract_summary(module, compact=True), "semantic_contract_summary"
    return (
        instructions + "\n\n[Canonical semantic contract 复核]\n" + contract_text,
        "instructions+semantic_contract",
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _json_sha256(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(payload)


def _resolution_metadata(module: dict) -> dict:
    """读取 v1/v1.1 兼容的 Resolver 元数据，不让用户模块的坏字段使节点崩溃。"""
    nested = module.get("resolution")
    nested = nested if isinstance(nested, dict) else {}
    conflicts = _string_list(module.get("conflicts"))
    for item in _string_list(nested.get("conflicts")):
        if item not in conflicts:
            conflicts.append(item)
    groups = _string_list(nested.get("exclusive_group", module.get("exclusive_group")))
    raw_priority = nested.get("priority", module.get("priority", 0))
    try:
        priority = int(raw_priority)
    except (TypeError, ValueError):
        priority = 0
    return {
        "requires": _string_list(nested.get("requires", module.get("requires"))),
        "conflicts": conflicts,
        "exclusive_group": groups,
        "priority": priority,
    }


def _metadata_issues(module: dict) -> list[str]:
    issues = []
    if not str(module.get("source") or "").strip():
        issues.append("缺少 source")
    if not isinstance(module.get("features"), list) or not module.get("features"):
        issues.append("features 必须是非空字符串数组")
    selectable = module.get("selectable")
    if selectable is not None and not isinstance(selectable, bool):
        issues.append("selectable 必须是布尔值")
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
    semantic_contract = module.get("semantic_contract")
    if semantic_contract is not None and not isinstance(semantic_contract, dict):
        issues.append("semantic_contract 必须是对象")
    ownership = module.get("ownership")
    if ownership is not None and not isinstance(ownership, dict):
        issues.append("ownership 必须是对象")
    elif isinstance(ownership, dict):
        for field in ("reads", "writes", "augments"):
            value = ownership.get(field)
            if value is not None and (
                not isinstance(value, list) or any(not isinstance(item, str) for item in value)
            ):
                issues.append(f"ownership.{field} 必须是字符串数组")
    resolution = module.get("resolution")
    if resolution is not None and not isinstance(resolution, dict):
        issues.append("resolution 必须是对象")
    elif isinstance(resolution, dict):
        for field in ("requires", "conflicts"):
            value = resolution.get(field)
            if value is not None and (
                not isinstance(value, list) or any(not isinstance(item, str) for item in value)
            ):
                issues.append(f"resolution.{field} 必须是字符串数组")
        group = resolution.get("exclusive_group")
        if group is not None and not (
            isinstance(group, str)
            or (isinstance(group, list) and all(isinstance(item, str) for item in group))
        ):
            issues.append("resolution.exclusive_group 必须是字符串或字符串数组")
        priority = resolution.get("priority")
        if priority is not None and (isinstance(priority, bool) or not isinstance(priority, int)):
            issues.append("resolution.priority 必须是整数")
    return issues


def _manifest_module(module: dict) -> dict:
    """生成可追踪但不含完整 instructions 的模块摘要。"""
    return {
        "id": module["id"],
        "title_zh": module.get("title_zh", module["id"]),
        "version": module.get("version", 1),
        "scope": module.get("scope", "全部"),
        "category": module.get("category", ""),
        "evidence_level": module.get("evidence_level", "experimental"),
        "source": module.get("source", ""),
        "source_url": module.get("source_url", ""),
        "features": module.get("features", []),
        "semantic_contract": module.get("semantic_contract", {}),
        "ownership": module.get("ownership", {}),
        "resolution": _resolution_metadata(module),
        "origin": module.get("_origin", "builtin"),
    }


def _resolve_modules(selected: list[dict], scope: str = "全部") -> tuple[list[dict], list[dict], list[str], list[dict]]:
    """确定性消解模块。

    优先级高者先占用互斥语义域；同优先级按用户槽位顺序决定，因而相同输入始终
    得到相同结果。返回（生效模块、被压制模块、诊断、完整解析轨迹）。
    """
    requested_ids = {str(module.get("id")) for module in selected if module.get("id")}
    indexed = list(enumerate(selected))
    candidates: list[tuple[int, dict]] = []
    trace_by_id: dict[str, dict] = {}
    issues: list[str] = []

    def suppress(index: int, module: dict, code: str, reason: str, winner: str = ""):
        module_id = str(module.get("id") or "")
        trace_by_id[module_id] = {
            "id": module_id,
            "title_zh": module.get("title_zh", module_id),
            "selected_index": index + 1,
            "priority": _resolution_metadata(module)["priority"],
            "status": "suppressed",
            "reason_code": code,
            "reason": reason,
            "winner": winner,
        }
        issues.append(reason)

    for index, module in indexed:
        module_id = str(module.get("id") or "")
        if not _scope_matches(scope, module.get("scope")):
            suppress(
                index, module, "scope_mismatch",
                f"规则未应用：{module.get('title_zh', module_id)} 适用 {module.get('scope') or '全部'}，当前 {scope}",
            )
            continue
        missing = [item for item in _resolution_metadata(module)["requires"] if item not in requested_ids]
        if missing:
            suppress(
                index, module, "missing_requirement",
                f"规则未应用：{module.get('title_zh', module_id)} 缺少依赖 {', '.join(missing)}",
            )
            continue
        candidates.append((index, module))

    ranked = sorted(
        candidates,
        key=lambda item: (-_resolution_metadata(item[1])["priority"], item[0], str(item[1].get("id"))),
    )
    accepted: list[tuple[int, dict]] = []
    for index, module in ranked:
        module_id = str(module.get("id") or "")
        meta = _resolution_metadata(module)
        blocked = False
        for winner_index, winner in accepted:
            winner_id = str(winner.get("id") or "")
            winner_meta = _resolution_metadata(winner)
            pairwise = winner_id in meta["conflicts"] or module_id in winner_meta["conflicts"]
            shared_groups = sorted(set(meta["exclusive_group"]) & set(winner_meta["exclusive_group"]))
            if not pairwise and not shared_groups:
                continue
            if pairwise:
                reason = (
                    f"规则冲突已消解：{module.get('title_zh', module_id)} 未应用；"
                    f"保留 {winner.get('title_zh', winner_id)}"
                )
                code = "conflict"
            else:
                reason = (
                    f"互斥规则已消解：{module.get('title_zh', module_id)} 未应用；"
                    f"{', '.join(shared_groups)} 已由 {winner.get('title_zh', winner_id)} 占用"
                )
                code = "exclusive_group"
            suppress(index, module, code, reason, winner_id)
            blocked = True
            break
        if not blocked:
            accepted.append((index, module))

    # 依赖项可能在冲突阶段被压制；此时依赖它的规则也不能继续生效。
    changed = True
    while changed:
        changed = False
        accepted_ids = {str(module.get("id")) for _, module in accepted}
        kept = []
        for index, module in accepted:
            missing = [item for item in _resolution_metadata(module)["requires"] if item not in accepted_ids]
            if missing:
                module_id = str(module.get("id") or "")
                suppress(
                    index, module, "suppressed_requirement",
                    f"规则未应用：{module.get('title_zh', module_id)} 的依赖未生效 {', '.join(missing)}",
                )
                changed = True
            else:
                kept.append((index, module))
        accepted = kept

    accepted.sort(key=lambda item: item[0])
    for index, module in accepted:
        module_id = str(module.get("id") or "")
        trace_by_id[module_id] = {
            "id": module_id,
            "title_zh": module.get("title_zh", module_id),
            "selected_index": index + 1,
            "priority": _resolution_metadata(module)["priority"],
            "status": "applied",
            "reason_code": "selected",
            "reason": "通过作用域、依赖与冲突消解",
            "winner": "",
        }
    trace = [trace_by_id[str(module.get("id"))] for _, module in indexed if str(module.get("id")) in trace_by_id]
    suppressed_ids = {item["id"] for item in trace if item["status"] == "suppressed"}
    suppressed = [module for module in selected if str(module.get("id")) in suppressed_ids]
    return [module for _, module in accepted], suppressed, list(dict.fromkeys(issues)), trace


def _folder_choices() -> list[str]:
    """规则文件夹下拉：正式目录、旧目录唯一扩展与用户库子文件夹。"""
    choices = [_FOLDER_NONE]
    seen = {_FOLDER_NONE}
    for root in (*_builtin_module_roots(), *_extra_module_roots()):
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if os.path.isdir(os.path.join(root, name)) and name not in seen:
                choices.append(name)
                seen.add(name)
    return choices


def _folder_choices_bare() -> list[str]:
    """文件夹加载器下拉：空（未选择）+ prompt_modules/ 直接子文件夹。
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
    """返回指定文件夹（相对 prompt_modules/ 或绝对路径）下的 json 文件名（排序）。
    越界/不存在返回空列表。"""
    folder = str(folder_path or "").strip()
    if not folder:
        # 空 folder = 正式规则库根目录平铺 JSON（默认视图）
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
    """folder（空/相对 prompt_modules/ /绝对）→ 绝对路径；越界返回空串。"""
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


def _module_items_from_path(path: str) -> list[dict]:
    """读取 JSON 中的模块对象；损坏文件统一返回空列表。"""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    items = data if isinstance(data, list) else [data]
    return [item for item in items if isinstance(item, dict)]


def _is_selectable_module(item: dict) -> bool:
    """普通可选规则判定；协议、迁移别名和参考资料均不可注入。"""
    module_id = str(item.get("id") or "")
    return bool(
        module_id
        and item.get("instructions")
        and item.get("selectable", True) is not False
        and module_id not in _PROTOCOL_IDS
        and module_id not in _LEGACY_PROTOCOL_IDS
        and module_id not in _REFERENCE_ONLY_IDS
    )


def _file_has_selectable_module(path: str) -> bool:
    """仅把至少含一个创作策略模块的合法 JSON 放进下拉。"""
    return any(_is_selectable_module(item) for item in _module_items_from_path(path))


def _selectable_module_ids(path: str) -> list[str]:
    """返回文件内可选规则 id；损坏文件与协议文件返回空列表。"""
    return [
        str(item["id"])
        for item in _module_items_from_path(path)
        if _is_selectable_module(item)
    ]


def _declared_module_ids(path: str) -> list[str]:
    """返回所有合法声明 id，用于正式目录压制 legacy 重复副本。"""
    return [
        str(item["id"])
        for item in _module_items_from_path(path)
        if item.get("id") and item.get("instructions")
    ]


def _module_file_index() -> dict[str, str]:
    """递归建立“根标签/相对路径.json → 绝对路径”索引。

    使用正斜杠是为了沿用 ComfyUI 模型加载器的路径分组显示；根标签让内置库与
    用户库即使有同名相对路径也不会冲突。
    """
    index: dict[str, str] = {}
    canonical_ids: set[str] = set()
    if _using_canonical_module_dir():
        for current, _dirnames, filenames in os.walk(_MODULES_DIR):
            for filename in filenames:
                if filename.lower().endswith(".json"):
                    canonical_ids.update(_declared_module_ids(os.path.join(current, filename)))
        roots = [("内置", _MODULES_DIR, set())]
        roots.extend(("旧目录", root, canonical_ids) for root in _legacy_module_roots())
        roots.extend(("用户库", root, set()) for root in _extra_module_roots())
    else:
        roots = [("内置", _MODULES_DIR, set())]
        roots.extend(("用户库", root, set()) for root in _extra_module_roots())
    for label, root, skip_ids in roots:
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
                module_ids = _selectable_module_ids(path)
                if (not _path_is_within(path, root) or not module_ids
                        or (skip_ids and set(module_ids) <= skip_ids)):
                    continue
                relative = os.path.relpath(path, root).replace(os.sep, "/")
                # 新统一目录优先；同名旧路径只作为兼容回退，不覆盖新文件。
                index.setdefault(f"{label}/{relative}", path)
    return dict(sorted(index.items(), key=lambda item: item[0].casefold()))


def _legacy_module_file_aliases(file_index: dict[str, str]) -> dict[str, str]:
    """旧工作流的 `内置/english_name.json` 静默映射到中文正式文件。"""
    legacy_roots = _legacy_module_roots()
    if not _using_canonical_module_dir() or not legacy_roots:
        return {}
    canonical_by_id: dict[str, str] = {}
    for selection, path in file_index.items():
        if not selection.startswith("内置/"):
            continue
        for module_id in _selectable_module_ids(path):
            canonical_by_id.setdefault(module_id, path)
    aliases: dict[str, str] = {}
    for legacy_root in legacy_roots:
        for current, dirnames, filenames in os.walk(legacy_root):
            dirnames[:] = sorted(name for name in dirnames if name not in {".git", "__pycache__"})
            for filename in sorted(filenames):
                if not filename.lower().endswith(".json"):
                    continue
                path = os.path.join(current, filename)
                module_ids = _selectable_module_ids(path)
                if not module_ids:
                    continue
                relative = os.path.relpath(path, legacy_root).replace(os.sep, "/")
                aliases.setdefault(
                    f"内置/{relative}",
                    next(
                        (canonical_by_id[module_id] for module_id in module_ids if module_id in canonical_by_id),
                        path,
                    ),
                )
    return aliases


def _module_file_choices() -> list[str]:
    """单下拉路径列表；每次获取节点定义时重新扫描目录。"""
    return [_MODULE_FILE_NONE, *_module_file_index().keys()]


def _empty_rule_pack(source_files=None) -> dict:
    return {
        "schema_version": "h3_prompt_rule_pack/1.0",
        "modules": [],
        "source_files": list(source_files or []),
    }


def _load_module_files(selections) -> tuple[str, list[str], int, int, dict]:
    """原样加载多个文件，返回文本、问题、模块数、文件数与结构化规则包。

    此入口刻意不运行 Resolver：它既可作为简单直连节点使用，也可把结构化规则包
    交给「创作规则组合」统一执行作用域、依赖、冲突和互斥组消解。
    """
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
        return "", issues, 0, 0, _empty_rule_pack()

    file_index = _module_file_index()
    legacy_aliases = None
    collected: list[dict] = []
    seen_ids: set[str] = set()
    for selected in selected_files:
        path = file_index.get(selected)
        if not path:
            # 只有旧工作流真的携带英文路径时才扫描兼容映射，日常中文路径不付这笔 I/O。
            if legacy_aliases is None:
                legacy_aliases = _legacy_module_file_aliases(file_index)
            path = legacy_aliases.get(selected)
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
            if not _is_selectable_module(item):
                issues.append(f"参考资料模块不可由独立加载器注入，已跳过：{module_id}")
                continue
            if module_id in seen_ids:
                issues.append(f"重复模块 id 已跳过：{module_id}")
                continue
            seen_ids.add(module_id)
            module = dict(item)
            module["title_zh"] = str(module.get("title_zh") or module_id)
            module["_origin"] = "file_loader"
            module["_metadata_issues"] = _metadata_issues(module)
            collected.append(module)

    sections: list[str] = []
    for module in collected:
        module_id = str(module["id"])
        title = str(module.get("title_zh") or module_id)
        evidence = str(module.get("evidence_level") or "experimental")
        sections.append(
            f"[{module_id} | {title} | v{module.get('version', 1)} | "
            f"{_EVIDENCE_LABELS.get(evidence, '未分级')}]\n{module['instructions']}"
        )

    merged = "\n\n".join(sections)
    if len(merged) > _PROMPT_MODULE_MAX_CHARS:
        merged = merged[:_PROMPT_MODULE_MAX_CHARS] + "\n（合并文本超过 6000 字符已截断）"
        issues.append("合并文本超过 6000 字符，末尾已截断")
    if not sections:
        issues.append("所选文件中没有可加载的创作策略模块")
    pack = _empty_rule_pack(selected_files)
    pack["modules"] = collected
    return merged, issues, len(sections), len(selected_files), pack


def _load_module_file(selection: str) -> tuple[str, list[str], int]:
    """单文件兼容入口；实际合并逻辑由多文件加载器统一处理。"""
    merged, issues, count, _, _pack = _load_module_files((selection,))
    return merged, issues, count


def _module_dirs(folder: str | None = None) -> list[str]:
    """按文件夹选择扫描目录；正式库优先，旧目录仅补充唯一规则。"""
    folder = str(folder or "").strip()
    if folder and folder != _FOLDER_NONE:
        builtins = []
        for root in _builtin_module_roots():
            path = os.path.abspath(os.path.join(root, folder))
            if _path_is_within(path, root) and os.path.isdir(path):
                builtins.append(path)
        if builtins:
            return builtins
        user_dirs = []
        for root in _extra_module_roots():
            path = os.path.abspath(os.path.join(root, folder))
            if not _path_is_within(path, root):
                LOGGER.warning("MiniMax H3 PromptModuleLoader: 非法文件夹 %s（越界）已忽略", folder)
                return []
            if os.path.isdir(path):
                user_dirs.append(path)
        return user_dirs
    return _builtin_module_roots()


def _load_modules(folder: str | None = None) -> list[dict]:
    """扫描模块 JSON；正式库优先，旧目录只补充不重复的用户规则。"""
    modules: list[dict] = []
    seen_ids = set()
    for scan_dir in _module_dirs(folder):
        if not os.path.isdir(scan_dir):
            continue
        for fname in sorted(os.listdir(scan_dir)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(scan_dir, fname)
            is_legacy_fallback = (
                _using_canonical_module_dir() and _path_is_within(path, _LEGACY_MODULES_DIR)
            )
            if is_legacy_fallback:
                legacy_ids = _selectable_module_ids(path)
                if not legacy_ids or set(legacy_ids) <= seen_ids:
                    continue
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else [data]
                for m in items:
                    if not (isinstance(m, dict) and m.get("id") and m.get("instructions")):
                        continue
                    if not _is_selectable_module(m):
                        continue  # 协议、迁移别名与参考资料均不列为可选模块
                    if m["id"] in seen_ids:
                        LOGGER.warning("MiniMax H3 PromptModuleLoader: 重复模块 id=%s（%s 已跳过）", m["id"], fname)
                        continue
                    seen_ids.add(m["id"])
                    item = dict(m)
                    item["_file"] = fname
                    item["_origin"] = "builtin" if _path_is_within(path, _MODULES_DIR) else "legacy"
                    item["_metadata_issues"] = _metadata_issues(item)
                    modules.append(item)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("MiniMax H3 PromptModuleLoader: 模块 %s 加载失败: %s", fname, exc)
    return modules


def _module_choices() -> list[str]:
    return [_PROMPT_MODULE_NONE] + [m["title_zh"] for m in _load_modules()]


def _modules_from_rule_pack(rule_pack) -> tuple[list[dict], list[str]]:
    """验证直载节点输出的运行时规则包；坏条目只记诊断，不中断工作流。"""
    if rule_pack in (None, ""):
        return [], []
    if not isinstance(rule_pack, dict):
        return [], ["外部规则包格式无效：应为对象"]
    raw_modules = rule_pack.get("modules")
    if not isinstance(raw_modules, list):
        return [], ["外部规则包格式无效：缺少 modules 数组"]
    modules: list[dict] = []
    issues: list[str] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_modules, start=1):
        if not (isinstance(raw, dict) and raw.get("id") and raw.get("instructions")):
            issues.append(f"外部规则包第 {index} 项字段不完整，已跳过")
            continue
        module_id = str(raw["id"])
        if module_id in _PROTOCOL_IDS or module_id in _LEGACY_PROTOCOL_IDS:
            issues.append(f"协议模块禁止由外部规则包注入，已跳过：{module_id}")
            continue
        if not _is_selectable_module(raw):
            issues.append(f"参考资料模块不可由外部规则包注入，已跳过：{module_id}")
            continue
        if module_id in seen_ids:
            issues.append(f"外部规则包重复模块 id 已跳过：{module_id}")
            continue
        seen_ids.add(module_id)
        module = dict(raw)
        module["title_zh"] = str(module.get("title_zh") or module_id)
        module["_origin"] = "file_loader"
        module["_metadata_issues"] = _metadata_issues(module)
        modules.append(module)
    return modules, issues


class MiniMaxH3PromptModuleLoader:
    """智能规则组合：内置槽位与外部规则包统一经过确定性 Resolver。"""

    @classmethod
    def INPUT_TYPES(cls):
        # 同一份节点定义只扫描一次磁盘；5 个槽位各拿列表副本，避免前端刷新时重复读 5 轮 JSON。
        module_choices = _module_choices()
        return {
            "required": {
                "scope": (_PROMPT_MODULE_SCOPES, {"default": "全部"}),
                "module_1": (list(module_choices), {"default": _PROMPT_MODULE_NONE}),
                "module_2": (list(module_choices), {"default": _PROMPT_MODULE_NONE}),
                "module_3": (list(module_choices), {"default": _PROMPT_MODULE_NONE}),
                "custom_instructions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "可选：只在当前工作流追加规则（如固定镜头/风格/负面约束）",
                }),
                # 新 widget 一律追加在末尾（ComfyUI 按位置恢复旧工作流值）
                "module_4": (list(module_choices), {"default": _PROMPT_MODULE_NONE}),
                "module_5": (list(module_choices), {"default": _PROMPT_MODULE_NONE}),
                "module_folder": (_folder_choices(), {"default": _FOLDER_NONE}),
                "render_profile": (_RENDER_PROFILES, {"default": "standard"}),
            },
            "optional": {
                "external_rule_pack": ("H3_PROMPT_RULE_PACK", {"forceInput": True}),
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
             custom_instructions="", module_folder=_FOLDER_NONE, render_profile="standard",
             external_rule_pack=None):
        render_profile = _normalize_render_profile(render_profile)
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

        external_modules, external_issues = _modules_from_rule_pack(external_rule_pack)
        issues.extend(external_issues)
        external_added = 0
        for module in external_modules:
            if module["id"] in seen_ids:
                issues.append(f"外部规则与已选规则 id 重复，已忽略：{module['id']}")
                continue
            seen_ids.add(module["id"])
            selected.append(module)
            external_added += 1

        for m in selected:
            issues.extend(f"{m['id']}: {item}" for item in m.get("_metadata_issues", []))
        resolved, suppressed, resolution_issues, resolution_trace = _resolve_modules(selected, scope)
        issues.extend(resolution_issues)

        sections = []
        rendered_modules = []
        for m in resolved:
            evidence = str(m.get("evidence_level") or "experimental")
            rendered_text, render_source = _render_module_instructions(m, render_profile)
            sections.append(
                f"[{m['id']} | {m['title_zh']} | v{m.get('version', 1)} | "
                f"{_EVIDENCE_LABELS.get(evidence, '未分级')}]\n{rendered_text}"
            )
            rendered_modules.append({
                "id": str(m["id"]),
                "version": m.get("version", 1),
                "render_source": render_source,
                "rendered_chars": len(rendered_text),
                "rendered_sha256": _sha256_text(rendered_text),
            })
            if render_source == "fallback_instructions":
                issues.append(
                    f"{m['id']}: 缺少可渲染 semantic_contract，{render_profile} 档已回退当前正文"
                )
        custom = (custom_instructions or "").strip()
        if custom:
            sections.append("[workflow_custom]\n" + custom)
        merged = "\n\n".join(sections)
        if len(merged) > _PROMPT_MODULE_MAX_CHARS:
            merged = merged[:_PROMPT_MODULE_MAX_CHARS] + "\n（合并文本过长已截断）"
            issues.append("合并文本超过 6000 字符，末尾已截断")

        preview_lines = [
            f"[已应用 {i + 1}] {m['title_zh']}（{_EVIDENCE_LABELS.get(m.get('evidence_level'), '未分级')}，"
            f"scope={m.get('scope') or '全部'}）"
            for i, m in enumerate(resolved)
        ]
        for item in resolution_trace:
            if item["status"] == "suppressed":
                preview_lines.append(f"[未应用] {item['reason']}")
        if custom:
            preview_lines.append("[自定义] workflow_custom（当前工作流规则，未做证据分级）")
        diag_lines = [
            f"模块文件夹={module_folder}，可选模块={len(modules)}，"
            f"已选={len(selected)}（外部规则={external_added}），已应用={len(resolved)}，"
            f"未应用={len(suppressed)}，渲染档={_RENDER_PROFILE_LABELS[render_profile]}，"
            f"合并字符={len(merged)}"
        ]
        if not modules and module_folder != _FOLDER_NONE:
            diag_lines.append("该文件夹无模块（下拉不随文件夹联动，可在模块槽手输该文件夹内模块的 title_zh）")
        diag_lines.extend(f"[提示] {item}" for item in dict.fromkeys(issues))
        module_set_sha256 = _json_sha256(rendered_modules)
        manifest = {
            "schema_version": "h3_prompt_modules/1.2",
            "scope": scope,
            "render_profile": render_profile,
            "render_profile_label": _RENDER_PROFILE_LABELS[render_profile],
            "module_set_sha256": module_set_sha256,
            "rendered_system_sha256": _sha256_text(merged),
            "rendered_modules": rendered_modules,
            "selected": [_manifest_module(m) for m in selected],
            "resolved": [_manifest_module(m) for m in resolved],
            "suppressed": [item for item in resolution_trace if item["status"] == "suppressed"],
            "resolution_trace": resolution_trace,
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
        # 空 folder = 正式规则库根目录（与下拉空选项一致）
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
    """轻量文件直载器：原样合并规则，并可把结构化规则包交给智能组合。"""

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

    # 前两个输出保持原顺序，旧工作流连线不会错位；规则包只在末尾追加。
    RETURN_TYPES = ("STRING", "STRING", "H3_PROMPT_RULE_PACK")
    RETURN_NAMES = ("system_prompt_module", "diagnostics", "rule_pack")
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
        merged, issues, count, selected_count, rule_pack = _load_module_files(selections)
        diag = (
            f"直载模式（不执行作用域/依赖/冲突消解）；已选文件={selected_count}，"
            f"有效模块={count}，合并字符={len(merged)}"
        )
        if issues:
            diag += "；" + "；".join(issues[:3])
        return merged, diag, rule_pack


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PromptModuleLoader": MiniMaxH3PromptModuleLoader,
    "MiniMaxH3ModuleFolderLoader": MiniMaxH3ModuleFolderLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PromptModuleLoader": "MiniMax H3 创作规则组合 (热加载, 实验)",
    "MiniMaxH3ModuleFolderLoader": "MiniMax H3 创作规则文件直载 (轻量, 热加载)",
}
