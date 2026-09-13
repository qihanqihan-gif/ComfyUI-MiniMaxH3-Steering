# -*- coding: utf-8 -*-
"""h3_compiler.py — MiniMax H3 提示词确定性编译与校验（纯 Python，零 LLM）。

v0.2 范围：
- validate_prompt：对最终提示词做确定性检查（时间码/标签/说话人/对白闭合/字段顺序）
- serialize_three_part：三段式序列化（字段前缀 + 镜头段，供后续 Planner 使用）

设计原则（GPT 5.6 第二轮结论）：编号、时间码、字段顺序、对白闭合这类
“程序能 100% 检查的东西”不应交给 LLM 保证，由本模块确定性处理。
"""
import json
import re

# ---------------------------------------------------------------------------
# 时间码
# ---------------------------------------------------------------------------

_TIME_RE = re.compile(r"(\d{2}):(\d{2})\.(\d{3})")

# [Shot N] 后紧跟 (At MM:SS.mmm) 的镜头行；首镜允许无时间戳
_SHOT_RE = re.compile(r"\[Shot\s+(\d+)\]\s*(?:At\s+(\d{2}):(\d{2})\.(\d{3}))?")
# 自由文本里的裸时间码（用于检测无镜头标记的时间戳）
_BARE_TIME_RE = re.compile(r"\bAt\s+(\d{2}):(\d{2})\.(\d{3})")


def parse_timecode(mm_ss_mmm: str) -> float:
    """'MM:SS.mmm' → 秒数（float）。格式错误抛 ValueError。"""
    m = _TIME_RE.fullmatch((mm_ss_mmm or "").strip())
    if not m:
        raise ValueError(f"时间码格式应为 MM:SS.mmm，收到 {mm_ss_mmm!r}")
    mm, ss, mmm = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if mm < 0 or ss < 0 or ss >= 60:
        raise ValueError(f"非法时间码 {mm_ss_mmm!r}（分/秒越界）")
    return mm * 60.0 + ss + mmm / 1000.0


def format_timecode(seconds: float) -> str:
    """秒数 → 'MM:SS.mmm'（两位分/秒 + 三位毫秒）。"""
    seconds = max(0.0, float(seconds))
    mm = int(seconds // 60)
    ss = int(seconds % 60)
    mmm = int(round((seconds - int(seconds)) * 1000.0))
    if mmm >= 1000:
        mmm -= 1000
        ss += 1
    if ss >= 60:
        ss -= 60
        mm += 1
    return f"{mm:02d}:{ss:02d}.{mmm:03d}"


# ---------------------------------------------------------------------------
# 标签 / 说话人 / 对白
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(r"<(Picture|Subject|Video|Audio)\s+(\d+)>", re.IGNORECASE)
_SPEAKER_RE = re.compile(r"\(S(\d+)\)")
_DIALOG_OPEN = re.compile(r"<d>", re.IGNORECASE)
_DIALOG_CLOSE = re.compile(r"</d>", re.IGNORECASE)

# 三字段/六字段（按官方字段名，行首冒号前缀）
_THREE_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
_SIX_FIELDS = ("subject_definitions", "summary", "retention_analysis",
               "detailed_description", "overall_soundscape", "non_diegetic_music")

_BASE_TASKS = ("T2VA", "I2VA", "FL2VA", "L2VA")
_TASKS = _BASE_TASKS + ("Ref2VA",)


def normalize_task_type(task_type: str | None, default: str = "T2VA") -> str:
    """把用户/API 输入规范成官方五种任务名。"""
    raw = str(task_type or default).strip().upper()
    mapping = {name.upper(): name for name in _TASKS}
    if raw not in mapping:
        raise ValueError(f"不支持的 H3 task_type={task_type!r}，可选：{', '.join(_TASKS)}")
    return mapping[raw]


def build_alignment_line(task_type: str, duration: float, final_shot_number: int = 1) -> str:
    """按官方 Base guide 生成首帧/首尾帧/尾帧对齐首行。"""
    task = normalize_task_type(task_type)
    final_shot = max(1, int(final_shot_number or 1))
    seconds = f"{max(0.0, float(duration)):.2f}"
    if task == "T2VA" or task == "Ref2VA":
        return ""
    if task == "I2VA":
        return (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
    if task == "FL2VA":
        return (
            "How the reference pictures align with the target video — "
            "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot {final_shot}) aligns with the {seconds}-second mark of the target video."
        )
    return (
        "How the reference pictures align with the target video — "
        f"<Picture 1> (from [Shot {final_shot}]) aligns with the {seconds}-second mark of the target video."
    )


def _strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl >= 0:
            text = text[first_nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _as_ir_object(value) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        raise ValueError("prompt_ir 必须是 JSON 对象或 JSON 字符串")
    parsed = json.loads(_strip_json_fence(value))
    if not isinstance(parsed, dict):
        raise ValueError("prompt_ir 顶层必须是 JSON 对象")
    return parsed


def _legacy_shots_to_description(shots) -> str:
    """兼容 alpha 的 shot_breakdown，把它确定性转成官方 [Shot N] 文本。"""
    if not isinstance(shots, list):
        return ""
    out = []
    for index, shot in enumerate(shots, 1):
        if not isinstance(shot, dict):
            continue
        description = str(shot.get("description") or "").strip()
        if not description:
            continue
        if description.startswith("[Shot "):
            out.append(description)
            continue
        if index == 1:
            out.append(f"[Shot 1] {description}")
        else:
            start = shot.get("start_s")
            if start is None:
                out.append(f"[Shot {index}] {description}")
            else:
                out.append(f"[Shot {index}] At {format_timecode(float(start))}, {description}")
    return " ".join(out)


def _max_shot_number(text: str) -> int:
    nums = [int(n) for n in re.findall(r"\[Shot\s+(\d+)\]", text or "", re.IGNORECASE)]
    return max(nums, default=1)


def normalize_prompt_ir(prompt_ir, task_type: str | None = None,
                        duration: float | None = None) -> dict:
    """接收 LLM JSON/Canonical IR，生成稳定的 H3 Prompt IR。"""
    source = _as_ir_object(prompt_ir)
    task = normalize_task_type(task_type or source.get("task_type") or "T2VA")
    resolved_duration = float(duration if duration is not None else source.get("duration_seconds", 5.0))
    if not 0.0 < resolved_duration <= 60.0:
        raise ValueError(f"duration_seconds 必须大于 0 且不超过 60，收到 {resolved_duration}")

    normalized = {
        "schema_version": "h3_prompt_ir/1.0",
        "task_type": task,
        "duration_seconds": resolved_duration,
    }
    if task == "Ref2VA":
        for field in _SIX_FIELDS:
            normalized[field] = str(source.get(field) or "").strip()
    else:
        integrated = str(source.get("integrated_multimodal_description") or "").strip()
        if not integrated:
            integrated = _legacy_shots_to_description(source.get("shot_breakdown"))
        normalized.update({
            "integrated_multimodal_description": integrated,
            "overall_soundscape": str(source.get("overall_soundscape") or "").strip(),
            "non_diegetic_music": str(source.get("non_diegetic_music") or "").strip(),
        })
        final_shot = source.get("final_shot_number") or _max_shot_number(integrated)
        normalized["final_shot_number"] = max(1, int(final_shot))
        normalized["alignment_line"] = str(source.get("alignment_line") or "").strip() or build_alignment_line(
            task, resolved_duration, normalized["final_shot_number"]
        )
    return normalized


def _safe_normalize_ir_structure(normalized: dict,
                                 media_inventory: dict[str, int] | None = None
                                 ) -> tuple[dict, list[str]]:
    """只修复不需要猜测语义的 IR 编号问题。"""
    result = dict(normalized)
    notes: list[str] = []
    task = result["task_type"]
    shot_field = "detailed_description" if task == "Ref2VA" else "integrated_multimodal_description"
    shot_text = str(result.get(shot_field) or "")
    shot_numbers = [int(value) for value in re.findall(
        r"\[Shot\s+(\d+)\]", shot_text, re.IGNORECASE,
    )]
    if (
        shot_numbers
        and len(set(shot_numbers)) == len(shot_numbers)
        and shot_numbers == list(range(shot_numbers[0], shot_numbers[0] + len(shot_numbers)))
        and shot_numbers[0] != 1
    ):
        offset = shot_numbers[0] - 1

        def replace_shot(match):
            return f"[Shot {int(match.group(1)) - offset}]"

        result[shot_field] = re.sub(
            r"\[Shot\s+(\d+)\]", replace_shot, shot_text, flags=re.IGNORECASE,
        )
        notes.append(
            f"镜头标签按出现顺序从 {shot_numbers[0]}..{shot_numbers[-1]} 规范为 1..{len(shot_numbers)}"
        )
        if task != "Ref2VA":
            result["final_shot_number"] = len(shot_numbers)
            result["alignment_line"] = build_alignment_line(
                task, result["duration_seconds"], len(shot_numbers),
            )

    inventory = {
        str(kind).casefold(): max(0, int(count))
        for kind, count in (media_inventory or {}).items()
    }
    if inventory.get("picture") == 1:
        content_fields = _SIX_FIELDS if task == "Ref2VA" else _THREE_FIELDS
        picture_numbers = sorted({
            int(number)
            for field in content_fields
            for value in (str(result.get(field) or ""),)
            for kind, number in _LABEL_RE.findall(value)
            if kind.casefold() == "picture"
        })
        if len(picture_numbers) == 1 and picture_numbers[0] != 1:
            old_number = picture_numbers[0]

            def replace_picture(match):
                if match.group(1).casefold() == "picture" and int(match.group(2)) == old_number:
                    return "<Picture 1>"
                return match.group(0)

            for field, value in list(result.items()):
                if isinstance(value, str):
                    result[field] = _LABEL_RE.sub(replace_picture, value)
            notes.append(
                f"本次仅有一个 Picture 素材，已把唯一的 <Picture {old_number}> 规范为 <Picture 1>"
            )
    return result, notes


def compile_prompt_ir(prompt_ir, task_type: str | None = None,
                      duration: float | None = None,
                      media_inventory: dict[str, int] | None = None) -> dict:
    """IR -> 官方 H3 Prompt，并立即执行确定性校验。"""
    normalized = normalize_prompt_ir(prompt_ir, task_type=task_type, duration=duration)
    normalized, normalizations = _safe_normalize_ir_structure(
        normalized, media_inventory=media_inventory,
    )
    task = normalized["task_type"]
    if task == "Ref2VA":
        prompt = serialize_six_part(
            normalized["subject_definitions"], normalized["summary"],
            normalized["retention_analysis"], normalized["detailed_description"],
            normalized["overall_soundscape"], normalized["non_diegetic_music"],
        )
        mode = "ref"
    else:
        prompt = serialize_three_part(
            normalized["integrated_multimodal_description"],
            normalized["overall_soundscape"], normalized["non_diegetic_music"],
            alignment_line=normalized["alignment_line"],
        )
        mode = "base"
    validation = validate_prompt(
        prompt, duration=normalized["duration_seconds"], mode=mode, check_fields=True,
        task_type=task, media_inventory=media_inventory,
    )
    return {
        "prompt": prompt,
        "ir": normalized,
        "validation": validation,
        "normalizations": normalizations,
    }


def _field_order(text: str, fields) -> list[str]:
    """按文本出现顺序收集字段（行首 `field:` 前缀，忽略大小写）。"""
    found = []
    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        for f in fields:
            if stripped.casefold().startswith(f.casefold() + ":"):
                found.append(f)
                break
    return found


def _field_value(text: str, field: str, fields) -> str:
    """提取一个字段直到下一个协议字段之间的完整内容。"""
    start_re = re.compile(rf"(?im)^\s*{re.escape(field)}\s*:\s*")
    match = start_re.search(text or "")
    if not match:
        return ""
    next_re = re.compile(
        r"(?im)^\s*(?:" + "|".join(re.escape(name) for name in fields) + r")\s*:\s*"
    )
    next_match = next_re.search(text, match.end())
    end = next_match.start() if next_match else len(text)
    return text[match.end():end].strip()


def _consecutive_numbers(nums: list[int]) -> list[str]:
    """编号连续性检查：返回缺失/跳号说明（空 = 通过）。"""
    issues = []
    seen = sorted(set(nums))
    if not seen:
        return issues
    if seen[0] != 1:
        issues.append(f"编号从 {seen[0]} 开始，应为 1")
    for a, b in zip(seen, seen[1:]):
        if b != a + 1:
            issues.append(f"编号跳号：{a} → {b}")
    return issues


def validate_prompt(text: str, duration: float | None = None, mode: str | None = None,
                    check_fields: bool = True, task_type: str | None = None,
                    media_inventory: dict[str, int] | None = None) -> dict:
    """确定性校验最终提示词。

    返回 {"errors": [...], "warnings": [...], "checks": {...}}。
    errors 为硬错误（应修复）；warnings 为软提示。
    mode: "base"（三段式）/ "ref"（六段式）/ None（自动探测）。
    check_fields: False 时跳过字段名检查（用于中文自然语言输出——
    字段名是英文协议的一部分，中文输出不适用，但时间码/标签/对白检查保留）。
    """
    errors: list[str] = []
    warnings: list[str] = []
    text = text or ""

    # 1) 字段顺序（自动探测后必须重新按目标字段取顺序）
    if check_fields:
        if mode not in (None, "base", "ref"):
            raise ValueError(f"mode 必须是 base/ref/None，收到 {mode!r}")
        if mode is None:
            ref_markers = _field_order(text, _SIX_FIELDS[:4])
            mode = "ref" if ref_markers else "base"
        fields = _SIX_FIELDS if mode == "ref" else _THREE_FIELDS
        order = _field_order(text, fields)
        missing = [f for f in fields if f not in order]
        for f in missing:
            errors.append(f"缺少字段 `{f}:`")
        duplicates = sorted({f for f in order if order.count(f) > 1})
        for f in duplicates:
            errors.append(f"字段 `{f}:` 出现多次")
        unique_order = list(dict.fromkeys(order))
        wanted = [f for f in fields if f in unique_order]
        if unique_order and unique_order != wanted:
            errors.append(f"字段顺序错误：实际 {unique_order}，应为 {wanted}")
        for f in fields:
            if f in order and not _field_value(text, f, fields):
                errors.append(f"字段 `{f}:` 内容为空")
    else:
        mode = mode or ("ref" if text.casefold().count("subject_definitions:") else "base")

    # 2) 镜头时间码：首镜无时间戳允许；后续严格递增；不超总时长
    # 注意：Python re.findall 对未参与匹配的可选组返回 ''（非 None），故用 truthy 判断
    shot_field = "detailed_description" if mode == "ref" else "integrated_multimodal_description"
    shot_text = _field_value(text, shot_field, _SIX_FIELDS if mode == "ref" else _THREE_FIELDS) or text
    shots = [(int(n), parse_timecode(f"{m}:{s}.{ms}") if m else None)
             for n, m, s, ms in _SHOT_RE.findall(shot_text)]
    if not shots and shot_text.strip():
        warnings.append(f"`{shot_field}` 缺少 [Shot 1]，镜头边界与时间线无法确定性校验")
    if shots:
        last_t = None
        for n, t in shots:
            if t is None:
                if n != 1:
                    warnings.append(f"[Shot {n}] 缺少时间戳（首镜外建议 At MM:SS.mmm）")
                continue
            if n == 1:
                warnings.append("[Shot 1] 不应带时间戳（官方格式从画面起点直接描述首镜）")
            if last_t is not None and t <= last_t:
                errors.append(f"[Shot {n}] 时间码 {format_timecode(t)} 未严格递增（前一镜 {format_timecode(last_t)}）")
            if duration is not None and t > duration + 0.001:
                errors.append(f"[Shot {n}] 时间码 {format_timecode(t)} 超过总时长 {duration}s")
            last_t = t
        shot_numbers = [n for n, _ in shots]
        if shot_numbers != list(range(1, len(shot_numbers) + 1)):
            errors.append(f"镜头编号应按出现顺序从 1 连续递增，实际 {shot_numbers}")
        # 裸时间码（无 [Shot N] 前缀）
        bare = [parse_timecode(f"{m}:{s}.{ms}") for m, s, ms in _BARE_TIME_RE.findall(shot_text)]
        if bare and len(bare) > len(shots):
            warnings.append("存在无 [Shot N] 前缀的裸时间码，建议统一镜头标记")

    # 3) 引用标签编号连续（各类型独立编号空间）
    labels: dict[str, list[int]] = {}
    for kind, num in _LABEL_RE.findall(text):
        labels.setdefault(kind.casefold(), []).append(int(num))
    for kind, nums in labels.items():
        for issue in _consecutive_numbers(nums):
            errors.append(f"<{kind.capitalize()}> {issue}")

    if task_type is not None:
        task = normalize_task_type(task_type)
        expected_pictures = {"T2VA": 0, "I2VA": 1, "FL2VA": 2, "L2VA": 1}.get(task)
        if expected_pictures is not None:
            picture_nums = sorted(set(labels.get("picture", [])))
            overflow = [num for num in picture_nums if num > expected_pictures]
            if overflow:
                errors.append(
                    f"{task} 官方 ImageToVideo 最多提供 {expected_pictures} 个 Picture 锚点，"
                    f"提示词却使用了 {overflow}；多参考素材应改用 Ref2VA"
                )

    # 导演调用可提供本次已连接/由用户明确声明的媒体数量；通用编译节点未知时跳过。
    if media_inventory is not None:
        inventory = {
            str(kind).casefold(): max(0, int(count))
            for kind, count in media_inventory.items()
        }
        for kind in ("picture", "video", "audio"):
            if kind not in inventory:
                continue
            allowed = inventory[kind]
            used = sorted(set(labels.get(kind, [])))
            overflow = [number for number in used if number > allowed]
            if overflow:
                display = kind.capitalize()
                available = f"1..{allowed}" if allowed else "无"
                errors.append(
                    f"<{display}> 使用了本次媒体清单外的编号 {overflow}；"
                    f"可用 {display} 编号为 {available}"
                )

    # Ref2VA：所有正文引用都应先在 subject_definitions 中定义。
    if mode == "ref":
        defs_text = _field_value(text, "subject_definitions", _SIX_FIELDS)
        defined = {(kind.casefold(), int(num)) for kind, num in _LABEL_RE.findall(defs_text)}
        used_parts = [_field_value(text, field, _SIX_FIELDS) for field in _SIX_FIELDS[1:]]
        used = {(kind.casefold(), int(num)) for kind, num in _LABEL_RE.findall("\n".join(used_parts))}
        for kind, num in sorted(used - defined):
            errors.append(f"正文使用了未在 subject_definitions 定义的 <{kind.capitalize()} {num}>")
        for kind, num in sorted(defined - used):
            warnings.append(f"subject_definitions 定义了未在后续字段使用的 <{kind.capitalize()} {num}>")

        retention = _field_value(text, "retention_analysis", _SIX_FIELDS)
        if _SPEAKER_RE.search(retention):
            warnings.append("retention_analysis 不应包含 (Sx)；说话人 ID 只属于实际发声描述")

        summary = _field_value(text, "summary", _SIX_FIELDS).lstrip()
        if summary and not summary.startswith("["):
            warnings.append("summary 建议以官方方括号任务类型开头，例如 [reference generation]")

        detailed = _field_value(text, "detailed_description", _SIX_FIELDS)
        first_shot = re.search(r"\[Shot\s+1\]", detailed, re.IGNORECASE)
        if first_shot and not detailed[:first_shot.start()].strip():
            warnings.append("detailed_description 建议在 [Shot 1] 前写 1–2 句全片视觉风格")

    # 4) 说话人 ID 连续
    speakers = [int(n) for n in _SPEAKER_RE.findall(text)]
    if speakers:
        for issue in _consecutive_numbers(speakers):
            warnings.append(f"(Sx) {issue}")

    # 5) <d> 对白闭合
    opens = len(_DIALOG_OPEN.findall(text))
    closes = len(_DIALOG_CLOSE.findall(text))
    if opens != closes:
        errors.append(f"<d> 对白标记未闭合：{opens} 开 vs {closes} 闭")

    # 6) non_diegetic_music: N/A 与静音一致性（软提示）
    na_music = bool(re.search(r"non_diegetic_music\s*:\s*N/?A", text, re.IGNORECASE))
    if na_music and re.search(r"\b(配乐|music|melody|orchestra)\b", text, re.IGNORECASE):
        warnings.append("non_diegetic_music 标为 N/A，但文本其他位置出现音乐相关词")

    # 7) 明显超长/超短（软提示，与 GPT 5.6「长度≠质量」结论一致）
    n_chars = len(text)
    if n_chars > 6000:
        warnings.append(f"提示词 {n_chars} 字符，超过 6000——社区实证长文可能稀释控制（按需裁剪）")
    elif 0 < n_chars < 80:
        warnings.append(f"提示词仅 {n_chars} 字符，可能过于简略（结构信息不足）")

    return {
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "mode": mode,
            "char_count": n_chars,
            "shot_count": len(shots),
            "labels": {k: len(v) for k, v in labels.items()},
            "dialogue_pairs": opens,
        },
    }


# ---------------------------------------------------------------------------
# 序列化（三段式 / 六段式骨架）
# ---------------------------------------------------------------------------

def serialize_three_part(integrated: str, soundscape: str, music: str,
                         alignment_line: str | None = None) -> str:
    """三段式序列化：可选对齐指令首行 + 三个字段。

    alignment_line：I2VA/FL2VA/L2VA 的首行对齐指令（如 I2VA 的
    'For the target video, at 0.00 seconds into the target video, '
    '<Picture 1> (from [Shot 1]) is fully referenced.'）；T2VA 传 None。
    """
    parts = []
    if alignment_line:
        parts.append(alignment_line.strip())
    body = []
    body.append("integrated_multimodal_description: " + (integrated or "").strip())
    body.append("overall_soundscape: " + (soundscape or "N/A").strip())
    body.append("non_diegetic_music: " + (music or "N/A").strip())
    parts.append("\n\n".join(body))
    return "\n\n".join(parts)


def serialize_six_part(subject_definitions: str, summary: str, retention_analysis: str,
                       detailed_description: str, soundscape: str, music: str) -> str:
    """六段式序列化（Ref2VA）：固定顺序，字段名严格。"""
    sections = [
        ("subject_definitions", subject_definitions),
        ("summary", summary),
        ("retention_analysis", retention_analysis),
        ("detailed_description", detailed_description),
        ("overall_soundscape", soundscape),
        ("non_diegetic_music", music),
    ]
    out = []
    for name, value in sections:
        value = (value or "").strip()
        if name in ("overall_soundscape", "non_diegetic_music") and not value:
            value = "N/A"
        out.append(f"{name}: {value}")
    return "\n\n".join(out)


def format_validation_report(result: dict) -> str:
    """把 validate_prompt 结果整理成适合 ComfyUI 节点查看的短报告。"""
    errors = list(result.get("errors") or [])
    warnings = list(result.get("warnings") or [])
    checks = dict(result.get("checks") or {})
    lines = [
        "状态：" + ("通过" if not errors else "存在错误"),
        f"模式={checks.get('mode', 'unknown')} 镜头={checks.get('shot_count', 0)} 字符={checks.get('char_count', 0)}",
    ]
    lines.extend(f"[错误] {item}" for item in errors)
    lines.extend(f"[警告] {item}" for item in warnings)
    return "\n".join(lines)


class MiniMaxH3CompileValidate:
    """编译 H3 Prompt IR；若输入不是 JSON，则只校验现有提示词。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_or_ir": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "粘贴 H3 Prompt IR JSON，或直接粘贴已有 H3 提示词进行校验",
                }),
                "task_type": (["AUTO", *_TASKS], {"default": "AUTO"}),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 60.0, "step": 0.1}),
                "fail_on_error": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("final_prompt", "validation_report", "normalized_ir", "is_valid")
    FUNCTION = "compile_or_validate"
    CATEGORY = "MiniMax H3 Lab/Prompt"

    def compile_or_validate(self, prompt_or_ir, task_type="AUTO", duration_seconds=5.0,
                            fail_on_error=False):
        text = str(prompt_or_ir or "").strip()
        if not text:
            raise ValueError("prompt_or_ir 不能为空")
        explicit_task = None if task_type == "AUTO" else task_type
        try:
            obj = _as_ir_object(text)
        except (json.JSONDecodeError, ValueError):
            mode = None if explicit_task is None else ("ref" if explicit_task == "Ref2VA" else "base")
            validation = validate_prompt(
                text, duration=duration_seconds, mode=mode, check_fields=True,
                task_type=explicit_task,
            )
            result = {"prompt": text, "ir": {}, "validation": validation}
        else:
            result = compile_prompt_ir(obj, task_type=explicit_task, duration=duration_seconds)

        report = format_validation_report(result["validation"])
        valid = not result["validation"]["errors"]
        if fail_on_error and not valid:
            raise ValueError(report)
        return (
            result["prompt"], report,
            json.dumps(result["ir"], ensure_ascii=False, indent=2) if result["ir"] else "{}",
            valid,
        )


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3CompileValidate": MiniMaxH3CompileValidate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3CompileValidate": "MiniMax H3 编译与校验",
}
