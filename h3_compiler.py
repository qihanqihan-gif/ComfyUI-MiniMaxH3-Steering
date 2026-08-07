# -*- coding: utf-8 -*-
"""h3_compiler.py — MiniMax H3 提示词确定性编译与校验（纯 Python，零 LLM）。

v0.2 范围：
- validate_prompt：对最终提示词做确定性检查（时间码/标签/说话人/对白闭合/字段顺序）
- serialize_three_part：三段式序列化（字段前缀 + 镜头段，供后续 Planner 使用）

设计原则（GPT 5.6 第二轮结论）：编号、时间码、字段顺序、对白闭合这类
“程序能 100% 检查的东西”不应交给 LLM 保证，由本模块确定性处理。
"""
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
                    check_fields: bool = True) -> dict:
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

    # 1) 字段顺序（自动探测模式）
    if check_fields:
        fields = _SIX_FIELDS if mode == "ref" else (
            _THREE_FIELDS if mode == "base" else None)
        if fields:
            order = _field_order(text, fields)
            missing = [f for f in fields if f not in order]
            for f in missing:
                errors.append(f"缺少字段 `{f}:`")
            if order:
                wanted = [f for f in fields if f in order]
                if order != wanted:
                    errors.append(f"字段顺序错误：实际 {order}，应为 {wanted}")
        else:
            # 自动：六字段全在 → ref；否则按三字段
            order = _field_order(text, _SIX_FIELDS)
            mode = "ref" if len(order) >= 4 else "base"
            if mode == "ref":
                for f in _SIX_FIELDS:
                    if f not in order:
                        errors.append(f"缺少字段 `{f}:`（六段式）")
            else:
                for f in _THREE_FIELDS:
                    if f not in order:
                        errors.append(f"缺少字段 `{f}:`（三段式）")
    else:
        mode = mode or ("ref" if text.casefold().count("subject_definitions:") else "base")

    # 2) 镜头时间码：首镜无时间戳允许；后续严格递增；不超总时长
    # 注意：Python re.findall 对未参与匹配的可选组返回 ''（非 None），故用 truthy 判断
    shots = [(int(n), parse_timecode(f"{m}:{s}.{ms}") if m else None)
             for n, m, s, ms in _SHOT_RE.findall(text)]
    if shots:
        last_t = None
        for n, t in shots:
            if t is None:
                if n != 1 and last_t is not None:
                    warnings.append(f"[Shot {n}] 缺少时间戳（首镜外建议 At MM:SS.mmm）")
                continue
            if last_t is not None and t <= last_t:
                errors.append(f"[Shot {n}] 时间码 {format_timecode(t)} 未严格递增（前一镜 {format_timecode(last_t)}）")
            if duration is not None and t > duration + 0.001:
                errors.append(f"[Shot {n}] 时间码 {format_timecode(t)} 超过总时长 {duration}s")
            last_t = t
        # 裸时间码（无 [Shot N] 前缀）
        bare = [parse_timecode(f"{m}:{s}.{ms}") for m, s, ms in _BARE_TIME_RE.findall(text)]
        if bare and len(bare) > len(shots):
            warnings.append("存在无 [Shot N] 前缀的裸时间码，建议统一镜头标记")

    # 3) 引用标签编号连续（各类型独立编号空间）
    labels: dict[str, list[int]] = {}
    for kind, num in _LABEL_RE.findall(text):
        labels.setdefault(kind.casefold(), []).append(int(num))
    for kind, nums in labels.items():
        for issue in _consecutive_numbers(nums):
            errors.append(f"<{kind.capitalize()}> {issue}")

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
    parts.append("\n".join(body))
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
        if not value:
            continue
        out.append(f"{name}: {value}")
    return "\n\n".join(out)
