# -*- coding: utf-8 -*-
"""h3_compiler 纯函数测试（确定性校验与序列化，零 LLM、零网络）。"""
import importlib.util
import os
import sys

LAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_mod():
    spec = importlib.util.spec_from_file_location("h3_compiler", os.path.join(LAB_ROOT, "h3_compiler.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h3_compiler"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 时间码
# ---------------------------------------------------------------------------

def test_parse_timecode_ok():
    mod = _load_mod()
    assert mod.parse_timecode("00:03.500") == 3.5
    assert mod.parse_timecode("01:02.250") == 62.25
    assert mod.parse_timecode("00:00.000") == 0.0


def test_parse_timecode_invalid():
    mod = _load_mod()
    for bad in ("3.5", "00:60.000", "aa:bb.ccc", ""):
        try:
            mod.parse_timecode(bad)
        except ValueError:
            continue
        raise AssertionError(f"应拒绝非法时间码 {bad!r}")


def test_format_timecode_roundtrip():
    mod = _load_mod()
    assert mod.format_timecode(3.5) == "00:03.500"
    assert mod.format_timecode(62.25) == "01:02.250"
    assert mod.format_timecode(0) == "00:00.000"


# ---------------------------------------------------------------------------
# 校验：字段
# ---------------------------------------------------------------------------

def test_validate_three_fields_ok():
    mod = _load_mod()
    text = (
        "integrated_multimodal_description: [Shot 1] A dog runs.\n"
        "overall_soundscape: footsteps.\n"
        "non_diegetic_music: N/A"
    )
    res = mod.validate_prompt(text, mode="base")
    assert res["errors"] == []


def test_validate_missing_field():
    mod = _load_mod()
    text = "integrated_multimodal_description: hi\noverall_soundscape: none"
    res = mod.validate_prompt(text, mode="base")
    assert any("non_diegetic_music" in e for e in res["errors"])


def test_validate_field_order():
    mod = _load_mod()
    text = (
        "non_diegetic_music: N/A\n"
        "integrated_multimodal_description: [Shot 1] A dog runs.\n"
        "overall_soundscape: bark."
    )
    res = mod.validate_prompt(text, mode="base")
    assert any("顺序" in e for e in res["errors"])


def test_validate_six_part_detected():
    mod = _load_mod()
    text = "\n".join([
        "subject_definitions: <Subject 1> is the woman.",
        "summary: [reference generation] A scene.",
        "retention_analysis: <Subject 1> (appears in [Shot 1]): fully_preserved.",
        "detailed_description: [Shot 1] A medium shot of <Subject 1>.",
        "overall_soundscape: room tone.",
        "non_diegetic_music: N/A",
    ])
    res = mod.validate_prompt(text)  # 自动探测 → ref
    assert res["checks"]["mode"] == "ref"
    assert res["errors"] == []


# ---------------------------------------------------------------------------
# 校验：镜头时间码
# ---------------------------------------------------------------------------

def test_validate_shot_timestamps_increasing():
    mod = _load_mod()
    text = (
        "integrated_multimodal_description: "
        "[Shot 1] Opening. [Shot 2] At 00:02.000, cut. [Shot 3] At 00:05.000, cut.\n"
        "overall_soundscape: N/A\nnon_diegetic_music: N/A"
    )
    res = mod.validate_prompt(text, mode="base", duration=10)
    assert res["errors"] == []


def test_validate_timestamp_not_increasing():
    mod = _load_mod()
    text = (
        "integrated_multimodal_description: "
        "[Shot 1] A. [Shot 2] At 00:05.000, B. [Shot 3] At 00:03.000, C.\n"
        "overall_soundscape: N/A\nnon_diegetic_music: N/A"
    )
    res = mod.validate_prompt(text, mode="base", duration=10)
    assert any("未严格递增" in e for e in res["errors"])


def test_validate_timestamp_exceeds_duration():
    mod = _load_mod()
    text = (
        "integrated_multimodal_description: "
        "[Shot 1] A. [Shot 2] At 00:08.000, B.\n"
        "overall_soundscape: N/A\nnon_diegetic_music: N/A"
    )
    res = mod.validate_prompt(text, mode="base", duration=5)
    assert any("超过总时长" in e for e in res["errors"])


def test_validate_first_shot_no_timestamp_allowed():
    mod = _load_mod()
    text = (
        "integrated_multimodal_description: "
        "[Shot 1] A. [Shot 2] At 00:02.000, B.\n"
        "overall_soundscape: N/A\nnon_diegetic_music: N/A"
    )
    res = mod.validate_prompt(text, mode="base")
    assert not any("[Shot 1]" in e for e in res["errors"])


# ---------------------------------------------------------------------------
# 校验：标签 / 说话人 / 对白
# ---------------------------------------------------------------------------

def test_validate_label_numbering():
    mod = _load_mod()
    text = (
        "integrated_multimodal_description: "
        "<Picture 2> appears. <Picture 1> appears first.\n"
        "overall_soundscape: N/A\nnon_diegetic_music: N/A"
    )
    res = mod.validate_prompt(text, mode="base")
    assert res["errors"] == []  # 编号 1、2 连续


def test_validate_label_skip():
    mod = _load_mod()
    text = (
        "integrated_multimodal_description: <Picture 3> only.\n"
        "overall_soundscape: N/A\nnon_diegetic_music: N/A"
    )
    res = mod.validate_prompt(text, mode="base")
    assert any("从 3 开始" in e for e in res["errors"]), "起始编号不是 1 应报错"


def test_validate_label_jump():
    mod = _load_mod()
    text = (
        "integrated_multimodal_description: "
        "<Picture 1> and <Picture 3>."
        "\noverall_soundscape: N/A\nnon_diegetic_music: N/A"
    )
    res = mod.validate_prompt(text, mode="base")
    assert any("跳号" in e for e in res["errors"]), "1→3 跳号应报错"


def test_validate_dialogue_closed():
    mod = _load_mod()
    text = (
        "integrated_multimodal_description: "
        '<Subject 1> (S1) says <d>[English] Hello.</d>\n'
        "overall_soundscape: N/A\nnon_diegetic_music: N/A"
    )
    res = mod.validate_prompt(text, mode="base")
    assert res["errors"] == []


def test_validate_dialogue_unclosed():
    mod = _load_mod()
    text = (
        "integrated_multimodal_description: "
        '<Subject 1> (S1) says <d>[English] Hello.\n'
        "overall_soundscape: N/A\nnon_diegetic_music: N/A"
    )
    res = mod.validate_prompt(text, mode="base")
    assert any("<d>" in e for e in res["errors"])


def test_validate_speaker_start_at_one():
    mod = _load_mod()
    text = (
        "integrated_multimodal_description: <Subject 1> (S2) speaks.\n"
        "overall_soundscape: N/A\nnon_diegetic_music: N/A"
    )
    res = mod.validate_prompt(text, mode="base")
    assert any("(Sx)" in w and "从 2 开始" in w for w in res["warnings"])


# ---------------------------------------------------------------------------
# 校验：软提示（长度 / 静音一致性）
# ---------------------------------------------------------------------------

def test_validate_length_warning():
    mod = _load_mod()
    long_text = (
        "integrated_multimodal_description: " + "word " * 1500 + "\n"
        "overall_soundscape: N/A\nnon_diegetic_music: N/A"
    )
    res = mod.validate_prompt(long_text, mode="base")
    assert any("超过 6000" in w for w in res["warnings"])


def test_validate_music_na_consistency():
    mod = _load_mod()
    text = (
        "integrated_multimodal_description: A piano melody plays.\n"
        "overall_soundscape: N/A\nnon_diegetic_music: N/A"
    )
    res = mod.validate_prompt(text, mode="base")
    assert any("音乐" in w for w in res["warnings"])


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------

def test_serialize_three_part_with_alignment():
    mod = _load_mod()
    out = mod.serialize_three_part(
        "A dog runs.",
        "footsteps",
        "N/A",
        alignment_line=(
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        ),
    )
    assert out.startswith("For the target video")
    assert "integrated_multimodal_description: A dog runs." in out
    assert "overall_soundscape: footsteps" in out
    assert "non_diegetic_music: N/A" in out
    # 空字段默认 N/A
    out2 = mod.serialize_three_part("desc", "", "")
    assert "overall_soundscape: N/A" in out2


def test_serialize_six_part_order():
    mod = _load_mod()
    out = mod.serialize_six_part("defs", "summary", "retention", "detail", "", "")
    idx = [out.index(name) for name in (
        "subject_definitions:", "summary:", "retention_analysis:",
        "detailed_description:", "overall_soundscape:", "non_diegetic_music:")]
    assert idx == sorted(idx)  # 顺序严格
    assert "overall_soundscape: N/A" in out  # 空音频字段默认 N/A
