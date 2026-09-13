import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / "example_workflows" / "MiniMax-H3-Lab-Ref2VA-研究模板.json"


def _load_workflow():
    return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_public_ref2va_workflow_uses_current_director_toolchain():
    workflow = _load_workflow()
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}

    cloud = nodes[271]
    assert cloud["type"] == "MiniMaxH3CloudDirector"
    assert cloud["properties"]["minimax_h3_cloud_widget_schema"] == 5
    assert [item["name"] for item in cloud["inputs"] if item.get("widget")] == [
        "prompt",
        "task_type",
        "duration_seconds",
        "shot_count",
        "rewrite_mode",
        "output_language",
        "cloud_provider",
        "api_model",
        "temperature",
        "max_tokens",
        "timeout_s",
        "api_reasoning",
        "analysis_mode",
        "frame_sequence_limit",
        "frame_selection_mode",
        "frame_selection_spec",
        "cloud_base_url",
        "my_preset",
        "gemini_video_route",
        "gemini_video_fps",
        "reference_fidelity",
    ]
    assert cloud["widgets_values"] == [
        "",
        "Ref2VA",
        5,
        0,
        "balanced",
        "English",
        "deepseek",
        "",
        0.2,
        16384,
        600,
        "auto",
        "auto",
        48,
        "uniform_full",
        "",
        "",
        "",
        "auto",
        0,
        "auto",
    ]

    assert nodes[283]["type"] == "MiniMaxH3ModuleFolderLoader"
    assert nodes[274]["type"] == "MiniMaxH3PromptModuleLoader"
    assert links[484] == [484, 283, 2, 274, 0, "H3_PROMPT_RULE_PACK"]
    assert links[474][1:5] == [274, 0, 271, 13]
    assert links[483][1:5] == [274, 3, 271, 14]
    assert links[478][1:5] == [271, 0, 279, 0]
    assert links[481][1:5] == [279, 0, 225, 9]
    assert nodes[279]["widgets_values"][-1] is True

    assert nodes[284]["type"] == "MiniMaxH3VideoContext"
    assert nodes[284]["outputs"][0]["links"] is None
    assert nodes[284]["outputs"][1]["links"] is None
    assert nodes[285]["type"] == "MiniMaxH3CloudVideoInput"
    assert nodes[285]["widgets_values"] == [""]

    note_titles = {node.get("title") for node in workflow["nodes"] if node["type"] == "Note"}
    assert "参考视频时间线：48 帧不是 48 秒" in note_titles
    assert "任务类型、素材编号与凭据安全" in note_titles


def test_public_ref2va_workflow_contains_no_private_connection_or_secret():
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = json.loads(text)
    nodes = {node["id"]: node for node in workflow["nodes"]}

    assert not re.search(r"sk-[A-Za-z0-9_-]{20,}", text)
    assert not re.search(r"Bearer\s+[A-Za-z0-9._-]{20,}", text, re.IGNORECASE)
    assert "C:\\Users\\" not in text
    assert "E:\\AI\\" not in text
    assert "deepseek-v4.1-flash-expires" not in text

    cloud_widget_names = [item["name"] for item in nodes[271]["inputs"] if item.get("widget")]
    cloud_values = dict(zip(cloud_widget_names, nodes[271]["widgets_values"], strict=True))
    assert cloud_values["api_model"] == ""
    assert cloud_values["cloud_base_url"] == ""
    assert cloud_values["my_preset"] == ""
    assert all(value == "（未选择）" for value in nodes[283]["widgets_values"])
