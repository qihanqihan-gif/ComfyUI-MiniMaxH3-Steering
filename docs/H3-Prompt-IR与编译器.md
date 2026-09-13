# H3 Prompt IR 与确定性编译器

这套设计不试图比大模型更会写创意，而是把“创意规划”和“官方格式”拆开。

```text
用户意图 / 参考图
        ↓
LLM 只生成结构化 Prompt IR（JSON）
        ↓
纯 Python 编译器固定字段、标签和首尾帧对齐语句
        ↓
纯 Python 校验器检查字段、镜头、时间码、标签和对白
        ↓
ComfyUI 官方 MiniMax H3 节点
```

## 为什么要有 IR

- 更换本地 VLM、云端 API 或 LoRA 时，后半段格式规则不变。
- LLM 不再负责逐字记住三段式、六段式和固定对齐句式。
- 编译结果可重复；同一份 IR 不会因为模型温度不同而改变字段顺序。
- 出错时可以判断是“创意内容有问题”，还是“协议格式有问题”。

它不会凭空提高画质，也不会替代提示词经验。它解决的是稳定性、可迁移性和可调试性。

## Base 模式 IR

适用于 T2VA、I2VA、FL2VA、L2VA。最小输入：

```json
{
  "task_type": "FL2VA",
  "duration_seconds": 5.0,
  "integrated_multimodal_description": "[Shot 1] ... [Shot 2] At 00:02.500, ...",
  "overall_soundscape": "海浪、衣料摩擦与脚步声",
  "non_diegetic_music": "N/A"
}
```

编译器会生成官方三字段结构，并按任务自动补首帧、首尾帧或尾帧对齐首行。`final_shot_number` 可显式提供；省略时从正文中的最大 `[Shot N]` 推导。

## Ref2VA 模式 IR

```json
{
  "task_type": "Ref2VA",
  "duration_seconds": 5.0,
  "subject_definitions": "<Subject 1> is ... from <Picture 1>.",
  "summary": "[Subject-driven generation] ...",
  "retention_analysis": "<Picture 1>: identity_preserved - ...",
  "detailed_description": "[Shot 1] ... [Shot 2] At 00:02.500, ...",
  "overall_soundscape": "...",
  "non_diegetic_music": "N/A"
}
```

编译器固定输出官方六字段顺序。缺失的主体、摘要、保留分析或详细描述会被校验器判为错误，而不会偷偷用虚构内容补齐。

## 安全规范化与媒体清单

导演调用编译器时会附带它实际看到、或用户在原始意图中明确声明的 Picture/Video/Audio 编号。模型输出若引用
清单外素材会停止，避免从字幕、画面或常识自行发明 `<Audio 1>`、`<Video 2>` 等条件。导演没有接到某类素材时，
该类型记为“未知”而不是武断地判定下游未连接；用户可在原始意图中显式写出外部 H3 标签。

同一 `<Picture N>` 可以在 subject_definitions、retention_analysis 和 detailed_description 等字段重复引用；
校验的是编号和素材关系，不是标签全文只能出现一次。

编译器只自动修复无需猜测语义的结构错误：连续镜头若整体误从 `[Shot 2]` 开始，会按出现顺序平移为
`[Shot 1]`；实际只有一张 Picture 且模型全文只使用同一个错误编号时，会规范成 `<Picture 1>`。跳号、多个冲突
Picture 编号、字段缺失或媒体清单越界仍然失败，不做猜测式改写。

## 两个节点如何使用

`MiniMaxH3PromptDirector` 会调用兼容 API 生成 IR，并在内部编译，输出：

1. `enhanced_prompt`：可直接接官方 H3 节点；
2. `report`：调用与校验摘要；
3. `reference_sheet`：分阶段识图时的素材事实表；
4. `prompt_ir`：可保存、修改或交给其他工具继续处理的 JSON。

`MiniMaxH3CompileValidate` 完全离线、零 API：

- 输入 JSON 时：规范化、编译并校验；
- 输入普通 H3 提示词时：保持原文不改，只校验；
- `fail_on_error=false` 适合研究和查看报告；发布工作流时可设为 `true` 防止错误提示词继续运行。

因此，导演节点后面不必重复接编译节点；独立编译节点主要用于手改 IR、接别的 LLM、导入社区提示词或做批量验证。

## 暂不塞进编译器的内容

- LoRA 的 trigger word、推荐强度和特殊语法：应放在独立的可选 Profile 层；
- 风格、运镜和场景模板：属于创作策略模块；
- 参考视频/音频解码和抽帧：属于素材处理层；
- 缓存、注意力和显存优化：属于推理层。

这些边界让基础编译器保持轻量，也避免某个社区 LoRA 或加速插件更新时拖垮整个节点包。
