# 示例工作流

## MiniMax-H3-Lab Ref2VA 研究模板

文件：[`MiniMax-H3-Lab-Ref2VA-研究模板.json`](./MiniMax-H3-Lab-Ref2VA-研究模板.json)

这是可直接替换素材后使用的完整研究画布。当前模板把近期稳定下来的导演、规则与时间线能力整理成
“主路径 + 可选工具区”：

```text
ModuleFolderLoader.rule_pack（可选）
  → PromptModuleLoader → CloudDirector → CompileValidate
  → 官方 MiniMaxH3ReferenceToVideo → 采样 → 视频/音频解码

VideoContext（可选）→ CloudDirector.video_frame_sequence + video_timeline_manifest
CloudVideoInput（可选，仅 Gemini）→ CloudDirector.cloud_video
```

模板只把云端导演放进执行主路径；本地 LM Studio 方案通过画布注释说明替换方式，不再复制第二个巨大导演
节点挤占画布。时间线和 Gemini 原生视频节点默认不接线，因此不会影响两张参考图的最小 Ref2VA 基线。

公开版本已经做过以下清理：

- 移除私人素材文件名和特定角色提示词，改成通用占位图片与原创角色示例。
- 文本编码器改为官方 INT8 ConvRot 文件名。
- CloudDirector、模块作用域和下游官方节点统一为 `Ref2VA`；模型 ID、Base URL 和已保存连接名保持空白。
- 模块的 `module_manifest` 第 4 输出已接回导演节点。
- 文件直载节点只通过结构化 `rule_pack` 接规则组合节点，默认不选择任何用户规则。
- `reference_fidelity=auto`、`rewrite_mode=balanced`，IR 编译校验失败时停止工作流。
- OpenCache 与 Steering 默认旁路，先跑无加速/无方向操控基线。

### 使用前必须做的事

1. 在两个 `Load Image` 节点中选择自己的参考图；仓库不分发占位图片。
2. 确认模型、文本编码器与两个 VAE 的本地文件名。
3. 在 CloudDirector 底部连接管理器中保存本机凭据；Key 位于 `ComfyUI/user`，不会写入工作流。
4. 第一次运行保持 5 秒、24 步、固定 seed，并保持 OpenCache/Steering 旁路。
5. 需要参考视频时，再把上游 IMAGE 帧批次接入 `VideoContext`：其两个输出分别连接导演的参考视频帧和
   时间线 manifest；完整视频还要另接官方 H3 的 `ref_video`。

### 画布中的新注释

- **从这里开始**：说明主路径、默认安全参数以及云端/本地导演替换关系。
- **参考视频时间线：48 帧不是 48 秒**：说明代表帧、FPS、时间线 manifest 与完整 H3 参考视频的分工。
- **任务类型、素材编号与凭据安全**：说明 I2VA/Ref2VA 边界、`reference_fidelity`、规则包接法和分享前清理项。
- **运行前检查**：集中列出素材、权重、凭据和固定基线检查。

### 节点依赖

核心 H3 提示词链只需要本仓库与包含 MiniMax-H3 原生节点的新版 ComfyUI。完整画布还保留了开发环境中的可选辅助节点：

| 节点 | 来源/处理方式 |
|---|---|
| `ResolutionSelector` / `PrimitiveFloat` / `ComfyMathExpression` | 新版 ComfyUI 核心 |
| `PathchSageAttentionKJ` | ComfyUI-KJNodes；缺少时删除并将模型加载器直接接 OpenCache |
| `Display Any (rgthree)` | rgthree-comfy；仅用于看提示词，可直接删除 |
| `LayerUtility: PurgeVRAM` | ComfyUI LayerStyle；只在“释放缓存”子图中 |
| `RAMCleanup` / `VRAMCleanup` | Comfyui-Memory_Cleanup；只在“释放缓存”子图中 |
| `easy cleanGpuUsed` | ComfyUI-Easy-Use；只在“释放缓存”子图中 |

如果缺少清理类节点，可删除右侧“释放缓存”子图，不影响生成主链。缺少 KJNodes 时要重新连接模型链，不能只删除节点后直接运行。

### 语义提醒

- `PromptDirector` 的图片端口只给提示词 API 识图，不会自动把图片送入 H3。
- I2VA 最多 1 个 Picture 锚点；多参考图或参考视频编辑必须显式选择 `Ref2VA`，节点不会自动改变任务类型。
- `VideoContext` 的代表帧数不是秒数，也不会裁剪上游视频；片段范围由上游加载/裁剪节点决定。
- 官方 `ReferenceToVideo` 的 `ref_image` 是软参考，不是硬首帧。
- 必须锁定第 0 帧时，请另建 `ImageToVideo` 工作流并连接 `first_frame`。
- `match` 会按生成画面面积等比缩小参考图；`max` 使用更高参考分辨率，可能慢数倍。

## 关于 Work-Fisher

本仓库不重新分发社区 Work-Fisher 工作流。本示例只是用户自己的研究画布，未复制或打包 Work-Fisher 原文件。
