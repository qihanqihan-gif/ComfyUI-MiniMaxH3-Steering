# 示例工作流

## MiniMax-H3-Lab Ref2VA 研究模板

文件：[`MiniMax-H3-Lab-Ref2VA-研究模板.json`](./MiniMax-H3-Lab-Ref2VA-研究模板.json)

这是开发期间使用的完整研究画布，包含：

```text
参考图 → PromptModuleLoader → PromptDirector → CompileValidate
      ↘ 官方 MiniMaxH3ReferenceToVideo → 采样 → 视频/音频解码
```

公开版本已经做过以下清理：

- 移除私人素材文件名和特定角色提示词，改成通用占位图片与原创角色示例。
- 文本编码器改为官方 INT8 ConvRot 文件名。
- PromptDirector、模块作用域和下游官方节点统一为 `Ref2VA`。
- 模块的 `module_manifest` 第 4 输出已接回导演节点。
- OpenCache 与 Steering 默认旁路，先跑无加速/无方向操控基线。

### 使用前必须做的事

1. 在两个 `Load Image` 节点中选择自己的参考图；仓库不分发占位图片。
2. 确认模型、文本编码器与两个 VAE 的本地文件名。
3. 启动 LM Studio 或填写其他 OpenAI-compatible API 地址和模型 ID。
4. 第一次运行保持 5 秒、24 步、固定 seed，并保持 OpenCache/Steering 旁路。

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
- 官方 `ReferenceToVideo` 的 `ref_image` 是软参考，不是硬首帧。
- 必须锁定第 0 帧时，请另建 `ImageToVideo` 工作流并连接 `first_frame`。
- `match` 会按生成画面面积等比缩小参考图；`max` 使用更高参考分辨率，可能慢数倍。

## 关于 Work-Fisher

本仓库不重新分发社区 Work-Fisher 工作流。本示例只是用户自己的研究画布，未复制或打包 Work-Fisher 原文件。
