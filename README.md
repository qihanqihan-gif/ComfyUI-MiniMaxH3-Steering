# ComfyUI-MiniMaxH3-Steering / MiniMaxH3-Lab

祈寒的个人 MiniMax-H3 实验节点目录。仓库从最初的 **加载时方向操控（activation steering）** 节点，逐步扩展为一套本地优先的提示词规划、确定性编译、参考素材检查与性能实验工具。

Steering 允许在不修改任何权重文件的前提下，对 MiniMax-H3 文本编码器的行为方向做运行时投影；PromptDirector/Prompt IR 则把“理解参考素材”和“严格生成 H3 格式”分开，方便本地小/中型多模态模型参与工作流。

所有节点均为独立实现，**不替换 ComfyUI 任何核心文件**。

> Keywords: `comfyui` · `minimax-h3` · `steering` · `activation-direction` · `abliteration` · `text-encoder` · `qwen3vl` · `prompt-director` · `prompt-modules` · `local-first` · `lm-studio`

## ✨ v0.2.0-alpha：H3 提示词实验与编译底座

**MiniMax-H3-Lab —— 面向本地小/中型多模态模型的 H3 Prompt 实验与编译工具。**

使用 LM Studio / OpenAI-compatible API，将中文创意与最多 9 张参考图整理为 MiniMax H3 结构化提示词。

- **Local-first**：默认连接 `http://127.0.0.1:1234/v1`（LM Studio），本地服务可留空 key；仍需先启动服务并加载可识图模型
- **单次 / 分阶段多图分析**：`auto` 模式 0-2 图单次、3-9 图分阶段（逐素材视觉分析 → 文字摘要 → 合并写作）；失败时给出诊断并安全直通原始提示词
- **面向小上下文 VLM 的 Reference Context 压缩**：8B/12B/27B 本地模型也能承担多参考 H3 PromptDirector
- **JSON 热加载 Prompt Modules**：26 个可选创作策略/场景/图生视频模板；带来源、证据等级、作用域、冲突与功能标签，改 JSON 即生效
- **Prompt IR → 官方格式的确定性编译**：LLM 只负责内容规划，纯 Python 负责三段式/六段式、首尾帧对齐句式和字段顺序
- **离线编译与校验节点**：既能编译 Prompt IR JSON，也能原样检查社区提示词；校验字段/镜头/时间码/标签/对白
- **核心协议自动加载**：T2VA/I2VA/FL2VA/L2VA → 三段式；Ref2VA → 六段式；用户只选创作策略，不会出现协议冲突
- **中文界面**：全部节点简体中文显示；正式 H3 Prompt 默认 English，中文输出保留为实验便捷模式
- **云端 OpenAI-compatible 接口实验性兼容**：任意兼容端点可直接填 base_url+key

> **Alpha 声明**：当前重点验证图片参考、本地 VLM 和官方格式编译。视频/音频参考由官方 H3 节点处理；不同云端兼容端点、Linux 与更多 Python/CUDA 组合仍需社区实测。
>
> **文档**：`docs/H3-Prompt-IR与编译器.md`（新编译层）· `docs/参考素材接口与首帧语义.md`（Picture/Video/Audio 与硬首帧区别）· `docs/H3短视频回归测试矩阵.md`（固定 6 例）· `docs/H3提示词编写规范-官方Skill核心提取.md` · `docs/提示词模块编写指南.md`。

## 节点清单

| 节点 | 作用 |
|---|---|
| **MiniMaxH3Steering** | ★ 主打节点：加载时方向操控。用 `tools/measure_directions.py` 自产方向向量，在文本编码器指定层（默认 40-49）的 o_proj 输出上做 `h -= λ·(h·d)·d` 投影移除。支持双方向（refusal / safety）+ 任意层区间 + 自定义 npy |
| MiniMaxH3OpenCache | 50 层 DiT 全层残差缓存（走官方 `double_block` 替换 hook），跳过未变化块的模型计算 |
| MiniMaxH3EasyCacheSafe | 官方 EasyCache 的鲁棒封装（多模型 key、跨图安全、异常直通） |
| MiniMaxH3PromptDirector | 顺序接口提示词导演：9 个“导演识图素材”接口 + OpenAI 兼容多模态 API（纯 urllib 零依赖）；图片只发给提示词 API，必须另接 H3 官方节点；输出正式提示词、诊断、参考素材表与 Prompt IR；支持模块 manifest 追踪与任务/作用域错配警告 |
| **MiniMaxH3CompileValidate** | ★ 离线 Prompt IR 编译与提示词校验：JSON 输入会编译成官方三段式/六段式，普通文本输入只校验不改写；输出最终提示词、报告、规范化 IR 与是否通过 |
| **MiniMaxH3PromptModuleLoader** | ★ 提示词模块节点（热加载）：26 个可选模块 + 自定义规则框；自动去重、作用域/冲突/元数据诊断，并输出可追踪 manifest（核心协议自动加载，不可选） |
| MiniMaxH3ReferenceMediaPrep | 参考素材预处理：等比缩放补边/裁剪选择、视频均匀抽帧、帧数对齐 |
| MiniMaxH3ReferenceInspector | 在官方 ReferenceToVideo 前按真实 `match/max` 和视频画布公式检查尺寸/帧数/参考 token/可用内存，并提示 ref_image 不等于首帧 |
| MiniMaxH3PerformanceProfiler | 采样墙钟、模型调用、CUDA 时间、峰值显存记录 |
| 汉化层 | 官方四个 H3 节点 + 官方工作流常用节点的简体中文显示（不改序列化值） |

## 快速开始：提示词链

推荐先只接最小链路，确认提示词输出正确后再加入缓存和 Steering：

```text
Input Text
  ├─> MiniMaxH3PromptModuleLoader.system_prompt_module
  ├─> MiniMaxH3PromptDirector.prompt
  └─> MiniMaxH3PromptDirector.ref_image_1..9（仅供提示词 API 识图）

PromptModuleLoader.module_manifest ─> PromptDirector.module_manifest
PromptDirector.enhanced_prompt ─> MiniMaxH3CompileValidate
CompileValidate.final_prompt ─> 官方 MiniMax H3 生成节点 prompt
```

参考图片必须**另外连接**到官方生成节点。导演节点的图片输入不会给 H3 增加视觉条件。

完整研究画布见 [`example_workflows/MiniMax-H3-Lab-Ref2VA-研究模板.json`](example_workflows/MiniMax-H3-Lab-Ref2VA-研究模板.json)。公开副本已移除私人素材名、改用官方编码器文件名、修正 Ref2VA 任务错配，并让 OpenCache/Steering 默认旁路。完整画布包含少量可选第三方辅助节点，依赖与删除方法见 [`example_workflows/README.md`](example_workflows/README.md)。

## MiniMaxH3Steering 说明

该节点**不修改权重文件**，运行时在文本编码器前向中做方向投影移除：

```text
h = o_proj 输出激活（band 内每层）
d = 方向向量（从 data/*.npy 加载，可自选）
h ← h − λ · (h·d) · d
```

- **方向向量完全自产**：`tools/measure_directions.py` 用两组对比提示词（如"触发拒绝的提示词" vs "中性提示词"）分别过一遍编码器，取激活均值差作为方向。**仓库不附带任何方向数据**，也无需任何外部数据集。
- **测量结果可复用**：脚本输出逐层归一化的 `.npy` 方向向量及 shape；效果评估需要用户用固定提示词、固定种子另做 A/B，对脚本没有实现的“拒绝率报告”不作承诺。
- 支持 1D 单方向（广播所有层）与 2D 每层独立方向（`[n_layers, hidden]`）两种 npy 规格。
- 天然可逆：λ=0 即完全关闭。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/qihanqihan-gif/ComfyUI-MiniMaxH3-Steering.git
```

或通过 ComfyUI-Manager 的 Custom Nodes 搜索安装（注册后）。重启 ComfyUI 即生效，无需额外依赖（纯 Python + torch/numpy）。

## 测试

从 **ComfyUI 根目录**运行（不要在插件目录内部直接收集，因为 pytest 会把带连字符的插件目录误当成顶层包）：

```bash
python -m pytest custom_nodes/ComfyUI-MiniMaxH3-Steering/tests -q --import-mode=importlib
```

当前工作树：`110 passed`（Python 3.11，2026-08-09；4 条来自 Triton 的弃用警告不属于测试失败）。

## 许可证与合规

- 本仓库代码：**GPL-3.0-only**（见 LICENSE）。
- **模型权重与编码器**：使用 MiniMax-H3 官方权重 / 文本编码器时，请遵守 [MiniMax-H3 社区许可证](https://huggingface.co/MiniMaxAI/MiniMax-H3)。本仓库**不包含任何权重**，方向数据也由用户自行测量生成。
- 第三方参考声明见 THIRD_PARTY_NOTICES.md（OpenCache 概念参考 `lihaoyun6/ComfyUI-MiniMaxH3-Cache`，均为新实现）。

## 安全说明

- **模型列表刷新接口**（`/minimaxh3lab/api/models`）：仅在本地 ComfyUI 服务上监听；出站目标限制为 http/https，并拒绝链路本地（169.254.0.0/16 云元数据等，含 IPv4-mapped IPv6 形式）与多播地址。回环（127.0.0.1，如本地 LM Studio）与局域网地址按用途放行——请勿将 ComfyUI 端口暴露到不可信网络。
- **API 密钥**：只从环境变量（`MINIMAX_H3_API_KEY` / `OPENAI_API_KEY`）或节点输入读取，不写日志、不随仓库分发。
- 本仓库不包含任何模型权重与方向数据；方向向量由 `tools/measure_directions.py` 在本地自行测量生成。

## 兼容性

- 当前实测环境：ComfyUI v0.30.x 开发线、Windows、Python 3.11。
- Prompt IR 编译/校验层为纯 Python 标准库逻辑，不依赖 Triton、CUDA 架构或第三方 wheel；PromptDirector 的图像/API 路径使用 ComfyUI 已带的 torch/numpy/Pillow。
- Linux 与 Python 3.12/3.13 按实现应可用，但尚未完成真机矩阵测试；不把“理论兼容”写成“已经验证”。
