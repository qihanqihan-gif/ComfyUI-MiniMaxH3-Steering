# ComfyUI-MiniMaxH3-Steering

祈寒的个人 MiniMax-H3 魔改节点目录。主打一个 **加载时方向操控（activation steering）** 节点——它允许你在不修改任何权重文件的前提下，对 MiniMax-H3 文本编码器的行为方向做加减法，例如**降低文本编码器的拒绝倾向（refusal-like behaviors）**。

所有节点均为独立实现，**不替换 ComfyUI 任何核心文件**。

> Keywords: `comfyui` · `minimax-h3` · `steering` · `activation-direction` · `abliteration` · `text-encoder` · `qwen3vl` · `prompt-director` · `prompt-modules` · `local-first` · `lm-studio`

## ✨ 新：H3 提示词实验节点（v0.1.0-alpha，2026-08-07）

**MiniMax-H3-Lab —— 面向本地小/中型多模态模型的 H3 Prompt 实验与编译工具。**

使用 LM Studio / OpenAI-compatible API，将中文创意与最多 9 张参考图整理为 MiniMax H3 结构化提示词。

- **Local-first，开箱即用**：默认 `http://127.0.0.1:1234/v1`（LM Studio），空 key 放行——零配置零成本
- **单次 / 分阶段多图分析**：`auto` 模式 0-2 图单次、3-9 图分阶段（逐素材视觉分析 → 文字摘要 → 合并写作），任一失败自动回退，绝不报错
- **面向小上下文 VLM 的 Reference Context 压缩**：8B/12B/27B 本地模型也能承担多参考 H3 PromptDirector
- **JSON 热加载 Prompt Modules**：21 个可选创作策略/场景/图生视频模板（官方 8 个场景 skill 全转换 + 5 个图生视频通用模板），改 JSON 即加新模块，无需重启
- **H3 官方结构编译与基础校验**：h3_compiler 确定性校验（字段/时间码/标签/对白），errors/warnings 进 report 不阻塞
- **核心协议自动加载**：T2VA/I2VA/FL2VA → 三段式；Ref2VA → 六段式；用户只选创作策略，不会出现协议冲突
- **中文界面**：全部节点简体中文显示 + 中文输出支持
- **云端 OpenAI-compatible 接口实验性兼容**：任意兼容端点可直接填 base_url+key

> **Alpha 声明**：当前重点验证图片参考与本地 VLM；视频/音频参考、不同云端供应商仍在持续测试。
>
> **文档**：`docs/H3提示词编写规范-官方Skill核心提取.md`（三段式/六段式协议规范）· `docs/提示词编写经验.md`（社区+实测经验）· `docs/提示词模块编写指南.md`（写自己的模块）。

## 节点清单

| 节点 | 作用 |
|---|---|
| **MiniMaxH3Steering** | ★ 主打节点：加载时方向操控。用 `tools/measure_directions.py` 自产方向向量，在文本编码器指定层（默认 40-49）的 o_proj 输出上做 `h -= λ·(h·d)·d` 投影移除。支持双方向（refusal / safety）+ 任意层区间 + 自定义 npy |
| MiniMaxH3OpenCache | 50 层 DiT 全层残差缓存（走官方 `double_block` 替换 hook），跳过未变化块的模型计算 |
| MiniMaxH3EasyCacheSafe | 官方 EasyCache 的鲁棒封装（多模型 key、跨图安全、异常直通） |
| MiniMaxH3PromptDirector | 顺序接口提示词导演：9 图接口（`<Picture i>` 一一对应）+ OpenAI 兼容多模态 API（纯 urllib 零依赖），输出 H3 三段式 JSON（integrated / soundscape / music）；v0.1：两阶段/单次/自动分析模式 + 协议自动注入 + h3_compiler 校验 + 参考素材分析表输出 |
| **MiniMaxH3PromptModuleLoader** | ★ 提示词模块节点（热加载）：21 个可选创作策略/场景/图生视频模块 + 自定义规则框，输出并入 PromptDirector（核心协议由导演节点自动加载，不可选） |
| MiniMaxH3ReferenceMediaPrep | 参考素材预处理：等比缩放补边/裁剪选择、视频均匀抽帧、帧数对齐 |
| MiniMaxH3ReferenceInspector | 在官方 ReferenceToVideo 前检查尺寸/帧数/张量内存/参考 token/可用显存 |
| MiniMaxH3PerformanceProfiler | 采样墙钟、模型调用、CUDA 时间、峰值显存记录 |
| 汉化层 | 官方四个 H3 节点 + 官方工作流常用节点的简体中文显示（不改序列化值） |

## MiniMaxH3Steering 说明

该节点**不修改权重文件**，运行时在文本编码器前向中做方向投影移除：

```text
h = o_proj 输出激活（band 内每层）
d = 方向向量（从 data/*.npy 加载，可自选）
h ← h − λ · (h·d) · d
```

- **方向向量完全自产**：`tools/measure_directions.py` 用两组对比提示词（如"触发拒绝的提示词" vs "中性提示词"）分别过一遍编码器，取激活均值差作为方向。**仓库不附带任何方向数据**，也无需任何外部数据集。
- **效果可量化**：测量脚本同时输出"拒绝率"变化报告（`--report`），例如这里 99% → 4%，完全中性描述。
- 支持 1D 单方向（广播所有层）与 2D 每层独立方向（`[n_layers, hidden]`）两种 npy 规格。
- 天然可逆：λ=0 即完全关闭。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/qihanqihan-gif/ComfyUI-MiniMaxH3-Steering.git
```

或通过 ComfyUI-Manager 的 Custom Nodes 搜索安装（注册后）。重启 ComfyUI 即生效，无需额外依赖（纯 Python + torch/numpy）。

## 测试

```bash
cd ComfyUI-MiniMaxH3-Steering
python -m pytest tests -q --import-mode=importlib   # 84 passed
```

## 许可证与合规

- 本仓库代码：**GPL-3.0-only**（见 LICENSE）。
- **模型权重与编码器**：使用 MiniMax-H3 官方权重 / 文本编码器时，请遵守 [MiniMax-H3 社区许可证](https://huggingface.co/MiniMaxAI/MiniMax-H3)。本仓库**不包含任何权重**，方向数据也由用户自行测量生成。
- 第三方参考声明见 THIRD_PARTY_NOTICES.md（OpenCache 概念参考 `lihaoyun6/ComfyUI-MiniMaxH3-Cache`，均为新实现）。

## 安全说明

- **模型列表刷新接口**（`/minimaxh3lab/api/models`）：仅在本地 ComfyUI 服务上监听；出站目标限制为 http/https，并拒绝链路本地（169.254.0.0/16 云元数据等，含 IPv4-mapped IPv6 形式）与多播地址。回环（127.0.0.1，如本地 LM Studio）与局域网地址按用途放行——请勿将 ComfyUI 端口暴露到不可信网络。
- **API 密钥**：只从环境变量（`MINIMAX_H3_API_KEY` / `OPENAI_API_KEY`）或节点输入读取，不写日志、不随仓库分发。
- 本仓库不包含任何模型权重与方向数据；方向向量由 `tools/measure_directions.py` 在本地自行测量生成。

## 兼容性

- ComfyUI 开发版与 v0.3.0+ 稳定版（官方 EasyCache 节点自 v0.3.52 起内置）。
- Windows / Linux 均可，Python 3.11+（与 ComfyUI 官方环境一致）。
