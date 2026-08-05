# ComfyUI-MiniMaxH3-Steering

祈寒的个人 MiniMax-H3 魔改节点目录。主打一个 **加载时方向操控（activation steering）** 节点——它允许你在不修改任何权重文件的前提下，对 MiniMax-H3 文本编码器的行为方向做加减法，例如**降低文本编码器的拒绝倾向（refusal-like behaviors）**。

所有节点均为独立实现，**不替换 ComfyUI 任何核心文件**。

> Keywords: `comfyui` · `minimax-h3` · `steering` · `activation-direction` · `abliteration` · `text-encoder` · `qwen3vl`

## 节点清单

| 节点 | 作用 |
|---|---|
| **MiniMaxH3Steering** | ★ 主打节点：加载时方向操控。用 `tools/measure_directions.py` 自产方向向量，在文本编码器指定层（默认 40-49）的 o_proj 输出上做 `h -= λ·(h·d)·d` 投影移除。支持双方向（refusal / safety）+ 任意层区间 + 自定义 npy |
| MiniMaxH3OpenCache | 50 层 DiT 全层残差缓存（走官方 `double_block` 替换 hook），跳过未变化块的模型计算 |
| MiniMaxH3EasyCacheSafe | 官方 EasyCache 的鲁棒封装（多模型 key、跨图安全、异常直通） |
| MiniMaxH3PromptDirector | 顺序接口提示词导演：9 图接口（`<Picture i>` 一一对应）+ OpenAI 兼容多模态 API（纯 urllib 零依赖），输出 H3 三段式 JSON（integrated / soundscape / music） |
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
python -m pytest tests -q --import-mode=importlib   # 49 passed
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
