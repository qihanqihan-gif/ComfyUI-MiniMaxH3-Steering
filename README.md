# ComfyUI-MiniMaxH3-Steering / MiniMaxH3-Lab

MiniMax-H3 实验节点目录。项目从最初的 **加载时方向操控（activation steering）** 节点，逐步扩展为一套本地优先的提示词规划、确定性编译、参考素材检查与性能实验工具。

Steering 允许在不修改任何权重文件的前提下，对 MiniMax-H3 文本编码器的行为方向做运行时投影；PromptDirector/Prompt IR 则把“理解参考素材”和“严格生成 H3 格式”分开，方便本地小/中型多模态模型参与工作流。

所有节点均为独立实现，**不替换 ComfyUI 任何核心文件**。

> Keywords: `comfyui` · `minimax-h3` · `steering` · `activation-direction` · `abliteration` · `text-encoder` · `qwen3vl` · `prompt-director` · `prompt-modules` · `local-first` · `lm-studio`

## ✨ v0.3.0-alpha.2：参考视频跟随、IR 纪律与公开模板

**MiniMax-H3-Lab —— 面向本地小/中型多模态模型的 H3 Prompt 实验与编译工具。**

使用 LM Studio / OpenAI-compatible API，将中文创意与最多 9 张参考图整理为 MiniMax H3 结构化提示词。

- **参考视频跟随档**：`auto/locked/structural/loose` 与改写自由度分离，避免创意模式越过动作、姿势、镜头和时点约束
- **真实时间线上下文**：可从完整 IMAGE 帧批次均匀选择代表帧，并向导演传递源索引、时间码和真实切镜；代表帧数不是秒数
- **IR 与媒体清单前置校验**：镜头编号、Picture 数量和未接入的媒体标签会在进入 H3 前失败；I2VA 多图在提示词 API 调用前停止
- **公开 Ref2VA 模板更新**：云端导演作为主路径，规则文件直载、时间线和 Gemini 原生视频入口分区展示并附接线/安全注释
- **Local-first**：默认连接 `http://127.0.0.1:1234/v1`（LM Studio），本地服务可留空 key；仍需先启动服务并加载可识图模型
- **单次 / 分阶段多图分析**：`auto` 模式 0-2 图单次、3-9 图分阶段（逐素材视觉分析 → 文字摘要 → 合并写作）；最终 API/JSON/IR 失败默认停止工作流，避免原提示词静默流入视频生成
- **Qwen3.8 分阶段思考适配**：自动模式把推理能力留给逐图/视频序列理解，最终 Prompt IR 编译改为 non-thinking；保留 `finish_reason` 与 prompt/completion/reasoning token 诊断，截断时自动直接作答重试一次
- **面向小上下文 VLM 的 Reference Context 压缩**：8B/12B/27B 本地模型也能承担多参考 H3 PromptDirector
- **JSON 热加载 Prompt Modules**：26 个可选创作策略/场景/图生视频模板 + 1 个保留但不注入的作者参考条目；所有规则统一放在 `prompt_modules/`，其中 `builtin/` 是中文正式库、`legacy/` 只负责旧工作流兼容、`user/` 用于个人规则
- **Prompt IR → 官方格式的确定性编译**：LLM 只负责内容规划，纯 Python 负责三段式/六段式、首尾帧对齐句式和字段顺序
- **离线编译与校验节点**：既能编译 Prompt IR JSON，也能原样检查社区提示词；校验字段/镜头/时间码/标签/对白
- **核心协议自动加载**：T2VA/I2VA/FL2VA/L2VA → 三段式；Ref2VA → 六段式；用户只选创作策略，不会出现协议冲突
- **中文界面**：全部节点简体中文显示；正式 H3 Prompt 默认 English，中文输出保留为实验便捷模式
- **云端 OpenAI-compatible 接口实验性兼容**：任意兼容端点可直接填 base_url+key

> **Alpha 声明**：当前重点验证图片参考、本地 VLM 和官方格式编译。视频/音频参考由官方 H3 节点处理；不同云端兼容端点、Linux 与更多 Python/CUDA 组合仍需社区实测。
>
> **文档**：`docs/关键结论与当前状态.md`（维护状态）· `docs/快速开始与回归测试.md`（现有视频替换工作流 A/B）· `docs/H3-Prompt-IR与编译器.md`（新编译层）· `docs/参考素材接口与首帧语义.md`（Picture/Video/Audio 与硬首帧区别）· `docs/H3短视频回归测试矩阵.md`（固定 6 例）· `docs/H3提示词编写规范-官方Skill核心提取.md` · `docs/提示词模块编写指南.md` · `docs/提示词模块重写路线图.md` · `docs/从Skill到创作规则模块.md` · `docs/模块作者Skill设计.md` · `docs/创作规则作者检查清单.md` · `docs/用户自定义创作规则.md`。

## 节点清单

| 节点 | 作用 |
|---|---|
| **MiniMaxH3Steering** | ★ 主打节点：加载时方向操控。用 `tools/measure_directions.py` 自产方向向量，在文本编码器指定层（默认 40-49）的 o_proj 输出上做 `h -= λ·(h·d)·d` 投影移除。支持双方向（refusal / safety）+ 任意层区间 + 自定义 npy |
| MiniMaxH3OpenCache | 50 层 DiT 全层残差缓存（走官方 `double_block` 替换 hook），跳过未变化块的模型计算 |
| MiniMaxH3EasyCacheSafe | 官方 EasyCache 的鲁棒封装（多模型 key、跨图安全、异常直通） |
| MiniMaxH3PromptDirector | 顺序接口提示词导演：9 个独立参考图接口 + 1 个参考视频抽帧/图像序列批次接口；自动模式会逐张分析参考图、联合分析有序视频帧，再合成 H3 Prompt IR。所有图片只发给提示词 API，必须另接 H3 官方节点；支持参考视频跟随档、媒体清单校验、模块 manifest 追踪与接线诊断 |
| **MiniMaxH3CloudDirector** | ★ 云端多模态导演：节点正面只暴露连接预设、媒体上传策略、参考视频跟随档和生成预算，供应商 host、默认模型、thinking/wire/schema 由受控预设与 adapter 解析。DeepSeek 可做最高 600 帧边界实验；Gemini 支持抽帧与原生视频 A/B。Key 从环境变量或 ComfyUI 用户目录本地凭据文件读取，不进入工作流 |
| **MiniMaxH3CloudVideoInput** | Gemini 原生视频输入：选择 ComfyUI input 内的视频，以不透明运行时对象连接云端导演；不解码为数百张前端预览，不把绝对路径、视频字节或远端 URI写入普通节点输出 |
| **MiniMaxH3CompileValidate** | ★ 离线 Prompt IR 编译与提示词校验：JSON 输入会编译成官方三段式/六段式，普通文本输入只校验不改写；输出最终提示词、报告、规范化 IR 与是否通过 |
| **MiniMaxH3PromptModuleLoader** | ★ 创作规则组合节点（热加载）：26 个可选规则 + 自定义规则框；在调用 LLM 前确定性处理去重、作用域、依赖、冲突与互斥组。默认 `standard` 保持当前正文，另有 `compact/strong` 实验渲染档；manifest 记录 applied/suppressed、档位与可复现哈希（核心协议与作者参考资料不可选） |
| **MiniMaxH3ModuleFolderLoader** | ★ 创作规则文件直载节点（轻量）：一次选择最多 5 个内置/用户 JSON；原样文本可直接接导演做简单对照，结构化规则包可接创作规则组合统一消解并生成 manifest |
| MiniMaxH3ReferenceMediaPrep | 参考素材预处理：等比缩放补边/裁剪选择、视频均匀抽帧、帧数对齐 |
| MiniMaxH3ReferenceInspector | 在官方 ReferenceToVideo 前按真实 `match/max` 和视频画布公式检查尺寸/帧数/参考 token/可用内存，并提示 ref_image 不等于首帧 |
| MiniMaxH3PerformanceProfiler | 采样墙钟、模型调用、CUDA 时间、峰值显存记录 |
| 汉化层 | 官方四个 H3 节点 + 官方工作流常用节点的简体中文显示（不改序列化值） |

## 快速开始：提示词链

公开模板默认使用凭据不进入工作流的云端导演。推荐先只接最小链路，确认提示词输出正确后再加入缓存和
Steering：

```text
MiniMaxH3ModuleFolderLoader.rule_pack（可选）
  └─> MiniMaxH3PromptModuleLoader.external_rule_pack

Input Text ────────────────────────────────> MiniMaxH3CloudDirector.prompt
MiniMaxH3PromptModuleLoader.system_prompt_module ─> MiniMaxH3CloudDirector.system_module
MiniMaxH3PromptModuleLoader.module_manifest ──────> MiniMaxH3CloudDirector.module_manifest
参考图 ───────────────────────────────────> MiniMaxH3CloudDirector.ref_image_1..9（仅供提示词 API 识图）

VHS_LoadVideo.IMAGE（完整帧 batch）
  ├─> MiniMaxH3VideoContext（可选；默认 48 张代表帧 + 真实时间线 manifest）
  │      ├─> MiniMaxH3CloudDirector.video_frame_sequence
  │      └─> MiniMaxH3CloudDirector.video_timeline_manifest
  └─> 官方 MiniMaxH3ReferenceToVideo.ref_video_0（仍需另接完整参考视频）

MiniMaxH3CloudDirector.enhanced_prompt ─> MiniMaxH3CompileValidate
CompileValidate.final_prompt ─> 官方 MiniMax H3 生成节点 prompt
```

`MiniMaxH3VideoContext` 的 48 表示“最多选择 48 张代表帧”，不是 48 秒时间窗口；节点不会裁剪原视频。
不需要真实时间码时，也可以把 IMAGE batch 直接接云端导演。若改用本地 LM Studio，把云端导演替换为
`MiniMaxH3PromptDirector` 并保留相同的规则、素材、时间线和编译校验接线即可。

自建规则推荐走结构化接线：

```text
ModuleFolderLoader.rule_pack ─> PromptModuleLoader.external_rule_pack
ModuleFolderLoader.system_prompt_module ─> PromptDirector.system_module  # 仅轻量直连 / A-B 对照时使用
```

文件直载的文本输出不会处理作用域、依赖或冲突；结构化规则包接入创作规则组合后，才会与节点槽位中的
内置规则一起解析。两条路径择一使用，不要把同一份规则重复接入。内置正式文件位于
`prompt_modules/builtin/`，个人规则放在 `prompt_modules/user/<自定义目录>/`；起步模板在 `templates/`。
`prompt_modules/legacy/` 仅供兼容，不要手动加入新规则。

现有内置规则不会一次性重写。`docs/提示词模块重写路线图.md` 把 26 个可选规则按风险分批，并要求先固定
canonical semantic contract，再生成 `instructions` 和做回归。用户若要把官方/社区 Skill 或 Agent 工作流
改成插件规则，应先使用 `docs/从Skill到创作规则模块.md` 和
`templates/Skill转规则模块工作单.example.md` 做四向分拣；完整 Skill 的工具、审批、模型调用、剪辑与交付流程
不属于单次提示词规则。

规则组合节点的渲染档仅用于固定样本 A/B：`standard` 是默认且保持当前正文，`compact` 只渲染 canonical
contract 摘要，`strong` 在当前正文后附同一契约复核。三档不代表质量排序；规则清单会记录档位、模块集合哈希
与最终规则文本哈希，只有在多个固定样本中反复出现同一失败后，才考虑增加窄范围模型 override。

参考图片必须**另外连接**到官方生成节点。导演节点的图片输入不会给 H3 增加视觉条件。
两个导演节点底部均提供“📎 插入 / 检查素材引用”：弹窗按当前实际连线显示
“物理参考图端口 → 密集 `<Picture N>` 编号 → 上游节点”，点击即可在用户意图文本框的光标处
插入官方标签，并非阻断地提醒错号、漏引用或无效 `<Video N>`。该交互不加载缩略图、不读取几百帧，
也不改写用户提示词；即使没有显式写标签，已连接素材仍会发送给提示词模型。
使用受支持的云端多模态 API 时可把上图中的 `MiniMaxH3PromptDirector` 换成
`MiniMaxH3CloudDirector`；其默认 Ref2VA、48 帧、16384 输出 token、600 秒超时，且始终
`fail-closed`。它仍然只是提示词导演，不会替代官方 H3 conditioning 或自动把素材传给 H3。

`video_frame_sequence` 接受普通 IMAGE、`VHS_LoadVideo` 的完整帧 batch 或
`VHS_SelectImages` 输出的已选图像序列。默认最多发送 4 帧；输入超过上限时包含首尾地均匀抽取，
并在报告中列出原 batch 索引。序列帧共同标记为 `<Video 1>` 时间线证据，不占用
`<Picture N>` 编号。日常上限仍建议 300；云端节点对官方 DeepSeek 单独开放最高 600 帧边界实验，
并把序列 Data URL 控制在约 28 MiB，超过预算时会进一步缩图，仍超限才确定性减少帧数。
Gemini inline 模式会把全部参考图与序列帧媒体控制在约 15 MiB，为其 20MB 总请求限制中的
文本、system、schema 与 JSON 外壳留出空间；两者都会按帧数自适应降低 JPEG 尺寸/质量。
若最低编码档仍超预算，会在报告中明确记录均匀减帧，避免请求直接撞到服务端请求体上限。

`analysis_mode=auto` 对本地模型沿用分阶段分析；当前 DeepSeek/Gemini 云端预设已接视频帧时，目标参考图与
`<Video 1>` 连续帧会在**同一次**请求中联合理解，避免阶段摘要被模型误当成一个需要重绘的新开场。
`single` 可显式固定该路径，`staged` 则保留为同素材 A/B；阶段分析失败会回退到最终调用直传
全部已选帧，不会静默丢帧。600 帧档只用于验证 DeepSeek API 边界，不是日常默认；报告会同时给出
请求上限、实际发送帧数、payload MiB、prompt token、transport 耗时和请求指纹，便于判断数量与耗时关系。

`reference_fidelity` 将提示词改写自由度与参考视频跟随度分开：`auto` 在 Ref2VA 视频任务中按
`structural` 执行，保留主要动作阶段、位移方向、真实切镜和顺序；`locked` 进一步锁定姿势、手势、机位和
相对时点；`loose` 才允许把参考视频作为灵感重新编排。即使 `rewrite_mode=creative`，也不能越过所选跟随档。
该字段对旧工作流为可选，缺失时由后端回退为 `auto`，不要求刷新浏览器或重建节点。I2VA/T2VA/FL2VA/L2VA
接入超过官方数量的参考图时会在调用提示词 API 前停止并提示改用 Ref2VA，不会自动改变任务类型。
导演还会按实际连接和用户显式声明生成 Picture/Video/Audio 清单；模型自行创造的越界标签会在进入 H3 前停止。

如需保留真实时间码，可先接 `MiniMaxH3VideoContext`：它输出选帧批次和确定性 timeline manifest，
二者分别接导演的 `video_frame_sequence` 与 `video_timeline_manifest`。两路同时连接时以上游选出的代表帧为准，
导演节点的帧数、选帧方式和自定义选帧参数不再二次生效，并继续使用源视频帧索引、
时间码与切镜分段。该节点不会裁剪输入视频，片段起止范围仍由上游视频加载/裁剪节点决定；
它只是轻量“时间线坐标层”，不做复杂剪辑工作台，也不生成数百张浏览器预览。

导演直连 IMAGE 批次时的序列帧快捷选择方式：

- `uniform_full`：均匀覆盖完整序列并包含首尾，默认推荐。
- `uniform_no_edges`：均匀覆盖约 10%–90%，适合片头/片尾可能有黑场或淡入淡出的视频。
- `custom_indices`：填写 `0,41,82,-1`；负数从末尾计数，`-1` 是最后一帧。
- `custom_percent`：填写 `0,33,66,100`，不需要知道视频总帧数。

自定义值会去重、按时间顺序排列；多于帧上限时仍在自定义候选中均匀压缩。
空值、越界值或格式错误会安全回退 `uniform_full`，report 与 reference sheet 都会记录原因。

### Qwen3.8 / 长输出设置

- `max_tokens` 新节点默认 `8192`，可选上限开放到 `131072`。这个数是请求的最大生成预算，
  实际可用值仍受 LM Studio 加载时的上下文长度以及本次输入 token 占用限制；把滑块调大不会自动扩大上下文。
- `api_reasoning=auto` 且模型 ID 能识别为 Qwen3.8 或 Gemma4 时：素材分析调用允许思考，最终固定 JSON
  编译调用关闭思考。连接本机 LM Studio 时，默认的分析/最终调用都会优先走原生 `/api/v1/chat`，
  由模型 ID 触发服务端按需/JIT 加载；`off` 映射原生 `reasoning=off`，`on/auto` 的分析阶段映射为
  `reasoning=on`。原生端点不可用才回退 `/v1/chat/completions`；`json_mode=force` 可显式强制兼容路径。
- 其他 OpenAI-compatible 最终调用使用 JSON Schema；如果服务器不支持，`json_mode=auto_retry`
  会退回纯提示约束。本地 Qwen3.8 原生接口没有 JSON Schema 参数，因此依靠字段契约、解析与一次恢复重试。
  如果 `finish_reason=length`、可见内容为空或 JSON 不完整，节点会追加紧凑直接作答要求并重试一次。
- 报告会显示首次/恢复调用的 prompt、completion 与 reasoning token。`API / JSON 失败策略`
  默认是 `stop`，防止 API 失败后原始模板被误送给 H3；只有需要旧行为时才选择 `passthrough`。

### 云端连接预设、媒体上传与 API Key

- `MiniMaxH3CloudDirector` 的节点标题和通用策略不绑定模型名；“云端连接 / 兼容预设”负责选择
  provider adapter。DeepSeek 固定官方 `https://api.deepseek.com`，默认
  `deepseek-v4-flash-vision-exp`；Gemini 固定 Google 原生 `v1beta:generateContent`，默认
  `gemini-3.1-flash-lite`。“模型 ID 覆盖”留空即跟随预设，仅在供应商更新模型 ID 或做 A/B 时填写。
  节点不提供会进入工作流的 API Key widget；点击底部“管理云端连接”可保存、替换、清除或测试当前凭据。
- 内置连接中，DeepSeek、Gemini 与 GLM 已有本项目真实探针/调用证据；豆包、Claude 的 wire 与探针已经实现，
  但在本项目拿到对应 Key 完成固定样本回归前仍标作“未实测”。MiniMax 官方公开文本 Chat 未证明能接收本节点的
  多图/视频帧内容块，因此不再展示一个虚构的 M3 视觉模型；旧工作流若仍保存该预设会在联网前明确阻断。
- “自定义 OpenAI 兼容”和“我的预设”不再是两个并列 provider。节点只显示一个 OpenAI 兼容入口；底部
  “连接管理”中可把表单临时用于当前节点，也可反复保存、更新、选择和删除 Base URL + 模型 ID。
  每个已保存连接生成稳定且独立的 `credential_id`，可以分别配置 Key；切回内置 provider 时前端会清除
  OpenAI 兼容专属的 Base URL/连接名，避免残留字段造成误解。旧工作流里的 `my_presets` 会迁移到同一入口。
- 点击“刷新 / 选择云端模型”才会按当前连接预设发出一次模型列表请求；不会在打开工作流时
  自动联网。Key 只由 ComfyUI 后端从当前 provider 的本地凭据/环境变量解析，浏览器只收到模型 ID、
  显示名、生成方法与公开 token 上限。内置 provider 走各自官方/静态适配；OpenAI 兼容连接会尝试
  标准 `GET <Base URL>/models`，没有该端点时仍可手填模型 ID。选择器支持搜索、恢复预设默认值，
  并用 `★` 标出适合文本 Prompt IR 的候选；“账户可见”不等于已通过本插件的多图、thinking 与严格 JSON 实测。
- “参考图与视频如何发送”直接说明媒体处理：`auto/single` 将按连接顺序编号的参考图与所选
  `<Video 1>` 连续帧同次上传；`staged` 先分别分析再汇总。节点底部的“参考图 / 视频如何发送”
  会再次提示：这些素材只供提示词 API 识别，仍需另接官方 H3 条件端口。
- Gemini 还可连接 `MiniMaxH3CloudVideoInput` 的 `cloud_video`：`auto` 在文件不超过 8 MiB 时使用
  `inlineData`，更大文件使用官方 Files API；也可强制选择 `inline`（保守上限 15 MiB）或 `file_api`。
  Files API 视频会等待 `ACTIVE` 后加入 `generateContent`，调用结束或处理失败时尽力删除远端临时文件。
  `Gemini 原生视频` 与 `IMAGE 视频帧批次` 必须二选一，避免一次请求里混入两份 `<Video 1>`；
  原生视频可包含音频，默认按供应商 1 FPS 解析，也可显式设为 0–24 FPS。此路径已通过本地契约/mock
  回归，尚未代替用户做真实 Key 的视频调用验收。
- 凭据目录**不是安装插件时创建**，也不会因加载节点或查看状态而创建。第一次成功保存 Key 时才创建：

```text
ComfyUI/user/default/MiniMaxH3-Lab/cloud_credentials.json
ComfyUI/user/default/MiniMaxH3-Lab/custom_presets.json
```

  `custom_presets.json` 只含预设名、Base URL、模型 ID 和非秘密凭据引用，不含 Key；它也只在第一次保存
  自定义预设时创建。DeepSeek 与 Gemini 分别使用 `deepseek_default` / `gemini_default`，可共存于同一凭据文件且不会
  交叉解析。清除最后一个本地 Key 会删除 `cloud_credentials.json`，允许保留空目录。DeepSeek 连接
  测试只请求官方 `/models`；Gemini 连接测试会先列模型，再用 32×32 红/蓝合成图执行一次很小的
  原生多模态 + strict JSON 探针，验证的不只是“列表可见”。重新打开弹窗不会回填或返回旧 Key。
- 本版按易用性取舍使用**本机明文 JSON**。它不会进入节点、workflow、history、PNG/MP4 元数据、
  Git 仓库或节点报告，但复制整个 `ComfyUI/user` 目录、开放 ComfyUI 服务或让他人读取本机文件时仍可能
  暴露。弹窗和本说明均明确显示真实保存路径；需要更强保护的用户可继续使用环境变量。
- 读取优先级为 `DEEPSEEK_API_KEY` 环境变量 > `deepseek_default` 本地凭据。环境变量存在时，弹窗保存的
  本地 Key 仍会落盘但不会生效，状态区会明确提示；清除本地 Key 不会删除 Windows 环境变量。
- 官方 `api.deepseek.com` 使用顶层 `thinking` / `reasoning_effort`；不会收到 LM Studio/Qwen 的
  `chat_template_kwargs`。`auto` 使用低思考，显式 `on` 使用高思考，恢复重试会关闭思考。
- DeepSeek Chat 结构化输出固定使用 `response_format={"type":"json_object"}`，不再误发 OpenAI
  `json_schema`；完整 schema 仍进入 system，并由本地 JSON parser、Prompt IR compiler 和 validator
  做格式/语义双层校验。云端节点使用强制 JSON，不会因 400/422 静默降级为无结构输出。
- Gemini 使用 `x-goog-api-key`、原生 `systemInstruction + contents`、图片 `inlineData`、视频
  `inlineData/fileData + videoMetadata` 与
  `responseMimeType + responseJsonSchema`；不会把 system role 混入 contents。通用“关闭思考”在
  Gemini 3 上映射为模型支持的最低级别（多数 Flash/Lite 为 `minimal`，3.7 Flash/3.1 Pro 为 `low`），
  因为这些模型不保证完全关闭 thinking。
- 供应商理论输出能力与本项目 Prompt IR 预算分开计算。旧通用导演默认 8192，云端导演默认 16384；
  DeepSeek、GLM、Claude、豆包及 OpenAI 兼容连接当前运行上限为 16384，Gemini 预设上限为 65536。
  旧工作流中 `131328`/`131072` 一类大值会按当前 provider 在运行时收敛并写入报告，不再照单全收。
- 旧的通用 `MiniMaxH3PromptDirector` 仍保留兼容用 `API 密钥` widget；其中明文填写的值会进入
  workflow、history，并可能随输出元数据写入 MP4/PNG。云端 DeepSeek 请改用 CloudDirector 的本地
  凭据弹窗或对应供应商环境变量（`DEEPSEEK_API_KEY` / `GEMINI_API_KEY`），不要因密码框显示圆点就把
  通用节点工作流当作已隔离。
- 外传工作流或成片前可运行只读扫描器（只报告文件与命中规则，不打印密钥值）：

```powershell
python tools/scan_embedded_secrets.py workflow.json output.mp4
```

如 MP4 命中，应先轮换已经暴露的密钥，再用 ffmpeg 生成无元数据副本，例如
`ffmpeg -i input.mp4 -map_metadata -1 -c copy safe.mp4`；扫描器本身不会修改原文件。

### 云端传输重试与实验指纹

- 首次发送前只序列化一次 JSON，并记录 `request_sha256` 与请求体 MiB。EOF、连接重置、响应不完整、
  可重试 SSL 断连、HTTP 408/429/5xx 最多补发一次；补发复用完全相同的 `body_bytes`。
- 429 优先读取 `Retry-After`，其余使用短指数退避与抖动。401/402/413/422 等不会原包重试；
  413 应回到帧数/压图预算处理，证书验证失败应修证书而不是重试。
- report 记录 prompt/system/module/media manifest 的 SHA-256，以及每次最终调用的 request SHA-256、
  payload MiB 和 transport attempt 数；不保存明文 prompt、图片 base64 或 Key。相同指纹证明客户端
  scaffold 一致，但不能证明供应商模型版本或采样结果确定。
- 网络重试与 JSON/Prompt IR 恢复是两类操作：前者请求体不变，后者会生成新的 request fingerprint；
  即使原包相同，供应商未提供幂等键时仍可能重复计费，也不能保证两次输出完全相同。

如需在送 API 前人工预览：`VHS_LoadVideo → VHS_SelectImages → Preview Image`；
例如 124 帧可选择 `0,41,82,123`。若不想手选，直接把完整 IMAGE batch 接入导演节点即可。

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

GitHub Release 提供两个资产：

- `MiniMaxH3-Lab-vX.Y.Z.zip`：完整插件，普通用户下载这个。
- `MiniMaxH3-Lab-Prompt-Modules-Only-vX.Y.Z.zip`：只含公开 `prompt_modules/` 的规则库覆盖包，**不能单独作为插件安装**；仅用于给同版本完整插件覆盖更新规则。`prompt_modules/user/` 中的个人 JSON 和 WF 本地副本内容不会进入该资产。

## 测试

从 **ComfyUI 根目录**运行（不要在插件目录内部直接收集，因为 pytest 会把带连字符的插件目录误当成顶层包）：

```bash
python -m pytest custom_nodes/ComfyUI-MiniMaxH3-Steering/tests -q --import-mode=importlib
```

当前发布副本：`245 passed`（ComfyUI 内置 Python + ComfyUI 根目录，2026-09-13；环境自带弃用警告不属于测试失败）。

## 许可证与合规

- 本仓库代码：**GPL-3.0-only**（见 LICENSE）。
- **模型权重与编码器**：使用 MiniMax-H3 官方权重 / 文本编码器时，请遵守 [MiniMax-H3 社区许可证](https://huggingface.co/MiniMaxAI/MiniMax-H3)。本仓库**不包含任何权重**，方向数据也由用户自行测量生成。
- 第三方参考声明见 THIRD_PARTY_NOTICES.md（OpenCache 概念参考 `lihaoyun6/ComfyUI-MiniMaxH3-Cache`，均为新实现）。

## 安全说明

- **模型列表刷新接口**（`/minimaxh3lab/api/models`）：仅在本地 ComfyUI 服务上监听；出站目标限制为 http/https，并拒绝链路本地（169.254.0.0/16 云元数据等，含 IPv4-mapped IPv6 形式）与多播地址。回环（127.0.0.1，如本地 LM Studio）与局域网地址按用途放行——请勿将 ComfyUI 端口暴露到不可信网络。
- **云端模型列表接口**（`/minimaxh3lab/cloud/models`）：内置 provider 使用固定 adapter/credential ID；
  OpenAI 兼容入口可使用用户明确填写或已保存的 Base URL，并复用链路本地/多播地址防护。响应不含 Key
  或供应商原始响应正文。DeepSeek 返回账户可见 ID，Gemini 只保留明确支持 `generateContent` 的条目；
  静态候选、账户可见与本插件实测不是同一证据等级。
- **API 密钥**：LocalDirector 兼容路径仍可从环境变量或旧节点输入读取；CloudDirector 优先使用供应商
  环境变量，其次使用 `ComfyUI/user/default/MiniMaxH3-Lab/cloud_credentials.json` 的本机凭据。Key 不写
  日志、不随仓库分发，也不由模型列表/状态接口返回前端。
- 本仓库不包含任何模型权重与方向数据；方向向量由 `tools/measure_directions.py` 在本地自行测量生成。

## 兼容性

- 当前实测环境：ComfyUI v0.30.x 开发线、Windows、Python 3.11。
- Prompt IR 编译/校验层为纯 Python 标准库逻辑，不依赖 Triton、CUDA 架构或第三方 wheel；PromptDirector 的图像/API 路径使用 ComfyUI 已带的 torch/numpy/Pillow。
- Linux 与 Python 3.12/3.13 按实现应可用，但尚未完成真机矩阵测试；不把“理论兼容”写成“已经验证”。
