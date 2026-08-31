# 从 Skill 到 MiniMaxH3-Lab 创作规则模块

本文说明如何把官方 Skill、社区 Skill、项目经验或一份类似 Skill 的长文档，转换成适合 MiniMaxH3-Lab 的创作规则 JSON。

官方 MiniMax-H3 仓库目前把内容分成 1 个提示词协议 Skill 与 8 个场景/风格生产 Skill。前者定义 H3 输出协议，后者通常是从需求确认到资产、生成、剪辑和交付的完整 Agent 工作流。两者不能用同一种方式直接塞进模块：

- [MiniMax H3 Skills 总览](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/README.md)
- [h3-prompt-writing](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing)

## Skill 和插件模块的区别

| Skill | 创作规则模块 |
|---|---|
| 决定何时触发、问什么、调用哪些工具 | 已由用户在节点中显式选择 |
| 可以浏览、读文件、生成资产、等待和迭代 | 不能调用工具，只为当前 Prompt Director 提供规则 |
| 可以管理项目、画布、审批和交付 | 只负责一次提示词规划中的有限语义 |
| 可以包含多阶段工作流 | 必须是可组合、可消解的原子规则或场景配方 |
| 正文就是 Agent 行为说明 | JSON contract 是事实源，instructions 是当前渲染文本 |

因此，转换不是“把 SKILL.md 缩短”，而是把其中**真正需要进入单次 H3 提示词规划的语义**抽出来。

## 先判断来源属于哪一类

### A. 协议 Skill

例如官方 `h3-prompt-writing`。它负责字段顺序、标签、时间码、对白、声音和模式差异，应进入自动 Protocol、Compiler 或 Validator，不做普通可选模块。

### B. 场景 / 风格生产 Skill

例如 3D 动画、品牌宣传、产品广告、MV 字幕、纸艺讲解。它通常混合了创作知识和 Agent 流程。只提取最终视频提示词必须知道的“单次生成语义”，其余流程删除或路由到其他节点。

### C. 原子技巧或社区经验

例如固定机位、缓慢推进、身份保持、循环、对白表演。可以成为原子模块，但必须说明证据等级、职责边界与冲突关系。案例中的偶然成功不能直接变成硬规则。

## 四向分拣：保留、转元数据、路由、删除

阅读来源时，把每条内容放入以下一类：

| 去向 | 内容 |
|---|---|
| 保留为 contract | 最终画面/动作/镜头/声音必须表达的语义；身份或产品必须保留的属性；真正影响一次生成的节拍与失败边界 |
| 转成元数据 | 适用任务→`scope`；主要职责→`ownership`；互斥/依赖→`resolution`；来源可靠性→`evidence_level` |
| 路由到其他层 | 字段顺序/标签语法→Protocol；时间码与首尾帧句式→Compiler；素材编号→Media Registry；时长/比例/帧数→节点输入；联网与 Key→连接管理 |
| 删除 | 触发词、选择卡、用户确认、画布节点、浏览/下载、生成参考图、模型选择、API 调用、等待、审批、剪辑、上传和交付步骤 |

如果一条规则离开 Skill 的工具和上下文就无法成立，不应伪装成 Prompt Module。

## 标准转换流程

### 1. 固定来源

记录来源 URL、文件名、读取日期或 commit。不要只写“官方 Skill”；同名 Skill 会演进。

### 2. 写目的和不适用场景

用两句话回答：这个模块解决什么？什么情况下不应选择？如果无法说清，来源可能需要拆成多个模块或只保留为文档。

### 3. 提取原子语义

把长段落改写为：

- `adds`：本模块新增什么。
- `preserves`：哪些已有事实不能被覆盖。
- `forbids`：哪些结果与任务目标不兼容。
- `defaults`：用户没说时才采用的弱默认值。

这里写模型无关的语义，不写“Gemini 要这样说”“DeepSeek 必须多重复一次”。

### 4. 分配所有权

- `writes`：模块主导的语义路径，数量尽量少。
- `augments`：只补充、不覆盖的路径。
- `reads`：模块需要查看但不能重写的已有信息。

两个模块若都写同一主路径，必须进入同一 `exclusive_group` 或声明成对冲突。

### 5. 声明消解关系

- `requires`：没有依赖规则时本模块不成立。
- `conflicts`：两个具体规则的意图不兼容。
- `exclusive_group`：多个候选都想成为同一语义域的主所有者。
- `priority`：只在竞争同一路径时比较，不能覆盖 Protocol 或用户硬要求。

不要把“冲突时听用户的”写进 instructions 交给 LLM 临场猜；Resolver 应在调用模型前完成选择。

### 6. 写 evidence

- `official_protocol`：仅官方协议定义。
- `official_skill_adaptation`：官方生产 Skill 的单次提示词语义改编。
- `official_aligned`：与官方规则一致，但包含项目自己的组织或推论。
- `community_heuristic`：社区经验，尚非官方保证。
- `experimental`：项目观察或待扩大样本的假设。

官方 Skill 中的某条工作流建议，不会因为来源官方就自动成为 H3 模型协议。

### 7. 从 contract 渲染 instructions

推荐顺序：

1. 一句目的与启用条件。
2. 主职责和必须保持的内容。
3. 素材在本规则中的作用，不重编号、不虚构标签。
4. 一次视频内部的动作/镜头/声音策略。
5. 仅保留与当前风险相关的失败边界。

不要在每个模块重复三段式/六段式字段顺序、`[Shot N]` 语法、声音字段定义或首尾帧编译句式。

### 8. 写回归工作单

至少准备：

- 应启用且应生效的正常样本。
- scope 不匹配时应被压制的样本。
- 与同一主路径模块冲突时的胜负样本。
- 缺少素材或标签时应提示/阻断的样本。
- 关闭模块后不应残留该模块语义的样本。

填写 `templates/Skill转规则模块工作单.example.md`，再提交 JSON。

## 三个官方 Skill 转换示例

### 品牌宣传 Skill

保留：品牌/产品事实不能编造，身份资产、颜色、包装、UI 与可见文字保持，短片只有一个清楚宣传焦点。

删除：浏览品牌站、下载媒体包、用户审批、生成资产、选择工具、后期拼接和交付流程。

路由：素材真实性由用户输入与素材检查负责；可见文字精确性可由 `ui_screen_text` 与 Validator 共同负责。

### 3D 动画短片 Skill

保留：当用户已经选择风格化 3D 方向时，角色比例、材质、表演方式、动作可读性和全片风格一致。

删除：项目简报、故事大纲、角色卡/场景卡制作、选择卡、模型选择、逐镜生成、BGM 合成和最终剪辑。

注意：官方 Skill 服务于完整中长流程；插件中的 `scene_3d_animation` 只是单次短片配方，不能声称完成整套制作。

### 极简产品广告 Skill

保留：产品主体颜色和材质是硬保真约束；参考图分别承担产品身份/细节/收束构图；节拍文字不能变成多格拼贴或把锚点图当连续分镜。

删除：开始门、风格选择卡、参考图生成、模型 fallback、独立音乐生成、节拍分析、音轨替换和交付检查。

注意：只有实际连接了对应素材时，模块才能引用它们；具体 `<Picture N>` 编号由 Media Registry 决定。

## 场景配方的两层互斥

完整 Skill 常同时规定交付类型与视觉媒介。转成插件规则时应分开判断：

1. 八个官方场景改编都属于 `scene.recipe.primary`，同一轮只能保留一个，避免“品牌宣传 + MV 字幕 + 纸艺讲解”被拼成含混混合模板。
2. 只有确实规定固定媒介的配方才同时写 `visual.medium.primary`。当前为 3D 动画、手绘×实拍、纸拼贴和纸艺定格；它们会与写实电影感等主媒介规则二选一。
3. 品牌宣传、游戏开场、极简产品广告和 MV 字幕只规定内容组织方式，不强绑画风，因此可以和独立视觉规则组合。
4. 镜头、时间线或声音若只是官方 Skill 的默认做法，应放入 `defaults` 或 `augments`，不要轻易取得新的主所有权；用户或其它原子规则明确指定时应能覆盖。

这套划分同样适用于用户自制场景模板：先问“它是在规定做什么，还是规定长什么样”，再决定是否需要第二个互斥组。

## 适合当前插件的 JSON 骨架

```json
{
  "id": "stable_english_id",
  "title_zh": "中文规则名",
  "version": 1,
  "scope": "I2VA,Ref2VA",
  "category": "camera",
  "source": "来源与版本",
  "source_url": "https://...",
  "evidence_level": "official_skill_adaptation",
  "features": ["one_primary_feature"],
  "semantic_contract": {
    "purpose": "一句话目的",
    "adds": [{"path": "camera.motion.primary", "value": "slow_push_in"}],
    "preserves": ["subject.identity"],
    "forbids": ["unmotivated_camera_motion"],
    "defaults": []
  },
  "ownership": {
    "reads": ["subject.identity"],
    "writes": ["camera.motion.primary"],
    "augments": ["continuity.composition"]
  },
  "resolution": {
    "exclusive_group": "camera.motion.primary",
    "priority": 50,
    "requires": [],
    "conflicts": []
  },
  "instructions": "由上述契约渲染出的简洁规则。"
}
```

当前 Resolver 已使用 scope、requires、conflicts、exclusive_group 和 priority；`semantic_contract` 是下一阶段统一渲染与更严格预检的事实源。旧式自由结构仍兼容，但新写模块应使用上面的 `purpose/adds/preserves/forbids/defaults` 形状。

## 发布前检查

- 来源许可证允许以当前方式引用或改编；不整段复制不必要的上游正文。
- `id` 不因中文改名或来源改名而变化。
- 模块不含本机路径、API Key、私人素材名或供应商凭据。
- 模块不承诺未经过回归验证的模型效果。
- 模块单独启用和与冲突模块组合时都有确定结果。
- `instructions` 删除后，contract 仍足以说明模块想做什么；否则说明契约还没有抽干净。
