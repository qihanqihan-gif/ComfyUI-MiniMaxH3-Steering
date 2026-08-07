# H3 提示词编写规范（官方 Skill 核心提取）

> 本文档从 MiniMax 官方 `MiniMax-H3` 仓库的 `skills/h3-prompt-writing`（base/ref 模板）提取浓缩，是 H3 视频提示词的**协议级规范**。插件中由「核心协议」模块自动加载（T2VA/I2VA/FL2VA → 三段式；Ref2VA → 六段式），本文档供手写与学习使用。

## 一、任务模式（五模式）

| 模式 | 全称 | 说明 |
|---|---|---|
| T2VA | Text-to-Video | 纯文本生成 |
| I2VA | Image-to-Video | 首帧/多帧引导 |
| FL2VA | First-and-Last-to-Video | 首尾帧路径 |
| L2VA | Last-to-Video | 尾帧倒推 |
| Ref2VA | Reference-to-Video | 参考素材（人物/物体/视频/音频） |

## 二、三段式结构（base：T2VA / I2VA / FL2VA / L2VA）

输出必须按固定顺序包含三个字段：

```
integrated_multimodal_description: 主字段——沿时间线的画面/动作/镜头/说话人/对白/剧情声
overall_soundscape: 全程环境声/动作声/非语言人声概括
non_diegetic_music: 仅观众可听的配乐
```

### 2.1 镜头标记与时间码

- `[Shot 1]` 首个镜头**不写时间戳**；后续镜头 `[Shot 2] At 00:01.500`（MM:SS.mmm）
- 时间码严格递增，不得超过总时长
- 镜头切换用语：`cuts to` / `transitions to` / `[cutoff]`

### 2.2 运镜三要素（类型 + 幅度 + 速度）

12 种运镜类型：`Zoom / Push / Pan / Truck / Tilt / Pedestal / Arc / Tracking / Static / Shake / POV / Roll`

写法示例：
- `a slow push-in toward her face`
- `a quick pan across the room`
- `static camera with subtle handheld breathing`

### 2.3 说话人与对白

- 说话人稳定 ID：`(S1)`、`(S2)`，跨镜头不换
- 对白保留原语言并标注：`(S1): <d>[EN] "Hello, how are you?" verbatim</d>`
- 画外旁白注明 `lips remain completely closed`
- 跨镜头连续说话：`<scenetrans>` / `<cutoff>`

### 2.4 屏幕文字

- 屏幕/UI 文字保留原语言，放入**英文双引号**：`A red neon sign reading "营业中"`
- 写明位置/样式/颜色

### 2.5 声音字段（overall_soundscape）

- 1-4 句一段，概括全程环境声
- 仅当完全静音才写 `N/A`

### 2.6 配乐字段（non_diegetic_music）

- 1-3 句，写乐器/速度/节奏/动态（如 `a gentle lo-fi piano track with a relaxed 70 BPM beat`）
- **禁用抽象情绪词**（不写 "epic music"，写具体构成）
- 剧情内音乐（diegetic）写进主字段，不写这里；禁止 BGM 时写 `N/A`

## 三、六段式结构（ref：Ref2VA）

固定顺序六个部分：

1. **subject_definitions**：定义引用内容与标签。四类标签独立编号：
   - `<Subject 1>` = 可复用可见内容（人物/物体/场景）
   - `<Picture 1>` = 帧锚点
   - `<Video 1>` = 视频资产
   - `<Audio 1>` = 音频资产
   - 编号跨段恒定（`<Subject 1>` 全文同义）
2. **summary**：以 `[任务类型]` 开头——`keyframe completion` / `reference generation` / `video editing` / `video continuation` / `audio reuse` / `audio reference`（可 `+` 组合）
3. **retention_analysis**：每标签一行，标记保留关系：
   - 视觉：`fully_preserved` / `partially_preserved` / `attribute_transfer` / `weak_reference`
   - 音频：`fully_copy` / `partially_copy` / `reference` / `weak_reference`
4. **detailed_description**：350-500 英文词，按播放顺序逐镜描述；风格总述句在 `[Shot 1]` 之前；标签插入首次出现处与作用点
5. **overall_soundscape** / 6. **non_diegetic_music**：规则同三段式

**核心分工**：参考图管身份（视觉通道），提示词管动作/时间线——不要重复描述参考图内容。

## 四、官方 8 个场景配方（浓缩）

| 配方 | 核心要求 |
|---|---|
| 3D 动画短片 | Pixar 风渲染、Q 版 2.5-3 头身、挤压拉伸表演；负面：写实/塑料皮肤/僵硬 |
| 品牌宣传 | 品牌资产溯源、单一宣传焦点、产品能力可视化、CTA 收尾 |
| 合作游戏开场 | 双角色+玩家卡、菜单交互动效、节拍对齐 |
| 手绘发光×实拍 | 粗糙发光笔触、物理接触真实感、延迟手持追逐、15s 16:9 |
| 极简产品广告 | 精炼卖点文案、产品锚点、节拍同步排版、留白 |
| MV 字幕 | 歌词逐字排版、节拍反应、角色/场景/文字分离 |
| 纸拼贴讲解 | 半调网点拼贴、视觉隐喻、纸声效（默认无 BGM/旁白） |
| 纸艺定格讲解 | 剪纸/立体书/分层 diorama、定格感、纸声设计 |

## 五、通用写作纪律（官方 + 社区 + 实测经验）

1. **具体大于抽象**：写「50mm prime lens, shallow depth of field」不写「beautiful shot」；写「gentle lo-fi piano, 70 BPM」不写「epic music」
2. **幅度与速度必限定**：任何运镜/动作带 slow/quick/gentle/rapid 级
3. **负面约束成句**：`no X, no Y` 英文短句，针对漂移/变形/闪烁
4. **身份走参考图**：Ref2VA 提示词不重复描述参考图内容
5. **对白/文字保留原语言**：只有描述性正文用英文
6. **时长对齐**：分镜时间码总和 ≤ 视频总时长
