# Skill → 创作规则模块工作单

## 1. 来源

- 来源名称：
- 来源类型：官方协议 Skill / 官方场景 Skill / 社区 Skill / 项目实验
- URL 或本地文档：
- commit / 版本 / 读取日期：
- 许可证或引用边界：

## 2. 目标模块

- 稳定 `id`：
- `title_zh`：
- 一句话目的：
- 不适用场景：
- 预计 `scope`：
- 证据等级：

## 3. 四向分拣

### 保留为一次提示词语义

- （填写）

### 转成 metadata

- scope：
- ownership：
- resolution：
- evidence：

### 路由到其他层

- Protocol / Compiler：
- Media Registry：
- 节点参数：
- Validator：
- 连接管理：

### 从模块删除

- 用户提问 / 选择卡：
- 浏览 / 下载 / 素材生成：
- 模型选择 / API 调用：
- 审批 / 等待 / 剪辑 / 交付：

## 4. Canonical semantic contract

```json
{
  "purpose": "",
  "adds": [],
  "preserves": [],
  "forbids": [],
  "defaults": []
}
```

## 5. Ownership

```json
{
  "reads": [],
  "writes": [],
  "augments": []
}
```

主要写入路径为何只能由本模块主导：

## 6. Resolution

```json
{
  "exclusive_group": "",
  "priority": 50,
  "requires": [],
  "conflicts": []
}
```

冲突胜负与理由：

## 7. Instructions 草案

目标：约 180–600 个中文字符；只表达当前调用需要的规则，不重复核心协议。

```text

```

## 8. 回归样本

| 样本 | 输入概要 | 预期模块状态 | 预期 Prompt IR 变化 | 不允许的变化 |
|---|---|---|---|---|
| 正常 |  | applied |  |  |
| scope 不匹配 |  | suppressed | 无 |  |
| 冲突 |  | applied/suppressed |  |  |
| 素材缺失 |  | warning/fail |  | 不虚构标签 |
| 关闭模块 |  | absent | 无残留 |  |

## 9. 审查结论

- [ ] 单一主要职责
- [ ] 与 Protocol/Compiler 无重复
- [ ] 未保留 Agent 工作流步骤
- [ ] contract 和 instructions 语义一致
- [ ] ownership 与 resolution 一致
- [ ] 来源与 evidence 等级匹配
- [ ] JSON、Resolver、Prompt IR 与编译测试通过
- [ ] 视频 A/B 尚未运行 / 已运行并附实验指纹
