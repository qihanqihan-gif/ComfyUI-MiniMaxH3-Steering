# Changelog

本项目按发布版本记录面向用户的变化。尚未发布的开发内容归入 `Unreleased`。

## Unreleased

## 0.3.0-alpha - 2026-09-01

### Added

- 新增 CloudDirector 云端多模态导演、OpenAI-compatible 自定义连接、每预设凭据引用、模型发现和连接测试。
- 新增 Provider capability registry、受控图片/视频上下文、视频时间线 manifest 与请求指纹。
- 创作规则库扩展为 26 个可选规则，并加入作用域、依赖、冲突、互斥组和稳定优先级解析。
- 新增 `standard`、`compact`、`strong` 三种规则渲染档，以及模块集合和最终规则文本哈希。
- 新增用户自定义规则模板、模块作者文档、快速开始与回归测试指南，以及 A/B 记录模板。

### Changed

- 统一云端自定义连接和“我的预设”交互；本地 LM Studio 优先使用原生 JIT 路由。
- 参考素材标签由实际 Media Registry 生成；创作规则不再自行虚构或分配 Picture、Video、Audio、Subject 编号。
- 规则存储统一到 `prompt_modules/`：`builtin/` 为正式中文库、`legacy/` 为只读兼容层、`user/` 为个人规则库。

### Fixed

- 修复旧工作流中 CloudDirector widget 插入导致的值错位，并归一化非法或越界值。
- 修复凭据测试读取真实用户目录、模块数量断言漂移和云端错误正文可能泄漏请求内容的问题。
- 隐藏或阻断未证实的云端视觉预设，避免把静态候选误报为账户可用或已实测模型。

### Security

- API Key 只从环境变量或 ComfyUI 用户目录的本地凭据文件解析，不写入工作流、普通报告或仓库。
- 出站 URL 校验拒绝链路本地、云元数据和多播目标；供应商错误只映射为安全错误码与固定建议。
- 增加个人规则目录、凭据文件、环境文件、方向数据和发布归档的 Git 忽略规则。

## 0.2.0-alpha

- 建立 H3 提示词导演、Prompt IR 编译/校验、创作规则热加载与官方格式输出底座。

## 0.1.0-alpha

- 首次公开提示词导演、规则模块节点、中文前端和基础测试集。
