# 贡献指南

感谢帮助改进 MiniMaxH3-Lab。提交改动前，请先确认工作流、日志、截图和测试夹具不含 API Key、Authorization header、私人素材或本机用户目录。

## 开发与测试

把仓库放在 `ComfyUI/custom_nodes/` 下。测试应从 ComfyUI 根目录运行，避免带连字符的插件目录被 pytest 当成普通顶层包导入：

```bash
python -m pytest -q --import-mode=importlib custom_nodes/ComfyUI-MiniMaxH3-Steering/tests
```

提交前至少完成：

```bash
git diff --check
python -m pytest -q --import-mode=importlib custom_nodes/ComfyUI-MiniMaxH3-Steering/tests
node --check custom_nodes/ComfyUI-MiniMaxH3-Steering/web/minimax_h3_zh.js
```

## 创作规则贡献

- 正式规则放在 `prompt_modules/builtin/`；个人规则放在被 Git 忽略的 `prompt_modules/user/<自定义目录>/`；不要向 `legacy/` 添加新规则。
- 保持稳定、小写下划线 `id`，不要用显示名称或文件名充当兼容标识。
- 规则只描述创作语义、素材职责与保留事实；字段顺序、时间码、素材编号和官方标签由 Protocol、Compiler 与 Media Registry 处理。
- 新规则需要说明作用域、主要 ownership、依赖/冲突、证据等级，并补相应解析或回归测试。
- 不要用单次生成结果宣称某个模型、供应商或渲染档更优；质量结论需要固定素材与参数的多轮 A/B。

## Pull Request 边界

- 一个 PR 聚焦一个可解释主题，避免把 Provider、规则正文、前端迁移和发布文档混成不可审查的大提交。
- 描述验证环境、测试命令、结果和已知限制。
- 涉及旧工作流 widget 布局时必须提供迁移测试。
- 涉及联网请求时必须覆盖凭据隔离、URL 校验、安全错误映射和“不因加载工作流自动联网”。
