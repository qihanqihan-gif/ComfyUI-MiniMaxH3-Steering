# Prompt modules

创作规则统一放在本目录，插件根目录不再平铺多个相近的模块文件夹。

```text
prompt_modules/
├── builtin/   插件正式内置规则
├── legacy/    旧英文规则兼容层
└── user/      用户自定义规则
```

- `builtin/` 随插件更新。可以参考其 JSON 结构，但不要把个人文件放进去。
- `legacy/` 只用于兼容旧工作流和旧英文选择值，不要编辑或新增规则。
- `user/` 用于个人规则；请在其中新建自己的子目录。用户 JSON 默认被 Git 忽略，更新插件前仍建议自行备份。

从 `templates/用户简单创作规则模板.example.json` 开始最省事。完整字段、接线与冲突消解说明见 `docs/用户自定义创作规则.md`。
