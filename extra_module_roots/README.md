# 用户提示词模块库

在本目录下按用途创建子文件夹，并把自建 JSON 模块放进去。例如：

```text
extra_module_roots/
└── 镜头/
    ├── 固定机位.json
    └── 慢速推进.json
```

`MiniMaxH3PromptModuleFileLoader` 会递归显示为
`用户库/镜头/固定机位.json`。一个节点只选择一个文件。需要组合多个文件时可以复制节点并拼接 `system_prompt_module`；如果还需要 PromptDirector 完整追踪全部 manifest，建议把这组模块放进同一个 JSON 数组 pack，由一个节点加载。

文件可以是一个模块对象，也可以是模块对象数组。每个模块至少需要：

```json
{
  "id": "my_fixed_camera",
  "title_zh": "我的固定机位",
  "version": 1,
  "scope": "全部",
  "source": "user_local",
  "evidence_level": "experimental",
  "features": ["camera"],
  "conflicts": [],
  "instructions": "保持固定机位，不新增镜头。"
}
```

注意：

- 修改已有 JSON 内容会在下次排队时重读。
- 新增、删除或改名后刷新浏览器节点定义；若列表仍未更新，重启 ComfyUI。
- 单文件上限 512 KiB，最多读取 64 个模块，合并输出上限 6000 字符。
- `protocol_base`、`protocol_ref` 及旧版规范模块不会从这里注入；核心协议仍由 PromptDirector 按任务类型自动加载。
- 目录外路径、越界路径与受控根目录外的链接目标不会被加载。
- 本目录的用户内容默认被 Git 忽略，更新代码前仍建议自行备份模板库。
