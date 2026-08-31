# 安全策略

## 报告漏洞

请优先通过 GitHub 仓库的 Private vulnerability reporting / Security Advisory 私下报告安全问题。不要在公开 Issue、工作流、截图或日志中粘贴真实 API Key、Authorization header、凭据文件内容或带签名的私有 URL。

若暂时无法使用私密报告入口，可以先创建一条不含利用细节和秘密值的公开 Issue，请维护者建立私密沟通渠道。

报告中建议包含：

- 受影响的 commit 或版本。
- 最小复现步骤与影响范围。
- 是否涉及工作流序列化、日志、浏览器前端、本地凭据文件或出站 URL。
- 已做过的安全删除或凭据轮换；不要附上秘密原值。

## 支持范围

安全修复以最新公开 release 和默认分支为主。较早 alpha 版本可能只提供升级建议，不保证单独回补。

## 凭据边界

- CloudDirector 的 Key 应保存在环境变量或 ComfyUI 用户目录下的本地凭据文件中。
- 不要把 Key 写入节点 widget、工作流 JSON、模板、测试夹具、录屏、发布 ZIP 或 Git 历史。
- 怀疑秘密已经进入 commit 时，应先吊销或轮换凭据；仅删除当前文件不能从 Git 历史中移除泄漏。
- 不要把 ComfyUI 服务直接暴露到不可信网络；本地和局域网连接放行不等于远程部署安全加固。
