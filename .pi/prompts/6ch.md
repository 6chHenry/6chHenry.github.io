---
description: 6ch 个人网站与 WritingVault 工作流
argument-hint: "<capture|daily|weekly|private|publish> [target]"
---

你正在协助我的个人网站与私人写作工作流。请按 `$1` 决定动作，参数为 `${@:2}`。

## 固定路径

- 公开网站仓库：`F:\EECS498\6ch`
- 公开 Markdown 源头：`F:\EECS498\6ch\docs`
- Astro 项目：`F:\EECS498\6ch\site-next`
- 私人写作库：`E:\WritingVault`

## 总原则

1. 默认所有新捕捉内容都是私人内容。
2. 私人内容不得复制到公开网站仓库，除非我明确说“发布/公开”。
3. 公开发布前必须检查隐私信息。
4. 发布网站时只 stage 本次相关文件，避免混入既有脏改动。
5. 文字润色要保留我的语气，不要改成 AI 腔。

## 动作说明

### capture

当我输入 `/6ch capture <内容>`：

1. 把内容追加到 `E:\WritingVault\inbox\YYYY-MM-DD.md`。
2. 如果内容来自语音转文字，先只做轻微断句，不要重写。
3. 不进入公开网站。
4. 追加后提示我是否需要整理成 card 或 daily。

可用命令：

```bash
cd E:/WritingVault && npm run capture -- "${@:2}"
```

### daily

当我输入 `/6ch daily [日期]`：

1. 读取 `E:\WritingVault\inbox\日期.md`。
2. 生成或更新 `E:\WritingVault\daily\日期.md`。
3. 保留原始记录。
4. 输出：我理解你的意思、今天学到的东西、重要想法、文章种子、公开/私人分流建议。

### weekly

当我输入 `/6ch weekly`：

1. 读取最近 7 天的 `inbox/` 和 `daily/`。
2. 生成 `weekly/YYYY-WW.md`。
3. 提取反复出现的主题、可发展选题、下周追问。

### private

当我输入 `/6ch private <文件或主题>`：

1. 将相关内容整理为私人文章。
2. 保存到 `E:\WritingVault\private-essays\`。
3. 可以 Git commit 到私人库。
4. 不得进入公开网站仓库。

### bot

当我输入 `/6ch bot`：

1. 优先使用飞书 Bot，不默认使用微信个人号 Bot。
2. 飞书 Bot 本地服务在 `E:\WritingVault\scripts\feishu-webhook.mjs`。
3. 启动命令：

```bash
cd E:/WritingVault && npm run feishu:webhook
```

4. 接入说明看：`E:\WritingVault\FEISHU_BOT.md`。
5. 如果要配置公网回调，先确认使用 Cloudflare Tunnel / ngrok / frp / 服务器反代中的哪一种。
6. 不要把飞书 App Secret、Verification Token、Encrypt Key 写进公开网站仓库。

### backup

当我输入 `/6ch backup`：

1. 检查 `E:\WritingVault` 是否已有 Git remote。
2. 如果没有 remote，先询问我要用哪个 private repo，不要擅自创建公开仓库。
3. 确认 repo 是 private 后，再执行首次 push。
4. 以后用于私人写作库的异地备份，不进入公开网站仓库。

推荐命令形态：

```bash
cd E:/WritingVault && git remote -v && git status --short
```

### publish

当我输入 `/6ch publish <文件>`：

1. 确认文件来自 `E:\WritingVault\publish-ready\` 或我明确指定为可公开。
2. 检查隐私信息。
3. 判断栏目并移动/复制到 `F:\EECS498\6ch\docs\...`。
4. 在 `site-next/` 运行构建：

```bash
cd F:/EECS498/6ch/site-next && npm run build
```

5. 如适合，使用现有发布脚本：

```bash
cd F:/EECS498/6ch/site-next && npm run publish-content -- --message "Publish content: 标题"
```

6. 只提交本次文章及迁移/构建生成的相关文件。

## 如果 `$1` 为空

先询问我要执行哪一个动作：`capture`、`daily`、`weekly`、`private`、`publish`。
