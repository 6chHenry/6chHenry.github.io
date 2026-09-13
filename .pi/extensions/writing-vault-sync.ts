import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const VAULT_ROOT = 'E:/WritingVault';
const LOG_FILE = 'E:/WritingVault/.tmp/pi-lark-sync.log';
const MANAGED_START = '<!-- daily-from-inbox:start -->';
const MANAGED_END = '<!-- daily-from-inbox:end -->';

function appendLog(message: string) {
  mkdirSync(dirname(LOG_FILE), { recursive: true });
  appendFileSync(LOG_FILE, `[${new Date().toISOString()}] ${message}\n`, 'utf8');
}

function localDate(date = new Date()) {
  return [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, '0'),
    String(date.getDate()).padStart(2, '0'),
  ].join('-');
}

function safeFilePart(text: string) {
  return text.replace(/[\\/:*?"<>|]/g, '').replace(/\s+/g, '-').slice(0, 80) || '未命名卡片';
}

function escapeYaml(text: string) {
  return JSON.stringify(text || '');
}

function runNpmScript(script: string): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const child = spawn('npm', ['run', script], {
      cwd: VAULT_ROOT,
      shell: process.platform === 'win32',
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        LARKSUITE_CLI_NO_UPDATE_NOTIFIER: '1',
        LARKSUITE_CLI_NO_SKILLS_NOTIFIER: '1',
      },
    });

    let stdout = '';
    let stderr = '';
    child.stdout?.setEncoding('utf8');
    child.stderr?.setEncoding('utf8');
    child.stdout?.on('data', (chunk) => {
      stdout += chunk;
    });
    child.stderr?.on('data', (chunk) => {
      stderr += chunk;
    });
    child.on('close', (code) => {
      resolve({ code, stdout, stderr });
    });
  });
}

function parseSyncSummary(output: string) {
  const match = output.match(/\[lark-sync\] done: captured=(\d+), skipped=(\d+), total=(\d+)/);
  if (!match) return null;
  return {
    captured: Number(match[1]),
    skipped: Number(match[2]),
    total: Number(match[3]),
  };
}

function upsertManagedBlock(existing: string, block: string) {
  if (existing.includes(MANAGED_START) && existing.includes(MANAGED_END)) {
    const pattern = new RegExp(`${MANAGED_START}[\\s\\S]*?${MANAGED_END}`);
    return existing.replace(pattern, block.trim());
  }
  return `${existing.trimEnd()}\n\n${block.trim()}\n`;
}

function makeDailyScaffold(date: string) {
  return `---\ntype: daily-review\ndate: ${date}\nvisibility: private\n---\n\n# ${date} 每日整理\n\n`;
}

function extractJson(text: string) {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i)?.[1];
  const raw = (fenced || text).trim();
  try {
    return JSON.parse(raw);
  } catch {
    const start = raw.indexOf('{');
    const end = raw.lastIndexOf('}');
    if (start >= 0 && end > start) return JSON.parse(raw.slice(start, end + 1));
    throw new Error('AI response is not valid JSON');
  }
}

function normalizeDailyMarkdown(markdown: string) {
  const cleaned = String(markdown || '').trim();
  if (!cleaned) throw new Error('AI dailyMarkdown is empty');
  return `${MANAGED_START}\n\n${cleaned}\n\n${MANAGED_END}`;
}

function makeCardMarkdown(date: string, card: { title?: string; tags?: string[]; markdown?: string }) {
  const title = String(card.title || '未命名卡片').trim();
  const tags = Array.isArray(card.tags) ? card.tags.map(String) : [];
  const body = String(card.markdown || '').trim();
  return `---\ntype: thought-card\ntitle: ${escapeYaml(title)}\ndate: ${date}\nsource: inbox/${date}.md\nvisibility: private\ntags: ${JSON.stringify(tags)}\nstatus: seed\ngeneratedBy: pi-ai-daily\n---\n\n# ${title}\n\n${body}\n`;
}

async function generateAiDailyAndCards(ctx: any) {
  if (!ctx.model) throw new Error('No active Pi model available');
  if (!ctx.modelRegistry.hasConfiguredAuth(ctx.model)) throw new Error(`No authentication configured for ${ctx.model.provider}/${ctx.model.id}`);

  const date = localDate();
  const inboxFile = join(VAULT_ROOT, 'inbox', `${date}.md`);
  if (!existsSync(inboxFile)) throw new Error(`Inbox file not found: ${inboxFile}`);

  const inbox = readFileSync(inboxFile, 'utf8').trim();
  if (!inbox || !inbox.includes('## ')) throw new Error(`No inbox entries for ${date}`);

  const prompt = `你是刘可唯的私人写作整理助手。请只基于下面的 inbox 内容，生成每日深度整理和思想卡片候选。\n\n要求：\n1. 用自然中文，像懂他的朋友和编辑，不要机械摘要。\n2. daily 要包含：今日主线、我理解你的意思、共情理解、犀利但温柔的提醒、可行动的小步、可以发展成文章的种子、公开/私人分流建议、明天追问。\n3. 不要把私人内容建议公开；默认私人。\n4. Cards 只生成真正值得沉淀的思想卡片，0 到 3 张。不要为了生成而生成。\n5. 每张 card 的 markdown 必须包含这些二级标题：原始记录、我理解你的意思、共情理解、犀利提醒、分流判断、可以展开的方向、下一步问题。\n6. 输出必须是 JSON，不要 Markdown fence，不要解释。\n\nJSON 结构：\n{\n  "dailyMarkdown": "不含 frontmatter 和一级标题，只写托管块里的 Markdown",\n  "cards": [\n    { "title": "卡片标题", "tags": ["标签"], "markdown": "卡片正文，不含 frontmatter 和一级标题" }\n  ]\n}\n\n日期：${date}\n\nINBOX：\n${inbox}`;

  const response = await ctx.modelRegistry.complete(
    ctx.model,
    {
      systemPrompt: '你是私人写作系统中的深度整理模块。你重视准确、克制、共情和边界感。只输出用户要求的 JSON。',
      messages: [
        {
          role: 'user' as const,
          content: [{ type: 'text' as const, text: prompt }],
          timestamp: Date.now(),
        },
      ],
    },
    {
      reasoningEffort: 'medium',
      cacheRetention: 'none',
      sessionId: randomUUID(),
    },
  );

  const text = response.content
    .filter((part: any): part is { type: 'text'; text: string } => part.type === 'text')
    .map((part: { text: string }) => part.text)
    .join('\n');

  const parsed = extractJson(text) as { dailyMarkdown?: string; cards?: Array<{ title?: string; tags?: string[]; markdown?: string }> };
  const dailyBlock = normalizeDailyMarkdown(parsed.dailyMarkdown || '');
  const dailyFile = join(VAULT_ROOT, 'daily', `${date}.md`);
  mkdirSync(dirname(dailyFile), { recursive: true });
  const existingDaily = existsSync(dailyFile) ? readFileSync(dailyFile, 'utf8') : makeDailyScaffold(date);
  writeFileSync(dailyFile, upsertManagedBlock(existingDaily, dailyBlock), 'utf8');

  const cards = Array.isArray(parsed.cards) ? parsed.cards.slice(0, 3) : [];
  const writtenCards: string[] = [];
  for (const card of cards) {
    const title = String(card.title || '').trim();
    if (!title || !String(card.markdown || '').trim()) continue;

    const file = join(VAULT_ROOT, 'cards', `${date}-${safeFilePart(title)}.md`);
    const content = makeCardMarkdown(date, card);
    if (existsSync(file)) {
      const existing = readFileSync(file, 'utf8');
      if (!existing.includes('generatedBy: pi-ai-daily')) {
        appendLog(`skip card because file exists and is not AI-generated: ${file}`);
        continue;
      }
    }
    mkdirSync(dirname(file), { recursive: true });
    writeFileSync(file, content, 'utf8');
    writtenCards.push(file.replace(/\\/g, '/'));
  }

  return { date, dailyFile: dailyFile.replace(/\\/g, '/'), cards: writtenCards };
}

export default function (pi: ExtensionAPI) {
  pi.on("session_start", async (event, ctx) => {
    if (!['startup', 'reload', 'new', 'resume'].includes(event.reason)) return;

    appendLog(`session_start reason=${event.reason}`);

    const sync = await runNpmScript('lark:sync');
    const syncOutput = `${sync.stdout}\n${sync.stderr}`;
    appendLog(`lark:sync exit=${sync.code ?? 'unknown'}\n${syncOutput.trim()}`);

    const summary = parseSyncSummary(syncOutput);
    if (sync.code === 0 && summary) {
      if (summary.captured <= 0) {
        ctx.ui.notify('WritingVault 没有新的飞书灵感需要同步', 'info');
        return;
      }

      try {
        const ai = await generateAiDailyAndCards(ctx);
        appendLog(`ai digest ok daily=${ai.dailyFile} cards=${ai.cards.length}`);
        ctx.ui.notify(`WritingVault 已同步 ${summary.captured} 条新灵感，AI 已更新 daily 和 ${ai.cards.length} 张卡片`, 'success');
      } catch (error) {
        appendLog(`ai digest failed: ${error instanceof Error ? error.stack || error.message : String(error)}`);
        const daily = await runNpmScript('daily:from-inbox');
        const dailyOutput = `${daily.stdout}\n${daily.stderr}`;
        appendLog(`fallback daily:from-inbox exit=${daily.code ?? 'unknown'}\n${dailyOutput.trim()}`);

        if (daily.code === 0) {
          ctx.ui.notify(`WritingVault 已同步 ${summary.captured} 条新灵感，但 AI 整理失败，已用本地模板兜底`, 'warning');
        } else {
          ctx.ui.notify(`WritingVault 已同步 ${summary.captured} 条新灵感，但 AI 和本地 daily 更新都失败`, 'warning');
        }
      }
      return;
    }

    const permissionHint = syncOutput.includes('权限不足') || syncOutput.includes('access denied') || syncOutput.includes('missing scope');
    if (permissionHint) {
      ctx.ui.notify('WritingVault 飞书同步失败：飞书应用权限不足，请先补消息读取权限。', 'warning');
      return;
    }

    ctx.ui.notify(`WritingVault 飞书同步失败：npm run lark:sync exited ${sync.code ?? 'unknown'}`, 'warning');
  });
}
