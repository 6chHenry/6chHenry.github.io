import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { spawn } from "node:child_process";
import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

const VAULT_ROOT = 'E:/WritingVault';
const LOG_FILE = 'E:/WritingVault/.tmp/pi-lark-sync.log';

function appendLog(message: string) {
  mkdirSync(dirname(LOG_FILE), { recursive: true });
  appendFileSync(LOG_FILE, `[${new Date().toISOString()}] ${message}\n`, 'utf8');
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

      const daily = await runNpmScript('daily:from-inbox');
      const dailyOutput = `${daily.stdout}\n${daily.stderr}`;
      appendLog(`daily:from-inbox exit=${daily.code ?? 'unknown'}\n${dailyOutput.trim()}`);

      if (daily.code === 0) {
        ctx.ui.notify(`WritingVault 已同步 ${summary.captured} 条新灵感，并更新今日 daily`, 'success');
      } else {
        ctx.ui.notify(`WritingVault 已同步 ${summary.captured} 条新灵感，但 daily 更新失败：npm run daily:from-inbox exited ${daily.code ?? 'unknown'}`, 'warning');
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
