import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { spawn } from "node:child_process";
import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

const LOG_FILE = 'E:/WritingVault/.tmp/pi-lark-sync.log';

function appendLog(message: string) {
  mkdirSync(dirname(LOG_FILE), { recursive: true });
  appendFileSync(LOG_FILE, `[${new Date().toISOString()}] ${message}\n`, 'utf8');
}

export default function (pi: ExtensionAPI) {
  pi.on("session_start", async (event, ctx) => {
    if (!['startup', 'reload', 'new', 'resume'].includes(event.reason)) return;

    appendLog(`session_start reason=${event.reason}`);

    const child = spawn('npm', ['run', 'lark:sync'], {
      cwd: 'E:/WritingVault',
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
      const output = `${stdout}\n${stderr}`;
      appendLog(`exit=${code ?? 'unknown'}\n${output.trim()}`);
      const match = output.match(/\[lark-sync\] done: captured=(\d+), skipped=(\d+), total=(\d+)/);
      if (code === 0 && match) {
        const captured = Number(match[1]);
        const total = Number(match[3]);
        if (captured > 0) {
          ctx.ui.notify(`WritingVault 已同步 ${captured} 条新灵感（共检查 ${total} 条）`, 'success');
        } else {
          ctx.ui.notify('WritingVault 没有新的飞书灵感需要同步', 'info');
        }
        return;
      }

      const permissionHint = output.includes('权限不足') || output.includes('access denied') || output.includes('missing scope');
      if (permissionHint) {
        ctx.ui.notify('WritingVault 飞书同步失败：飞书应用权限不足，请先补消息读取权限。', 'warning');
        return;
      }

      ctx.ui.notify(`WritingVault 飞书同步失败：npm run lark:sync exited ${code ?? 'unknown'}`, 'warning');
    });
  });
}
