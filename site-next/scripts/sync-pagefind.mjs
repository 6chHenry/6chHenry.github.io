#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const SOURCE = path.join(ROOT, 'dist/pagefind');
const TARGET = path.join(ROOT, 'public/pagefind');

function copyDir(from, to) {
  fs.mkdirSync(to, { recursive: true });
  for (const entry of fs.readdirSync(from, { withFileTypes: true })) {
    const srcPath = path.join(from, entry.name);
    const destPath = path.join(to, entry.name);
    if (entry.isDirectory()) {
      copyDir(srcPath, destPath);
      continue;
    }
    fs.copyFileSync(srcPath, destPath);
  }
}

function main() {
  if (!fs.existsSync(SOURCE)) {
    console.warn('Skip pagefind sync: dist/pagefind not found (run npm run build first).');
    return;
  }

  if (fs.existsSync(TARGET)) {
    fs.rmSync(TARGET, { recursive: true, force: true });
  }

  copyDir(SOURCE, TARGET);
  console.log('Synced pagefind index to public/pagefind for local dev.');
}

main();
