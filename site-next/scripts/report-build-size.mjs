#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.resolve(__dirname, '../dist');
const LIMIT_BYTES = 900 * 1024 * 1024;

function collect(dir, files = []) {
  if (!fs.existsSync(dir)) return files;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) collect(full, files);
    else if (entry.isFile()) files.push(full);
  }
  return files;
}

const files = collect(DIST);
const bytes = files.reduce((sum, file) => sum + fs.statSync(file).size, 0);
console.log(`Final dist size: ${(bytes / 1024 / 1024).toFixed(2)} MB across ${files.length} files`);
if (bytes >= LIMIT_BYTES) {
  console.error('Final dist exceeds the 900 MB safety limit.');
  process.exitCode = 1;
}
