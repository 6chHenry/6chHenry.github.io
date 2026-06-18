#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const cacheDir = path.resolve(__dirname, '../.astro');
const siteRoot = path.resolve(__dirname, '..');

if (cacheDir.startsWith(siteRoot)) {
  fs.rmSync(cacheDir, { recursive: true, force: true });
}
