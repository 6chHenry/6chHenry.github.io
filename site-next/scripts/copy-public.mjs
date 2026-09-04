#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '../..');
const DOCS = path.join(ROOT, 'docs');
const PUBLIC = path.join(ROOT, 'site-next/public');

const COLLECTIONS = {
  notes: ['notes'],
  essay: ['essay', 'summary'],
  projects: ['projects'],
  gallery: ['gallery'],
};

function copyDir(src, dest) {
  if (!fs.existsSync(src)) {
    console.warn(`Skip missing directory: ${src}`);
    return;
  }
  fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const from = path.join(src, entry.name);
    const to = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDir(from, to);
      continue;
    }
    try {
      if (fs.existsSync(to)) {
        const srcStat = fs.statSync(from);
        const destStat = fs.statSync(to);
        if (srcStat.size === destStat.size && srcStat.mtimeMs <= destStat.mtimeMs) {
          continue;
        }
      }
      fs.copyFileSync(from, to);
    } catch (error) {
      console.warn(`Skip locked or unreadable file: ${from} (${error.code ?? error.message})`);
    }
  }
}

function copyFile(src, dest) {
  if (!fs.existsSync(src)) return;
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.copyFileSync(src, dest);
}

function copyAssetDirs() {
  for (const collection of Object.keys(COLLECTIONS)) {
    if (collection === 'gallery') continue;
    for (const sourceDirName of COLLECTIONS[collection]) {
      const sourceRoot = path.join(DOCS, sourceDirName);
      if (!fs.existsSync(sourceRoot)) continue;
      for (const entry of fs.readdirSync(sourceRoot, { withFileTypes: true, recursive: true })) {
        if (!entry.isDirectory() || !entry.name.endsWith('.assets')) continue;
        const from = path.join(entry.parentPath ?? entry.path, entry.name);
        const rel = path.relative(sourceRoot, from);
        const to = path.join(PUBLIC, 'assets', collection, rel);
        copyDir(from, to);
      }
    }
  }
}

/* 笔记里散置的图片（不在 .assets 文件夹内的）也按同一路径约定复制，
   供 `/assets/notes/<相对路径>` 形式的引用使用 */
const NOTE_IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp']);

function copyLooseNoteImages() {
  const sourceRoot = path.join(DOCS, 'notes');
  if (!fs.existsSync(sourceRoot)) return;
  for (const entry of fs.readdirSync(sourceRoot, { withFileTypes: true, recursive: true })) {
    if (!entry.isFile()) continue;
    const parentName = path.basename(entry.parentPath ?? entry.path);
    if (parentName.endsWith('.assets')) continue;
    if (!NOTE_IMAGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) continue;
    const from = path.join(entry.parentPath ?? entry.path, entry.name);
    const rel = path.relative(sourceRoot, from);
    copyFile(from, path.join(PUBLIC, 'assets', 'notes', rel));
  }
}

function main() {
  copyDir(path.join(ROOT, 'overrides/img'), path.join(PUBLIC, 'img'));
  copyDir(path.join(ROOT, 'docs/audio'), path.join(PUBLIC, 'audio'));
  copyDir(path.join(ROOT, 'docs/academy.assets'), path.join(PUBLIC, 'academy.assets'));
  copyFile(path.join(ROOT, 'site-next/icon.png'), path.join(PUBLIC, 'academy.assets/academy-avatar.png'));
  copyDir(path.join(ROOT, 'assets'), path.join(PUBLIC, 'assets'));
  copyAssetDirs();
  copyLooseNoteImages();
  copyFile(path.join(ROOT, 'docs/academy.md'), path.join(PUBLIC, 'academy-legacy.md'));
  console.log('Public assets copied to site-next/public');
}

main();
