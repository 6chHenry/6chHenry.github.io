#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '../..');
const DOCS = path.join(ROOT, 'docs');
const CONTENT = path.join(ROOT, 'site-next/src/content');

const COLLECTIONS = {
  notes: ['notes'],
  essay: ['essay', 'summary'],
  projects: ['projects'],
  gallery: ['gallery'],
};

const SKIP_FILES = new Set(['index.md', 'math-test.md', 'academy.md']);

const HIDDEN_ESSAY_PATHS = new Set([
  '旅游记账',
  'Life/情感',
  'Life/复合之后',
  'Life/19岁后记',
  'Life/19生日',
  'Life/我的大一上学期',
  'Life/写给18岁的信',
  'Misc/test',
]);

const HIDDEN_SUMMARY_PATHS = [/^[^/]+$/, /^2025\//];

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function getSourceUpdatedAt(absPath, previousUpdatedAt) {
  if (process.env.GITHUB_ACTIONS && previousUpdatedAt) return previousUpdatedAt;
  return new Date(fs.statSync(absPath).mtime).toISOString();
}

function walkMarkdown(dir, files = []) {
  if (!fs.existsSync(dir)) return files;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name.endsWith('.assets')) continue;
      walkMarkdown(full, files);
    } else if (entry.isFile() && entry.name.endsWith('.md')) {
      files.push(full);
    }
  }
  return files;
}

function parseFrontmatter(raw) {
  const start = raw.match(/^---\r?\n/);
  if (!start) {
    return { data: {}, body: raw };
  }
  const endMatch = /\r?\n---\r?\n/.exec(raw.slice(start[0].length));
  if (!endMatch) return { data: {}, body: raw };

  const end = start[0].length + endMatch.index;
  const bodyStart = end + endMatch[0].length;
  const yaml = raw.slice(start[0].length, end);
  const body = raw.slice(bodyStart);
  const data = {};
  const lines = yaml.split('\n');

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const match = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (!match) continue;

    const [, key, value] = match;
    const trimmed = value.replace(/^['"]|['"]$/g, '');

    if (trimmed === '' && i + 1 < lines.length && lines[i + 1].match(/^\s+-\s+/)) {
      const items = [];
      let j = i + 1;
      while (j < lines.length) {
        const itemMatch = lines[j].match(/^\s+-\s+(.+)$/);
        if (!itemMatch) break;
        items.push(itemMatch[1].replace(/^['"]|['"]$/g, ''));
        j++;
      }
      if (items.length > 0) {
        data[key] = key === 'tags' ? items.join(',') : items;
        i = j - 1;
        continue;
      }
    }

    data[key] = trimmed;
  }
  return { data, body };
}

function isDraft(data) {
  return String(data.draft ?? '').trim().toLowerCase() === 'true';
}

function isHiddenLocalNote(sourceDirName, relPath) {
  const normalized = relPath.replace(/\\/g, '/');
  if (sourceDirName === 'essay') {
    return HIDDEN_ESSAY_PATHS.has(normalized);
  }
  if (sourceDirName === 'summary') {
    return HIDDEN_SUMMARY_PATHS.some((pattern) => pattern.test(normalized));
  }
  return false;
}

function stringifyFrontmatter(data) {
  const lines = ['---'];
  for (const [key, value] of Object.entries(data)) {
    if (value === undefined || value === null || value === '') continue;
    if (Array.isArray(value)) {
      if (value.length === 0) continue;
      lines.push(`${key}:`);
      for (const item of value) lines.push(`  - ${JSON.stringify(item)}`);
    } else if (typeof value === 'boolean') {
      lines.push(`${key}: ${value}`);
    } else {
      lines.push(`${key}: ${JSON.stringify(String(value))}`);
    }
  }
  lines.push('---', '');
  return lines.join('\n');
}

function extractTitle(body, fallback) {
  const match = body.match(/^#\s+(.+)$/m);
  return match ? match[1].trim() : fallback;
}

function inferTags(relPath, collection) {
  const tags = new Set();
  const parts = relPath.split(/[\\/]/).filter(Boolean);
  if (collection === 'notes' && parts.length > 1) tags.add(parts[0]);
  if (collection === 'essay') tags.add('essay');
  if (collection === 'projects') tags.add('projects');
  if (collection === 'gallery') tags.add('gallery');
  return [...tags];
}

function transformBody(body, collection, relPath) {
  let output = body;

  output = output.replace(
    /^!!!\s+(\w+)\s+"([^"]+)"\n((?:    .+\n?)*)/gm,
    (_, type, title, content) => {
      const cleaned = content.replace(/^    /gm, '').trim();
      return `<div class="admonition admonition--${type}">\n<div class="admonition__title">${title}</div>\n\n${cleaned}\n\n</div>\n\n`;
    },
  );

  output = output.replace(/^!!!\s+(\w+)\s*\n((?:    .+\n?)*)/gm, (_, type, content) => {
    const cleaned = content.replace(/^    /gm, '').trim();
    return `<div class="admonition admonition--${type}">\n\n${cleaned}\n\n</div>\n\n`;
  });

  output = output.replace(/^===\s+"([^"]+)"\n([\s\S]*?)(?=^===\s+"|^##\s|^#\s|\Z)/gm, (_, title, content) => {
    return `### ${title}\n\n${content.trim()}\n\n`;
  });

  output = output.replace(/\{ \.[^}]+\}/g, '');
  output = output.replace(/<div markdown="0">\s*/g, '<div>\n');
  output = output.replace(/<\/div>\s*(?=\n|$)/g, '</div>\n');
  output = output.replace(/data-src="\/audio\//g, 'data-src="/audio/');
  output = output.replace(/<div class="home-hero"[\s\S]*?<\/div>/g, '');
  output = output.replace(/<nav class="home-actions"[\s\S]*?<\/nav>/g, '');

  output = output.replace(/!\[([^\]]*)\]\(\.\/([^)]+)\)/g, (_, alt, assetPath) => {
    const publicPath = `/assets/${collection}/${path.posix.dirname(relPath)}/${assetPath}`.replace(/\/+/g, '/');
    return `![${alt}](${publicPath})`;
  });

  output = output.replace(/!\[([^\]]*)\]\((?!https?:|\/|mailto:|#|\.\/)([^)]+)\)/g, (_, alt, assetPath) => {
    const publicPath = `/assets/${collection}/${path.posix.dirname(relPath)}/${assetPath}`.replace(/\/+/g, '/');
    return `![${alt}](${publicPath})`;
  });

  return output.trim() + '\n';
}

function toContentPath(sourceFile, sourceRoot) {
  const rel = path.relative(sourceRoot, sourceFile);
  return rel.replace(/\\/g, '/').replace(/\.md$/, '');
}

function migrateFile(sourceFile, sourceRoot, sourceDirName, collection, previousUpdatedAtByPath) {
  const rel = path.relative(sourceRoot, sourceFile);
  if (SKIP_FILES.has(path.basename(sourceFile))) return null;

  const raw = fs.readFileSync(sourceFile, 'utf8');
  const { data: oldData, body } = parseFrontmatter(raw);
  if (isDraft(oldData)) return null;

  const relPath = toContentPath(sourceFile, sourceRoot);
  if (isHiddenLocalNote(sourceDirName, relPath)) return null;

  const title = oldData.title || extractTitle(body, path.basename(sourceFile, '.md'));
  const legacyPath = `/${collection}/${relPath}/`.replace(/\/+/g, '/');
  const destFile = path.join(CONTENT, collection, `${relPath}.md`);
  const destKey = path.relative(CONTENT, destFile).replace(/\\/g, '/');
  const updatedAt = getSourceUpdatedAt(sourceFile, previousUpdatedAtByPath.get(destKey));

  const newData = {
    title,
    description: oldData.description || undefined,
    date: oldData.date || undefined,
    updatedAt,
    tags: oldData.tags
      ? String(oldData.tags)
          .split(',')
          .map((t) => t.trim())
          .filter(Boolean)
      : inferTags(relPath, collection),
    draft: false,
    legacyPath,
  };

  if (collection === 'gallery') {
    if (oldData.cover) newData.cover = oldData.cover;
    if (oldData.images) {
      newData.images = Array.isArray(oldData.images)
        ? oldData.images
        : String(oldData.images).split(',').map((s) => s.trim()).filter(Boolean);
    }
    if (oldData.category) newData.category = oldData.category;
  }

  const transformed = transformBody(body, collection, relPath.replace(/\\/g, '/'));
  const destDir = path.join(CONTENT, collection, path.dirname(relPath));
  ensureDir(destDir);
  if (fs.existsSync(destFile)) {
    console.warn(`Skip duplicate content path: ${destFile}`);
    return null;
  }
  fs.writeFileSync(destFile, stringifyFrontmatter(newData) + transformed, 'utf8');
  return destFile;
}

function resetContentDir(collection) {
  const dir = path.join(CONTENT, collection);
  if (fs.existsSync(dir)) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
  ensureDir(dir);
}

function removeDiariesContent() {
  const dir = path.join(CONTENT, 'diaries');
  if (fs.existsSync(dir)) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

function collectPreviousUpdatedAt() {
  const values = new Map();
  for (const collection of Object.keys(COLLECTIONS)) {
    const dir = path.join(CONTENT, collection);
    for (const file of walkMarkdown(dir)) {
      const raw = fs.readFileSync(file, 'utf8');
      const { data } = parseFrontmatter(raw);
      if (!data.updatedAt) continue;
      const key = path.relative(CONTENT, file).replace(/\\/g, '/');
      values.set(key, data.updatedAt);
    }
  }
  return values;
}

import { writeNotesNavJson } from './mkdocs-nav.mjs';

function main() {
  ensureDir(CONTENT);
  const previousUpdatedAtByPath = collectPreviousUpdatedAt();
  removeDiariesContent();

  let count = 0;
  let skipped = 0;

  for (const [collection, sourceDirs] of Object.entries(COLLECTIONS)) {
    resetContentDir(collection);
    for (const sourceDirName of sourceDirs) {
      const sourceRoot = path.join(DOCS, sourceDirName);
      const files = walkMarkdown(sourceRoot);
      for (const file of files) {
        const dest = migrateFile(file, sourceRoot, sourceDirName, collection, previousUpdatedAtByPath);
        if (dest) count += 1;
        else skipped += 1;
      }
    }
  }

  console.log(`Migrated ${count} published files from docs/ (${skipped} draft/index/skipped files ignored)`);

  const navTree = writeNotesNavJson(path.join(ROOT, 'site-next/src/data/notes-nav.json'));
  console.log(`Notes nav categories: ${navTree.length} top-level groups`);
}

main();
