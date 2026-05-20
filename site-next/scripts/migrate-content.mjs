#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '../..');
const DOCS = path.join(ROOT, 'docs');
const CONTENT = path.join(ROOT, 'site-next/src/content');
const MKDOCS_YML = path.join(ROOT, 'mkdocs.yml');

/** Only migrate markdown paths listed in mkdocs.yml nav (non-comment lines). */
const COLLECTIONS = {
  notes: ['notes'],
  essay: ['essay', 'summary'],
  projects: ['projects'],
};

const SKIP_FILES = new Set(['index.md', 'math-test.md', 'academy.md']);

const SOURCE_TO_COLLECTION = {
  notes: 'notes',
  essay: 'essay',
  summary: 'essay',
  projects: 'projects',
};

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

/** Parse mkdocs.yml nav and return allowed docs-relative paths like `notes/Merge.md`. */
export function parseMkdocsNavAllowlist(mkdocsPath = MKDOCS_YML) {
  const raw = fs.readFileSync(mkdocsPath, 'utf8');
  const lines = raw.split('\n');
  const navStart = lines.findIndex((line) => /^nav:\s*$/.test(line));
  if (navStart === -1) return new Set();

  const allowed = new Set();
  for (let i = navStart + 1; i < lines.length; i += 1) {
    const line = lines[i];
    if (/^[a-zA-Z_][\w-]*:\s/.test(line) && !line.startsWith(' ')) break;

    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;

    const content = trimmed.split('#')[0].trim();
    if (!content) continue;

    const mdMatch = content.match(/(?::\s*|index:\s*)(.+\.md)\s*$/);
    if (!mdMatch) continue;

    allowed.add(mdMatch[1].replace(/\\/g, '/'));
  }
  return allowed;
}

function mapToCollection(docsRelativePath) {
  const top = docsRelativePath.split('/')[0];
  return SOURCE_TO_COLLECTION[top] ?? null;
}

function getSourceUpdatedAt(absPath) {
  try {
    const iso = execSync(`git log -1 --format=%cI -- "${absPath}"`, {
      cwd: ROOT,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
    if (iso) return iso;
  } catch {
    // fall through to file mtime
  }
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
  if (!raw.startsWith('---\n')) {
    return { data: {}, body: raw };
  }
  const end = raw.indexOf('\n---\n', 4);
  if (end === -1) return { data: {}, body: raw };
  const yaml = raw.slice(4, end);
  const body = raw.slice(end + 5);
  const data = {};
  for (const line of yaml.split('\n')) {
    const match = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (!match) continue;
    const [, key, value] = match;
    data[key] = value.replace(/^['"]|['"]$/g, '');
  }
  return { data, body };
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

function migrateFile(sourceFile, sourceRoot, collection, allowlist) {
  const docsRelative = path.relative(DOCS, sourceFile).replace(/\\/g, '/');
  if (!allowlist.has(docsRelative)) return null;

  const rel = path.relative(sourceRoot, sourceFile);
  if (SKIP_FILES.has(path.basename(sourceFile))) return null;

  const raw = fs.readFileSync(sourceFile, 'utf8');
  const { data: oldData, body } = parseFrontmatter(raw);
  const relPath = toContentPath(sourceFile, sourceRoot);
  const title = oldData.title || extractTitle(body, path.basename(sourceFile, '.md'));
  const legacyPath = `/${collection}/${relPath}/`.replace(/\/+/g, '/');
  const updatedAt = getSourceUpdatedAt(sourceFile);

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

  const transformed = transformBody(body, collection, relPath.replace(/\\/g, '/'));
  const destDir = path.join(CONTENT, collection, path.dirname(relPath));
  ensureDir(destDir);
  const destFile = path.join(CONTENT, collection, `${relPath}.md`);
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

import { writeNotesNavJson } from './mkdocs-nav.mjs';

function main() {
  const allowlist = parseMkdocsNavAllowlist();
  ensureDir(CONTENT);
  removeDiariesContent();

  let count = 0;
  let skipped = 0;

  for (const [collection, sourceDirs] of Object.entries(COLLECTIONS)) {
    resetContentDir(collection);
    for (const sourceDirName of sourceDirs) {
      const sourceRoot = path.join(DOCS, sourceDirName);
      const files = walkMarkdown(sourceRoot);
      for (const file of files) {
        const docsRelative = path.relative(DOCS, file).replace(/\\/g, '/');
        if (!allowlist.has(docsRelative)) {
          skipped += 1;
          continue;
        }
        const dest = migrateFile(file, sourceRoot, collection, allowlist);
        if (dest) count += 1;
      }
    }
  }

  console.log(`MkDocs nav allowlist: ${allowlist.size} paths`);
  console.log(`Migrated ${count} published files (${skipped} private/unlisted skipped)`);

  const navTree = writeNotesNavJson(path.join(ROOT, 'site-next/src/data/notes-nav.json'));
  console.log(`Notes nav categories: ${navTree.length} top-level groups`);
}

main();
