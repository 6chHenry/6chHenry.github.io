import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const DOCS_NOTES_DIR = path.resolve(__dirname, '../../docs/notes');

const SKIP_FILES = new Set(['index.md', 'math-test.md', 'academy.md']);

function sortEntries(entries) {
  return entries.sort((a, b) => {
    if (a.isDirectory() !== b.isDirectory()) return a.isDirectory() ? -1 : 1;
    return a.name.localeCompare(b.name, 'zh-Hans-CN', { numeric: true, sensitivity: 'base' });
  });
}

function readFrontmatterAndBody(raw) {
  const start = raw.match(/^---\r?\n/);
  if (!start) return { data: {}, body: raw };
  const endMatch = /\r?\n---\r?\n/.exec(raw.slice(start[0].length));
  if (!endMatch) return { data: {}, body: raw };

  const data = {};
  const end = start[0].length + endMatch.index;
  for (const line of raw.slice(start[0].length, end).split(/\r?\n/)) {
    const match = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (!match) continue;
    data[match[1]] = match[2].replace(/^['"]|['"]$/g, '');
  }

  return { data, body: raw.slice(end + endMatch[0].length) };
}

function isDraft(data) {
  return String(data.draft ?? '').trim().toLowerCase() === 'true';
}

function titleFromMarkdown(filePath) {
  const raw = fs.readFileSync(filePath, 'utf8');
  const { data, body } = readFrontmatterAndBody(raw);
  if (isDraft(data)) return null;
  if (data.title) return data.title;

  const heading = body.match(/^#\s+(.+)$/m);
  return heading ? heading[1].trim() : path.basename(filePath, '.md');
}

export function slugifyNavTitle(title) {
  const base = title
    .trim()
    .toLowerCase()
    .replace(/[^\w\u4e00-\u9fff]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 72);
  return base || 'section';
}

function assignSlugPaths(nodes, prefix = []) {
  const seen = new Map();

  for (const node of nodes) {
    const baseSlug = slugifyNavTitle(node.slugSource ?? node.title);
    const count = seen.get(baseSlug) ?? 0;
    const slug = count > 0 ? `${baseSlug}-${count + 1}` : baseSlug;
    seen.set(baseSlug, count + 1);

    node.slug = slug;
    node.slugPath = [...prefix, slug];
    delete node.slugSource;
    if (node.children.length > 0) assignSlugPaths(node.children, node.slugPath);
  }
}

function buildDirectoryNode(dirPath, relativeParts = []) {
  const entries = sortEntries(fs.readdirSync(dirPath, { withFileTypes: true }));
  const children = [];

  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;

    const fullPath = path.join(dirPath, entry.name);
    if (entry.isDirectory()) {
      if (entry.name.endsWith('.assets')) continue;
      const child = buildDirectoryNode(fullPath, [...relativeParts, entry.name]);
      if (child.children.length > 0) children.push(child);
      continue;
    }

    if (!entry.isFile() || !entry.name.endsWith('.md') || SKIP_FILES.has(entry.name)) continue;

    const title = titleFromMarkdown(fullPath);
    if (!title) continue;

    const docsPath = path.posix.join('notes', ...relativeParts, entry.name);
    children.push({
      title,
      slugSource: path.basename(entry.name, '.md'),
      docsPath,
      children: [],
    });
  }

  return {
    title: relativeParts.at(-1) ?? 'notes',
    slugSource: relativeParts.at(-1) ?? 'notes',
    children,
  };
}

/** Build notes navigation from the docs/notes directory tree. */
export function parseNotesNavTree(notesDir = DOCS_NOTES_DIR) {
  if (!fs.existsSync(notesDir)) return [];
  const root = buildDirectoryNode(notesDir);
  assignSlugPaths(root.children);
  return root.children;
}

export function walkNavNodes(nodes, visitor, trail = []) {
  for (const node of nodes) {
    const nextTrail = [...trail, node];
    visitor(node, nextTrail);
    if (node.children.length > 0) walkNavNodes(node.children, visitor, nextTrail);
  }
}

export function collectCategoryNodes(tree) {
  const categories = [];
  walkNavNodes(tree, (node) => {
    if (node.children.length > 0) categories.push(node);
  });
  return categories;
}

export function findNavNodeBySlugPath(tree, slugParts) {
  let current = tree;
  let found = null;

  for (const part of slugParts) {
    found = current.find((node) => node.slug === part);
    if (!found) return null;
    current = found.children;
  }

  return found;
}

export function docsPathToNoteId(docsPath) {
  return docsPath.replace(/^notes\//, '').replace(/\.md$/i, '');
}

export function writeNotesNavJson(outputPath, notesDir = DOCS_NOTES_DIR) {
  const tree = parseNotesNavTree(notesDir);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(tree, null, 2), 'utf8');
  return tree;
}
