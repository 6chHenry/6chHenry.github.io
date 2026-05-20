import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const MKDOCS_YML = path.resolve(__dirname, '../../mkdocs.yml');

export function slugifyNavTitle(title) {
  const base = title
    .trim()
    .toLowerCase()
    .replace(/[^\w\u4e00-\u9fff]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 72);
  return base || 'section';
}

function parseNavLine(line) {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith('#')) return null;

  const match = line.match(/^(\s*)- (.+)$/);
  if (!match) return null;

  const indent = match[1].length;
  const rest = match[2].split('#')[0].trim();
  if (!rest || /^index:\s*.+\.md/i.test(rest)) return null;

  let title = rest;
  let docsPath;

  const colonIndex = rest.indexOf(':');
  if (colonIndex !== -1) {
    title = rest.slice(0, colonIndex).trim().replace(/^["']|["']$/g, '');
    const after = rest.slice(colonIndex + 1).trim();
    if (after.endsWith('.md')) docsPath = after.replace(/\\/g, '/');
  }

  title = title.replace(/^["']|["']$/g, '');
  if (!title) return null;

  return { indent, title, docsPath };
}

function assignSlugPaths(nodes, prefix = []) {
  const seen = new Map();

  for (const node of nodes) {
    let slug = slugifyNavTitle(node.title);
    const count = seen.get(slug) ?? 0;
    if (count > 0) slug = `${slug}-${count + 1}`;
    seen.set(slugifyNavTitle(node.title), count + 1);

    node.slug = slug;
    node.slugPath = [...prefix, slug];
    if (node.children.length > 0) {
      assignSlugPaths(node.children, node.slugPath);
    }
  }
}

/** Parse mkdocs.yml notes nav into a nested tree. */
export function parseNotesNavTree(mkdocsPath = MKDOCS_YML) {
  const lines = fs.readFileSync(mkdocsPath, 'utf8').split('\n');
  const navStart = lines.findIndex((line) => /^nav:\s*$/.test(line));
  if (navStart === -1) return [];

  let notesStart = -1;
  for (let i = navStart + 1; i < lines.length; i += 1) {
    const trimmed = lines[i].trim();
    if (trimmed.startsWith('#')) continue;
    if (/^- 笔记:/.test(trimmed)) {
      notesStart = i;
      break;
    }
  }
  if (notesStart === -1) return [];

  const root = [];
  const stack = [{ indent: 2, children: root }];

  for (let i = notesStart + 1; i < lines.length; i += 1) {
    const line = lines[i];
    const trimmed = line.trim();

    if (trimmed && !trimmed.startsWith('#') && /^  - /.test(line) && !/^    /.test(line)) {
      break;
    }

    const parsed = parseNavLine(line);
    if (!parsed || parsed.indent < 4) continue;

    const node = {
      title: parsed.title,
      docsPath: parsed.docsPath,
      children: [],
    };

    while (stack.length > 1 && stack[stack.length - 1].indent >= parsed.indent) {
      stack.pop();
    }

    stack[stack.length - 1].children.push(node);

    if (!parsed.docsPath) {
      stack.push({ indent: parsed.indent, children: node.children });
    }
  }

  assignSlugPaths(root);
  return root;
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

export function writeNotesNavJson(outputPath, mkdocsPath = MKDOCS_YML) {
  const tree = parseNotesNavTree(mkdocsPath);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(tree, null, 2), 'utf8');
  return tree;
}
