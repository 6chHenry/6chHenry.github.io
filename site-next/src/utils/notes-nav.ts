import notesNavTree from '../data/notes-nav.json';

export interface NotesNavNode {
  title: string;
  slug: string;
  slugPath: string[];
  docsPath?: string;
  children: NotesNavNode[];
}

export const notesNav = notesNavTree as NotesNavNode[];

export function normalizeNoteId(id: string): string {
  return id.replace(/\\/g, '/').replace(/\.md$/i, '');
}

export function docsPathToNoteId(docsPath: string): string {
  return normalizeNoteId(docsPath.replace(/^notes[/\\]/, ''));
}

export function resolveNoteEntry<T extends { id: string }>(
  docsPath: string | undefined,
  entryById: Map<string, T>,
): T | undefined {
  if (!docsPath) return undefined;
  return entryById.get(docsPathToNoteId(docsPath));
}

export function buildNoteEntryMap<T extends { id: string }>(entries: T[]): Map<string, T> {
  return new Map(entries.map((entry) => [normalizeNoteId(entry.id), entry]));
}

export function buildCategoryUrl(slugPath: string[], base = import.meta.env.BASE_URL): string {
  return `${base}notes/category/${slugPath.join('/')}/`.replace(/\/{2,}/g, '/');
}

export function findNavNodeBySlugPath(slugParts: string[]): NotesNavNode | null {
  let current = notesNav;
  let found: NotesNavNode | null = null;

  for (const part of slugParts) {
    found = current.find((node) => node.slug === part) ?? null;
    if (!found) return null;
    current = found.children;
  }

  return found;
}

export function walkNavNodes(
  nodes: NotesNavNode[],
  visitor: (node: NotesNavNode, trail: NotesNavNode[]) => void,
  trail: NotesNavNode[] = [],
) {
  for (const node of nodes) {
    const nextTrail = [...trail, node];
    visitor(node, nextTrail);
    if (node.children.length > 0) walkNavNodes(node.children, visitor, nextTrail);
  }
}

export function collectCategoryNodes(nodes: NotesNavNode[] = notesNav): NotesNavNode[] {
  const categories: NotesNavNode[] = [];
  walkNavNodes(nodes, (node) => {
    if (node.children.length > 0) categories.push(node);
  });
  return categories;
}

export function collectDocsPaths(node: NotesNavNode): string[] {
  const paths: string[] = [];
  walkNavNodes([node], (item) => {
    if (item.docsPath) paths.push(item.docsPath);
  });
  return paths;
}

export function countNotesInNode(node: NotesNavNode): number {
  return collectDocsPaths(node).length;
}

export interface NoteLink {
  href: string;
  title: string;
  description?: string;
  date?: string;
}

export interface NavTreeItem {
  kind: 'folder' | 'note';
  title: string;
  slugPath: string[];
  count?: number;
  note?: NoteLink;
  children?: NavTreeItem[];
}

export function buildNavPanelId(slugPath: string[]): string {
  return `notes-panel-${slugPath.join('--')}`;
}

export function buildNavTree<T extends { id: string; data: { title: string; description?: string } }>(
  nodes: NotesNavNode[],
  entryById: Map<string, T>,
  linkFor: (entry: T) => NoteLink,
): NavTreeItem[] {
  const items: NavTreeItem[] = [];

  for (const node of nodes) {
    if (node.children.length > 0) {
      items.push({
        kind: 'folder',
        title: node.title,
        slugPath: node.slugPath,
        count: countNotesInNode(node),
        children: buildNavTree(node.children, entryById, linkFor),
      });
      continue;
    }

    const entry = resolveNoteEntry(node.docsPath, entryById);
    if (!entry) continue;

    items.push({
      kind: 'note',
      title: node.title,
      slugPath: node.slugPath,
      note: linkFor(entry),
    });
  }

  return items;
}
