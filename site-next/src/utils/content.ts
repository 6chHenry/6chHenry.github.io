export interface ContentEntry {
  id: string;
  slug: string;
  collection: string;
  data: {
    title: string;
    description?: string;
    date?: Date;
    tags?: string[];
    draft?: boolean;
    legacyPath?: string;
  };
}

export function getEntryTimestamp(entry: { data: { updatedAt?: Date; date?: Date } }): number {
  return entry.data.updatedAt?.getTime() ?? entry.data.date?.getTime() ?? 0;
}

export function sortByRecency<T extends { data: { updatedAt?: Date; date?: Date } }>(entries: T[]): T[] {
  return [...entries].sort((a, b) => getEntryTimestamp(b) - getEntryTimestamp(a));
}

/** @deprecated use sortByRecency */
export function sortByDate<T extends { data: { updatedAt?: Date; date?: Date } }>(entries: T[]): T[] {
  return sortByRecency(entries);
}

export function formatDate(date?: Date): string | undefined {
  if (!date) return undefined;
  return date.toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

export function formatRecencyDate(entry: { data: { updatedAt?: Date; date?: Date } }): string | undefined {
  const ts = getEntryTimestamp(entry);
  if (!ts) return undefined;
  return formatDate(new Date(ts));
}

export function getSlugParts(id: string): string[] {
  return id.split('/').filter(Boolean);
}

export function normalizeSlug(id: string): string {
  return id.replace(/\.md$/i, '');
}

export function buildContentUrl(collection: string, id: string, base = import.meta.env.BASE_URL): string {
  const slug = getSlugParts(normalizeSlug(id)).join('/');
  return `${base}${collection}/${slug}/`.replace(/\/{2,}/g, '/');
}

export function collectTags<T extends { data: { tags?: string[] } }>(entries: T[]): Map<string, number> {
  const tags = new Map<string, number>();
  for (const entry of entries) {
    for (const tag of entry.data.tags ?? []) {
      tags.set(tag, (tags.get(tag) ?? 0) + 1);
    }
  }
  return tags;
}

export function excerpt(text: string | undefined, max = 140): string {
  if (!text) return '';
  const plain = text
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/[#>*_\[\]()!`~-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  if (plain.length <= max) return plain;
  return `${plain.slice(0, max).trim()}…`;
}
