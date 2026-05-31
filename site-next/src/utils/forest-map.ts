import { buildCharacterTreeSvg } from './character-tree';
import { buildContentUrl, formatRecencyDate, sortByRecency } from './content';

export interface ForestMapEntry {
  id: string;
  collection: string;
  body?: string;
  data: {
    title: string;
    date?: Date;
    updatedAt?: Date;
    tags?: string[];
  };
}

export interface ForestTreeNode {
  id: string;
  title: string;
  href: string;
  collection: string;
  timestamp: number;
  x?: number;
  y?: number;
  scale?: number;
  dateLabel?: string;
  summary: string;
  treeSvg?: string;
  recentRank?: number;
}

export interface ForestMapData {
  trees: ForestTreeNode[];
  recentTrees: ForestTreeNode[];
  totalCount: number;
}

function getTimestamp(entry: ForestMapEntry): number {
  return entry.data.updatedAt?.getTime() ?? entry.data.date?.getTime() ?? 0;
}

const GROVE_TREE_LIMIT = 11;

const GROVE_POSITIONS: Record<string, Array<{ x: number; y: number; scale: number }>> = {
  notes: [
    { x: 17, y: 69, scale: 0.92 },
    { x: 27, y: 54, scale: 1.08 },
    { x: 39, y: 72, scale: 0.98 },
    { x: 50, y: 49, scale: 1.18 },
    { x: 62, y: 68, scale: 0.9 },
    { x: 73, y: 55, scale: 1.04 },
    { x: 84, y: 73, scale: 0.86 },
    { x: 22, y: 86, scale: 0.78 },
    { x: 56, y: 84, scale: 0.82 },
    { x: 78, y: 86, scale: 0.76 },
    { x: 36, y: 39, scale: 0.84 },
  ],
  essay: [
    { x: 16, y: 74, scale: 0.88 },
    { x: 27, y: 57, scale: 1.02 },
    { x: 39, y: 76, scale: 0.94 },
    { x: 51, y: 51, scale: 1.16 },
    { x: 63, y: 68, scale: 0.98 },
    { x: 75, y: 49, scale: 0.9 },
    { x: 84, y: 78, scale: 0.82 },
    { x: 22, y: 88, scale: 0.72 },
    { x: 46, y: 88, scale: 0.78 },
    { x: 68, y: 87, scale: 0.76 },
    { x: 57, y: 34, scale: 0.82 },
  ],
  projects: [
    { x: 17, y: 72, scale: 0.86 },
    { x: 29, y: 56, scale: 1.08 },
    { x: 42, y: 75, scale: 0.92 },
    { x: 55, y: 50, scale: 1.22 },
    { x: 68, y: 70, scale: 0.96 },
    { x: 80, y: 56, scale: 0.84 },
    { x: 34, y: 88, scale: 0.76 },
    { x: 58, y: 87, scale: 0.8 },
    { x: 76, y: 84, scale: 0.72 },
    { x: 22, y: 42, scale: 0.78 },
    { x: 70, y: 37, scale: 0.74 },
  ],
};

function sanitizeClipId(value: string): string {
  return value.replace(/[^\w\u4e00-\u9fff-]+/g, '-').slice(0, 64);
}

function buildShowcaseLookup(entries: ForestMapEntry[]) {
  const byCollection = new Map<string, ForestMapEntry[]>();

  for (const entry of sortByRecency(entries)) {
    const group = byCollection.get(entry.collection) ?? [];
    group.push(entry);
    byCollection.set(entry.collection, group);
  }

  const lookup = new Map<string, { index: number; position: { x: number; y: number; scale: number } }>();
  for (const [collection, group] of byCollection) {
    const positions = GROVE_POSITIONS[collection] ?? GROVE_POSITIONS.notes;
    group.slice(0, GROVE_TREE_LIMIT).forEach((entry, index) => {
      lookup.set(`${entry.collection}:${entry.id}`, {
        index,
        position: positions[index % positions.length],
      });
    });
  }

  return lookup;
}

function cleanMarkdownForSummary(body: string): string {
  return body
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/<div[^>]*data-route=["'][^"']+["'][\s\S]*?<\/div>/g, ' ')
    .replace(/!\[[^\]]*]\([^)]+\)/g, ' ')
    .replace(/\[[^\]]+]\([^)]+\)/g, (match) => match.replace(/^\[|\]\([^)]+\)$/g, ''))
    .replace(/^---[\s\S]*?---/g, ' ')
    .replace(/^#{1,6}\s+/gm, ' ')
    .replace(/[`*_~>|#-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function buildSummary(entry: ForestMapEntry): string {
  const cleaned = cleanMarkdownForSummary(entry.body ?? '');
  const sentences = cleaned
    .split(/(?<=[。！？!?；;])\s*/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence.length >= 12 && !/^#+\s*/.test(sentence));

  const source = sentences.find((sentence) => !sentence.includes('data-route')) ?? cleaned;
  const fallback = entry.data.tags?.length
    ? `关于 ${entry.data.tags.slice(0, 3).join('、')} 的一篇笔记。`
    : '这篇内容还没有足够的正文可供生成摘要。';

  const summary = (source || fallback).replace(/\s+/g, ' ').trim();
  if (summary.length < 24 || /详见|\.pdf\b|by\s+6ch/i.test(summary)) {
    if (entry.collection === 'essay') {
      return `${entry.data.title} 的随笔，记录经历、观察与当时的思考。`;
    }
    if (entry.collection === 'projects') {
      return `${entry.data.title} 的项目记录，概述目标、实现思路与阶段结果。`;
    }
    return `${entry.data.title} 的学习笔记，整理相关概念、推导过程与实现要点。`;
  }
  if (summary.length <= 86) return summary;
  return `${summary.slice(0, 84).replace(/[，,、：:；;。.!！?？\s]+$/g, '')}...`;
}

export function buildForestMapStops(entries: ForestMapEntry[], base = import.meta.env.BASE_URL): ForestMapData {
  const showcaseLookup = buildShowcaseLookup(entries);

  const trees = entries
    .map((entry) => {
      const key = `${entry.collection}:${entry.id}`;
      const showcase = showcaseLookup.get(key);
      const timestamp = getTimestamp(entry);

      return {
        id: key,
        title: entry.data.title,
        href: buildContentUrl(entry.collection, entry.id, base),
        collection: entry.collection,
        timestamp,
        ...(showcase
          ? {
              x: showcase.position.x,
              y: showcase.position.y,
              scale: showcase.position.scale,
              treeSvg: buildCharacterTreeSvg({
                seed: key,
                title: entry.data.title,
                body: entry.body ?? '',
                clipId: `grove-tree-${showcase.index}-${sanitizeClipId(key)}`,
                variant: 'compact',
              }),
            }
          : {}),
        dateLabel: formatRecencyDate(entry),
        summary: buildSummary(entry),
      };
    })
    .sort((a, b) => a.title.localeCompare(b.title, 'zh-CN'));

  const treeById = new Map(trees.map((tree) => [tree.id, tree]));
  const recentTrees = sortByRecency(entries)
    .filter((entry) => getTimestamp(entry) > 0)
    .slice(0, 9)
    .map((entry, index) => {
      const tree = treeById.get(`${entry.collection}:${entry.id}`);
      if (!tree) return undefined;
      tree.recentRank = index + 1;
      return tree;
    })
    .filter((tree): tree is ForestTreeNode => Boolean(tree));

  return {
    trees,
    recentTrees,
    totalCount: entries.length,
  };
}
