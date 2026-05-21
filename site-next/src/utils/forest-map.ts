import { buildCharacterTreeSvg } from './character-tree';
import { buildContentUrl, formatRecencyDate, sortByRecency } from './content';

export const FOREST_MAP_WIDTH = 1280;
export const FOREST_MAP_HEIGHT = 720;

export interface ForestMapEntry {
  id: string;
  collection: string;
  body: string;
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
  x: number;
  y: number;
  size: number;
  dateLabel?: string;
  summary: string;
  treeSvg: string;
  recentRank?: number;
}

export interface ForestMapData {
  trees: ForestTreeNode[];
  recentTrees: ForestTreeNode[];
  totalCount: number;
}

function hashString(input: string): number {
  let hash = 2166136261;
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function createRng(seed: number) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function collectionBand(collection: string) {
  if (collection === 'notes') {
    return { x: 110, y: 100, width: 620, height: 430, label: 'Notes Grove' };
  }
  if (collection === 'essay') {
    return { x: 660, y: 185, width: 480, height: 330, label: 'Essay Grove' };
  }
  return { x: 360, y: 430, width: 580, height: 190, label: 'Project Grove' };
}

function getTimestamp(entry: ForestMapEntry): number {
  return entry.data.updatedAt?.getTime() ?? entry.data.date?.getTime() ?? 0;
}

function computeTreeSize(entry: ForestMapEntry, newestTs: number, oldestTs: number): number {
  const bodyWeight = clamp(Math.log10(Math.max(80, entry.body.length)) / 4.2, 0.56, 1);
  const ts = getTimestamp(entry);
  const recency = newestTs > oldestTs && ts > 0 ? (ts - oldestTs) / (newestTs - oldestTs) : 0.5;
  return 0.72 + bodyWeight * 0.42 + recency * 0.18;
}

function sanitizeClipId(value: string): string {
  return value.replace(/[^\w\u4e00-\u9fff-]+/g, '-').slice(0, 60);
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
  const cleaned = cleanMarkdownForSummary(entry.body);
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

export function buildCurvePath(points: Array<{ x: number; y: number }>): string {
  if (points.length === 0) return '';
  if (points.length === 1) return `M ${points[0].x} ${points[0].y}`;

  let path = `M ${points[0].x.toFixed(1)} ${points[0].y.toFixed(1)}`;
  for (let i = 0; i < points.length - 1; i += 1) {
    const current = points[i];
    const next = points[i + 1];
    const midX = (current.x + next.x) / 2;
    path += ` C ${midX.toFixed(1)} ${current.y.toFixed(1)}, ${midX.toFixed(1)} ${next.y.toFixed(1)}, ${next.x.toFixed(1)} ${next.y.toFixed(1)}`;
  }
  return path;
}

export function buildForestMapStops(entries: ForestMapEntry[], base = import.meta.env.BASE_URL): ForestMapData {
  const dated = entries.filter((entry) => getTimestamp(entry) > 0);
  const newestTs = Math.max(...dated.map(getTimestamp), 0);
  const oldestTs = Math.min(...dated.map(getTimestamp), newestTs);

  const trees = entries
    .map((entry, index) => {
      const band = collectionBand(entry.collection);
      const rng = createRng(hashString(`${entry.collection}:${entry.id}`));
      const rowBias = entry.collection === 'projects' ? 0.25 : 0;
      const x = band.x + rng() * band.width;
      const y = band.y + (rng() * 0.72 + rowBias) * band.height;
      const size = computeTreeSize(entry, newestTs, oldestTs);
      const clipId = `map-tree-${index}-${sanitizeClipId(entry.id)}`;

      return {
        id: `${entry.collection}:${entry.id}`,
        title: entry.data.title,
        href: buildContentUrl(entry.collection, entry.id, base),
        collection: entry.collection,
        x,
        y,
        size,
        dateLabel: formatRecencyDate(entry),
        summary: buildSummary(entry),
        treeSvg: buildCharacterTreeSvg({
          seed: `${entry.collection}:${entry.id}`,
          title: entry.data.title,
          body: entry.body,
          clipId,
          variant: 'compact',
        }),
      };
    })
    .sort((a, b) => a.y - b.y);

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
