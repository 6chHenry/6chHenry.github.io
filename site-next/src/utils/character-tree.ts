import { weighDocument, type CharPortrait } from './character-lexicon';

function createRng(seed: number) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashString(input: string): number {
  let hash = 2166136261;
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

/** Simple pine-tree polygon (viewBox 0 0 120 160). */
const TREE_POLYGON: Array<[number, number]> = [
  [60, 6],
  [72, 34],
  [66, 34],
  [80, 58],
  [73, 58],
  [86, 82],
  [78, 82],
  [90, 106],
  [68, 106],
  [68, 154],
  [52, 154],
  [52, 106],
  [30, 106],
  [42, 82],
  [34, 82],
  [47, 58],
  [40, 58],
  [54, 34],
  [48, 34],
];

function pointInPolygon(x: number, y: number, polygon: Array<[number, number]>): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    const intersects = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 0.0001) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

function escapeSvgText(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function shuffle<T>(items: T[], rng: () => number): T[] {
  for (let i = items.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
}

function treePath(): string {
  return TREE_POLYGON.map(([x, y], index) => `${index === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`).join(' ');
}

interface GlyphSeed {
  char: string;
  count: number;
  source: 'title' | 'heading' | 'body';
  heading?: string;
  distinctive: boolean;
  rank: number;
}

function collectGlyphPool(
  title: string,
  headings: string[],
  body: string,
  portraits: CharPortrait[],
  limit: number,
  seed: string,
): GlyphSeed[] {
  const weights = weighDocument(title, headings, body);
  const distinctive = new Map(portraits.map((item) => [item.char, item]));
  const maxScore = Math.max(...portraits.map((item) => item.score), 0.001);
  const expanded: GlyphSeed[] = [];

  for (const [char, stat] of weights) {
    const portrait = distinctive.get(char);
    const copies = Math.min(8, Math.max(1, Math.round(Math.sqrt(stat.count))));
    const source: GlyphSeed['source'] = stat.inTitle ? 'title' : stat.inHeading ? 'heading' : 'body';
    for (let i = 0; i < copies; i += 1) {
      expanded.push({
        char,
        count: stat.count,
        source,
        heading: stat.heading,
        distinctive: Boolean(portrait),
        rank: portrait ? portrait.score / maxScore : 0.35,
      });
    }
  }

  if (expanded.length === 0) {
    for (const char of '林间笔记森树') {
      expanded.push({ char, count: 1, source: 'body', distinctive: false, rank: 0.3 });
    }
  }

  const rng = createRng(hashString(`${seed}:${body.length}`));
  shuffle(expanded, rng);

  const guaranteed: GlyphSeed[] = [];
  for (const portrait of portraits) {
    const found = expanded.find((glyph) => glyph.char === portrait.char);
    guaranteed.push(
      found ?? {
        char: portrait.char,
        count: portrait.count,
        source: portrait.inTitle ? 'title' : portrait.inHeading ? 'heading' : 'body',
        heading: portrait.heading,
        distinctive: true,
        rank: portrait.score / maxScore,
      },
    );
  }

  const rest = expanded.filter((glyph) => !glyph.distinctive);
  const pool: GlyphSeed[] = [];
  const mixed = [...guaranteed, ...rest];
  while (pool.length < limit && mixed.length > 0) pool.push(...mixed);
  return pool.slice(0, limit);
}

export interface CharacterTreeOptions {
  seed: string;
  portraits?: CharPortrait[];
  title?: string;
  headings?: string[];
  body?: string;
  clipId?: string;
  variant?: 'default' | 'compact' | 'year';
  hrefForChar?: (char: string) => string;
}

function buildYearTreeSvg({
  seed,
  portraits,
  clipId,
  hrefForChar,
}: Required<Pick<CharacterTreeOptions, 'seed' | 'clipId'>> & Pick<CharacterTreeOptions, 'portraits' | 'hrefForChar'>): string {
  const list = portraits ?? [];
  const rng = createRng(hashString(seed));
  const canopy: Array<{ x: number; y: number }> = [];
  for (let y = 28; y <= 120; y += 13) {
    for (let x = 22; x <= 98; x += 12) {
      const px = x + (rng() - 0.5) * 5;
      const py = y + (rng() - 0.5) * 5;
      if (pointInPolygon(px, py, TREE_POLYGON)) canopy.push({ x: px, y: py });
    }
  }
  canopy.sort((a, b) => a.y - b.y || a.x - b.x);
  const maxScore = Math.max(...list.map((item) => item.score), 0.001);
  const take = Math.min(list.length, 14);
  const step = Math.max(1, Math.floor(canopy.length / Math.max(take, 1)));
  const labels = list.slice(0, take).map((portrait, index) => {
    const slot = canopy[Math.min(index * step, canopy.length - 1)] ?? { x: 60, y: 48 };
    const rank = portrait.score / maxScore;
    const size = 13 + rank * 11;
    const rotation = (rng() - 0.5) * 7;
    const tone = rng();
    const fill =
      tone > 0.66 ? 'var(--ch-accent-secondary)' : tone > 0.33 ? 'var(--ch-accent-tertiary)' : 'var(--ch-accent)';
    const source = portrait.inTitle ? 'title' : portrait.inHeading ? 'heading' : 'body';
    const href = hrefForChar?.(portrait.char);
    const text = `<text class="character-tree__glyph" data-char="${escapeSvgText(portrait.char)}" data-count="${portrait.count}" data-source="${source}" data-heading="${escapeSvgText(portrait.heading ?? '')}" data-rank="${rank.toFixed(3)}" x="${slot.x.toFixed(1)}" y="${slot.y.toFixed(1)}" fill="${fill}" font-size="${size.toFixed(1)}" transform="rotate(${rotation.toFixed(1)} ${slot.x.toFixed(1)} ${slot.y.toFixed(1)})" font-family="var(--ch-text-font)" style="--char-opacity:${(0.72 + rank * 0.28).toFixed(2)}">${escapeSvgText(portrait.char)}</text>`;
    return href
      ? `<a class="character-tree__link" href="${escapeSvgText(href)}" aria-label="查看汉字「${escapeSvgText(portrait.char)}」">${text}</a>`
      : text;
  }).join('');

  return `<svg class="character-tree__svg" id="${escapeSvgText(clipId)}" viewBox="0 0 120 160" focusable="false" xmlns="http://www.w3.org/2000/svg"><path class="character-tree__silhouette" d="${treePath()} Z" fill="none" /><g>${labels}</g></svg>`;
}

export function buildCharacterTreeSvg({
  seed,
  portraits = [],
  title = '',
  headings = [],
  body = '',
  clipId = 'tree-clip',
  variant = 'default',
  hrefForChar,
}: CharacterTreeOptions): string {
  if (variant === 'year') {
    return buildYearTreeSvg({ seed, portraits, clipId, hrefForChar });
  }

  const compact = variant === 'compact';
  const rng = createRng(hashString(seed));
  const pool = collectGlyphPool(title, headings, body, portraits, compact ? 48 : 180, seed);
  const slots: Array<{ x: number; y: number }> = [];

  for (let y = 10; y <= 150; y += 9) {
    for (let x = 18; x <= 102; x += 9) {
      const px = x + (rng() - 0.5) * 4;
      const py = y + (rng() - 0.5) * 4;
      if (pointInPolygon(px, py, TREE_POLYGON)) slots.push({ x: px, y: py });
    }
  }
  shuffle(slots, rng);

  const count = Math.min(slots.length, pool.length, compact ? 28 : 110);
  const labels = Array.from({ length: count }, (_, index) => {
    const slot = slots[index];
    const glyph = pool[index];
    const size = compact ? 7.5 + rng() * 4 : 10.5 + rng() * 5.5;
    const rotation = (rng() - 0.5) * (compact ? 8 : 10);
    const opacity = compact ? 0.58 + rng() * 0.34 : 0.52 + rng() * 0.4;
    const tone = rng();
    const fill =
      tone > 0.66 ? 'var(--ch-accent-secondary)' : tone > 0.33 ? 'var(--ch-accent-tertiary)' : 'var(--ch-accent)';
    const href = glyph.distinctive ? hrefForChar?.(glyph.char) : undefined;
    const text = `<text class="character-tree__glyph" data-char="${escapeSvgText(glyph.char)}" data-count="${glyph.count}" data-source="${glyph.source}" data-heading="${escapeSvgText(glyph.heading ?? '')}" data-rank="${glyph.rank.toFixed(3)}" x="${slot.x.toFixed(1)}" y="${slot.y.toFixed(1)}" fill="${fill}" font-size="${size.toFixed(1)}" transform="rotate(${rotation.toFixed(1)} ${slot.x.toFixed(1)} ${slot.y.toFixed(1)})" font-family="var(--ch-text-font)" style="--char-opacity:${opacity.toFixed(2)}">${escapeSvgText(glyph.char)}</text>`;
    return href
      ? `<a class="character-tree__link" href="${escapeSvgText(href)}" aria-label="查看汉字「${escapeSvgText(glyph.char)}」">${text}</a>`
      : text;
  }).join('');

  return `<svg class="character-tree__svg" viewBox="0 0 120 160" focusable="false" xmlns="http://www.w3.org/2000/svg"><defs><clipPath id="${escapeSvgText(clipId)}"><path d="${treePath()} Z" /></clipPath></defs><path class="character-tree__silhouette" d="${treePath()} Z" fill="none" /><g clip-path="url(#${escapeSvgText(clipId)})">${labels}</g></svg>`;
}

export function treeClipId(entryId: string, prefix = 'tree'): string {
  return `${prefix}-${entryId.replace(/[^\w\u4e00-\u9fff-]+/g, '-').slice(0, 56)}`;
}
