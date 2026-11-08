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

/* 字形以 (x, baselineY) 锚定且 text-anchor: middle。CJK 视觉盒约在
   基线上方 86% / 下方 14%，再留 12% 余量给小幅旋转；
   四个角都落在树冠内，这个字才算放得下。 */
function glyphFits(x: number, baselineY: number, size: number): boolean {
  const half = size * 0.56;
  const cy = baselineY - size * 0.36;
  return (
    pointInPolygon(x - half, cy - half, TREE_POLYGON) &&
    pointInPolygon(x + half, cy - half, TREE_POLYGON) &&
    pointInPolygon(x - half, cy + half, TREE_POLYGON) &&
    pointInPolygon(x + half, cy + half, TREE_POLYGON)
  );
}

const SIZE_LADDER = [26, 22, 18.5, 16, 14, 12.4, 11, 9.6, 8.4, 7.4];

interface TreeSlot {
  x: number;
  y: number;
  /** 这个槽位塞得下的最大字号 */
  size: number;
}

/** 网格撒点 + 抖动，逐个测出可容纳的最大字号；只保留读得清的槽位。 */
function buildSlots(
  rng: () => number,
  stepX: number,
  stepY: number,
  xStart: number,
  xEnd: number,
  yStart: number,
  yEnd: number,
): TreeSlot[] {
  const slots: TreeSlot[] = [];
  for (let y = yStart; y <= yEnd; y += stepY) {
    for (let x = xStart; x <= xEnd; x += stepX) {
      const px = x + (rng() - 0.5) * stepX * 0.5;
      const py = y + (rng() - 0.5) * stepY * 0.5;
      const size = SIZE_LADDER.find((candidate) => glyphFits(px, py, candidate));
      if (size !== undefined) slots.push({ x: px, y: py, size });
    }
  }
  return slots;
}

/** 均匀取 count 个（items 需已按空间顺序排好），保证疏密一致。 */
function pickEvenly<T>(items: T[], count: number): T[] {
  if (items.length <= count) return items;
  if (count <= 1) return [items[Math.floor(items.length / 2)]];
  const out: T[] = [];
  for (let i = 0; i < count; i += 1) {
    out.push(items[Math.round((i * (items.length - 1)) / (count - 1))]);
  }
  return out;
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
  /** 返回 undefined 时该字不生成链接（例如语料库里还没有它的 forest 页）。 */
  hrefForChar?: (char: string) => string | undefined;
}

interface GlyphVisual {
  char: string;
  count: number;
  source: 'title' | 'heading' | 'body';
  heading?: string;
  rank: number;
  href?: string;
}

function renderGlyph(slot: TreeSlot, glyph: GlyphVisual, size: number, rotation: number, opacity: number, fill: string): string {
  const x = slot.x.toFixed(1);
  const y = slot.y.toFixed(1);
  const text = `<text class="character-tree__glyph" data-char="${escapeSvgText(glyph.char)}" data-count="${glyph.count}" data-source="${glyph.source}" data-heading="${escapeSvgText(glyph.heading ?? '')}" data-rank="${glyph.rank.toFixed(3)}" x="${x}" y="${y}" text-anchor="middle" fill="${fill}" font-size="${Math.min(size, slot.size).toFixed(1)}" transform="rotate(${rotation.toFixed(1)} ${x} ${y})" font-family="var(--ch-text-font)" style="--char-opacity:${opacity.toFixed(2)}">${escapeSvgText(glyph.char)}</text>`;
  return glyph.href
    ? `<a class="character-tree__link" href="${escapeSvgText(glyph.href)}" aria-label="查看汉字「${escapeSvgText(glyph.char)}」">${text}</a>`
    : text;
}

function pickFill(rng: () => number): string {
  const tone = rng();
  return tone > 0.66 ? 'var(--ch-accent-secondary)' : tone > 0.33 ? 'var(--ch-accent-tertiary)' : 'var(--ch-accent)';
}

function buildYearTreeSvg({
  seed,
  portraits,
  clipId,
  hrefForChar,
}: Required<Pick<CharacterTreeOptions, 'seed' | 'clipId'>> & Pick<CharacterTreeOptions, 'portraits' | 'hrefForChar'>): string {
  const list = portraits ?? [];
  const rng = createRng(hashString(seed));
  /* 密网格撒点后再均匀取用：从树顶往下占满整个树冠；
     字量不够时轮转复用，让年轮长满一棵树 */
  const dense = buildSlots(rng, 4.5, 4.8, 18, 102, 24, 124).sort((a, b) => a.y - b.y || a.x - b.x);
  const slots = pickEvenly(dense, Math.min(dense.length, 36));
  const maxScore = Math.max(...list.map((item) => item.score), 0.001);
  const fallback = ['林', '间', '年', '轮'];

  const labels = slots
    .map((slot, index) => {
      const portrait = list.length > 0 ? list[index % list.length] : undefined;
      const char = portrait?.char ?? fallback[index % fallback.length];
      const rank = portrait ? portrait.score / maxScore : 0.3;
      const size = Math.min(slot.size, 11.5 + rank * 10.5);
      const rotation = (rng() - 0.5) * 6;
      const glyph: GlyphVisual = {
        char,
        count: portrait?.count ?? 1,
        source: portrait ? (portrait.inTitle ? 'title' : portrait.inHeading ? 'heading' : 'body') : 'body',
        heading: portrait?.heading,
        rank,
        href: hrefForChar?.(char),
      };
      return renderGlyph(slot, glyph, size, rotation, 0.72 + rank * 0.28, pickFill(rng));
    })
    .join('');

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
  /* 密网格覆盖整棵树（含树干），再均匀取样，避免某一层扎堆、某一层空着 */
  const dense = buildSlots(rng, 4, 4.5, 14, 106, 10, 154).sort((a, b) => a.y - b.y || a.x - b.x);
  const count = Math.min(dense.length, pool.length, compact ? 28 : 110);
  const chosen = pickEvenly(dense, count);
  const glyphs = pickEvenly(pool, count);

  const labels = chosen
    .map((slot, index) => {
      const glyph = glyphs[index];
      const size = compact ? 7.5 + rng() * 4 : 10.5 + rng() * 5.5;
      const rotation = (rng() - 0.5) * (compact ? 8 : 10);
      const opacity = compact ? 0.58 + rng() * 0.34 : 0.52 + rng() * 0.4;
      const visual: GlyphVisual = {
        char: glyph.char,
        count: glyph.count,
        source: glyph.source,
        heading: glyph.heading,
        rank: glyph.rank,
        href: glyph.distinctive ? hrefForChar?.(glyph.char) : undefined,
      };
      return renderGlyph(slot, visual, size, rotation, opacity, pickFill(rng));
    })
    .join('');

  return `<svg class="character-tree__svg" viewBox="0 0 120 160" focusable="false" xmlns="http://www.w3.org/2000/svg"><defs><clipPath id="${escapeSvgText(clipId)}"><path d="${treePath()} Z" /></clipPath></defs><path class="character-tree__silhouette" d="${treePath()} Z" fill="none" /><g clip-path="url(#${escapeSvgText(clipId)})">${labels}</g></svg>`;
}

export function treeClipId(entryId: string, prefix = 'tree'): string {
  return `${prefix}-${entryId.replace(/[^\w\u4e00-\u9fff-]+/g, '-').slice(0, 56)}`;
}
