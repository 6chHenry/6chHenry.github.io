import type { CharPortrait } from './character-lexicon';

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

export interface CharacterTreeOptions {
  seed: string;
  portraits: CharPortrait[];
  clipId?: string;
  variant?: 'default' | 'compact' | 'year';
  hrefForChar?: (char: string) => string;
}

export function buildCharacterTreeSvg({
  seed,
  portraits,
  clipId = 'tree-clip',
  variant = 'default',
  hrefForChar,
}: CharacterTreeOptions): string {
  const compact = variant === 'compact';
  const year = variant === 'year';
  if (portraits.length === 0 && !compact && !year) return '';
  const rng = createRng(hashString(seed));
  const canopy: Array<{ x: number; y: number }> = [];

  for (let y = 12; y <= (compact ? 100 : 112); y += compact ? 14 : 11) {
    for (let x = 22; x <= 98; x += compact ? 14 : 12) {
      const px = x + (rng() - 0.5) * 5;
      const py = y + (rng() - 0.5) * 5;
      if (pointInPolygon(px, py, TREE_POLYGON)) canopy.push({ x: px, y: py });
    }
  }

  canopy.sort((a, b) => a.y - b.y || a.x - b.x);
  const maxScore = Math.max(...portraits.map((item) => item.score), 0.001);
  const take = Math.min(portraits.length, compact ? 6 : year ? 14 : 12);
  const step = Math.max(1, Math.floor(canopy.length / Math.max(take, 1)));

  const treePath = TREE_POLYGON.map(([x, y], index) => `${index === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`).join(' ');
  const labels = portraits.slice(0, take).map((portrait, index) => {
    const slot = canopy[Math.min(index * step, canopy.length - 1)] ?? { x: 60, y: 48 };
    const rank = portrait.score / maxScore;
    const size = compact
      ? 8.5 + rank * 6
      : year
        ? 13 + rank * 11
        : 13.5 + rank * 12;
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

  return `<svg class="character-tree__svg" id="${escapeSvgText(clipId)}" viewBox="0 0 120 160" focusable="false" xmlns="http://www.w3.org/2000/svg"><path class="character-tree__silhouette" d="${treePath} Z" fill="none" /><g>${labels}</g></svg>`;
}

export function treeClipId(entryId: string, prefix = 'tree'): string {
  return `${prefix}-${entryId.replace(/[^\w\u4e00-\u9fff-]+/g, '-').slice(0, 56)}`;
}
