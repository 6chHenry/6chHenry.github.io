/** Seeded PRNG (mulberry32). */
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

function isCjk(char: string): boolean {
  const code = char.charCodeAt(0);
  return (
    (code >= 0x4e00 && code <= 0x9fff) ||
    (code >= 0x3400 && code <= 0x4dbf) ||
    (code >= 0xf900 && code <= 0xfaff)
  );
}

/** Collect weighted Chinese characters from note content. */
export function collectCharacterPool(
  title: string,
  headings: string[],
  body: string,
  limit = 180,
): string[] {
  const weights = new Map<string, number>();

  const addText = (text: string, weight: number) => {
    for (const char of text.replace(/\s+/g, '')) {
      if (!isCjk(char)) continue;
      weights.set(char, (weights.get(char) ?? 0) + weight);
    }
  };

  addText(title, 6);
  for (const heading of headings) addText(heading, 4);

  const cjkWords = body.match(/[\u4e00-\u9fff]{2,4}/g) ?? [];
  for (const word of cjkWords) addText(word, 2);

  const cjkChars = body.match(/[\u4e00-\u9fff]/g) ?? [];
  for (const char of cjkChars) addText(char, 1);

  if (weights.size === 0) {
    for (const char of '林间笔记森树') addText(char, 1);
  }

  const expanded: string[] = [];
  for (const [char, weight] of weights) {
    const copies = Math.min(8, Math.max(1, Math.round(Math.sqrt(weight))));
    for (let i = 0; i < copies; i += 1) expanded.push(char);
  }

  const rng = createRng(hashString(`${title}:${body.length}`));
  for (let i = expanded.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [expanded[i], expanded[j]] = [expanded[j], expanded[i]];
  }

  const pool: string[] = [];
  while (pool.length < limit && expanded.length > 0) {
    pool.push(...expanded);
  }

  return pool.slice(0, limit);
}

/** Simple pine-tree polygon for hit testing (viewBox 0 0 120 160). */
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

export interface CharacterTreeOptions {
  seed: string;
  title: string;
  headings?: string[];
  body: string;
  clipId?: string;
  variant?: 'default' | 'compact';
}

function escapeSvgText(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function buildEntryCharacterTree(
  entry: { id: string; body: string; data: { title: string } },
  headingTexts: string[] = [],
): string {
  const clipId = `tree-${entry.id.replace(/[^\w\u4e00-\u9fff-]+/g, '-').slice(0, 56)}`;
  return buildCharacterTreeSvg({
    seed: entry.id,
    title: entry.data.title,
    headings: headingTexts,
    body: entry.body,
    clipId,
  });
}

export function buildCharacterTreeSvg({
  seed,
  title,
  headings = [],
  body,
  clipId = 'tree-clip',
  variant = 'default',
}: CharacterTreeOptions): string {
  const compact = variant === 'compact';
  const rng = createRng(hashString(seed));
  const pool = collectCharacterPool(title, headings, body, compact ? 48 : 180);
  const slots: Array<{ x: number; y: number }> = [];

  for (let y = 10; y <= 150; y += 9) {
    for (let x = 18; x <= 102; x += 9) {
      const jitterX = (rng() - 0.5) * 4;
      const jitterY = (rng() - 0.5) * 4;
      const px = x + jitterX;
      const py = y + jitterY;
      if (pointInPolygon(px, py, TREE_POLYGON)) {
        slots.push({ x: px, y: py });
      }
    }
  }

  for (let i = slots.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [slots[i], slots[j]] = [slots[j], slots[i]];
  }

  const count = Math.min(slots.length, pool.length, compact ? 28 : 110);
  const treePath = TREE_POLYGON.map(([x, y], index) => `${index === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`).join(' ');

  const labels = Array.from({ length: count }, (_, index) => {
    const slot = slots[index];
    const char = pool[index];
    const size = compact ? 7.5 + rng() * 4 : 10.5 + rng() * 5.5;
    const rotation = (rng() - 0.5) * (compact ? 8 : 10);
    const opacity = compact ? 0.58 + rng() * 0.34 : 0.52 + rng() * 0.4;
    const tone = rng();
    const fill =
      tone > 0.66 ? 'var(--ch-accent-secondary)' : tone > 0.33 ? 'var(--ch-accent-tertiary)' : 'var(--ch-accent)';

    return `<text x="${slot.x.toFixed(1)}" y="${slot.y.toFixed(1)}" fill="${fill}" font-size="${size.toFixed(1)}" opacity="${opacity.toFixed(2)}" transform="rotate(${rotation.toFixed(1)} ${slot.x.toFixed(1)} ${slot.y.toFixed(1)})" font-family="var(--ch-text-font)">${escapeSvgText(char)}</text>`;
  }).join('');

  return `<svg class="character-tree__svg" viewBox="0 0 120 160" aria-hidden="true" focusable="false" xmlns="http://www.w3.org/2000/svg"><defs><clipPath id="${clipId}"><path d="${treePath} Z" /></clipPath></defs><g clip-path="url(#${clipId})">${labels}</g></svg>`;
}
