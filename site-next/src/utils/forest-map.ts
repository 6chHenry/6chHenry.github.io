import { buildCharacterTreeSvg } from './character-tree';
import { buildContentUrl } from './content';

export const FOREST_MAP_WIDTH = 1400;
export const FOREST_MAP_HEIGHT = 560;
export const FOREST_MAP_MAX_STOPS = 18;

export interface ForestMapEntry {
  id: string;
  collection: string;
  body: string;
  data: {
    title: string;
    date?: Date;
    updatedAt?: Date;
  };
}

export interface ForestMapStop {
  href: string;
  title: string;
  dateLabel: string;
  collection: string;
  treeSvg: string;
  x: number;
  y: number;
}

export function formatMapDate(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}/${month}/${day}`;
}

export function getCreationTimestamp(entry: ForestMapEntry): number {
  return entry.data.date?.getTime() ?? entry.data.updatedAt?.getTime() ?? 0;
}

export function sortByCreation<T extends ForestMapEntry>(entries: T[]): T[] {
  return [...entries].sort((a, b) => getCreationTimestamp(a) - getCreationTimestamp(b));
}

/** Pick chronological milestones: always earliest + latest, evenly sample the rest. */
export function selectForestMapEntries<T extends ForestMapEntry>(
  entries: T[],
  maxStops = FOREST_MAP_MAX_STOPS,
): T[] {
  const dated = sortByCreation(entries).filter((entry) => getCreationTimestamp(entry) > 0);
  if (dated.length <= maxStops) return dated;

  const indices = new Set<number>([0, dated.length - 1]);
  const slots = maxStops - 2;

  for (let i = 1; i <= slots; i += 1) {
    indices.add(Math.round((i / (slots + 1)) * (dated.length - 1)));
  }

  return [...indices]
    .sort((a, b) => a - b)
    .map((index) => dated[index]);
}

/** Winding trail coordinate for normalized progress 0..1 (left = past, right = present). */
export function computeTrailPoint(t: number): { x: number; y: number } {
  const clamped = Math.min(1, Math.max(0, t));
  const x = 96 + clamped * (FOREST_MAP_WIDTH - 192);
  const wave =
    Math.sin(clamped * Math.PI * 2.35) * 78 +
    Math.sin(clamped * Math.PI * 4.8 + 0.55) * 26;
  const y = FOREST_MAP_HEIGHT * 0.54 + wave;
  return { x, y };
}

export function buildTrailPath(points: Array<{ x: number; y: number }>): string {
  if (points.length === 0) return '';
  if (points.length === 1) {
    return `M ${points[0].x.toFixed(1)} ${points[0].y.toFixed(1)}`;
  }

  let path = `M ${points[0].x.toFixed(1)} ${points[0].y.toFixed(1)}`;
  for (let i = 0; i < points.length - 1; i += 1) {
    const current = points[i];
    const next = points[i + 1];
    const midX = (current.x + next.x) / 2;
    path += ` C ${midX.toFixed(1)} ${current.y.toFixed(1)}, ${midX.toFixed(1)} ${next.y.toFixed(1)}, ${next.x.toFixed(1)} ${next.y.toFixed(1)}`;
  }
  return path;
}

export function buildForestMapStops(entries: ForestMapEntry[], base = import.meta.env.BASE_URL): ForestMapStop[] {
  const selected = selectForestMapEntries(entries);
  const count = selected.length;

  return selected.map((entry, index) => {
    const t = count === 1 ? 0.5 : index / (count - 1);
    const { x, y } = computeTrailPoint(t);
    const timestamp = getCreationTimestamp(entry);
    const date = new Date(timestamp);
    const clipId = `map-tree-${entry.collection}-${index}`;

    return {
      href: buildContentUrl(entry.collection, entry.id, base),
      title: entry.data.title,
      dateLabel: formatMapDate(date),
      collection: entry.collection,
      treeSvg: buildCharacterTreeSvg({
        seed: entry.id,
        title: entry.data.title,
        body: entry.body,
        clipId,
        variant: 'compact',
      }),
      x,
      y,
    };
  });
}
