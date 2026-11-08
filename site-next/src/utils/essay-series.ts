import { getSlugParts, normalizeSlug } from './content';

export interface EssaySeriesMeta {
  key: string;
  /** 合集在 content 中的文件夹名，例如 `Life/北平/…` 里的 `北平`。 */
  folder: string;
  categoryFolder: string;
  title: string;
  subtitle: string;
  /** 相对 base 的合集页路径。 */
  href: string;
  kicker: string;
  description: string;
  epigraph: { text: string; source: string };
  /** 相对 base 的实拍配图（卡片图 4:3、合集页横幅 3:1）。 */
  coverImage: string;
  coverAlt: string;
  bannerImage: string;
  bannerAlt: string;
  photoCredit: string;
}

export const BEIPING_SERIES: EssaySeriesMeta = {
  key: 'beiping',
  folder: '北平',
  categoryFolder: 'Life',
  title: '北平',
  subtitle: '北京手记',
  href: 'essay/beiping/',
  kicker: 'Peiping papers',
  description: '关于一座城的前奏与回声：落叶、初雪，和还没兑现的抵达。',
  epigraph: {
    text: '谁道飘零不可怜，旧游时节好花天，断肠人去自经年',
    source: '纳兰性德《浣溪沙》',
  },
  coverImage: 'images/beiping/corner-tower-card.jpg',
  coverAlt: '故宫西北角楼与筒子河，秋末',
  bannerImage: 'images/beiping/corner-tower-banner.jpg',
  bannerAlt: '故宫西北角楼与筒子河，秋末的北京',
  photoCredit: 'N509FZ · CC BY-SA 4.0',
};

const SERIES_BY_FOLDER: EssaySeriesMeta[] = [BEIPING_SERIES];

export interface EssayLikeEntry {
  id: string;
  data: { title: string; description?: string; updatedAt?: Date; date?: Date };
  body?: string;
}

/** 条目 id 形如 `Life/北平/北平（前）`，第二段是合集文件夹名。 */
export function getSeriesForEntryId(id: string): EssaySeriesMeta | undefined {
  const parts = getSlugParts(normalizeSlug(id));
  if (parts.length < 3) return undefined;
  return SERIES_BY_FOLDER.find(
    (series) => parts[0] === series.categoryFolder && parts[1] === series.folder,
  );
}

export function getSeriesEntries<T extends EssayLikeEntry>(
  series: EssaySeriesMeta,
  entries: T[],
): T[] {
  return entries.filter((entry) => getSeriesForEntryId(entry.id)?.key === series.key);
}

const CHAPTER_NUMERALS = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '十一', '十二'];

/** 标题带「（前）」的是序卷，其余按落笔顺序编号；indexAmongNumbered 传当前已编号数。 */
export function getChapterNumeral(title: string, indexAmongNumbered: number): string {
  if (/（\s*前\s*）|\(\s*前\s*\)/.test(title)) return '序';
  return CHAPTER_NUMERALS[indexAmongNumbered] ?? String(indexAmongNumbered + 1);
}

export function withSeriesBase(href: string, base = import.meta.env.BASE_URL): string {
  return `${base}${href}`.replace(/\/{2,}/g, '/');
}
