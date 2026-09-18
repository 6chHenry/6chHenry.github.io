export interface GalleryChapter {
  slug: string;
  kanji: string;
  en: string;
  start: number;
  count: number;
}

export const CITY_STOPS: Record<string, { kanji: string; en: string }> = {
  osaka: { kanji: '大阪', en: 'Osaka' },
  nara: { kanji: '奈良', en: 'Nara' },
  uji: { kanji: '宇治', en: 'Uji' },
  kyoto: { kanji: '京都', en: 'Kyoto' },
  kobe: { kanji: '神戸', en: 'Kobe' },
  himeji: { kanji: '姫路', en: 'Himeji' },
  guangzhou: { kanji: '广州', en: 'Guangzhou' },
  macau: { kanji: '澳门', en: 'Macau' },
  'hong-kong': { kanji: '香港', en: 'Hong Kong' },
  wuhan: { kanji: '武汉', en: 'Wuhan' },
  lushan: { kanji: '庐山', en: 'Lushan' },
  nanchang: { kanji: '南昌', en: 'Nanchang' },
  taiyuan: { kanji: '太原', en: 'Taiyuan' },
  datong: { kanji: '大同', en: 'Datong' },
  qingdao: { kanji: '青岛', en: 'Qingdao' },
  yantai: { kanji: '烟台', en: 'Yantai' },
  geermu: { kanji: '格尔木', en: 'Golmud' },
};

/* 多城旅程由多个城市条目组成，照片数要逐城累加；单城旅程就是它自己 */
export const TRIP_CITIES: Record<string, string[]> = {
  'hubei-jiangxi': ['wuhan', 'lushan', 'nanchang'],
  geermu: ['geermu'],
  'greater-bay-area': ['guangzhou', 'macau', 'hong-kong'],
  japan: ['osaka', 'nara', 'uji', 'kyoto', 'kobe', 'himeji'],
  'summer-2026-four-cities': ['summer-2026-four-cities'],
};

export function cityStop(slug: string): { kanji: string; en: string } {
  return CITY_STOPS[slug] ?? { kanji: slug, en: slug };
}

export function mergeChapters(keys: string[]): GalleryChapter[] {
  const chapters: GalleryChapter[] = [];
  for (const [index, key] of keys.entries()) {
    if (!key) continue;
    const last = chapters[chapters.length - 1];
    if (last && last.slug === key) {
      last.count += 1;
      continue;
    }
    const stop = cityStop(key);
    chapters.push({
      slug: key,
      kanji: stop.kanji,
      en: stop.en,
      start: index,
      count: 1,
    });
  }
  return chapters;
}
