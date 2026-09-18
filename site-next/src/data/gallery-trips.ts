export interface CityStop {
  kanji: string;
  en: string;
  color: string;
}

export interface GalleryChapter {
  slug: string;
  kanji: string;
  en: string;
  color: string;
  start: number;
  count: number;
}

export const CITY_STOPS: Record<string, CityStop> = {
  osaka: { kanji: '大阪', en: 'Osaka', color: '#e60012' },
  nara: { kanji: '奈良', en: 'Nara', color: '#8b4513' },
  uji: { kanji: '宇治', en: 'Uji', color: '#4a9e6e' },
  kyoto: { kanji: '京都', en: 'Kyoto', color: '#5a9e4b' },
  kobe: { kanji: '神戸', en: 'Kobe', color: '#0066b3' },
  himeji: { kanji: '姫路', en: 'Himeji', color: '#9e9e9e' },
  guangzhou: { kanji: '广州', en: 'Guangzhou', color: '#c41e3a' },
  macau: { kanji: '澳门', en: 'Macau', color: '#2d7a5f' },
  'hong-kong': { kanji: '香港', en: 'Hong Kong', color: '#1a5276' },
  wuhan: { kanji: '武汉', en: 'Wuhan', color: '#4a90d9' },
  lushan: { kanji: '庐山', en: 'Lushan', color: '#2d6a4f' },
  nanchang: { kanji: '南昌', en: 'Nanchang', color: '#c0392b' },
  taiyuan: { kanji: '太原', en: 'Taiyuan', color: '#8a6a34' },
  datong: { kanji: '大同', en: 'Datong', color: '#96522c' },
  qingdao: { kanji: '青岛', en: 'Qingdao', color: '#1a5276' },
  yantai: { kanji: '烟台', en: 'Yantai', color: '#c2571d' },
  geermu: { kanji: '格尔木', en: 'Golmud', color: '#2d7a5f' },
};

/* 多城旅程由多个城市条目组成，照片数要逐城累加；单城旅程就是它自己 */
export const TRIP_CITIES: Record<string, string[]> = {
  'hubei-jiangxi': ['wuhan', 'lushan', 'nanchang'],
  geermu: ['geermu'],
  'greater-bay-area': ['guangzhou', 'macau', 'hong-kong'],
  japan: ['osaka', 'nara', 'uji', 'kyoto', 'kobe', 'himeji'],
  'summer-2026-four-cities': ['summer-2026-four-cities'],
};

export function cityStop(slug: string): CityStop {
  return CITY_STOPS[slug] ?? { kanji: slug, en: slug, color: '' };
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
      color: stop.color,
      start: index,
      count: 1,
    });
  }
  return chapters;
}

export interface GalleryTripLink {
  slug: string;
  title: string;
  en: string;
  subtitle: string;
  cover: string;
  href: string;
}

export const GALLERY_TRIP_SEQUENCE: GalleryTripLink[] = [
  {
    slug: 'hubei-jiangxi',
    title: '鄂赣行记',
    en: 'Hubei · Jiangxi',
    subtitle: '武汉 · 庐山 · 南昌',
    cover: '/assets/gallery/photography/wuhan.assets/yellow_crane_tower.jpg',
    href: 'gallery/hubei-jiangxi/',
  },
  {
    slug: 'geermu',
    title: '格尔木',
    en: 'Golmud',
    subtitle: '察尔汗盐湖 · 昆仑山',
    cover: '/assets/gallery/photography/geermu.assets/qarhan_saltlake_1.jpg',
    href: 'gallery/photography/geermu/',
  },
  {
    slug: 'greater-bay-area',
    title: '粤港澳三城游',
    en: 'Greater Bay Area',
    subtitle: '广州 · 澳门 · 香港',
    cover: '/assets/gallery/photography/guangzhou.assets/Yat-sen_Mausoleum_front.JPG',
    href: 'gallery/bay-area/',
  },
  {
    slug: 'japan',
    title: '関西旅路',
    en: 'Kansai Journey',
    subtitle: '大阪 · 奈良 · 宇治 · 京都 · 神户 · 姬路',
    cover: '/assets/gallery/photography/osaka.assets/glico.jpg',
    href: 'gallery/japan/',
  },
  {
    slug: 'summer-2026-four-cities',
    title: '山之东西',
    en: 'Shanxi · Shandong',
    subtitle: '太原 · 大同 · 青岛 · 烟台',
    cover: '/assets/gallery/photography/summer-2026-four-cities.assets/qingdao_05.jpg',
    href: 'gallery/four-cities/',
  },
];

export function nextGalleryTrip(slug: string): GalleryTripLink | null {
  const index = GALLERY_TRIP_SEQUENCE.findIndex((trip) => trip.slug === slug);
  if (index < 0) return null;
  return GALLERY_TRIP_SEQUENCE[index + 1] ?? null;
}

export function isGalleryTrip(slug: string): boolean {
  return GALLERY_TRIP_SEQUENCE.some((trip) => trip.slug === slug);
}
