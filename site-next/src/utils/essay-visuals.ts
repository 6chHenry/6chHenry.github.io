export interface TravelGalleryLink {
  title: string;
  href: string;
  cover: string;
  coverAlt: string;
  note: string;
}

export interface TravelPosterData {
  galleryHref: string;
  galleryTitle: string;
  cities: string;
  photos: string[];
}

const TRAVEL_POSTERS: Record<string, TravelPosterData> = {
  'Travel/2026寒假三城游': {
    galleryHref: 'gallery/hubei-jiangxi/',
    galleryTitle: '鄂赣行记',
    cities: '武汉·庐山·南昌',
    photos: [
      'assets/gallery/photography/wuhan.assets/yellow_crane_tower.jpg',
      'assets/gallery/photography/wuhan.assets/snow_yellow_crane_tower.jpg',
      'assets/gallery/photography/lushan.assets/lushan_sunset.jpg',
      'assets/gallery/photography/nanchang.assets/tengwang_pavilion.jpg',
      'assets/gallery/photography/lushan.assets/lushan_snow.jpg',
    ],
  },
  'Travel/日本游记': {
    galleryHref: 'gallery/japan/',
    galleryTitle: '关西旅路',
    cities: '大阪·京都·奈良·宇治·神戸·姫路',
    photos: [
      'assets/gallery/photography/osaka.assets/glico.jpg',
      'assets/gallery/photography/kyoto.assets/kinkaku-ji.jpg',
      'assets/gallery/photography/nara.assets/deer.jpg',
      'assets/gallery/photography/kobe.assets/kobe_tower.jpg',
      'assets/gallery/photography/himeji.assets/himeji_castle.jpg',
      'assets/gallery/photography/kyoto.assets/kiyomizudera_temple.jpg',
    ],
  },
  'Travel/港澳手记': {
    galleryHref: 'gallery/bay-area/',
    galleryTitle: '粤港澳三城游',
    cities: '广州·澳门·香港',
    photos: [
      'assets/gallery/photography/hong-kong.assets/Victoria_Peak.JPG',
      'assets/gallery/photography/macau.assets/Landmark.JPEG',
      'assets/gallery/photography/guangzhou.assets/shamian_island.JPG',
      'assets/gallery/photography/hong-kong.assets/Metro_Sign.JPG',
      'assets/gallery/photography/macau.assets/Londonese.JPG',
    ],
  },
  'Travel/2026暑假四城游': {
    galleryHref: 'gallery/four-cities/',
    galleryTitle: '山之东西',
    cities: '太原·大同·青岛·烟台',
    photos: [
      'assets/gallery/photography/summer-2026-four-cities.assets/taiyuan_04.jpg',
      'assets/gallery/photography/summer-2026-four-cities.assets/datong_02.jpg',
      'assets/gallery/photography/summer-2026-four-cities.assets/datong_07.jpg',
      'assets/gallery/photography/summer-2026-four-cities.assets/qingdao_05.jpg',
      'assets/gallery/photography/summer-2026-four-cities.assets/qingdao_04.jpg',
      'assets/gallery/photography/summer-2026-four-cities.assets/yantai_03.jpg',
    ],
  },
};

const TRAVEL_GALLERY_LINKS: Record<string, TravelGalleryLink> = {
  'Travel/2026寒假三城游': {
    title: '鄂赣行记',
    href: 'gallery/hubei-jiangxi/',
    cover: 'assets/gallery/photography/wuhan.assets/yellow_crane_tower.jpg',
    coverAlt: '雪后黄鹤楼与长江',
    note: '武汉、庐山、南昌，路上拍下来的。',
  },
  'Travel/日本游记': {
    title: '关西旅路',
    href: 'gallery/japan/',
    cover: 'assets/gallery/photography/osaka.assets/glico.jpg',
    coverAlt: '大阪道顿堀格力高招牌',
    note: '大阪、京都、奈良、宇治、神户、姬路，照片里的那条路。',
  },
  'Travel/港澳手记': {
    title: '粤港澳三城游',
    href: 'gallery/bay-area/',
    cover: 'assets/gallery/photography/hong-kong.assets/Victoria_Peak.JPG',
    coverAlt: '太平山俯瞰维港',
    note: '广州、香港、澳门，路上拍的。',
  },
  'Travel/香港红绿灯': {
    title: '香港',
    href: 'gallery/photography/hong-kong/',
    cover: 'assets/gallery/photography/hong-kong.assets/Metro_Sign.JPG',
    coverAlt: '香港地铁标识',
    note: '香港，街和声音。',
  },
  'Travel/2026暑假四城游': {
    title: '山之东西',
    href: 'gallery/four-cities/',
    cover: 'assets/gallery/photography/summer-2026-four-cities.assets/qingdao_05.jpg',
    coverAlt: '崂山之巅俯瞰海湾与索道',
    note: '太原、大同、青岛、烟台，从山里走到海边。',
  },
};

function trimBase(base: string): string {
  return base.endsWith('/') ? base : `${base}/`;
}

function withBase(path: string, base: string): string {
  if (/^(https?:|mailto:|#|\/)/.test(path)) return path;
  return `${trimBase(base)}${path}`;
}

export function getTravelPoster(entryId: string, base = import.meta.env.BASE_URL): (TravelPosterData & { galleryHref: string }) | undefined {
  const normalizedId = entryId.replace(/\.md$/, '');
  const item = TRAVEL_POSTERS[normalizedId];
  if (!item) return undefined;
  return {
    ...item,
    galleryHref: withBase(item.galleryHref, base),
    photos: item.photos.map((p) => withBase(p, base)),
  };
}

export function getTravelGalleryLink(entryId: string, base = import.meta.env.BASE_URL): TravelGalleryLink | undefined {
  const item = TRAVEL_GALLERY_LINKS[entryId.replace(/\.md$/, '')];
  if (!item) return undefined;

  return {
    ...item,
    href: withBase(item.href, base),
    cover: withBase(item.cover, base),
  };
}

/* 海报一律走本站静态资源，外链图源（维基共享、门户 CDN）在境内常被墙或失效 */
const MOVIE_POSTER_FALLBACK: Record<string, string> = {
  'Movies/3 Idiots': 'images/posters/3_idiots.jpg',
  'Movies/Hachi': 'images/posters/hachi.jpg',
  'Movies/机器人之梦': 'images/posters/robot-dreams.webp',
  'Movies/河狸变身计划': 'images/watchlist/hoppers.webp',
  'Movies/给阿嬷的情书': 'images/watchlist/dear-grandma.webp',
  'Movies/绿色大门': 'images/posters/blue-gate-crossing.webp',
  'Movies/欢迎来龙餐馆': 'images/watchlist/dragon-restaurant.webp',
  'Movies/肖申克的救赎': 'images/watchlist/shawshank-redemption.webp',
};

export function getMoviePosterFromBody(
  body = '',
  entryId = '',
  base = import.meta.env.BASE_URL,
): string | undefined {
  const image = body.match(/!\[[^\]]*]\(([^)\s]+)(?:\s+"[^"]*")?\)/);
  if (image) return image[1];
  if (entryId) {
    const normalizedId = entryId.replace(/\.md$/, '');
    const poster = MOVIE_POSTER_FALLBACK[normalizedId];
    if (poster) return withBase(poster, base);
  }
  return undefined;
}

/**
 * Extract the first image from essay body for use as card cover / background.
 * Returns null for movie entries (they use posters separately).
 */
export function getEntryCoverImage(body = ''): string | null {
  const image = body.match(/!\[[^\]]*]\(([^)\s]+)(?:\s+"[^"]*")?\)/);
  if (!image) return null;
  return image[1];
}

export function buildEssayExcerpt(body = '', fallback = ''): string {
  const cleaned = body
    .replace(/^---[\s\S]*?---/g, ' ')
    .replace(/!\[[^\]]*]\([^)]+\)/g, ' ')
    .replace(/\[[^\]]+]\([^)]+\)/g, (match) => match.replace(/^\[|\]\([^)]+\)$/g, ''))
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/^#{1,6}\s+/gm, ' ')
    .replace(/[>*_`~#-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  const source = cleaned || fallback;
  if (source.length <= 74) return source;
  return `${source.slice(0, 72).replace(/[，,、：:；;。.!！?？\s]+$/g, '')}...`;
}
