import type { CollectionEntry } from 'astro:content';
import { CITY_STOPS, GALLERY_TRIP_SEQUENCE, TRIP_CITIES } from './gallery-trips';
import { buildContentUrl, formatDate } from '../utils/content';
import { resolveGalleryImage, type GalleryImageAsset } from '../utils/gallery-images';

export interface AtlasPlace {
  id: string;
  name: string;
  en: string;
  region: string;
  lat: number;
  lng: number;
  trip: string;
  tripTitle: string;
  date: string;
  detailUrl: string;
  images: GalleryImageAsset[];
  imageDescs: string[];
}

// Regional anchors, not camera GPS. Albums include nearby excursions (e.g. 应县).
// Add an anchor here when publishing a new city; unrecognized chapters stay unmapped.
const anchors: Record<string, { lat: number; lng: number; region: string }> = {
  wuhan: { lat: 30.59, lng: 114.30, region: '湖北 · 武汉' },
  lushan: { lat: 29.56, lng: 115.98, region: '江西 · 庐山' },
  nanchang: { lat: 28.68, lng: 115.86, region: '江西 · 南昌' },
  geermu: { lat: 36.40, lng: 94.90, region: '青海 · 格尔木及周边' },
  guangzhou: { lat: 23.13, lng: 113.26, region: '广东 · 广州' },
  macau: { lat: 22.20, lng: 113.54, region: '澳门' },
  'hong-kong': { lat: 22.32, lng: 114.17, region: '香港' },
  osaka: { lat: 34.69, lng: 135.50, region: '日本 · 大阪' },
  nara: { lat: 34.68, lng: 135.83, region: '日本 · 奈良' },
  uji: { lat: 34.89, lng: 135.80, region: '日本 · 宇治' },
  kyoto: { lat: 35.01, lng: 135.77, region: '日本 · 京都' },
  kobe: { lat: 34.69, lng: 135.20, region: '日本 · 神户' },
  himeji: { lat: 34.83, lng: 134.69, region: '日本 · 姬路' },
  taiyuan: { lat: 37.87, lng: 112.55, region: '山西 · 太原' },
  datong: { lat: 40.09, lng: 113.30, region: '山西 · 大同及周边' },
  qingdao: { lat: 36.07, lng: 120.38, region: '山东 · 青岛' },
  yantai: { lat: 37.46, lng: 121.45, region: '山东 · 烟台' },
};

export function buildGalleryAtlas(entries: CollectionEntry<'gallery'>[], base: string) {
  const published = entries.filter(({ data }) => !data.draft && data.category === 'photography');
  const bySlug = new Map(published.map((entry) => [entry.id.split('/').pop()!.replace(/\.md$/, ''), entry]));
  const groups = new Map<string, AtlasPlace>();
  const seen = new Set<string>();
  const consumed = new Set<string>();
  let unmappedCount = 0;
  const trips: { slug: string; title: string }[] = [];

  const collect = (entry: CollectionEntry<'gallery'>, member: string, trip: string, title: string, href: string) => {
    consumed.add(member);
    const sources = [entry.data.cover, ...entry.data.images];
    const descriptions = [entry.data.coverDesc || '', ...entry.data.imageDescs];
    sources.forEach((source, index) => {
      if (seen.has(source)) return;
      seen.add(source);
      const city = (index === 0 ? entry.data.coverChapter : entry.data.imageChapters[index - 1]) || member;
      const anchor = anchors[city];
      const stop = CITY_STOPS[city];
      if (!anchor || !stop) {
        unmappedCount += 1;
        return;
      }
      const id = `${trip}-${city}`;
      let group = groups.get(id);
      if (!group) {
        group = {
          id, name: stop.kanji, en: stop.en, ...anchor,
          trip, tripTitle: title, detailUrl: href,
          date: entry.data.date ? formatDate(entry.data.date) : '',
          images: [], imageDescs: [],
        };
        groups.set(id, group);
      }
      group.images.push(resolveGalleryImage(source, base));
      group.imageDescs.push(descriptions[index] || '');
    });
  };

  for (const trip of GALLERY_TRIP_SEQUENCE) {
    // Trip overview covers repeat city photos; consume only member albums.
    consumed.add(trip.slug);
    for (const member of TRIP_CITIES[trip.slug] || [trip.slug]) {
      const entry = bySlug.get(member);
      if (entry) collect(entry, member, trip.slug, trip.title, `${base}${trip.href}`);
    }
    if (Array.from(groups.values()).some((place) => place.trip === trip.slug)) {
      trips.push({ slug: trip.slug, title: trip.title });
    }
  }
  for (const [slug, entry] of bySlug) {
    if (consumed.has(slug)) continue;
    collect(entry, slug, slug, entry.data.title, buildContentUrl('gallery', entry.id, base));
    if (Array.from(groups.values()).some((place) => place.trip === slug)) {
      trips.push({ slug, title: entry.data.title });
    }
  }
  return { places: Array.from(groups.values()), trips, unmappedCount };
}
