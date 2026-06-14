import fs from 'node:fs';

export interface GalleryImageAsset {
  src: string;
  webpSrcset: string;
  width?: number;
  height?: number;
  original: string;
}

interface GalleryImageManifest {
  images?: Record<string, GalleryImageAsset>;
}

const manifestUrl = new URL('../../public/assets/gallery-manifest.json', import.meta.url);
let manifest: GalleryImageManifest = {};

try {
  manifest = JSON.parse(fs.readFileSync(manifestUrl, 'utf8')) as GalleryImageManifest;
} catch {
  manifest = {};
}

function withBase(url: string, base: string): string {
  if (!url || /^(https?:|data:|blob:|mailto:|#)/.test(url)) return url;
  const normalizedBase = base.endsWith('/') ? base.slice(0, -1) : base;
  const normalizedUrl = url.startsWith('/') ? url : `/${url}`;
  return `${normalizedBase}${normalizedUrl}` || '/';
}

function withoutBase(url: string, base: string): string {
  if (!url.startsWith('/')) return url;
  const normalizedBase = base.endsWith('/') ? base.slice(0, -1) : base;
  if (normalizedBase && url.startsWith(`${normalizedBase}/`)) {
    return url.slice(normalizedBase.length);
  }
  return url;
}

function withBaseSrcset(srcset: string, base: string): string {
  return srcset
    .split(',')
    .map((candidate) => {
      const [url, descriptor] = candidate.trim().split(/\s+/, 2);
      return descriptor ? `${withBase(url, base)} ${descriptor}` : withBase(url, base);
    })
    .join(', ');
}

export function resolveGalleryImage(
  source: string,
  base = import.meta.env.BASE_URL,
): GalleryImageAsset {
  const lookupKey = withoutBase(source, base);
  const asset = manifest.images?.[lookupKey];
  if (!asset) {
    return {
      src: withBase(source, base),
      webpSrcset: '',
      original: withBase(source, base),
    };
  }
  return {
    ...asset,
    src: withBase(asset.src, base),
    webpSrcset: withBaseSrcset(asset.webpSrcset, base),
    original: withBase(asset.original, base),
  };
}

export function resolveGalleryImages(
  sources: string[],
  base = import.meta.env.BASE_URL,
): GalleryImageAsset[] {
  return sources.map((source) => resolveGalleryImage(source, base));
}
