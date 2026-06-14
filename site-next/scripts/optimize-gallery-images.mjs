#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SITE_ROOT = path.resolve(__dirname, '..');
const REPO_ROOT = path.resolve(SITE_ROOT, '..');
const SOURCE_ROOT = path.join(REPO_ROOT, 'docs/gallery');
const PUBLIC_ROOT = path.join(SITE_ROOT, 'public');
const BROWSE_ROOT = path.join(PUBLIC_ROOT, 'assets/gallery');
const ORIGINAL_ROOT = path.join(PUBLIC_ROOT, 'assets/gallery-original');
const MANIFEST_PATH = path.join(PUBLIC_ROOT, 'assets/gallery-manifest.json');
const CACHE_PATH = path.join(SITE_ROOT, '.cache/gallery-images.json');

const CONFIG_VERSION = 1;
const WEBP_WIDTHS = [640, 1280, 2048, 3200];
const WEBP_QUALITY = 86;
const JPEG_QUALITY = 88;
const FALLBACK_MAX_EDGE = 3200;
const CONCURRENCY = Math.max(1, Math.min(3, Number(process.env.GALLERY_IMAGE_CONCURRENCY) || 2));
const IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.avif']);

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function readJson(file, fallback) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch {
    return fallback;
  }
}

function toPosix(value) {
  return value.replace(/\\/g, '/');
}

function walkFiles(dir, predicate, files = []) {
  if (!fs.existsSync(dir)) return files;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walkFiles(full, predicate, files);
    else if (entry.isFile() && predicate(full)) files.push(full);
  }
  return files;
}

function walkImages(dir) {
  return walkFiles(dir, (file) => IMAGE_EXTENSIONS.has(path.extname(file).toLowerCase()));
}

function walkAllFiles(dir) {
  return walkFiles(dir, () => true);
}

function fileSize(file) {
  try {
    return fs.statSync(file).size;
  } catch {
    return 0;
  }
}

function directorySize(dir) {
  return walkAllFiles(dir).reduce((sum, file) => sum + fileSize(file), 0);
}

function formatMb(bytes) {
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function cacheKey(stat) {
  return `${CONFIG_VERSION}:${stat.size}:${Math.round(stat.mtimeMs)}`;
}

function urlToPublicFile(url) {
  return path.join(PUBLIC_ROOT, decodeURIComponent(url.replace(/^\//, '')));
}

function outputExists(asset) {
  const urls = [
    asset.src,
    asset.original,
    ...asset.webpSrcset.split(',').map((item) => item.trim().split(/\s+/)[0]),
  ];
  return urls.every((url) => fs.existsSync(urlToPublicFile(url)));
}

async function writeFallback(sourceFile, destFile) {
  ensureDir(path.dirname(destFile));
  const extension = path.extname(sourceFile).toLowerCase();
  let pipeline = sharp(sourceFile)
    .rotate()
    .resize({
      width: FALLBACK_MAX_EDGE,
      height: FALLBACK_MAX_EDGE,
      fit: 'inside',
      withoutEnlargement: true,
    });

  if (extension === '.jpg' || extension === '.jpeg') {
    pipeline = pipeline.jpeg({ quality: JPEG_QUALITY, mozjpeg: true });
  } else if (extension === '.png') {
    pipeline = pipeline.png({ compressionLevel: 9, adaptiveFiltering: true });
  } else if (extension === '.webp') {
    pipeline = pipeline.webp({ quality: WEBP_QUALITY, effort: 5 });
  } else if (extension === '.avif') {
    pipeline = pipeline.avif({ quality: 65, effort: 5 });
  }

  await pipeline.toFile(destFile);
  return sharp(destFile).metadata();
}

async function writeWebpVariants(sourceFile, relPath, sourceWidth) {
  const parsed = path.parse(relPath);
  const widths = WEBP_WIDTHS.filter((width) => width <= sourceWidth);
  widths.push(Math.min(sourceWidth, WEBP_WIDTHS.at(-1)));
  const variants = [];

  for (const requestedWidth of [...new Set(widths)].sort((a, b) => a - b)) {
    const filename = `${parsed.name}.${requestedWidth}.webp`;
    const relOutput = path.join(parsed.dir, filename);
    const output = path.join(BROWSE_ROOT, relOutput);
    ensureDir(path.dirname(output));
    const info = await sharp(sourceFile)
      .rotate()
      .resize({ width: requestedWidth, withoutEnlargement: true })
      .webp({ quality: WEBP_QUALITY, effort: 5, smartSubsample: true })
      .toFile(output);
    variants.push({
      url: `/assets/gallery/${toPosix(relOutput)}`,
      width: info.width,
    });
  }

  return variants
    .filter((variant, index, items) => items.findIndex((item) => item.width === variant.width) === index)
    .sort((a, b) => a.width - b.width);
}

async function processImage(sourceFile, previousCache) {
  const relPath = path.relative(SOURCE_ROOT, sourceFile);
  const relPosix = toPosix(relPath);
  const stat = fs.statSync(sourceFile);
  const key = cacheKey(stat);
  const cached = previousCache.files?.[relPosix];
  if (cached?.key === key && cached.asset && outputExists(cached.asset)) {
    return { relPath: relPosix, key, asset: cached.asset, reused: true };
  }

  const sourceMetadata = await sharp(sourceFile).metadata();
  const orientationSwapsAxes = sourceMetadata.orientation && sourceMetadata.orientation >= 5;
  const sourceWidth = orientationSwapsAxes ? sourceMetadata.height : sourceMetadata.width;
  if (!sourceWidth) throw new Error(`Unable to read image width: ${sourceFile}`);

  const fallbackFile = path.join(BROWSE_ROOT, relPath);
  const fallbackMetadata = await writeFallback(sourceFile, fallbackFile);
  const webpVariants = await writeWebpVariants(sourceFile, relPath, sourceWidth);

  const originalFile = path.join(ORIGINAL_ROOT, relPath);
  ensureDir(path.dirname(originalFile));
  fs.copyFileSync(sourceFile, originalFile);

  return {
    relPath: relPosix,
    key,
    reused: false,
    asset: {
      src: `/assets/gallery/${relPosix}`,
      webpSrcset: webpVariants.map((variant) => `${variant.url} ${variant.width}w`).join(', '),
      width: fallbackMetadata.width ?? sourceMetadata.width,
      height: fallbackMetadata.height ?? sourceMetadata.height,
      original: `/assets/gallery-original/${relPosix}`,
    },
  };
}

async function mapConcurrent(items, worker, concurrency) {
  const results = new Array(items.length);
  let cursor = 0;
  async function run() {
    while (cursor < items.length) {
      const index = cursor++;
      results[index] = await worker(items[index], index);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, run));
  return results;
}

function cleanUnexpectedFiles(root, expected) {
  for (const file of walkAllFiles(root)) {
    if (!expected.has(path.resolve(file))) fs.rmSync(file, { force: true });
  }
  if (!fs.existsSync(root)) return;
  const directories = [];
  const collect = (dir) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const full = path.join(dir, entry.name);
      collect(full);
      directories.push(full);
    }
  };
  collect(root);
  for (const dir of directories) {
    if (fs.existsSync(dir) && fs.readdirSync(dir).length === 0) fs.rmdirSync(dir);
  }
}

function expectedFilesFor(results) {
  const browse = new Set();
  const originals = new Set();
  for (const result of results) {
    browse.add(path.resolve(urlToPublicFile(result.asset.src)));
    originals.add(path.resolve(urlToPublicFile(result.asset.original)));
    for (const item of result.asset.webpSrcset.split(',')) {
      browse.add(path.resolve(urlToPublicFile(item.trim().split(/\s+/)[0])));
    }
  }
  return { browse, originals };
}

async function main() {
  const started = performance.now();
  ensureDir(BROWSE_ROOT);
  ensureDir(ORIGINAL_ROOT);
  ensureDir(path.dirname(MANIFEST_PATH));
  ensureDir(path.dirname(CACHE_PATH));

  const sourceFiles = walkImages(SOURCE_ROOT).sort((a, b) => a.localeCompare(b, 'en'));
  const previousCache = readJson(CACHE_PATH, { version: CONFIG_VERSION, files: {} });
  const results = await mapConcurrent(
    sourceFiles,
    (sourceFile) => processImage(sourceFile, previousCache),
    CONCURRENCY,
  );

  const images = {};
  const files = {};
  for (const result of results) {
    images[`/assets/gallery/${result.relPath}`] = result.asset;
    files[result.relPath] = { key: result.key, asset: result.asset };
  }

  const expected = expectedFilesFor(results);
  cleanUnexpectedFiles(BROWSE_ROOT, expected.browse);
  cleanUnexpectedFiles(ORIGINAL_ROOT, expected.originals);

  fs.writeFileSync(
    MANIFEST_PATH,
    `${JSON.stringify({ version: CONFIG_VERSION, generatedAt: new Date().toISOString(), images }, null, 2)}\n`,
    'utf8',
  );
  fs.writeFileSync(
    CACHE_PATH,
    `${JSON.stringify({ version: CONFIG_VERSION, files }, null, 2)}\n`,
    'utf8',
  );

  const sourceBytes = sourceFiles.reduce((sum, file) => sum + fileSize(file), 0);
  const browseBytes = directorySize(BROWSE_ROOT);
  const originalBytes = directorySize(ORIGINAL_ROOT);
  const elapsedSeconds = (performance.now() - started) / 1000;
  const generated = results.filter((result) => !result.reused).length;

  console.log(`Gallery images: ${results.length} (${generated} generated, ${results.length - generated} reused)`);
  console.log(`Source originals: ${formatMb(sourceBytes)}`);
  console.log(`Browse assets: ${formatMb(browseBytes)} (${((browseBytes / sourceBytes) * 100).toFixed(1)}% of originals)`);
  console.log(`Original downloads: ${formatMb(originalBytes)}`);
  console.log(`Optimization time: ${elapsedSeconds.toFixed(2)}s with concurrency ${CONCURRENCY}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
