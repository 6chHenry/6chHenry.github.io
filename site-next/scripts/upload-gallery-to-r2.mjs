#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { S3Client, HeadObjectCommand, PutObjectCommand } from '@aws-sdk/client-s3';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SITE_ROOT = path.resolve(__dirname, '..');
const REPO_ROOT = path.resolve(SITE_ROOT, '..');
const SOURCE_ROOT = path.join(REPO_ROOT, 'docs/gallery');
const ENV_PATH = path.join(SITE_ROOT, '.env');

const IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.avif', '.gif']);

function loadEnv(file) {
  if (!fs.existsSync(file)) return;
  for (const line of fs.readFileSync(file, 'utf8').split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const index = trimmed.indexOf('=');
    if (index === -1) continue;
    const key = trimmed.slice(0, index).trim();
    const value = trimmed.slice(index + 1).trim();
    if (!(key in process.env)) process.env[key] = value;
  }
}

function requireEnv(name) {
  const value = process.env[name]?.trim();
  if (!value) throw new Error(`Missing required env var: ${name} (set it in site-next/.env)`);
  return value;
}

function walkImages(dir, files = []) {
  if (!fs.existsSync(dir)) return files;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walkImages(full, files);
    else if (IMAGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) files.push(full);
  }
  return files;
}

function contentType(file) {
  switch (path.extname(file).toLowerCase()) {
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg';
    case '.png':
      return 'image/png';
    case '.webp':
      return 'image/webp';
    case '.avif':
      return 'image/avif';
    case '.gif':
      return 'image/gif';
    default:
      return 'application/octet-stream';
  }
}

async function exists(client, bucket, key) {
  try {
    await client.send(new HeadObjectCommand({ Bucket: bucket, Key: key }));
    return true;
  } catch {
    return false;
  }
}

async function main() {
  loadEnv(ENV_PATH);

  const accountId = requireEnv('R2_ACCOUNT_ID');
  const accessKeyId = requireEnv('R2_ACCESS_KEY_ID');
  const secretAccessKey = requireEnv('R2_SECRET_ACCESS_KEY');
  const bucket = requireEnv('R2_BUCKET_NAME');
  const prefix = (process.env.R2_PREFIX ?? 'gallery').replace(/^\/+|\/+$/g, '');

  const client = new S3Client({
    region: 'auto',
    endpoint: `https://${accountId}.r2.cloudflarestorage.com`,
    credentials: { accessKeyId, secretAccessKey },
  });

  const sourceFiles = walkImages(SOURCE_ROOT).sort((a, b) => a.localeCompare(b, 'en'));
  let uploaded = 0;
  let skipped = 0;

  for (const sourceFile of sourceFiles) {
    const rel = path.relative(SOURCE_ROOT, sourceFile).replace(/\\/g, '/');
    const key = prefix ? `${prefix}/${rel}` : rel;
    if (await exists(client, bucket, key)) {
      skipped += 1;
      continue;
    }

    await client.send(
      new PutObjectCommand({
        Bucket: bucket,
        Key: key,
        Body: fs.createReadStream(sourceFile),
        ContentType: contentType(sourceFile),
      }),
    );
    uploaded += 1;
    console.log(`Uploaded ${key}`);
  }

  const baseUrl = process.env.GALLERY_ORIGINAL_BASE_URL?.replace(/\/$/, '');
  console.log(`Done: ${uploaded} uploaded, ${skipped} skipped, ${sourceFiles.length} total.`);
  if (baseUrl) {
    console.log(`Public originals base: ${baseUrl}`);
    console.log('Set GALLERY_ORIGINAL_BASE_URL in GitHub Actions variables to enable download links in production.');
  } else {
    console.log('Tip: set GALLERY_ORIGINAL_BASE_URL in site-next/.env after enabling the bucket public URL.');
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
