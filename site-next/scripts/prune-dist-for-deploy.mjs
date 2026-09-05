#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.resolve(__dirname, '../dist');

const PRUNE_PATHS = [
  'assets/gallery-original',
];

function removeIfExists(target) {
  if (!fs.existsSync(target)) return false;
  fs.rmSync(target, { recursive: true, force: true });
  return true;
}

function main() {
  if (!fs.existsSync(DIST)) {
    console.warn('Skip dist prune: dist directory not found.');
    return;
  }

  let removed = 0;
  for (const rel of PRUNE_PATHS) {
    const target = path.join(DIST, rel);
    if (removeIfExists(target)) {
      removed += 1;
      console.log(`Pruned deploy artifact: dist/${rel}`);
    }
  }

  if (removed === 0) {
    console.log('No deploy artifacts needed pruning.');
  }
}

main();
