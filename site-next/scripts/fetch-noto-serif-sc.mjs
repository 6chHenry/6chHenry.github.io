// One-off helper: self-host Noto Serif SC (weights 400/600) formerly served by Google Fonts.
// Downloads the unicode-range sliced woff2 set, rewrites the CSS to local URLs and writes
// public/fonts/noto-serif-sc.css. Rerunnable; skips re-download when files exist.
import { execFileSync } from 'node:child_process';
import { promises as fs } from 'node:fs';
import path from 'node:path';

const OUT_DIR = path.resolve('public/fonts/noto-serif-sc');
const CSS_OUT = path.resolve('public/fonts/noto-serif-sc.css');
const WEIGHTS = [400, 600];
const UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36';

function curl(url, outFile) {
  execFileSync('curl', ['-s', '--fail', '-L', '--max-time', '60', '-A', UA, '-o', outFile, url], {
    stdio: ['ignore', 'ignore', 'pipe'],
  });
}

await fs.mkdir(OUT_DIR, { recursive: true });

let css = '';
for (const weight of WEIGHTS) {
  const url = `https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@${weight}&display=swap`;
  const tmp = path.join(OUT_DIR, `_src-${weight}.css`);
  console.log(`fetching css for weight ${weight} …`);
  curl(url, tmp);
  css += await fs.readFile(tmp, 'utf8');
  await fs.rm(tmp);
}

// Download every referenced woff2 (skip ones already fetched on reruns)
const urls = [...new Set([...css.matchAll(/url\((https:[^)]+)\)/g)].map((m) => m[1]))];
console.log(`${urls.length} font slices to ensure`);

let done = 0;
const CONCURRENCY = 8;
const queue = [...urls];
await Promise.all(
  Array.from({ length: CONCURRENCY }, async () => {
    for (;;) {
      const remote = queue.pop();
      if (!remote) return;
      const name = remote.split('/').pop().split('?')[0];
      const local = path.join(OUT_DIR, name);
      try {
        await fs.access(local);
      } catch {
        curl(remote, local);
      }
      done += 1;
      if (done % 50 === 0) console.log(`  ${done}/${urls.length}`);
    }
  }),
);

for (const remote of urls) {
  const name = remote.split('/').pop().split('?')[0];
  css = css.split(remote).join(`/fonts/noto-serif-sc/${name}`);
}
css = css.replace(/\bfont-display:\s*swap;/g, 'font-display: swap;');
await fs.writeFile(CSS_OUT, css);

const sizes = await Promise.all(urls.map((r) => fs.stat(path.join(OUT_DIR, r.split('/').pop().split('?')[0]))));
const totalMb = sizes.reduce((sum, s) => sum + s.size, 0) / 1024 / 1024;
console.log(`wrote ${CSS_OUT} with ${urls.length} slices, ${totalMb.toFixed(1)} MB total`);
