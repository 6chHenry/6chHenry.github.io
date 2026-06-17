#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SITE_ROOT = path.resolve(__dirname, '..');
const REPO_ROOT = path.resolve(SITE_ROOT, '..');

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const noPush = args.includes('--no-push') || dryRun;
const allowNonMain = args.includes('--allow-non-main');

function readArgValue(name) {
  const index = args.indexOf(name);
  if (index === -1) return undefined;
  return args[index + 1];
}

const explicitMessage = readArgValue('--message') ?? readArgValue('-m');

const SOURCE_COLLECTIONS = new Map([
  ['notes', 'notes'],
  ['essay', 'essay'],
  ['summary', 'essay'],
  ['projects', 'projects'],
  ['gallery', 'gallery'],
]);

let allowedPaths = ['docs'];

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd ?? REPO_ROOT,
    encoding: options.encoding ?? 'utf8',
    stdio: options.stdio ?? 'pipe',
    shell: false,
  });
  if (result.status !== 0) {
    const detail = [result.stdout, result.stderr].filter(Boolean).join('\n').trim();
    throw new Error(`Command failed: ${command} ${args.join(' ')}${detail ? `\n${detail}` : ''}`);
  }
  return result.stdout ?? '';
}

function runInherit(command, args, cwd) {
  const result = spawnSync(command, args, {
    cwd,
    stdio: 'inherit',
    shell: process.platform === 'win32',
  });
  if (result.status !== 0) {
    throw new Error(`Command failed: ${command} ${args.join(' ')}`);
  }
}

function parsePorcelain(output) {
  if (!output) return [];
  const records = output.split('\0').filter(Boolean);
  const entries = [];
  for (let i = 0; i < records.length; i++) {
    const record = records[i];
    const status = record.slice(0, 2);
    let file = record.slice(3);
    if (status.startsWith('R') || status.startsWith('C')) {
      file = records[i + 1] ?? file;
      i += 1;
    }
    entries.push({ status, file: file.replace(/\\/g, '/') });
  }
  return entries;
}

function gitStatus(paths) {
  return parsePorcelain(run('git', ['status', '--porcelain=v1', '-z', '-uall', '--', ...paths]));
}

function gitDiffNameOnly(args) {
  return run('git', ['diff', '--name-only', '-z', ...args])
    .split('\0')
    .filter(Boolean)
    .map((file) => file.replace(/\\/g, '/'));
}

function isAllowedPath(file) {
  const normalized = file.replace(/\\/g, '/');
  return allowedPaths.some((allowed) => normalized === allowed || normalized.startsWith(`${allowed}/`));
}

function printEntries(title, entries) {
  console.log(title);
  for (const entry of entries) {
    console.log(`  ${entry.status} ${entry.file}`);
  }
}

function titleFromMarkdown(file) {
  try {
    const raw = fs.readFileSync(path.join(REPO_ROOT, file), 'utf8');
    const title = raw.match(/^title:\s*["']?(.+?)["']?\s*$/m)?.[1]?.trim();
    if (title) return title;
  } catch {
    // Fall back to filename below.
  }
  return path.basename(file, path.extname(file));
}

function defaultCommitMessage(docsEntries) {
  const markdownChanges = docsEntries
    .map((entry) => entry.file)
    .filter((file) => file.endsWith('.md'));
  if (markdownChanges.length === 1) {
    return `Publish content: ${titleFromMarkdown(markdownChanges[0])}`;
  }
  return 'Publish content updates';
}

function ensureNoDisallowedChanges(entries, phase) {
  const disallowed = entries.filter((entry) => !isAllowedPath(entry.file));
  if (disallowed.length === 0) return;
  printEntries(`Refusing to continue: unrelated changes detected ${phase}.`, disallowed);
  throw new Error('Commit or stash unrelated changes first.');
}

function stripExtension(file) {
  return file.replace(/\.[^.]+$/, '');
}

function derivedGeneratedPaths(docsEntries) {
  const paths = new Set();
  for (const entry of docsEntries) {
    const file = entry.file.replace(/\\/g, '/');
    if (!file.startsWith('docs/')) continue;

    const rel = file.slice('docs/'.length);
    if (rel === 'academy.md') {
      paths.add('site-next/public/academy-legacy.md');
      continue;
    }
    if (rel.startsWith('academy.assets/')) {
      paths.add(`site-next/public/${rel}`);
      continue;
    }
    if (rel.startsWith('audio/')) {
      paths.add(`site-next/public/${rel}`);
      continue;
    }

    const [sourceDir, ...restParts] = rel.split('/');
    const collection = SOURCE_COLLECTIONS.get(sourceDir);
    if (!collection || restParts.length === 0) continue;

    const sourceRel = restParts.join('/');
    if (sourceRel.endsWith('.md')) {
      paths.add(`site-next/src/content/${collection}/${stripExtension(sourceRel)}.md`);
      if (collection === 'notes') paths.add('site-next/src/data/notes-nav.json');
      continue;
    }

    const assetIndex = restParts.findIndex((part) => part.endsWith('.assets'));
    if (assetIndex !== -1 && collection !== 'gallery') {
      paths.add(`site-next/public/assets/${collection}/${sourceRel}`);
      if (collection === 'notes') paths.add('site-next/src/data/notes-nav.json');
    }
  }
  return [...paths];
}

function main() {
  const branch = run('git', ['branch', '--show-current']).trim();
  if (!branch) throw new Error('Not currently on a branch.');
  if (!allowNonMain && branch !== 'main') {
    throw new Error(`Refusing to publish from '${branch}'. Switch to main or pass --allow-non-main.`);
  }

  console.log('Checking docs changes...');
  const docsEntries = gitStatus(['docs']);
  if (docsEntries.length === 0) {
    throw new Error('No changes found under docs/. Nothing to publish.');
  }
  printEntries('Docs changes to publish:', docsEntries);
  allowedPaths = ['docs', ...derivedGeneratedPaths(docsEntries)];

  const currentEntries = gitStatus(['.']);
  ensureNoDisallowedChanges(currentEntries, 'before build');

  if (branch === 'main') {
    console.log('Checking remote main ancestry...');
    run('git', ['fetch', 'origin', 'main'], { stdio: 'inherit' });
    const ancestor = spawnSync('git', ['merge-base', '--is-ancestor', 'origin/main', 'HEAD'], {
      cwd: REPO_ROOT,
      stdio: 'ignore',
      shell: false,
    });
    if (ancestor.status !== 0) {
      throw new Error('Local main does not contain origin/main. Pull/rebase first, then rerun.');
    }
  }

  console.log('Building site...');
  runInherit(process.platform === 'win32' ? 'npm.cmd' : 'npm', ['run', 'build'], SITE_ROOT);

  console.log('Staging docs and generated content outputs...');
  if (!dryRun) {
    run('git', ['add', '-A', '--', ...allowedPaths]);
  }

  const stagedFiles = dryRun
    ? [...new Set(gitStatus(allowedPaths).map((entry) => entry.file))]
    : gitDiffNameOnly(['--cached']);
  const disallowedStaged = stagedFiles.filter((file) => !isAllowedPath(file));
  if (disallowedStaged.length > 0) {
    console.log('Disallowed staged files:');
    for (const file of disallowedStaged) console.log(`  ${file}`);
    throw new Error('Refusing to commit files outside docs/generated content paths.');
  }

  if (stagedFiles.length === 0) {
    console.log('No effective changes after build.');
    return;
  }

  console.log('Staged files:');
  for (const file of stagedFiles) console.log(`  ${file}`);

  const remainingEntries = gitStatus(['.']);
  ensureNoDisallowedChanges(remainingEntries, 'after staging');

  const message = explicitMessage ?? defaultCommitMessage(docsEntries);
  if (dryRun) {
    console.log(`Dry run: would commit with message: ${message}`);
    return;
  }

  console.log(`Committing: ${message}`);
  run('git', ['commit', '-m', message], { stdio: 'inherit' });

  if (noPush) {
    console.log('Skipping push.');
    return;
  }

  console.log(`Pushing ${branch} to origin...`);
  run('git', ['push', 'origin', branch], { stdio: 'inherit' });
}

try {
  main();
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
