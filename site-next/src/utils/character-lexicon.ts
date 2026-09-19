import { getCollection } from 'astro:content';
import { buildContentUrl } from './content';

export function isCjk(char: string): boolean {
  const code = char.charCodeAt(0);
  return (
    (code >= 0x4e00 && code <= 0x9fff) ||
    (code >= 0x3400 && code <= 0x4dbf) ||
    (code >= 0xf900 && code <= 0xfaff)
  );
}

export interface CharStat {
  count: number;
  inTitle: boolean;
  inHeading: boolean;
  heading?: string;
}

export interface CharPortrait {
  char: string;
  count: number;
  score: number;
  inTitle: boolean;
  inHeading: boolean;
  heading?: string;
}

export interface CorpusDoc {
  key: string;
  collection: string;
  id: string;
  title: string;
  href: string;
  date?: Date;
  portraits: CharPortrait[];
  weights: Map<string, CharStat>;
}

export interface CharacterCorpus {
  docs: CorpusDoc[];
  df: Map<string, number>;
  docCount: number;
  byKey: Map<string, CorpusDoc>;
  inverted: Map<string, Array<{ key: string; title: string; href: string; collection: string; count: number; score: number }>>;
}

const STOP_DF = 0.72;
const SPARSE_UNIQUE = 8;

export function extractMarkdownHeadings(body: string): string[] {
  const plain = body.replace(/```[\s\S]*?```/g, '\n');
  return [...plain.matchAll(/^#{1,3}\s+(.+)$/gm)].map((match) =>
    match[1].replace(/[#*`[\]]/g, '').trim(),
  );
}

export function weighDocument(title: string, headings: string[], body: string): Map<string, CharStat> {
  const weights = new Map<string, CharStat>();

  const bump = (text: string, amount: number, flags: { inTitle?: boolean; inHeading?: boolean; heading?: string }) => {
    for (const char of text.replace(/\s+/g, '')) {
      if (!isCjk(char)) continue;
      const prev = weights.get(char) ?? { count: 0, inTitle: false, inHeading: false };
      prev.count += amount;
      if (flags.inTitle) prev.inTitle = true;
      if (flags.inHeading) {
        prev.inHeading = true;
        if (!prev.heading && flags.heading) prev.heading = flags.heading;
      }
      weights.set(char, prev);
    }
  };

  bump(title, 6, { inTitle: true });
  for (const heading of headings) bump(heading, 4, { inHeading: true, heading });
  bump(body.replace(/```[\s\S]*?```/g, '\n'), 1, {});
  return weights;
}

export function isSparseCjk(weights: Map<string, CharStat>): boolean {
  let total = 0;
  for (const stat of weights.values()) total += stat.count;
  return weights.size < SPARSE_UNIQUE || total < 24;
}

export function pickPortraits(
  weights: Map<string, CharStat>,
  df: Map<string, number>,
  docCount: number,
  limit = 12,
): CharPortrait[] {
  if (isSparseCjk(weights)) return [];

  const scored: CharPortrait[] = [];
  for (const [char, stat] of weights) {
    const dfCount = df.get(char) ?? 1;
    const idf = Math.log((docCount + 1) / (dfCount + 0.5));
    let score = stat.count * idf;
    if (stat.inTitle) score *= 1.35;
    if (stat.inHeading) score *= 1.15;
    if (docCount > 4 && dfCount / docCount > STOP_DF) score *= 0.12;
    if (score <= 0) continue;
    scored.push({
      char,
      count: stat.count,
      score,
      inTitle: stat.inTitle,
      inHeading: stat.inHeading,
      heading: stat.heading,
    });
  }

  scored.sort((a, b) => b.score - a.score || b.count - a.count);
  return scored.slice(0, limit);
}

type CorpusSource = {
  id: string;
  collection: string;
  body?: string;
  data: { title: string; date?: Date; draft?: boolean };
};

export function buildCorpusFromEntries(entries: CorpusSource[], base = '/'): CharacterCorpus {
  const prepared = entries.map((entry) => {
    const headings = extractMarkdownHeadings(entry.body ?? '');
    const weights = weighDocument(entry.data.title, headings, entry.body ?? '');
    return { entry, weights };
  });

  const df = new Map<string, number>();
  for (const { weights } of prepared) {
    for (const char of weights.keys()) df.set(char, (df.get(char) ?? 0) + 1);
  }

  const docCount = Math.max(1, prepared.length);
  const docs: CorpusDoc[] = prepared.map(({ entry, weights }) => {
    const portraits = pickPortraits(weights, df, docCount);
    return {
      key: `${entry.collection}:${entry.id}`,
      collection: entry.collection,
      id: entry.id,
      title: entry.data.title,
      href: buildContentUrl(entry.collection, entry.id, base),
      date: entry.data.date,
      portraits,
      weights,
    };
  });

  const inverted = new Map<string, Array<{ key: string; title: string; href: string; collection: string; count: number; score: number }>>();
  for (const doc of docs) {
    for (const portrait of doc.portraits) {
      const list = inverted.get(portrait.char) ?? [];
      list.push({
        key: doc.key,
        title: doc.title,
        href: doc.href,
        collection: doc.collection,
        count: portrait.count,
        score: portrait.score,
      });
      inverted.set(portrait.char, list);
    }
  }
  for (const list of inverted.values()) {
    list.sort((a, b) => b.score - a.score || b.count - a.count);
  }

  return {
    docs,
    df,
    docCount,
    byKey: new Map(docs.map((doc) => [doc.key, doc])),
    inverted,
  };
}

export function portraitsForRendered(
  corpus: CharacterCorpus,
  entry: { id: string; collection: string; body?: string; data: { title: string } },
  headings: string[],
  limit = 12,
): CharPortrait[] {
  const weights = weighDocument(entry.data.title, headings, entry.body ?? '');
  return pickPortraits(weights, corpus.df, corpus.docCount, limit);
}

let cached: CharacterCorpus | null = null;
let cachedBase = '';

export async function loadCharacterCorpus(base = '/'): Promise<CharacterCorpus> {
  if (cached && cachedBase === base) return cached;
  const [notes, essays, projects] = await Promise.all([
    getCollection('notes', ({ data }) => !data.draft),
    getCollection('essay', ({ data }) => !data.draft),
    getCollection('projects', ({ data }) => !data.draft),
  ]);
  cached = buildCorpusFromEntries(
    [
      ...notes.map((entry) => ({ ...entry, collection: 'notes' })),
      ...essays.map((entry) => ({ ...entry, collection: 'essay' })),
      ...projects.map((entry) => ({ ...entry, collection: 'projects' })),
    ],
    base,
  );
  cachedBase = base;
  return cached;
}

export function yearPortraits(corpus: CharacterCorpus, year: number, limit = 14): CharPortrait[] {
  const pooled = new Map<string, CharStat>();
  for (const doc of corpus.docs) {
    if (!doc.date || doc.date.getFullYear() !== year) continue;
    for (const [char, stat] of doc.weights) {
      const prev = pooled.get(char) ?? { count: 0, inTitle: false, inHeading: false };
      prev.count += stat.count;
      prev.inTitle = prev.inTitle || stat.inTitle;
      prev.inHeading = prev.inHeading || stat.inHeading;
      if (!prev.heading && stat.heading) prev.heading = stat.heading;
      pooled.set(char, prev);
    }
  }
  if (pooled.size === 0) {
    for (const doc of corpus.docs) {
      for (const [char, stat] of doc.weights) {
        const prev = pooled.get(char) ?? { count: 0, inTitle: false, inHeading: false };
        prev.count += stat.count;
        pooled.set(char, prev);
      }
    }
  }
  return pickPortraits(pooled, corpus.df, corpus.docCount, limit);
}
