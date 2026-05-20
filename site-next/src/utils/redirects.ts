// Legacy MkDocs path -> new Astro path (without base prefix)
export const legacyRedirects: Record<string, string> = {
  '/notes/': '/notes/',
  '/essay/': '/essay/',
  '/essay/日本游记/': '/essay/日本游记/',
  '/about/': '/about/',
  '/projects/': '/projects/',
};

export function resolveLegacyRedirect(pathname: string): string | null {
  const normalized = pathname.endsWith('/') ? pathname : `${pathname}/`;
  return legacyRedirects[normalized] ?? null;
}
