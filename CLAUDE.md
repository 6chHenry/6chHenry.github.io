# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See also `AGENTS.md` for project structure, commands, coding style, and PR guidelines. This file covers architecture that spans multiple files.

## Content Pipeline

Content is authored in `docs/` (canonical Markdown) and migrated into Astro at build time via `scripts/migrate-content.mjs`. The `prebuild` hook runs migration + `copy-public.mjs` before every `astro build`. Key behaviors of the migration script:

- **Admonition conversion**: `!!! note` / `!!! warning` / `!!! info` / `!!! question` / `!!! tip` blocks are converted to HTML `<div class="admonition ...">` with a title paragraph and nested content.
- **Image path rewriting**: Paths like `../images/` are rewritten to `../../public/images/`; `../../assets/images/` is handled.
- **Tag inference**: Directory structure implies tags (e.g., `docs/notes/CS/ML/` → tag `ML`). Tags can also be explicit in frontmatter.
- **Draft/hidden filtering**: Files with `draft: true`, `hidden: true`, or in `_private/` directories are skipped. Essays default to `draft: false`; notes default to `draft: true`.
- **Static asset copying**: `copy-public.mjs` copies `docs/images/`, `docs/audio/`, `docs/academy/` into `site-next/public/`.

## Content Collections & Schemas

Defined in `src/content/config.ts` using Zod. Three collections — `notes`, `essay`, `projects` — share a base schema:

```
title, description, date (ISO string), updatedAt?, tags[], draft, legacyPath?, summary?
```

Essays add `series?` and `seriesIndex?`. Projects add `repo?`, `demo?`, `status?` (enum: planned/in-progress/completed/archived), and `tech[]`. All collections use `glob()` with `../docs/**` patterns for type-safe content queries. Content is loaded via `getCollection()` and `getEntry()` in Astro pages.

## Design System (Forest Theme)

`src/styles/tokens.css` defines a cohesive CSS custom property system:

- **Forest-inspired palette**: `--color-void` (deep dark), `--color-sky` (light warm), `--color-hill-*` (green-brown series), `--color-canopy-*` (green series), `--color-gold-*` (warm accents)
- **Glass-morphism cards**: `--glass-bg`, `--glass-border`, `--glass-shadow`; surfacing on hover via `--glass-hover-*`
- **Typography**: Cormorant Garamond (Latin headings/serifs) + LXGW WenKai (CJK body), with fallback stacks in `--font-serif`, `--font-sans`, `--font-mono`
- **Light/dark modes** via `[data-theme="dark"]` on `<html>`, toggled by `src/scripts/theme.ts`
- **Spatial tokens**: `--space-*`, `--radius-*`, `--text-*` size scale

Components use these tokens rather than hardcoded values. Backdrop gradients and header glass effects are in `global.css`.

## Chinese Character Trees

A unique visualization: `src/utils/character-tree.ts` extracts Chinese characters from article content (weighted by position: title > headings > body), builds a frequency map, and generates an SVG tree where each leaf/particle is a character. Used by `CharacterTree.astro` on article pages and by `ForestMap.astro` / `forest-map.ts` on the homepage. The homepage forest map positions content entries as trees grouped into "groves" (Notes, Essay, Project), with a golden path connecting recently updated entries.

## Pagefind Search

Search indexing runs after `astro build` via `npx pagefind --site dist`. The `sync-pagefind.mjs` script then copies the index back to `site-next/public/pagefind/` so `npm run dev` can serve it (Pagefind regenerates on rebuild, the sync prevents missing index during dev).

Two search surfaces, both mounting via `src/scripts/search.ts`:
- **SearchModal.astro**: Overlay modal triggered by header button or Ctrl+K
- **`/search` page**: Full-page search at `pages/search.astro`

## Path Aliases & URL Strategy

- TypeScript path alias `@/*` → `src/*` (configured in `tsconfig.json`)
- `trailingSlash: 'always'` in Astro config
- Base path defaults to `/` but can be set via `SITE_BASE` env var (e.g., `/next` for hybrid preview)
- Legacy redirects: `src/utils/redirects.ts` maps old MkDocs URLs to new Astro routes; essay pages generate redirect entries in `getStaticPaths()`

## Deploy Modes (GitHub Actions)

`deploy-site.yml` supports three modes:
- **astro-only** (default on push): Builds only the Astro site, deploys to root of `gh-pages`
- **hybrid**: Builds Astro at `/next/` + MkDocs at `/`, combined into one deploy directory
- **mkdocs-only**: Legacy fallback, builds only MkDocs

The Astro `prebuild` hook runs via `npm run prebuild` which calls `migrate-content.mjs` then `copy-public.mjs`.

## Notes Navigation

`src/data/notes-nav.json` is auto-generated from the filesystem structure of content files. `src/utils/notes-nav.ts` provides tree-building utilities. `NotesNavItem.astro` renders the collapsible sidebar tree on notes pages, driven by `src/scripts/notes-tree.ts` for client-side expand/collapse interaction.
