# Repository Guidelines

## Project Structure & Module Organization

This repository hosts a personal static site. The active production site is `site-next/`, an Astro project. Source code lives under `site-next/src/`: pages in `src/pages/`, layouts in `src/layouts/`, UI in `src/components/`, utilities in `src/utils/`, scripts in `src/scripts/`, styles in `src/styles/`, and migrated content in `src/content/`. Static assets live in `site-next/public/`.

Legacy MkDocs content remains in `docs/`, with theme overrides in `overrides/` and artifacts in root-level `assets/`, `search/`, `index.html`, and `404.html`. Treat `docs/` as the canonical Markdown source migrated into Astro.

Gallery source content lives in `docs/gallery/` and is migrated into the Astro `gallery` content collection at `site-next/src/content/gallery/`. Gallery pages live in `site-next/src/pages/gallery/`, shared gallery UI in `site-next/src/components/Gallery*.astro`, gallery browser scripts in `site-next/src/scripts/gallery.ts` and `site-next/src/scripts/japan-gallery.ts`, and gallery styles in `site-next/src/styles/gallery.css` and `site-next/src/styles/japan-gallery.css`. Gallery image asset folders should sit next to their source Markdown as `*.assets/`; migration copies them to `site-next/public/assets/gallery/`.

## Build, Test, and Development Commands

Run frontend commands from `site-next/`.

- `npm install`: install Astro, Pagefind, and TypeScript dependencies.
- `npm run dev`: start the local Astro development server.
- `npm run migrate`: copy Markdown and assets from `../docs` into Astro content collections.
- `npm run build`: run migration hooks, build the site, and generate/sync the Pagefind index.
- `npm run preview`: preview the production build locally.

For legacy MkDocs work, install Python dependencies with `pip install -r requirements.txt` and use `mkdocs serve` or `mkdocs build`.

For gallery updates, edit the source Markdown and adjacent assets under `docs/gallery/`, then run `npm run migrate` or `npm run build` from `site-next/` so `site-next/src/content/gallery/` and `site-next/public/assets/gallery/` stay in sync.

## Coding Style & Naming Conventions

Use TypeScript and Astro patterns already present in `site-next/src`. Prefer PascalCase component filenames, for example `SearchModal.astro`; use kebab-case for CSS such as `audio-player.css`; keep utility modules lowercase or kebab-case. Use two-space indentation in Astro, TypeScript, CSS, JSON, and Markdown frontmatter. Preserve existing Chinese content filenames when appropriate.

Gallery Markdown requires frontmatter compatible with `site-next/src/content/config.ts`: `title`, optional `description`, `date`, `updatedAt`, `tags`, `draft`, `legacyPath`, required `cover`, optional `coverDesc`, `images`, `imageDescs`, and `category` (`illustration`, `photography`, `design`, `ui-ux`, or `other`). Keep gallery image URLs rooted at `/assets/gallery/...` so they match migrated public assets.

## Testing Guidelines

There is no dedicated test suite. Validate changes with `npm run build` before opening a PR. For UI, gallery, or search changes, also run `npm run dev` and manually check navigation, content pages, theme switching, math rendering, audio embeds, gallery filtering, gallery lightbox behavior, the Japan immersive gallery, and Pagefind search. When editing migration scripts, compare generated files in `site-next/src/content/` and `site-next/public/`.

## Commit & Pull Request Guidelines

Recent history uses short, imperative, sentence-style subjects, for example `Fix Pagefind search by loading the UI script correctly...` and `Replace MkDocs with Astro as the production GitHub Pages site.` Keep commits focused.

Pull requests should include a summary, affected areas (`docs`, `site-next`, migration, search, styling), verification commands, and screenshots for visible UI changes. Link related issues when available and note generated content or asset synchronization.

## Agent-Specific Instructions

Avoid editing generated build output unless the task explicitly requires it. Prefer changing source content in `docs/` or Astro source files in `site-next/src/`, then run the appropriate migration or build command. For gallery entries, update `docs/gallery/` first and let migration refresh the Astro content and public gallery assets.
