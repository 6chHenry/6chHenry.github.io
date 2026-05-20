# Site deployment guide

## Current state

- **Astro (production)**: served at `https://6chHenry.github.io/`
- **MkDocs (legacy)**: source of truth for markdown in `docs/`; no longer deployed by default

Every push to `main` runs an Astro-only deploy. MkDocs can still be built manually for rollback via workflow dispatch.

## Local development

```bash
cd site-next
npm install
npm run dev
```

Content is synced from `docs/` via `npm run migrate` (runs automatically before build).

## Rollback to MkDocs

1. GitHub Actions → **Deploy Site** → **Run workflow**
2. Choose `mkdocs-only` or `hybrid` (Astro preview at `/next/`)

## Hybrid preview (legacy)

Run workflow with `hybrid` to publish MkDocs at root and Astro at `/next/` (uses `SITE_BASE=/next`).

## Preview URL redirect

Bookmarks to `/next/...` from the hybrid rollout are redirected to the matching root path via `404.html`.
