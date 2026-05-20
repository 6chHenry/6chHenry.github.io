# 6ch. personal site (Astro)

Production site built with Astro. Markdown sources live in `../docs/` and are synced before each build.

## Commands

```bash
cd site-next
npm install
npm run migrate      # copy markdown + assets from ../docs
npm run dev          # local dev server
npm run build        # build static site + Pagefind index
npm run preview      # preview production build
```

## Deployment

- Production: `https://6chHenry.github.io/` (auto-deploy on push to `main`)
- Rollback / hybrid preview: see [ROLLOUT.md](./ROLLOUT.md)

## Content collections

- `notes` — from `docs/notes`
- `essay` — from `docs/essay` and `docs/summary`
- `projects` — from `docs/projects`

Run `npm run migrate` before build to refresh content from MkDocs sources.
