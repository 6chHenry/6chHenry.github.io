# Paimon Case Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a polished Paimon Asks Everything personal case study page to the Astro projects section and feature it on the projects index.

**Architecture:** Keep the projects collection as the source of routing and metadata. Add a dedicated Astro component for this one premium case study and conditionally render it from the existing project detail route, while all other projects continue through `ContentLayout`. Copy the two provided local PNG assets into `site-next/public/assets/projects/paimon-asks-everything/` with ASCII names.

**Tech Stack:** Astro 5 content collections, Markdown frontmatter, scoped Astro CSS, existing `6ch.` design tokens, static public assets, `npm run build`.

## Global Constraints

- Work on branch `paimon-case-study`, not `main`.
- Do not modify generated build output.
- Do not touch unrelated migrated essay or notes content.
- Keep existing project pages on the generic `ContentLayout` path.
- Use `banner.png` as the primary hero visual.
- Copy the non-banner local PNG as `design-map.png`.
- No horizontal scroll at 375px width.
- Link buttons must be at least 44px tall.
- Images need descriptive alt text and stable aspect-ratio wrappers.
- Use restrained project-specific navy/gold styling while preserving existing forest editorial tokens.

---

### Task 1: Copy Stable Project Assets

**Files:**
- Create: `site-next/public/assets/projects/paimon-asks-everything/banner.png`
- Create: `site-next/public/assets/projects/paimon-asks-everything/design-map.png`

**Interfaces:**
- Produces: public URLs `/assets/projects/paimon-asks-everything/banner.png` and `/assets/projects/paimon-asks-everything/design-map.png`
- Consumes: local source files `F:\PAIMON\figures\banner.png` and the other PNG in `F:\PAIMON\figures`

- [ ] **Step 1: Create the asset directory**

Run:

```powershell
New-Item -ItemType Directory -Force 'site-next\public\assets\projects\paimon-asks-everything' | Out-Null
```

Expected: command exits successfully.

- [ ] **Step 2: Copy the banner with a stable name**

Run:

```powershell
Copy-Item -LiteralPath 'F:\PAIMON\figures\banner.png' -Destination 'site-next\public\assets\projects\paimon-asks-everything\banner.png' -Force
```

Expected: `site-next/public/assets/projects/paimon-asks-everything/banner.png` exists.

- [ ] **Step 3: Copy the mind map with an ASCII name**

Run:

```powershell
$mindMap = Get-ChildItem -LiteralPath 'F:\PAIMON\figures' -File -Filter '*.png' | Where-Object { $_.Name -ne 'banner.png' } | Select-Object -First 1
Copy-Item -LiteralPath $mindMap.FullName -Destination 'site-next\public\assets\projects\paimon-asks-everything\design-map.png' -Force
```

Expected: `site-next/public/assets/projects/paimon-asks-everything/design-map.png` exists.

- [ ] **Step 4: Verify asset sizes**

Run:

```powershell
Get-ChildItem -LiteralPath 'site-next\public\assets\projects\paimon-asks-everything' | Select-Object Name,Length
```

Expected: both files are present and larger than 1 MB.

### Task 2: Add The Project Collection Entry

**Files:**
- Create: `site-next/src/content/projects/PaimonAsksEverything/paimon-asks-everything.md`

**Interfaces:**
- Produces: projects collection entry id `PaimonAsksEverything/paimon-asks-everything.md`
- Produces: generated page URL `/projects/PaimonAsksEverything/paimon-asks-everything/`
- Consumes: existing `projects` collection schema in `site-next/src/content/config.ts`

- [ ] **Step 1: Create the content directory**

Run:

```powershell
New-Item -ItemType Directory -Force 'site-next\src\content\projects\PaimonAsksEverything' | Out-Null
```

Expected: command exits successfully.

- [ ] **Step 2: Add the Markdown entry**

Create `site-next/src/content/projects/PaimonAsksEverything/paimon-asks-everything.md` with this content:

```markdown
---
title: "Paimon Asks Everything"
description: "A spoiler-aware, evidence-grounded bilingual lore agent for Genshin players and release insight exploration."
date: "2026-06-23"
updatedAt: "2026-06-30T04:00:00.000Z"
tags:
  - "AI Agent"
  - "Retrieval"
  - "Product Design"
  - "Genshin Impact"
draft: false
featured: true
status: "Demo"
period: "2026"
role: "Product design, agent workflow, retrieval, and full-stack implementation"
techStack:
  - "TypeScript"
  - "Next.js"
  - "AI Agent"
  - "Retrieval"
  - "Supabase"
  - "Vercel"
repo: "https://github.com/6chHenry/paimon-asks-everything"
links:
  - {"label":"Live Demo","href":"https://paimon-asks-everything.vercel.app","type":"demo"}
  - {"label":"GitHub","href":"https://github.com/6chHenry/paimon-asks-everything","type":"repo"}
accent: "gold"
summary: "A case study for a spoiler-aware lore agent that combines bilingual retrieval, source governance, structured citations, and release-team insight signals."
---

This project uses a dedicated case-study layout.
```

Expected: frontmatter matches the existing schema and the body is intentionally minimal.

- [ ] **Step 3: Run Astro content validation through build later**

No command in this task. Task 5 runs `npm run build`.

### Task 3: Create The Dedicated Case Study Component

**Files:**
- Create: `site-next/src/components/PaimonCaseStudy.astro`

**Interfaces:**
- Consumes prop `entry: CollectionEntry<'projects'>`
- Consumes prop `previous?: { title: string; href: string; description?: string }`
- Consumes prop `next?: { title: string; href: string; description?: string }`
- Produces a full `BaseLayout` page with `data-pagefind-body`

- [ ] **Step 1: Create the Astro component**

Create `site-next/src/components/PaimonCaseStudy.astro` with:

```astro
---
import type { CollectionEntry } from 'astro:content';
import BaseLayout from '../layouts/BaseLayout.astro';

interface ArticleNavLink {
  title: string;
  href: string;
  description?: string;
}

interface Props {
  entry: CollectionEntry<'projects'>;
  previous?: ArticleNavLink;
  next?: ArticleNavLink;
}

const { entry, previous, next } = Astro.props;
const base = import.meta.env.BASE_URL;
const assetBase = `${base}assets/projects/paimon-asks-everything/`;
const resolveHref = (href: string) => (href.startsWith('/') ? `${base}${href.slice(1)}` : href);

const links = entry.data.links ?? [];
const pillars = [
  {
    label: 'Spoiler-safe Q&A',
    text: 'Answers adapt to player progress and keep high-risk lore behind explicit gates.',
  },
  {
    label: 'Evidence-grounded answers',
    text: 'Responses are constrained by controlled knowledge, whitelisted wiki search, and structured citations.',
  },
  {
    label: 'Snezhnaya relationship map',
    text: 'Characters, factions, and concepts become explorable nodes instead of scattered notes.',
  },
  {
    label: 'Release insights',
    text: 'Anonymous aggregate signals point to confusion, interest, and content opportunities.',
  },
];

const architecture = [
  'Progress and language profile',
  'Controlled bilingual retrieval',
  'Spoiler gate and source policy',
  'Whitelist wiki and web fallback',
  'Citation-checked answer generation',
  'Anonymous insight aggregation',
];

const reliability = [
  'Citation validation before answers are shown as supported.',
  'Source classes separate official, trusted wiki, community, and unknown web material.',
  'Weak evidence downgrades confidence instead of inventing certainty.',
  'Fixed evaluation questions catch retrieval, spoiler, citation, and API regressions.',
];
---

<BaseLayout title={`${entry.data.title} | 6ch. Projects`} description={entry.data.description}>
  <article class="paimon-case" data-pagefind-body>
    <section class="paimon-hero" aria-labelledby="paimon-title">
      <div class="paimon-hero__copy">
        <p class="section-kicker">Case Study / AI Agent</p>
        <h1 id="paimon-title">{entry.data.title}</h1>
        <p class="paimon-hero__subtitle">Paimon Sanqianwen</p>
        <p class="paimon-hero__summary">{entry.data.description}</p>
        <div class="paimon-hero__actions" aria-label="Project links">
          {links.map((link) => (
            <a class:list={['paimon-link', { 'paimon-link--primary': link.type === 'demo' }]} href={resolveHref(link.href)} target={link.href.startsWith('http') ? '_blank' : undefined} rel={link.href.startsWith('http') ? 'noopener' : undefined}>
              {link.label}
            </a>
          ))}
        </div>
        <ul class="paimon-meta" aria-label="Project metadata">
          {(entry.data.techStack ?? []).slice(0, 4).map((tech) => <li>{tech}</li>)}
        </ul>
      </div>
      <figure class="paimon-hero__visual">
        <img src={`${assetBase}banner.png`} alt="Paimon Asks Everything promotional banner showing Paimon, a starry blue background, and four product feature cards." width="2048" height="680" loading="eager" decoding="async" />
      </figure>
    </section>

    <section class="paimon-section paimon-overview">
      <div>
        <p class="paimon-label">Overview</p>
        <h2>A lore assistant shaped as a product system</h2>
      </div>
      <p>
        I built this demo as a bridge between player-facing lore assistance and release-team observation. The core challenge was not simply answering questions. It was answering with progress awareness, source boundaries, spoiler control, bilingual retrieval, and enough traceability that the answer could be audited.
      </p>
    </section>

    <section class="paimon-section">
      <div class="paimon-section__heading">
        <p class="paimon-label">Problem</p>
        <h2>Genshin lore is rich, fragmented, and easy to spoil</h2>
      </div>
      <div class="paimon-problem-grid">
        <p>Story details are spread across quests, artifacts, character profiles, version events, wiki pages, and community summaries.</p>
        <p>Players arrive with different progress levels, so a useful assistant must know when an answer becomes a spoiler.</p>
        <p>Release teams need to understand confusion and interest without collecting unnecessary personal data.</p>
      </div>
    </section>

    <section class="paimon-section">
      <div class="paimon-section__heading">
        <p class="paimon-label">Product Strategy</p>
        <h2>Four pillars made the demo concrete</h2>
      </div>
      <div class="paimon-pillar-grid">
        {pillars.map((pillar, index) => (
          <article class="paimon-pillar">
            <span>{String(index + 1).padStart(2, '0')}</span>
            <h3>{pillar.label}</h3>
            <p>{pillar.text}</p>
          </article>
        ))}
      </div>
    </section>

    <section class="paimon-section paimon-map-section">
      <div class="paimon-section__heading">
        <p class="paimon-label">System Design</p>
        <h2>The product frame before the agent workflow</h2>
      </div>
      <figure class="paimon-map">
        <img src={`${assetBase}design-map.png`} alt="Design mind map for Paimon Asks Everything, connecting core positioning, reasons for the product, design principles, release value, Snezhnaya adaptation, and core features." width="1794" height="1356" loading="lazy" decoding="async" />
        <figcaption>
          The map kept the demo grounded: Paimon as guide, not replacement; accuracy over fabrication; spoiler control before fluency; release insight as a byproduct of useful player support.
        </figcaption>
      </figure>
    </section>

    <section class="paimon-section paimon-architecture">
      <div class="paimon-section__heading">
        <p class="paimon-label">Technical Architecture</p>
        <h2>Evidence first, generation second</h2>
      </div>
      <ol class="paimon-flow" aria-label="Agent workflow">
        {architecture.map((item) => <li>{item}</li>)}
      </ol>
      <p>
        The agent starts from player preferences, expands aliases across Chinese and English, filters by spoiler level, searches controlled and whitelisted sources, grades citations, and only then asks generation to turn evidence into a structured answer.
      </p>
    </section>

    <section class="paimon-section paimon-reliability">
      <div class="paimon-section__heading">
        <p class="paimon-label">Reliability</p>
        <h2>Keeping the demo honest</h2>
      </div>
      <ul>
        {reliability.map((item) => <li>{item}</li>)}
      </ul>
    </section>

    <section class="paimon-section paimon-retrospective">
      <div>
        <p class="paimon-label">Retrospective</p>
        <h2>What remains intentionally unfinished</h2>
      </div>
      <div>
        <p>
          The Vercel deployment can still cold-start, the release insight page has room for real external trend signals, and the public demo uses lightweight rate limiting. Next I would push the UI closer to the game language, deepen the interaction model, and improve search reliability with a stronger local corpus.
        </p>
      </div>
    </section>

    {(previous || next) && (
      <nav class="paimon-nav" aria-label="Project navigation">
        {previous ? <a href={previous.href}><span>Previous</span><strong>{previous.title}</strong></a> : <span></span>}
        {next ? <a href={next.href}><span>Next</span><strong>{next.title}</strong></a> : <span></span>}
      </nav>
    )}
  </article>
</BaseLayout>
```

Expected: the component compiles after Task 4 connects it.

- [ ] **Step 2: Add scoped CSS to the component**

Append this `<style>` block to the component:

```astro
<style>
  .paimon-case {
    --paimon-navy: #07142f;
    --paimon-navy-soft: #12264a;
    --paimon-gold: var(--ch-accent-warm);
    width: min(calc(100% - 2.5rem), 78rem);
    margin: 0 auto;
    padding: 2.5rem 0 4.5rem;
  }

  .paimon-hero {
    display: grid;
    grid-template-columns: minmax(0, 0.72fr) minmax(20rem, 1fr);
    gap: clamp(1.25rem, 3vw, 2.5rem);
    align-items: center;
    min-height: min(720px, calc(100dvh - var(--ch-header-height) - 1rem));
    padding: clamp(1.2rem, 3vw, 2rem);
    border: 1px solid color-mix(in srgb, var(--paimon-gold) 30%, var(--ch-border-soft));
    border-radius: 14px;
    background:
      linear-gradient(135deg, color-mix(in srgb, var(--ch-surface-solid) 88%, transparent), color-mix(in srgb, var(--paimon-navy) 20%, var(--ch-glass-bg))),
      radial-gradient(circle at 18% 14%, color-mix(in srgb, var(--paimon-gold) 18%, transparent), transparent 36%);
    overflow: hidden;
  }

  .paimon-hero__copy {
    position: relative;
    z-index: 1;
  }

  .paimon-hero h1,
  .paimon-section h2 {
    font-family: var(--ch-display-font);
    color: var(--ch-fg);
  }

  .paimon-hero h1 {
    margin: 0;
    font-size: clamp(3rem, 8vw, 6.4rem);
    line-height: 0.92;
  }

  .paimon-hero__subtitle {
    margin: 0.8rem 0 0;
    color: var(--paimon-gold);
    font-family: var(--ch-mono-font);
    font-size: 0.82rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
  }

  .paimon-hero__summary,
  .paimon-section p,
  .paimon-pillar p,
  .paimon-map figcaption,
  .paimon-reliability li {
    color: var(--ch-fg-muted);
    line-height: 1.75;
  }

  .paimon-hero__summary {
    max-width: 34rem;
    margin: 1.1rem 0 0;
    font-size: 1.05rem;
  }

  .paimon-hero__actions,
  .paimon-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
  }

  .paimon-hero__actions {
    margin-top: 1.4rem;
  }

  .paimon-link {
    display: inline-flex;
    min-height: 44px;
    align-items: center;
    justify-content: center;
    padding: 0.62rem 1rem;
    border: 1px solid color-mix(in srgb, var(--paimon-gold) 36%, var(--ch-border-soft));
    border-radius: 999px;
    background: color-mix(in srgb, var(--ch-surface-solid) 72%, transparent);
    color: var(--ch-accent);
    font-family: var(--ch-academic-font);
    font-weight: 650;
    text-decoration: none;
  }

  .paimon-link--primary {
    background: color-mix(in srgb, var(--paimon-gold) 20%, var(--ch-surface-solid));
    color: color-mix(in srgb, var(--ch-accent-ink) 78%, var(--paimon-gold));
  }

  .paimon-meta {
    margin: 1rem 0 0;
    padding: 0;
    list-style: none;
  }

  .paimon-meta li,
  .paimon-label,
  .paimon-pillar span {
    font-family: var(--ch-mono-font);
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }

  .paimon-meta li {
    padding: 0.24rem 0.52rem;
    border: 1px solid var(--ch-border-soft);
    border-radius: 999px;
    color: var(--ch-accent-secondary);
  }

  .paimon-hero__visual,
  .paimon-map {
    margin: 0;
    overflow: hidden;
    border: 1px solid color-mix(in srgb, var(--paimon-gold) 30%, var(--ch-border-soft));
    border-radius: 12px;
    background: var(--ch-surface-solid);
  }

  .paimon-hero__visual {
    aspect-ratio: 2048 / 680;
    box-shadow: 0 28px 90px color-mix(in srgb, var(--paimon-navy) 22%, transparent);
  }

  .paimon-hero__visual img,
  .paimon-map img {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .paimon-section {
    display: grid;
    grid-template-columns: minmax(12rem, 0.38fr) minmax(0, 1fr);
    gap: clamp(1rem, 3vw, 2.5rem);
    padding: clamp(2.4rem, 5vw, 4.5rem) 0 0;
  }

  .paimon-section__heading h2,
  .paimon-overview h2,
  .paimon-retrospective h2 {
    margin: 0;
    font-size: clamp(1.8rem, 4vw, 3.4rem);
    line-height: 1;
  }

  .paimon-label {
    margin: 0 0 0.6rem;
    color: var(--paimon-gold);
  }

  .paimon-overview > p,
  .paimon-retrospective p {
    margin: 0;
    font-size: 1.08rem;
  }

  .paimon-problem-grid,
  .paimon-pillar-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.8rem;
  }

  .paimon-problem-grid p,
  .paimon-pillar,
  .paimon-reliability li {
    margin: 0;
    padding: 1rem;
    border: 1px solid var(--ch-border-soft);
    border-radius: 10px;
    background: color-mix(in srgb, var(--ch-glass-bg) 86%, transparent);
  }

  .paimon-pillar-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  .paimon-pillar h3 {
    margin: 0.55rem 0 0.4rem;
    color: var(--ch-fg);
    font-family: var(--ch-display-font);
    font-size: 1.18rem;
    line-height: 1.2;
  }

  .paimon-pillar span {
    color: var(--paimon-gold);
  }

  .paimon-map-section {
    grid-template-columns: 1fr;
  }

  .paimon-map {
    aspect-ratio: 1794 / 1356;
  }

  .paimon-map figcaption {
    padding: 0.9rem 1rem;
    border-top: 1px solid var(--ch-border-soft);
    background: color-mix(in srgb, var(--ch-surface-solid) 88%, var(--ch-accent-dim));
    font-size: 0.92rem;
  }

  .paimon-architecture,
  .paimon-reliability {
    align-items: start;
  }

  .paimon-flow {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 0.5rem;
    margin: 0;
    padding: 0;
    list-style: none;
    counter-reset: flow;
  }

  .paimon-flow li {
    counter-increment: flow;
    min-height: 8rem;
    padding: 0.85rem;
    border: 1px solid color-mix(in srgb, var(--paimon-gold) 24%, var(--ch-border-soft));
    border-radius: 10px;
    background: linear-gradient(180deg, color-mix(in srgb, var(--paimon-navy-soft) 12%, var(--ch-glass-bg)), var(--ch-glass-bg));
    color: var(--ch-fg);
    font-family: var(--ch-academic-font);
    line-height: 1.35;
  }

  .paimon-flow li::before {
    display: block;
    margin-bottom: 0.55rem;
    color: var(--paimon-gold);
    font-family: var(--ch-mono-font);
    font-size: 0.68rem;
    content: counter(flow, decimal-leading-zero);
  }

  .paimon-architecture > p {
    grid-column: 2;
    margin: 1rem 0 0;
  }

  .paimon-reliability ul {
    display: grid;
    gap: 0.65rem;
    margin: 0;
    padding: 0;
    list-style: none;
  }

  .paimon-nav {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
    margin-top: 3rem;
    padding-top: 1.25rem;
    border-top: 1px solid var(--ch-border-soft);
  }

  .paimon-nav a {
    min-height: 5.5rem;
    padding: 1rem;
    border: 1px solid var(--ch-border-soft);
    border-radius: 10px;
    background: var(--ch-glass-bg);
    text-decoration: none;
  }

  .paimon-nav span {
    display: block;
    color: var(--paimon-gold);
    font-family: var(--ch-mono-font);
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }

  .paimon-nav strong {
    display: block;
    margin-top: 0.35rem;
    color: var(--ch-fg);
    font-family: var(--ch-display-font);
    font-size: 1.05rem;
  }

  @media (max-width: 980px) {
    .paimon-hero,
    .paimon-section,
    .paimon-architecture > p {
      grid-template-columns: 1fr;
    }

    .paimon-problem-grid,
    .paimon-pillar-grid,
    .paimon-flow {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (max-width: 640px) {
    .paimon-case {
      width: min(calc(100% - 1rem), 78rem);
      padding-top: 1rem;
    }

    .paimon-hero {
      min-height: auto;
      padding: 1rem;
    }

    .paimon-hero h1 {
      font-size: clamp(2.6rem, 18vw, 4rem);
    }

    .paimon-hero__visual {
      aspect-ratio: 1.45;
    }

    .paimon-problem-grid,
    .paimon-pillar-grid,
    .paimon-flow,
    .paimon-nav {
      grid-template-columns: 1fr;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .paimon-link,
    .paimon-nav a {
      transition: none;
    }
  }
</style>
```

Expected: no global selectors beyond media queries and no changes to shared CSS files.

### Task 4: Route The Paimon Entry To The Dedicated Component

**Files:**
- Modify: `site-next/src/pages/projects/[...slug].astro`

**Interfaces:**
- Consumes: `PaimonCaseStudy` component from `../../components/PaimonCaseStudy.astro`
- Produces: conditional rendering for `entry.id === 'PaimonAsksEverything/paimon-asks-everything.md'`
- Preserves: existing `ContentLayout` rendering for all other projects

- [ ] **Step 1: Import the component**

Add:

```astro
import PaimonCaseStudy from '../../components/PaimonCaseStudy.astro';
```

near the existing imports.

- [ ] **Step 2: Add the entry guard**

After `const { entry } = Astro.props;`, add:

```ts
const isPaimonCaseStudy = entry.id === 'PaimonAsksEverything/paimon-asks-everything.md';
```

- [ ] **Step 3: Conditionally render the page**

Replace the final `ContentLayout` block with this conditional render:

```astro
{
  isPaimonCaseStudy ? (
    <PaimonCaseStudy entry={entry} previous={previousEntry && {
      title: previousEntry.data.title,
      description: previousEntry.data.description,
      href: buildContentUrl('projects', previousEntry.id),
    }} next={nextEntry && {
      title: nextEntry.data.title,
      description: nextEntry.data.description,
      href: buildContentUrl('projects', nextEntry.id),
    }} />
  ) : (
    <ContentLayout
      title={entry.data.title}
      description={entry.data.description}
      date={entry.data.date}
      tags={entry.data.tags}
      headings={filteredHeadings.map((h) => ({ depth: h.depth, slug: h.slug, text: h.text }))}
      characterTreeSvg={characterTreeSvg}
      previous={previousEntry && {
        title: previousEntry.data.title,
        description: previousEntry.data.description,
        href: buildContentUrl('projects', previousEntry.id),
      }}
      next={nextEntry && {
        title: nextEntry.data.title,
        description: nextEntry.data.description,
        href: buildContentUrl('projects', nextEntry.id),
      }}
    >
      <Content />
    </ContentLayout>
  )
}
```

Expected: the Paimon detail page uses the dedicated component; older projects remain unchanged.

### Task 5: Build And Visual Sanity Check

**Files:**
- No source changes expected unless verification reveals an issue.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: successful Astro production build.

- [ ] **Step 1: Run the production build**

Run:

```powershell
npm run build
```

from `site-next/`.

Expected: build succeeds and Pagefind completes.

- [ ] **Step 2: If the build fails, fix only the failing files**

Use the error output to patch the smallest relevant file. Do not touch unrelated dirty files.

Expected: rerunning `npm run build` succeeds.

- [ ] **Step 3: Inspect the changed file set**

Run:

```powershell
git status --short
```

Expected: changes from this task are limited to the plan, Paimon assets, Paimon content entry, Paimon component, and project detail route. Existing unrelated dirty files may still appear and must not be staged with this work.

- [ ] **Step 4: Commit the implementation files**

Stage only relevant files:

```powershell
git add -- docs/superpowers/plans/2026-06-30-paimon-case-study-implementation.md site-next/public/assets/projects/paimon-asks-everything site-next/src/content/projects/PaimonAsksEverything site-next/src/components/PaimonCaseStudy.astro 'site-next/src/pages/projects/[...slug].astro'
git commit -m "Add Paimon case study project page"
```

Expected: commit succeeds without staging unrelated changes.
