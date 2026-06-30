# Paimon Asks Everything Case Study Page Design

## Goal

Add a featured personal project case study for `6chHenry/paimon-asks-everything` to the Astro personal site.

The page should feel like a polished product case study rather than a generic project note. It must still belong to the existing `6ch.` visual system: forest editorial tokens, misty glass surfaces, serif display typography, restrained gold accents, and content-first pacing.

The project should communicate:

- What problem the demo solves for Genshin players and release teams.
- How the product is positioned: a spoiler-aware lore agent with evidence-grounded answers and release insight support.
- What the author designed and implemented.
- Which systems make the demo credible: bilingual retrieval, spoiler gating, source governance, citations, event storage, and deterministic evaluation.

## Confirmed Direction

Use a **personal case study with a premium product-style hero**.

The page should not become a pure marketing landing page. The product presentation should create immediate appeal, while the body should explain the design reasoning, technical architecture, and tradeoffs.

## Source Inputs

Project repository:

```text
https://github.com/6chHenry/paimon-asks-everything
```

Live demo:

```text
https://paimon-asks-everything.vercel.app
```

Local promotional assets:

```text
F:\PAIMON\figures\banner.png
F:\PAIMON\figures\*.png, using the non-banner mind map image as design-map.png
```

The banner should be used as the primary hero visual. The mind map should be used in the case study body as design logic or system overview material.

## Visual System

The page extends the site's existing forest editorial style instead of replacing it with a game UI skin.

Use:

- Deep navy and star-map imagery from the banner as a project-specific layer.
- Existing site gold accent through `--ch-accent-warm`.
- Existing forest greens for continuity with the rest of the site.
- Misty glass panels and subtle borders already used by the site.
- Serif display type for the case title and section headings.
- Mono or sans labels for metadata, tags, and small system annotations.

Avoid:

- One-note blue/purple fantasy palette.
- Oversized marketing hero that hides the rest of the case study.
- Decorative orbs, generic gradients, or random glowing blobs.
- Game-like UI chrome that clashes with the personal site.
- Emoji as structural icons.

## Page Structure

### Hero

Create a custom case-study hero for this project.

Content:

- Kicker: `Case Study / AI Agent`
- Title: `Paimon Asks Everything`
- Subtitle: `Paimon Sanqianwen`
- Summary: a spoiler-aware, evidence-grounded bilingual lore agent for Genshin players and release insight exploration.
- Primary links:
  - Live Demo
  - GitHub
- Metadata chips:
  - `TypeScript`
  - `Next.js`
  - `AI Agent`
  - `Evidence-grounded QA`

Visual:

- Use `banner.png` as a large first-viewport visual.
- The hero should expose enough of the next section on desktop and mobile, so it does not become a full landing page wall.
- The image needs explicit dimensions or an aspect-ratio wrapper to avoid layout shift.
- The image should have descriptive alt text.

### Case Overview

Introduce the project in first-person case-study language:

- The demo sits between player-facing lore assistance and release-team insight exploration.
- The product problem is not only answering questions, but answering with progress awareness, source boundaries, spoiler control, and bilingual retrieval.
- The author role includes product framing, agent workflow design, retrieval and generation implementation, interaction design, and evaluation.

### Problem Definition

Explain why this product exists:

- Genshin lore is distributed across versions, quests, artifacts, wiki pages, and community summaries.
- Spoilers are hard to avoid when players have different progress levels.
- Wiki and community information can be mixed, duplicated, incomplete, or speculative.
- Release teams need to detect where players are confused without storing unnecessary personal data.

### Product Strategy

Describe the product positioning using four pillars:

- Spoiler-safe Q&A.
- Evidence-grounded answers.
- Snezhnaya relationship map.
- Release insights.

Each pillar should be short, concrete, and tied to user value.

### System Design

Use the local design mind map image as an overview visual, copied into the site as `design-map.png`, with a concise explanation below it.

Key points:

- Core positioning connects "Paimon notebook AI-ification", player guide system, and release observation.
- Design principles include accuracy over fabrication, spoiler control, bilingual consistency, and not replacing the game experience.
- Release value comes from finding interest points, understanding breakpoints, guiding FAQ, and reducing comprehension friction.

### Technical Architecture

Present the architecture as readable case-study content, not as a wall of implementation notes.

Include:

- Controlled bilingual lexical retrieval.
- Spoiler level gating.
- External whitelist wiki search.
- General web search fallback.
- Source grading: official, trusted wiki, community, unknown web.
- Evidence-constrained answer generation with structured citations.
- Anonymous event capture and aggregate release signals.
- Deterministic evaluation set for regression checks.

Use small system cards or a flow section if implementing custom markup.

### Evaluation And Reliability

Explain how the demo is kept honest:

- Citation validation.
- Refusal or downgrade behavior when sources are weak.
- Fixed question set evaluation.
- Source governance that avoids treating community wiki content as official.
- Privacy default that avoids storing raw user questions unless authorized.

### Limitations And Next Steps

List limitations honestly:

- Vercel cold start can make the first response fail.
- White-listed wiki search remains a community-indexed layer.
- External trend layer in release insights is not fully connected.
- Public demo rate limiting is lightweight.

Next steps:

- Improve UI to feel closer to the game while preserving site consistency.
- Add richer interaction.
- Improve search reliability.
- Explore local corpus or fine-tuned Paimon-style language.

## Project Listing Behavior

The project should appear in the Projects index as a featured work.

Frontmatter should include:

- `featured: true`
- `status: "Demo"`
- `period: "2026"`
- `role: "Product design, agent workflow, retrieval, and full-stack implementation"`
- `techStack`: TypeScript, Next.js, AI Agent, Retrieval, Supabase, Vercel
- Links to GitHub and Live Demo.
- `accent: "gold"` or another value that maps to a warm premium treatment.

The featured card should feel consistent with existing cards. If the listing card is enhanced for image support, it must degrade gracefully for older projects without images.

## Implementation Boundaries

Preferred implementation:

- Add the source project content under `site-next/src/content/projects/PaimonAsksEverything/`.
- Copy the local images to a stable public project asset folder, for example:

```text
site-next/public/assets/projects/paimon-asks-everything/banner.png
site-next/public/assets/projects/paimon-asks-everything/design-map.png
```

- Add page-specific styles in an Astro component or a small imported stylesheet.
- Keep generated build output untouched.
- Do not modify migrated essay or notes content unrelated to this project.

If the generic Markdown renderer is too limiting, add a conditional project detail template keyed by `entry.id` or frontmatter. Keep the fallback renderer unchanged for existing projects.

## Responsive Requirements

- No horizontal scroll at 375px width.
- Hero image remains legible and does not overlap title or CTA text.
- Link buttons are at least 44px tall.
- Long labels wrap instead of clipping.
- Hero and case sections keep stable aspect ratios to avoid layout shift.
- Mobile layout should prioritize title, summary, links, then image.
- Desktop layout can use a two-column hero, but the next section should remain hinted below the fold.

## Accessibility Requirements

- Hero and mind-map images need descriptive alt text.
- Links must have clear visible labels.
- Focus states must remain visible.
- Text contrast must remain readable in both light and dark themes.
- Motion should be minimal and respect `prefers-reduced-motion`.
- Do not rely on color alone to explain architecture or status.

## Verification

Run from `site-next/`:

```powershell
npm run build
```

Manual checks:

- Projects index shows the new featured project.
- Project detail page renders the hero image and mind map.
- GitHub and Live Demo links work.
- Page renders correctly in light and dark themes.
- 375px, tablet, and desktop widths have no overlaps or horizontal scroll.
- Pagefind/search build does not fail.
- Existing project pages still render with the generic layout.

## Out Of Scope

- Rebuilding the Paimon app itself.
- Fetching live GitHub metrics at build time.
- Creating new AI-generated Paimon artwork.
- Adding a full interactive graph demo inside the personal site.
- Reworking the entire projects index design.
