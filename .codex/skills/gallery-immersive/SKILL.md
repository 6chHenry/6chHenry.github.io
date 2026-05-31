---
name: gallery-immersive
description: "Create immersive travel gallery pages for the 6ch. Astro site. Use when the user wants to add a new travel photography collection with multiple cities — phrases like 'add a gallery trip', 'create immersive gallery', 'new travel gallery', 'add city photo collection', 'create a photo journey page', 'like the Kansai page', '类似关西旅路'. Handles creating city content files, the immersive page with hero carousel + timeline, wiring up gallery card redirects, and updating sort order."
---

# Gallery Immersive Creator

Creates a complete immersive travel gallery page: hero carousel, sticky timeline, per-city photo sections with lightbox. Follow the Kansai Journey / Greater Bay Area pattern exactly.

## Workflow

### 1. Gather Input

Ask the user for these. Use AskUserQuestion for anything ambiguous, but don't delay — if they've already described cities and trip, deduce what you can.

Required data:
- **Trip name CN**: Chinese title, e.g. `関西旅路`
- **Trip name EN**: English subtitle, e.g. `Kansai Journey`
- **Trip slug**: URL-safe, e.g. `japan`
- **Year**: e.g. `2026`
- **Lead description**: 1-2 sentences for page header
- **Cities list**: Each with: slug, Chinese name, English name, hex color, 1-sentence description
- **Aggregate cover city**: which city's cover to use for the main gallery card

Pick distinctive, non-clashing colors for cities.

### 2. Discover Images

For each city, list the image files in its `.assets/` directory. Sort alphabetically — first becomes `cover`, rest become `images`.

If no images exist, generate placeholder paths using `https://picsum.photos/seed/{city-slug}-{n}/800/600`. Remind user to replace.

Generate 1-sentence placeholder `imageDescs` from filenames (convert snake_case/kebab-case to readable Chinese).

### 3. Create Content Files

#### 3a. City entries

Create `docs/gallery/photography/{city-slug}.md`. Read `assets/city-template.md` for the exact format.

#### 3b. Aggregate card

Create `docs/gallery/photography/{trip-slug}.md`. Read `assets/trip-template.md` for the exact format. Pick 3-4 images from different cities for the card.

#### 3c. Immersive page

Create `src/pages/gallery/{trip-slug}.astro`. Read `assets/page-template.astro` for the exact format. Substitute ALL placeholders:
- `{TRIP_SLUG}` `{TRIP_NAME_CN}` `{TRIP_NAME_EN}` `{YEAR}` `{LEAD_DESCRIPTION}`
- `{CITY_ORDER_ARRAY}` — e.g. `['osaka', 'nara', 'uji'] as const`
- `{CITY_META_OBJECT}` — object with kanji, en, color, desc per city

### 4. Wire Up Existing Files

#### 4a. Sort order — `src/data/gallery-order.ts`

Append city IDs to the `GALLERY_CITY_ORDER` array with a comment:

```ts
  // Trip Name
  'photography/{city1}.md',
  'photography/{city2}.md',
```

#### 4b. Gallery filter — `src/pages/gallery/index.astro`

Add city slugs to the `hiddenCities` Set.

#### 4c. Card redirect — `src/scripts/gallery.ts`

Add a redirect block after the last existing one, before `event.preventDefault()`:

```ts
const is{TripSlugPascal}Card = href.includes('/gallery/photography/{trip-slug}');
if (is{TripSlugPascal}Card) {
  window.location.href = href.replace('/gallery/photography/{trip-slug}', '/gallery/{trip-slug}');
  event.preventDefault();
  return;
}
```

### 5. Build & Verify

```bash
node site-next/scripts/migrate-content.mjs
```

Then build: `cd site-next && npx astro build`. Must succeed with zero errors. Remind user to add real images to `.assets/` directories and rebuild.
