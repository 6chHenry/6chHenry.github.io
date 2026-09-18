interface GalleryImageAsset {
  src: string;
  webpSrcset: string;
  width?: number;
  height?: number;
  original: string;
}

interface GalleryChapter {
  slug: string;
  kanji: string;
  en: string;
  color?: string;
  start: number;
  count: number;
}

interface GalleryMeta {
  images: GalleryImageAsset[];
  imageDescs: string[];
  chapters?: GalleryChapter[];
  title: string;
  description: string;
  category: string;
  date: string;
  detailUrl: string;
}

let currentMeta: GalleryMeta | null = null;
let currentIndex = 0;
let touchStartX = 0;
let touchEndX = 0;

const categoryLabels: Record<string, string> = {
  illustration: '插画',
  photography: '摄影',
  design: '设计',
  'ui-ux': 'UI/UX',
  other: '其他',
};

export function initGalleryFilters() {
  const filterBar = document.querySelector<HTMLElement>('.gallery-filters');
  const grid = document.querySelector<HTMLElement>('.gallery-grid');
  if (!filterBar || !grid) return;

  const cards = grid.querySelectorAll<HTMLElement>('.gallery-card');

  filterBar.addEventListener('click', (event) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>('.gallery-filter');
    if (!button) return;

    const category = button.dataset.category;

    for (const btn of filterBar.querySelectorAll('.gallery-filter')) {
      btn.setAttribute('aria-pressed', btn === button ? 'true' : 'false');
    }

    cards.forEach((card) => {
      if (!category || card.dataset.category === category) {
        card.classList.remove('gallery-card--hidden');
      } else {
        card.classList.add('gallery-card--hidden');
      }
    });
  });
}

export function initGalleryTimelineHighlights() {
  const timelineNodes = document.querySelectorAll<HTMLElement>('[data-trip-target]');
  const galleryCards = document.querySelectorAll<HTMLElement>('[data-gallery-slug]');
  if (timelineNodes.length === 0 || galleryCards.length === 0) return;

  const clearHighlights = () => {
    timelineNodes.forEach((node) => node.classList.remove('is-linked'));
    galleryCards.forEach((card) => card.classList.remove('is-linked'));
  };

  const setHighlight = (slug: string | undefined) => {
    clearHighlights();
    if (!slug) return;

    document.querySelector<HTMLElement>(`[data-trip-target="${slug}"]`)?.classList.add('is-linked');
    document.querySelector<HTMLElement>(`[data-gallery-slug="${slug}"]`)?.classList.add('is-linked');
  };

  timelineNodes.forEach((node) => {
    const slug = node.dataset.tripTarget;
    node.addEventListener('pointerenter', () => setHighlight(slug));
    node.addEventListener('focus', () => setHighlight(slug));
    node.addEventListener('pointerleave', clearHighlights);
    node.addEventListener('blur', clearHighlights);
  });

  galleryCards.forEach((card) => {
    const slug = card.dataset.gallerySlug;
    card.addEventListener('pointerenter', () => setHighlight(slug));
    card.addEventListener('focus', () => setHighlight(slug));
    card.addEventListener('pointerleave', clearHighlights);
    card.addEventListener('blur', clearHighlights);
  });
}

export function initGalleryLightbox() {
  const lightbox = document.getElementById('gallery-lightbox');
  if (!lightbox) return;
  if (lightbox.parentElement !== document.body) {
    document.body.appendChild(lightbox);
  }

  const img = lightbox.querySelector<HTMLImageElement>('.glbx__image');
  const source = lightbox.querySelector<HTMLSourceElement>('.glbx__source');
  const counter = lightbox.querySelector<HTMLElement>('.glbx__counter');
  const catEl = lightbox.querySelector<HTMLElement>('.glbx__category');
  const titleEl = lightbox.querySelector<HTMLElement>('.glbx__title');
  const descEl = lightbox.querySelector<HTMLElement>('.glbx__desc');
  const imageDescEl = lightbox.querySelector<HTMLElement>('.glbx__image-desc');
  const dateEl = lightbox.querySelector<HTMLElement>('.glbx__date');
  const detailBtn = lightbox.querySelector<HTMLAnchorElement>('.glbx__detail-btn');
  const prevBtn = lightbox.querySelector<HTMLButtonElement>('.glbx__nav--prev');
  const nextBtn = lightbox.querySelector<HTMLButtonElement>('.glbx__nav--next');
  const stopEl = lightbox.querySelector<HTMLElement>('.glbx__stop');
  const stopKanji = lightbox.querySelector<HTMLElement>('.glbx__stop-kanji');
  const stopEn = lightbox.querySelector<HTMLElement>('.glbx__stop-en');
  const filmEl = lightbox.querySelector<HTMLElement>('.glbx__film');
  const filmTrack = lightbox.querySelector<HTMLElement>('.glbx__film-track');
  const imageArea = lightbox.querySelector<HTMLElement>('.glbx__image-area');
  const panel = lightbox.querySelector<HTMLElement>('.glbx__panel');
  const inspectBtn = lightbox.querySelector<HTMLButtonElement>('.glbx__inspect');

  let loaded = false;
  let inspecting = false;
  let inspectScale = 1;
  let inspectX = 0;
  let inspectY = 0;
  let dragging = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let dragOriginX = 0;
  let dragOriginY = 0;

  const applyInspectTransform = () => {
    if (!img) return;
    img.style.transform = inspecting
      ? `translate(${inspectX}px, ${inspectY}px) scale(${inspectScale})`
      : '';
  };

  const resetInspectTransform = () => {
    inspectScale = inspecting ? 1.28 : 1;
    inspectX = 0;
    inspectY = 0;
    applyInspectTransform();
  };

  const setInspecting = (next: boolean) => {
    inspecting = next;
    lightbox.classList.toggle('is-inspect', inspecting);
    inspectBtn?.setAttribute('aria-pressed', inspecting ? 'true' : 'false');
    inspectBtn?.setAttribute('aria-label', inspecting ? '退出放大' : '放大查看');
    if (!inspecting) {
      inspectScale = 1;
      inspectX = 0;
      inspectY = 0;
    } else if (inspectScale < 1.15) {
      inspectScale = 1.28;
    }
    applyInspectTransform();
  };

  const chapterAt = (index: number) => {
    const chapters = currentMeta?.chapters;
    if (!chapters?.length) return null;
    return chapters.find((chapter) => index >= chapter.start && index < chapter.start + chapter.count) ?? null;
  };

  const updateChrome = (index: number) => {
    if (!currentMeta) return;
    const chapter = chapterAt(index);
    const multiCity = (currentMeta.chapters?.length ?? 0) > 1;

    if (counter) {
      if (multiCity && chapter) {
        counter.textContent = `${chapter.kanji}  ${index - chapter.start + 1} / ${chapter.count}`;
      } else {
        counter.textContent = `${index + 1} / ${currentMeta.images.length}`;
      }
    }

    if (stopEl && stopKanji && stopEn) {
      if (multiCity && chapter) {
        stopEl.hidden = false;
        stopKanji.textContent = chapter.kanji;
        stopEn.textContent = chapter.en;
      } else {
        stopEl.hidden = true;
      }
    }

    if (panel) {
      const accent = chapter?.color;
      if (accent) {
        panel.style.setProperty('--gb-accent', accent);
        panel.style.setProperty('--gb-accent-dim', `color-mix(in srgb, ${accent} 16%, transparent)`);
      } else {
        panel.style.removeProperty('--gb-accent');
        panel.style.removeProperty('--gb-accent-dim');
      }
    }

    resetInspectTransform();

    if (imageDescEl) imageDescEl.textContent = currentMeta.imageDescs?.[index] ?? '';
    if (prevBtn) prevBtn.style.visibility = index > 0 ? '' : 'hidden';
    if (nextBtn) nextBtn.style.visibility = index < currentMeta.images.length - 1 ? '' : 'hidden';
    updateFilm(index);
  };

  const renderFilm = () => {
    if (!filmEl || !filmTrack || !currentMeta) return;
    const chapters = currentMeta.chapters ?? [];
    const showGaps = chapters.length > 1;
    const starts = new Set(chapters.map((chapter) => chapter.start));
    filmTrack.replaceChildren();

    currentMeta.images.forEach((asset, index) => {
      const item = document.createElement('button');
      item.type = 'button';
      item.className = 'glbx__film-item';
      if (showGaps && starts.has(index) && index > 0) item.classList.add('is-chapter-start');
      item.dataset.index = String(index);
      item.setAttribute('aria-label', `第 ${index + 1} 张`);
      const thumb = document.createElement('img');
      thumb.alt = '';
      thumb.loading = 'lazy';
      if (asset.webpSrcset) {
        thumb.srcset = asset.webpSrcset;
        thumb.sizes = '48px';
      }
      thumb.src = asset.src;
      item.appendChild(thumb);
      filmTrack.appendChild(item);
    });

    const showFilm = currentMeta.images.length > 1;
    filmEl.hidden = !showFilm;
    imageArea?.classList.toggle('has-film', showFilm);
  };

  const updateFilm = (index: number) => {
    if (!filmTrack) return;
    const items = filmTrack.querySelectorAll<HTMLElement>('.glbx__film-item');
    items.forEach((item, itemIndex) => {
      const active = itemIndex === index;
      item.classList.toggle('is-active', active);
      if (active) {
        item.scrollIntoView({ inline: 'center', block: 'nearest', behavior: 'smooth' });
      }
    });
  };

  const preloadAdjacent = (index: number) => {
    if (!currentMeta) return;
    const asset = currentMeta.images[index + 1] ?? currentMeta.images[index - 1];
    if (!asset) return;
    const preload = new Image();
    preload.srcset = asset.webpSrcset;
    preload.sizes = '100vw';
    preload.src = asset.src;
  };

  const setImageAsset = (asset: GalleryImageAsset) => {
    if (!img) return;
    if (source) {
      source.srcset = asset.webpSrcset;
      source.sizes = '100vw';
    }
    img.src = asset.src;
    if (asset.width) img.width = asset.width;
    else img.removeAttribute('width');
    if (asset.height) img.height = asset.height;
    else img.removeAttribute('height');
  };

  const showImage = (index: number) => {
    if (!img || !counter || !currentMeta) return;

    const isNav = loaded && index !== currentIndex;

    if (isNav) {
      img.classList.add('is-switching');
      setTimeout(() => {
        currentIndex = index;
        setImageAsset(currentMeta.images[currentIndex]);
        const onDone = () => {
          img.classList.remove('is-switching');
          img.onload = null;
          img.onerror = null;
        };
        img.onload = onDone;
        img.onerror = onDone;
        updateChrome(index);
        preloadAdjacent(index);
      }, 150);
    } else {
      currentIndex = index;
      setImageAsset(currentMeta.images[currentIndex]);
      img.classList.remove('is-switching');
      loaded = true;
      updateChrome(index);
      preloadAdjacent(index);
    }
  };

  const open = (meta: GalleryMeta, startIndex: number) => {
    currentMeta = meta;
    currentIndex = startIndex;

    if (catEl) catEl.textContent = categoryLabels[meta.category] ?? meta.category;
    if (titleEl) titleEl.textContent = meta.title;
    if (descEl) descEl.textContent = meta.description || '';
    if (dateEl) dateEl.textContent = meta.date || '';
    if (detailBtn) detailBtn.href = meta.detailUrl;

    renderFilm();
    showImage(startIndex);
    lightbox.hidden = false;
    document.body.classList.add('glbx-open');
    document.body.style.overflow = 'hidden';
  };

  const close = () => {
    lightbox.hidden = true;
    document.body.classList.remove('glbx-open');
    document.body.style.overflow = '';
    currentMeta = null;
    currentIndex = 0;
    loaded = false;
    filmTrack?.replaceChildren();
    imageArea?.classList.remove('has-film');
    if (filmEl) filmEl.hidden = true;
    panel?.style.removeProperty('--gb-accent');
    panel?.style.removeProperty('--gb-accent-dim');
    setInspecting(false);
  };

  const prev = () => {
    if (currentIndex > 0) showImage(currentIndex - 1);
  };

  const next = () => {
    if (currentIndex < (currentMeta?.images.length ?? 1) - 1) showImage(currentIndex + 1);
  };

  const DETAIL_ENTER = '.gallery-card__enter, .trip-timeline__enter';
  const PREVIEW_HOST = '.gallery-card, .trip-timeline__node';
  let previewTimer: number | null = null;

  const readMeta = (host: HTMLElement): GalleryMeta | null => {
    const rawMeta = host.dataset.galleryMeta;
    if (!rawMeta) return null;
    try {
      const meta = JSON.parse(rawMeta) as GalleryMeta;
      if (!meta.images || meta.images.length === 0) return null;
      return meta;
    } catch {
      return null;
    }
  };

  const openFromHost = (host: HTMLElement) => {
    const meta = readMeta(host);
    if (!meta) return;
    const startIdx = parseInt(host.dataset.galleryIndex || '0', 10);
    open(meta, startIdx);
  };

  document.addEventListener('click', (event) => {
    const target = event.target as HTMLElement;
    if (target.closest(DETAIL_ENTER)) return;

    const btn = target.closest<HTMLElement>('[data-gallery-meta]');
    if (!btn || btn.closest('#gallery-lightbox')) return;

    event.preventDefault();
    if (btn.matches(PREVIEW_HOST)) {
      if (previewTimer) window.clearTimeout(previewTimer);
      previewTimer = window.setTimeout(() => {
        previewTimer = null;
        openFromHost(btn);
      }, 180);
      return;
    }

    openFromHost(btn);
  });

  document.addEventListener('dblclick', (event) => {
    const target = event.target as HTMLElement;
    if (target.closest(DETAIL_ENTER)) return;
    const host = target.closest<HTMLElement>(PREVIEW_HOST);
    if (!host) return;
    if (previewTimer) {
      window.clearTimeout(previewTimer);
      previewTimer = null;
    }
    const meta = readMeta(host);
    if (!meta?.detailUrl) return;
    event.preventDefault();
    window.location.href = meta.detailUrl;
  });

  lightbox.querySelector('.glbx__backdrop')?.addEventListener('click', close);
  lightbox.querySelector('.glbx__close')?.addEventListener('click', close);
  prevBtn?.addEventListener('click', prev);
  nextBtn?.addEventListener('click', next);
  inspectBtn?.addEventListener('click', () => setInspecting(!inspecting));
  img?.addEventListener('dblclick', () => setInspecting(!inspecting));
  filmTrack?.addEventListener('click', (event) => {
    const item = (event.target as HTMLElement).closest<HTMLElement>('.glbx__film-item');
    if (!item) return;
    const index = parseInt(item.dataset.index || '0', 10);
    showImage(index);
  });

  imageArea?.addEventListener('wheel', (event) => {
    if (lightbox.hidden || !inspecting) return;
    event.preventDefault();
    inspectScale = Math.min(3, Math.max(1, inspectScale + (event.deltaY < 0 ? 0.18 : -0.18)));
    if (inspectScale === 1) {
      inspectX = 0;
      inspectY = 0;
    }
    applyInspectTransform();
  }, { passive: false });

  imageArea?.addEventListener('pointerdown', (event) => {
    if (!inspecting || event.button !== 0) return;
    if ((event.target as HTMLElement).closest('button')) return;
    event.preventDefault();
    dragging = true;
    dragStartX = event.clientX;
    dragStartY = event.clientY;
    dragOriginX = inspectX;
    dragOriginY = inspectY;
    imageArea.classList.add('is-panning');
    imageArea.setPointerCapture(event.pointerId);
  });

  imageArea?.addEventListener('pointermove', (event) => {
    if (!dragging) return;
    inspectX = dragOriginX + (event.clientX - dragStartX);
    inspectY = dragOriginY + (event.clientY - dragStartY);
    applyInspectTransform();
  });

  const endPan = () => {
    dragging = false;
    imageArea?.classList.remove('is-panning');
  };
  imageArea?.addEventListener('pointerup', endPan);
  imageArea?.addEventListener('pointercancel', endPan);

  document.addEventListener('keydown', (event) => {
    if (lightbox.hidden) {
      const host = event.target as HTMLElement;
      if ((event.key === 'Enter' || event.key === ' ') && host.matches?.(PREVIEW_HOST)) {
        event.preventDefault();
        openFromHost(host);
      }
      return;
    }
    if (event.key === 'Escape') {
      event.preventDefault();
      if (inspecting) setInspecting(false);
      else close();
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault();
      prev();
    } else if (event.key === 'ArrowRight') {
      event.preventDefault();
      next();
    }
  });

  lightbox.addEventListener('touchstart', (event) => {
    touchStartX = event.changedTouches[0].screenX;
  }, { passive: true });

  lightbox.addEventListener('touchend', (event) => {
    if (inspecting) return;
    touchEndX = event.changedTouches[0].screenX;
    const diff = touchStartX - touchEndX;
    if (Math.abs(diff) > 60) {
      if (diff > 0) next();
      else prev();
    }
  });
}
