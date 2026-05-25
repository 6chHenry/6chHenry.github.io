interface GalleryMeta {
  images: string[];
  title: string;
  description: string;
  category: string;
  date: string;
  detailUrl: string;
}

let currentMeta: GalleryMeta | null = null;
let currentIndex = 0;

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

export function initGalleryLightbox() {
  const lightbox = document.getElementById('gallery-lightbox');
  if (!lightbox) return;

  const img = lightbox.querySelector<HTMLImageElement>('.glbx__image');
  const counter = lightbox.querySelector<HTMLElement>('.glbx__counter');
  const catEl = lightbox.querySelector<HTMLElement>('.glbx__category');
  const titleEl = lightbox.querySelector<HTMLElement>('.glbx__title');
  const descEl = lightbox.querySelector<HTMLElement>('.glbx__desc');
  const dateEl = lightbox.querySelector<HTMLElement>('.glbx__date');
  const detailBtn = lightbox.querySelector<HTMLAnchorElement>('.glbx__detail-btn');
  const prevBtn = lightbox.querySelector<HTMLButtonElement>('.glbx__nav--prev');
  const nextBtn = lightbox.querySelector<HTMLButtonElement>('.glbx__nav--next');

  let loaded = false;

  const showImage = (index: number) => {
    if (!img || !counter || !currentMeta) return;

    const isNav = loaded && index !== currentIndex;

    if (isNav) {
      img.classList.add('is-switching');
      setTimeout(() => {
        currentIndex = index;
        img.src = currentMeta.images[currentIndex];
        const onDone = () => {
          img.classList.remove('is-switching');
          img.onload = null;
          img.onerror = null;
        };
        img.onload = onDone;
        img.onerror = onDone;
        counter.textContent = `${index + 1} / ${currentMeta.images.length}`;
        if (prevBtn) prevBtn.style.visibility = index > 0 ? '' : 'hidden';
        if (nextBtn) nextBtn.style.visibility = index < currentMeta.images.length - 1 ? '' : 'hidden';
      }, 150);
    } else {
      currentIndex = index;
      img.src = currentMeta.images[currentIndex];
      img.classList.remove('is-switching');
      loaded = true;
      counter.textContent = `${index + 1} / ${currentMeta.images.length}`;
      if (prevBtn) prevBtn.style.visibility = index > 0 ? '' : 'hidden';
      if (nextBtn) nextBtn.style.visibility = index < currentMeta.images.length - 1 ? '' : 'hidden';
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

    showImage(startIndex);
    lightbox.hidden = false;
    document.body.style.overflow = 'hidden';
  };

  const close = () => {
    lightbox.hidden = true;
    document.body.style.overflow = '';
    currentMeta = null;
    currentIndex = 0;
    loaded = false;
  };

  const prev = () => {
    if (currentIndex > 0) showImage(currentIndex - 1);
  };

  const next = () => {
    if (currentIndex < (currentMeta?.images.length ?? 1) - 1) showImage(currentIndex + 1);
  };

  document.addEventListener('click', (event) => {
    const btn = (event.target as HTMLElement).closest<HTMLElement>('[data-gallery-meta]');
    if (!btn) return;
    const card = (event.target as HTMLElement).closest<HTMLElement>('.gallery-card');
    if (card) {
      const href = (card as HTMLAnchorElement).href || '';
      const isJapanCard = href.includes('/gallery/photography/japan');
      if (isJapanCard) {
        window.location.href = href.replace('/gallery/photography/japan', '/gallery/japan');
        event.preventDefault();
        return;
      }
      event.preventDefault();
    }

    const rawMeta = btn.dataset.galleryMeta;
    if (!rawMeta) return;

    let meta: GalleryMeta;
    try {
      meta = JSON.parse(rawMeta);
    } catch {
      return;
    }

    if (!meta.images || meta.images.length === 0) return;
    const startIdx = parseInt(btn.dataset.galleryIndex || '0', 10);
    open(meta, startIdx);
  });

  lightbox.querySelector('.glbx__backdrop')?.addEventListener('click', close);
  lightbox.querySelector('.glbx__close')?.addEventListener('click', close);
  prevBtn?.addEventListener('click', prev);
  nextBtn?.addEventListener('click', next);

  document.addEventListener('keydown', (event) => {
    if (lightbox.hidden) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      close();
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
    touchEndX = event.changedTouches[0].screenX;
    const diff = touchStartX - touchEndX;
    if (Math.abs(diff) > 60) {
      if (diff > 0) next();
      else prev();
    }
  });
}
