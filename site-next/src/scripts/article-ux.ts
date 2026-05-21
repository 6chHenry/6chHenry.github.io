const RING_RADIUS = 18;
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;

function getHeaderOffset(): number {
  const raw = getComputedStyle(document.documentElement).getPropertyValue('--ch-header-height').trim();
  const parsed = Number.parseFloat(raw);
  return Number.isFinite(parsed) ? parsed : 68;
}

function prefersReducedMotion(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function computeArticleProgress(article: HTMLElement): number {
  const headerOffset = getHeaderOffset();
  const rect = article.getBoundingClientRect();
  const viewport = window.innerHeight;
  const scrollable = article.offsetHeight - (viewport - headerOffset) * 0.35;
  if (scrollable <= 0) return 0;

  const start = window.scrollY + rect.top - headerOffset;
  const progress = (window.scrollY - start) / scrollable;
  return Math.min(1, Math.max(0, progress));
}

function initReadingProgress() {
  const root = document.querySelector<HTMLElement>('[data-reading-progress-root]');
  const bar = document.querySelector<HTMLElement>('[data-reading-progress]');
  const label = document.querySelector<HTMLElement>('[data-reading-progress-label]');
  const article = document.querySelector<HTMLElement>('.content-article');
  if (!root || !bar || !article) return;

  let raf = 0;
  let lastPercent = -1;

  const apply = (progress: number) => {
    const percent = Math.round(progress * 100);
    bar.style.transform = `scaleX(${progress})`;
    root.setAttribute('aria-valuenow', String(percent));
    if (label) label.textContent = `${percent}%`;

    if (percent !== lastPercent) {
      lastPercent = percent;
      root.classList.toggle('is-complete', percent >= 100);
    }
  };

  const measure = () => {
    const progress = computeArticleProgress(article);
    const scrollable = article.offsetHeight > window.innerHeight * 0.92;
    root.hidden = !scrollable;
    if (!scrollable) {
      apply(0);
      return;
    }
    apply(progress);
  };

  const schedule = () => {
    if (raf) return;
    raf = window.requestAnimationFrame(() => {
      raf = 0;
      measure();
    });
  };

  measure();
  window.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('resize', schedule);
  window.addEventListener('load', schedule);

  if ('ResizeObserver' in window) {
    const observer = new ResizeObserver(schedule);
    observer.observe(article);
  }
}

function initBackToTop() {
  const button = document.querySelector<HTMLButtonElement>('[data-back-to-top]');
  const ring = document.querySelector<SVGCircleElement>('[data-back-to-top-ring]');
  const article = document.querySelector<HTMLElement>('.content-article');
  if (!button || !article) return;

  let raf = 0;

  const update = () => {
    const progress = computeArticleProgress(article);
    const show = window.scrollY > window.innerHeight * 0.45;
    button.hidden = !show;
    button.classList.toggle('is-visible', show);

    if (ring) {
      ring.style.strokeDasharray = `${RING_CIRCUMFERENCE}`;
      ring.style.strokeDashoffset = `${RING_CIRCUMFERENCE * (1 - progress)}`;
    }

    const percent = Math.round(progress * 100);
    button.setAttribute('aria-label', percent > 0 ? `返回顶部，已阅读 ${percent}%` : '返回顶部');
  };

  const schedule = () => {
    if (raf) return;
    raf = window.requestAnimationFrame(() => {
      raf = 0;
      update();
    });
  };

  button.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
    button.blur();
  });

  update();
  window.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('resize', schedule);
}

function initTocSpy() {
  const tocLinks = Array.from(document.querySelectorAll<HTMLAnchorElement>('.content-toc a[href^="#"]'));
  if (tocLinks.length === 0) return;

  const linkById = new Map<string, HTMLAnchorElement>();
  for (const link of tocLinks) {
    const id = decodeURIComponent(link.hash.slice(1));
    if (id) linkById.set(id, link);
  }

  const headings = Array.from(linkById.keys())
    .map((id) => document.getElementById(id))
    .filter((heading): heading is HTMLElement => Boolean(heading));
  if (headings.length === 0) return;

  let activeId = '';
  const setActive = (id: string) => {
    if (id === activeId) return;
    activeId = id;
    for (const link of tocLinks) {
      const isActive = linkById.get(id) === link;
      link.classList.toggle('is-active', isActive);
      if (isActive) link.setAttribute('aria-current', 'true');
      else link.removeAttribute('aria-current');
    }
  };

  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
      if (visible?.target.id) setActive(visible.target.id);
    },
    {
      rootMargin: '-18% 0px -66% 0px',
      threshold: [0, 1],
    },
  );

  headings.forEach((heading) => observer.observe(heading));

  const fallback = () => {
    const current = [...headings].reverse().find((heading) => heading.getBoundingClientRect().top <= 140);
    if (current?.id) setActive(current.id);
  };
  fallback();
  window.addEventListener('scroll', fallback, { passive: true });
}

function initCodeCopyButtons() {
  const blocks = Array.from(document.querySelectorAll<HTMLPreElement>('.prose pre'));
  for (const pre of blocks) {
    if (pre.closest('.code-block')) continue;

    const wrapper = document.createElement('div');
    wrapper.className = 'code-block';
    pre.parentNode?.insertBefore(wrapper, pre);
    wrapper.appendChild(pre);

    const button = document.createElement('button');
    button.className = 'code-copy';
    button.type = 'button';
    button.textContent = '复制';
    button.setAttribute('aria-label', '复制代码');
    wrapper.appendChild(button);

    button.addEventListener('click', async () => {
      const code = pre.querySelector('code')?.textContent ?? pre.textContent ?? '';
      try {
        await navigator.clipboard.writeText(code);
        button.textContent = '已复制';
        button.classList.add('is-copied');
        window.setTimeout(() => {
          button.textContent = '复制';
          button.classList.remove('is-copied');
        }, 1400);
      } catch {
        button.textContent = '复制失败';
        window.setTimeout(() => {
          button.textContent = '复制';
        }, 1400);
      }
    });
  }
}

export function initArticleUx() {
  initReadingProgress();
  initBackToTop();
  initTocSpy();
  initCodeCopyButtons();
}
