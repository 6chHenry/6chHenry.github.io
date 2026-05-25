export function initJapanCarousel() {
  const carousel = document.querySelector<HTMLElement>('.jg-carousel');
  if (!carousel) return;

  const track = carousel.querySelector<HTMLElement>('.jg-carousel__track');
  const dots = carousel.querySelectorAll<HTMLElement>('.jg-carousel__dot');
  const prevBtn = carousel.querySelector<HTMLButtonElement>('.jg-carousel__nav--prev');
  const nextBtn = carousel.querySelector<HTMLButtonElement>('.jg-carousel__nav--next');
  if (!track || dots.length === 0) return;

  let current = 0;
  const total = dots.length;
  let interval: ReturnType<typeof setInterval> | null = null;

  const scrollToSection = (index: number) => {
    const slides = carousel.querySelectorAll<HTMLElement>('.jg-carousel__slide');
    const targetId = slides[index]?.dataset.target;
    if (!targetId) return;
    const section = document.getElementById(targetId);
    if (!section) return;
    const timeline = document.querySelector<HTMLElement>('.jg-timeline');
    const offset = (timeline?.offsetHeight ?? 0) + 24;
    const top = section.getBoundingClientRect().top + window.scrollY - offset;
    window.scrollTo({ top, behavior: 'smooth' });
  };

  const goTo = (index: number) => {
    current = ((index % total) + total) % total;
    track.style.transform = `translateX(-${current * 100}%)`;
    dots.forEach((d, i) => d.classList.toggle('is-active', i === current));
  };

  const next = () => goTo(current + 1);
  const prev = () => goTo(current - 1);

  prevBtn?.addEventListener('click', prev);
  nextBtn?.addEventListener('click', next);

  dots.forEach((dot, i) => {
    dot.addEventListener('click', () => goTo(i));
  });

  carousel.querySelectorAll('.jg-carousel__slide').forEach((slide, i) => {
    slide.addEventListener('click', () => scrollToSection(i));
  });

  const startAuto = () => {
    stopAuto();
    interval = setInterval(next, 5000);
  };

  const stopAuto = () => {
    if (interval) { clearInterval(interval); interval = null; }
  };

  startAuto();
  carousel.addEventListener('mouseenter', stopAuto);
  carousel.addEventListener('mouseleave', startAuto);
  carousel.addEventListener('touchstart', stopAuto, { passive: true });
  carousel.addEventListener('touchend', () => setTimeout(startAuto, 3000));
}

export function initJapanTimeline() {
  const timeline = document.querySelector<HTMLElement>('.jg-timeline');
  if (!timeline) return;

  const stops = timeline.querySelectorAll<HTMLElement>('.jg-timeline__stop');
  const sections = document.querySelectorAll<HTMLElement>('.jg-section');

  if (stops.length === 0 || sections.length === 0) return;

  stops.forEach((stop) => {
    stop.addEventListener('click', () => {
      const targetId = stop.dataset.target;
      if (!targetId) return;
      const section = document.getElementById(targetId);
      if (!section) return;
      const offset = timeline.offsetHeight + 24;
      const top = section.getBoundingClientRect().top + window.scrollY - offset;
      window.scrollTo({ top, behavior: 'smooth' });
    });
  });

  const observer = new IntersectionObserver(
    (entries) => {
      let activeIdx = -1;
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const idx = Array.from(sections).indexOf(entry.target as HTMLElement);
          if (idx > activeIdx) activeIdx = idx;
        }
      });
      if (activeIdx >= 0) {
        stops.forEach((s, i) => {
          const isActive = i === activeIdx;
          s.classList.toggle('is-active', isActive);
          const dot = s.querySelector<HTMLElement>('.jg-timeline__dot');
          if (dot) {
            const color = getComputedStyle(s).getPropertyValue('--stop-color').trim();
            dot.style.background = isActive ? color : 'var(--ch-surface-solid)';
          }
        });
      }
    },
    { rootMargin: '-20% 0px -60% 0px', threshold: 0 },
  );

  sections.forEach((section) => observer.observe(section));

  // Set initial active
  stops[0]?.classList.add('is-active');
}

export function initJapanSections() {
  const sections = document.querySelectorAll<HTMLElement>('.jg-section');
  if (sections.length === 0) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
        }
      });
    },
    { rootMargin: '0px 0px -8% 0px', threshold: 0.1 },
  );

  sections.forEach((section) => observer.observe(section));
}
