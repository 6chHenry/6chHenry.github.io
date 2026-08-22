/**
 * 杂谈页「抽屉索引牌」：页首一排分类索引牌，
 * 滚动时吸顶成毛玻璃条，金色指示珠滑向当前所在的抽屉，点击平滑跳转。
 */

export function initEssayDrawerNav(): void {
  const nav = document.querySelector<HTMLElement>('[data-drawer-nav]');
  if (!nav) return;
  const rail = nav.querySelector<HTMLElement>('.essay-drawers__rail');
  const bead = nav.querySelector<HTMLElement>('.essay-drawers__bead');
  const tabs = Array.from(nav.querySelectorAll<HTMLAnchorElement>('[data-drawer-tab]'));
  if (!rail || !bead || tabs.length === 0) return;

  const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let current = tabs[0];

  const place = (tab: HTMLAnchorElement, instant = false) => {
    current = tab;
    for (const t of tabs) {
      if (t === tab) t.setAttribute('aria-current', 'true');
      else t.removeAttribute('aria-current');
    }
    if (instant) bead.style.transition = 'none';
    bead.style.width = `${tab.offsetWidth}px`;
    bead.style.transform = `translateX(${tab.offsetLeft}px)`;
    if (instant) {
      requestAnimationFrame(() => {
        bead.style.transition = '';
      });
    }
    /* 横向溢出时让活动牌保持可见 */
    const target = tab.offsetLeft - rail.clientWidth / 2 + tab.offsetWidth / 2;
    rail.scrollTo({ left: Math.max(0, target), behavior: instant || reduced() ? 'auto' : 'smooth' });
  };

  place(tabs[0], true);

  /* 点击：平滑跳转（留出吸顶条的高度），并立刻把珠子滑过去 */
  tabs.forEach((tab) => {
    tab.addEventListener('click', (event) => {
      const section = document.getElementById(tab.hash.slice(1));
      if (!section) return;
      event.preventDefault();
      const top = section.getBoundingClientRect().top + window.scrollY - nav.offsetHeight - 14;
      window.scrollTo({ top: Math.max(0, top), behavior: reduced() ? 'auto' : 'smooth' });
      place(tab);
      history.replaceState(null, '', tab.hash);
    });
  });

  /* 滚动侦测：视口中带（35%–65%）落在哪个抽屉，珠子就滑向谁 */
  const sections = tabs
    .map((tab) => document.getElementById(tab.hash.slice(1)))
    .filter((section): section is HTMLElement => section !== null);
  const tabFor = new Map(sections.map((section) => [section.id, tabForTab(section.id)]));

  function tabForTab(id: string): HTMLAnchorElement {
    return tabs.find((tab) => tab.hash === `#${id}`) ?? tabs[0];
  }

  const spy = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          place(tabFor.get(entry.target.id) ?? tabs[0]);
          break;
        }
      }
    },
    { rootMargin: '-32% 0px -62% 0px', threshold: 0 },
  );
  sections.forEach((section) => spy.observe(section));

  /* 页底兜底：最后一个抽屉太短时也能点亮 */
  window.addEventListener(
    'scroll',
    () => {
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) {
        place(tabs[tabs.length - 1]);
      }
    },
    { passive: true },
  );

  /* 吸顶侦测：哨兵离开视口即进入吸顶状态 */
  const sentinel = document.querySelector('.essay-drawers-sentinel');
  if (sentinel) {
    new IntersectionObserver(([entry]) => nav.classList.toggle('is-stuck', !entry.isIntersecting), {
      threshold: 0,
    }).observe(sentinel);
  }

  window.addEventListener('resize', () => place(current, true));
}
