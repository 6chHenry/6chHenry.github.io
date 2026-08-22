/**
 * 杂谈页「抽屉索引牌」（仅展阅形态）：
 * 吸顶毛玻璃标签条 + 滚动侦测高亮当前分类 + 点击平滑跳转。
 */

export function initEssayDrawerNav(): void {
  const nav = document.querySelector<HTMLElement>('[data-drawer-nav]');
  if (!nav) return;
  const rail = nav.querySelector<HTMLElement>('.essay-drawers__rail');
  const tabs = Array.from(nav.querySelectorAll<HTMLAnchorElement>('[data-drawer-tab]'));
  if (!rail || tabs.length === 0) return;

  const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let current = tabs[0];

  const setActive = (tab: HTMLAnchorElement) => {
    current = tab;
    for (const t of tabs) {
      if (t === tab) t.setAttribute('aria-current', 'true');
      else t.removeAttribute('aria-current');
    }
    /* 横向溢出时让活动牌保持可见 */
    const target = tab.offsetLeft - rail.clientWidth / 2 + tab.offsetWidth / 2;
    rail.scrollTo({ left: Math.max(0, target), behavior: reduced() ? 'auto' : 'smooth' });
  };

  setActive(tabs[0]);

  /* 点击：平滑跳转（留出吸顶条的高度），并立刻点亮目标 */
  tabs.forEach((tab) => {
    tab.addEventListener('click', (event) => {
      const section = document.getElementById(tab.hash.slice(1));
      if (!section) return;
      event.preventDefault();
      const top = section.getBoundingClientRect().top + window.scrollY - nav.offsetHeight - 14;
      window.scrollTo({ top: Math.max(0, top), behavior: reduced() ? 'auto' : 'smooth' });
      setActive(tab);
      history.replaceState(null, '', tab.hash);
    });
  });

  /* 滚动侦测：视口中带（35%–65%）落在哪个抽屉，就点亮谁 */
  const sections = tabs
    .map((tab) => document.getElementById(tab.hash.slice(1)))
    .filter((section): section is HTMLElement => section !== null);
  const tabFor = new Map<string, HTMLAnchorElement>(
    sections.map((section) => [section.id, tabs.find((tab) => tab.hash === `#${section.id}`) ?? tabs[0]]),
  );

  const spy = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          setActive(tabFor.get(entry.target.id) ?? tabs[0]);
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
        setActive(tabs[tabs.length - 1]);
      }
    },
    { passive: true },
  );

  /* 吸顶侦测 */
  const sentinel = document.querySelector('.essay-drawers-sentinel');
  if (sentinel) {
    new IntersectionObserver(([entry]) => nav.classList.toggle('is-stuck', !entry.isIntersecting), {
      threshold: 0,
    }).observe(sentinel);
  }

  window.addEventListener('resize', () => {
    rail.scrollTo({ left: Math.max(0, current.offsetLeft - 24), behavior: 'auto' });
  });

  /* 从别的页面带着 #essay-section-* 锚点进来时（如关于页的爱好小径），
     原生平滑滚动常被首屏布局位移打断而停在页顶——这里显式补一跳 */
  if (window.location.hash.startsWith('#essay-section-')) {
    const jumpToHash = () => {
      const id = decodeURIComponent(window.location.hash.slice(1));
      const section = document.getElementById(id);
      if (!section) return false;
      const top = section.getBoundingClientRect().top + window.scrollY - nav.offsetHeight - 14;
      window.scrollTo({ top: Math.max(0, top), behavior: reduced() ? 'auto' : 'smooth' });
      setActive(tabs.find((tab) => tab.hash === `#${id}`) ?? tabs[0]);
      return true;
    };

    requestAnimationFrame(() => requestAnimationFrame(jumpToHash));
    window.addEventListener('load', () => {
      window.setTimeout(jumpToHash, 80);
    });
  }
}
