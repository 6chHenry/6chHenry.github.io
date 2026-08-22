/**
 * 杂谈页双形态：展阅（分组列表）⇄ 收纳（野外采集柜的标本木盒）。
 * 记住用户上次的选择；点击木盒切回展阅并平滑跳到对应分区。
 */

const VIEW_KEY = 'essay-view:v1';

export function initEssayView(): void {
  const page = document.querySelector<HTMLElement>('.essay-page');
  const switchEl = document.querySelector<HTMLElement>('[data-view-switch]');
  const buttons = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-view-btn]'));
  const groups = page?.querySelector<HTMLElement>('.essay-groups');
  const trays = document.querySelector<HTMLElement>('[data-essay-trays]');
  const nav = document.querySelector<HTMLElement>('[data-drawer-nav]');
  if (!page || !switchEl || buttons.length === 0 || !groups || !trays) return;

  const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const load = (): 'expanded' | 'collected' => {
    try {
      return localStorage.getItem(VIEW_KEY) === 'collected' ? 'collected' : 'expanded';
    } catch {
      return 'expanded';
    }
  };

  let view: 'expanded' | 'collected' = load();

  const apply = (next: 'expanded' | 'collected') => {
    view = next;
    page.classList.toggle('is-collected', next === 'collected');
    switchEl.dataset.view = next;
    for (const btn of buttons) {
      btn.setAttribute('aria-pressed', btn.dataset.viewBtn === next ? 'true' : 'false');
    }
    groups.hidden = next === 'collected';
    trays.hidden = next !== 'collected';
    nav.hidden = next === 'collected';
    document.querySelector('.essay-drawers-sentinel')?.toggleAttribute('hidden', next === 'collected');
    try {
      localStorage.setItem(VIEW_KEY, next);
    } catch {
      /* 隐私模式下静默失败 */
    }
  };

  apply(view);

  buttons.forEach((btn) => {
    btn.addEventListener('click', () => {
      if (btn.dataset.viewBtn !== view) apply(btn.dataset.viewBtn as 'expanded' | 'collected');
    });
  });

  /* 木盒点击：切回展阅，等它出现后再平滑滚到分区 */
  trays.querySelectorAll<HTMLAnchorElement>('a.essay-tray').forEach((tray) => {
    tray.addEventListener('click', (event) => {
      const section = document.getElementById(tray.hash.slice(1));
      if (!section) return;
      event.preventDefault();
      apply('expanded');
      const jump = () => {
        const navH = nav && !nav.hidden ? nav.offsetHeight + 14 : 14;
        window.scrollTo({
          top: Math.max(0, section.getBoundingClientRect().top + window.scrollY - navH),
          behavior: reduced() ? 'auto' : 'smooth',
        });
        history.replaceState(null, '', tray.hash);
      };
      window.setTimeout(jump, reduced() ? 0 : 90);
    });
  });
}
