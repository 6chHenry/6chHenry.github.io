export function initHomeStats() {
  const hero = document.querySelector('.home-hero');
  if (!hero) return;

  document.body.classList.add('page-home');

  const statsBtn = document.getElementById('home-stats-toggle');
  const statistics = document.getElementById('statistics');
  statsBtn?.addEventListener('click', () => {
    statistics?.classList.toggle('is-visible');
    const visible = statistics?.classList.contains('is-visible');
    statsBtn.setAttribute('aria-expanded', visible ? 'true' : 'false');
  });

  const el = document.getElementById('web-time');
  if (!el) return;

  const start = new Date('2025/02/28 22:00:00').getTime();
  const update = () => {
    let diff = Date.now() - start;
    const y = Math.floor(diff / (365 * 24 * 3600 * 1000));
    diff -= y * 365 * 24 * 3600 * 1000;
    const d = Math.floor(diff / (24 * 3600 * 1000));
    const h = Math.floor((diff / (3600 * 1000)) % 24);
    const m = Math.floor((diff / (60 * 1000)) % 60);
    el.innerHTML =
      y > 0
        ? `${y}<span> </span>y<span> </span>${d}<span> </span>d<span> </span>${h}<span> </span>h<span> </span>${m}<span> </span>m`
        : `${d}<span> </span>d<span> </span>${h}<span> </span>h<span> </span>${m}<span> </span>m`;
  };

  update();
  window.setInterval(update, 60_000);
}
