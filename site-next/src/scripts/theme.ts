export function initThemeToggle() {
  const button = document.getElementById('theme-toggle');
  if (!button) return;

  /* 开关的 aria-checked 始终跟随当前主题（初始可能由内联脚本按偏好设置） */
  const syncState = () => {
    const dark = document.documentElement.dataset.theme === 'dark';
    button.setAttribute('aria-checked', dark ? 'true' : 'false');
  };
  syncState();

  button.addEventListener('click', () => {
    const current = document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('theme', next);
    syncState();
  });
}
