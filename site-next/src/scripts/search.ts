let pagefindLoaded = false;

const base = import.meta.env.BASE_URL;

async function assetExists(url: string): Promise<boolean> {
  try {
    const response = await fetch(url, { method: 'HEAD' });
    return response.ok;
  } catch {
    return false;
  }
}

async function ensurePagefindStyles(): Promise<boolean> {
  const href = `${base}pagefind/pagefind-ui.css`;
  if (document.querySelector('link[data-pagefind-ui]')) return true;
  if (!(await assetExists(href))) return false;

  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = href;
  link.dataset.pagefindUi = 'true';
  document.head.appendChild(link);
  return true;
}

async function ensurePagefind(): Promise<boolean> {
  if (pagefindLoaded) return true;

  const jsUrl = `${base}pagefind/pagefind.js`;
  if (!(await assetExists(jsUrl))) return false;

  // @ts-expect-error pagefind is injected at build time
  window.pagefind = await import(/* @vite-ignore */ jsUrl);
  pagefindLoaded = true;
  return true;
}

function showSearchUnavailable(container: HTMLElement) {
  container.innerHTML =
    '<p class="search-unavailable">搜索索引尚未生成。本地开发请先运行 <code>npm run build</code>，或使用 <code>npm run preview</code> 预览。</p>';
  container.dataset.mounted = 'true';
}

async function mountSearch(containerSelector: string) {
  const container = document.querySelector<HTMLElement>(containerSelector);
  if (!container || container.dataset.mounted === 'true') return;

  const stylesReady = await ensurePagefindStyles();
  const pagefindReady = await ensurePagefind();
  if (!stylesReady || !pagefindReady) {
    showSearchUnavailable(container);
    return;
  }

  // @ts-expect-error pagefind UI loaded dynamically
  const { PagefindUI } = await import(/* @vite-ignore */ `${base}pagefind/pagefind-ui.js`);
  new PagefindUI({
    element: containerSelector,
    showImages: false,
    resetStyles: false,
  });
  container.dataset.mounted = 'true';
}

export function initSearchModal() {
  const modal = document.getElementById('search-modal');
  const openBtn = document.getElementById('search-open');
  if (!modal || !openBtn) return;

  const open = async () => {
    modal.hidden = false;
    document.body.style.overflow = 'hidden';
    await mountSearch('#pagefind-search');
    const input = modal.querySelector<HTMLInputElement>('input[type="search"], input');
    input?.focus();
  };

  const close = () => {
    modal.hidden = true;
    document.body.style.overflow = '';
  };

  openBtn.addEventListener('click', open);
  modal.querySelectorAll('[data-search-close]').forEach((el) => el.addEventListener('click', close));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !modal.hidden) close();
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      open();
    }
  });
}

export { mountSearch };
