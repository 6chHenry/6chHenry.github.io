let pagefindUiLoaded = false;

const base = import.meta.env.BASE_URL;
const bundlePath = `${base}pagefind/`;

declare global {
  interface Window {
    PagefindUI?: new (options: Record<string, unknown>) => {
      triggerSearch?: (term: string) => void;
      destroy?: () => void;
    };
  }
}

async function assetExists(url: string): Promise<boolean> {
  try {
    const response = await fetch(url, { method: 'GET', cache: 'no-store' });
    return response.ok;
  } catch {
    return false;
  }
}

function loadStylesheet(href: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`link[data-pagefind-ui][href="${href}"]`)) {
      resolve();
      return;
    }

    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = href;
    link.dataset.pagefindUi = 'true';
    link.onload = () => resolve();
    link.onerror = () => reject(new Error(`Failed to load stylesheet: ${href}`));
    document.head.appendChild(link);
  });
}

function loadPagefindUiScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (window.PagefindUI) {
      resolve();
      return;
    }

    const existing = document.querySelector<HTMLScriptElement>(`script[data-pagefind-ui-script="${src}"]`);
    if (existing) {
      existing.addEventListener('load', () => resolve(), { once: true });
      existing.addEventListener('error', () => reject(new Error(`Failed to load script: ${src}`)), { once: true });
      return;
    }

    const script = document.createElement('script');
    script.src = src;
    script.dataset.pagefindUiScript = src;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    document.head.appendChild(script);
  });
}

async function ensurePagefindUi(): Promise<boolean> {
  if (pagefindUiLoaded && window.PagefindUI) return true;

  const cssUrl = `${bundlePath}pagefind-ui.css`;
  const jsUrl = `${bundlePath}pagefind-ui.js`;

  const [cssReady, jsReady] = await Promise.all([assetExists(cssUrl), assetExists(jsUrl)]);
  if (!cssReady || !jsReady) return false;

  await loadStylesheet(cssUrl);
  await loadPagefindUiScript(jsUrl);

  pagefindUiLoaded = Boolean(window.PagefindUI);
  return pagefindUiLoaded;
}

function showSearchUnavailable(container: HTMLElement) {
  container.innerHTML =
    '<p class="search-unavailable">搜索索引尚未就绪。请先运行 <code>npm run build</code>（会自动同步索引到本地 dev），或使用 <code>npm run preview</code> 预览生产站点。</p>';
  container.dataset.mounted = 'true';
}

async function mountSearch(containerSelector: string) {
  const container = document.querySelector<HTMLElement>(containerSelector);
  if (!container || container.dataset.mounted === 'true') return;

  const ready = await ensurePagefindUi();
  if (!ready || !window.PagefindUI) {
    showSearchUnavailable(container);
    return;
  }

  new window.PagefindUI({
    element: containerSelector,
    bundlePath,
    pageSize: 10,
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
