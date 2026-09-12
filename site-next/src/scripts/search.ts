let pagefindUiLoaded = false;

const base = import.meta.env.BASE_URL;
const bundlePath = `${base}pagefind/`;

/**
 * Pagefind UI 自带的是英文文案，这里整体换成果园语气的中文。
 * 占位符（[SEARCH_TERM] / [COUNT] 等）由 Pagefind 自己替换，不要改动拼写。
 */
const PAGEFIND_ZH: Record<string, string> = {
  placeholder: '搜一搜这片林子',
  clear_search: '清空',
  load_more: '再翻一页',
  search_label: '站内搜索',
  filters_label: '筛一筛',
  zero_results: '林子里没找到「[SEARCH_TERM]」',
  many_results: '「[SEARCH_TERM]」有 [COUNT] 处踪迹',
  one_result: '「[SEARCH_TERM]」有 [COUNT] 处踪迹',
  total_zero_results: '什么也没找着',
  total_one_result: '[COUNT] 处踪迹',
  total_many_results: '[COUNT] 处踪迹',
  alt_search: '「[SEARCH_TERM]」没找着，先看看「[DIFFERENT_TERM]」',
  search_suggestion: '「[SEARCH_TERM]」没找着，要不试试这些：',
  searching: '正在翻找「[SEARCH_TERM]」……',
  results_label: '搜索结果',
  keyboard_navigate: '切换',
  keyboard_select: '打开',
  keyboard_clear: '清空',
  keyboard_close: '关闭',
  keyboard_search: '搜索',
  error_search: '搜索出了点问题，稍后再试试',
  filter_selected_one: '选中 [COUNT] 项',
  filter_selected_many: '选中 [COUNT] 项',
  input_hint: '边打字边出结果',
  loading: '正在翻找……',
};

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
    translations: PAGEFIND_ZH,
  });
  container.dataset.mounted = 'true';
}

/** 空态提示：输入框为空时露出来，一开始搜就收起来 */
function syncSearchHint(modal: HTMLElement, hint: HTMLElement) {
  const query = modal.querySelector<HTMLInputElement>('input')?.value.trim() ?? '';
  hint.hidden = query.length > 0;
}

export function initSearchModal() {
  const modal = document.getElementById('search-modal');
  const openBtn = document.getElementById('search-open');
  if (!modal || !openBtn) return;

  const hint = modal.querySelector<HTMLElement>('[data-search-hint]');

  const open = async () => {
    modal.hidden = false;
    document.body.style.overflow = 'hidden';
    await mountSearch('#pagefind-search');
    const input = modal.querySelector<HTMLInputElement>('input[type="search"], input');
    input?.focus();
    if (hint) syncSearchHint(modal, hint);
  };

  const close = () => {
    modal.hidden = true;
    document.body.style.overflow = '';
  };

  if (hint) {
    const sync = () => syncSearchHint(modal, hint);
    modal.addEventListener('input', sync);
    // 清空按钮走的是点击，不一定会派发 input 事件
    modal.addEventListener('click', sync);
    modal.addEventListener('keyup', sync);
  }

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
