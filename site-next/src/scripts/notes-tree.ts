export function initNotesTree(root: ParentNode = document) {
  const trees = root.querySelectorAll<HTMLElement>('.notes-nav-tree');

  trees.forEach((tree) => {
    if (tree.dataset.treeBound === 'true') return;
    tree.dataset.treeBound = 'true';

    tree.addEventListener('click', (event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;

      const toggle = target.closest<HTMLButtonElement>('.notes-nav-item__toggle');
      if (!toggle || !tree.contains(toggle)) return;

      event.preventDefault();
      setExpanded(toggle, toggle.getAttribute('aria-expanded') !== 'true');
    });
  });

  openPanelsFromHash(root);
}

function setExpanded(toggle: HTMLButtonElement, open: boolean) {
  const panelId = toggle.getAttribute('aria-controls');
  const panel = panelId ? document.getElementById(panelId) : null;

  toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
  toggle.classList.toggle('is-open', open);
  panel?.classList.toggle('is-open', open);
}

function openPanelsFromHash(root: ParentNode) {
  const hash = window.location.hash.slice(1);
  if (!hash.startsWith('folder-')) return;

  const slugParts = hash.slice('folder-'.length).split('--').filter(Boolean);
  for (let i = 1; i <= slugParts.length; i += 1) {
    const panelId = `notes-panel-${slugParts.slice(0, i).join('--')}`;
    const toggle = root.querySelector<HTMLButtonElement>(`[data-panel-id="${panelId}"]`);
    if (toggle) setExpanded(toggle, true);
  }
}
