function parsePoint(raw: string | undefined): [number, number] {
  const [x = 0, y = 0] = (raw ?? '').split(',').map(Number);
  return [x, y];
}

function setPoint(element: SVGCircleElement, point: [number, number]) {
  element.setAttribute('cx', String(point[0]));
  element.setAttribute('cy', String(point[1]));
}

function parseHighlights(raw: string | undefined): string[] {
  if (!raw) return [];
  try {
    const value = JSON.parse(raw);
    return Array.isArray(value) ? value.map(String) : [];
  } catch {
    return [];
  }
}

export function initNordicTravel() {
  const root = document.querySelector<HTMLElement>('[data-nordic-travel]');
  if (!root || root.dataset.nordicReady === 'true') return;
  root.dataset.nordicReady = 'true';

  const tabs = [...root.querySelectorAll<HTMLButtonElement>('[data-nordic-day]')];
  const route = root.querySelector<SVGPathElement>('[data-nordic-route]');
  const origin = root.querySelector<SVGCircleElement>('[data-nordic-origin]');
  const destination = root.querySelector<SVGCircleElement>('[data-nordic-destination]');
  const ticket = root.querySelector<HTMLElement>('[data-nordic-ticket]');
  const scene = root.querySelector<HTMLElement>('[data-nordic-scene]');

  if (!route || !origin || !destination || !ticket || !scene) return;

  const ticketCode = ticket.querySelector<HTMLElement>('[data-ticket-code]');
  const ticketMode = ticket.querySelector<HTMLElement>('[data-ticket-mode]');
  const ticketRoute = ticket.querySelector<HTMLElement>('[data-ticket-route]');
  const ticketSummary = ticket.querySelector<HTMLElement>('[data-ticket-summary]');
  const ticketDetail = ticket.querySelector<HTMLElement>('[data-ticket-detail]');
  const sceneNumber = scene.querySelector<HTMLElement>('[data-scene-number]');
  const sceneTitle = scene.querySelector<HTMLElement>('[data-scene-title]');
  const sceneSummary = scene.querySelector<HTMLElement>('[data-scene-summary]');
  const sceneDetail = scene.querySelector<HTMLElement>('[data-scene-detail]');

  let activeDay = tabs.find((tab) => tab.getAttribute('aria-selected') === 'true') ?? tabs[0];
  let dayTransition = 0;

  const activate = (tab: HTMLButtonElement, moveFocus = false) => {
    if (tab === activeDay) {
      if (moveFocus) tab.focus();
      return;
    }
    activeDay = tab;
    const transition = ++dayTransition;

    for (const item of tabs) {
      const selected = item === tab;
      item.setAttribute('aria-selected', String(selected));
      item.tabIndex = selected ? 0 : -1;
    }
    if (moveFocus) tab.focus();

    const path = tab.dataset.routePath ?? '';
    route.classList.remove('is-active');
    destination.classList.remove('is-arriving');
    route.setAttribute('d', path);
    route.toggleAttribute('hidden', path.length === 0);
    origin.toggleAttribute('hidden', path.length === 0);
    setPoint(origin, parsePoint(tab.dataset.origin));
    setPoint(destination, parsePoint(tab.dataset.destination));

    if (path) {
      void route.getBoundingClientRect();
      route.classList.add('is-active');
    }
    void destination.getBoundingClientRect();
    destination.classList.add('is-arriving');

    ticket.classList.add('is-changing');
    scene.classList.add('is-changing');
    window.setTimeout(() => {
      if (transition !== dayTransition) return;

      if (ticketCode) ticketCode.textContent = tab.dataset.ticketCode ?? '';
      if (ticketMode) ticketMode.textContent = tab.dataset.ticketMode ?? '';
      if (ticketRoute) ticketRoute.textContent = tab.dataset.ticketRoute ?? '';
      if (ticketSummary) ticketSummary.textContent = tab.dataset.ticketSummary ?? '';
      if (ticketDetail) ticketDetail.textContent = tab.dataset.ticketDetail ?? '';
      if (sceneNumber) sceneNumber.textContent = tab.dataset.sceneNumber ?? '';
      if (sceneTitle) sceneTitle.textContent = tab.dataset.sceneTitle ?? '';
      if (sceneSummary) sceneSummary.textContent = tab.dataset.sceneSummary ?? '';
      if (sceneDetail) sceneDetail.textContent = tab.dataset.sceneDetail ?? '';
      ticket.classList.remove('is-changing');
      scene.classList.remove('is-changing');
    }, 140);
  };

  tabs.forEach((tab, index) => {
    tab.tabIndex = index === 0 ? 0 : -1;
    tab.addEventListener('click', () => activate(tab));
    tab.addEventListener('keydown', (event) => {
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const current = tabs.indexOf(tab);
      let next = current;
      if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = tabs.length - 1;
      else if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (current + 1) % tabs.length;
      else next = (current - 1 + tabs.length) % tabs.length;
      activate(tabs[next], true);
    });
  });

  const cityTabs = [...root.querySelectorAll<HTMLButtonElement>('[data-city-tab]')];
  const cityFeature = root.querySelector<HTMLElement>('[data-city-feature]');
  const cityName = root.querySelector<HTMLElement>('[data-city-name]');
  const cityCn = root.querySelector<HTMLElement>('[data-city-cn]');
  const cityRole = root.querySelector<HTMLElement>('[data-city-role]');
  const cityDuration = root.querySelector<HTMLElement>('[data-city-duration]');
  const cityTransport = root.querySelector<HTMLElement>('[data-city-transport]');
  const cityHighlights = root.querySelector<HTMLElement>('[data-city-highlights]');
  const cityNote = root.querySelector<HTMLElement>('[data-city-note]');
  let activeCity = cityTabs.find((tab) => tab.getAttribute('aria-selected') === 'true') ?? cityTabs[0];
  let cityTransition = 0;

  const activateCity = (tab: HTMLButtonElement, moveFocus = false) => {
    if (tab === activeCity) {
      if (moveFocus) tab.focus();
      return;
    }
    activeCity = tab;
    const transition = ++cityTransition;

    for (const item of cityTabs) {
      const selected = item === tab;
      item.setAttribute('aria-selected', String(selected));
      item.tabIndex = selected ? 0 : -1;
    }
    if (moveFocus) tab.focus();

    cityFeature?.classList.add('is-changing');
    window.setTimeout(() => {
      if (transition !== cityTransition) return;

      if (cityName) cityName.textContent = tab.dataset.cityName ?? '';
      if (cityCn) cityCn.textContent = tab.dataset.cityCn ?? '';
      if (cityRole) cityRole.textContent = tab.dataset.cityRole ?? '';
      if (cityDuration) cityDuration.textContent = tab.dataset.cityDuration ?? '';
      if (cityTransport) cityTransport.textContent = tab.dataset.cityTransport ?? '';
      if (cityNote) cityNote.textContent = tab.dataset.cityNote ?? '';
      if (cityHighlights) {
        cityHighlights.replaceChildren(
          ...parseHighlights(tab.dataset.cityHighlights).map((highlight) => {
            const item = document.createElement('li');
            item.textContent = highlight;
            return item;
          }),
        );
      }
      cityFeature?.classList.remove('is-changing');
    }, 140);
  };

  cityTabs.forEach((tab, index) => {
    tab.tabIndex = index === 0 ? 0 : -1;
    tab.addEventListener('click', () => activateCity(tab));
    tab.addEventListener('keydown', (event) => {
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const current = cityTabs.indexOf(tab);
      let next = current;
      if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = cityTabs.length - 1;
      else if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (current + 1) % cityTabs.length;
      else next = (current - 1 + cityTabs.length) % cityTabs.length;
      activateCity(cityTabs[next], true);
    });
  });

  const reveals = [...root.querySelectorAll<HTMLElement>('[data-cinematic-reveal]')];
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches || !('IntersectionObserver' in window)) {
    reveals.forEach((element) => element.classList.add('is-visible'));
  } else {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      },
      { threshold: 0.12 },
    );
    reveals.forEach((element) => observer.observe(element));
  }
}
