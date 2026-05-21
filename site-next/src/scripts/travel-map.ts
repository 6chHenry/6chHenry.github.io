import L from 'leaflet';
import { travelRoutes, type TravelRoute, type TravelStop } from '../data/travel-routes';

function resolveTilePath(path: string): string {
  if (path.startsWith('http')) return path;
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  return `${base}${path.startsWith('/') ? path : `/${path}`}`;
}

function buildStopList(stops: TravelStop[]): string {
  return stops
    .map(
      (stop, index) => `
        <li>
          <span class="travel-map__stop-index">${index + 1}</span>
          <span class="travel-map__stop-main">
            <strong>${stop.name}</strong>
            ${stop.time ? `<em>${stop.time}</em>` : ''}
          </span>
        </li>`,
    )
    .join('');
}

function markerIcon(index: number) {
  return L.divIcon({
    className: 'travel-map-marker',
    html: `<span>${index}</span>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14],
    popupAnchor: [0, -16],
  });
}

function renderShell(root: HTMLElement, route: TravelRoute) {
  root.innerHTML = `
    <div class="travel-map__header">
      <div>
        <p class="travel-map__eyebrow">TRAVEL ROUTE</p>
        <h3>${route.title}</h3>
      </div>
      <p>${route.subtitle}</p>
    </div>
    <div class="travel-map__canvas" role="img" aria-label="${route.title} route map"></div>
    <div class="travel-map__footer">
      <p>Approximate route based on key stops from the travel note.</p>
      <ol>${buildStopList(route.stops)}</ol>
    </div>`;
}

function initTravelMap(root: HTMLElement) {
  if (root.dataset.travelMapReady === 'true') return;
  const routeId = root.dataset.route;
  if (!routeId) return;
  const route = travelRoutes[routeId];
  if (!route) return;

  root.dataset.travelMapReady = 'true';
  root.classList.add('travel-map--ready');
  renderShell(root, route);

  const canvas = root.querySelector<HTMLElement>('.travel-map__canvas');
  if (!canvas) return;

  const map = L.map(canvas, {
    center: route.center,
    zoom: route.zoom,
    scrollWheelZoom: false,
    zoomControl: true,
  });

  L.tileLayer(resolveTilePath('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'), {
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19,
  }).addTo(map);

  const coordinates = route.stops.map((stop) => L.latLng(stop.lat, stop.lng));
  const line = L.polyline(coordinates, {
    color: '#22c55e',
    weight: 4,
    opacity: 0.86,
    lineCap: 'round',
    lineJoin: 'round',
  }).addTo(map);

  route.stops.forEach((stop, index) => {
    L.marker([stop.lat, stop.lng], { icon: markerIcon(index + 1) })
      .bindPopup(
        `<strong>${stop.name}</strong>${stop.time ? `<br><span>${stop.time}</span>` : ''}${
          stop.note ? `<br>${stop.note}` : ''
        }`,
      )
      .addTo(map);
  });

  map.fitBounds(line.getBounds(), { padding: [24, 24] });
  setTimeout(() => map.invalidateSize(), 50);
}

export function initTravelMaps() {
  document.querySelectorAll<HTMLElement>('[data-travel-map]').forEach(initTravelMap);
}
