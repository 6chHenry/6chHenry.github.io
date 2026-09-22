import L from 'leaflet';
import type { AtlasPlace } from '../data/gallery-atlas';

const root = document.querySelector<HTMLElement>('[data-atlas]');

if (root) {
  const canvas = root.querySelector<HTMLElement>('[data-atlas-canvas]');
  const sheet = root.querySelector<HTMLElement>('[data-atlas-sheet]');
  const payload = root.querySelector('[data-atlas-data]')?.textContent;
  if (canvas && sheet && payload) {
    const { places } = JSON.parse(payload) as { places: AtlasPlace[] };
    const byId = new Map(places.map((place) => [place.id, place]));
    const initialSheet = sheet.innerHTML;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    const status = root.querySelector<HTMLElement>('[data-atlas-status]');
    const retry = root.querySelector<HTMLButtonElement>('[data-atlas-retry]');
    const zoomIn = root.querySelector<HTMLButtonElement>('[data-atlas-zoom-in]');
    const zoomOut = root.querySelector<HTMLButtonElement>('[data-atlas-zoom-out]');
    let activeTrip = '';
    let selectedId = '';
    let map: L.Map | undefined;
    let tiles: L.TileLayer | undefined;
    let markers: L.LayerGroup | undefined;
    let failedTiles = 0;
    let renderedMarkers: { marker: L.Marker; ids: string[] }[] = [];

    const visiblePlaces = () => places.filter((place) => !activeTrip || place.trip === activeTrip);
    const setStatus = (message: string, canRetry = false) => {
      if (status) status.textContent = message;
      if (retry) retry.hidden = !canRetry;
    };
    const text = (tag: string, className: string, value: string) => {
      const element = document.createElement(tag);
      element.className = className;
      element.textContent = value;
      return element;
    };
    const picture = (place: AtlasPlace, index: number, sizes: string) => {
      const asset = place.images[index];
      const element = document.createElement('picture');
      if (asset.webpSrcset) {
        const source = document.createElement('source');
        source.type = 'image/webp';
        source.srcset = asset.webpSrcset;
        source.sizes = sizes;
        element.append(source);
      }
      const img = document.createElement('img');
      img.src = asset.src;
      img.alt = place.imageDescs[index] || `${place.name} · 第 ${index + 1} 张照片`;
      img.loading = 'lazy';
      img.decoding = 'async';
      if (asset.width) img.width = asset.width;
      if (asset.height) img.height = asset.height;
      element.append(img);
      return element;
    };
    const updateSelection = () => {
      root.querySelectorAll<HTMLButtonElement>('[data-atlas-place]').forEach((button) => {
        button.setAttribute('aria-pressed', String(button.dataset.atlasPlace === selectedId));
      });
      for (const { marker, ids } of renderedMarkers) {
        marker.getElement()?.querySelector('button')?.setAttribute('aria-pressed', String(ids.includes(selectedId)));
      }
    };
    const showPlace = (place: AtlasPlace, move = true) => {
      selectedId = place.id;
      const heading = document.createElement('div');
      heading.className = 'atlas-sheet__heading';
      heading.append(
        text('p', 'atlas-eyebrow', place.region),
        text('h3', '', place.name),
        text('p', 'atlas-sheet__meta', `${place.en} · ${place.images.length} 张照片`),
      );
      const note = text('p', 'atlas-sheet__note', '按相册归属标记城市／区域，非逐张拍摄坐标。');
      const grid = document.createElement('div');
      grid.className = 'atlas-sheet__photos';
      const meta = JSON.stringify({
        images: place.images,
        imageDescs: place.imageDescs,
        title: `${place.name} · ${place.tripTitle}`,
        description: `${place.region} · 城市／区域级归档，非拍摄 GPS`,
        category: 'photography',
        date: place.date,
        detailUrl: place.detailUrl,
      });
      place.images.forEach((_, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'atlas-photo';
        button.dataset.galleryMeta = meta;
        button.dataset.galleryIndex = String(index);
        button.setAttribute('aria-label', `查看${place.name}第 ${index + 1} 张照片：${place.imageDescs[index] || '未题名'}`);
        button.append(picture(place, index, '(max-width: 640px) 30vw, 140px'));
        const number = text('span', 'atlas-photo__number', String(index + 1).padStart(2, '0'));
        number.setAttribute('aria-hidden', 'true');
        button.append(number);
        grid.append(button);
      });
      const link = document.createElement('a');
      link.className = 'atlas-sheet__link';
      link.href = place.detailUrl;
      link.textContent = `走进「${place.tripTitle}」 →`;
      sheet.replaceChildren(heading, note, grid, link);
      sheet.scrollTop = 0;
      updateSelection();
      if (move && map) {
        map.setView([place.lat, place.lng], Math.max(map.getZoom(), 9), { animate: !reducedMotion.matches });
      }
      const announcement = root.querySelector<HTMLElement>('[data-atlas-announcement]');
      if (announcement) announcement.textContent = `已展开${place.name}的 ${place.images.length} 张照片`;
    };

    const fitPlaces = () => {
      if (!map) return;
      const visible = visiblePlaces();
      if (!visible.length) return;
      map.fitBounds(L.latLngBounds(visible.map((place) => [place.lat, place.lng])), {
        padding: [canvas.clientWidth < 500 ? 40 : 70, 70], maxZoom: 10, animate: !reducedMotion.matches,
      });
    };

    // Cluster in screen space, so close cities open naturally as the map zooms in.
    const renderMarkers = () => {
      if (!map || !markers) return;
      const visible = visiblePlaces();
      const focusedId = (document.activeElement as HTMLElement | null)?.closest<HTMLElement>('[data-atlas-marker]')?.dataset.atlasMarker;
      markers.clearLayers();
      renderedMarkers = [];
      const points = visible.map((place) => map!.latLngToLayerPoint([place.lat, place.lng]));
      const remaining = new Set(visible.map((_, index) => index));
      while (remaining.size) {
        const first = remaining.values().next().value!;
        const group = [first];
        remaining.delete(first);
        const point = points[first];
        for (const index of remaining) {
          if (Math.abs(point.x - points[index].x) < 92 && Math.abs(point.y - points[index].y) < 100) {
            group.push(index);
            remaining.delete(index);
          }
        }
        const members = group.map((index) => visible[index]);
        const cover = members.find((place) => place.id === selectedId) || members[0];
        const count = members.reduce((sum, place) => sum + place.images.length, 0);
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `atlas-pin${members.length > 1 ? ' is-cluster' : ''}`;
        button.dataset.atlasMarker = cover.id;
        button.setAttribute('aria-label', members.length > 1
          ? `放大查看${members.map((place) => place.name).join('、')}，共 ${count} 张照片`
          : `展开${cover.name}，${count} 张照片`);
        button.setAttribute('aria-pressed', String(members.some((place) => place.id === selectedId)));
        const image = document.createElement('span');
        image.className = 'atlas-pin__image';
        image.append(picture(cover, 0, '72px'));
        button.append(image, text('span', 'atlas-pin__count', String(count)), text('span', 'atlas-pin__label', members.length > 1 ? `${members.length} 处足迹` : cover.name));
        const anchor: L.LatLngExpression = [
          members.reduce((sum, place) => sum + place.lat, 0) / members.length,
          members.reduce((sum, place) => sum + place.lng, 0) / members.length,
        ];
        const marker = L.marker(anchor, {
          icon: L.divIcon({ className: 'atlas-marker', html: button, iconSize: [80, 96], iconAnchor: [40, 88] }),
          keyboard: false, riseOnHover: true,
        }).addTo(markers);
        // The nested native button is the single accessible interactive target.
        marker.getElement()?.removeAttribute('role');
        button.addEventListener('click', (event) => {
          event.stopPropagation();
          if (members.length === 1) {
            showPlace(cover);
          } else {
            map!.fitBounds(L.latLngBounds(members.map((place) => [place.lat, place.lng])), {
              padding: [85, 95], maxZoom: 13, animate: !reducedMotion.matches,
            });
          }
          canvas.focus({ preventScroll: true });
        });
        renderedMarkers.push({ marker, ids: members.map((place) => place.id) });
      }
      if (focusedId) {
        const current = renderedMarkers.find(({ ids }) => ids.includes(focusedId));
        current?.marker.getElement()?.querySelector('button')?.focus({ preventScroll: true });
      }
      if (zoomIn) zoomIn.disabled = map.getZoom() >= map.getMaxZoom();
      if (zoomOut) zoomOut.disabled = map.getZoom() <= map.getMinZoom();
    };

    const startMap = () => {
      if (map || !places.length) return;
      map = L.map(canvas, {
        center: [32, 117], zoom: 4, minZoom: 2, maxZoom: 13,
        zoomControl: false, scrollWheelZoom: false,
        attributionControl: true, preferCanvas: true,
        zoomAnimation: !reducedMotion.matches, fadeAnimation: !reducedMotion.matches,
        maxBounds: [[-80, -180], [85, 180]], maxBoundsViscosity: 1,
      });
      map.attributionControl.setPrefix(false);
      markers = L.layerGroup().addTo(map);
      tiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19, noWrap: true,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a>',
      });
      tiles.on('loading', () => {
        failedTiles = 0;
        setStatus('正在展开地图…');
      });
      tiles.on('tileerror', () => {
        failedTiles += 1;
        setStatus('底图暂时不可用，仍可从地点索引浏览照片。', true);
      });
      tiles.on('load', () => {
        setStatus(failedTiles ? '部分底图未能加载，地点和照片仍可浏览。' : '拖动探索 · 双指缩放 · 点击相纸看照片', failedTiles > 0);
      });
      tiles.addTo(map);
      map.on('zoomend', renderMarkers);
      fitPlaces();
      renderMarkers();
      if (selectedId) {
        const selected = byId.get(selectedId);
        if (selected) map.setView([selected.lat, selected.lng], 9, { animate: false });
      }
      const resize = new ResizeObserver(() => map?.invalidateSize({ pan: false }));
      resize.observe(canvas);
      window.addEventListener('pagehide', () => resize.disconnect(), { once: true });
      root.dataset.atlasReady = 'true';
    };

    const filter = (trip: string) => {
      activeTrip = trip;
      selectedId = '';
      sheet.innerHTML = initialSheet;
      root.querySelectorAll<HTMLButtonElement>('[data-atlas-filter]').forEach((button) => {
        button.setAttribute('aria-pressed', String(button.dataset.atlasFilter === trip));
      });
      root.querySelectorAll<HTMLButtonElement>('[data-atlas-place]').forEach((button) => {
        const place = byId.get(button.dataset.atlasPlace || '');
        button.hidden = !!trip && place?.trip !== trip;
      });
      const visible = visiblePlaces();
      const count = root.querySelector<HTMLElement>('[data-atlas-count]');
      if (count) count.textContent = `${visible.length} 处足迹 · ${visible.reduce((sum, place) => sum + place.images.length, 0)} 张照片`;
      startMap();
      fitPlaces();
      renderMarkers();
      updateSelection();
      if (trip && visible.length) showPlace(visible[0], false);
    };
    const lightbox = document.getElementById('gallery-lightbox');
    let photoTrigger: HTMLButtonElement | null = null;
    if (lightbox) {
      new MutationObserver(() => {
        if (!photoTrigger) return;
        if (lightbox.hidden) {
          photoTrigger.focus({ preventScroll: true });
          photoTrigger = null;
        } else {
          lightbox.querySelector<HTMLButtonElement>('.glbx__close')?.focus({ preventScroll: true });
        }
      }).observe(lightbox, { attributes: true, attributeFilter: ['hidden'] });
      lightbox.addEventListener('keydown', (event) => {
        if (!photoTrigger || event.key !== 'Tab') return;
        const controls = Array.from(lightbox.querySelectorAll<HTMLElement>('button, a[href]'))
          .filter((element) => element.getClientRects().length && getComputedStyle(element).visibility !== 'hidden');
        const first = controls[0];
        const last = controls[controls.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      });
    }
    root.addEventListener('click', (event) => {
      const target = event.target as HTMLElement;
      const photo = target.closest<HTMLButtonElement>('.atlas-photo');
      if (photo) photoTrigger = photo;
      const tripButton = target.closest<HTMLElement>('[data-atlas-filter]');
      if (tripButton) filter(tripButton.dataset.atlasFilter || '');
      const placeButton = target.closest<HTMLElement>('[data-atlas-place]');
      if (placeButton) {
        const place = byId.get(placeButton.dataset.atlasPlace || '');
        if (place) { startMap(); showPlace(place); }
      }
      if (target.closest('[data-atlas-reset]')) filter('');
      if (target.closest('[data-atlas-zoom-in]')) { startMap(); map?.zoomIn(); }
      if (target.closest('[data-atlas-zoom-out]')) { startMap(); map?.zoomOut(); }
      if (target.closest('[data-atlas-retry]')) tiles?.redraw();
    });
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver((entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        startMap();
        observer.disconnect();
      }, { rootMargin: '240px' });
      observer.observe(canvas);
    } else {
      startMap();
    }
  }
}
