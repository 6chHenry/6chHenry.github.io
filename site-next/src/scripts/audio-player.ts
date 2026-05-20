const ICON_PLAY =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>';
const ICON_PAUSE =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h4v14H6zm8 0h4v14h-4z"/></svg>';

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s < 10 ? '0' : ''}${s}`;
}

function prefersReducedMotion(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function resolveAssetPath(src: string): string {
  if (src.startsWith('http://') || src.startsWith('https://')) return src;
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  if (src.startsWith('/')) return `${base}${src}`;
  return `${base}/${src}`.replace(/\/{2,}/g, '/');
}

function buildPlayer(root: HTMLElement) {
  const rawSrc = root.getAttribute('data-src');
  if (!rawSrc) return;
  const src = resolveAssetPath(rawSrc);

  const title = root.getAttribute('data-title') || '录音';
  const label = root.getAttribute('data-label') || 'AUDIO LOG';
  const reduced = prefersReducedMotion();

  root.classList.add('ch-audio--ready');
  root.innerHTML = `
    <div class="ch-audio__shell">
      <div class="ch-audio__fx" aria-hidden="true">
        <div class="ch-audio__grid"></div>
        <div class="ch-audio__glow"></div>
        <div class="ch-audio__scan"></div>
      </div>
      <div class="ch-audio__top">
        <span class="ch-audio__badge">${label}</span>
        <h3 class="ch-audio__title">${title}</h3>
        <div class="ch-audio__viz" aria-hidden="true"></div>
      </div>
      <div class="ch-audio__controls">
        <button type="button" class="ch-audio__btn ch-audio__btn--play" aria-label="播放">${ICON_PLAY}</button>
        <button type="button" class="ch-audio__btn ch-audio__btn--skip" data-skip="-15" aria-label="后退 15 秒">−15</button>
        <div class="ch-audio__track-wrap">
          <input type="range" class="ch-audio__seek" min="0" max="1000" value="0" step="1" aria-label="播放进度" />
          <div class="ch-audio__track-fill"></div>
        </div>
        <button type="button" class="ch-audio__btn ch-audio__btn--skip" data-skip="15" aria-label="前进 15 秒">+15</button>
        <span class="ch-audio__time"><span class="ch-audio__cur">0:00</span> / <span class="ch-audio__dur">0:00</span></span>
      </div>
      <audio class="ch-audio__el" preload="metadata" src="${src}"></audio>
    </div>`;

  const shell = root.querySelector<HTMLElement>('.ch-audio__shell');
  const audio = root.querySelector<HTMLAudioElement>('.ch-audio__el');
  const playBtn = root.querySelector<HTMLButtonElement>('.ch-audio__btn--play');
  const seek = root.querySelector<HTMLInputElement>('.ch-audio__seek');
  const fill = root.querySelector<HTMLElement>('.ch-audio__track-fill');
  const curEl = root.querySelector<HTMLElement>('.ch-audio__cur');
  const durEl = root.querySelector<HTMLElement>('.ch-audio__dur');
  const viz = root.querySelector<HTMLElement>('.ch-audio__viz');
  const skipBtns = root.querySelectorAll<HTMLButtonElement>('[data-skip]');

  if (!shell || !audio || !playBtn || !seek || !fill || !curEl || !durEl || !viz) return;

  const bars: HTMLElement[] = [];
  const barCount = reduced ? 0 : 28;
  let rafId: number | null = null;
  let ticker = 0;
  let seeking = false;

  for (let i = 0; i < barCount; i += 1) {
    const bar = document.createElement('span');
    bar.className = 'ch-audio__bar';
    viz.appendChild(bar);
    bars.push(bar);
  }

  const getDuration = () => (Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : 0);

  const setProgressFromTime = (time: number) => {
    const dur = getDuration();
    const pct = dur > 0 ? (time / dur) * 100 : 0;
    seek.value = String(Math.round(pct * 10));
    fill.style.width = `${pct}%`;
    curEl.textContent = formatTime(time);
    durEl.textContent = dur > 0 ? formatTime(dur) : '0:00';
  };

  const updateProgress = () => {
    if (seeking) return;
    setProgressFromTime(audio.currentTime);
  };

  const seekToRatio = (ratio: number) => {
    const dur = getDuration();
    if (dur <= 0) return;
    const clamped = Math.max(0, Math.min(1, ratio));
    const target = clamped * dur;
    audio.currentTime = target;
    setProgressFromTime(target);
  };

  const setPlayingUI = (playing: boolean) => {
    root.classList.toggle('is-playing', playing);
    playBtn.innerHTML = playing ? ICON_PAUSE : ICON_PLAY;
    playBtn.setAttribute('aria-label', playing ? '暂停' : '播放');
  };

  const stopVisualizer = () => {
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    bars.forEach((bar) => bar.style.setProperty('--h', '0.15'));
  };

  const tickVisualizer = () => {
    if (audio.paused || audio.ended) {
      stopVisualizer();
      return;
    }
    ticker += 1;
    const phase = audio.currentTime * 7 + ticker * 0.09;
    for (let i = 0; i < barCount; i += 1) {
      const sinA = Math.sin(phase + i * 0.42);
      const sinB = Math.sin(phase * 0.63 + i * 0.78);
      const v = (sinA * 0.55 + sinB * 0.45 + 1) / 2;
      bars[i].style.setProperty('--h', (0.18 + v * 0.82).toFixed(3));
    }
    rafId = requestAnimationFrame(tickVisualizer);
  };

  const pauseOthers = () => {
    document.querySelectorAll<HTMLElement>('[data-ch-audio].is-playing').forEach((other) => {
      if (other === root) return;
      const otherAudio = other.querySelector<HTMLAudioElement>('.ch-audio__el');
      const otherBtn = other.querySelector<HTMLButtonElement>('.ch-audio__btn--play');
      if (otherAudio && !otherAudio.paused) {
        otherAudio.pause();
        other.classList.remove('is-playing');
        if (otherBtn) {
          otherBtn.innerHTML = ICON_PLAY;
          otherBtn.setAttribute('aria-label', '播放');
        }
      }
    });
  };

  playBtn.addEventListener('click', () => {
    if (audio.paused) {
      pauseOthers();
      audio.play().catch(() => shell.classList.add('ch-audio__shell--error'));
    } else {
      audio.pause();
    }
  });

  skipBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const delta = parseFloat(btn.getAttribute('data-skip') || '0');
      const dur = getDuration();
      let next = audio.currentTime + delta;
      next = dur > 0 ? Math.max(0, Math.min(dur, next)) : Math.max(0, next);
      audio.currentTime = next;
      setProgressFromTime(next);
    });
  });

  seek.addEventListener('pointerdown', () => {
    seeking = true;
  });

  seek.addEventListener('input', () => {
    const pct = parseFloat(seek.value) / 1000;
    fill.style.width = `${pct * 100}%`;
    const dur = getDuration();
    if (dur <= 0) return;
    const target = pct * dur;
    audio.currentTime = target;
    curEl.textContent = formatTime(target);
  });

  seek.addEventListener('change', () => {
    seeking = false;
    seekToRatio(parseFloat(seek.value) / 1000);
  });

  audio.addEventListener('loadedmetadata', updateProgress);
  audio.addEventListener('durationchange', updateProgress);
  audio.addEventListener('timeupdate', updateProgress);
  audio.addEventListener('ended', () => {
    setPlayingUI(false);
    stopVisualizer();
  });
  audio.addEventListener('play', () => {
    setPlayingUI(true);
    if (!reduced && barCount > 0) tickVisualizer();
  });
  audio.addEventListener('pause', () => {
    setPlayingUI(false);
    stopVisualizer();
  });
  audio.addEventListener('error', () => {
    shell.classList.add('ch-audio__shell--error');
    durEl.textContent = '无法加载';
  });
}

export function initAudioPlayers() {
  document.querySelectorAll<HTMLElement>('[data-ch-audio]').forEach(buildPlayer);
}
