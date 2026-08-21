/**
 * 首页 Hero 的萤火虫氛围：一小群暖光在林间飘，
 * 鼠标划过时像气流一样把它们轻轻推开。
 * 仅在首屏可见时运行；prefers-reduced-motion 下不启动。
 */

interface Firefly {
  x: number;
  y: number;
  heading: number;
  speed: number;
  size: number;
  baseAlpha: number;
  phaseA: number;
  phaseB: number;
  freqA: number;
  freqB: number;
  pushX: number;
  pushY: number;
}

const WIND_RADIUS = 170;
const WIND_STRENGTH = 0.16;

export function initFireflies(): void {
  const canvas = document.querySelector<HTMLCanvasElement>('.home-fireflies');
  const hero = canvas?.closest<HTMLElement>('.home-hero');
  if (!canvas || !hero) return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    canvas.remove();
    return;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  let width = 0;
  let height = 0;
  let running = false;
  let raf = 0;
  let last = 0;
  let elapsed = 0;

  const resize = () => {
    const rect = hero.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = rect.width;
    height = rect.height;
    canvas.width = Math.max(1, Math.round(width * dpr));
    canvas.height = Math.max(1, Math.round(height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  resize();
  window.addEventListener('resize', resize);

  /* 预渲染一枚柔光贴图，避免每帧画径向渐变 */
  const sprite = document.createElement('canvas');
  sprite.width = 64;
  sprite.height = 64;
  const sctx = sprite.getContext('2d');
  if (sctx) {
    const grad = sctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, 'rgba(255, 226, 168, 0.9)');
    grad.addColorStop(0.25, 'rgba(228, 178, 106, 0.5)');
    grad.addColorStop(0.6, 'rgba(196, 138, 61, 0.14)');
    grad.addColorStop(1, 'rgba(196, 138, 61, 0)');
    sctx.fillStyle = grad;
    sctx.fillRect(0, 0, 64, 64);
  }

  const count = () => (width < 720 ? 9 : 15);
  let flies: Firefly[] = [];

  const spawn = (): Firefly => ({
    x: Math.random() * width,
    y: height * 0.3 + Math.random() * height * 0.65,
    heading: Math.random() * Math.PI * 2,
    speed: 7 + Math.random() * 16,
    size: 7 + Math.random() * 12,
    baseAlpha: 0.28 + Math.random() * 0.4,
    phaseA: Math.random() * Math.PI * 2,
    phaseB: Math.random() * Math.PI * 2,
    freqA: 0.4 + Math.random() * 0.5,
    freqB: 0.13 + Math.random() * 0.2,
    pushX: 0,
    pushY: 0,
  });

  const seed = () => {
    flies = Array.from({ length: count() }, spawn);
  };
  seed();

  /* 鼠标气流：记录指针在 hero 内的移动速度 */
  let pointerX = -9999;
  let pointerY = -9999;
  let windX = 0;
  let windY = 0;
  let lastPointerX = -9999;
  let lastPointerY = -9999;

  hero.addEventListener('pointermove', (event) => {
    const rect = hero.getBoundingClientRect();
    pointerX = event.clientX - rect.left;
    pointerY = event.clientY - rect.top;
  });
  hero.addEventListener('pointerleave', () => {
    pointerX = -9999;
    pointerY = -9999;
    windX = 0;
    windY = 0;
  });

  const frame = (now: number) => {
    if (!running) return;
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    elapsed += dt;

    if (lastPointerX > -9000) {
      windX = windX * 0.8 + (pointerX - lastPointerX) * 0.2;
      windY = windY * 0.8 + (pointerY - lastPointerY) * 0.2;
    }
    lastPointerX = pointerX;
    lastPointerY = pointerY;

    ctx.clearRect(0, 0, width, height);
    ctx.globalCompositeOperation = 'lighter';

    for (const fly of flies) {
      /* 蜿蜒游走 */
      fly.heading +=
        Math.sin(elapsed * fly.freqA + fly.phaseA) * 1.4 * dt +
        Math.cos(elapsed * fly.freqB + fly.phaseB) * 0.9 * dt;

      let vx = Math.cos(fly.heading) * fly.speed;
      let vy = Math.sin(fly.heading) * fly.speed * 0.6;

      /* 气流推力，随距离衰减 */
      const dx = fly.x - pointerX;
      const dy = fly.y - pointerY;
      const distSq = dx * dx + dy * dy;
      if (distSq < WIND_RADIUS * WIND_RADIUS) {
        const falloff = 1 - Math.sqrt(distSq) / WIND_RADIUS;
        vx += windX * WIND_STRENGTH * falloff * 60 * dt * 10;
        vy += windY * WIND_STRENGTH * falloff * 60 * dt * 10;
      }

      fly.pushX = (fly.pushX + vx * dt) * 0.96 + vx * dt * 0.04;
      fly.pushY = (fly.pushY + vy * dt) * 0.96 + vy * dt * 0.04;
      fly.x += fly.pushX;
      fly.y += fly.pushY;

      /* 越界环绕 */
      const margin = 30;
      if (fly.x < -margin) fly.x = width + margin;
      if (fly.x > width + margin) fly.x = -margin;
      if (fly.y < height * 0.12) {
        fly.y = height * 0.12;
        fly.heading = -fly.heading;
      }
      if (fly.y > height + margin) fly.y = -margin;

      const twinkle = 0.72 + Math.sin(elapsed * (1.1 + fly.freqA) + fly.phaseB) * 0.28;
      const alpha = fly.baseAlpha * twinkle;
      const drawSize = fly.size * (0.92 + twinkle * 0.16);

      ctx.globalAlpha = alpha;
      ctx.drawImage(sprite, fly.x - drawSize / 2, fly.y - drawSize / 2, drawSize, drawSize);

      /* 亮芯 */
      ctx.globalAlpha = alpha * 0.85;
      ctx.fillStyle = 'rgba(255, 240, 205, 0.95)';
      ctx.beginPath();
      ctx.arc(fly.x, fly.y, 1.1, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.globalAlpha = 1;
    raf = requestAnimationFrame(frame);
  };

  const start = () => {
    if (running || document.hidden) return;
    running = true;
    last = performance.now();
    raf = requestAnimationFrame(frame);
  };

  const stop = () => {
    running = false;
    cancelAnimationFrame(raf);
    ctx.clearRect(0, 0, width, height);
  };

  new IntersectionObserver(([entry]) => {
    if (entry.isIntersecting) start();
    else stop();
  }).observe(hero);

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stop();
    else start();
  });

  window.addEventListener('resize', () => {
    if (flies.length !== count()) seed();
  });
}
