/**
 * 林间小精灵「苔苔」——一只住在站里的苔藓光茧精。
 * 可拖拽（位置持久化）、点击聊天（情境感知）、双击送回角落。
 * 纯原生实现，无依赖；尊重 prefers-reduced-motion。
 */

const POS_KEY = 'forest-sprite:v1';
const TALK_MS = 4200;
const DOUBLE_TAP_MS = 350;
const DRAG_THRESHOLD = 6;

interface SpritePos {
  x: number;
  y: number;
}

type LinePool = Record<string, string[]>;

const LINES: LinePool = {
  general: [
    '我是住在站里的苔藓精，叫我苔苔就好。',
    '有什么想找的？Ctrl + K 可以唤出搜索。',
    '双击可以把我送回角落，我有点恋家。',
    '这片林子里的每棵树都是一篇文章哦。',
    '拖着我到处走走也行，我抓得很稳的。',
  ],
  home: [
    '欢迎回到林子里～',
    '今天想走哪条路？',
    '风把最新的痕迹吹到首页了。',
    '林口的金色小径又更新啦。',
  ],
  notes: [
    '笔记林里全是知识的年轮。',
    '慢慢看，树不会跑的。',
    '这一片林的根系长得越来越深了。',
  ],
  essay: [
    '杂谈林保存着季节感，适合慢慢逛。',
    '去别处看看吧，就现在。',
    '这里的风里都是故事的味道。',
  ],
  projects: [
    '这些都是长出来的枝条呀。',
    '点一棵树看看结了什么果子？',
    '做东西的手，是不会骗人的。',
  ],
  gallery: [
    '照片是时间的标本。',
    '这一带的风景不错吧？',
    '快门按下去的那一刻就不一样了。',
  ],
  about: [
    '这就是种林子的人啦。',
    '嘘，他正在找新的问题。',
  ],
  search: [
    '找什么？我帮你闻闻味儿。',
    '林子虽大，一句话就能定位。',
  ],
  tags: ['顺着标签走，也是一种路标。'],
  dawn: ['早啊，林子刚醒。', '晨雾还没散呢。'],
  day: ['阳光正好，适合翻翻笔记。', '今天的林子很安静。'],
  dusk: ['黄昏的林子是金色的。', '天边烧起来了，看一眼？'],
  night: ['夜深了，萤火虫都出来了。', '晚上的林子另一种味道。'],
  lateNight: ['还没睡呀？别熬太久哦。', '星星都困了，你也早点休息。'],
  themeDark: ['天黑了……我把小灯点亮。', '夜里也要记得回来呀。'],
  themeLight: ['天亮啦！伸个懒腰——', '光进来了。'],
  dragFar: ['哇，飞起来了！', '换个地方住也不错。', '轻点儿，苔藓会晕的……', '新视野！记下了记下了。'],
};

const SECTION_PATTERNS: Array<[string, string]> = [
  ['notes', 'notes'],
  ['essay', 'essay'],
  ['projects', 'projects'],
  ['gallery', 'gallery'],
  ['about', 'about'],
  ['search', 'search'],
  ['tags', 'tags'],
];

function sectionOf(pathname: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  let path = pathname;
  if (base && path.startsWith(base)) path = path.slice(base.length);
  path = path.replace(/^\//, '');
  for (const [prefix, section] of SECTION_PATTERNS) {
    if (path.startsWith(prefix)) return section;
  }
  if (path === '' || path === '/') return 'home';
  return 'general';
}

function timePool(): string {
  const hour = new Date().getHours();
  if (hour >= 5 && hour < 8) return 'dawn';
  if (hour >= 8 && hour < 17) return 'day';
  if (hour >= 17 && hour < 23) return 'dusk';
  return hour >= 23 ? 'lateNight' : 'night';
}

export function initForestSprite(): void {
  const root = document.querySelector<HTMLElement>('.forest-sprite');
  if (!root) return;
  const tilt = root.querySelector<HTMLElement>('.forest-sprite__tilt');
  const bubble = root.querySelector<HTMLElement>('.forest-sprite__bubble');
  const button = root.querySelector<HTMLButtonElement>('.forest-sprite__body');
  if (!tilt || !bubble || !button) return;

  const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const size = () => root.getBoundingClientRect();
  const bounds = () => ({
    minX: 10,
    maxX: window.innerWidth - size().width - 10,
    minY: 74,
    maxY: window.innerHeight - size().height - 10,
  });

  const clamp = (p: SpritePos): SpritePos => {
    const b = bounds();
    return {
      x: Math.min(Math.max(p.x, b.minX), Math.max(b.minX, b.maxX)),
      y: Math.min(Math.max(p.y, b.minY), Math.max(b.minY, b.maxY)),
    };
  };

  const homePos = (): SpritePos => ({
    x: window.innerWidth - size().width - 26,
    y: window.innerHeight - size().height - 26,
  });

  const loadPos = (): SpritePos => {
    try {
      const raw = localStorage.getItem(POS_KEY);
      if (!raw) return homePos();
      const saved = JSON.parse(raw) as { xPct?: number; yPct?: number };
      if (typeof saved.xPct !== 'number' || typeof saved.yPct !== 'number') return homePos();
      return clamp({ x: saved.xPct * window.innerWidth, y: saved.yPct * window.innerHeight });
    } catch {
      return homePos();
    }
  };

  let pos = loadPos();
  let target: SpritePos = { ...pos };
  let rotation = 0;
  let raf = 0;
  let animating = false;
  let dragging = false;
  let grabOffset: SpritePos = { x: 0, y: 0 };
  let pointerStart: SpritePos = { x: 0, y: 0 };
  let moved = false;
  let lastTapAt = 0;
  let hideTimer = 0;
  let talkFxTimer = 0;
  let skitTimer = 0;
  let skitEndTimer = 0;
  let lastLine = '';

  const apply = () => {
    root.style.transform = `translate(${pos.x}px, ${pos.y}px)`;
    tilt.style.transform = rotation === 0 ? '' : `rotate(${rotation.toFixed(2)}deg)`;
    root.classList.toggle('is-edge-left', pos.x + size().width / 2 < window.innerWidth / 2);
  };

  const tick = () => {
    const dx = target.x - pos.x;
    const dy = target.y - pos.y;
    const dr = -dx * 0.08;
    pos.x += dx * 0.28;
    pos.y += dy * 0.28;
    rotation += (dr - rotation) * 0.2;
    if (Math.abs(rotation) > 12) rotation = Math.sign(rotation) * 12;
    apply();
    if (!dragging && Math.hypot(dx, dy) < 0.5 && Math.abs(rotation) < 0.2) {
      pos = { ...target };
      rotation = 0;
      apply();
      animating = false;
      return;
    }
    raf = requestAnimationFrame(tick);
  };

  const wake = () => {
    if (!animating) {
      animating = true;
      raf = requestAnimationFrame(tick);
    }
  };

  const moveTo = (p: SpritePos, instant = false) => {
    target = clamp(p);
    if (instant || reducedMotion()) {
      pos = { ...target };
      rotation = 0;
      apply();
      return;
    }
    wake();
  };

  /* ── 对话 ── */

  const pick = (poolName: string): string => {
    const pool = LINES[poolName];
    if (!pool || pool.length === 0) return LINES.general[0];
    const candidates = pool.length > 1 ? pool.filter((line) => line !== lastLine) : pool;
    const line = candidates[Math.floor(Math.random() * candidates.length)];
    lastLine = line;
    return line;
  };

  const say = (line: string) => {
    bubble.textContent = line;
    bubble.classList.add('is-visible');
    window.clearTimeout(hideTimer);
    hideTimer = window.setTimeout(() => bubble.classList.remove('is-visible'), TALK_MS);
  };

  const talk = () => {
    const roll = Math.random();
    if (roll < 0.55) {
      const section = sectionOf(window.location.pathname);
      say(pick(LINES[section] ? section : 'general'));
    } else if (roll < 0.85) {
      say(pick(timePool()));
    } else {
      say(pick('general'));
    }
    root.classList.add('is-talking');
    window.clearTimeout(talkFxTimer);
    talkFxTimer = window.setTimeout(() => root.classList.remove('is-talking'), 640);
  };

  /* ── 拖拽 ── */

  const onPointerDown = (event: PointerEvent) => {
    if (event.button !== 0) return;
    dragging = true;
    moved = false;
    pointerStart = { x: event.clientX, y: event.clientY };
    grabOffset = { x: event.clientX - pos.x, y: event.clientY - pos.y };
    button.setPointerCapture(event.pointerId);
    root.classList.add('is-dragging');
    bubble.classList.remove('is-visible');
  };

  const onPointerMove = (event: PointerEvent) => {
    if (!dragging) return;
    if (!moved && Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > DRAG_THRESHOLD) {
      moved = true;
    }
    moveTo({ x: event.clientX - grabOffset.x, y: event.clientY - grabOffset.y });
  };

  const onPointerUp = (event: PointerEvent) => {
    if (!dragging) return;
    dragging = false;
    button.releasePointerCapture?.(event.pointerId);
    root.classList.remove('is-dragging');

    if (moved) {
      try {
        localStorage.setItem(
          POS_KEY,
          JSON.stringify({
            xPct: pos.x / Math.max(1, window.innerWidth),
            yPct: pos.y / Math.max(1, window.innerHeight),
          }),
        );
      } catch {
        /* 隐私模式下静默失败 */
      }
      const travelled = Math.hypot(pos.x - pointerStart.x, pos.y - pointerStart.y);
      if (travelled > 140 && Math.random() < 0.45) say(pick('dragFar'));
      return;
    }

    const now = performance.now();
    if (now - lastTapAt < DOUBLE_TAP_MS) {
      lastTapAt = 0;
      goHome();
    } else {
      lastTapAt = now;
      talk();
    }
  };

  const goHome = () => {
    try {
      localStorage.removeItem(POS_KEY);
    } catch {
      /* 同上 */
    }
    moveTo(homePos());
    window.setTimeout(() => say(pick('home')), reducedMotion() ? 0 : 480);
  };

  button.addEventListener('pointerdown', onPointerDown);
  button.addEventListener('pointermove', onPointerMove);
  button.addEventListener('pointerup', onPointerUp);
  button.addEventListener('pointercancel', onPointerUp);

  button.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      talk();
    }
  });

  bubble.addEventListener('click', () => bubble.classList.remove('is-visible'));

  /* ── 主题切换反应 ── */

  let lastTheme = document.documentElement.dataset.theme ?? '';
  new MutationObserver(() => {
    const theme = document.documentElement.dataset.theme ?? '';
    if (theme === lastTheme) return;
    lastTheme = theme;
    say(pick(theme === 'dark' ? 'themeDark' : 'themeLight'));
    root.classList.add('is-happy');
    window.setTimeout(() => root.classList.remove('is-happy'), 900);
  }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  window.addEventListener('resize', () => moveTo(pos));

  /* ── 目光跟随与悬停倾身：视线和小身子都朝鼠标偏一点 ── */

  button.addEventListener('pointermove', (event) => {
    if (dragging || reducedMotion()) return;
    const rect = button.getBoundingClientRect();
    const dx = event.clientX - (rect.left + rect.width / 2);
    const dy = event.clientY - (rect.top + rect.height / 2);
    const dist = Math.hypot(dx, dy) || 1;
    const reach = Math.min(1, dist / 120);
    root.style.setProperty('--eye-x', `${((dx / dist) * 2 * reach).toFixed(2)}px`);
    root.style.setProperty('--eye-y', `${((dy / dist) * 1.6 * reach).toFixed(2)}px`);
    root.style.setProperty('--lean', `${Math.max(-7, Math.min(7, dx * 0.06)).toFixed(2)}deg`);
  });

  button.addEventListener('pointerleave', () => {
    root.style.removeProperty('--eye-x');
    root.style.removeProperty('--eye-y');
    root.style.removeProperty('--lean');
  });

  /* ── 闲置小剧场：东张西望 / 扭一扭 / 打瞌睡 / 蹦一下 ── */

  const IDLE_SKITS: Array<[string, number]> = [
    ['look', 1700],
    ['wiggle', 1000],
    ['doze', 2200],
    ['hop', 800],
  ];
  let lastSkit = '';

  const scheduleSkit = () => {
    window.clearTimeout(skitTimer);
    skitTimer = window.setTimeout(playSkit, 9000 + Math.random() * 9000);
  };

  const playSkit = () => {
    const busy =
      dragging ||
      document.hidden ||
      reducedMotion() ||
      root.classList.contains('is-talking') ||
      root.classList.contains('is-happy');
    if (busy) {
      scheduleSkit();
      return;
    }
    const pool = IDLE_SKITS.filter(([name]) => name !== lastSkit);
    const [name, duration] = pool[Math.floor(Math.random() * pool.length)];
    lastSkit = name;
    root.classList.add(`is-${name}`);
    window.clearTimeout(skitEndTimer);
    skitEndTimer = window.setTimeout(() => {
      root.classList.remove(`is-${name}`);
      scheduleSkit();
    }, duration);
  };

  scheduleSkit();

  apply();
}
