/**
 * 林间小精灵「苔苔」——一只住在站里的苔藓光茧精。
 * 可拖拽（位置持久化）、点击聊天（情境感知）、双击送回角落。
 * 纯原生实现，无依赖；尊重 prefers-reduced-motion。
 */

import { daypartOf } from '../utils/daypart';

const POS_KEY = 'forest-sprite:v1';
const SPOKEN_KEY = 'forest-sprite:spoken:v1';
const GREETED_KEY = 'forest-sprite:greeted:v1';
const MET_KEY = 'forest-sprite:met:v1';
const TALK_MS = 4200;
const GREET_DELAY_MS = 1600;
const SECTION_DELAY_MS = 1200;
const HOVER_TALK_MS = 1400;
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
    '有什么想找的？Ctrl + K 能唤出搜索。',
    '双击可以把我送回角落，我有点恋家。',
    '拖着我到处走走也行，我抓得很稳的。',
    '不知道看什么，随便点一篇也行。',
    '不催你，慢慢来。',
    '又见面了，还是老样子。',
  ],
  home: [
    '欢迎回来～',
    '今天想往哪走？',
    '首页有新的。',
  ],
  notes: [
    '慢慢看，又不会跑。',
    '这些是写给自己以后看的。',
    '看不懂的地方折个角，回头再来。',
  ],
  essay: [
    '这儿适合慢慢逛。',
    '有些话是写给自己听的，也顺便给你看。',
    '对上了就多停一会儿。',
  ],
  projects: [
    '这些是做出来的。',
    '点开一个看看？',
    '做得歪一点也没关系。',
    '中间全是试错。',
    '有的还在做，别急着下结论。',
  ],
  gallery: [
    '这几张还行吧？',
    '有些地方，去过一次就长进身体里了。',
  ],
  about: [
    '这就是他。',
    '嘘，他正在找新的问题。',
    '这里写的是他，也是他在意的东西。',
  ],
  search: [
    '找什么？我帮你闻闻味儿。',
    '给个词就行。',
    '词越具体越好找。',
    '搜不到也别灰心，可能还没写。',
  ],
  tags: ['顺着标签走也行。', '点一个词，看它牵出多少事情。'],
  academy: ['学院那页写得比较板正。', '这里记着他念书的事。', '简历和兴趣都摊在这儿了，随便看。'],
  dawn: ['早啊。', '晨雾还没散呢。', '这个点来的人不多，安静得刚刚好。'],
  day: ['阳光正好，适合翻翻笔记。', '今天很安静。'],
  dusk: ['黄昏了。', '天边烧起来了，看一眼？', '这会儿的影子拉得最长。'],
  night: ['夜深了，萤火虫都出来了。', '黑下来之后，字反而更亮一点。'],
  lateNight: [
    '还没睡呀？别熬太久哦。',
    '星星都困了，你也早点休息。',
    '这个点还亮着屏幕的，就剩你我。',
    '再读一篇就去睡，好不好？',
  ],
  themeDark: ['天黑了……我把小灯点亮。', '夜里也要记得回来呀。'],
  themeLight: ['天亮啦！伸个懒腰——', '光进来了。', '有点晃眼……等我适应一下。'],
  dragFar: ['哇，飞起来了！', '换个地方住也不错。', '轻点儿，苔藓会晕的……', '这么远啊，那我坐着歇会儿。'],
  firstVisit: [
    '第一次见吧？我叫苔苔，住在这儿。',
    '欢迎，随便走，不催你。',
  ],
  returnVisit: ['又见面啦。', '你上次停在哪一页？', '老地方，我一直在这儿。'],
  hover: ['痒。', '要说话就点我一下。', '别一直盯着看，我会不好意思。', '拽我也行，我不咬人的。', '有事？没事我就继续站着。'],
  pageEnd: ['到底啦。', '下面没有了。', '这页看完了。', '要不要回首页？'],
  lost: ['这页没有，跟我回首页吧。', '走错了？不丢人，我也常走岔。', '这里什么都没有。'],
};

const SECTION_PATTERNS: Array<[string, string]> = [
  ['notes', 'notes'],
  ['essay', 'essay'],
  ['projects', 'projects'],
  ['gallery', 'gallery'],
  ['academy', 'academy'],
  ['about', 'about'],
  ['search', 'search'],
  ['tags', 'tags'],
];

/* 去掉部署 base 与前导斜杠，得到站内相对路径 */
function normalizePath(pathname: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  let path = pathname;
  if (base && path.startsWith(base)) path = path.slice(base.length);
  return path.replace(/^\/+/, '');
}

function sectionOf(pathname: string): string {
  const path = normalizePath(pathname);
  for (const [prefix, section] of SECTION_PATTERNS) {
    if (path.startsWith(prefix)) return section;
  }
  if (path === '') return 'home';
  return 'general';
}

/* 功能区的落地页（/notes/、/essay/ 这类），文章内页不算 */
function sectionIndex(pathname: string): string | null {
  const section = sectionOf(pathname);
  if (section === 'home' || !LINES[section]) return null;
  const path = normalizePath(pathname);
  return path === `${section}/` || path === section ? section : null;
}

function timePool(): string {
  const hour = new Date().getHours();
  const part = daypartOf(hour);
  /* 过了 23 点算深夜，苔苔的话说得更轻一些 */
  return part === 'night' && hour >= 23 ? 'lateNight' : part;
}

export function initForestSprite(): void {
  const root = document.querySelector<HTMLElement>('.forest-sprite');
  if (!root) return;
  const tilt = root.querySelector<HTMLElement>('.forest-sprite__tilt');
  const bubble = root.querySelector<HTMLElement>('.forest-sprite__bubble');
  const bubbleText = root.querySelector<HTMLElement>('.forest-sprite__bubble-text');
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
  let swapTimer = 0;
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

  /* 说过的话记在会话里：翻页也不太会听到重复的一句，
     一个池子讲完了就翻篇，从头再轮一轮。 */
  const spoken = loadSpoken();

  function loadSpoken(): Record<string, number[]> {
    try {
      const raw = sessionStorage.getItem(SPOKEN_KEY);
      const parsed = raw ? (JSON.parse(raw) as unknown) : null;
      return parsed && typeof parsed === 'object' ? (parsed as Record<string, number[]>) : {};
    } catch {
      return {};
    }
  }

  const rememberSpoken = () => {
    try {
      sessionStorage.setItem(SPOKEN_KEY, JSON.stringify(spoken));
    } catch {
      /* 隐私模式下静默失败，去重退化为本页内存 */
    }
  };

  const pick = (poolName: string): string => {
    const key = LINES[poolName] ? poolName : 'general';
    const pool = LINES[key];
    let fresh = pool
      .map((_, index) => index)
      .filter((index) => !(spoken[key] ?? []).includes(index) && pool[index] !== lastLine);
    if (fresh.length === 0) {
      spoken[key] = [];
      fresh = pool.map((_, index) => index).filter((index) => pool[index] !== lastLine);
      if (fresh.length === 0) fresh = pool.map((_, index) => index);
    }
    const index = fresh[Math.floor(Math.random() * fresh.length)];
    spoken[key] = [...(spoken[key] ?? []), index];
    rememberSpoken();
    lastLine = pool[index];
    return lastLine;
  };

  const say = (line: string) => {
    /* 气泡正开着又换了一句：给新台词一次淡入，避免文字硬切 */
    const target = bubbleText ?? bubble;
    if (bubbleText && target.textContent !== line && bubble.classList.contains('is-visible')) {
      bubble.classList.remove('is-swapping');
      void bubble.offsetWidth;
      bubble.classList.add('is-swapping');
      window.clearTimeout(swapTimer);
      swapTimer = window.setTimeout(() => bubble.classList.remove('is-swapping'), 380);
    }
    target.textContent = line;
    bubble.classList.add('is-visible');
    window.clearTimeout(hideTimer);
    hideTimer = window.setTimeout(() => bubble.classList.remove('is-visible'), TALK_MS);
  };

  /* 说一句话并抖一下身子，主动搭话与点击聊天共用 */
  const speak = (line: string) => {
    say(line);
    root.classList.add('is-talking');
    window.clearTimeout(talkFxTimer);
    talkFxTimer = window.setTimeout(() => root.classList.remove('is-talking'), 640);
  };

  /* 迷失页没有自己的 section，靠场景认领 */
  const currentSection = (): string => {
    if (document.querySelector('.not-found')) return 'lost';
    const section = sectionOf(window.location.pathname);
    return LINES[section] ? section : 'general';
  };

  const talk = () => {
    const roll = Math.random();
    if (roll < 0.55) speak(pick(currentSection()));
    else if (roll < 0.85) speak(pick(timePool()));
    else speak(pick('general'));
  };

  /* ── 主动开口的时机 ── */

  /* 本次会话的第一面：先让落定动画走完，再打个招呼。
     功能区落地页不在这儿开口，交给 enterSection 说本区的话。 */
  const greet = () => {
    try {
      if (sessionStorage.getItem(GREETED_KEY)) return;
      sessionStorage.setItem(GREETED_KEY, '1');
    } catch {
      /* 隐私模式下每次都会打招呼，无伤大雅 */
    }
    if (sectionIndex(window.location.pathname)) return;
    let known = true;
    try {
      known = localStorage.getItem(MET_KEY) === '1';
      localStorage.setItem(MET_KEY, '1');
    } catch {
      /* 同上 */
    }
    window.setTimeout(() => {
      if (bubble.classList.contains('is-visible')) return;
      speak(pick(document.querySelector('.not-found') ? 'lost' : known ? 'returnVisit' : 'firstVisit'));
    }, GREET_DELAY_MS);
  };

  /* 走进某个功能区（笔记/杂谈/项目/画廊/学术/标签/关于/搜索）时，先招呼一句这个区的事 */
  const enterSection = () => {
    const section = sectionIndex(window.location.pathname);
    if (!section) return;
    window.setTimeout(() => {
      if (bubble.classList.contains('is-visible')) return;
      speak(pick(section));
    }, SECTION_DELAY_MS);
  };

  /* 鼠标在苔苔身上停一会儿，她会先开口；挪开就算了 */
  let hoverTimer = 0;
  const cancelHoverTalk = () => window.clearTimeout(hoverTimer);

  button.addEventListener('pointerenter', () => {
    if (dragging) return;
    window.clearTimeout(hoverTimer);
    hoverTimer = window.setTimeout(() => {
      if (dragging || bubble.classList.contains('is-visible') || root.classList.contains('is-talking')) return;
      speak(pick('hover'));
    }, HOVER_TALK_MS);
  });

  button.addEventListener('pointerdown', cancelHoverTalk);
  button.addEventListener('pointerleave', cancelHoverTalk);

  /* 滑到页尾才感慨一句，一页只说一次；短页面不凑热闹 */
  let saidPageEnd = false;
  let endCheckFrame = 0;
  window.addEventListener(
    'scroll',
    () => {
      window.cancelAnimationFrame(endCheckFrame);
      endCheckFrame = window.requestAnimationFrame(() => {
        if (saidPageEnd || dragging) return;
        const total = document.documentElement.scrollHeight;
        if (total < window.innerHeight * 1.4) return;
        if (window.scrollY + window.innerHeight < total - 4) return;
        saidPageEnd = true;
        speak(pick('pageEnd'));
      });
    },
    { passive: true },
  );

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
      /* 存 target（松手后的静止点）而非还在插值途中的 pos，避免存下半途坐标 */
      try {
        localStorage.setItem(
          POS_KEY,
          JSON.stringify({
            xPct: target.x / Math.max(1, window.innerWidth),
            yPct: target.y / Math.max(1, window.innerHeight),
          }),
        );
      } catch {
        /* 隐私模式下静默失败 */
      }
      const travelled = Math.hypot(pos.x - pointerStart.x, pos.y - pointerStart.y);
      if (travelled > 140 && Math.random() < 0.45) speak(pick('dragFar'));
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
    window.setTimeout(() => speak(pick('home')), reducedMotion() ? 0 : 480);
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
    speak(pick(theme === 'dark' ? 'themeDark' : 'themeLight'));
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
  greet();
  enterSection();

  apply();
}
