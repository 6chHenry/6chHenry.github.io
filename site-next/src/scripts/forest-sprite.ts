/**
 * 林间小精灵「苔苔」——一只住在站里的苔藓光茧精。
 * 可拖拽（位置持久化）、点击聊天（情境感知）、双击送回角落。
 * 纯原生实现，无依赖；尊重 prefers-reduced-motion。
 */

const POS_KEY = 'forest-sprite:v1';
const SPOKEN_KEY = 'forest-sprite:spoken:v1';
const GREETED_KEY = 'forest-sprite:greeted:v1';
const MET_KEY = 'forest-sprite:met:v1';
const TALK_MS = 4200;
const GREET_DELAY_MS = 1600;
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
    '这片林子里的每棵树，都是一篇文章。',
    '拖着我到处走走也行，我抓得很稳的。',
    '苔藓长得慢，但每天都在长一点点。',
    '我负责看林子，你负责看风景。',
    '风一吹，我就知道有人翻页了。',
    '不知道往哪走的时候，随便点一棵树也行。',
    '林子不催人，慢慢来。',
    '有句话我憋很久了：这里的空气真好。',
    '又见面了，林子里还是老样子。',
  ],
  home: [
    '欢迎回到林子里～',
    '今天想走哪条路？',
    '风把最新的痕迹吹到首页了。',
    '林口的金色小径又更新啦。',
    '四条路都通着，没有死胡同。',
    '脚边这点苔藓，是从第一颗种子长起来的。',
  ],
  notes: [
    '笔记林里全是知识的年轮。',
    '慢慢看，树不会跑的。',
    '这一片林的根系长得越来越深了。',
    '年轮多一圈，说明有人认真走过一遍。',
    '笔记是留给自己认路的路标。',
    '看不懂的地方折个角，回头再来。',
  ],
  essay: [
    '杂谈林保存着季节感，适合慢慢逛。',
    '这里的风里都是故事的味道。',
    '写下来的日子，就不会白白过去。',
    '有些话是写给自己听的，也顺便给你看。',
    '读到有共鸣的地方，可以多停一会儿。',
  ],
  projects: [
    '这些都是长出来的枝条呀。',
    '点一棵树看看结了什么果子？',
    '做东西的手，是不会骗人的。',
    '枝条歪一点没关系，能结果就行。',
    '从种子到果子，中间全是试错。',
    '有一棵还在长，别急着下结论。',
  ],
  gallery: [
    '照片是时间的标本。',
    '这一带的风景不错吧？',
    '快门按下去的那一刻就不一样了。',
    '光只在那几秒里是那个样子。',
    '有些地方，去过一次就长进身体里了。',
  ],
  about: [
    '这就是种林子的人啦。',
    '嘘，他正在找新的问题。',
    '他种树的样子，比说得好听。',
    '这里写的是他，也是他在意的东西。',
  ],
  search: [
    '找什么？我帮你闻闻味儿。',
    '林子虽大，一句话就能定位。',
    '关键词给得越具体，我找得越快。',
    '搜不到也别灰心，可能它还没长出来。',
  ],
  tags: ['顺着标签走，也是一种路标。', '标签是叶脉，连着一整片林子。', '点一个词，看它牵出多少事情。'],
  dawn: ['早啊，林子刚醒。', '晨雾还没散呢。', '这个点来的人不多，安静得刚刚好。'],
  day: ['阳光正好，适合翻翻笔记。', '今天的林子很安静。', '白天里的光和影子，都很直白。'],
  dusk: ['黄昏的林子是金色的。', '天边烧起来了，看一眼？', '这会儿的影子拉得最长。'],
  night: ['夜深了，萤火虫都出来了。', '晚上的林子是另一种味道。', '黑下来之后，字反而更亮一点。'],
  lateNight: [
    '还没睡呀？别熬太久哦。',
    '星星都困了，你也早点休息。',
    '这个点还亮着屏幕的，就剩你我。',
    '再读一篇就去睡，好不好？',
  ],
  themeDark: ['天黑了……我把小灯点亮。', '夜里也要记得回来呀。', '灯关了，萤火虫值班。'],
  themeLight: ['天亮啦！伸个懒腰——', '光进来了。', '有点晃眼……等我适应一下。'],
  dragFar: ['哇，飞起来了！', '换个地方住也不错。', '轻点儿，苔藓会晕的……', '新视野！记下了记下了。', '这么远啊，那我坐着歇会儿。'],
  firstVisit: [
    '第一次见吧？我叫苔苔，住在这片林子里。',
    '你是新来的吧，风没提过你的味道。',
    '欢迎，随便走，这里不催人。',
  ],
  returnVisit: ['又见面啦。', '你上次停在哪一页？', '风说你会回来，果然。', '老地方，我一直在这儿。'],
  hover: ['痒。', '要说话就点我一下。', '别一直盯着看，我会不好意思。', '拽我也行，我不咬人的。', '有事？没事我就继续站着。'],
  pageEnd: ['到底啦，这一棵看完了。', '下面没有了，风也停了。', '走到林子这一头了。', '要不要回林口，重新挑条路？'],
  lost: ['这条路不存在，跟我回林口吧。', '迷路了？不丢人，我也常走岔。', '这里没有树，只有雾。'],
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

  /* 本次会话的第一面：先让落定动画走完，再打个招呼 */
  const greet = () => {
    try {
      if (sessionStorage.getItem(GREETED_KEY)) return;
      sessionStorage.setItem(GREETED_KEY, '1');
    } catch {
      /* 隐私模式下每次都会打招呼，无伤大雅 */
    }
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

  apply();
}
