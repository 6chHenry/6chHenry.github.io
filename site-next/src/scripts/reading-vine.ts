/**
 * 阅读藤蔓：文章页左侧一根随阅读进度生长的藤，
 * 走过一段距离就抽一片新叶，读到最后顶端结出一颗芽。
 * 样式见 styles/content.css 的「Reading vine」段。
 */

const NS = 'http://www.w3.org/2000/svg';
const VIEW_W = 40;
const VIEW_H = 620;
const STEM_D =
  'M20 10 C32 92 8 168 19 252 C30 332 10 408 21 488 C28 546 16 578 19 606';
/* 每片叶子的位置（占茎长的比例），出现阈值略微滞后 */
const LEAF_SPOTS = [0.06, 0.17, 0.29, 0.41, 0.53, 0.65, 0.77, 0.88];
const LEAF_D = 'M0 0 C-6.5 -4 -8.5 -13 0 -21 C8.5 -13 6.5 -4 0 0 Z';

export function initReadingVine(): void {
  const host = document.querySelector<HTMLElement>('.reading-vine');
  if (!host || host.childElementCount > 0) return;

  const el = (name: string, attrs: Record<string, string | number>) => {
    const node = document.createElementNS(NS, name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    return node;
  };

  const svg = el('svg', { viewBox: `0 0 ${VIEW_W} ${VIEW_H}`, 'preserveAspectRatio': 'xMidYMid meet' });

  el('path', { d: STEM_D, class: 'reading-vine__stem reading-vine__stem--ghost' });
  const stem = el('path', { d: STEM_D, class: 'reading-vine__stem' });
  svg.append(stem);

  const leaves: Array<{ node: SVGPathElement; showAt: number }> = [];
  const stemLength = stem.getTotalLength();
  stem.style.strokeDasharray = `${stemLength}`;
  stem.style.strokeDashoffset = `${stemLength}`;

  LEAF_SPOTS.forEach((spot, index) => {
    const point = stem.getPointAtLength(stemLength * spot);
    /* 叶子随机长在茎的左侧或右侧，张角和大小都带抖动，避免对称感 */
    const side = Math.random() < 0.5 ? -1 : 1;
    const angle = side * (26 + Math.random() * 24);
    const scale = 0.78 + Math.random() * 0.38;

    const wrapper = el('g', {
      transform: `translate(${point.x.toFixed(1)} ${point.y.toFixed(1)}) rotate(${angle.toFixed(1)}) scale(${scale.toFixed(2)})`,
    });
    const leaf = el('path', {
      d: LEAF_D,
      class: `reading-vine__leaf${index % 3 === 1 ? ' reading-vine__leaf--alt' : ''}`,
    });
    leaf.dataset.showAt = (spot + 0.02).toFixed(3);
    wrapper.append(leaf);
    svg.append(wrapper);
    leaves.push({ node: leaf, showAt: spot + 0.02 });
  });

  const tip = stem.getPointAtLength(stemLength);
  const bud = el('circle', {
    cx: tip.x.toFixed(1),
    cy: tip.y.toFixed(1),
    r: 3.4,
    class: 'reading-vine__bud',
  });
  svg.append(bud);
  host.append(svg);

  let ticking = false;
  const update = () => {
    ticking = false;
    const scrollable = document.documentElement.scrollHeight - window.innerHeight;
    const progress = scrollable > 0 ? Math.min(1, Math.max(0, window.scrollY / scrollable)) : 0;
    stem.style.strokeDashoffset = `${stemLength * (1 - progress)}`;
    for (const leaf of leaves) {
      leaf.node.classList.toggle('is-grown', progress >= leaf.showAt);
    }
    bud.classList.toggle('is-grown', progress >= 0.985);
  };

  const onScroll = () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(update);
  };

  window.addEventListener('scroll', onScroll, { passive: true });
  update();
}
