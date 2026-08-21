/**
 * 时辰天色：按访客本地时间给首页 Hero 打上
 * 黎明 / 白昼 / 黄昏 / 夜四种天色（样式见 home.css 的 daypart 段）。
 */

export function initDaypartSky(): void {
  const hero = document.querySelector<HTMLElement>('.home-hero');
  if (!hero) return;

  const hour = new Date().getHours();
  const part =
    hour >= 5 && hour < 8 ? 'dawn' :
    hour >= 8 && hour < 17 ? 'day' :
    hour >= 17 && hour < 20 ? 'dusk' :
    'night';

  hero.dataset.daypart = part;
}
