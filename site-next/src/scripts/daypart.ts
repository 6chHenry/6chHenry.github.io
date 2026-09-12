/**
 * 时辰天色：按访客本地时间给首页 Hero 打上
 * 黎明 / 白昼 / 黄昏 / 夜四种天色（样式见 home.css 的 daypart 段）。
 */

import { daypartOf } from '../utils/daypart';

export function initDaypartSky(): void {
  const hero = document.querySelector<HTMLElement>('.home-hero');
  if (!hero) return;

  hero.dataset.daypart = daypartOf(new Date().getHours());
}
