/**
 * 时辰判定：首页的天色与苔苔的时辰台词共用同一把尺子，
 * 免得两边对「黄昏」和「夜」的分界各有各的说法。
 */

export type Daypart = 'dawn' | 'day' | 'dusk' | 'night';

export function daypartOf(hour: number): Daypart {
  if (hour >= 5 && hour < 8) return 'dawn';
  if (hour >= 8 && hour < 17) return 'day';
  if (hour >= 17 && hour < 20) return 'dusk';
  return 'night';
}
