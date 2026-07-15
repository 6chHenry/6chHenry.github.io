export interface WatchedFilm {
  title: string;
  poster: string;
  posterAlt: string;
  reviewId?: string;
  description?: string;
}

export const watchlist2026: WatchedFilm[] = [
  {
    title: '河狸变身计划',
    poster: 'images/watchlist/hoppers.webp',
    posterAlt: '电影《河狸变身计划》海报',
    reviewId: 'Movies/河狸变身计划',
  },
  {
    title: '给阿嬷的情书',
    poster: 'images/watchlist/dear-grandma.webp',
    posterAlt: '电影《给阿嬷的情书》海报',
    reviewId: 'Movies/给阿嬷的情书',
  },
  {
    title: '小黄人与大怪兽',
    poster: 'images/watchlist/minions-monsters.webp',
    posterAlt: '电影《小黄人与大怪兽》海报',
    description: '20 世纪 20 年代，一群小黄人误闯好莱坞，意外成为电影明星。热度消退后，詹姆斯与亨利决定自制怪兽片，却召唤出真正的怪兽，只好联手收拾这场失控的混乱。',
  },
  {
    title: '三国第一部：争洛阳',
    poster: 'images/watchlist/three-kingdoms-luoyang.webp',
    posterAlt: '电影《三国第一部：争洛阳》海报',
    description: '东汉末年，宦官、外戚与士族在洛阳角力。曹操、袁绍等青年豪杰被卷入汉室倾颓的权力风暴，在乱世开端寻找各自的破局之路。',
  },
];
