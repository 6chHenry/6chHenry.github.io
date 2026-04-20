---
hide:
  - date
  - navigation
  - toc
home: true
nostatistics: true
comments: false
icon: material/home
---

# Welcome to 6ch.'s Website ✨

<div class="home-hero" markdown="1">

你好，我是 **6ch.**，在这里记录学习、思考和生活。

<div class="home-hero-actions" markdown="1">

[了解我 :octicons-info-16:](./about/index.md){ .md-button .md-button--primary }
[学术主页 :academicons-google-scholar:](./academy.md){ .md-button }
[查看统计 :material-chart-line:](javascript:toggle_statistics();){ .md-button }

</div>

</div>

<div id="statistics" class="home-stat card" markdown="1">

- Website Operating Time: <span id="web-time"></span>
- Total Visitors: <span id="busuanzi_value_site_uv"></span> people
- Total Visits: <span id="busuanzi_value_site_pv"></span> times

</div>

## 快速开始

<div class="home-grid">
  <a class="home-grid-card" href="./notes/index.md">
    <h3>🧠 学习笔记</h3>
    <p>算法、机器学习与课程笔记，持续更新。</p>
  </a>
  <a class="home-grid-card" href="./projects/index.md">
    <h3>🚀 项目记录</h3>
    <p>正在做什么、怎么做的、踩过哪些坑。</p>
  </a>
  <a class="home-grid-card" href="./diaries/index.md">
    <h3>🌿 日记与随笔</h3>
    <p>日常观察、旅行想法与阶段性复盘。</p>
  </a>
</div>

<script>
function updateTime() {
    var date = new Date();
    var now = date.getTime();
    var startDate = new Date("2025/02/28 22:00:00");
    var start = startDate.getTime();
    var diff = now - start;
    var y, d, h, m;
    y = Math.floor(diff / (365 * 24 * 3600 * 1000));
    diff -= y * 365 * 24 * 3600 * 1000;
    d = Math.floor(diff / (24 * 3600 * 1000));
    h = Math.floor(diff / (3600 * 1000) % 24);
    m = Math.floor(diff / (60 * 1000) % 60);
    if (y == 0) {
        document.getElementById("web-time").innerHTML = d + "<span> </span>d<span> </span>" + h + "<span> </span>h<span> </span>" + m + "<span> </span>m";
    } else {
        document.getElementById("web-time").innerHTML = y + "<span> </span>y<span> </span>" + d + "<span> </span>d<span> </span>" + h + "<span> </span>h<span> </span>" + m + "<span> </span>m";
    }
    setTimeout(updateTime, 1000 * 60);
}
updateTime();
function toggle_statistics() {
    var statistics = document.getElementById("statistics");
  if (!statistics.classList.contains("is-visible")) {
    statistics.classList.add("is-visible");
    } else {
    statistics.classList.remove("is-visible");
    }
}
</script>