(function () {
  "use strict";

  function initHomePage() {
    var hero = document.querySelector(".home-hero");
    if (!hero) {
      return;
    }

    document.body.classList.add("page-home");

    var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!reducedMotion) {
      requestAnimationFrame(function () {
        hero.classList.add("is-ready");
      });
    } else {
      hero.classList.add("is-ready");
    }

    var statsBtn = document.getElementById("home-stats-toggle");
    var statistics = document.getElementById("statistics");
    if (statsBtn && statistics) {
      statsBtn.addEventListener("click", function (e) {
        e.preventDefault();
        statistics.classList.toggle("is-visible");
        var visible = statistics.classList.contains("is-visible");
        statsBtn.setAttribute("aria-expanded", visible ? "true" : "false");
      });
    }

    updateTime();
  }

  function updateTime() {
    var el = document.getElementById("web-time");
    if (!el) {
      return;
    }

    var now = Date.now();
    var start = new Date("2025/02/28 22:00:00").getTime();
    var diff = now - start;
    var y = Math.floor(diff / (365 * 24 * 3600 * 1000));
    diff -= y * 365 * 24 * 3600 * 1000;
    var d = Math.floor(diff / (24 * 3600 * 1000));
    var h = Math.floor((diff / (3600 * 1000)) % 24);
    var m = Math.floor((diff / (60 * 1000)) % 60);

    if (y === 0) {
      el.innerHTML =
        d + "<span> </span>d<span> </span>" +
        h + "<span> </span>h<span> </span>" +
        m + "<span> </span>m";
    } else {
      el.innerHTML =
        y + "<span> </span>y<span> </span>" +
        d + "<span> </span>d<span> </span>" +
        h + "<span> </span>h<span> </span>" +
        m + "<span> </span>m";
    }

    setTimeout(updateTime, 60000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initHomePage);
  } else {
    initHomePage();
  }
})();
