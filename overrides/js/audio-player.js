(function () {
  "use strict";

  var ICON_PLAY =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>';
  var ICON_PAUSE =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h4v14H6zm8 0h4v14h-4z"/></svg>';

  function formatTime(seconds) {
    if (!isFinite(seconds) || seconds < 0) {
      return "0:00";
    }
    var m = Math.floor(seconds / 60);
    var s = Math.floor(seconds % 60);
    return m + ":" + (s < 10 ? "0" : "") + s;
  }

  function prefersReducedMotion() {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  function buildPlayer(root) {
    var src = root.getAttribute("data-src");
    if (!src) {
      return;
    }

    var title = root.getAttribute("data-title") || "录音";
    var label = root.getAttribute("data-label") || "AUDIO LOG";
    var reduced = prefersReducedMotion();

    root.classList.add("ch-audio--ready");
    root.innerHTML =
      '<div class="ch-audio__shell">' +
      '  <div class="ch-audio__fx" aria-hidden="true">' +
      '    <div class="ch-audio__grid"></div>' +
      '    <div class="ch-audio__glow"></div>' +
      '    <div class="ch-audio__scan"></div>' +
      "  </div>" +
      '  <div class="ch-audio__top">' +
      '    <span class="ch-audio__badge">' +
      label +
      "</span>" +
      '    <h3 class="ch-audio__title">' +
      title +
      "</h3>" +
      '    <div class="ch-audio__viz" aria-hidden="true"></div>' +
      "  </div>" +
      '  <div class="ch-audio__controls">' +
      '    <button type="button" class="ch-audio__btn ch-audio__btn--play" aria-label="播放">' +
      ICON_PLAY +
      "</button>" +
      '    <button type="button" class="ch-audio__btn ch-audio__btn--skip" data-skip="-15" aria-label="后退 15 秒">−15</button>' +
      '    <div class="ch-audio__track-wrap">' +
      '      <input type="range" class="ch-audio__seek" min="0" max="1000" value="0" step="1" aria-label="播放进度" />' +
      '      <div class="ch-audio__track-fill"></div>' +
      "    </div>" +
      '    <button type="button" class="ch-audio__btn ch-audio__btn--skip" data-skip="15" aria-label="前进 15 秒">+15</button>' +
      '    <span class="ch-audio__time"><span class="ch-audio__cur">0:00</span> / <span class="ch-audio__dur">0:00</span></span>' +
      "  </div>" +
      '  <audio class="ch-audio__el" preload="metadata" src="' +
      src +
      '"></audio>' +
      "</div>";

    var shell = root.querySelector(".ch-audio__shell");
    var audio = root.querySelector(".ch-audio__el");
    var playBtn = root.querySelector(".ch-audio__btn--play");
    var seek = root.querySelector(".ch-audio__seek");
    var fill = root.querySelector(".ch-audio__track-fill");
    var curEl = root.querySelector(".ch-audio__cur");
    var durEl = root.querySelector(".ch-audio__dur");
    var viz = root.querySelector(".ch-audio__viz");
    var skipBtns = root.querySelectorAll("[data-skip]");

    var bars = [];
    var barCount = reduced ? 0 : 28;
    var rafId = null;
    var ticker = 0;
    var seeking = false;

    if (barCount > 0) {
      for (var i = 0; i < barCount; i++) {
        var bar = document.createElement("span");
        bar.className = "ch-audio__bar";
        viz.appendChild(bar);
        bars.push(bar);
      }
    }

    function getDuration() {
      return isFinite(audio.duration) && audio.duration > 0 ? audio.duration : 0;
    }

    function setProgressFromTime(time) {
      var dur = getDuration();
      var pct = dur > 0 ? (time / dur) * 100 : 0;
      seek.value = String(Math.round(pct * 10));
      fill.style.width = pct + "%";
      curEl.textContent = formatTime(time);
      durEl.textContent = dur > 0 ? formatTime(dur) : "0:00";
    }

    function updateProgress() {
      if (seeking) {
        return;
      }
      setProgressFromTime(audio.currentTime);
    }

    function seekToRatio(ratio) {
      var dur = getDuration();
      if (dur <= 0) {
        return;
      }
      var clamped = Math.max(0, Math.min(1, ratio));
      var target = clamped * dur;
      audio.currentTime = target;
      setProgressFromTime(target);
    }

    function setPlayingUI(playing) {
      root.classList.toggle("is-playing", playing);
      playBtn.innerHTML = playing ? ICON_PAUSE : ICON_PLAY;
      playBtn.setAttribute("aria-label", playing ? "暂停" : "播放");
    }

    function stopVisualizer() {
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      bars.forEach(function (bar) {
        bar.style.setProperty("--h", "0.15");
      });
    }

    function tickVisualizer() {
      if (audio.paused || audio.ended) {
        stopVisualizer();
        return;
      }
      ticker += 1;
      var phase = audio.currentTime * 7 + ticker * 0.09;
      for (var i = 0; i < barCount; i++) {
        var sinA = Math.sin(phase + i * 0.42);
        var sinB = Math.sin(phase * 0.63 + i * 0.78);
        var v = (sinA * 0.55 + sinB * 0.45 + 1) / 2;
        var h = 0.18 + v * 0.82;
        bars[i].style.setProperty("--h", h.toFixed(3));
      }
      rafId = requestAnimationFrame(tickVisualizer);
    }

    function pauseOthers() {
      document.querySelectorAll("[data-ch-audio].is-playing").forEach(function (other) {
        if (other === root) {
          return;
        }
        var otherAudio = other.querySelector(".ch-audio__el");
        if (otherAudio && !otherAudio.paused) {
          otherAudio.pause();
          other.classList.remove("is-playing");
          var otherBtn = other.querySelector(".ch-audio__btn--play");
          if (otherBtn) {
            otherBtn.innerHTML = ICON_PLAY;
            otherBtn.setAttribute("aria-label", "播放");
          }
        }
      });
    }

    playBtn.addEventListener("click", function () {
      if (audio.paused) {
        pauseOthers();
        audio.play().catch(function () {
          shell.classList.add("ch-audio__shell--error");
        });
      } else {
        audio.pause();
      }
    });

    skipBtns.forEach(function (btn) {
      btn.addEventListener("click", function () {
        var delta = parseFloat(btn.getAttribute("data-skip"), 10) || 0;
        var dur = getDuration();
        var next = audio.currentTime + delta;
        if (dur > 0) {
          next = Math.max(0, Math.min(dur, next));
        } else {
          next = Math.max(0, next);
        }
        audio.currentTime = next;
        setProgressFromTime(next);
      });
    });

    seek.addEventListener("pointerdown", function () {
      seeking = true;
    });

    seek.addEventListener("input", function () {
      var pct = parseFloat(seek.value, 10) / 1000;
      fill.style.width = pct * 100 + "%";
      var dur = getDuration();
      if (dur <= 0) {
        return;
      }
      var target = pct * dur;
      audio.currentTime = target;
      curEl.textContent = formatTime(target);
    });

    seek.addEventListener("change", function () {
      seeking = false;
      seekToRatio(parseFloat(seek.value, 10) / 1000);
    });

    audio.addEventListener("loadedmetadata", updateProgress);
    audio.addEventListener("durationchange", updateProgress);
    audio.addEventListener("timeupdate", updateProgress);
    audio.addEventListener("ended", function () {
      setPlayingUI(false);
      stopVisualizer();
    });
    audio.addEventListener("play", function () {
      setPlayingUI(true);
      if (!reduced && barCount > 0) {
        tickVisualizer();
      }
    });
    audio.addEventListener("pause", function () {
      setPlayingUI(false);
      stopVisualizer();
    });
    audio.addEventListener("error", function () {
      shell.classList.add("ch-audio__shell--error");
      durEl.textContent = "无法加载";
    });
  }

  function init() {
    document.querySelectorAll("[data-ch-audio]").forEach(buildPlayer);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
