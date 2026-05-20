const k='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>',C='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h4v14H6zm8 0h4v14h-4z"/></svg>';function A(t){if(!Number.isFinite(t)||t<0)return"0:00";const r=Math.floor(t/60),d=Math.floor(t%60);return`${r}:${d<10?"0":""}${d}`}function B(){return window.matchMedia("(prefers-reduced-motion: reduce)").matches}function I(t){if(t.startsWith("http://")||t.startsWith("https://"))return t;const r="/".replace(/\/$/,"");return t.startsWith("/")?`${r}${t}`:`${r}/${t}`.replace(/\/{2,}/g,"/")}function N(t){const r=t.getAttribute("data-src");if(!r)return;const d=I(r),x=t.getAttribute("data-title")||"录音",T=t.getAttribute("data-label")||"AUDIO LOG",M=B();t.classList.add("ch-audio--ready"),t.innerHTML=`
    <div class="ch-audio__shell">
      <div class="ch-audio__fx" aria-hidden="true">
        <div class="ch-audio__grid"></div>
        <div class="ch-audio__glow"></div>
        <div class="ch-audio__scan"></div>
      </div>
      <div class="ch-audio__top">
        <span class="ch-audio__badge">${T}</span>
        <h3 class="ch-audio__title">${x}</h3>
        <div class="ch-audio__viz" aria-hidden="true"></div>
      </div>
      <div class="ch-audio__controls">
        <button type="button" class="ch-audio__btn ch-audio__btn--play" aria-label="播放">${k}</button>
        <button type="button" class="ch-audio__btn ch-audio__btn--skip" data-skip="-15" aria-label="后退 15 秒">−15</button>
        <div class="ch-audio__track-wrap">
          <input type="range" class="ch-audio__seek" min="0" max="1000" value="0" step="1" aria-label="播放进度" />
          <div class="ch-audio__track-fill"></div>
        </div>
        <button type="button" class="ch-audio__btn ch-audio__btn--skip" data-skip="15" aria-label="前进 15 秒">+15</button>
        <span class="ch-audio__time"><span class="ch-audio__cur">0:00</span> / <span class="ch-audio__dur">0:00</span></span>
      </div>
      <audio class="ch-audio__el" preload="metadata" src="${d}"></audio>
    </div>`;const h=t.querySelector(".ch-audio__shell"),i=t.querySelector(".ch-audio__el"),u=t.querySelector(".ch-audio__btn--play"),c=t.querySelector(".ch-audio__seek"),_=t.querySelector(".ch-audio__track-fill"),p=t.querySelector(".ch-audio__cur"),v=t.querySelector(".ch-audio__dur"),S=t.querySelector(".ch-audio__viz"),w=t.querySelectorAll("[data-skip]");if(!h||!i||!u||!c||!_||!p||!v||!S)return;const f=[],b=M?0:28;let l=null,q=0,m=!1;for(let e=0;e<b;e+=1){const a=document.createElement("span");a.className="ch-audio__bar",S.appendChild(a),f.push(a)}const o=()=>Number.isFinite(i.duration)&&i.duration>0?i.duration:0,y=e=>{const a=o(),s=a>0?e/a*100:0;c.value=String(Math.round(s*10)),_.style.width=`${s}%`,p.textContent=A(e),v.textContent=a>0?A(a):"0:00"},g=()=>{m||y(i.currentTime)},P=e=>{const a=o();if(a<=0)return;const n=Math.max(0,Math.min(1,e))*a;i.currentTime=n,y(n)},L=e=>{t.classList.toggle("is-playing",e),u.innerHTML=e?C:k,u.setAttribute("aria-label",e?"暂停":"播放")},E=()=>{l&&(cancelAnimationFrame(l),l=null),f.forEach(e=>e.style.setProperty("--h","0.15"))},$=()=>{if(i.paused||i.ended){E();return}q+=1;const e=i.currentTime*7+q*.09;for(let a=0;a<b;a+=1){const s=Math.sin(e+a*.42),n=Math.sin(e*.63+a*.78),z=(s*.55+n*.45+1)/2;f[a].style.setProperty("--h",(.18+z*.82).toFixed(3))}l=requestAnimationFrame($)},F=()=>{document.querySelectorAll("[data-ch-audio].is-playing").forEach(e=>{if(e===t)return;const a=e.querySelector(".ch-audio__el"),s=e.querySelector(".ch-audio__btn--play");a&&!a.paused&&(a.pause(),e.classList.remove("is-playing"),s&&(s.innerHTML=k,s.setAttribute("aria-label","播放")))})};u.addEventListener("click",()=>{i.paused?(F(),i.play().catch(()=>h.classList.add("ch-audio__shell--error"))):i.pause()}),w.forEach(e=>{e.addEventListener("click",()=>{const a=parseFloat(e.getAttribute("data-skip")||"0"),s=o();let n=i.currentTime+a;n=s>0?Math.max(0,Math.min(s,n)):Math.max(0,n),i.currentTime=n,y(n)})}),c.addEventListener("pointerdown",()=>{m=!0}),c.addEventListener("input",()=>{const e=parseFloat(c.value)/1e3;_.style.width=`${e*100}%`;const a=o();if(a<=0)return;const s=e*a;i.currentTime=s,p.textContent=A(s)}),c.addEventListener("change",()=>{m=!1,P(parseFloat(c.value)/1e3)}),i.addEventListener("loadedmetadata",g),i.addEventListener("durationchange",g),i.addEventListener("timeupdate",g),i.addEventListener("ended",()=>{L(!1),E()}),i.addEventListener("play",()=>{L(!0),!M&&b>0&&$()}),i.addEventListener("pause",()=>{L(!1),E()}),i.addEventListener("error",()=>{h.classList.add("ch-audio__shell--error"),v.textContent="无法加载"})}function O(){document.querySelectorAll("[data-ch-audio]").forEach(N)}O();
