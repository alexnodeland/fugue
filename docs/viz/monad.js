// docs/viz/monad.js — "The Model Is a Score", rebuilt DATA-FIRST.
//
// HERO: a live distribution over the mean mu — blue prior density, green exact
// conjugate posterior, five large draggable yellow data dots, a coral tick at
// the current run's mu. Drag a dot and the green curve slides in real time.
//
// MACHINERY STRIP (below, compact): the CPS chain as small chips
// (SampleF64 mu -> ObserveF64 y1..y5 -> Pure); Step/Play walk it and each
// Observe chip lights ITS data dot in the hero. "Perform x200" rains prior
// draws (blue cloud, PriorHandler) or stacks them on one mu (coral spike,
// ReplayHandler) — improvising vs replaying, told as distribution mass.
//
// Model:  mu ~ Normal(0, 2),  y_i ~ Normal(mu, 1)  for i = 1..5.
// Self-contained IIFE; consumes window.FugueViz (loaded first).
(function () {
  "use strict";
  if (typeof window === "undefined" || !window.FugueViz) return;

  window.FugueViz.register("monad", function (root, FV) {
    var MONO = 'ui-monospace, "Source Code Pro", SFMono-Regular, Menlo, monospace';
    var SANS = "system-ui, sans-serif";
    var SUB = ["₁", "₂", "₃", "₄", "₅"]; // y1..y5

    // ---- model constants ----
    var PRIOR_MU = 0.0, PRIOR_SD = 2.0, OBS_SD = 1.0;
    var PRIOR_VAR = PRIOR_SD * PRIOR_SD, OBS_VAR = OBS_SD * OBS_SD;

    // ---- data (draggable) ----
    var DATA0 = [1.3, 0.7, 2.1, 0.4, 1.5]; // seeded default; sum = 6.0
    var data = DATA0.slice();
    var N = data.length;

    // ---- seeds ----
    var baseSeed = parseInt(root.getAttribute("data-seed"), 10);
    if (!isFinite(baseSeed)) baseSeed = 11;
    baseSeed = baseSeed >>> 0;
    var liveSeed = baseSeed;
    // The ReplayHandler's fixed recording: the canonical prior draw at the page
    // seed. It never changes, whatever the live seed does.
    var recordMu = FV.dist.normal.sample(FV.rng(baseSeed), PRIOR_MU, PRIOR_SD);

    // ---- run state ----
    var mode = "prior";       // "prior" | "replay"
    var muCurrent = drawMu(); // the current run's sampled mu (coral tick)

    // ---- machinery walk state ----
    // nodes: 0 = SampleF64 mu, 1..5 = ObserveF64 y_i, 6 = Pure(mu)
    var NODES = 7, PURE = 6;
    var committed = newBoolArray(NODES);
    var cursor = 0;           // next node to interpret
    var walkActive = -1;      // node currently highlighted (persists after commit)
    var autoPlay = false;
    var stepClock = 0;        // seconds since last auto-advance
    var STEP_GAP = 0.55;
    var holdClock = 0;        // seconds resting on a finished performance
    var HOLD = 1.4;           // pause on Pure before the ambient reloop

    // ---- rain (Perform x200) ----
    var rainTicks = null;     // array of mu draws, or null
    var rainRole = "prior";

    // ---- posterior (exact conjugate; variance is constant) ----
    var POST_VAR = 1 / (1 / PRIOR_VAR + N / OBS_VAR);
    var POST_SD = Math.sqrt(POST_VAR);
    var POST_PEAK = 1 / (POST_SD * Math.sqrt(2 * Math.PI));
    function postMean() {
      var sy = 0;
      for (var i = 0; i < N; i++) sy += data[i];
      return POST_VAR * (PRIOR_MU / PRIOR_VAR + sy / OBS_VAR);
    }

    function drawMu() {
      return mode === "replay"
        ? recordMu
        : FV.dist.normal.sample(FV.rng(liveSeed), PRIOR_MU, PRIOR_SD);
    }
    function newBoolArray(n) { var a = []; for (var i = 0; i < n; i++) a.push(false); return a; }

    // ------------------------------------------------------------------ tallies
    function logPrior() {
      return committed[0] ? FV.dist.normal.logpdf(muCurrent, PRIOR_MU, PRIOR_SD) : null;
    }
    function logLike() {
      var s = 0, any = false;
      for (var k = 1; k <= 5; k++) {
        if (committed[k]) { s += FV.dist.normal.logpdf(data[k - 1], muCurrent, OBS_SD); any = true; }
      }
      return any ? s : null;
    }

    // ------------------------------------------------------------------ walking
    function applyStep(i) { committed[i] = true; }
    function runAllInstant() {
      committed = newBoolArray(NODES);
      muCurrent = drawMu();
      for (var i = 0; i < NODES; i++) applyStep(i);
      cursor = NODES; walkActive = -1;
      autoPlay = false; updatePlayLabel();
      updateReadouts(); render();
    }
    function resetWalk() {
      committed = newBoolArray(NODES);
      cursor = 0; walkActive = -1;
      autoPlay = false; stepClock = 0; updatePlayLabel();
      muCurrent = drawMu();
      rainTicks = null; // §A.6: a Perform ×200 cloud must not survive Reset
      if (loopApi) loopApi.pause();
      updateReadouts(); render();
    }
    function stepOnce() {
      if (cursor >= NODES) return;
      applyStep(cursor);
      walkActive = cursor;
      cursor++;
      updateReadouts(); render();
    }
    function reseed(seed) {
      liveSeed = seed >>> 0;
      rainTicks = null;
      runAllInstant();
    }
    function setMode(replayOn) {
      mode = replayOn ? "replay" : "prior";
      rainTicks = null;
      runAllInstant();
    }
    function performRain() {
      var r = FV.rng((liveSeed ^ 0x9e3779b9) >>> 0);
      rainTicks = [];
      rainRole = mode === "replay" ? "hot" : "prior";
      for (var i = 0; i < 200; i++) {
        rainTicks.push(mode === "replay" ? recordMu : FV.dist.normal.sample(r, PRIOR_MU, PRIOR_SD));
      }
      render();
    }

    // ------------------------------------------------------------------ tick
    function tick(dt) {
      if (autoPlay) {
        if (cursor >= NODES) {
          // A full performance is on the strip — rest on it, then reloop with a
          // fresh improvisation so the widget stays quietly alive on the page.
          holdClock += dt;
          if (holdClock >= HOLD) {
            holdClock = 0;
            committed = newBoolArray(NODES);
            cursor = 0; walkActive = -1; stepClock = 0;
            muCurrent = drawMu();
            updateReadouts();
          }
        } else {
          stepClock += dt;
          while (stepClock >= STEP_GAP && cursor < NODES) {
            stepClock -= STEP_GAP;
            stepOnce();
          }
        }
      }
      render();
    }

    // ------------------------------------------------------------------ DOM
    function elem(tag, cls, parent) {
      var e = document.createElement(tag);
      if (cls) e.className = cls;
      if (parent) parent.appendChild(e);
      return e;
    }

    var controls = elem("div", "fv-controls", root);
    var btnRoot = FV.buttons(controls, [
      { label: "Step", title: "Interpret the next node", onClick: onStep },
      { label: "Play", title: "Walk the whole chain", primary: true, onClick: onPlay },
      { label: "Reset", title: "Rewind to the first node", onClick: resetWalk },
      { label: "Perform ×200", title: "Run the handler 200 times", onClick: performRain }
    ]);
    var playBtn = btnRoot.fvButtons["Play"];
    FV.toggle(controls, {
      label: "Replay handler", value: false,
      onChange: function (on) { setMode(on); }
    });

    // Hero canvas: the live distribution over mu.
    var hero = FV.canvas(root, { height: 400, onResize: function () { render(); } });
    var hctx = hero.ctx;

    var instr = elem("div", "fv-instruction", root);
    instr.textContent = "Drag a yellow data point left or right — the green posterior slides to follow it.";

    // Machinery strip: a second, short canvas.
    var strip = FV.canvas(root, { height: 116, onResize: function () { render(); } });
    var sctx = strip.ctx;

    var readouts = elem("div", "fv-readouts", root);
    var roMu = FV.readout(readouts, { label: "mu (current run)" });
    var roMean = FV.readout(readouts, { label: "posterior mean" });
    var roSd = FV.readout(readouts, { label: "posterior sd" });
    var roTot = FV.readout(readouts, { label: "total log-weight" });

    var hint = elem("div", "fv-hint", root);
    hint.textContent = "try: press Perform ×200 in PriorHandler mode (a blue cloud of guesses), then flip on Replay and press it again (one coral spike).";

    // Seed scrub lives in the prose (id=monad-seed). Bind it if present.
    var seedSpan = document.getElementById("monad-seed");
    if (seedSpan && FV.scrub) {
      FV.scrub(seedSpan, {
        min: 1, max: 40, step: 1, value: liveSeed,
        fmt: function (v) { return String(v); },
        onInput: function (v) { reseed(v >>> 0); }
      });
    }

    function updateReadouts() {
      roMu.set(muCurrent == null ? "—" : muCurrent.toFixed(3), "hot");
      roMean.set(postMean().toFixed(3), "post");
      roSd.set(POST_SD.toFixed(3), "post");
      var lp = logPrior(), ll = logLike();
      if (lp == null) roTot.set("—", "post");
      else roTot.set((lp + (ll || 0)).toFixed(3), "post");
    }
    function updatePlayLabel() { if (playBtn) playBtn.textContent = autoPlay ? "Pause" : "Play"; }

    function onStep() {
      autoPlay = false; updatePlayLabel();
      if (cursor >= NODES) resetWalk();
      stepOnce();
      if (loopApi) loopApi.pause();
    }
    function onPlay() {
      if (autoPlay) { autoPlay = false; updatePlayLabel(); if (loopApi) loopApi.pause(); return; }
      if (cursor >= NODES) resetWalk();
      if (loopApi.reduced) { while (cursor < NODES) stepOnce(); return; }
      autoPlay = true; stepClock = 0; updatePlayLabel();
      loopApi.play();
    }

    // ---------------------------------------------------------- hero geometry
    var heroLayout = null;
    function computeHeroLayout() {
      var w = hero.w, h = hero.h;
      var padL = 30, padR = 16, padT = 16, padB = 34;
      var plot = { x: padL, y: padT, w: w - padL - padR, h: h - padT - padB };
      var xs = FV.scale([-4, 6], [plot.x, plot.x + plot.w]);
      var ymax = POST_PEAK * 1.14;
      var baseY = plot.y + plot.h;
      var ys = FV.scale([0, ymax], [baseY, plot.y]);
      return { plot: plot, xs: xs, ys: ys, baseY: baseY, dotY: baseY };
    }

    // ---------------------------------------------------------- hero drawing
    function pdfCurve(xs, ys, mu, sd) {
      var pts = [], x0 = xs.domain[0], x1 = xs.domain[1], steps = 180;
      for (var i = 0; i <= steps; i++) {
        var xv = x0 + (x1 - x0) * (i / steps);
        var d = Math.exp(FV.dist.normal.logpdf(xv, mu, sd));
        pts.push([xs(xv), ys(d)]);
      }
      return pts;
    }

    function renderHero() {
      if (!hero) return;
      var col = FV.theme().colors, now = Date.now();
      hero.clear();
      var L = heroLayout = computeHeroLayout();
      var xs = L.xs, ys = L.ys, ctx = hctx;

      // baseline + x ticks + axis label
      ctx.save();
      ctx.strokeStyle = col.grid; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(L.plot.x, L.baseY); ctx.lineTo(L.plot.x + L.plot.w, L.baseY); ctx.stroke();
      ctx.fillStyle = col.ink; ctx.font = "11px " + MONO;
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      for (var t = -4; t <= 6; t += 2) {
        var px = xs(t);
        ctx.globalAlpha = 0.5;
        ctx.beginPath(); ctx.moveTo(px, L.baseY); ctx.lineTo(px, L.baseY + 4); ctx.stroke();
        ctx.globalAlpha = 0.75;
        ctx.fillText(String(t), px, L.baseY + 6);
      }
      ctx.globalAlpha = 0.6; ctx.textAlign = "right";
      ctx.fillText("μ  /  y", L.plot.x + L.plot.w, L.baseY + 6);
      ctx.restore();

      // rain (Perform x200): a mini-histogram under the curves
      if (rainTicks) {
        FV.histogram(ctx, rainTicks, {
          bins: 48, xscale: xs, yscale: ys,
          color: rainRole === "hot" ? col.hot : col.prior, alpha: 0.28
        });
      }

      // prior (blue) — broad and low
      FV.curve(ctx, pdfCurve(xs, ys, PRIOR_MU, PRIOR_SD), { color: col.prior, width: 2 });
      // posterior (green) — the answer, exact conjugate
      var pm = postMean();
      var postPts = pdfCurve(xs, ys, pm, POST_SD);
      // translucent fill under posterior
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(postPts[0][0], L.baseY);
      for (var i = 0; i < postPts.length; i++) ctx.lineTo(postPts[i][0], postPts[i][1]);
      ctx.lineTo(postPts[postPts.length - 1][0], L.baseY);
      ctx.closePath();
      ctx.globalAlpha = 0.12; ctx.fillStyle = col.post; ctx.fill();
      ctx.restore();
      FV.curve(ctx, postPts, { color: col.post, width: 2.4 });

      // legend
      legend(ctx, L, col);

      // current mu coral tick (from baseline up to posterior height at mu)
      if (muCurrent != null) {
        var mx = xs(muCurrent);
        var mdens = Math.exp(FV.dist.normal.logpdf(muCurrent, pm, POST_SD));
        var topPix = ys(Math.min(mdens, ys.domain[1]));
        ctx.save();
        ctx.strokeStyle = col.hot; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(mx, L.baseY); ctx.lineTo(mx, topPix - 4); ctx.stroke();
        // marker triangle at top
        ctx.fillStyle = col.hot;
        ctx.beginPath();
        ctx.moveTo(mx, topPix - 4); ctx.lineTo(mx - 4, topPix - 11); ctx.lineTo(mx + 4, topPix - 11);
        ctx.closePath(); ctx.fill();
        ctx.font = "600 10px " + MONO; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
        ctx.fillText("μ", mx, topPix - 12);
        ctx.restore();
      }

      // data dots (yellow), draggable; pulse when their Observe chip is active
      for (var d = 0; d < N; d++) {
        var dx = xs(data[d]), dy = L.dotY;
        var activeObs = (walkActive === d + 1);
        var grabbing = dragHandle && dragHandle.grabbed && dragHandle.target === d;
        var pulse = (activeObs && !loopApi.reduced) ? 0.5 + 0.5 * Math.sin(now / 200) : 0;
        var rad = 7 + (activeObs ? 2 : 0) + (grabbing ? 2 : 0);
        if (grabbing) FV.halo(ctx, dx, dy, rad + 8); // §A.2 grab halo
        ctx.save();
        if (activeObs) {
          ctx.globalAlpha = 0.25 + 0.35 * pulse;
          ctx.beginPath(); ctx.arc(dx, dy, rad + 6, 0, 6.2832);
          ctx.fillStyle = col.hot; ctx.fill();
          ctx.globalAlpha = 1;
        }
        ctx.beginPath(); ctx.arc(dx, dy, rad, 0, 6.2832);
        ctx.fillStyle = col.data; ctx.fill();
        ctx.lineWidth = 1.5; ctx.strokeStyle = activeObs ? col.hot : col.data; ctx.stroke();
        ctx.restore();
      }
    }

    function legend(ctx, L, col) {
      var items = [["prior  μ~Normal(0,2)", col.prior], ["posterior", col.post], ["data y", col.data]];
      ctx.save();
      ctx.font = "11px " + SANS; ctx.textBaseline = "middle"; ctx.textAlign = "left";
      var lx = L.plot.x + 8, ly = L.plot.y + 10;
      for (var i = 0; i < items.length; i++) {
        ctx.fillStyle = items[i][1];
        ctx.beginPath(); ctx.arc(lx + 4, ly, 4, 0, 6.2832); ctx.fill();
        ctx.fillStyle = col.ink; ctx.globalAlpha = 0.85;
        ctx.fillText(items[i][0], lx + 13, ly + 0.5);
        ctx.globalAlpha = 1;
        ly += 16;
      }
      ctx.restore();
    }

    // ---------------------------------------------------------- strip drawing
    var chipRects = [];
    function renderStrip() {
      if (!strip) return;
      var col = FV.theme().colors, now = Date.now();
      strip.clear();
      var ctx = sctx, w = strip.w, h = strip.h;
      var narrow = w < 520;
      var padX = 8;
      var chipY = 26, chipH = 40;
      var gap = narrow ? 7 : 13;
      var inner = w - 2 * padX;
      var chipW = (inner - gap * (NODES - 1)) / NODES;
      var fs = narrow ? 9 : 10.5;

      // title
      ctx.fillStyle = col.ink; ctx.globalAlpha = 0.6;
      ctx.font = "10px " + SANS; ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText("MACHINERY — the CPS chain fugue steps through", padX, 6);
      ctx.globalAlpha = 1;

      chipRects = [];
      for (var i = 0; i < NODES; i++) {
        var x = padX + i * (chipW + gap);
        var q = { x: x, y: chipY, w: chipW, h: chipH };
        chipRects.push(q);
        drawChip(ctx, i, q, col, fs, now);
        if (i < NODES - 1) drawArrow(ctx, x + chipW, chipY + chipH / 2, gap, col, committed[i]);
      }

      // tallies row
      var ty = chipY + chipH + 20;
      var lp = logPrior(), ll = logLike();
      var tot = lp == null ? null : lp + (ll || 0);
      ctx.font = "600 " + (narrow ? 10 : 11.5) + "px " + MONO;
      ctx.textBaseline = "middle"; ctx.textAlign = "left";
      var segs = [
        ["log_prior ", lp, col.prior],
        ["log_lik ", ll, col.data],
        ["total ", tot, col.post]
      ];
      var cx = padX;
      for (var s = 0; s < segs.length; s++) {
        var lab = segs[s][0], val = segs[s][1], c = segs[s][2];
        var txt = lab + (val == null ? "—" : val.toFixed(2));
        ctx.fillStyle = c; ctx.globalAlpha = val == null ? 0.5 : 1;
        ctx.fillText(txt, cx, ty);
        cx += ctx.measureText(txt).width + (narrow ? 14 : 26);
        ctx.globalAlpha = 1;
      }
    }

    function chipRole(i) { return i === 0 ? "prior" : i === PURE ? "post" : "data"; }
    function chipLines(i) {
      if (i === 0) return ["SampleF64", "μ"];
      if (i === PURE) return ["Pure", "μ"];
      return ["ObserveF64", "y" + SUB[i - 1]];
    }
    function roleColor(col, role) {
      return role === "prior" ? col.prior : role === "data" ? col.data :
             role === "post" ? col.post : col.hot;
    }
    function drawChip(ctx, i, q, col, fs, now) {
      var role = chipRole(i), lines = chipLines(i);
      var isActive = (i === walkActive) || (i === cursor && cursor < NODES && !autoPlay && walkActive === -1);
      var pulse = (isActive && !loopApi.reduced) ? 0.5 + 0.5 * Math.sin(now / 200) : 1;
      var alpha = committed[i] ? 1 : (i === cursor ? 0.9 : 0.4);

      ctx.save();
      ctx.globalAlpha = alpha;
      rr(ctx, q.x, q.y, q.w, q.h, 6);
      ctx.fillStyle = col.panel; ctx.fill();
      var border = committed[i] ? roleColor(col, role) : col.grid;
      if (isActive) border = col.hot;
      ctx.lineWidth = isActive ? 2 : 1;
      ctx.strokeStyle = border; ctx.stroke();
      if (isActive && !loopApi.reduced) {
        ctx.globalAlpha = alpha * 0.5 * pulse;
        rr(ctx, q.x - 2, q.y - 2, q.w + 4, q.h + 4, 8);
        ctx.strokeStyle = col.hot; ctx.lineWidth = 1.5; ctx.stroke();
        ctx.globalAlpha = alpha;
      }
      var cx = q.x + q.w / 2;
      ctx.textAlign = "center";
      ctx.fillStyle = col.ink; ctx.globalAlpha = alpha * 0.8;
      ctx.font = (fs - 1.5) + "px " + MONO; ctx.textBaseline = "alphabetic";
      ctx.fillText(lines[0], cx, q.y + q.h / 2 - 1);
      ctx.globalAlpha = alpha;
      ctx.fillStyle = roleColor(col, role);
      ctx.font = "600 " + (fs + 1) + "px " + MONO;
      ctx.fillText(lines[1], cx, q.y + q.h / 2 + 13);
      ctx.restore();
    }
    function drawArrow(ctx, x, cy, gap, col, on) {
      ctx.save();
      ctx.globalAlpha = on ? 0.85 : 0.4;
      ctx.strokeStyle = col.flow; ctx.lineWidth = 1.3;
      var x1 = x + 2, x2 = x + gap - 2;
      ctx.beginPath(); ctx.moveTo(x1, cy); ctx.lineTo(x2, cy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x2 - 3, cy - 3); ctx.lineTo(x2, cy); ctx.lineTo(x2 - 3, cy + 3); ctx.stroke();
      ctx.restore();
    }

    function rr(ctx, x, y, w, h, r) {
      if (w <= 0 || h <= 0) { ctx.beginPath(); return; }
      r = Math.max(0, Math.min(r, w / 2, h / 2));
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }

    function render() {
      if (!hero || !strip || !loopApi) return;
      renderHero();
      renderStrip();
    }

    // ---------------------------------------------------------- interactions
    // Drag a data dot in the hero, via the shared pointer manager. fullCapture is
    // false: the hero is a tall (≈400px) chart whose bulk is non-interactive, so
    // the canvas stays touch-action:pan-y — a thumb swiping the chart body scrolls
    // the page, and only a pointerdown that actually lands on a dot (generous
    // coarse-pointer slop ≥22px, §A.2) is claimed so the drag never scrolls.
    var dragHandle = FV.drag(hero.el, {
      inflate: 14,
      fullCapture: false,
      hitTest: function (x, y, slop) {
        if (!heroLayout) return -1;
        var xs = heroLayout.xs, dy = heroLayout.dotY;
        var r = Math.max(14, slop), best = -1, bestD = r * r;
        for (var d = 0; d < N; d++) {
          var dx = xs(data[d]);
          var dd = (x - dx) * (x - dx) + (y - dy) * (y - dy);
          if (dd <= bestD) { bestD = dd; best = d; }
        }
        return best; // -1 = miss (page scrolls, rain untouched)
      },
      onStart: function () { rainTicks = null; },
      onDrag: function (idx, x) {
        if (!heroLayout) return;
        var v = heroLayout.xs.invert(x);
        v = Math.max(-3.5, Math.min(5.5, v));
        data[idx] = v;
        updateReadouts(); render();
      }
    });

    // Tap (not swipe) the strip to Step. A plain `click` fires only on a tap and
    // is suppressed by the browser after a scroll, so a thumb swiping over the
    // strip scrolls the page instead of stepping (§A.1 — no preventDefault here).
    strip.el.addEventListener("click", function () { onStep(); });

    // ---------------------------------------------------------------- boot
    var loopApi = FV.loop(root, tick);
    FV.onThemeChange(function () { render(); });

    // Pre-warm: start fully interpreted — one full performance already on the strip
    // (filled tallies, coral tick present). This is also the reduced-motion frame.
    runAllInstant();
    // autoplay: gently reloop the walk when motion is allowed. runAllInstant left
    // the chain at Pure, so the first thing the loop does is hold on the completed
    // performance, then walk a fresh one — never a dead canvas.
    if (!loopApi.reduced) {
      autoPlay = true; holdClock = 0; stepClock = 0;
      updatePlayLabel();
      loopApi.play();
    }
  });
})();
