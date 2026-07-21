/*
 * docs/viz/anatomy.js — "Anatomy of a Probabilistic Program"
 *
 * The coin-flip Bayes loop, fully touchable. Prior Beta(alpha, beta) in blue,
 * data as clickable H/T coin chips in yellow, posterior Beta(alpha+h, beta+t)
 * in green — the literal product blue x yellow = green, normalizing as you
 * play. Conjugacy is what the widget exploits; the page's Rust runs fugue's MH
 * and lands on the same number.
 *
 * v2 upgrades (data-first law):
 *   - Bigger hero (400px).
 *   - Posterior GHOST TRAIL: when the data or prior changes, the previous
 *     green curve is kept as a fading ghost, so "updating in real time" leaves
 *     a visible trail of learning.
 *   - Sequential-updating REPLAY: press Replay (or Step) and the flips enter
 *     one at a time, the posterior morphing per flip — Bayesian updating, live.
 *   - Coin chips FLIP with a subtle squash animation on toggle.
 *
 * Self-contained IIFE. Consumes window.FugueViz (loaded first per book.toml).
 */
(function () {
  "use strict";
  if (typeof window === "undefined" || !window.FugueViz) return;
  var FV = window.FugueViz;

  // ---- Math beyond FugueViz: regularized incomplete beta + Beta quantiles ----
  // I_x(a,b) via the Lentz continued fraction (Numerical Recipes betacf/betai).
  // Needed for the 90% credible interval readout; FugueViz has no CDF/quantile.
  function betacf(x, a, b) {
    var MAXIT = 200, EPS = 3e-12, FPMIN = 1e-300;
    var qab = a + b, qap = a + 1, qam = a - 1;
    var c = 1, d = 1 - (qab * x) / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    var h = d, m, m2, aa, del;
    for (m = 1; m <= MAXIT; m++) {
      m2 = 2 * m;
      aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = -((a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; del = d * c; h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }
  function betainc(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    var lbeta = FV.lgamma(a) + FV.lgamma(b) - FV.lgamma(a + b);
    var front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lbeta);
    if (x < (a + 1) / (a + b + 2)) return (front * betacf(x, a, b)) / a;
    return 1 - (front * betacf(1 - x, b, a)) / b;
  }
  function betaQuantile(p, a, b) {
    if (p <= 0) return 0;
    if (p >= 1) return 1;
    var lo = 0, hi = 1, mid;
    for (var i = 0; i < 80; i++) {
      mid = 0.5 * (lo + hi);
      if (betainc(mid, a, b) < p) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
  }
  function betaMode(a, b) {
    // Interior mode exists only for a>1 and b>1; otherwise mass piles at an edge.
    if (a > 1 && b > 1) return (a - 1) / (a + b - 2);
    if (a <= 1 && b > 1) return 0;
    if (a > 1 && b <= 1) return 1;
    return -1; // a<=1 && b<=1: bimodal at both edges — no single interior mode
  }

  var START_DATA = [true, false, true, true, false, true, true, false, true, true];

  FV.register("anatomy", function (root, FV) {
    // ---- DOM shell: controls / canvas / readouts / instruction / hint --------
    var controls = document.createElement("div");
    controls.className = "fv-controls";
    root.appendChild(controls);

    var canvasWrap = document.createElement("div");
    root.appendChild(canvasWrap);

    var instruction = document.createElement("div");
    instruction.className = "fv-instruction";
    instruction.textContent = "Click a coin to flip it. Press Replay to watch the posterior update one flip at a time. Scrub α and β in the text below.";
    root.appendChild(instruction);

    var readouts = document.createElement("div");
    readouts.className = "fv-readouts";
    root.appendChild(readouts);

    var hint = document.createElement("div");
    hint.className = "fv-hint";
    hint.textContent = "press Replay — the green curve walks from the prior to the posterior, one flip at a time, leaving a fading trail of every belief it held on the way.";
    root.appendChild(hint);

    // ---- State ---------------------------------------------------------------
    var seed = parseInt(root.getAttribute("data-seed"), 10);
    if (!isFinite(seed)) seed = 11;
    var alpha = 2, beta = 2;
    // Curated starting data: 7 heads, 3 tails (mirrors the page's Rust example).
    var data = START_DATA.slice();
    var revealCount = data.length;   // how many flips are currently "live"
    var showLik = true;
    var chips = [];                  // hit-test rects, CSS px, filled each draw

    var ghosts = [];                 // {a, b, life} fading previous posteriors
    var chipAnims = {};              // idx -> {kind:'flip'|'enter', t, dur, from}
    var playMode = false;            // auto sequential-updating replay running
    var revealTimer = 0;
    // Cached 90% credible bounds — recomputed only when the posterior TARGET
    // changes (updateReadouts), NOT per animation frame. betaQuantile is a
    // Lentz continued fraction inside an 80-step bisection; recomputing it twice
    // every rAF during a replay tween was a needless per-frame hot spot (§A.3).
    var credLo = 0, credHi = 1;

    function activeN() { var n = Math.min(revealCount, data.length); return n < 0 ? 0 : n; }
    function activeHeads() { var n = 0, m = activeN(); for (var i = 0; i < m; i++) if (data[i]) n++; return n; }
    function activeTails() { return activeN() - activeHeads(); }

    // tweened posterior params + their targets
    var targetA = alpha + activeHeads(), targetB = beta + activeTails();
    var animA = targetA, animB = targetB;

    // ---- Canvas (bigger hero per v2) -----------------------------------------
    var cv = FV.canvas(canvasWrap, { height: 400, onResize: function () { if (cv) draw(); } });

    // ---- Loop: drives posterior tween, ghost fade, chip flips, replay --------
    var loopApi = FV.loop(root, tick);

    function tweenSettled() {
      return Math.abs(targetA - animA) < 1e-3 && Math.abs(targetB - animB) < 1e-3;
    }

    function tick(dt) {
      if (dt > 0.1) dt = 0.1; // clamp long frames (tab refocus etc.)
      var active = false;

      // --- Sequential replay: reveal one flip on a timer -------------------
      if (playMode) {
        revealTimer -= dt;
        if (revealTimer <= 0 && revealCount < data.length) {
          revealCount++;
          chipAnims[revealCount - 1] = { kind: "enter", t: 0, dur: 0.4 };
          revealTimer = 0.5;
          setTarget(true, true); // ghost the previous posterior, tween to new
        }
        if (revealCount >= data.length && tweenSettled() && ghosts.length === 0) {
          playMode = false;
          syncPlayLabel();
        } else {
          active = true;
        }
      }

      // --- Posterior tween -------------------------------------------------
      var rate = 1 - Math.exp(-dt * 11);
      animA += (targetA - animA) * rate;
      animB += (targetB - animB) * rate;
      if (!tweenSettled()) active = true;
      else { animA = targetA; animB = targetB; }

      // --- Ghost fade ------------------------------------------------------
      if (ghosts.length) {
        for (var gi = ghosts.length - 1; gi >= 0; gi--) {
          ghosts[gi].life -= dt * 0.9;
          if (ghosts[gi].life <= 0) ghosts.splice(gi, 1);
        }
        if (ghosts.length) active = true;
      }

      // --- Chip flip / enter animations ------------------------------------
      for (var key in chipAnims) {
        if (!chipAnims.hasOwnProperty(key)) continue;
        var ca = chipAnims[key];
        ca.t += dt / (ca.dur || 0.4);
        if (ca.t >= 1) delete chipAnims[key];
        else active = true;
      }

      draw();
      if (!active) loopApi.pause();
    }

    function pushGhost() {
      ghosts.push({ a: targetA, b: targetB, life: 1 });
      while (ghosts.length > 6) ghosts.shift();
    }

    // Recompute the posterior target from the currently-live flips. `animate`
    // tweens (unless reduced motion); `ghost` leaves the old curve fading.
    function setTarget(animate, ghost) {
      if (ghost) pushGhost();
      targetA = alpha + activeHeads();
      targetB = beta + activeTails();
      updateReadouts();
      if (!animate || loopApi.reduced) {
        animA = targetA; animB = targetB;
        draw();
      } else {
        loopApi.play();
      }
    }

    // ---- Readouts ------------------------------------------------------------
    var roData = FV.readout(readouts, { label: "Data (H / T)" });
    var roMean = FV.readout(readouts, { label: "Posterior mean" });
    var roCI = FV.readout(readouts, { label: "90% credible" });
    var roMap = FV.readout(readouts, { label: "MAP" });
    function updateReadouts() {
      var h = activeHeads(), t = activeTails();
      var a = alpha + h, b = beta + t;
      roData.set(h + " H / " + t + " T", "data");
      roMean.set((a / (a + b)).toFixed(3), "post");
      var lo = betaQuantile(0.05, a, b), hi = betaQuantile(0.95, a, b);
      credLo = lo; credHi = hi; // cache for draw()'s credible band (§A.3)
      roCI.set("[" + lo.toFixed(2) + ", " + hi.toFixed(2) + "]", "post");
      var mode = betaMode(a, b);
      roMap.set(mode >= 0 ? mode.toFixed(3) : "—", "hot");
    }

    // ---- Replay controls -----------------------------------------------------
    function startReplay() {
      if (loopApi.reduced) {
        // Reduced motion: no autoplay — just show the full posterior.
        playMode = false;
        revealCount = data.length;
        ghosts = []; chipAnims = {};
        setTarget(false, false);
        return;
      }
      playMode = true;
      ghosts = []; chipAnims = {};
      revealCount = 0;
      animA = alpha; animB = beta; targetA = alpha; targetB = beta; // start at prior
      revealTimer = 0.35;
      updateReadouts();
      loopApi.play();
      syncPlayLabel();
    }
    function pauseReplay() {
      playMode = false;
      loopApi.pause();
      syncPlayLabel();
    }
    function toggleReplay() {
      if (playMode && loopApi.playing) pauseReplay();
      else startReplay();
    }
    // Manual sequential updating — one flip per press (also the reduced-motion path).
    function stepReveal() {
      if (revealCount >= data.length) {
        // Restart the walk from the prior.
        ghosts = []; chipAnims = {};
        revealCount = 0;
        animA = alpha; animB = beta; targetA = alpha; targetB = beta;
        updateReadouts();
      }
      playMode = false;
      revealCount++;
      chipAnims[revealCount - 1] = { kind: "enter", t: 0, dur: 0.4 };
      setTarget(true, true);
      syncPlayLabel();
    }

    // ---- Controls ------------------------------------------------------------
    var btnRoot = FV.buttons(controls, [
      { label: "Replay", title: "Replay the flips one at a time — watch the posterior update", onClick: toggleReplay, primary: true },
      { label: "Step", title: "Reveal one more flip (one Bayesian update)", onClick: stepReveal },
      { label: "+ Heads", title: "Add a heads flip", onClick: function () { playMode = false; data.push(true); revealCount = data.length; setTarget(true, true); syncPlayLabel(); } },
      { label: "+ Tails", title: "Add a tails flip", onClick: function () { playMode = false; data.push(false); revealCount = data.length; setTarget(true, true); syncPlayLabel(); } },
      { label: "Remove", title: "Remove the last flip", onClick: function () { if (data.length) { playMode = false; data.pop(); revealCount = data.length; setTarget(true, true); syncPlayLabel(); } } },
      { label: "Deal", title: "Deal 12 fresh flips from the current seed (reproducible)", onClick: function () { deal(seedScrub ? seedScrub.fvGet() : seed); } },
      { label: "Reset", title: "Restore the prior and the starting data", onClick: reset }
    ]);
    var playBtn = btnRoot.fvButtons.Replay;
    function syncPlayLabel() {
      if (playBtn) playBtn.textContent = (playMode && loopApi.playing) ? "Pause" : "Replay";
    }

    FV.toggle(controls, {
      label: "show likelihood",
      value: showLik,
      onChange: function (v) { showLik = v; draw(); }
    });

    function deal(s) {
      playMode = false;
      var rand = FV.rng((s >>> 0) || 11);
      var bias = 0.62, n = 12, out = [];
      for (var i = 0; i < n; i++) out.push(rand() < bias);
      data = out;
      revealCount = data.length;
      chipAnims = {};
      setTarget(true, true);
      syncPlayLabel();
    }
    function reset() {
      playMode = false;
      alpha = 2; beta = 2;
      data = START_DATA.slice();
      revealCount = data.length;
      ghosts = []; chipAnims = {};
      if (alphaScrub) alphaScrub.fvSet(2);
      if (betaScrub) betaScrub.fvSet(2);
      setTarget(false, false);
      syncPlayLabel();
    }

    // ---- Prose scrubs (alpha, beta, seed live inside the sentences) ----------
    var alphaScrub = null, betaScrub = null, seedScrub = null;
    var aEl = document.getElementById("fv-anatomy-alpha");
    var bEl = document.getElementById("fv-anatomy-beta");
    var sEl = document.getElementById("fv-anatomy-seed");
    if (aEl) alphaScrub = FV.scrub(aEl, {
      min: 0.5, max: 20, step: 0.5, value: alpha,
      fmt: function (v) { return String(v); },
      onInput: function (v) { playMode = false; alpha = v; setTarget(false, false); syncPlayLabel(); } // direct manipulation → snap
    });
    if (bEl) betaScrub = FV.scrub(bEl, {
      min: 0.5, max: 20, step: 0.5, value: beta,
      fmt: function (v) { return String(v); },
      onInput: function (v) { playMode = false; beta = v; setTarget(false, false); syncPlayLabel(); }
    });
    if (sEl) seedScrub = FV.scrub(sEl, {
      min: 1, max: 99, step: 1, value: seed,
      fmt: function (v) { return String(v); },
      onInput: function (v) { seed = v; deal(v); } // reproducible re-deal
    });

    // ---- Pointer: tap a coin chip to flip it (shared drag manager) -----------
    // Chips are tap-targets, not drags, but FV.drag gives us exactly the touch
    // semantics §A wants: it claims the gesture (setPointerCapture + preventDefault)
    // ONLY when the pointerdown actually lands on a chip — so a thumb on a chip
    // flips it and never scrolls, while a thumb anywhere else (the whole 400px
    // plot region) still scrolls the page. fullCapture:false keeps the canvas at
    // touch-action:pan-y so that ambient plot area stays scrollable (§A.1). The
    // hit radius comes from `slop`, which the manager inflates to >=22 CSS px on
    // coarse pointers (§A.2) while the drawn chip stays a crisp 12px.
    function chipHitTest(x, y, slop) {
      var best = null, bestD = Infinity;
      for (var i = 0; i < chips.length; i++) {
        var ch = chips[i];
        var dx = x - ch.x, dy = y - ch.y, d = dx * dx + dy * dy;
        var rr = Math.max(ch.r, slop); // coarse pointers -> >=22px pick radius
        if (d <= rr * rr && d < bestD) { best = ch; bestD = d; }
      }
      return best; // truthy chip object on a hit, null on a miss (index 0 is a
                   // valid chip, so we must return the object, never the index)
    }
    function flipChip(ch) {
      playMode = false;
      var was = data[ch.i];
      data[ch.i] = !was;
      chipAnims[ch.i] = { kind: "flip", t: 0, dur: 0.4, from: was };
      setTarget(true, true);
      syncPlayLabel();
    }
    FV.drag(cv.el, {
      fullCapture: false,           // plot area must still scroll on a swipe
      hitTest: chipHitTest,
      onStart: function (ch) { flipChip(ch); } // tap = flip on pointerdown
    });

    // ---- Drawing -------------------------------------------------------------
    function pdf(x, a, b) {
      var lp = FV.dist.beta.logpdf(x, a, b);
      return lp === -Infinity ? 0 : Math.exp(lp);
    }
    function sampleCurve(a, b, xs) {
      var pts = [];
      for (var i = 0; i < xs.length; i++) pts.push([xs[i], pdf(xs[i], a, b)]);
      return pts;
    }

    function draw() {
      var w = cv.w, h = cv.h, ctx = cv.ctx;
      var colors = FV.theme().colors;
      cv.clear();

      var padL = 8, padR = 8;
      var plotLeft = padL, plotRight = w - padR;
      var plotW = plotRight - plotLeft;

      // --- DATA strip: coin chips (yellow H, hollow T), clickable -------------
      chips = [];
      ctx.save();
      ctx.font = "600 10px var(--mono-font, monospace)";
      ctx.fillStyle = colors.ink;
      ctx.globalAlpha = 0.65;
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText("DATA", plotLeft, 6);
      ctx.restore();

      var r = 12, gap = 6, rowH = 2 * r + gap;
      var perRow = Math.max(1, Math.floor((plotW + gap) / (2 * r + gap)));
      var n = data.length;
      var shown = activeN();
      var rowsNeeded = Math.max(1, Math.ceil(n / perRow));
      var rowsShown = Math.min(rowsNeeded, 3);
      var chipsTop = 22;
      for (var idx = 0; idx < shown; idx++) {
        var rIdx = Math.floor(idx / perRow);
        if (rIdx >= 3) break; // clip overflow beyond 3 rows
        var cIdx = idx % perRow;
        var cx = plotLeft + r + cIdx * (2 * r + gap);
        var cy = chipsTop + r + rIdx * rowH;
        var ca = chipAnims[idx];
        var sx = 1, sy = 1, alph = 1, face = data[idx];
        if (ca) {
          if (ca.kind === "flip") {
            sx = Math.abs(Math.cos(Math.PI * ca.t)); // squash through the flip
            face = ca.t < 0.5 ? ca.from : data[idx];
          } else if (ca.kind === "enter") {
            var e = ca.t;
            alph = e;
            sx = sy = 0.5 + 0.5 * e; // pop in
          }
        }
        drawChip(ctx, cx, cy, r, face, colors, sx, sy, alph);
        chips.push({ x: cx, y: cy, r: r + 2, i: idx });
      }
      if (rowsNeeded > 3) {
        ctx.save();
        ctx.fillStyle = colors.ink; ctx.globalAlpha = 0.55;
        ctx.font = "11px var(--mono-font, monospace)";
        ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText("+" + (n - 3 * perRow) + " more", plotLeft, chipsTop + r + 3 * rowH - rowH / 2);
        ctx.restore();
      }

      // --- Plot region -------------------------------------------------------
      var plotTop = chipsTop + rowsShown * rowH + 12;
      var plotBottom = h - 22;
      if (plotBottom - plotTop < 60) plotTop = plotBottom - 60;

      var xs = [];
      var N = 180;
      for (var i = 0; i <= N; i++) xs.push(i / N);

      var h0 = activeHeads(), t0 = activeTails();
      var priorPts = sampleCurve(alpha, beta, xs);
      var likPts = sampleCurve(h0 + 1, t0 + 1, xs);   // normalized likelihood = Beta(h+1,t+1)
      var postPts = sampleCurve(animA, animB, xs);

      var ymax = 0;
      function accum(pts) { for (var i = 0; i < pts.length; i++) { var v = pts[i][1]; if (isFinite(v) && v > ymax) ymax = v; } }
      accum(priorPts); accum(postPts);
      if (showLik && h0 + t0 > 0) accum(likPts);
      if (!(ymax > 0)) ymax = 1;
      ymax *= 1.12;

      var xsc = FV.scale([0, 1], [plotLeft, plotRight]);
      var ysc = FV.scale([0, ymax], [plotBottom, plotTop]);

      FV.axes(ctx, { x: plotLeft, y: plotTop, w: plotW, h: plotBottom - plotTop, xscale: xsc, theme: FV.theme() });

      function toPix(pts) {
        var out = [];
        for (var i = 0; i < pts.length; i++) {
          var v = pts[i][1];
          out.push([xsc(pts[i][0]), isFinite(v) ? ysc(Math.min(v, ymax)) : NaN]);
        }
        return out;
      }
      function fillUnder(pts, color, alpha) {
        var px = toPix(pts);
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(px[0][0], ysc(0));
        for (var i = 0; i < px.length; i++) if (isFinite(px[i][1])) ctx.lineTo(px[i][0], px[i][1]);
        ctx.lineTo(px[px.length - 1][0], ysc(0));
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      }

      var a2 = animA, b2 = animB;

      // Posterior credible band (green, faint) under the posterior curve.
      // Bounds are cached (updateReadouts) — targetA/targetB only change on a
      // user action, so there is no need to re-solve the quantile every frame.
      var qLo = credLo, qHi = credHi;
      ctx.save();
      ctx.globalAlpha = 0.16;
      ctx.fillStyle = colors.post;
      ctx.beginPath();
      var started = false;
      for (var i = 0; i < postPts.length; i++) {
        var xv = postPts[i][0];
        if (xv < qLo || xv > qHi) continue;
        var pxx = xsc(xv), pyy = ysc(Math.min(postPts[i][1], ymax));
        if (!started) { ctx.moveTo(pxx, ysc(0)); ctx.lineTo(pxx, pyy); started = true; }
        else ctx.lineTo(pxx, pyy);
      }
      if (started) { ctx.lineTo(xsc(Math.min(qHi, 1)), ysc(0)); ctx.closePath(); ctx.fill(); }
      ctx.restore();

      // Posterior GHOST TRAIL: fading green curves of prior beliefs, behind.
      for (var gi = 0; gi < ghosts.length; gi++) {
        var g = ghosts[gi];
        ctx.save();
        ctx.globalAlpha = 0.18 * Math.max(0, g.life);
        FV.curve(ctx, toPix(sampleCurve(g.a, g.b, xs)), { color: colors.post, width: 1.5 });
        ctx.restore();
      }

      // Prior (blue), likelihood (yellow, dashed), posterior (green) overlaid.
      fillUnder(priorPts, colors.prior, 0.10);
      FV.curve(ctx, toPix(priorPts), { color: colors.prior, width: 2 });
      if (showLik && h0 + t0 > 0) {
        FV.curve(ctx, toPix(likPts), { color: colors.data, width: 2, dash: [5, 4] });
      }
      fillUnder(postPts, colors.post, 0.20);
      FV.curve(ctx, toPix(postPts), { color: colors.post, width: 2.5 });

      // Posterior mean tick + MAP marker (coral vertical line to the mode).
      var meanX = targetA / (targetA + targetB);
      drawVLine(ctx, xsc(meanX), plotTop, plotBottom, colors.post, 1.5, [2, 3]);
      var mode = betaMode(a2, b2);
      if (mode > 0 && mode < 1) {
        var modePx = xsc(mode);
        var modePy = ysc(Math.min(pdf(mode, a2, b2), ymax));
        drawVLine(ctx, modePx, modePy, plotBottom, colors.hot, 2, null);
        ctx.save();
        ctx.fillStyle = colors.hot;
        ctx.beginPath(); ctx.arc(modePx, modePy, 3.5, 0, 2 * Math.PI); ctx.fill();
        ctx.restore();
      }

      // Colored legend: prior x likelihood -> posterior (the color algebra).
      drawLegend(ctx, plotLeft + 4, plotTop + 4, colors);

      // x-axis label
      ctx.save();
      ctx.fillStyle = colors.ink; ctx.globalAlpha = 0.6;
      ctx.font = "11px var(--mono-font, monospace)";
      ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      ctx.fillText("bias  p", (plotLeft + plotRight) / 2, h - 4);
      ctx.restore();
    }

    function drawChip(ctx, cx, cy, r, isHead, colors, sx, sy, alph) {
      sx = sx == null ? 1 : sx;
      sy = sy == null ? 1 : sy;
      alph = alph == null ? 1 : alph;
      ctx.save();
      ctx.globalAlpha *= alph;
      ctx.translate(cx, cy);
      ctx.scale(sx, sy);
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, 2 * Math.PI);
      if (isHead) {
        ctx.fillStyle = colors.data;
        ctx.fill();
        ctx.fillStyle = "#1f2328";
        ctx.font = "600 12px var(--mono-font, monospace)";
      } else {
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = colors.data;
        ctx.globalAlpha = 0.7 * alph;
        ctx.stroke();
        ctx.globalAlpha = alph;
        ctx.fillStyle = colors.data;
        ctx.font = "12px var(--mono-font, monospace)";
      }
      // Only render the glyph when the chip isn't edge-on (avoids stretched text).
      if (Math.abs(sx) > 0.25) {
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(isHead ? "H" : "T", 0, 0.5);
      }
      ctx.restore();
    }

    function drawVLine(ctx, x, y0, y1, color, width, dash) {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(x, y0);
      ctx.lineTo(x, y1);
      ctx.stroke();
      ctx.restore();
    }

    function drawLegend(ctx, x, y, colors) {
      ctx.save();
      ctx.font = "600 11px var(--mono-font, monospace)";
      ctx.textBaseline = "top";
      ctx.textAlign = "left";
      var parts = [
        ["prior", colors.prior],
        [" × ", colors.ink],
        ["likelihood", colors.data],
        [" → ", colors.ink],
        ["posterior", colors.post]
      ];
      var cx = x;
      for (var i = 0; i < parts.length; i++) {
        ctx.fillStyle = parts[i][1];
        ctx.globalAlpha = parts[i][1] === colors.ink ? 0.6 : 1;
        ctx.fillText(parts[i][0], cx, y);
        cx += ctx.measureText(parts[i][0]).width;
      }
      ctx.restore();
    }

    // ---- Theme + init --------------------------------------------------------
    FV.onThemeChange(function () { draw(); });
    updateReadouts();
    // Pre-warm: dealt data with a settled posterior painted synchronously — this is
    // also the reduced-motion static frame (never an empty axis).
    draw();
    // autoplay: replay the sequential Bayesian updating the moment the widget scrolls
    // into view. startReplay no-ops the animation under reduced motion — it just
    // re-shows the settled posterior we already painted — so it is safe either way.
    startReplay();
  });
})();
