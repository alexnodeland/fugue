// docs/viz/minis.js — page-support mini-figures (ambient, "clear & smooth").
//
// Eight small, param-driven figures woven into the six explorable hero pages'
// explanation sections, embeddable via
//   <div class="fugue-explorable fv-inline" data-viz="NAME" data-...></div>
// Same conventions as inline.js: an ambient autoplay loop that advances discrete
// STATE at a few Hz with tweened rendering between states; one unobtrusive
// pause/play glyph (top-right); optional one-line caption from data-caption;
// seeded via data-seed (default 11); all math via FugueViz; theme-aware; the
// color algebra everywhere (data = yellow, prior = blue, posterior = green,
// current = coral, structure = violet).
//
// SMOOTHNESS (v4 §A): these are AMBIENT figures. They must never eat page
// scroll. Only `sigma-sweep` has a draggable control (a coral σ marker), and it
// claims the gesture ONLY when a pointerdown actually lands on the marker
// (coarse-pointer hit radius >= 22px) via setPointerCapture; its canvas uses
// touch-action:pan-y so a vertical swipe always scrolls the page. Every other
// mini attaches no pointer handlers at all, leaving the canvas fully scrollable.
//
// The math is real: real random-walk Metropolis + autocorrelation/ESS, real
// leapfrog (velocity Verlet) energy accounting, real Beta conjugate updates,
// real systematic-resample-style ESS. Known-value checks live in the agent
// report (verified with docs/viz/minis.verify.js against fugue-viz.js).
//
// Self-contained IIFE; assumes fugue-viz.js has loaded first (book.toml order).
// The mount scaffold is COPIED from inline.js's pattern (not imported), then
// hardened for touch per §A.
(function () {
  "use strict";
  if (typeof window === "undefined" || !window.FugueViz) return;
  var FV = window.FugueViz;

  // ==========================================================================
  // Small generic helpers
  // ==========================================================================

  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function nextSeed(s) { return (Math.imul(s, 1664525) + 1013904223) >>> 0; }
  function fmtNum(v) {
    if (v == null || !isFinite(v)) return "—";
    var a = Math.abs(v);
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    return v.toFixed(2);
  }

  // ---- canvas drawing primitives (all take resolved theme colors) -----------

  function baseline(g, x0, x1, y, c) {
    g.save(); g.strokeStyle = c.ink; g.globalAlpha = 0.22; g.lineWidth = 1;
    g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke(); g.restore();
  }
  function label(g, txt, x, y, c) {
    g.save(); g.fillStyle = c.ink; g.globalAlpha = 0.7;
    g.font = "11px var(--mono-font, monospace)"; g.textBaseline = "top"; g.textAlign = "left";
    g.fillText(txt, x, y); g.restore();
  }
  function labelRight(g, txt, x, y, c, role) {
    g.save(); g.fillStyle = role ? c[role] : c.ink; g.globalAlpha = role ? 0.95 : 0.7;
    g.font = "11px var(--mono-font, monospace)"; g.textBaseline = "top"; g.textAlign = "right";
    g.fillText(txt, x, y); g.restore();
  }
  function fillUnder(g, pts, y0, col, alpha) {
    if (!pts.length) return;
    g.save(); g.globalAlpha = alpha; g.fillStyle = col; g.beginPath();
    var started = false, i;
    for (i = 0; i < pts.length; i++) {
      if (!isFinite(pts[i][1])) continue;
      if (!started) { g.moveTo(pts[i][0], y0); g.lineTo(pts[i][0], pts[i][1]); started = true; }
      else g.lineTo(pts[i][0], pts[i][1]);
    }
    if (started) { g.lineTo(pts[pts.length - 1][0], y0); g.closePath(); g.fill(); }
    g.restore();
  }
  function densCurve(dom, f, xs, ys, n) {
    var pts = [], i, x;
    for (i = 0; i <= n; i++) { x = dom[0] + (dom[1] - dom[0]) * i / n; pts.push([xs(x), ys(f(x))]); }
    return pts;
  }
  function roundRect(g, x, y, w, h, r) {
    g.beginPath();
    g.moveTo(x + r, y);
    g.arcTo(x + w, y, x + w, y + h, r);
    g.arcTo(x + w, y + h, x, y + h, r);
    g.arcTo(x, y + h, x, y, r);
    g.arcTo(x, y, x + w, y, r);
    g.closePath();
  }

  // ==========================================================================
  // Shared statistical math (real formulas; verified in the agent report)
  // ==========================================================================

  // Random-walk Metropolis targeting the standard normal N(0,1). Returns the
  // chain and the empirical acceptance rate. Deterministic given `rng`.
  function mhChain(rng, sigma, n, x0) {
    var x = (x0 == null) ? 0 : x0, out = [x], accepts = 0, i;
    for (i = 1; i < n; i++) {
      var prop = x + sigma * FV.randn(rng);
      // log target ratio: -0.5(prop^2 - x^2)
      if (Math.log(rng()) < -0.5 * (prop * prop - x * x)) { x = prop; accepts++; }
      out.push(x);
    }
    return { chain: out, accept: accepts / (n - 1) };
  }

  // Sample autocorrelation rho_k, k = 0..K (biased/1-over-n estimator, the one
  // used for ESS). rho_0 = 1 by construction.
  function autocorr(x, K) {
    var n = x.length, mean = 0, i, k;
    for (i = 0; i < n; i++) mean += x[i];
    mean /= n;
    var c0 = 0;
    for (i = 0; i < n; i++) { var d = x[i] - mean; c0 += d * d; }
    c0 /= n;
    var rho = [];
    for (k = 0; k <= K; k++) {
      var ck = 0;
      for (i = 0; i < n - k; i++) ck += (x[i] - mean) * (x[i + k] - mean);
      ck /= n;
      rho[k] = c0 > 0 ? ck / c0 : (k === 0 ? 1 : 0);
    }
    return rho;
  }

  // Effective sample size via the initial-positive-sequence truncation of the
  // autocorrelation sum: ESS = N / (1 + 2 * sum_{k>=1, rho_k>0} rho_k).
  function essFromChain(x) {
    var K = Math.min(x.length - 1, 120);
    var rho = autocorr(x, K), s = 0, k;
    for (k = 1; k <= K; k++) { if (rho[k] <= 0) break; s += rho[k]; }
    var tau = 1 + 2 * s;
    return x.length / (tau > 1e-9 ? tau : 1e-9);
  }

  // One full leapfrog / velocity-Verlet step for Hamiltonian H = U(q) + p^2/2.
  function vstep(q, p, eps, dU) {
    var ph = p - 0.5 * eps * dU(q);
    var qn = q + eps * ph;
    var pn = ph - 0.5 * eps * dU(qn);
    return [qn, pn];
  }
  // |ΔH| after L leapfrog steps of size eps from (q0,p0) on Hamiltonian (U,dU).
  function leapfrogDeltaH(q0, p0, eps, L, U, dU) {
    var q = q0, p = p0, i, H0 = U(q0) + 0.5 * p0 * p0;
    for (i = 0; i < L; i++) { var r = vstep(q, p, eps, dU); q = r[0]; p = r[1]; }
    var H1 = U(q) + 0.5 * p * p;
    return Math.abs(H1 - H0);
  }

  // Normalized effective sample fraction of a positive weight vector:
  // ESS/N = (sum w)^2 / (N * sum w^2). Uniform -> 1; degenerate -> 1/N.
  function essFraction(w) {
    var n = w.length, s1 = 0, s2 = 0, i;
    for (i = 0; i < n; i++) { s1 += w[i]; s2 += w[i] * w[i]; }
    if (s2 <= 0) return 0;
    return (s1 * s1) / (n * s2);
  }

  // ==========================================================================
  // The mount scaffold — one ambient loop, glyph, caption, reduced-motion frame.
  // COPIED from inline.js's pattern; pointer handling hardened for touch (§A):
  // hit-gated pointer capture, pan-y touch-action, coarse hit radius.
  // ==========================================================================

  function mount(root, spec) {
    var seed = parseInt(root.getAttribute("data-seed"), 10);
    if (!(seed >= 0)) seed = 11;

    // A resize resets the canvas backing store. The ambient loop repaints every
    // frame so the clear is invisible while playing; a paused / reduced-motion
    // widget renders once, so we repaint on resize to avoid a blank canvas.
    var ready = false;
    var cv = FV.canvas(root, { height: spec.height || 150, onResize: function () { if (ready) renderFrame(); } });
    var g = cv.ctx;

    var capText = root.getAttribute("data-caption");
    if (capText) {
      var cap = document.createElement("div");
      cap.className = "fv-caption";
      cap.textContent = capText;
      root.appendChild(cap);
    }

    var S = { seed: seed, reloopSeed: seed, rng: FV.rng(seed) };
    spec.build(S, FV);

    var hz = spec.hz || 4, interval = 1 / hz, acc = 0, T = 1;
    function colors() { return FV.theme().colors; }
    function renderFrame() {
      cv.clear();
      try { spec.render(g, S, cv.w, cv.h, T, colors(), FV); } catch (e) { /* keep the page quiet */ }
    }

    var loopApi = FV.loop(root, function (dt) {
      if (dt > 0.1) dt = 0.1;
      acc += dt;
      while (acc >= interval) { acc -= interval; spec.advance(S, FV); }
      T = acc / interval; if (T > 1) T = 1;
      renderFrame();
    });

    // pause/play glyph
    var glyph = document.createElement("button");
    glyph.type = "button";
    glyph.className = "fv-glyph";
    glyph.setAttribute("aria-label", "Pause or play this animation");
    function glyphUpdate() {
      glyph.textContent = loopApi.playing ? "‖" : "▶"; // ‖ / ▶
      glyph.title = loopApi.playing ? "Pause" : "Play";
    }
    glyph.addEventListener("click", function () {
      if (loopApi.playing) loopApi.pause(); else loopApi.play();
      glyphUpdate(); renderFrame();
    });
    root.appendChild(glyph);

    // Pointer interaction — ONLY when the widget declares spec.pointer. Ambient
    // minis omit it entirely and never touch the canvas's default scroll.
    if (spec.pointer) {
      var coarse = false;
      try { coarse = !!(window.matchMedia && window.matchMedia("(pointer: coarse)").matches); } catch (e) { coarse = false; }
      // pan-y: a vertical swipe always scrolls the page; horizontal motion is
      // ours to interpret (the σ marker rides a horizontal axis).
      cv.el.style.touchAction = "pan-y";
      var dragging = false;
      function localXY(e) { var r = cv.el.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
      cv.el.addEventListener("pointerdown", function (e) {
        var p = localXY(e);
        // Claim the gesture only if the pointerdown actually grabs something.
        var hit = spec.hitTest ? spec.hitTest(S, p[0], p[1], cv.w, cv.h, coarse) : true;
        if (!hit) return; // let the page scroll / do nothing
        dragging = true;
        try { cv.el.setPointerCapture(e.pointerId); } catch (_) {}
        spec.pointer(S, p[0], p[1], cv.w, cv.h, "down", FV);
        renderFrame();
        if (e.cancelable) e.preventDefault();
      });
      cv.el.addEventListener("pointermove", function (e) {
        if (!dragging) return;
        var p = localXY(e);
        spec.pointer(S, p[0], p[1], cv.w, cv.h, "move", FV);
        renderFrame();
        if (e.cancelable) e.preventDefault();
      });
      function endDrag(e) {
        if (!dragging) return;
        dragging = false;
        try { cv.el.releasePointerCapture(e.pointerId); } catch (_) {}
        var p = localXY(e);
        spec.pointer(S, p[0], p[1], cv.w, cv.h, "up", FV);
        renderFrame();
      }
      cv.el.addEventListener("pointerup", endDrag);
      cv.el.addEventListener("pointercancel", endDrag);
      cv.el.style.cursor = "ew-resize";
    }

    FV.onThemeChange(function () { renderFrame(); });

    ready = true; // future resizes may now repaint the current frame
    if (loopApi.reduced) {
      // reduced motion: render a fully-formed static frame, never an empty axis.
      if (spec.staticFrame) spec.staticFrame(S, FV);
      else { var n = spec.settleN || 30; for (var i = 0; i < n; i++) spec.advance(S, FV); }
      T = 1; renderFrame();
      glyph.style.display = "none";
    } else {
      renderFrame();
      loopApi.play();
      glyphUpdate();
    }
  }

  // ==========================================================================
  // 1. acf-decay — one MH trace (top) + its autocorrelation bars ρ_k (bottom).
  //    data-sigma "0.05" | "0.4" | "3": small σ → slow ACF decay; good → fast.
  // ==========================================================================

  function acfDecay(root) {
    var sigma = parseFloat(root.getAttribute("data-sigma"));
    if (!(sigma > 0)) sigma = 0.4;
    var N = 130, K = 16, BATCH = 3;

    mount(root, {
      height: 150, hz: 6,
      staticFrame: function (S) { resetC(S); while (S.chain.length < N) grow(S); recompute(S); },
      build: function (S) { resetC(S); },
      advance: function (S) {
        if (S.chain.length < N) { for (var b = 0; b < BATCH; b++) if (S.chain.length < N) grow(S); recompute(S); }
        else { S.hold = (S.hold || 0) + 1; if (S.hold > 14) { S.reloopSeed = nextSeed(S.reloopSeed); S.rng = FV.rng(S.reloopSeed); resetC(S); } }
      },
      render: function (g, S, w, h, T, c) { drawAcf(g, S, w, h, c); }
    });

    function resetC(S) { S.chain = [0]; S.hold = 0; S.rho = [1]; }
    function grow(S) {
      var x = S.chain[S.chain.length - 1];
      var prop = x + sigma * FV.randn(S.rng);
      if (Math.log(S.rng()) < -0.5 * (prop * prop - x * x)) x = prop;
      S.chain.push(x);
    }
    function recompute(S) { S.rho = autocorr(S.chain, Math.min(K, S.chain.length - 1)); }

    function drawAcf(g, S, w, h, c) {
      var pad = { l: 10, r: 10, t: 14, b: 8 };
      var splitY = h * 0.5;
      // ---- top: the trace ----
      var xs = FV.scale([0, N], [pad.l, w - pad.r]);
      var ys = FV.scale([-3.4, 3.4], [splitY - 6, pad.t]);
      baseline(g, pad.l, w - pad.r, ys(0), c);
      var pts = [], i, ch = S.chain;
      for (i = 0; i < ch.length; i++) pts.push([xs(i), ys(clamp(ch[i], -3.4, 3.4))]);
      FV.curve(g, pts, { color: c.post, width: 1.2 });
      if (ch.length) {
        g.save(); g.fillStyle = c.hot;
        g.beginPath(); g.arc(xs(ch.length - 1), ys(clamp(ch[ch.length - 1], -3.4, 3.4)), 2.6, 0, 6.2832); g.fill(); g.restore();
      }
      label(g, "σ " + fmtNum(sigma) + " trace", pad.l, 1, c);
      // ---- bottom: ACF bars ρ_k (structure = violet) ----
      var bx = FV.scale([0, K], [pad.l + 4, w - pad.r]);
      var by = FV.scale([-0.25, 1], [h - pad.b, splitY + 8]);
      g.save(); g.strokeStyle = c.ink; g.globalAlpha = 0.22; g.lineWidth = 1;
      g.beginPath(); g.moveTo(pad.l, by(0)); g.lineTo(w - pad.r, by(0)); g.stroke(); g.restore();
      var bw = Math.max(3, (bx(1) - bx(0)) * 0.55);
      for (i = 0; i <= K && i < S.rho.length; i++) {
        var rv = S.rho[i], px = bx(i), py = by(clamp(rv, -0.25, 1));
        g.save(); g.globalAlpha = i === 0 ? 0.35 : 0.85; g.fillStyle = c.flow;
        g.fillRect(px - bw / 2, Math.min(py, by(0)), bw, Math.abs(py - by(0)));
        g.restore();
      }
      labelRight(g, "ρ_k", w - pad.r, splitY + 4, c, "flow");
    }
  }

  // ==========================================================================
  // 2. sigma-sweep — acceptance (green) & ESS (violet) over a log-σ sweep, the
  //    Goldilocks band shaded, a draggable coral σ marker. THE gesture mini.
  // ==========================================================================

  function sigmaSweep(root) {
    var NPTS = 26, LOSIG = 0.02, HISIG = 20, CHAINLEN = 520;

    mount(root, {
      height: 175, hz: 4,
      staticFrame: function (S) { buildSweep(S); S.reveal = NPTS; S.mi = Math.floor(NPTS * 0.62); },
      build: function (S) { buildSweep(S); S.reveal = 1; S.mi = NPTS * 0.62; S.phase = 0; },
      advance: function (S) {
        if (S.reveal < NPTS) S.reveal += 1;
        if (!S.touched) { S.phase += 0.05; S.mi = (NPTS - 1) * (0.5 + 0.42 * Math.sin(S.phase)); }
      },
      render: function (g, S, w, h, T, c) { drawSweep(g, S, w, h, c); },
      hitTest: function (S, x, y, w, h, coarse) {
        var mx = markerX(S, w);
        var r = coarse ? 24 : 14;
        return Math.abs(x - mx) <= r;
      },
      pointer: function (S, x, y, w, h, phase) {
        S.touched = true;
        S.dragging = (phase !== "up");
        var pad = geom(w);
        var frac = clamp((x - pad.l) / (pad.r - pad.l), 0, 1);
        S.mi = frac * (NPTS - 1);
      }
    });

    function geom(w) { return { l: 30, r: w - 12 }; }
    function markerX(S, w) {
      var pad = geom(w);
      return lerp(pad.l, pad.r, clamp(S.mi, 0, NPTS - 1) / (NPTS - 1));
    }
    function buildSweep(S) {
      S.sig = []; S.acc = []; S.ess = []; S.touched = false; S.dragging = false; S.hold = 0;
      var i, maxE = 1e-9;
      for (i = 0; i < NPTS; i++) {
        var f = i / (NPTS - 1);
        var sg = LOSIG * Math.pow(HISIG / LOSIG, f); // log-spaced
        var r = mhChain(S.rng, sg, CHAINLEN, 0);
        var e = essFromChain(r.chain) / CHAINLEN; // ESS fraction
        S.sig.push(sg); S.acc.push(r.accept); S.ess.push(e);
        if (e > maxE) maxE = e;
      }
      S.maxE = maxE;
      // Goldilocks band: contiguous σ where ESS >= 0.7 * peak.
      var lo = -1, hi = -1;
      for (i = 0; i < NPTS; i++) if (S.ess[i] >= 0.7 * maxE) { if (lo < 0) lo = i; hi = i; }
      S.bandLo = lo; S.bandHi = hi;
    }
    function drawSweep(g, S, w, h, c) {
      var pad = geom(w), top = 16, bot = h - 22;
      var xs = FV.scale([0, NPTS - 1], [pad.l, pad.r]);
      var ya = FV.scale([0, 1], [bot, top]);       // acceptance 0..1
      var ye = FV.scale([0, S.maxE * 1.1], [bot, top]); // ESS fraction
      // Goldilocks band
      if (S.bandLo >= 0) {
        g.save(); g.globalAlpha = 0.12; g.fillStyle = c.post;
        var xa = xs(Math.max(0, S.bandLo - 0.5)), xb = xs(Math.min(NPTS - 1, S.bandHi + 0.5));
        g.fillRect(xa, top, xb - xa, bot - top); g.restore();
      }
      baseline(g, pad.l, pad.r, bot, c);
      // curves, revealed left-to-right
      var nrev = Math.min(S.reveal, NPTS), i;
      var accPts = [], essPts = [];
      for (i = 0; i < nrev; i++) { accPts.push([xs(i), ya(S.acc[i])]); essPts.push([xs(i), ye(S.ess[i])]); }
      FV.curve(g, accPts, { color: c.post, width: 2 });
      FV.curve(g, essPts, { color: c.flow, width: 2 });
      // σ ticks (log): 0.1, 1, 10
      g.save(); g.font = "10px var(--mono-font, monospace)"; g.fillStyle = c.ink; g.globalAlpha = 0.5;
      g.textAlign = "center"; g.textBaseline = "top";
      var ticks = [0.1, 1, 10], t;
      for (t = 0; t < ticks.length; t++) {
        var fr = Math.log(ticks[t] / LOSIG) / Math.log(HISIG / LOSIG);
        if (fr < 0 || fr > 1) continue;
        var px = lerp(pad.l, pad.r, fr);
        g.fillText("σ=" + ticks[t], px, bot + 5);
      }
      g.restore();
      // marker
      var mi = clamp(S.mi, 0, nrev - 1 < 0 ? 0 : nrev - 1);
      var mx = markerX(S, w);
      var sgV = S.sig[Math.round(mi)] || S.sig[0];
      var accV = S.acc[Math.round(mi)], essV = S.ess[Math.round(mi)];
      if (S.dragging) { // subtle halo while grabbed
        g.save(); g.globalAlpha = 0.16; g.fillStyle = c.hot;
        g.beginPath(); g.arc(mx, (top + bot) / 2, 16, 0, 6.2832); g.fill(); g.restore();
      }
      g.save(); g.strokeStyle = c.hot; g.globalAlpha = 0.85; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(mx, top); g.lineTo(mx, bot); g.stroke();
      g.fillStyle = c.hot; g.globalAlpha = 1;
      g.beginPath(); g.arc(mx, bot, 4, 0, 6.2832); g.fill(); g.restore();
      // readouts
      label(g, "accept", pad.l, 1, c);
      labelRight(g, "ESS", pad.r, 1, c, "flow");
      g.save(); g.font = "11px var(--mono-font, monospace)"; g.textAlign = "center"; g.textBaseline = "top";
      g.fillStyle = c.hot;
      g.fillText("σ " + fmtNum(sgV) + "  ·  acc " + Math.round(accV * 100) + "%  ·  ESS " + Math.round(essV * 100) + "%", (pad.l + pad.r) / 2, top - 14 < 0 ? 0 : 1);
      g.restore();
    }
  }

  // ==========================================================================
  // 3. well-1d — a 1D double-well U(q); a ball driven by momentum kicks (violet
  //    arrows) + leapfrog crosses the barrier; a random-walk ghost cannot.
  // ==========================================================================

  function well1d(root) {
    var DOM = [-2.1, 2.1], EPS = 0.09, L = 46, KICK = 1.15, GSIG = 0.16;
    function U(q) { var t = q * q - 1; return 0.6 * t * t; }
    function dU(q) { return 0.6 * 4 * q * (q * q - 1); }

    mount(root, {
      height: 155, hz: 20,
      staticFrame: function (S) { resetW(S); for (var i = 0; i < 80; i++) stepW(S); },
      build: function (S) { resetW(S); },
      advance: function (S) { stepW(S); },
      render: function (g, S, w, h, T, c) { drawWell(g, S, w, h, T, c); }
    });

    function kick(S) {
      S.p = KICK * FV.randn(S.rng);
      S.step = 0; S.epiStartQ = S.q; S.p0 = S.p; S.epiCount = (S.epiCount || 0) + 1;
    }
    function resetW(S) {
      S.q = -1; S.prevQ = -1; S.ghost = -1; S.epiCount = 0; kick(S);
    }
    function stepW(S) {
      S.prevQ = S.q;
      var r = vstep(S.q, S.p, EPS, dU); S.q = clamp(r[0], DOM[0], DOM[1]); S.p = r[1];
      S.step++;
      if (S.step >= L) {
        // random-walk ghost takes one Metropolis step per episode (target ∝ e^-U)
        var gp = S.ghost + GSIG * FV.randn(S.rng);
        if (gp > DOM[0] && gp < DOM[1] && Math.log(S.rng()) < U(S.ghost) - U(gp)) S.ghost = gp;
        if (S.epiCount >= 10) { S.reloopSeed = nextSeed(S.reloopSeed); S.rng = FV.rng(S.reloopSeed); resetW(S); }
        else kick(S);
      }
    }
    function drawWell(g, S, w, h, T, c) {
      var pad = { l: 12, r: 12, t: 10, b: 12 };
      var xs = FV.scale(DOM, [pad.l, w - pad.r]);
      var umax = U(DOM[0]);
      var ys = FV.scale([-0.25 * umax, umax * 1.05], [h - pad.b, pad.t]);
      // potential curve
      FV.curve(g, densCurve(DOM, U, xs, ys, 120), { color: c.ink, width: 1.6 });
      // ghost (random walk, trapped) — faint
      var gq = S.ghost;
      g.save(); g.globalAlpha = 0.4; g.strokeStyle = c.data; g.fillStyle = c.data; g.lineWidth = 1.4;
      g.beginPath(); g.arc(xs(gq), ys(U(gq)) - 5, 3.4, 0, 6.2832); g.stroke(); g.restore();
      // ball (leapfrog) — coral, tween between steps
      var q = lerp(S.prevQ, S.q, T);
      var bx = xs(q), byv = ys(U(q)) - 5;
      // momentum kick arrow (violet), time-based fade over the episode
      var age = clamp((S.step + T) / L, 0, 1);
      if (age < 0.7) {
        var a = 1 - age / 0.7;
        var dir = S.p0 >= 0 ? 1 : -1, len = clamp(Math.abs(S.p0) * 16 + 8, 10, 34);
        g.save(); g.globalAlpha = 0.35 + 0.5 * a; g.strokeStyle = c.flow; g.fillStyle = c.flow; g.lineWidth = 2;
        var ax0 = bx, ax1 = bx + dir * len, ay = byv - 12;
        g.beginPath(); g.moveTo(ax0, ay); g.lineTo(ax1, ay); g.stroke();
        g.beginPath(); g.moveTo(ax1, ay); g.lineTo(ax1 - dir * 5, ay - 4); g.lineTo(ax1 - dir * 5, ay + 4); g.closePath(); g.fill();
        g.restore();
      }
      g.save(); g.fillStyle = c.hot;
      g.beginPath(); g.arc(bx, byv, 4.5, 0, 6.2832); g.fill(); g.restore();
      label(g, "leapfrog + momentum", pad.l, 0, c);
      labelRight(g, "random-walk ghost", w - pad.r, 0, c, "data");
    }
  }

  // ==========================================================================
  // 4. eps-divergence — energy error |ΔH| vs step size ε as a live log-log
  //    scatter; a coral divergence line. data-L = leapfrog steps.
  // ==========================================================================

  function epsDivergence(root) {
    var L = parseInt(root.getAttribute("data-L"), 10);
    if (!(L > 0)) L = 25;
    var EPS_LO = 0.05, EPS_HI = 3.2, DH_LO = 1e-4, DH_HI = 1e6, DIVLINE = 1e3, CAP = 220, BATCH = 5;
    function U(q) { return 0.5 * q * q; }
    function dU(q) { return q; }

    mount(root, {
      height: 175, hz: 5,
      staticFrame: function (S) { resetE(S); for (var i = 0; i < 40; i++) addPts(S); },
      build: function (S) { resetE(S); },
      advance: function (S) {
        if (S.pts.length < CAP) addPts(S);
        else { S.hold = (S.hold || 0) + 1; if (S.hold > 16) { S.reloopSeed = nextSeed(S.reloopSeed); S.rng = FV.rng(S.reloopSeed); resetE(S); } }
      },
      render: function (g, S, w, h, T, c) { drawEps(g, S, w, h, c); }
    });

    function resetE(S) { S.pts = []; S.hold = 0; }
    function addPts(S) {
      for (var b = 0; b < BATCH; b++) {
        var f = S.rng();
        var eps = EPS_LO * Math.pow(EPS_HI / EPS_LO, f);
        var q0 = FV.randn(S.rng), p0 = FV.randn(S.rng);
        var dh = leapfrogDeltaH(q0, p0, eps, L, U, dU);
        if (!(dh > 0)) dh = DH_LO;
        S.pts.push([eps, dh]);
      }
    }
    function drawEps(g, S, w, h, c) {
      var pad = { l: 30, r: 10, t: 16, b: 20 };
      var lx = function (e) { return pad.l + (Math.log(e / EPS_LO) / Math.log(EPS_HI / EPS_LO)) * (w - pad.l - pad.r); };
      var ly = function (d) { var dd = clamp(d, DH_LO, DH_HI); return (h - pad.b) - (Math.log(dd / DH_LO) / Math.log(DH_HI / DH_LO)) * (h - pad.b - pad.t); };
      // divergence threshold line (coral)
      g.save(); g.strokeStyle = c.hot; g.globalAlpha = 0.8; g.lineWidth = 1.5; g.setLineDash([5, 4]);
      g.beginPath(); g.moveTo(pad.l, ly(DIVLINE)); g.lineTo(w - pad.r, ly(DIVLINE)); g.stroke(); g.restore();
      labelRight(g, "|ΔH|=10³ divergence", w - pad.r, ly(DIVLINE) - 12, c, "hot");
      // ε ticks
      g.save(); g.font = "10px var(--mono-font, monospace)"; g.fillStyle = c.ink; g.globalAlpha = 0.5;
      g.textAlign = "center"; g.textBaseline = "top";
      var et = [0.1, 0.5, 1, 2], i;
      for (i = 0; i < et.length; i++) { if (et[i] < EPS_LO || et[i] > EPS_HI) continue; g.fillText(String(et[i]), lx(et[i]), h - pad.b + 4); }
      g.textAlign = "left"; g.fillText("ε", w - pad.r - 8, h - pad.b + 4);
      g.restore();
      // scatter — green when stable, coral when diverged
      for (i = 0; i < S.pts.length; i++) {
        var e = S.pts[i][0], d = S.pts[i][1], div = d >= DIVLINE;
        g.save(); g.globalAlpha = 0.7; g.fillStyle = div ? c.hot : c.post;
        g.beginPath(); g.arc(lx(e), ly(d), 2.4, 0, 6.2832); g.fill(); g.restore();
      }
      label(g, "L=" + L + " leapfrog  ·  |ΔH| vs ε", pad.l, 1, c);
    }
  }

  // ==========================================================================
  // 5. bind-chain — .map (transform in one lane) vs .bind (a new effect node
  //    appears); a coral value pulse flows through a growing pipeline.
  // ==========================================================================

  function bindChain(root) {
    var NB = 3; // three .bind links after pure(a)
    var OPS = [["+1", function (x) { return x + 1; }], ["×2", function (x) { return x * 2; }], ["−3", function (x) { return x - 3; }]];

    mount(root, {
      height: 170, hz: 3,
      staticFrame: function (S) { resetB(S); S.grown = NB; S.pulse = NB + 1; },
      build: function (S) { resetB(S); },
      advance: function (S) {
        if (S.grown < NB) { S.grown++; return; }
        S.pulse++;
        if (S.pulse > NB + 3) { S.a0 = 1 + Math.floor(S.rng() * 4); S.pulse = 0; S.grown = 0; }
      },
      render: function (g, S, w, h, T, c) { drawBind(g, S, w, h, T, c); }
    });

    function resetB(S) { S.a0 = 1 + Math.floor(S.rng() * 4); S.grown = 0; S.pulse = 0; }

    function drawBind(g, S, w, h, T, c) {
      var pad = 12, ncell = NB + 1;
      var cellW = (w - pad * 2) / ncell;
      var mapY = h * 0.30, bindY = h * 0.72, boxH = 26, boxW = Math.min(cellW - 14, 58);
      // pulse position along the pipeline (0..NB+1), tweened
      var pulseF = clamp(S.pulse + T, 0, ncell);
      // running value at the pulse
      function valAt(k) { var v = S.a0, i; for (i = 0; i < k && i < NB; i++) v = OPS[i][1](v); return v; }

      // ---------- map lane (top): ONE node, value transforms in-flight --------
      label(g, ".map  —  transform stays in one lane", pad, mapY - boxH / 2 - 14, c);
      var mx = pad + cellW * 0.5;
      // input wire
      g.save(); g.strokeStyle = c.ink; g.globalAlpha = 0.4; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(pad, mapY); g.lineTo(w - pad, mapY); g.stroke(); g.restore();
      // the single node
      g.save(); g.strokeStyle = c.prior; g.lineWidth = 2; g.globalAlpha = 0.9;
      roundRect(g, mx - boxW / 2, mapY - boxH / 2, boxW, boxH, 6); g.stroke(); g.restore();
      g.save(); g.font = "12px var(--mono-font, monospace)"; g.fillStyle = c.prior; g.textAlign = "center"; g.textBaseline = "middle";
      g.fillText("map f", mx, mapY); g.restore();
      // coral pulse gliding across, value shown
      var mpx = lerp(pad, w - pad, clamp(pulseF / ncell, 0, 1));
      var mv = (mpx > mx) ? OPS[0][1](S.a0) : S.a0;
      drawPulse(g, mpx, mapY, mv, c);

      // ---------- bind lane (bottom): nodes APPEAR, effects compose -----------
      label(g, ".bind  —  each bind spawns a new effect node", pad, bindY - boxH / 2 - 14, c);
      var visible = 1 + Math.min(S.grown, NB); // pure(a) + grown binds
      var k;
      // wires + nodes
      for (k = 0; k < visible; k++) {
        var cx = pad + cellW * (k + 0.5);
        if (k > 0) {
          var px0 = pad + cellW * (k - 0.5) + boxW / 2, px1 = cx - boxW / 2;
          g.save(); g.strokeStyle = c.ink; g.globalAlpha = 0.4; g.lineWidth = 1.5;
          g.beginPath(); g.moveTo(px0, bindY); g.lineTo(px1, bindY); g.stroke(); g.restore();
        }
        var isPure = k === 0;
        var col = isPure ? c.post : c.flow; // structure nodes = violet; pure value = green
        g.save(); g.strokeStyle = col; g.lineWidth = 2; g.globalAlpha = 0.9;
        roundRect(g, cx - boxW / 2, bindY - boxH / 2, boxW, boxH, 6); g.stroke(); g.restore();
        g.save(); g.font = "11px var(--mono-font, monospace)"; g.fillStyle = col; g.textAlign = "center"; g.textBaseline = "middle";
        g.fillText(isPure ? ("pure " + S.a0) : ("bind " + OPS[k - 1][0]), cx, bindY); g.restore();
      }
      // pulse through the bind pipeline (only after fully grown)
      if (S.grown >= NB) {
        var bf = clamp(pulseF, 0, ncell);
        var seg = Math.floor(bf), frac = bf - seg;
        var x0 = pad + cellW * (Math.min(seg, NB) + 0.5);
        var x1 = pad + cellW * (Math.min(seg + 1, NB) + 0.5);
        var bpx = lerp(x0, x1, frac);
        drawPulse(g, bpx, bindY, valAt(Math.min(seg + 1, NB)), c);
      }
    }
    function drawPulse(g, x, y, v, c) {
      g.save();
      g.globalAlpha = 0.25; g.fillStyle = c.hot;
      g.beginPath(); g.arc(x, y, 9, 0, 6.2832); g.fill();
      g.globalAlpha = 1; g.fillStyle = c.hot;
      g.beginPath(); g.arc(x, y, 5, 0, 6.2832); g.fill();
      g.font = "10px var(--mono-font, monospace)"; g.fillStyle = c.hot; g.textAlign = "center"; g.textBaseline = "bottom";
      g.fillText(String(v), x, y - 11);
      g.restore();
    }
  }

  // ==========================================================================
  // 6. seq-update — Beta posterior after n = 0,1,2,4,8,16 flips as a small-
  //    multiples strip sharpening left→right; loops with fresh data.
  // ==========================================================================

  function seqUpdate(root) {
    var NS = [0, 1, 2, 4, 8, 16];

    mount(root, {
      height: 150, hz: 2.5,
      staticFrame: function (S) { resetS(S); S.shown = NS.length; },
      build: function (S) { resetS(S); },
      advance: function (S) {
        if (S.shown < NS.length) S.shown++;
        else { S.hold = (S.hold || 0) + 1; if (S.hold > 5) { S.reloopSeed = nextSeed(S.reloopSeed); S.rng = FV.rng(S.reloopSeed); resetS(S); } }
      },
      render: function (g, S, w, h, T, c) { drawSeq(g, S, w, h, T, c); }
    });

    function resetS(S) {
      S.hold = 0; S.shown = 1; S.p = 0.25 + 0.5 * S.rng();
      // one shared stream of 16 flips; panel n uses the first n of them
      S.flips = []; var i;
      for (i = 0; i < 16; i++) S.flips.push(S.rng() < S.p ? 1 : 0);
    }
    // posterior params after n flips: Beta(1 + heads, 1 + tails)
    function ab(S, n) { var hh = 0, i; for (i = 0; i < n; i++) hh += S.flips[i]; return [1 + hh, 1 + (n - hh)]; }

    function drawSeq(g, S, w, h, T, c) {
      var pad = { l: 8, r: 8, t: 20, b: 14 };
      var m = NS.length, cw = (w - pad.l - pad.r) / m;
      for (var j = 0; j < m; j++) {
        var vis = j < S.shown ? 1 : 0;
        if (!vis) continue;
        var grow = (j === S.shown - 1) ? T : 1; // newest panel eases in
        var x0 = pad.l + cw * j + 3, x1 = pad.l + cw * (j + 1) - 3;
        var xs = FV.scale([0, 1], [x0, x1]);
        var pr = ab(S, NS[j]), A = pr[0], B = pr[1];
        function post(x) { return Math.exp(FV.dist.beta.logpdf(x, A, B)); }
        var ymax = 1e-6, k, xx;
        for (k = 1; k < 40; k++) { xx = k / 40; var pv = post(xx); if (isFinite(pv) && pv > ymax) ymax = pv; }
        var ys = FV.scale([0, ymax * 1.15], [h - pad.b, pad.t + (1 - grow) * 20]);
        // baseline
        g.save(); g.strokeStyle = c.ink; g.globalAlpha = 0.18 * grow; g.lineWidth = 1;
        g.beginPath(); g.moveTo(x0, ys(0)); g.lineTo(x1, ys(0)); g.stroke(); g.restore();
        var pts = densCurve([0, 1], post, xs, ys, 48);
        g.save(); g.globalAlpha = grow;
        fillUnder(g, pts, ys(0), c.post, 0.13 * grow);
        FV.curve(g, pts, { color: c.post, width: 1.6 });
        g.restore();
        // n label + posterior mean
        var mean = A / (A + B);
        g.save(); g.globalAlpha = grow; g.font = "10px var(--mono-font, monospace)"; g.fillStyle = c.ink;
        g.textAlign = "center"; g.textBaseline = "top";
        g.globalAlpha = 0.6 * grow; g.fillText("n=" + NS[j], (x0 + x1) / 2, pad.t - 16);
        g.restore();
        // mean tick (coral) at posterior mean
        g.save(); g.globalAlpha = 0.7 * grow; g.strokeStyle = c.hot; g.lineWidth = 1;
        g.beginPath(); g.moveTo(xs(mean), ys(0)); g.lineTo(xs(mean), ys(0) + 4); g.stroke(); g.restore();
      }
      label(g, "Beta(1+k, 1+n−k)  sharpening", pad.l, 2, c);
    }
  }

  // ==========================================================================
  // 7. ess-timeline — SMC ESS/N over time as a live area chart, the 0.5
  //    threshold line, resampling events as violet ticks. data-adaptive on|off.
  // ==========================================================================

  function essTimeline(root) {
    var adaptive = (root.getAttribute("data-adaptive") || "on").toLowerCase() !== "off";
    var NPART = 60, MAXT = 46, THRESH = 0.5;

    mount(root, {
      height: 155, hz: 5,
      staticFrame: function (S) { resetT(S); while (S.hist.length < MAXT) stepT(S); },
      build: function (S) { resetT(S); },
      advance: function (S) {
        if (S.hist.length < MAXT) stepT(S);
        else { S.hold = (S.hold || 0) + 1; if (S.hold > 16) { S.reloopSeed = nextSeed(S.reloopSeed); S.rng = FV.rng(S.reloopSeed); resetT(S); } }
      },
      render: function (g, S, w, h, T, c) { drawEss(g, S, w, h, c); }
    });

    function resetT(S) {
      S.w = []; var i; for (i = 0; i < NPART; i++) S.w.push(1 / NPART);
      S.hist = [1]; S.resampleAt = []; S.hold = 0; S.t = 0;
    }
    function stepT(S) {
      // reweight: each particle's incremental likelihood factor (heavy-tailed →
      // weights spread → ESS falls). lognormal-ish factor keeps weights positive.
      var i, s = 0;
      for (i = 0; i < NPART; i++) { var lf = Math.exp(0.9 * FV.randn(S.rng)); S.w[i] *= lf; s += S.w[i]; }
      for (i = 0; i < NPART; i++) S.w[i] /= s; // normalize
      var frac = essFraction(S.w);
      if (adaptive && frac < THRESH) {
        for (i = 0; i < NPART; i++) S.w[i] = 1 / NPART; // resample → weights reset
        frac = 1;
        S.resampleAt.push(S.hist.length);
      }
      S.hist.push(frac); S.t++;
    }
    function drawEss(g, S, w, h, c) {
      var pad = { l: 26, r: 10, t: 14, b: 16 };
      var xs = FV.scale([0, MAXT], [pad.l, w - pad.r]);
      var ys = FV.scale([0, 1], [h - pad.b, pad.t]);
      // threshold line
      g.save(); g.strokeStyle = c.hot; g.globalAlpha = 0.7; g.lineWidth = 1.2; g.setLineDash([5, 4]);
      g.beginPath(); g.moveTo(pad.l, ys(THRESH)); g.lineTo(w - pad.r, ys(THRESH)); g.stroke(); g.restore();
      labelRight(g, "0.5", pad.l - 4, ys(THRESH) - 6, c, "hot");
      // area + line
      var pts = [], i, hst = S.hist;
      for (i = 0; i < hst.length; i++) pts.push([xs(i), ys(hst[i])]);
      fillUnder(g, pts, ys(0), c.post, 0.16);
      FV.curve(g, pts, { color: c.post, width: 2 });
      // resample ticks (violet)
      for (i = 0; i < S.resampleAt.length; i++) {
        var rx = xs(S.resampleAt[i]);
        g.save(); g.strokeStyle = c.flow; g.globalAlpha = 0.8; g.lineWidth = 1.5;
        g.beginPath(); g.moveTo(rx, pad.t); g.lineTo(rx, h - pad.b); g.stroke(); g.restore();
      }
      // y axis label
      g.save(); g.font = "10px var(--mono-font, monospace)"; g.fillStyle = c.ink; g.globalAlpha = 0.5;
      g.textAlign = "right"; g.textBaseline = "middle";
      g.fillText("1", pad.l - 4, ys(1)); g.fillText("0", pad.l - 4, ys(0)); g.restore();
      label(g, "ESS/N  ·  resample " + (adaptive ? "adaptive" : "off"), pad.l, 1, c);
    }
  }

  // ==========================================================================
  // 8. type-flow — four sampling lanes, each pulsing a sample of its natural
  //    return type into a typed slot (mono type names).
  // ==========================================================================

  function typeFlow(root) {
    var LANES = [
      { dist: "Normal", type: "f64", role: "prior", draw: function (r) { return (FV.dist.normal.sample(r, 0, 1)).toFixed(2); } },
      { dist: "Bernoulli", type: "bool", role: "post", draw: function (r) { return FV.dist.bernoulli.sample(r, 0.5) ? "true" : "false"; } },
      { dist: "Poisson", type: "u64", role: "data", draw: function (r) { return String(FV.dist.poisson.sample(r, 3)); } },
      { dist: "Categorical", type: "usize", role: "flow", draw: function (r) { return String(FV.dist.categorical.sample(r, [0.25, 0.25, 0.25, 0.25])); } }
    ];

    mount(root, {
      height: 180, hz: 2.4,
      staticFrame: function (S) { newSamples(S); S.slotFilled = [1, 1, 1, 1]; },
      build: function (S) { newSamples(S); S.slotFilled = [0, 0, 0, 0]; S.tick = 0; },
      advance: function (S) {
        S.tick++;
        // each lane fills on its own staggered beat, then all refresh
        var k; for (k = 0; k < 4; k++) if (S.tick === k + 1) S.slotFilled[k] = 1;
        if (S.tick > 7) { S.reloopSeed = nextSeed(S.reloopSeed); S.rng = FV.rng(S.reloopSeed); newSamples(S); S.slotFilled = [0, 0, 0, 0]; S.tick = 0; }
      },
      render: function (g, S, w, h, T, c) { drawFlow(g, S, w, h, T, c); }
    });

    function newSamples(S) { S.vals = []; for (var k = 0; k < 4; k++) S.vals.push(LANES[k].draw(S.rng)); }

    function drawFlow(g, S, w, h, T, c) {
      var padL = 12, padR = 12, padT = 10, padB = 10;
      var laneH = (h - padT - padB) / 4;
      for (var k = 0; k < 4; k++) {
        var yc = padT + laneH * (k + 0.5);
        var lane = LANES[k], col = c[lane.role];
        var x0 = padL, x1 = w - padR;
        var slotW = 62, slotX = x1 - slotW;
        // lane rail
        g.save(); g.strokeStyle = c.ink; g.globalAlpha = 0.25; g.lineWidth = 1;
        g.beginPath(); g.moveTo(x0 + 66, yc); g.lineTo(slotX, yc); g.stroke(); g.restore();
        // source label (dist name)
        g.save(); g.font = "12px var(--mono-font, monospace)"; g.fillStyle = col; g.globalAlpha = 0.95;
        g.textAlign = "left"; g.textBaseline = "middle"; g.fillText(lane.dist, x0, yc); g.restore();
        // typed slot
        g.save(); g.strokeStyle = col; g.globalAlpha = 0.7; g.lineWidth = 1.5;
        roundRect(g, slotX, yc - laneH * 0.34, slotW, laneH * 0.68, 5); g.stroke(); g.restore();
        // pulse travelling into the slot (each lane staggered by its beat)
        var beat = clamp((S.tick + T - k) / 1.0, 0, 1.4);
        if (beat > 0 && beat < 1) {
          var px = lerp(x0 + 66, slotX, beat);
          g.save(); g.globalAlpha = 0.3; g.fillStyle = c.hot;
          g.beginPath(); g.arc(px, yc, 7, 0, 6.2832); g.fill();
          g.globalAlpha = 1; g.beginPath(); g.arc(px, yc, 4, 0, 6.2832); g.fill(); g.restore();
        }
        // value in slot + type name (mono)
        if (S.slotFilled[k]) {
          g.save(); g.font = "12px var(--mono-font, monospace)"; g.textBaseline = "middle";
          g.fillStyle = c.hot; g.globalAlpha = 0.95; g.textAlign = "center";
          g.fillText(S.vals[k], slotX + slotW / 2, yc); g.restore();
        }
        // type annotation under the slot
        g.save(); g.font = "10px var(--mono-font, monospace)"; g.fillStyle = c.ink; g.globalAlpha = 0.55;
        g.textAlign = "right"; g.textBaseline = "top";
        g.fillText(": " + lane.type, slotX + slotW, yc + laneH * 0.34 + 1); g.restore();
      }
    }
  }

  // ==========================================================================
  // Registration
  // ==========================================================================

  FV.register("acf-decay", function (root) { acfDecay(root); });
  FV.register("sigma-sweep", function (root) { sigmaSweep(root); });
  FV.register("well-1d", function (root) { well1d(root); });
  FV.register("eps-divergence", function (root) { epsDivergence(root); });
  FV.register("bind-chain", function (root) { bindChain(root); });
  FV.register("seq-update", function (root) { seqUpdate(root); });
  FV.register("ess-timeline", function (root) { essTimeline(root); });
  FV.register("type-flow", function (root) { typeFlow(root); });
})();
