// docs/viz/inline.js — the micro-widget family (ambient, "alive everywhere").
//
// Ten small, param-driven visualizations embeddable on ANY docs page via
//   <div class="fugue-explorable fv-inline" data-viz="NAME" data-...></div>
// Shared traits: autoplay ambient loop that advances discrete STATE at 2–6 Hz
// with tweened rendering between states; one unobtrusive pause/play glyph
// (top-right); optional one-line caption from data-caption; seeded via
// data-seed (default 11); all math via FugueViz; theme-aware; the color algebra
// everywhere (data = yellow, prior = blue, posterior = green, current = coral,
// structure = violet).
//
// The math is real: split-R̂ is ported line-for-line from
// src/inference/diagnostics.rs (Vehtari et al. 2021 split variant); EM
// responsibilities, Beta/Normal conjugate updates, and the logistic likelihood
// are the genuine formulas (known-value checks live in the inline agent report).
//
// Self-contained IIFE; assumes fugue-viz.js has loaded first (book.toml order).
(function () {
  "use strict";
  if (typeof window === "undefined" || !window.FugueViz) return;
  var FV = window.FugueViz;

  // ==========================================================================
  // Small generic helpers
  // ==========================================================================

  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function clone(o) { var r = {}; for (var k in o) if (o.hasOwnProperty(k)) r[k] = o[k]; return r; }
  function nextSeed(s) { return (Math.imul(s, 1664525) + 1013904223) >>> 0; }

  // Coarse (touch) pointers get inflated hit targets (§A.2: ≥22 CSS px). Cached.
  var _coarse = null;
  function coarsePointer() {
    if (_coarse === null) _coarse = !!(window.matchMedia && window.matchMedia("(pointer: coarse)").matches);
    return _coarse;
  }

  // Numerically-stable softplus: log(1 + e^z) = max(z,0) + log1p(e^-|z|).
  function softplus(z) { var az = Math.abs(z); return Math.max(z, 0) + FV.log1p(Math.exp(-az)); }

  function parseRGB(col) {
    col = (col || "").trim();
    var m = /^#?([0-9a-f]{6})$/i.exec(col);
    if (m) { var n = parseInt(m[1], 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
    var r = /rgba?\(([^)]+)\)/.exec(col);
    if (r) { var p = r[1].split(","); return [parseInt(p[0], 10), parseInt(p[1], 10), parseInt(p[2], 10)]; }
    return [128, 128, 128];
  }
  function blend(c1, c2, t) {
    var a = parseRGB(c1), b = parseRGB(c2);
    return "rgb(" + Math.round(lerp(a[0], b[0], t)) + "," + Math.round(lerp(a[1], b[1], t)) + "," + Math.round(lerp(a[2], b[2], t)) + ")";
  }
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
  function densCurve(g, dom, f, xs, ys, n) {
    var pts = [], i, x;
    for (i = 0; i <= n; i++) { x = dom[0] + (dom[1] - dom[0]) * i / n; pts.push([xs(x), ys(f(x))]); }
    return pts;
  }

  // ==========================================================================
  // Shared statistical math (real formulas; verified in the agent report)
  // ==========================================================================

  // Split-R̂ — ported from src/inference/diagnostics.rs (split_f64_chains +
  // r_hat_from_f64_chains). Each chain is halved before the between/within
  // comparison (Vehtari et al. 2021), so within-chain trends inflate R̂.
  function splitChains(chs) {
    var out = [], i;
    for (i = 0; i < chs.length; i++) {
      var c = chs[i], half = Math.floor(c.length / 2);
      if (half === 0) { out.push(c.slice()); continue; }
      out.push(c.slice(0, half));
      out.push(c.slice(half, 2 * half));
    }
    return out;
  }
  function rhatCore(ch) {
    if (ch.length < 2) return 1.0;
    var m = ch.length, i, k;
    for (i = 0; i < m; i++) if (ch[i].length === 0) return NaN;
    var n = ch[0].length;
    var means = [], overall = 0;
    for (i = 0; i < m; i++) { var s = 0, v = ch[i]; for (k = 0; k < v.length; k++) s += v[k]; means[i] = s / v.length; overall += means[i]; }
    overall /= m;
    var b = 0; for (i = 0; i < m; i++) { var d = means[i] - overall; b += d * d; } b *= n / (m - 1);
    var w = 0;
    for (i = 0; i < m; i++) { var vv = ch[i], mu = means[i], s2 = 0; for (k = 0; k < vv.length; k++) { var e = vv[k] - mu; s2 += e * e; } w += s2 / (n - 1); }
    w /= m;
    if (!(w > 0)) return NaN;
    var varPlus = ((n - 1) / n) * w + (1 / n) * b;
    return Math.sqrt(varPlus / w);
  }
  function splitRhat(chs) { return rhatCore(splitChains(chs)); }

  // Normal–Normal conjugate update, known observation variance. Prior
  // μ ~ Normal(m0, s0); observations y_i ~ Normal(μ, sigma). Returns the exact
  // posterior {mean, sd} over μ.
  function normalPosterior(m0, s0, sigma, ys) {
    var prec = 1 / (s0 * s0), num = m0 / (s0 * s0), i, iv = 1 / (sigma * sigma);
    for (i = 0; i < ys.length; i++) { prec += iv; num += ys[i] * iv; }
    var pv = 1 / prec;
    return { mean: num * pv, sd: Math.sqrt(pv) };
  }

  // One EM step for a 2-component 1-D Gaussian mixture. Returns updated params
  // and the per-point responsibilities r_i (of component 2), computed in
  // log-space. This IS the E-step responsibility formula and the weighted M-step.
  function emStep(xs, p) {
    var n = xs.length, r = new Array(n), i, g = FV.dist.normal;
    for (i = 0; i < n; i++) {
      var l1 = Math.log(p.pi) + g.logpdf(xs[i], p.m1, p.s1);
      var l2 = Math.log(1 - p.pi) + g.logpdf(xs[i], p.m2, p.s2);
      var mx = l1 > l2 ? l1 : l2;
      var e1 = Math.exp(l1 - mx), e2 = Math.exp(l2 - mx);
      r[i] = e2 / (e1 + e2);
    }
    var n2 = 0; for (i = 0; i < n; i++) n2 += r[i]; var n1 = n - n2;
    var m1 = 0, m2 = 0; for (i = 0; i < n; i++) { m1 += (1 - r[i]) * xs[i]; m2 += r[i] * xs[i]; }
    m1 /= (n1 || 1e-9); m2 /= (n2 || 1e-9);
    var v1 = 0, v2 = 0; for (i = 0; i < n; i++) { v1 += (1 - r[i]) * (xs[i] - m1) * (xs[i] - m1); v2 += r[i] * (xs[i] - m2) * (xs[i] - m2); }
    v1 /= (n1 || 1e-9); v2 /= (n2 || 1e-9);
    return { pi: n1 / n, m1: m1, s1: Math.sqrt(Math.max(v1, 0.04)), m2: m2, s2: Math.sqrt(Math.max(v2, 0.04)), r: r };
  }

  // Logistic-regression log-likelihood + Normal(0,3) prior over weights
  // w = [w0, w1x, w1y]. Stable via softplus: log σ(z) = −softplus(−z),
  // log(1−σ(z)) = −softplus(z).
  function logisticLogPost(w, pts) {
    var s = FV.dist.normal.logpdf(w[0], 0, 3) + FV.dist.normal.logpdf(w[1], 0, 3) + FV.dist.normal.logpdf(w[2], 0, 3), i;
    for (i = 0; i < pts.length; i++) {
      var z = w[0] + w[1] * pts[i].x + w[2] * pts[i].y;
      s += pts[i].c ? -softplus(-z) : -softplus(z);
    }
    return s;
  }

  // ==========================================================================
  // The mount scaffold — one ambient loop, glyph, caption, reduced-motion frame
  // ==========================================================================

  function evtXY(elm, e) {
    var r = elm.getBoundingClientRect();
    // On touchend/touchcancel e.touches is empty; the released point lives in
    // changedTouches. Reading e.touches[0] there threw (undefined.clientX),
    // which crashed the "up" handler and left drag state stuck.
    var t = null;
    if (e.touches && e.touches.length) t = e.touches[0];
    else if (e.changedTouches && e.changedTouches.length) t = e.changedTouches[0];
    var cx = t ? t.clientX : e.clientX;
    var cy = t ? t.clientY : e.clientY;
    return [cx - r.left, cy - r.top];
  }

  function mount(root, spec) {
    var seed = parseInt(root.getAttribute("data-seed"), 10);
    if (!(seed >= 0)) seed = 11;
    // A resize resets the canvas backing store (clearing it). When the ambient
    // loop is running it repaints every frame so the clear is invisible; but a
    // paused or reduced-motion widget renders exactly once, so without repainting
    // on resize its static frame would be wiped to an empty canvas. Repaint here.
    var ready = false;
    var cv = FV.canvas(root, { height: spec.height || 150, onResize: function () { if (ready) renderFrame(); } });
    var g = cv.ctx;

    // optional caption
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

    // pointer interactions (mouse + touch). Ambient micros must NEVER eat page
    // scroll (§A.1): touch gestures are claimed only when they actually engage a
    // draggable. Two modes:
    //   • spec.scrub  — the whole canvas is a horizontal control. touch-action
    //     pan-y keeps native vertical page scrolling; the browser then only
    //     delivers *cancelable* horizontal-intent moves to us, so a vertical
    //     swipe scrolls the page and a horizontal drag scrubs.
    //   • point drag  — claim the gesture only when pointerdown actually hits a
    //     target (spec.pointer returns truthy on "down"); a miss (empty canvas)
    //     is left untouched so the page scrolls normally.
    if (spec.pointer) {
      var coarse = coarsePointer();
      var down = false, claimed = false;
      function fire(e, phase) {
        var p = evtXY(cv.el, e);
        var r = spec.pointer(S, p[0], p[1], cv.w, cv.h, phase, FV, coarse);
        renderFrame();
        return r;
      }
      // Mouse never fights page scroll — wire it straight through.
      cv.el.addEventListener("mousedown", function (e) { down = true; fire(e, "down"); });
      window.addEventListener("mousemove", function (e) { if (down) fire(e, "move"); });
      window.addEventListener("mouseup", function (e) { if (down) { down = false; fire(e, "up"); } });
      if (spec.scrub) {
        // The whole canvas is a horizontal control, yet a *vertical* swipe must
        // still scroll the page (§A.1). touch-action:pan-y asks the browser to
        // own vertical scrolling, but with a non-passive touchmove listener the
        // pan-y moves can still arrive `cancelable` (they do under some engines
        // and headless/emulated touch), so keying off `e.cancelable` alone would
        // let this control eat the page scroll. Instead latch the gesture axis on
        // the first move past a small dead-zone: only a horizontally-dominant
        // drag is claimed for scrubbing; a vertical one is left to the page.
        cv.el.style.touchAction = "pan-y";
        var sx0 = 0, sy0 = 0, axis = 0; // 0 undecided, 1 scrub (horizontal), -1 scroll (vertical)
        cv.el.addEventListener("touchstart", function (e) {
          down = true; axis = 0;
          var t = e.touches && e.touches[0];
          if (t) { sx0 = t.clientX; sy0 = t.clientY; }
          fire(e, "down");
        }, { passive: true });
        cv.el.addEventListener("touchmove", function (e) {
          if (!down) return;
          var t = e.touches && e.touches[0]; if (!t) return;
          if (axis === 0) {
            var dx = Math.abs(t.clientX - sx0), dy = Math.abs(t.clientY - sy0);
            if (dx < 6 && dy < 6) return;   // below dead-zone: wait, let native scroll begin
            axis = dx > dy ? 1 : -1;        // decide once, then latch for the gesture
          }
          if (axis === 1 && e.cancelable) { fire(e, "move"); e.preventDefault(); }
          // axis === -1 (vertical intent): never preventDefault — the page scrolls
        }, { passive: false });
        cv.el.addEventListener("touchend", function (e) { if (down) { down = false; fire(e, "up"); } axis = 0; });
        cv.el.style.cursor = "ew-resize";
      } else {
        cv.el.addEventListener("touchstart", function (e) {
          claimed = !!fire(e, "down"); down = claimed;
          if (claimed && e.cancelable) e.preventDefault();
        }, { passive: false });
        cv.el.addEventListener("touchmove", function (e) {
          if (down && claimed) { fire(e, "move"); if (e.cancelable) e.preventDefault(); }
        }, { passive: false });
        cv.el.addEventListener("touchend", function (e) { if (down) { down = false; fire(e, "up"); } claimed = false; });
        cv.el.style.cursor = "grab";
      }
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
  // 1. dist-strip — a distribution, forever raining samples into a histogram.
  // ==========================================================================

  var DISCRETE = { bernoulli: 1, binomial: 1, poisson: 1, categorical: 1 };
  function defaultParams(name) {
    return ({ normal: [0, 1], lognormal: [0, 0.5], beta: [2, 2], gamma: [2, 1], exponential: [1], uniform: [0, 1], bernoulli: [0.5], binomial: [10, 0.5], poisson: [4], categorical: [0.2, 0.3, 0.3, 0.2] })[name] || [0, 1];
  }
  function firstParamLabel(name) {
    return ({ normal: "μ", lognormal: "μ", beta: "α", gamma: "k", exponential: "λ", uniform: "lo", bernoulli: "p", binomial: "n", poisson: "λ", categorical: "" })[name] || "";
  }
  function p0Range(name) {
    return ({ normal: 8, lognormal: 3, beta: 12, gamma: 12, exponential: 6, uniform: 4, bernoulli: 1.2, binomial: 34, poisson: 24 })[name] || 6;
  }
  function clampP0(name, v) {
    switch (name) {
      case "beta": case "gamma": return clamp(v, 0.2, 20);
      case "exponential": return clamp(v, 0.1, 10);
      case "poisson": return clamp(v, 0.2, 40);
      case "bernoulli": return clamp(v, 0.02, 0.98);
      case "binomial": return Math.round(clamp(v, 1, 40));
      default: return v;
    }
  }
  function distDomain(name, p) {
    switch (name) {
      case "normal": return [p[0] - 4 * p[1], p[0] + 4 * p[1]];
      case "lognormal": return [0, Math.exp(p[0] + 3 * p[1])];
      case "beta": return [0, 1];
      case "gamma": { var m = p[0] / p[1], sd = Math.sqrt(p[0]) / p[1]; return [0, Math.max(m + 4 * sd, 0.01)]; }
      case "exponential": return [0, 6 / p[0]];
      case "uniform": { var lo = Math.min(p[0], p[1] - 0.1), hi = Math.max(p[1], lo + 0.1), s = (hi - lo) * 0.12; return [lo - s, hi + s]; }
      case "bernoulli": return [-0.6, 1.6];
      case "binomial": return [-0.6, p[0] + 0.6];
      case "poisson": return [-0.6, Math.ceil(p[0] + 4 * Math.sqrt(p[0]) + 1) + 0.6];
      case "categorical": return [-0.6, p.length - 0.4];
    }
    return [0, 1];
  }
  function distSample(name, rand, p) {
    var d = FV.dist[name];
    switch (name) {
      case "exponential": return d.sample(rand, p[0]);
      case "poisson": return d.sample(rand, p[0]);
      case "bernoulli": return d.sample(rand, p[0]) ? 1 : 0;
      case "categorical": return d.sample(rand, p);
      default: return d.sample(rand, p[0], p[1]);
    }
  }
  function distDensity(name, x, p) {
    var d = FV.dist[name];
    switch (name) {
      case "normal": case "lognormal": case "beta": case "gamma": return Math.exp(d.logpdf(x, p[0], p[1]));
      case "exponential": return Math.exp(d.logpdf(x, p[0]));
      case "uniform": return Math.exp(d.logpdf(x, p[0], p[1]));
      case "bernoulli": return Math.exp(d.logpmf(x, p[0]));
      case "binomial": return Math.exp(d.logpmf(x, p[0], p[1]));
      case "poisson": return Math.exp(d.logpmf(x, p[0]));
      case "categorical": return Math.exp(d.logpmf(x, p));
    }
    return 0;
  }

  function distStrip(root) {
    var name = (root.getAttribute("data-dist") || "normal").toLowerCase();
    var praw = root.getAttribute("data-params");
    var params = praw ? praw.split(",").map(function (s) { return parseFloat(s); }) : defaultParams(name);
    var isDisc = !!DISCRETE[name];
    var CAP = 480, BATCH = 6, RANGE = p0Range(name);

    mount(root, {
      height: 150, hz: 5, settleN: 70, scrub: true,
      build: function (S) { S.params = params.slice(); S.buf = []; S.dragX = null; },
      advance: function (S) {
        for (var i = 0; i < BATCH; i++) S.buf.push(distSample(name, S.rng, S.params));
        while (S.buf.length > CAP) S.buf.shift();
      },
      render: function (g, S, w, h, T, c) {
        var pad = { l: 10, r: 10, t: 15, b: 12 };
        var dom = distDomain(name, S.params);
        var xs = FV.scale(dom, [pad.l, w - pad.r]);
        var ymax = 1e-6, k, x, d;
        if (isDisc) { for (k = Math.max(0, Math.ceil(dom[0])); k <= Math.floor(dom[1]); k++) { d = distDensity(name, k, S.params); if (d > ymax) ymax = d; } }
        else { for (k = 0; k <= 120; k++) { x = dom[0] + (dom[1] - dom[0]) * k / 120; d = distDensity(name, x, S.params); if (isFinite(d) && d > ymax) ymax = d; } }
        var ys = FV.scale([0, ymax * 1.25], [h - pad.b, pad.t]);
        baseline(g, pad.l, w - pad.r, ys(0), c);
        if (isDisc) drawDiscHist(g, S.buf, xs, ys, c.post);
        else FV.histogram(g, S.buf, { bins: 28, xscale: xs, yscale: ys, color: c.post, alpha: 0.5 });
        if (isDisc) drawStems(g, name, S.params, dom, xs, ys, c.prior);
        else { var pts = densCurve(g, dom, function (xx) { return distDensity(name, xx, S.params); }, xs, ys, 140); fillUnder(g, pts, ys(0), c.prior, 0.08); FV.curve(g, pts, { color: c.prior, width: 2 }); }
        if (firstParamLabel(name)) label(g, firstParamLabel(name) + " " + fmtNum(S.params[0]), pad.l, pad.t - 4, c);
        label(g, isDisc ? "" : "n=" + S.buf.length, pad.l, pad.t - 4, c);
        if (firstParamLabel(name)) labelRight(g, name, w - pad.r, pad.t - 4, c);
      },
      pointer: (name === "categorical") ? null : function (S, x, y, w, h, phase) {
        if (phase === "down") { S.dragX = x; S.drag0 = S.params[0]; }
        else if (phase === "move" && S.dragX != null) {
          var frac = (x - S.dragX) / w;
          S.params[0] = clampP0(name, S.drag0 + frac * RANGE);
          if (name === "uniform" && S.params[0] > S.params[1] - 0.2) S.params[0] = S.params[1] - 0.2;
          S.buf.length = 0;
        } else if (phase === "up") { S.dragX = null; }
      }
    });

    function drawDiscHist(g, buf, xs, ys, col) {
      if (!buf.length) return;
      var counts = {}, n = 0, i, v;
      for (i = 0; i < buf.length; i++) { v = Math.round(buf[i]); counts[v] = (counts[v] || 0) + 1; n++; }
      var bw = Math.max(4, (xs(1) - xs(0)) * 0.6);
      g.save(); g.globalAlpha = 0.5; g.fillStyle = col;
      for (var kk in counts) if (counts.hasOwnProperty(kk)) {
        var kv = parseInt(kk, 10), prob = counts[kk] / n, px = xs(kv), py = ys(prob);
        g.fillRect(px - bw / 2, py, bw, ys(0) - py);
      }
      g.restore();
    }
    function drawStems(g, nm, p, dom, xs, ys, col) {
      var lo = Math.max(0, Math.ceil(dom[0])), hi = Math.floor(dom[1]), k;
      g.save(); g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 2;
      for (k = lo; k <= hi; k++) {
        var d = distDensity(nm, k, p); if (!isFinite(d) || d <= 0) continue;
        var px = xs(k), py = ys(d), y0 = ys(0);
        g.beginPath(); g.moveTo(px, y0); g.lineTo(px, py); g.stroke();
        g.beginPath(); g.arc(px, py, 2.6, 0, 6.2832); g.fill();
      }
      g.restore();
    }
  }

  // ==========================================================================
  // 2. posterior-morph — observations arrive one at a time; posterior sharpens.
  // ==========================================================================

  function posteriorMorph(root) {
    var kind = (root.getAttribute("data-kind") || "beta").toLowerCase();
    var NOBS = 12;

    mount(root, {
      height: 160, hz: 3, settleN: NOBS,
      staticFrame: function (S) { reset(S); for (var i = 0; i < NOBS; i++) obs(S); },
      build: function (S) { reset(S); },
      advance: function (S) {
        if (S.phase === "obs") { obs(S); if (S.n >= NOBS) { S.phase = "hold"; S.hold = 0; } }
        else { S.hold++; if (S.hold > 4) { S.reloopSeed = nextSeed(S.reloopSeed); S.rng = FV.rng(S.reloopSeed); reset(S); } }
      },
      render: function (g, S, w, h, T, c) { drawMorph(g, S, w, h, T, c); }
    });

    function reset(S) {
      S.phase = "obs"; S.n = 0; S.hold = 0; S.lastObs = null;
      if (kind === "beta") { S.a0 = 2; S.b0 = 2; S.hh = 0; S.tt = 0; S.ptrue = 0.2 + 0.6 * S.rng(); }
      else { S.m0 = 0; S.s0 = 2; S.sigma = 1; S.ys = []; S.mtrue = -1.5 + 3 * S.rng(); }
      S.toA = null; S.toB = null; snap(S); S.fromA = S.toA; S.fromB = S.toB;
    }
    function obs(S) {
      if (kind === "beta") { var f = S.rng() < S.ptrue ? 1 : 0; if (f) S.hh++; else S.tt++; S.lastObs = f; }
      else { var y = FV.dist.normal.sample(S.rng, S.mtrue, 1); S.ys.push(y); S.lastObs = y; }
      S.n++; snap(S);
    }
    function snap(S) {
      var toA, toB;
      if (kind === "beta") { toA = S.a0 + S.hh; toB = S.b0 + S.tt; }
      else { var p = normalPosterior(S.m0, S.s0, S.sigma, S.ys); toA = p.mean; toB = p.sd; }
      S.fromA = (S.toA == null) ? toA : S.toA;
      S.fromB = (S.toB == null) ? toB : S.toB;
      S.toA = toA; S.toB = toB;
    }
    function drawMorph(g, S, w, h, T, c) {
      var pad = { l: 10, r: 10, t: 15, b: 14 };
      var dom = kind === "beta" ? [0, 1] : [-4.2, 4.2];
      var xs = FV.scale(dom, [pad.l, w - pad.r]);
      var A = lerp(S.fromA, S.toA, T), B = lerp(S.fromB, S.toB, T);
      function post(x) { return kind === "beta" ? Math.exp(FV.dist.beta.logpdf(x, A, B)) : Math.exp(FV.dist.normal.logpdf(x, A, B)); }
      function prior(x) { return kind === "beta" ? Math.exp(FV.dist.beta.logpdf(x, S.a0, S.b0)) : Math.exp(FV.dist.normal.logpdf(x, S.m0, S.s0)); }
      var ymax = 1e-6, k, x;
      for (k = 1; k < 120; k++) { x = dom[0] + (dom[1] - dom[0]) * k / 120; var pv = post(x); if (isFinite(pv) && pv > ymax) ymax = pv; var qv = prior(x); if (isFinite(qv) && qv > ymax) ymax = qv; }
      var ys = FV.scale([0, ymax * 1.15], [h - pad.b, pad.t]);
      baseline(g, pad.l, w - pad.r, ys(0), c);
      FV.curve(g, densCurve(g, dom, prior, xs, ys, 120), { color: c.prior, width: 1.5, dash: [4, 3] });
      var qq = densCurve(g, dom, post, xs, ys, 120);
      fillUnder(g, qq, ys(0), c.post, 0.15); FV.curve(g, qq, { color: c.post, width: 2 });
      if (S.lastObs != null) {
        var ox = kind === "beta" ? (S.lastObs ? xs(1) : xs(0)) : xs(clamp(S.lastObs, dom[0], dom[1]));
        g.save(); g.globalAlpha = 0.35 + 0.6 * (1 - T); g.fillStyle = c.data;
        g.beginPath(); g.arc(ox, ys(0) - 5, 3 + 4 * (1 - T), 0, 6.2832); g.fill(); g.restore();
      }
      label(g, "n " + S.n, pad.l, pad.t - 4, c);
      var meanTxt = kind === "beta" ? (S.toA / (S.toA + S.toB)) : S.toA;
      labelRight(g, "post μ " + fmtNum(meanTxt), w - pad.r, pad.t - 4, c, "post");
    }
  }

  // ==========================================================================
  // 3. trace-ticker — a 3-site model runs, trace rows type in, tally, reloop.
  // ==========================================================================

  function traceTicker(root) {
    mount(root, {
      height: 150, hz: 2.6,
      staticFrame: function (S) { buildRun(S); S.row = 3; S.phase = "hold"; S.hold = 0; },
      build: function (S) { buildRun(S); S.row = 0; S.phase = "type"; S.hold = 0; },
      advance: function (S) {
        if (S.phase === "type") { S.row++; if (S.row >= 3) { S.phase = "hold"; S.hold = 0; } }
        else { S.hold++; if (S.hold > 3) { S.reloopSeed = nextSeed(S.reloopSeed); S.rng = FV.rng(S.reloopSeed); buildRun(S); S.row = 0; S.phase = "type"; } }
      },
      render: function (g, S, w, h, T, c) { drawTrace(g, S, w, h, T, c); }
    });

    function buildRun(S) {
      var g = FV.dist.normal;
      var mu = Math.round(g.sample(S.rng, 0, 1) * 100) / 100;
      var d1 = Math.round(g.sample(S.rng, mu, 1) * 100) / 100;
      var d2 = Math.round(g.sample(S.rng, mu, 1) * 100) / 100;
      S.rows = [
        { addr: "mu", val: mu, logw: g.logpdf(mu, 0, 1), role: "prior" },
        { addr: "y1", val: d1, logw: g.logpdf(d1, mu, 1), role: "data" },
        { addr: "y2", val: d2, logw: g.logpdf(d2, mu, 1), role: "data" }
      ];
    }
    function drawTrace(g, S, w, h, T, c) {
      var pad = 12, headH = 16, rowH = (h - pad * 2 - headH) / 3;
      var shown = S.phase === "type" ? S.row : 3;
      g.save(); g.font = "12px var(--mono-font, monospace)"; g.textBaseline = "middle";
      // header
      g.textAlign = "left"; g.fillStyle = c.ink; g.globalAlpha = 0.5;
      g.fillText("addr", pad + 8, pad + headH / 2);
      g.textAlign = "center"; g.fillText("value", w * 0.52, pad + headH / 2);
      g.textAlign = "right"; g.fillText("log w", w - pad, pad + headH / 2);
      var total = 0;
      for (var i = 0; i < 3; i++) {
        var r = S.rows[i], yy = pad + headH + i * rowH + rowH / 2;
        var vis = i < shown ? 1 : (i === shown && S.phase === "type" ? T : 0);
        if (vis <= 0) continue;
        total += r.logw * vis;
        if (i === shown - 1 && S.phase === "type") { g.globalAlpha = 0.9 * vis; g.fillStyle = c.hot; g.fillRect(pad, yy - rowH / 2 + 2, 3, rowH - 4); }
        g.globalAlpha = 0.6 * vis; g.textAlign = "left"; g.fillStyle = c.ink; g.fillText(r.addr, pad + 8, yy);
        g.globalAlpha = vis; g.textAlign = "center"; g.fillStyle = c[r.role]; g.fillText(fmtNum(r.val), w * 0.52, yy);
        g.globalAlpha = 0.85 * vis; g.textAlign = "right"; g.fillStyle = c.ink; g.fillText(fmtNum(r.logw), w - pad, yy);
      }
      // running tally
      g.globalAlpha = 0.5; g.strokeStyle = c.ink; g.lineWidth = 1;
      g.beginPath(); g.moveTo(pad, h - pad - 2); g.lineTo(w - pad, h - pad - 2); g.stroke();
      g.globalAlpha = 0.85; g.textAlign = "left"; g.fillStyle = c.ink; g.fillText("Σ log w", pad + 8, h - pad + 6);
      g.globalAlpha = 1; g.textAlign = "right"; g.fillStyle = c.post; g.fillText(fmtNum(total), w - pad, h - pad + 6);
      g.restore();
    }
  }

  // ==========================================================================
  // 4. rhat-spark — three over-dispersed chains COLLAPSE onto a shared band and
  //    the live split-R̂ falls past 1.1 (coral→green) as they merge — or, in
  //    "bad" mode, stay trapped in three separate modes with R̂ pinned high.
  //
  //    The convergence STORY has to be visible, which means the vertical
  //    ENVELOPE must shrink: three thin threads far apart (dark space between
  //    them), a held separated phase, then a crisp funnel onto one shared
  //    stationary Normal(0,1) band. Over-dispersion only reads if the start
  //    separation is several stationary SDs and the warmup wiggle is tight —
  //    otherwise "before" and "after" span the same pixels and nothing
  //    collapses. That relaxation-to-a-shared-target is what an AR(1)/Langevin
  //    move toward N(0,1) does in expectation (mean decays, variance settles);
  //    we drive mean-collapse and wiggle-growth on one eased schedule.
  //
  //    The readout R̂ is real split-R̂ (same rhatCore as everywhere) computed on
  //    the recent (post-warmup) draws — a trailing window — so it falls to ~1.00
  //    and turns green when the chains actually merge, instead of dragging the
  //    over-dispersed warmup along forever. A thin R̂-over-time sparkline under
  //    the traces re-tells the same fall, crossing the dashed 1.1 threshold.
  // ==========================================================================

  function rhatSpark(root) {
    var mode = (root.getAttribute("data-mode") || "good").toLowerCase();
    var GOOD = mode !== "bad";
    var NCH = 3, MAXN = 140;
    var SEPN = Math.round(MAXN * 0.22); // fully separated threads until here…
    var MIXN = Math.round(MAXN * 0.42); // …then funnel; fully merged by here
    var NPHI = 0.2;                    // mixing autocorrelation (low → clean R̂)
    var SD0 = 0.32, SD1 = 1.0;         // thin warmup threads → full stationary band
    var DWIN = 38;                     // trailing window for the live (post-warmup) R̂
    var BPHI = 0.82;                   // bad-mode within-mode autocorrelation

    // Hold separated, then smoothstep-funnel onto the shared band.
    function ease(u) { return u * u * (3 - 2 * u); }
    function phase01(k) { return k <= SEPN ? 0 : k >= MIXN ? 1 : (k - SEPN) / (MIXN - SEPN); }
    function collapse(k) { return 1 - ease(phase01(k)); } // 1→0 mean scale
    function sdScale(k) { return SD0 + (SD1 - SD0) * ease(phase01(k)); }

    mount(root, {
      height: 150, hz: 6,
      staticFrame: function (S) { resetR(S); for (var i = 0; i < MAXN; i++) stepR(S); },
      build: function (S) { resetR(S); },
      advance: function (S) { if (S.k < MAXN) stepR(S); else { S.hold = (S.hold || 0) + 1; if (S.hold > 12) { S.reloopSeed = nextSeed(S.reloopSeed); S.rng = FV.rng(S.reloopSeed); resetR(S); } } },
      render: function (g, S, w, h, T, c) { drawR(g, S, w, h, c); }
    });

    function resetR(S) {
      S.k = 0; S.hold = 0; S.ch = [[], [], []]; S.rhat = 1; S.rhist = [];
      S.eta = []; S.cur = [];
      // good: dispersed starts, one shared Normal(0,1) target → chains mix.
      // bad:  chains trapped in three separate modes → split-R̂ stuck ≫ 1.1.
      S.starts = GOOD ? [-3.4, 0.0, 3.4] : [-1.7, 0.1, 1.7];
      S.centers = GOOD ? [0, 0, 0] : [-1.5, 0.0, 1.5];
      for (var i = 0; i < NCH; i++) { S.eta[i] = FV.randn(S.rng); S.cur[i] = S.starts[i]; }
    }
    function stepR(S) {
      var k = S.k, i;
      if (GOOD) {
        // x = (collapsing separated mean) + (growing stationary N(0,1) wiggle).
        for (i = 0; i < NCH; i++) {
          S.eta[i] = NPHI * S.eta[i] + Math.sqrt(1 - NPHI * NPHI) * FV.randn(S.rng);
          S.ch[i].push(S.starts[i] * collapse(k) + sdScale(k) * S.eta[i]);
        }
      } else {
        // AR(1) trapped in its own mode: never leaves, so chains never agree.
        for (i = 0; i < NCH; i++) {
          S.cur[i] = S.centers[i] + BPHI * (S.cur[i] - S.centers[i]) + FV.randn(S.rng) * Math.sqrt(1 - BPHI * BPHI);
          S.ch[i].push(S.cur[i]);
        }
      }
      S.k++;
      // live split-R̂ over the most recent DWIN draws (drop the warmup).
      var lo = Math.max(0, S.ch[0].length - DWIN);
      S.rhat = splitRhat([S.ch[0].slice(lo), S.ch[1].slice(lo), S.ch[2].slice(lo)]);
      S.rhist.push(S.rhat);
    }
    function drawR(g, S, w, h, c) {
      var pad = { l: 8, r: 8, t: 16 };
      var sparkH = 18, sparkGap = 6, sparkBot = h - 6, sparkTop = sparkBot - sparkH;
      var plotTop = pad.t, plotBot = sparkTop - sparkGap;
      var xs = FV.scale([0, MAXN], [pad.l, w - pad.r]);
      var ys = FV.scale([-5, 5], [plotBot, plotTop]);
      baseline(g, pad.l, w - pad.r, ys(0), c);
      var roleFor = ["prior", "post", "flow"];
      for (var i = 0; i < NCH; i++) {
        var pts = [], v = S.ch[i], k;
        for (k = 0; k < v.length; k++) pts.push([xs(k), ys(clamp(v[k], -5, 5))]);
        FV.curve(g, pts, { color: c[roleFor[i]], width: 1.2 });
      }
      var rh = S.rhat;
      var conv = isFinite(rh) && rh <= 1.1;
      // R̂ readout, top-right, drawn INSIDE the canvas and cleared of the pause
      // glyph (a ~20–30px DOM button top-right): reserve glyphClear px so the
      // number never hides under it at narrow (phone) widths.
      g.save();
      g.font = "13px var(--mono-font, monospace)"; g.textAlign = "right"; g.textBaseline = "top";
      var rtxt = "R̂ " + (isFinite(rh) ? rh.toFixed(2) : "—");
      var glyphClear = coarsePointer() ? 34 : 24;
      var rRight = w - pad.r - glyphClear;
      var rtxtW = g.measureText(rtxt).width;
      g.fillStyle = conv ? c.post : c.hot;
      g.fillText(rtxt, rRight, 3);
      g.restore();
      // "3 chains" label — only if it clears the readout at this width.
      g.save(); g.font = "11px var(--mono-font, monospace)";
      var labW = g.measureText("3 chains").width; g.restore();
      if (pad.l + labW + 10 < rRight - rtxtW) label(g, "3 chains", pad.l, plotTop - 8, c);
      drawSpark(g, S, w, sparkTop, sparkBot, c, conv);
    }
    // Thin R̂-over-time sparkline: fall past the dashed 1.1 threshold, segments
    // recolored coral→green so the crossing reads at a glance.
    function drawSpark(g, S, w, top, bot, c, conv) {
      var pl = 8, pr = 8, rLo = 1.0, rHi = 1.85;
      var sx = FV.scale([0, MAXN], [pl, w - pr]);
      var sy = FV.scale([rLo, rHi], [bot, top]);
      g.save(); g.strokeStyle = c.ink; g.globalAlpha = 0.18; g.lineWidth = 1; g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(pl, sy(1.1)); g.lineTo(w - pr, sy(1.1)); g.stroke(); g.restore();
      var hh = S.rhist; if (hh.length < 2) return;
      for (var i = 1; i < hh.length; i++) {
        var a = clamp(hh[i - 1], rLo, rHi), b = clamp(hh[i], rLo, rHi);
        g.save(); g.strokeStyle = hh[i] > 1.1 ? c.hot : c.post; g.globalAlpha = 0.85; g.lineWidth = 1.4;
        g.beginPath(); g.moveTo(sx(i - 1), sy(a)); g.lineTo(sx(i), sy(b)); g.stroke(); g.restore();
      }
      var last = clamp(hh[hh.length - 1], rLo, rHi);
      g.save(); g.fillStyle = conv ? c.post : c.hot; g.beginPath(); g.arc(sx(hh.length - 1), sy(last), 2, 0, 6.2832); g.fill(); g.restore();
    }
  }

  // ==========================================================================
  // 5. shrinkage — hierarchical partial pooling; a violet τ slides the estimates.
  // ==========================================================================

  function shrinkage(root) {
    var NG = 8;
    mount(root, {
      height: 170, hz: 4, scrub: true,
      staticFrame: function (S) { S.tau = 1.0; },
      build: function (S) {
        S.y = []; S.se = []; var i;
        for (i = 0; i < NG; i++) { S.y.push(FV.randn(S.rng) * 1.7); S.se.push(0.5 + 0.6 * S.rng()); }
        S.mu = 0; for (i = 0; i < NG; i++) S.mu += S.y[i]; S.mu /= NG;
        S.phase = 0; S.tau = 1.0; S.touched = false;
      },
      advance: function (S) { if (!S.touched) { S.phase += 0.16; S.tau = Math.exp(Math.sin(S.phase) * 1.5 - 0.4); } },
      render: function (g, S, w, h, T, c) { drawShrink(g, S, w, h, c); },
      // Only commit on an actual drag ("move"). Under scrub/pan-y a scroll-intent
      // touch still fires "down"; acting on it would jump τ and permanently kill
      // the ambient breathing just from swiping past the widget.
      pointer: function (S, x, y, w, h, phase) { if (phase === "move") { S.touched = true; var frac = clamp((x - 12) / (w - 24), 0, 1); S.tau = Math.exp(lerp(-3.2, 1.8, frac)); } }
    });

    function drawShrink(g, S, w, h, c) {
      var pad = { l: 14, r: 14, t: 16, b: 14 };
      var ys = FV.scale([-4.5, 4.5], [h - pad.b, pad.t]);
      var colW = (w - pad.l - pad.r) / NG;
      // grand mean line
      g.save(); g.strokeStyle = c.ink; g.globalAlpha = 0.3; g.setLineDash([4, 4]); g.lineWidth = 1;
      g.beginPath(); g.moveTo(pad.l, ys(S.mu)); g.lineTo(w - pad.r, ys(S.mu)); g.stroke(); g.restore();
      var t2 = S.tau * S.tau;
      for (var j = 0; j < NG; j++) {
        var px = pad.l + colW * (j + 0.5);
        var iv = 1 / (S.se[j] * S.se[j]), pr = iv + 1 / t2;
        var theta = (S.y[j] * iv + S.mu / t2) / pr;
        var psd = Math.sqrt(1 / pr);
        // raw observed (faint hollow)
        g.save(); g.strokeStyle = c.data; g.globalAlpha = 0.3; g.lineWidth = 1;
        g.beginPath(); g.arc(px, ys(S.y[j]), 3, 0, 6.2832); g.stroke(); g.restore();
        // CI
        g.save(); g.strokeStyle = c.data; g.globalAlpha = 0.55; g.lineWidth = 1.5;
        g.beginPath(); g.moveTo(px, ys(theta - psd)); g.lineTo(px, ys(theta + psd)); g.stroke(); g.restore();
        // shrunk estimate
        g.save(); g.fillStyle = c.data; g.beginPath(); g.arc(px, ys(theta), 3.4, 0, 6.2832); g.fill(); g.restore();
      }
      labelRight(g, "τ " + fmtNum(S.tau), w - pad.r, pad.t - 6, c, "flow");
      label(g, S.touched ? "your τ" : "τ breathing", pad.l, pad.t - 6, c);
    }
  }

  // ==========================================================================
  // 6. regression-mini — draggable points + ambient posterior spaghetti.
  // ==========================================================================

  function regressionMini(root) {
    var N = 10, SIG = 0.8, PSD = 2.5, TRAIL = 34, DX = [-3.4, 3.4], DY = [-4.2, 4.2];
    mount(root, {
      height: 170, hz: 4, settleN: 100,
      build: function (S) {
        S.pts = []; var i;
        for (i = 0; i < N; i++) { var x = DX[0] + 0.4 + (DX[1] - DX[0] - 0.8) * i / (N - 1); S.pts.push({ x: x, y: 1.0 * x - 0.2 + FV.randn(S.rng) * 0.7 }); }
        S.a = 0; S.b = 0; S.trail = []; S.drag = -1;
      },
      advance: function (S) { for (var s = 0; s < 3; s++) mh(S); },
      render: function (g, S, w, h, T, c) { drawReg(g, S, w, h, c); },
      pointer: function (S, x, y, w, h, phase, FV, coarse) { return regPtr(S, x, y, w, h, phase, coarse); }
    });

    function xsOf(w) { return FV.scale(DX, [16, w - 10]); }
    function ysOf(h) { return FV.scale(DY, [h - 12, 12]); }
    function logp(S, a, b) {
      var lp = FV.dist.normal.logpdf(a, 0, PSD) + FV.dist.normal.logpdf(b, 0, PSD), i;
      for (i = 0; i < S.pts.length; i++) lp += FV.dist.normal.logpdf(S.pts[i].y, a * S.pts[i].x + b, SIG);
      return lp;
    }
    function mh(S) {
      var pa = S.a + FV.randn(S.rng) * 0.14, pb = S.b + FV.randn(S.rng) * 0.14;
      if (Math.log(S.rng()) < logp(S, pa, pb) - logp(S, S.a, S.b)) { S.a = pa; S.b = pb; }
      S.trail.push([S.a, S.b]); while (S.trail.length > TRAIL) S.trail.shift();
    }
    function drawReg(g, S, w, h, c) {
      var xs = xsOf(w), ys = ysOf(h);
      baseline(g, 16, w - 10, ys(0), c);
      var i, tr = S.trail;
      for (i = 0; i < tr.length; i++) {
        var age = (i + 1) / tr.length, last = i === tr.length - 1;
        var a = tr[i][0], b = tr[i][1];
        g.save(); g.globalAlpha = last ? 1 : 0.1 + 0.35 * age; g.strokeStyle = last ? c.hot : c.post; g.lineWidth = last ? 2 : 1;
        g.beginPath(); g.moveTo(xs(DX[0]), ys(a * DX[0] + b)); g.lineTo(xs(DX[1]), ys(a * DX[1] + b)); g.stroke(); g.restore();
      }
      for (i = 0; i < S.pts.length; i++) { g.save(); g.fillStyle = c.data; g.beginPath(); g.arc(xs(S.pts[i].x), ys(S.pts[i].y), 4, 0, 6.2832); g.fill(); g.restore(); }
      // grabbed-point halo (§A.2)
      if (S.drag >= 0 && S.drag < S.pts.length) {
        var pd = S.pts[S.drag];
        g.save(); g.strokeStyle = c.hot; g.globalAlpha = 0.55; g.lineWidth = 2;
        g.beginPath(); g.arc(xs(pd.x), ys(pd.y), coarsePointer() ? 11 : 9, 0, 6.2832); g.stroke(); g.restore();
      }
      labelRight(g, "slope " + fmtNum(S.a), w - 10, 2, c, "hot");
    }
    function regPtr(S, x, y, w, h, phase, coarse) {
      var xs = xsOf(w), ys = ysOf(h), i;
      if (phase === "down") {
        S.drag = -1;
        // Inflate the hit target on coarse pointers so a thumb can grab a point
        // (§A.2: ≥22 CSS px). Visual dots stay 4px; only the test is generous.
        var hitR = coarse ? 26 : 15, best = hitR * hitR;
        for (i = 0; i < S.pts.length; i++) { var dx = xs(S.pts[i].x) - x, dy = ys(S.pts[i].y) - y, d = dx * dx + dy * dy; if (d < best) { best = d; S.drag = i; } }
        return S.drag >= 0; // claim the gesture only on an actual hit
      }
      else if (phase === "move" && S.drag >= 0) { S.pts[S.drag].x = clamp(xs.invert(x), DX[0], DX[1]); S.pts[S.drag].y = clamp(ys.invert(y), DY[0], DY[1]); return true; }
      else if (phase === "up") S.drag = -1;
      return false;
    }
  }

  // ==========================================================================
  // 7. mixture-resp — 2-Gaussian mixture; points colored by EM responsibility.
  // ==========================================================================

  function mixtureResp(root) {
    var N = 42, DOM = [-4.6, 4.6], STEPS = 16;
    mount(root, {
      height: 170, hz: 2.8,
      staticFrame: function (S) { newRun(S); for (var i = 0; i < 14; i++) emAdv(S, true); },
      build: function (S) { newRun(S); },
      advance: function (S) { emAdv(S, false); },
      render: function (g, S, w, h, T, c) { drawMix(g, S, w, h, T, c); }
    });

    function newRun(S) {
      var tm1 = -1.7 + (S.rng() - 0.5), tm2 = 1.7 + (S.rng() - 0.5), i;
      S.xs = [];
      for (i = 0; i < N; i++) S.xs.push((S.rng() < 0.5 ? tm1 : tm2) + FV.randn(S.rng) * 0.7);
      S.p = { pi: 0.5, m1: -0.6 + S.rng(), s1: 1.0, m2: 0.6 + S.rng(), s2: 1.0 };
      S.r = null; S.toP = clone(S.p); S.fromP = clone(S.p); S.steps = 0;
    }
    function emAdv(S, force) {
      if (S.steps >= STEPS && !force) { S.reloopSeed = nextSeed(S.reloopSeed); S.rng = FV.rng(S.reloopSeed); newRun(S); return; }
      var np = emStep(S.xs, S.p);
      S.fromP = clone(S.p);
      S.p = { pi: np.pi, m1: np.m1, s1: np.s1, m2: np.m2, s2: np.s2 };
      S.toP = clone(S.p); S.r = np.r; S.steps++;
    }
    function drawMix(g, S, w, h, T, c) {
      var pad = { l: 10, r: 10, t: 14, b: 16 };
      var xs = FV.scale(DOM, [pad.l, w - pad.r]);
      var p = { pi: lerp(S.fromP.pi, S.toP.pi, T), m1: lerp(S.fromP.m1, S.toP.m1, T), s1: lerp(S.fromP.s1, S.toP.s1, T), m2: lerp(S.fromP.m2, S.toP.m2, T), s2: lerp(S.fromP.s2, S.toP.s2, T) };
      function c1(x) { return p.pi * Math.exp(FV.dist.normal.logpdf(x, p.m1, p.s1)); }
      function c2(x) { return (1 - p.pi) * Math.exp(FV.dist.normal.logpdf(x, p.m2, p.s2)); }
      var ymax = 1e-6, k, x;
      for (k = 0; k <= 120; k++) { x = DOM[0] + (DOM[1] - DOM[0]) * k / 120; var s = c1(x) + c2(x); if (s > ymax) ymax = s; }
      var ys = FV.scale([0, ymax * 1.25], [h - pad.b, pad.t]);
      baseline(g, pad.l, w - pad.r, ys(0), c);
      FV.curve(g, densCurve(g, DOM, c1, xs, ys, 120), { color: c.prior, width: 1.6 });
      FV.curve(g, densCurve(g, DOM, c2, xs, ys, 120), { color: c.post, width: 1.6 });
      var by = ys(0) + 6;
      for (var i = 0; i < S.xs.length; i++) {
        var r = S.r ? S.r[i] : 0.5;
        g.save(); g.globalAlpha = 0.85; g.fillStyle = blend(c.prior, c.post, r);
        g.beginPath(); g.arc(xs(S.xs[i]), by, 3, 0, 6.2832); g.fill(); g.restore();
      }
      label(g, "EM step " + S.steps, pad.l, pad.t - 4, c);
    }
  }

  // ==========================================================================
  // 8. logistic-boundary — Bayesian logistic regression; wobbling green lines.
  // ==========================================================================

  function logisticBoundary(root) {
    var N = 34, SPA = 14, DOM = [-3.2, 3.2];
    mount(root, {
      height: 175, hz: 5, settleN: 90,
      build: function (S) {
        S.pts = []; var i;
        for (i = 0; i < N; i++) { var cls = S.rng() < 0.5 ? 1 : 0, cx = cls ? 1.1 : -1.1, cy = cls ? 0.9 : -0.9; S.pts.push({ x: cx + FV.randn(S.rng) * 0.9, y: cy + FV.randn(S.rng) * 0.9, c: cls }); }
        S.w = [0, 0.6, 0.6]; S.draws = [];
      },
      advance: function (S) { for (var s = 0; s < 2; s++) wStep(S); },
      render: function (g, S, w, h, T, c) { drawLogi(g, S, w, h, c); }
    });

    function wStep(S) {
      var pw = [S.w[0] + FV.randn(S.rng) * 0.25, S.w[1] + FV.randn(S.rng) * 0.25, S.w[2] + FV.randn(S.rng) * 0.25];
      if (Math.log(S.rng()) < logisticLogPost(pw, S.pts) - logisticLogPost(S.w, S.pts)) S.w = pw;
      S.draws.push(S.w.slice()); while (S.draws.length > SPA) S.draws.shift();
    }
    function drawLogi(g, S, w, h, c) {
      var pad = { l: 10, r: 10, t: 12, b: 10 };
      var xs = FV.scale(DOM, [pad.l, w - pad.r]);
      var ys = FV.scale(DOM, [h - pad.b, pad.t]);
      var i, dr = S.draws;
      for (i = 0; i < dr.length; i++) {
        var wi = dr[i], last = i === dr.length - 1, age = (i + 1) / dr.length;
        if (Math.abs(wi[2]) < 1e-6) continue;
        function yb(x) { return -(wi[0] + wi[1] * x) / wi[2]; }
        g.save(); g.globalAlpha = last ? 1 : 0.12 + 0.4 * age; g.strokeStyle = c.post; g.lineWidth = last ? 2 : 1;
        g.beginPath(); g.moveTo(xs(DOM[0]), ys(clamp(yb(DOM[0]), DOM[0] - 4, DOM[1] + 4))); g.lineTo(xs(DOM[1]), ys(clamp(yb(DOM[1]), DOM[0] - 4, DOM[1] + 4))); g.stroke(); g.restore();
      }
      for (i = 0; i < S.pts.length; i++) {
        g.save(); g.fillStyle = S.pts[i].c ? c.data : c.prior; g.globalAlpha = 0.9;
        g.beginPath(); g.arc(xs(S.pts[i].x), ys(S.pts[i].y), 3.2, 0, 6.2832); g.fill(); g.restore();
      }
      label(g, "posterior boundaries", pad.l, pad.t - 2, c);
    }
  }

  // ==========================================================================
  // 9. elbo-climb — a Gaussian guide climbs a skewed target; ELBO plateaus.
  // ==========================================================================

  function elboClimb(root) {
    var HALF = 0.5 * Math.log(2 * Math.PI * Math.E), DOM = [-4, 4], MAXK = 70;
    function U(x) { var s = x < 0 ? 0.7 : 1.5, z = x / s; return 0.5 * z * z; }
    function dU(x) { var s = x < 0 ? 0.7 : 1.5; return x / (s * s); }

    mount(root, {
      height: 175, hz: 6,
      staticFrame: function (S) { resetE(S); for (var i = 0; i < MAXK; i++) stepE(S); },
      build: function (S) { resetE(S); },
      advance: function (S) { if (S.k < MAXK) stepE(S); else { S.hold = (S.hold || 0) + 1; if (S.hold > 12) { S.reloopSeed = nextSeed(S.reloopSeed); S.rng = FV.rng(S.reloopSeed); resetE(S); } } },
      render: function (g, S, w, h, T, c) { drawElbo(g, S, w, h, c); }
    });

    function resetE(S) { S.mu = -2.5 + 5 * S.rng(); S.L = Math.log(0.3 + 1.4 * S.rng()); S.k = 0; S.hold = 0; S.elbos = []; }
    function stepE(S) {
      var M = 10, gm = 0, gl = 0, el = 0, i, sig = Math.exp(S.L);
      for (i = 0; i < M; i++) { var eps = FV.randn(S.rng), x = S.mu + sig * eps, du = dU(x); gm += -du; gl += -du * sig * eps; el += -U(x); }
      gm /= M; gl = gl / M + 1; el = el / M + HALF + S.L;
      var lr = 0.08;
      S.mu += lr * gm; S.L += lr * gl;
      S.L = clamp(S.L, -2.5, 1.5);
      S.elbos.push(el); while (S.elbos.length > 160) S.elbos.shift();
      S.k++;
    }
    function drawElbo(g, S, w, h, c) {
      var splitY = h * 0.64, pad = { l: 10, r: 10, t: 12 };
      var xs = FV.scale(DOM, [pad.l, w - pad.r]);
      var sig = Math.exp(S.L);
      function tgt(x) { return Math.exp(-U(x)); }
      function guide(x) { return Math.exp(FV.dist.normal.logpdf(x, S.mu, sig)); }
      var ymax = 1e-6, k, x;
      for (k = 0; k <= 120; k++) { x = DOM[0] + (DOM[1] - DOM[0]) * k / 120; var a = tgt(x); if (a > ymax) ymax = a; var b = guide(x); if (b > ymax) ymax = b; }
      var ys = FV.scale([0, ymax * 1.15], [splitY - 8, pad.t]);
      baseline(g, pad.l, w - pad.r, ys(0), c);
      FV.curve(g, densCurve(g, DOM, tgt, xs, ys, 120), { color: c.ink, width: 1.4 });
      var gg = densCurve(g, DOM, guide, xs, ys, 120); fillUnder(g, gg, ys(0), c.post, 0.14); FV.curve(g, gg, { color: c.post, width: 2 });
      // ELBO sparkline
      var e = S.elbos;
      if (e.length > 1) {
        var lo = Infinity, hi = -Infinity, i;
        for (i = 0; i < e.length; i++) { if (e[i] < lo) lo = e[i]; if (e[i] > hi) hi = e[i]; }
        if (hi - lo < 1e-6) hi = lo + 1;
        var sx = FV.scale([0, MAXK], [pad.l, w - pad.r]);
        var sy = FV.scale([lo, hi], [h - 8, splitY + 6]);
        var pts = []; for (i = 0; i < e.length; i++) pts.push([sx(i), sy(e[i])]);
        FV.curve(g, pts, { color: c.flow, width: 1.5 });
      }
      label(g, "guide vs target", pad.l, pad.t - 2, c);
      labelRight(g, "ELBO", w - pad.r, splitY + 4, c, "flow");
    }
  }

  // ==========================================================================
  // 10. abc-eps — prior draws fall; ε shrinks; the accepted cloud tightens.
  // ==========================================================================

  function abcEps(root) {
    var DOM = [-6, 6], MAXK = 26;
    mount(root, {
      height: 175, hz: 3,
      staticFrame: function (S) { resetA(S); S.eps = 0.3; for (var i = 0; i < 40; i++) genAcc(S); },
      build: function (S) { resetA(S); },
      advance: function (S) { stepA(S); },
      render: function (g, S, w, h, T, c) { drawAbc(g, S, w, h, T, c); }
    });

    function resetA(S) { S.yobs = -1.5 + 3 * S.rng(); S.eps = 3.0; S.accepted = []; S.drops = []; S.k = 0; }
    function stepA(S) {
      if (S.k > MAXK) { S.reloopSeed = nextSeed(S.reloopSeed); S.rng = FV.rng(S.reloopSeed); resetA(S); return; }
      S.drops = [];
      for (var i = 0; i < 10; i++) {
        var th = FV.dist.normal.sample(S.rng, 0, 2.5), ok = Math.abs(th - S.yobs) <= S.eps;
        S.drops.push({ th: th, ok: ok, off: S.rng() });
        if (ok) S.accepted.push(th);
      }
      S.eps = Math.max(0.3, S.eps * 0.86); S.k++;
      while (S.accepted.length > 400) S.accepted.shift();
    }
    function genAcc(S) { for (var i = 0; i < 10; i++) { var th = FV.dist.normal.sample(S.rng, 0, 2.5); if (Math.abs(th - S.yobs) <= 0.3) S.accepted.push(th); } }
    function drawAbc(g, S, w, h, T, c) {
      var pad = { l: 10, r: 10 }, splitY = h * 0.56;
      var xs = FV.scale(DOM, [pad.l, w - pad.r]);
      // epsilon band (violet) around observed
      g.save(); g.globalAlpha = 0.14; g.fillStyle = c.flow;
      g.fillRect(xs(S.yobs - S.eps), 12, xs(S.yobs + S.eps) - xs(S.yobs - S.eps), splitY - 12); g.restore();
      // observed marker
      g.save(); g.strokeStyle = c.flow; g.globalAlpha = 0.7; g.lineWidth = 1.5; g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(xs(S.yobs), 12); g.lineTo(xs(S.yobs), splitY); g.stroke(); g.restore();
      // falling drops
      for (var i = 0; i < S.drops.length; i++) {
        var d = S.drops[i], y = lerp(14, splitY - 8, Math.min(1, T + d.off * 0.4));
        g.save(); g.globalAlpha = d.ok ? 0.95 : 0.35 * (1 - T) + 0.2; g.fillStyle = d.ok ? c.post : c.hot;
        g.beginPath(); g.arc(xs(clamp(d.th, DOM[0], DOM[1])), y, 3, 0, 6.2832); g.fill(); g.restore();
      }
      // accepted posterior histogram
      var ys = FV.scale([0, 1], [h - 10, splitY + 8]);
      if (S.accepted.length) {
        var bins = 40, counts = new Array(bins), b, n = 0, lo = DOM[0], hi = DOM[1], bw = (hi - lo) / bins;
        for (b = 0; b < bins; b++) counts[b] = 0;
        for (i = 0; i < S.accepted.length; i++) { var idx = Math.floor((S.accepted[i] - lo) / bw); if (idx >= 0 && idx < bins) { counts[idx]++; n++; } }
        var mx = 1; for (b = 0; b < bins; b++) if (counts[b] > mx) mx = counts[b];
        g.save(); g.globalAlpha = 0.6; g.fillStyle = c.post;
        for (b = 0; b < bins; b++) { if (!counts[b]) continue; var xa = xs(lo + b * bw), xb = xs(lo + (b + 1) * bw), hh = (counts[b] / mx) * (h - 18 - splitY); g.fillRect(xa, h - 10 - hh, xb - xa, hh); }
        g.restore();
      }
      label(g, "ε " + fmtNum(S.eps), pad.l, 2, c);
      labelRight(g, "accepted " + S.accepted.length, w - pad.r, 2, c, "post");
    }
  }

  // ==========================================================================
  // Registration
  // ==========================================================================

  FV.register("dist-strip", function (root) { distStrip(root); });
  FV.register("posterior-morph", function (root) { posteriorMorph(root); });
  FV.register("trace-ticker", function (root) { traceTicker(root); });
  FV.register("rhat-spark", function (root) { rhatSpark(root); });
  FV.register("shrinkage", function (root) { shrinkage(root); });
  FV.register("regression-mini", function (root) { regressionMini(root); });
  FV.register("mixture-resp", function (root) { mixtureResp(root); });
  FV.register("logistic-boundary", function (root) { logisticBoundary(root); });
  FV.register("elbo-climb", function (root) { elboClimb(root); });
  FV.register("abc-eps", function (root) { abcEps(root); });
})();
