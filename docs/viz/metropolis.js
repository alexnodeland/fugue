// docs/viz/metropolis.js — "Random Walks in Posterior Space"
// v2 DATA-FIRST rebuild: Bayesian linear regression, twin-panel.
//   LEFT  (data space):  ~12 draggable (x,y) points; posterior spaghetti — the
//                        last ~60 accepted (slope,intercept) drawn as thin green
//                        lines through the data; current fit coral; rejected
//                        proposals flash coral-dashed and vanish.
//   RIGHT (param space):  live 2-D posterior heatmap over (slope, intercept),
//                        recomputed whenever a data point moves (offscreen grid,
//                        rebuilt only when dirty, so the sampler stays butter-
//                        smooth); chain trails, proposal arrows, accept/reject.
// Model: y ~ Normal(a*x + b, 0.8), priors a,b ~ Normal(0, 2.5). σ_obs fixed 0.8.
// Diagnostics (acceptance / split-R̂ / ESS) ported from src/inference/{diagnostics,
// mcmc_utils}.rs. Self-contained IIFE; assumes fugue-viz.js has loaded first.
(function () {
  "use strict";
  if (typeof window === "undefined" || !window.FugueViz) return;
  var FV = window.FugueViz;

  // ---- The model. logp(a,b) is THE log posterior over (slope, intercept). ----
  var SIGMA_OBS = 0.8;   // fixed observation noise (stated on the page)
  var PRIOR_SD = 2.5;    // Normal(0, 2.5) prior on both params

  // Fixed plot domains (kept stable so dragging never makes the axes jump).
  // Windows shared verbatim with viz/hmc.js — the two pages present the SAME
  // regression problem so the MH→HMC comparison is apples-to-apples.
  var DX = [-3.6, 3.6], DY = [-4.8, 3.2];   // data space
  var PA = [-0.2, 2.0], PB = [-2.5, 1.2];   // param space: slope × intercept

  // Dispersed chain starts so several chains reveal (or fail to reveal) mixing.
  var STARTA = [0.0, 1.8, 1.2, 0.3];
  var STARTB = [1.0, -2.2, 0.2, -1.5];

  var SPAGHETTI = 60;   // accepted fits retained as green lines
  var MAXSAMP = 1200;   // capped sample history per chain used for diagnostics
  var MAXTRAIL = 320;   // capped param-space trail length for drawing
  var LAGCAP = 256;     // autocorrelation lag cap (Rust uses 2048; see report)

  // ===== Diagnostics ported from src/inference/diagnostics.rs + mcmc_utils.rs ===
  function splitChains(chs) {
    var out = [];
    for (var i = 0; i < chs.length; i++) {
      var c = chs[i], half = Math.floor(c.length / 2);
      if (half === 0) { out.push(c.slice()); continue; }
      out.push(c.slice(0, half));
      out.push(c.slice(half, 2 * half));
    }
    return out;
  }
  function rhatFrom(ch) {
    if (ch.length < 2) return 1.0;
    for (var i = 0; i < ch.length; i++) if (ch[i].length === 0) return NaN;
    var m = ch.length, n = ch[0].length, k;
    var means = ch.map(function (v) { var s = 0; for (k = 0; k < v.length; k++) s += v[k]; return s / v.length; });
    var overall = 0; for (i = 0; i < m; i++) overall += means[i]; overall /= m;
    var b = 0; for (i = 0; i < m; i++) { var d = means[i] - overall; b += d * d; } b *= n / (m - 1);
    var w = 0;
    for (i = 0; i < m; i++) {
      var v = ch[i], mu = means[i], s = 0;
      for (k = 0; k < v.length; k++) { var e = v[k] - mu; s += e * e; }
      w += s / (n - 1);
    }
    w /= m;
    var varplus = ((n - 1) / n) * w + (1 / n) * b;
    return Math.sqrt(varplus / w);
  }
  function splitRhat(chs) { return rhatFrom(splitChains(chs)); }

  function autocov(x, maxLag) {
    var n = x.length, mean = 0, i;
    for (i = 0; i < n; i++) mean += x[i]; mean /= n;
    var c = new Array(n); for (i = 0; i < n; i++) c[i] = x[i] - mean;
    var acov = new Array(maxLag + 1);
    for (var lag = 0; lag <= maxLag; lag++) {
      var s = 0; for (i = 0; i < n - lag; i++) s += c[i] * c[i + lag];
      acov[lag] = s / n;
    }
    return acov;
  }
  // Multi-chain ESS (Vehtari et al. 2021 / Stan), Geyer initial positive sequence.
  function essFromChains(chains) {
    var m = chains.length; if (m === 0) return 0;
    var n = chains[0].length, i;
    var uneq = false; for (i = 0; i < m; i++) if (chains[i].length !== n) uneq = true;
    if (n < 4 || uneq) { var tot = 0; for (i = 0; i < m; i++) tot += chains[i].length; return Math.max(tot, 1); }
    var maxLag = Math.min(n - 1, LAGCAP);
    var acovs = chains.map(function (c) { return autocov(c, maxLag); });
    var nf = n, mf = m;
    var means = chains.map(function (c) { var s = 0; for (var k = 0; k < c.length; k++) s += c[k]; return s / nf; });
    var vars = acovs.map(function (a) { return a[0] * nf / (nf - 1); });
    var meanVar = 0; for (i = 0; i < m; i++) meanVar += vars[i]; meanVar /= mf;
    if (meanVar <= 0) return m * n;
    var varplus = meanVar * (nf - 1) / nf;
    if (m > 1) {
      var overall = 0; for (i = 0; i < m; i++) overall += means[i]; overall /= mf;
      var between = 0; for (i = 0; i < m; i++) { var d = means[i] - overall; between += d * d; } between /= (mf - 1);
      varplus += between;
    }
    var rho = function (t) { var s = 0; for (var j = 0; j < m; j++) s += acovs[j][t]; s /= mf; return 1 - (meanVar - s) / varplus; };
    var rhoHat = new Array(maxLag + 1); for (i = 0; i <= maxLag; i++) rhoHat[i] = 0; rhoHat[0] = 1;
    if (maxLag >= 1) rhoHat[1] = rho(1);
    var t = 1, maxT = Math.min(1, maxLag);
    while (t + 2 <= maxLag) {
      var re = rho(t + 1), ro = rho(t + 2);
      if (re + ro < 0) break;
      rhoHat[t + 1] = re; rhoHat[t + 2] = ro; maxT = t + 2; t += 2;
    }
    var kk = 1;
    while (kk + 2 <= maxT) {
      var prev = rhoHat[kk - 1] + rhoHat[kk], cur = rhoHat[kk + 1] + rhoHat[kk + 2];
      if (cur > prev) { var avg = prev / 2; rhoHat[kk + 1] = avg; rhoHat[kk + 2] = avg; }
      kk += 2;
    }
    var sum = 0; for (i = 0; i <= maxT; i++) sum += rhoHat[i];
    var tau = Math.max(-1 + 2 * sum, 1);
    return (m * n) / tau;
  }

  // small hex/rgb parser (FugueViz keeps its own private; we need one for the ramp)
  function toRgb(col) {
    col = (col || "").trim();
    var m = /^#?([0-9a-f]{6})$/i.exec(col);
    if (m) { var n = parseInt(m[1], 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
    var r = /rgba?\(([^)]+)\)/.exec(col);
    if (r) { var p = r[1].split(","); return [parseInt(p[0], 10), parseInt(p[1], 10), parseInt(p[2], 10)]; }
    return [86, 211, 100];
  }

  FV.register("metropolis", function (root, FV) {
    // ------------------------------------------------------------------ state
    var seed0 = parseInt(root.getAttribute("data-seed") || "11", 10);
    var params = { sigma: 0.35, nChains: 3, speed: 8, seed: seed0 };
    var pts = [];              // draggable data points {x, y}
    var chains = [];
    var spaghetti = [];        // recent accepted [a, b] across all chains
    var accProp = 0, accAcc = 0;
    var rand = FV.rng(seed0 >>> 0);
    var diag = { rhat: NaN, ess: 0, acc: 0 };
    var acc = 0;               // fractional step accumulator for play mode
    var lastDiag = 0;

    // The regression log posterior over (slope a, intercept b).
    function logp(a, b) {
      var lp = FV.dist.normal.logpdf(a, 0, PRIOR_SD) + FV.dist.normal.logpdf(b, 0, PRIOR_SD);
      for (var i = 0; i < pts.length; i++) {
        lp += FV.dist.normal.logpdf(pts[i].y, a * pts[i].x + b, SIGMA_OBS);
      }
      return lp;
    }

    // Seeded default dataset, identical to viz/hmc.js makeSeedData(11):
    // true line y = 0.8·x − 0.4 + Normal(0, 0.8), 12 points evenly on [-3, 3].
    function makeData() {
      var dr = FV.rng(11);
      pts = [];
      var N = 12;
      for (var i = 0; i < N; i++) {
        var x = -3 + 6 * i / (N - 1);
        var y = 0.8 * x - 0.4 + 0.8 * FV.randn(dr);
        if (y < DY[0] + 0.3) y = DY[0] + 0.3;
        if (y > DY[1] - 0.3) y = DY[1] - 0.3;
        pts.push({ x: x, y: y });
      }
    }

    function newChains() {
      rand = FV.rng(params.seed >>> 0);
      chains = [];
      spaghetti = [];
      for (var i = 0; i < params.nChains; i++) {
        var a = STARTA[i % 4], b = STARTB[i % 4];
        chains.push({ a: a, b: b, lp: logp(a, b), as: [a], bs: [b], trail: [[a, b]], flash: null });
      }
      accProp = 0; accAcc = 0; acc = 0;
      diag = { rhat: NaN, ess: 0, acc: 0 };
      lastDiag = 0;
    }

    // Data changed (drag / reseed): posterior moved, so every chain's cached
    // log-density is stale — recompute it and mark the heatmap dirty.
    function reweightChains() {
      for (var i = 0; i < chains.length; i++) chains[i].lp = logp(chains[i].a, chains[i].b);
      heatDirty = true;
    }

    function doStep() {
      var s = params.sigma;
      for (var i = 0; i < chains.length; i++) {
        var ch = chains[i];
        var pa = ch.a + s * FV.randn(rand);
        var pb = ch.b + s * FV.randn(rand);
        var plp = logp(pa, pb);
        var logA = plp - ch.lp;               // symmetric proposal: ratio of targets
        var accept = Math.log(rand() + 1e-300) < logA;
        accProp++;
        ch.flash = { oa: ch.a, ob: ch.b, pa: pa, pb: pb, accepted: accept, life: 1 };
        if (accept) {
          ch.a = pa; ch.b = pb; ch.lp = plp; accAcc++;
          spaghetti.push([pa, pb]);
          if (spaghetti.length > SPAGHETTI) spaghetti.shift();
        }
        ch.as.push(ch.a); ch.bs.push(ch.b);
        if (ch.as.length > MAXSAMP) { ch.as.shift(); ch.bs.shift(); }
        ch.trail.push([ch.a, ch.b]);
        if (ch.trail.length > MAXTRAIL) ch.trail.shift();
      }
    }

    function refreshDiag(now) {
      if (now - lastDiag < 250) return;
      lastDiag = now;
      var as = [], bs = [], i;
      for (i = 0; i < chains.length; i++) { as.push(chains[i].as); bs.push(chains[i].bs); }
      var ra = splitRhat(as), rb = splitRhat(bs);
      var rhat = Math.max(isFinite(ra) ? ra : -Infinity, isFinite(rb) ? rb : -Infinity);
      if (!isFinite(rhat)) rhat = NaN;
      var ess = Math.min(essFromChains(as), essFromChains(bs)); // worst coordinate
      diag = { rhat: rhat, ess: ess, acc: accProp ? accAcc / accProp : 0 };
      renderReadouts();
    }

    // --------------------------------------------------------------- DOM shell
    var controls = document.createElement("div");
    controls.className = "fv-controls";
    root.appendChild(controls);

    // proposal sigma on a log scale (0.01 .. 5)
    var LOGLO = Math.log(0.01), LOGHI = Math.log(5);
    function sigmaOf(t) { return Math.exp(LOGLO + t * (LOGHI - LOGLO)); }
    FV.slider(controls, {
      label: "PROPOSAL σ", min: 0, max: 1, step: 0.001, value: (Math.log(params.sigma) - LOGLO) / (LOGHI - LOGLO),
      fmt: function (t) { return sigmaOf(t).toFixed(2); },
      onInput: function (t) { params.sigma = sigmaOf(t); requestDraw(); }
    });

    FV.slider(controls, {
      label: "CHAINS", min: 1, max: 4, step: 1, value: params.nChains,
      fmt: function (v) { return String(v | 0); },
      onInput: function (v) { params.nChains = v | 0; newChains(); renderReadouts(); requestDraw(); }
    });

    FV.slider(controls, {
      label: "SPEED", min: 1, max: 40, step: 1, value: params.speed,
      fmt: function (v) { return (v | 0) + "/s"; },
      onInput: function (v) { params.speed = v | 0; }
    });

    FV.slider(controls, {
      label: "SEED", min: 1, max: 99, step: 1, value: params.seed,
      fmt: function (v) { return String(v | 0); },
      onInput: function (v) { params.seed = v | 0; newChains(); renderReadouts(); requestDraw(); }
    });

    var btns = FV.buttons(controls, [
      { label: "Play", title: "Run the chains", primary: true, onClick: function () { togglePlay(); } },
      { label: "Step", title: "One proposal per chain", onClick: function () { loopApi.step(); } },
      { label: "Reset", title: "Restart chains from dispersed seeds", onClick: function () { newChains(); renderReadouts(); requestDraw(); } }
    ]);

    var cv = FV.canvas(root, { height: 400, onResize: function () { heatDirty = true; draw(); } });
    var ctx = cv.ctx;

    var instr = document.createElement("div");
    instr.className = "fv-instruction";
    instr.textContent = "drag a yellow point — the posterior heatmap (right) and the chain react instantly · coral = current fit · green = recent accepted fits";
    root.appendChild(instr);

    var readouts = document.createElement("div");
    readouts.className = "fv-readouts";
    root.appendChild(readouts);
    var rAcc = FV.readout(readouts, { label: "ACCEPTANCE" });
    var rRhat = FV.readout(readouts, { label: "SPLIT-R̂" });
    var rEss = FV.readout(readouts, { label: "ESS" });

    var hint = document.createElement("div");
    hint.className = "fv-hint";
    hint.textContent = "try: the chains are already walking — drag a point far from the line and the heatmap morphs and the whole chain migrates to the new best fit, live.";
    root.appendChild(hint);

    function renderReadouts() {
      var a = diag.acc;
      rAcc.set((a * 100).toFixed(1) + "%", (a >= 0.2 && a <= 0.5) ? "post" : "hot");
      if (isFinite(diag.rhat)) rRhat.set(diag.rhat.toFixed(3), diag.rhat < 1.1 ? "post" : "hot");
      else rRhat.set("—");
      rEss.set(diag.ess >= 1 ? diag.ess.toFixed(0) : "—");
    }

    // -------------------------------------------------------------- heatmap
    var heatDirty = true;
    var heatCanvas = document.createElement("canvas");
    var heatCtx = heatCanvas.getContext("2d");
    // Rebuild the offscreen posterior heatmap over (slope, intercept). Called
    // only when the data changed / panel resized — never every animation frame,
    // so the sampler runs smoothly and a drag still updates same-frame.
    function rebuildHeat(iw, ih, col) {
      iw = Math.max(1, Math.round(iw)); ih = Math.max(1, Math.round(ih));
      heatCanvas.width = iw; heatCanvas.height = ih;
      var step = 4;
      var cols = Math.ceil(iw / step), rows = Math.ceil(ih / step);
      var vals = new Array(cols * rows), maxv = -Infinity, ix, iy;
      for (iy = 0; iy < rows; iy++) {
        var b = PB[1] - ((iy * step + step / 2) / ih) * (PB[1] - PB[0]);
        for (ix = 0; ix < cols; ix++) {
          var a = PA[0] + ((ix * step + step / 2) / iw) * (PA[1] - PA[0]);
          var v = logp(a, b);
          if (!isFinite(v)) v = -Infinity;
          vals[iy * cols + ix] = v;
          if (v > maxv) maxv = v;
        }
      }
      var rgb = toRgb(col);
      heatCtx.clearRect(0, 0, iw, ih);
      if (isFinite(maxv)) {
        for (iy = 0; iy < rows; iy++) {
          for (ix = 0; ix < cols; ix++) {
            var lv = vals[iy * cols + ix];
            var al = lv === -Infinity ? 0 : Math.exp(lv - maxv);
            if (al <= 0.004) continue;
            if (al > 1) al = 1;
            heatCtx.fillStyle = "rgba(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + "," + al + ")";
            heatCtx.fillRect(ix * step, iy * step, step, step);
          }
        }
      }
      heatDirty = false;
    }

    // ---------------------------------------------------------------- layout
    // Returns {data:{...plot}, param:{...plot}} where each plot carries its inner
    // rect + scales. Side-by-side (~55/45) on wide, stacked on narrow.
    var dataPlot = null, paramPlot = null;
    function layout(w, h) {
      var gap = 16, stacked = w < 560;
      var dRect, pRect;
      if (stacked) {
        var ph = (h - gap) / 2;
        dRect = { x: 0, y: 0, w: w, h: ph };
        pRect = { x: 0, y: ph + gap, w: w, h: ph };
      } else {
        var dw = (w - gap) * 0.55;
        dRect = { x: 0, y: 0, w: dw, h: h };
        pRect = { x: dw + gap, y: 0, w: (w - gap) - dw, h: h };
      }
      dataPlot = mkPlot(dRect, DX, DY);
      paramPlot = mkPlot(pRect, PA, PB);
    }
    function mkPlot(rect, domX, domY) {
      var pad = { l: 38, r: 10, t: 12, b: 26 };
      var iw = Math.max(10, rect.w - pad.l - pad.r);
      var ih = Math.max(10, rect.h - pad.t - pad.b);
      var ix = rect.x + pad.l, iy = rect.y + pad.t;
      return {
        rect: rect, ix: ix, iy: iy, iw: iw, ih: ih,
        sx: FV.scale(domX, [ix, ix + iw]),
        sy: FV.scale(domY, [iy + ih, iy])
      };
    }

    // ------------------------------------------------------------------- draw
    function drawArrow(x0, y0, x1, y1, color, alpha, dash) {
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.5;
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      ctx.setLineDash([]);
      var ang = Math.atan2(y1 - y0, x1 - x0), hl = 6;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - hl * Math.cos(ang - 0.4), y1 - hl * Math.sin(ang - 0.4));
      ctx.lineTo(x1 - hl * Math.cos(ang + 0.4), y1 - hl * Math.sin(ang + 0.4));
      ctx.closePath(); ctx.fill();
      ctx.restore();
    }
    function dotRaw(x, y, r, color) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); }
    function dot(x, y, r, color, alpha) { ctx.save(); ctx.globalAlpha = alpha; dotRaw(x, y, r, color); ctx.restore(); }

    // a fit-line y = a*x + b clipped to the data panel
    function fitLine(p, a, b, color, width, alpha, dash) {
      var y0 = a * DX[0] + b, y1 = a * DX[1] + b;
      ctx.save();
      ctx.beginPath();
      ctx.rect(p.ix, p.iy, p.iw, p.ih); ctx.clip();
      ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width;
      ctx.lineCap = "round";
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(p.sx(DX[0]), p.sy(y0));
      ctx.lineTo(p.sx(DX[1]), p.sy(y1));
      ctx.stroke();
      ctx.restore();
    }

    function draw() {
      if (!cv) return; // onResize can fire during canvas() before cv is assigned
      var w = cv.w, h = cv.h;
      cv.clear();
      var th = FV.theme(), C = th.colors;
      layout(w, h);
      var dp = dataPlot, pp = paramPlot;

      // ---- PARAM panel: posterior heatmap ----
      if (heatDirty || heatCanvas.width !== Math.round(pp.iw) || heatCanvas.height !== Math.round(pp.ih)) {
        rebuildHeat(pp.iw, pp.ih, C.post);
      }
      ctx.drawImage(heatCanvas, pp.ix, pp.iy, pp.iw, pp.ih);
      FV.axes(ctx, { x: pp.ix, y: pp.iy, w: pp.iw, h: pp.ih, xscale: pp.sx, yscale: pp.sy, xlabel: "slope a", ylabel: "intercept b", theme: th });

      // param trails (the "recording" so far) — ink, faded
      ctx.save();
      ctx.globalAlpha = 0.45;
      for (var i = 0; i < chains.length; i++) {
        var tr = chains[i].trail, ptsPix = new Array(tr.length);
        for (var j = 0; j < tr.length; j++) ptsPix[j] = [pp.sx(tr[j][0]), pp.sy(tr[j][1])];
        FV.curve(ctx, ptsPix, { color: C.ink, width: 1.1 });
      }
      ctx.restore();

      // param proposal flashes: blue ghost arrow + green/coral outcome
      for (i = 0; i < chains.length; i++) {
        var fl = chains[i].flash;
        if (!fl || fl.life <= 0) continue;
        var ax0 = pp.sx(fl.oa), ay0 = pp.sy(fl.ob), ax1 = pp.sx(fl.pa), ay1 = pp.sy(fl.pb);
        drawArrow(ax0, ay0, ax1, ay1, C.prior, 0.5 * fl.life, [4, 3]);
        if (fl.accepted) dot(ax1, ay1, 4, C.post, 0.9 * fl.life);
        else { dot(ax1, ay1, 3.5, C.hot, 0.85 * fl.life); drawArrow(ax1, ay1, ax0, ay0, C.hot, 0.35 * fl.life, [2, 3]); }
      }

      // param current states — coral with a glow
      for (i = 0; i < chains.length; i++) {
        var ch = chains[i], cx = pp.sx(ch.a), cy = pp.sy(ch.b);
        ctx.save(); ctx.globalAlpha = 0.28; dotRaw(cx, cy, 9, C.hot); ctx.restore();
        dot(cx, cy, 4.5, C.hot, 1);
        ctx.save();
        ctx.strokeStyle = th.dark ? "rgba(13,17,23,0.9)" : "rgba(255,255,255,0.9)";
        ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(cx, cy, 4.5, 0, 2 * Math.PI); ctx.stroke();
        ctx.restore();
      }

      // ---- DATA panel: spaghetti + points ----
      FV.axes(ctx, { x: dp.ix, y: dp.iy, w: dp.iw, h: dp.ih, xscale: dp.sx, yscale: dp.sy, xlabel: "x", ylabel: "y", theme: th });

      // posterior spaghetti: recent accepted fits, oldest faint -> newest bright
      var N = spaghetti.length;
      for (i = 0; i < N; i++) {
        var age = (i + 1) / N;
        fitLine(dp, spaghetti[i][0], spaghetti[i][1], C.post, 1.1, 0.10 + 0.42 * age);
      }

      // rejected-proposal flashes in data space: coral dashed, vanishing
      for (i = 0; i < chains.length; i++) {
        var f2 = chains[i].flash;
        if (f2 && f2.life > 0 && !f2.accepted) {
          fitLine(dp, f2.pa, f2.pb, C.hot, 1.2, 0.5 * f2.life, [5, 4]);
        }
      }

      // current fit(s) — coral, thicker
      for (i = 0; i < chains.length; i++) {
        fitLine(dp, chains[i].a, chains[i].b, C.hot, 2.2, 0.95);
      }

      // data points — yellow, draggable
      for (i = 0; i < pts.length; i++) {
        var px = dp.sx(pts[i].x), py = dp.sy(pts[i].y);
        var active = (dragIdx === i);
        ctx.save();
        ctx.globalAlpha = active ? 0.35 : 0.22; dotRaw(px, py, active ? 11 : 8, C.data); ctx.restore();
        dot(px, py, active ? 5.5 : 4.5, C.data, 1);
        ctx.save();
        ctx.strokeStyle = th.dark ? "rgba(13,17,23,0.9)" : "rgba(255,255,255,0.9)";
        ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(px, py, active ? 5.5 : 4.5, 0, 2 * Math.PI); ctx.stroke();
        ctx.restore();
      }

      // decay flashes for the next frame
      for (i = 0; i < chains.length; i++) if (chains[i].flash) chains[i].flash.life -= 0.05;
    }

    // ------------------------------------------------------- pointer / dragging
    var dragIdx = -1;
    function evtPos(e) {
      var r = cv.el.getBoundingClientRect();
      var cx = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
      var cy = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
      return { x: cx, y: cy };
    }
    function hitTest(pos) {
      if (!dataPlot) return -1;
      var dp = dataPlot, best = -1, bestD = 15 * 15;
      for (var i = 0; i < pts.length; i++) {
        var dx = dp.sx(pts[i].x) - pos.x, dy = dp.sy(pts[i].y) - pos.y;
        var d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = i; }
      }
      return best;
    }
    function onDown(e) {
      var pos = evtPos(e);
      var hit = hitTest(pos);
      if (hit >= 0) {
        dragIdx = hit;
        cv.el.style.cursor = "grabbing";
        if (e.cancelable) e.preventDefault();
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
        window.addEventListener("touchmove", onMove, { passive: false });
        window.addEventListener("touchend", onUp);
      }
    }
    function onMove(e) {
      if (dragIdx < 0 || !dataPlot) return;
      var pos = evtPos(e), dp = dataPlot;
      var nx = dp.sx.invert(pos.x), ny = dp.sy.invert(pos.y);
      if (nx < DX[0]) nx = DX[0]; if (nx > DX[1]) nx = DX[1];
      if (ny < DY[0]) ny = DY[0]; if (ny > DY[1]) ny = DY[1];
      pts[dragIdx].x = nx; pts[dragIdx].y = ny;
      reweightChains();       // posterior moved -> heatDirty + stale lp fixed
      requestDraw();
      if (e.cancelable) e.preventDefault();
    }
    function onUp() {
      dragIdx = -1;
      cv.el.style.cursor = "";
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchmove", onMove);
      window.removeEventListener("touchend", onUp);
      requestDraw();
    }
    function onHover(e) {
      if (dragIdx >= 0) return;
      cv.el.style.cursor = hitTest(evtPos(e)) >= 0 ? "grab" : "";
    }
    cv.el.addEventListener("mousedown", onDown);
    cv.el.addEventListener("touchstart", onDown, { passive: false });
    cv.el.addEventListener("mousemove", onHover);

    // schedule a single draw when paused (during play the loop already draws)
    var drawQueued = false;
    function requestDraw() {
      if (loopApi.playing) return;
      if (drawQueued) return;
      drawQueued = true;
      window.requestAnimationFrame(function () { drawQueued = false; draw(); });
    }

    // ------------------------------------------------------------------- loop
    // autoplay: start walking the moment the widget scrolls into view (the loop
    // honors reduced-motion internally — no animation there, just the pre-warmed
    // static frame below).
    var loopApi = FV.loop(root, function (dt) {
      if (dt === 0) { doStep(); }        // Step button (or reduced-motion)
      else {
        acc += dt * params.speed;
        var n = Math.floor(acc);
        if (n > 0) { acc -= n; if (n > 60) n = 60; for (var i = 0; i < n; i++) doStep(); }
      }
      refreshDiag(nowMs());
      draw();
    }, { autoplay: true });

    function nowMs() { return (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now(); }

    function togglePlay() {
      if (loopApi.playing) { loopApi.pause(); btns.fvButtons["Play"].textContent = "Play"; }
      else { loopApi.play(); if (loopApi.playing) btns.fvButtons["Play"].textContent = "Pause"; }
    }

    if (loopApi.reduced) {
      btns.fvButtons["Play"].textContent = "Play";
      btns.fvButtons["Play"].disabled = true;
      btns.fvButtons["Play"].title = "Reduced motion is on — use Step";
      btns.fvButtons["Step"].classList.add("fv-primary");
    }

    FV.onThemeChange(function () { heatDirty = true; draw(); });

    // first paint: pre-warm ~40 burn-in steps so the posterior spaghetti and the
    // param-space trails already exist at first paint (and so reduced-motion, which
    // never animates, still shows a rich converged frame — never an empty axis).
    makeData();
    newChains();
    for (var pw = 0; pw < 40; pw++) doStep();
    refreshDiag(nowMs());
    renderReadouts();
    draw();
    // The loop autoplays itself (see FV.loop {autoplay:true}); reflect that on the
    // button. play() already ran and is a no-op under reduced motion.
    if (loopApi.playing) btns.fvButtons["Play"].textContent = "Pause";
  });
})();
