// docs/viz/hmc.js — "Rolling, Not Guessing: Hamiltonian Monte Carlo" (v2, data-first).
// Self-contained IIFE; consumes window.FugueViz (loaded first via book.toml).
//
// Twin panel over a real Bayesian linear regression — the SAME seeded dataset and
// model as the Metropolis explorable, so a reader flowing metropolis -> hmc meets
// the same problem instantly:
//     y_i ~ Normal(a*x_i + b, sigma_obs),  a ~ Normal(0, 2.5),  b ~ Normal(0, 2.5)
//     sigma_obs fixed at 0.8.
//
//   LEFT  (data space):  12 draggable yellow points; posterior SPAGHETTI — the last
//                        accepted (slope,intercept) samples as thin green lines through
//                        the cloud; the current sample's line coral and thick; on a
//                        rejected/divergent proposal the proposed line flashes coral-dashed.
//   RIGHT (param space): live 2D posterior heatmap over (slope, intercept), recomputed
//                        whenever a data point moves; the HMC state rolls through it with
//                        a violet leapfrog trajectory + momentum arrow; divergences glow
//                        coral; a compact energy strip-chart sits below.
//
// The link is the lesson: a point in parameter space (right) IS a line in data space
// (left). MH side-by-side runs a random-walk chain at a MATCHED gradient budget and
// paints its spaghetti in DIMMER green, so HMC's coverage advantage shows in data space.
(function () {
  "use strict";

  // ------------------------------------------------------------------
  // The regression problem — dataset + model. Hardcoded IDENTICALLY to the
  // metropolis page: seed 11, 12 points, x evenly spaced in [-3, 3], true
  // slope 0.8, intercept -0.4, observation noise 0.8.
  // ------------------------------------------------------------------
  var N = 12;
  var A_TRUE = 0.8, B_TRUE = -0.4, NOISE = 0.8, DATA_SEED = 11;
  var SIGMA_OBS = 0.8;      // fixed observation noise (stated on the page)
  var PRIOR_SD = 2.5;       // Normal(0, 2.5) prior on both params
  var INV_S2 = 1 / (SIGMA_OBS * SIGMA_OBS);
  var INV_T2 = 1 / (PRIOR_SD * PRIOR_SD);

  // Parameter-space window over (slope a, intercept b).
  var DOM_A = [-0.2, 2.0];
  var DOM_B = [-2.5, 1.2];
  // Data-space window.
  var DAT_X = [-3.6, 3.6];
  var DAT_Y = [-4.8, 3.2];
  var DIV_THRESHOLD = 1000;    // |dH| beyond this => divergent (Stan's default)
  var SPAG_CAP = 60;           // spaghetti lines retained per sampler

  function makeSeedData(seed) {
    var rng = FugueViz.rng(seed >>> 0);
    var xs = new Array(N), ys = new Array(N);
    for (var i = 0; i < N; i++) {
      var x = -3 + 6 * i / (N - 1);
      xs[i] = x;
      ys[i] = A_TRUE * x + B_TRUE + NOISE * FugueViz.randn(rng);
    }
    return { xs: xs, ys: ys };
  }

  // Log-posterior over (a, b) for the current data. Gaussian likelihood +
  // Gaussian priors — proper, honest math (no conjugacy shortcut in the sampler).
  function logpost(a, b, D) {
    var xs = D.xs, ys = D.ys, s = 0;
    for (var i = 0; i < N; i++) {
      var r = ys[i] - (a * xs[i] + b);
      s += -0.5 * INV_S2 * r * r;
    }
    s += -0.5 * INV_T2 * (a * a + b * b);
    return s - N * (Math.log(SIGMA_OBS) + 0.918938533204673); // + const (0.5 ln 2pi)
  }
  // Analytic gradient of the log-posterior (verified vs finite differences).
  function gradLogpost(a, b, D) {
    var xs = D.xs, ys = D.ys, ga = 0, gb = 0;
    for (var i = 0; i < N; i++) {
      var r = (ys[i] - (a * xs[i] + b)) * INV_S2;
      ga += r * xs[i];
      gb += r;
    }
    ga -= a * INV_T2;
    gb -= b * INV_T2;
    return [ga, gb];
  }

  // ------------------------------------------------------------------
  // Effective sample size — ported from src/inference/mcmc_utils.rs
  // (single-chain Geyer initial-positive + monotone sequence estimator).
  // ------------------------------------------------------------------
  function autocov(x, maxLag) {
    var n = x.length, i, lag;
    var mean = 0;
    for (i = 0; i < n; i++) mean += x[i];
    mean /= n;
    var c = new Array(n);
    for (i = 0; i < n; i++) c[i] = x[i] - mean;
    var out = new Array(maxLag + 1);
    for (lag = 0; lag <= maxLag; lag++) {
      var s = 0;
      for (i = 0; i < n - lag; i++) s += c[i] * c[i + lag];
      out[lag] = s / n;
    }
    return out;
  }
  function essSingle(x) {
    var n = x.length;
    if (n < 4) return n;
    var maxLag = Math.min(n - 1, 2048);
    var acov = autocov(x, maxLag);
    var W = acov[0] * n / (n - 1);
    if (!(W > 0)) return n;
    var varPlus = W * (n - 1) / n;
    function rho(t) { return 1 - (W - acov[t]) / varPlus; }
    var rhoHat = new Array(maxLag + 1), k;
    for (k = 0; k <= maxLag; k++) rhoHat[k] = 0;
    rhoHat[0] = 1;
    if (maxLag >= 1) rhoHat[1] = rho(1);
    var t = 1, maxT = Math.min(1, maxLag);
    while (t + 2 <= maxLag) {
      var re = rho(t + 1), ro = rho(t + 2);
      if (re + ro < 0) break;
      rhoHat[t + 1] = re; rhoHat[t + 2] = ro;
      maxT = t + 2; t += 2;
    }
    k = 1;
    while (k + 2 <= maxT) {
      var prev = rhoHat[k - 1] + rhoHat[k];
      var cur = rhoHat[k + 1] + rhoHat[k + 2];
      if (cur > prev) { var avg = prev / 2; rhoHat[k + 1] = avg; rhoHat[k + 2] = avg; }
      k += 2;
    }
    var sum = 0;
    for (k = 0; k <= maxT; k++) sum += rhoHat[k];
    var tau = Math.max(1, -1 + 2 * sum);
    return n / tau;
  }

  // Expose the math for the node gate to import.
  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      makeSeedData: makeSeedData, logpost: logpost, gradLogpost: gradLogpost,
      essSingle: essSingle, N: N, SIGMA_OBS: SIGMA_OBS, PRIOR_SD: PRIOR_SD
    };
  }

  // ------------------------------------------------------------------
  if (typeof FugueViz === "undefined" || !FugueViz.register) return;

  FugueViz.register("hmc", function (root, FV) {
    var seed0 = parseInt(root.getAttribute("data-seed"), 10);
    if (!isFinite(seed0)) seed0 = DATA_SEED;

    // ---- tunables / chain state ----
    var eps = 0.08, L = 25, speed = 14, sideBySide = false;
    var curSeed = seed0 >>> 0;
    var data = makeSeedData(DATA_SEED);      // dataset is fixed to seed 11
    var rng = FV.rng(curSeed);
    function nrm() { return FV.randn(rng); }

    var q, logpCur, trail, samplesA, spag, accepts, total, divergences, lastDH;
    var active = null, reveal = 0, applied = false, flash = null, rejLine = null;

    // MH side chain (matched budget: one proposal per leapfrog step)
    var MH_SIGMA = 0.12;
    var mhQ, mhLogp, mhTrail, mhSpag, mhAccepts, mhTotal;

    function pushCapped(arr, v, cap) { arr.push(v); if (arr.length > cap) arr.shift(); }

    function resetChain() {
      rng = FV.rng(curSeed);
      q = [0.0, 0.0];
      logpCur = logpost(q[0], q[1], data);
      trail = [[q[0], q[1]]];
      samplesA = [q[0]];
      spag = [];
      accepts = 0; total = 0; divergences = 0; lastDH = 0;
      mhQ = [0.0, 0.0]; mhLogp = logpost(mhQ[0], mhQ[1], data);
      mhTrail = [[mhQ[0], mhQ[1]]]; mhSpag = []; mhAccepts = 0; mhTotal = 0;
      active = null; reveal = 0; applied = false; flash = null; rejLine = null;
      nextTransition();
    }

    // The data changed (a point was dragged): recompute densities but keep the
    // chain where it is so you SEE it migrate onto the new posterior.
    function onDataChanged() {
      logpCur = logpost(q[0], q[1], data);
      mhLogp = logpost(mhQ[0], mhQ[1], data);
      nextTransition();
      bgDirty = true;
    }

    // Simulate a full leapfrog trajectory from the current q using the analytic force.
    function nextTransition() {
      var p = [nrm(), nrm()]; // identity mass => N(0,1) momentum
      var U0 = -logpost(q[0], q[1], data);
      var H0 = U0 + 0.5 * (p[0] * p[0] + p[1] * p[1]);
      var qa = q[0], qb = q[1], pa = p[0], pb = p[1];
      var qs = [[qa, qb]], Hs = [H0];
      var grad = gradLogpost(qa, qb, data);
      var divergent = false;
      for (var i = 0; i < L; i++) {
        pa += 0.5 * eps * grad[0]; pb += 0.5 * eps * grad[1];
        qa += eps * pa;            qb += eps * pb;
        grad = gradLogpost(qa, qb, data);
        pa += 0.5 * eps * grad[0]; pb += 0.5 * eps * grad[1];
        var Hh = -logpost(qa, qb, data) + 0.5 * (pa * pa + pb * pb);
        qs.push([qa, qb]); Hs.push(Hh);
        if (!isFinite(Hh) || Math.abs(Hh - H0) > DIV_THRESHOLD) { divergent = true; break; }
      }
      var last = Hs.length - 1;
      var dH = Hs[last] - H0;
      var acceptProb = divergent ? 0 : Math.min(1, Math.exp(-dH));
      var accept = !divergent && (rng() < acceptProb);
      active = {
        qs: qs, Hs: Hs, H0: H0, p0: p, start: [q[0], q[1]],
        divergent: divergent, dH: dH, accept: accept,
        qEnd: [qa, qb], logpEnd: logpost(qa, qb, data)
      };
      reveal = 0; applied = false;
    }

    function applyDecision() {
      total++;
      if (active.divergent) {
        divergences++;
        flash = { kind: "div", a: 1, at: [active.qEnd[0], active.qEnd[1]] };
        rejLine = { a: active.qEnd[0], b: active.qEnd[1], life: 1 };
      } else if (active.accept) {
        q = [active.qEnd[0], active.qEnd[1]];
        logpCur = active.logpEnd;
        accepts++;
        flash = { kind: "accept", a: 1, at: [q[0], q[1]] };
      } else {
        flash = { kind: "reject", a: 1, at: [active.qEnd[0], active.qEnd[1]] };
        rejLine = { a: active.qEnd[0], b: active.qEnd[1], life: 1 };
      }
      lastDH = active.dH;
      pushCapped(trail, [q[0], q[1]], 240);
      pushCapped(samplesA, q[0], 4000);
      pushCapped(spag, [q[0], q[1]], SPAG_CAP);
      if (sideBySide) mhAdvance(active.qs.length - 1);
      updateReadouts();
    }

    // Random-walk Metropolis: one proposal per leapfrog step => matched budget.
    function mhAdvance(steps) {
      for (var i = 0; i < steps; i++) {
        var na = mhQ[0] + nrm() * MH_SIGMA;
        var nb = mhQ[1] + nrm() * MH_SIGMA;
        var lp = logpost(na, nb, data);
        if (Math.log(rng() + 1e-300) < lp - mhLogp) { mhQ = [na, nb]; mhLogp = lp; mhAccepts++; }
        mhTotal++;
        pushCapped(mhTrail, [mhQ[0], mhQ[1]], 1200);
      }
      pushCapped(mhSpag, [mhQ[0], mhQ[1]], SPAG_CAP);
    }

    // ------------------------------------------------------------------
    // DOM shell
    // ------------------------------------------------------------------
    var controls = document.createElement("div");
    controls.className = "fv-controls";
    root.appendChild(controls);

    var canvasHost = document.createElement("div");
    root.appendChild(canvasHost);
    var stripHost = document.createElement("div");
    root.appendChild(stripHost);

    var readouts = document.createElement("div");
    readouts.className = "fv-readouts";
    root.appendChild(readouts);

    var instr = document.createElement("div");
    instr.className = "fv-instruction";
    instr.textContent = "Left: drag a yellow data point. Right: violet = one leapfrog roll; coral dot = the current (slope, intercept).";
    root.appendChild(instr);

    var hint = document.createElement("div");
    hint.className = "fv-hint";
    hint.textContent = "try: drag the rightmost point far up — the heatmap tilts and the coral ball rolls after it within a few transitions.";
    root.appendChild(hint);

    // ---- canvases ----
    var mainApi = FV.canvas(canvasHost, { height: 400, onResize: onMainResize });
    var stripApi = FV.canvas(stripHost, { height: 92, onResize: function () { scheduleDraw(); } });

    var dataView = null, paramView = null, bgDirty = true;
    var PADL = 40, PADR = 12, PADT = 12, PADB = 26, GAP = 26;

    function makeView(x0, y0, w, h, domx, domy) {
      return {
        x0: x0, y0: y0, w: w, h: h,
        xs: FV.scale(domx, [x0, x0 + w]),
        ys: FV.scale(domy, [y0 + h, y0]),
        bg: null
      };
    }
    function onMainResize(api) {
      var plotW = api.w - PADL - PADR - GAP;
      var plotH = api.h - PADT - PADB;
      if (plotW < 40 || plotH < 20) return;
      var dataW = plotW * 0.56;
      var paramW = plotW - dataW;
      dataView = makeView(PADL, PADT, dataW, plotH, DAT_X, DAT_Y);
      paramView = makeView(PADL + dataW + GAP, PADT, paramW, plotH, DOM_A, DOM_B);
      bgDirty = true;
      scheduleDraw();
    }

    function buildBg(view, dpr) {
      var off = document.createElement("canvas");
      off.width = Math.max(1, Math.round(view.w * dpr));
      off.height = Math.max(1, Math.round(view.h * dpr));
      var octx = off.getContext("2d");
      octx.setTransform(dpr, 0, 0, dpr, 0, 0);
      var xsL = FV.scale(DOM_A, [0, view.w]);
      var ysL = FV.scale(DOM_B, [view.h, 0]);
      FV.heatmap(octx, function (a, b) { return Math.exp(logpost(a, b, data)); },
        { xscale: xsL, yscale: ysL, w: view.w, h: view.h, colormap: "post", step: 4 });
      view.bg = off;
    }

    // ------------------------------------------------------------------
    // Drawing primitives
    // ------------------------------------------------------------------
    function arrow(ctx, x1, y1, x2, y2, color, width) {
      var dx = x2 - x1, dy = y2 - y1;
      var len = Math.sqrt(dx * dx + dy * dy);
      if (len < 0.5) return;
      var ux = dx / len, uy = dy / len;
      ctx.save();
      ctx.strokeStyle = color; ctx.fillStyle = color;
      ctx.lineWidth = width || 2; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      var ah = 7;
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - ah * ux + ah * 0.55 * uy, y2 - ah * uy - ah * 0.55 * ux);
      ctx.lineTo(x2 - ah * ux - ah * 0.55 * uy, y2 - ah * uy + ah * 0.55 * ux);
      ctx.closePath(); ctx.fill();
      ctx.restore();
    }
    function dot(ctx, x, y, r, color) {
      ctx.save(); ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill(); ctx.restore();
    }
    function ring(ctx, x, y, r, color, alpha, width) {
      ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = color;
      ctx.lineWidth = width || 3;
      ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.stroke(); ctx.restore();
    }
    // A regression line a*x+b drawn across the data panel, clipped to it.
    function regLine(ctx, view, a, b, color, alpha, width, dash) {
      var xLo = DAT_X[0], xHi = DAT_X[1];
      ctx.save();
      ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width;
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(view.xs(xLo), view.ys(a * xLo + b));
      ctx.lineTo(view.xs(xHi), view.ys(a * xHi + b));
      ctx.stroke();
      ctx.restore();
    }

    // ------------------------------------------------------------------
    // Data-space panel (the hero): spaghetti through draggable points.
    // ------------------------------------------------------------------
    function drawDataPanel(ctx, colors, th) {
      var view = dataView;
      FV.axes(ctx, {
        x: view.x0, y: view.y0, w: view.w, h: view.h,
        xscale: view.xs, yscale: view.ys, xlabel: "x", theme: th
      });
      ctx.save();
      ctx.beginPath(); ctx.rect(view.x0, view.y0, view.w, view.h); ctx.clip();

      var i;
      // MH spaghetti — dimmer green (only when comparing)
      if (sideBySide) {
        for (i = 0; i < mhSpag.length; i++) {
          var age = i / Math.max(1, mhSpag.length - 1);
          regLine(ctx, view, mhSpag[i][0], mhSpag[i][1], colors.post, 0.04 + 0.06 * age, 1);
        }
      }
      // HMC spaghetti — bright green, opacity ramps with recency
      for (i = 0; i < spag.length; i++) {
        var a2 = i / Math.max(1, spag.length - 1);
        regLine(ctx, view, spag[i][0], spag[i][1], colors.post, 0.08 + 0.28 * a2, 1);
      }
      // rejected/divergent proposal flashes coral-dashed then vanishes
      if (rejLine && rejLine.life > 0) {
        regLine(ctx, view, rejLine.a, rejLine.b, colors.hot, 0.5 * rejLine.life, 1.5, [5, 4]);
      }
      // current sample line — coral, thick
      regLine(ctx, view, q[0], q[1], colors.hot, 0.95, 2.4);
      ctx.restore();

      // data points — yellow, draggable
      for (i = 0; i < N; i++) {
        var px = view.xs(data.xs[i]), py = view.ys(data.ys[i]);
        var isHot = (dragIdx === i);
        ctx.save();
        ctx.globalAlpha = 0.22; dot(ctx, px, py, isHot ? 11 : 8, colors.data); ctx.restore();
        dot(ctx, px, py, isHot ? 6 : 5, colors.data);
        ctx.save();
        ctx.strokeStyle = th.dark ? "rgba(13,17,23,0.9)" : "rgba(255,255,255,0.9)";
        ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(px, py, isHot ? 6 : 5, 0, 2 * Math.PI); ctx.stroke();
        ctx.restore();
      }
      // label
      ctx.save();
      ctx.fillStyle = colors.ink; ctx.globalAlpha = 0.8;
      ctx.font = "600 11px var(--mono-font, monospace)";
      ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText("DATA SPACE  y = a·x + b", view.x0 + 6, view.y0 + 5);
      ctx.restore();
    }

    // ------------------------------------------------------------------
    // Parameter-space panel: heatmap + leapfrog roll.
    // ------------------------------------------------------------------
    function drawParamPanel(ctx, colors, th) {
      var view = paramView;
      if (!view.bg || bgDirty) buildBg(view, mainApi.dpr);
      ctx.drawImage(view.bg, view.x0, view.y0, view.w, view.h);
      FV.axes(ctx, {
        x: view.x0, y: view.y0, w: view.w, h: view.h,
        xscale: view.xs, yscale: view.ys, xlabel: "slope a", theme: th
      });
      ctx.save();
      ctx.beginPath(); ctx.rect(view.x0, view.y0, view.w, view.h); ctx.clip();
      var xs = view.xs, ys = view.ys, i, pts;

      // MH chain trail (faint blue) when comparing
      if (sideBySide) {
        pts = [];
        for (i = 0; i < mhTrail.length; i++) pts.push([xs(mhTrail[i][0]), ys(mhTrail[i][1])]);
        ctx.globalAlpha = 0.4;
        FV.curve(ctx, pts, { color: colors.prior, width: 1 });
        ctx.globalAlpha = 1;
        dot(ctx, xs(mhQ[0]), ys(mhQ[1]), 3.5, colors.prior);
      }

      // HMC chain trail (ink)
      pts = [];
      for (i = 0; i < trail.length; i++) pts.push([xs(trail[i][0]), ys(trail[i][1])]);
      ctx.globalAlpha = 0.5;
      FV.curve(ctx, pts, { color: colors.ink, width: 1 });
      ctx.globalAlpha = 1;

      if (active) {
        var traj = active.qs;
        var idx = Math.floor(reveal);
        if (idx > traj.length - 1) idx = traj.length - 1;
        var frac = reveal - idx;
        var trajColor = active.divergent ? colors.hot : colors.flow;
        pts = [];
        for (i = 0; i <= idx; i++) pts.push([xs(traj[i][0]), ys(traj[i][1])]);
        var cx, cy;
        if (idx < traj.length - 1 && frac > 0) {
          cx = traj[idx][0] + frac * (traj[idx + 1][0] - traj[idx][0]);
          cy = traj[idx][1] + frac * (traj[idx + 1][1] - traj[idx][1]);
          pts.push([xs(cx), ys(cy)]);
        } else { cx = traj[idx][0]; cy = traj[idx][1]; }
        FV.curve(ctx, pts, { color: trajColor, width: 2 });
        for (i = 1; i <= idx; i++) dot(ctx, xs(traj[i][0]), ys(traj[i][1]), 2, trajColor);
        // momentum arrow at the start, fading as the trajectory reveals
        var af = Math.max(0, 1 - reveal / 3);
        if (af > 0.02) {
          ctx.globalAlpha = af;
          var s = active.start;
          arrow(ctx, xs(s[0]), ys(s[1]),
            xs(s[0] + active.p0[0] * 0.18), ys(s[1] + active.p0[1] * 0.18),
            colors.flow, 2.5);
          ctx.globalAlpha = 1;
        }
        dot(ctx, xs(cx), ys(cy), 4, trajColor);
        if (active.divergent && idx >= traj.length - 1) {
          ring(ctx, xs(cx), ys(cy), 10, colors.hot, 0.9, 2);
        }
      }
      // current state (coral)
      dot(ctx, xs(q[0]), ys(q[1]), 5, colors.hot);
      if (flash && flash.at) {
        var fc = flash.kind === "accept" ? colors.post : colors.hot;
        ring(ctx, xs(flash.at[0]), ys(flash.at[1]), 8 + (1 - flash.a) * 12, fc, flash.a, 3);
      }
      ctx.restore();

      ctx.save();
      ctx.fillStyle = colors.ink; ctx.globalAlpha = 0.8;
      ctx.font = "600 11px var(--mono-font, monospace)";
      ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText("PARAMETER SPACE  (a, b)", view.x0 + 6, view.y0 + 5);
      ctx.restore();
    }

    function draw() {
      if (!dataView || !paramView || !mainApi || !stripApi) return;
      var th = FV.theme();
      var colors = th.colors;
      var ctx = mainApi.ctx;
      mainApi.clear();
      drawDataPanel(ctx, colors, th);
      drawParamPanel(ctx, colors, th);
      bgDirty = false;
      drawStrip(colors, th);
    }

    function drawStrip(colors, th) {
      var ctx = stripApi.ctx;
      stripApi.clear();
      var w = stripApi.w, h = stripApi.h;
      var pl = PADL, pr = 12, pt = 14, pb = 14;
      var pw = w - pl - pr, ph = h - pt - pb;
      if (pw < 10 || ph < 10) return;
      ctx.save();
      ctx.fillStyle = colors.ink; ctx.globalAlpha = 0.7;
      ctx.font = "10px var(--mono-font, monospace)";
      ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText("ENERGY H(q,p) ALONG TRAJECTORY  —  flat = conserved", pl, 2);
      ctx.restore();
      if (!active) return;
      var Hs = active.Hs, n = Hs.length;
      var lo = Infinity, hi = -Infinity, i;
      for (i = 0; i < n; i++) { if (Hs[i] < lo) lo = Hs[i]; if (Hs[i] > hi) hi = Hs[i]; }
      if (!isFinite(lo) || !isFinite(hi)) return;
      var pad = (hi - lo) * 0.15 || 0.5;
      lo -= pad; hi += pad;
      var xs = FV.scale([0, Math.max(1, n - 1)], [pl, pl + pw]);
      var ys = FV.scale([lo, hi], [pt + ph, pt]);
      ctx.save();
      ctx.strokeStyle = colors.ink; ctx.globalAlpha = 0.3; ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(pl, ys(active.H0)); ctx.lineTo(pl + pw, ys(active.H0)); ctx.stroke();
      ctx.restore();
      var pts = [];
      for (i = 0; i < n; i++) pts.push([xs(i), ys(Hs[i])]);
      var col = active.divergent ? colors.hot : colors.flow;
      FV.curve(ctx, pts, { color: col, width: 2 });
      var idx = Math.min(Math.floor(reveal), n - 1);
      dot(ctx, xs(idx), ys(Hs[idx]), 3, col);
      ctx.save();
      ctx.strokeStyle = colors.ink; ctx.globalAlpha = 0.25; ctx.lineWidth = 1;
      ctx.strokeRect(pl, pt, pw, ph); ctx.restore();
    }

    // Coalesced redraw (keeps drags to one heatmap rebuild per frame).
    var rafPending = false;
    function scheduleDraw() {
      if (rafPending) return;
      rafPending = true;
      (window.requestAnimationFrame || function (f) { setTimeout(f, 16); })(function () {
        rafPending = false;
        draw();
      });
    }

    // ------------------------------------------------------------------
    // Readouts
    // ------------------------------------------------------------------
    var roAccept = FV.readout(readouts, { label: "ACCEPT" });
    var roDH = FV.readout(readouts, { label: "ΔH (last)" });
    var roEss = FV.readout(readouts, { label: "ESS (slope)" });
    var roDiv = FV.readout(readouts, { label: "DIVERGENCES" });

    function updateReadouts() {
      var ar = total > 0 ? accepts / total : 0;
      roAccept.set((ar * 100).toFixed(0) + "%", ar >= 0.6 ? "post" : "hot");
      var adH = Math.abs(lastDH);
      roDH.set((lastDH >= 0 ? "+" : "") + lastDH.toFixed(2), adH > 1 ? "hot" : "flow");
      roEss.set(samplesA.length > 3 ? essSingle(samplesA).toFixed(1) : "—", "post");
      roDiv.set(String(divergences), divergences > 0 ? "hot" : null);
    }

    // ------------------------------------------------------------------
    // Controls
    // ------------------------------------------------------------------
    FV.slider(controls, {
      label: "STEP ε", min: 0.01, max: 0.4, step: 0.005, value: eps,
      fmt: function (v) { return v.toFixed(3); },
      onInput: function (v) { eps = v; nextTransition(); scheduleDraw(); }
    });
    FV.slider(controls, {
      label: "LEAPFROG L", min: 1, max: 50, step: 1, value: L,
      fmt: function (v) { return String(v); },
      onInput: function (v) { L = v | 0; nextTransition(); scheduleDraw(); }
    });
    FV.slider(controls, {
      label: "SPEED", min: 2, max: 40, step: 1, value: speed,
      fmt: function (v) { return String(v); },
      onInput: function (v) { speed = v; }
    });
    FV.toggle(controls, {
      label: "MH SIDE-BY-SIDE", value: false,
      onChange: function (on) { sideBySide = on; scheduleDraw(); }
    });

    // ------------------------------------------------------------------
    // Pointer: drag the yellow data points (mouse + touch).
    // ------------------------------------------------------------------
    var dragIdx = -1;
    function evtPos(e) {
      var rect = mainApi.el.getBoundingClientRect();
      var cx = e.touches ? e.touches[0].clientX : e.clientX;
      var cy = e.touches ? e.touches[0].clientY : e.clientY;
      return { x: cx - rect.left, y: cy - rect.top };
    }
    function hitPoint(px, py) {
      if (!dataView) return -1;
      for (var i = 0; i < N; i++) {
        var dx = px - dataView.xs(data.xs[i]);
        var dy = py - dataView.ys(data.ys[i]);
        if (dx * dx + dy * dy <= 144) return i;
      }
      return -1;
    }
    function onDown(e) {
      var pos = evtPos(e);
      var idx = hitPoint(pos.x, pos.y);
      if (idx >= 0) {
        dragIdx = idx;
        if (e.cancelable) e.preventDefault();
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
        window.addEventListener("touchmove", onMove, { passive: false });
        window.addEventListener("touchend", onUp);
        scheduleDraw();
      }
    }
    function onMove(e) {
      if (dragIdx < 0) return;
      var pos = evtPos(e);
      var y = dataView.ys.invert(pos.y);
      if (y < DAT_Y[0]) y = DAT_Y[0];
      if (y > DAT_Y[1]) y = DAT_Y[1];
      data.ys[dragIdx] = y;         // x stays fixed; drag adjusts the response
      onDataChanged();
      scheduleDraw();
      if (e.cancelable) e.preventDefault();
    }
    function onUp() {
      dragIdx = -1;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchmove", onMove);
      window.removeEventListener("touchend", onUp);
      scheduleDraw();
    }
    function onHover(e) {
      if (dragIdx >= 0) return;
      var pos = evtPos(e);
      mainApi.el.style.cursor = hitPoint(pos.x, pos.y) >= 0 ? "grab" : "default";
    }
    mainApi.el.addEventListener("mousedown", onDown);
    mainApi.el.addEventListener("touchstart", onDown, { passive: false });
    mainApi.el.addEventListener("mousemove", onHover);

    // ------------------------------------------------------------------
    // Animation engine
    // ------------------------------------------------------------------
    var loopApi = FV.loop(root, function (dt) {
      if (!active) nextTransition();
      reveal += dt * speed;
      var end = active.qs.length - 1;
      if (reveal >= end) {
        reveal = end;
        if (!applied) { applyDecision(); applied = true; }
        nextTransition();
      }
      if (flash) { flash.a -= dt * 1.6; if (flash.a <= 0) flash = null; }
      if (rejLine) { rejLine.life -= dt * 1.4; if (rejLine.life <= 0) rejLine = null; }
      draw();
    });

    function stepOnce() {
      if (!active) nextTransition();
      if (applied) nextTransition();
      reveal = active.qs.length - 1;
      applyDecision();
      applied = true;
      draw();
    }

    var btns = FV.buttons(controls, [
      { label: "Play", title: "Play / pause the sampler", primary: true, onClick: togglePlay },
      { label: "Step", title: "Run one full HMC transition", onClick: function () { stepOnce(); } },
      { label: "Reset", title: "Restore the seeded data and restart the chain", onClick: function () { data = makeSeedData(DATA_SEED); resetChain(); bgDirty = true; draw(); } }
    ]);
    var playBtn = btns.fvButtons["Play"];
    if (loopApi.reduced) {
      playBtn.disabled = true;
      playBtn.title = "Reduced motion is on — use Step";
      instr.textContent = "Reduced motion: drag the yellow points; press Step to run one transition.";
    }
    function togglePlay() {
      if (loopApi.playing) { loopApi.pause(); playBtn.textContent = "Play"; }
      else { loopApi.play(); if (loopApi.playing) playBtn.textContent = "Pause"; }
    }

    // ------------------------------------------------------------------
    // Seed scrub (in prose) + theme + init
    // ------------------------------------------------------------------
    var seedSpan = document.getElementById("hmc-seed");
    if (seedSpan && FV.scrub) {
      FV.scrub(seedSpan, {
        min: 1, max: 40, step: 1, value: curSeed,
        fmt: function (v) { return String(v); },
        onInput: function (v) { curSeed = v >>> 0; resetChain(); scheduleDraw(); }
      });
    }

    FV.onThemeChange(function () { bgDirty = true; draw(); });

    resetChain();
    draw();
  });
})();
