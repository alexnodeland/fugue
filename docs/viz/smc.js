// docs/viz/smc.js — "Particles That Tell Stories"
// A 1-D bootstrap particle filter (Sequential Monte Carlo) you can watch
// breathe: propagate -> weight -> resample, time flowing left to right.
// Self-contained IIFE; assumes fugue-viz.js (window.FugueViz) loaded first.
(function () {
  "use strict";
  if (!window.FugueViz) return;

  window.FugueViz.register("smc", function (root, FV) {
    // Wait for the wasm loader's verdict (module namespace or null), then
    // init once. When the loader is absent we behave exactly as before.
    (FV.wasmReady || Promise.resolve(null)).then(function (W) {
      realInit(root, FV, W);
    });
  });

  function realInit(root, FV, W) {
    // ---- fixed model constants --------------------------------------------
    var T = 22; // number of time steps (observations)
    var SIG_STEP = 0.7; // latent random-walk step std (x_t | x_{t-1})
    var SIG_GEN = 0.6; // TRUE generating observation noise
    var PRIOR_SIG = 1.6; // spread of the x_0 prior particle cloud

    // ---- mutable widget state ---------------------------------------------
    var st = {
      N: 80, // particle count
      sigObs: 0.6, // the FILTER's assumed observation noise (slider)
      adaptive: true, // adaptive resampling (ESS/N < 0.5) vs never resample
      seed: parseInt(root.getAttribute("data-seed") || "11", 10) >>> 0,
    };

    var colors = FV.theme().colors;

    // ---- wasm backend (real fugue crate) -----------------------------------
    // When W is non-null the particle-filter math runs in the real crate via
    // WasmParticleFilter; the JS below stays as the mirrored fallback. The
    // wasm filter draws its own RNG stream, so a given seed yields a
    // different — but equally reproducible — particle history than JS mode.
    var pf = null; // live WasmParticleFilter, or null (JS math)
    var wasmFailed = false; // construction/step threw once -> permanent JS fallback
    root.setAttribute("data-fugue-backend", W ? "wasm" : "js");

    function wasmBail(e) {
      if (!wasmFailed) {
        try {
          console.warn(
            "[fugue-viz] smc: wasm particle filter failed (" +
              (e && e.message ? e.message : e) +
              ") — using mirrored JS math"
          );
        } catch (e2) { /* readout only */ }
      }
      wasmFailed = true;
      pf = null;
      root.setAttribute("data-fugue-backend", "js");
    }

    function buildFilter() {
      pf = null;
      if (!W || wasmFailed) return;
      try {
        // essThreshold 0 disables resampling (ESS/N is never < 0), which
        // mirrors the JS path's "adaptive off = never resample" toggle.
        pf = new W.WasmParticleFilter(
          st.N,
          PRIOR_SIG,
          SIG_STEP,
          st.sigObs,
          st.adaptive ? 0.5 : 0,
          BigInt(st.seed)
        );
      } catch (e) {
        wasmBail(e);
      }
    }

    // ---- DOM shell ---------------------------------------------------------
    var controls = document.createElement("div");
    controls.className = "fv-controls";
    root.appendChild(controls);

    var canvasHost = document.createElement("div");
    root.appendChild(canvasHost);

    var readouts = document.createElement("div");
    readouts.className = "fv-readouts";
    root.appendChild(readouts);

    var hint = document.createElement("div");
    hint.className = "fv-hint";
    hint.appendChild(
      document.createTextNode(
        "drop the particle count to 10 and watch the cloud collapse onto a single lineage."
      )
    );
    root.appendChild(hint);

    // ---- canvas ------------------------------------------------------------
    var cv = FV.canvas(canvasHost, {
      height: 340,
      onResize: function () {
        draw();
      },
    });

    // ---- data + particle bookkeeping --------------------------------------
    var truth = []; // true latent path
    var ys = []; // observations
    var yDomain = [-4, 4];
    var rand = null; // the single deterministic rng stream (a replayable trace)

    // particle population at rest (belongs to time `curT`)
    var s = []; // states
    var w = []; // normalized weights (sum = 1)
    var curT = -1; // last observed time index (-1 = prior, nothing seen)
    var est = []; // filtered mean estimate per observed time
    var sd = []; // filtered std (sqrt weighted variance) per observed time
    var logEv = 0; // cumulative log-evidence  log p(y_1:t)
    var lastEss = 1; // ESS/N of the most recent step (for the readout band)

    // in-flight step animation
    var anim = { active: false, time: 0, phases: [], durs: [], stp: null };
    var mode = "idle"; // 'idle' | 'playing' | 'stepping'

    var DUR = { propagate: 0.5, weight: 0.6, resample: 0.7 };

    // -----------------------------------------------------------------------
    function genData() {
      truth = [];
      ys = [];
      var x = FV.randn(rand) * 1.2;
      for (var t = 0; t < T; t++) {
        if (t > 0) x += FV.randn(rand) * SIG_STEP;
        truth.push(x);
        ys.push(x + FV.randn(rand) * SIG_GEN);
      }
      // fixed y-domain from the whole series (so the frame doesn't jump)
      var lo = Infinity,
        hi = -Infinity,
        i;
      for (i = 0; i < T; i++) {
        lo = Math.min(lo, truth[i], ys[i]);
        hi = Math.max(hi, truth[i], ys[i]);
      }
      var pad = 0.18 * (hi - lo) + 1.2;
      yDomain = [lo - pad, hi + pad];
    }

    function reset() {
      rand = FV.rng(st.seed);
      genData(); // consumes a fixed number of draws (independent of N)
      s = [];
      w = [];
      var uni = 1 / st.N;
      for (var i = 0; i < st.N; i++) {
        s.push(FV.randn(rand) * PRIOR_SIG); // x_0 prior cloud
        w.push(uni);
      }
      // wasm mode: fresh filter from the same seed (loop restarts replay).
      // The JS prior cloud above still paints the at-rest frame; the first
      // wasm step supplies its own prior via r.prev.
      buildFilter();
      curT = -1;
      est = [];
      sd = [];
      logEv = 0;
      lastEss = 1;
      anim.active = false;
      anim.time = 0;
      mode = "idle";
      setPlayLabel(false);
      updateReadouts();
      draw();
    }

    // radius so that AREA is proportional to weight; uniform weight -> r0.
    function radius(wi) {
      var r0 = Math.max(2.2, Math.min(9, 55 / Math.sqrt(st.N)));
      var r = r0 * Math.sqrt(Math.max(wi * st.N, 0.02));
      return Math.min(r, r0 * 3.2);
    }
    function uniformR() {
      return radius(1 / st.N);
    }

    // systematic resampling -> parent indices (mirrors fugue's default method)
    function systematic(weights) {
      var n = weights.length,
        idx = new Array(n),
        u = rand() / n,
        cw = 0,
        i = 0,
        j;
      for (j = 0; j < n; j++) {
        var thr = u + j / n;
        while (cw < thr && i < n) {
          cw += weights[i];
          i++;
        }
        idx[j] = Math.max(0, Math.min(i - 1, n - 1));
      }
      return idx;
    }

    // -----------------------------------------------------------------------
    // One time-step through the REAL fugue crate (WasmParticleFilter.step).
    // Maps the returned JSON onto the exact anim.stp shape the JS path
    // builds; render-only extras (chosen, jitter) are recomputed here just
    // as the JS path does.
    function beginStepWasm() {
      var toT = curT + 1;
      if (toT >= T) return false;
      var n = st.N,
        i;
      var hasProp = curT >= 0;

      var r = JSON.parse(pf.step(ys[toT]));
      var newS = r.propagated;
      var newW = r.weights; // already normalized

      // posterior mean / weighted variance — same estimator as the JS path
      var mean = 0;
      for (i = 0; i < n; i++) mean += newW[i] * newS[i];
      var vari = 0;
      for (i = 0; i < n; i++) {
        var dm = newS[i] - mean;
        vari += newW[i] * dm * dm;
      }

      var doResample = !!r.resampled;
      var parents = doResample ? r.parents : null,
        postS = doResample ? r.posterior : newS,
        postW = newW,
        jitter = null,
        chosen = null;
      if (doResample) {
        chosen = {};
        for (i = 0; i < n; i++) chosen[parents[i]] = (chosen[parents[i]] || 0) + 1;
        // deterministic small vertical fan so duplicated children separate
        jitter = new Array(n);
        var seen = {};
        var span = 0.28 * (yDomain[1] - yDomain[0]) * 0.06;
        for (i = 0; i < n; i++) {
          var p = parents[i];
          var k = seen[p] || 0;
          var cnt = chosen[p];
          jitter[i] = cnt > 1 ? (k - (cnt - 1) / 2) * span : 0;
          seen[p] = k + 1;
        }
        postW = new Array(n);
        for (i = 0; i < n; i++) postW[i] = 1 / n;
      }

      anim.stp = {
        fromT: curT,
        toT: toT,
        hasProp: hasProp,
        prevS: r.prev,
        newS: newS,
        newW: newW,
        doResample: doResample,
        parents: parents,
        chosen: chosen,
        jitter: jitter,
        postS: postS,
        postW: postW,
        essN: r.ess_frac, // already ESS/N, matching the JS field's units
        logZ: r.log_z_inc, // incremental log-evidence; finalize accumulates
        estMean: mean,
        estVar: vari,
      };
      anim.phases = [];
      anim.durs = [];
      if (hasProp) {
        anim.phases.push("propagate");
        anim.durs.push(DUR.propagate);
      }
      anim.phases.push("weight");
      anim.durs.push(DUR.weight);
      if (doResample) {
        anim.phases.push("resample");
        anim.durs.push(DUR.resample);
      }
      anim.time = 0;
      anim.active = true;
      return true;
    }

    // -----------------------------------------------------------------------
    // Compute one full time-step of the filter (pure-ish; commits on finalize).
    function beginStep() {
      if (pf) {
        try {
          return beginStepWasm();
        } catch (e) {
          // fall back permanently; s/w are maintained identically in both
          // modes, so the mirrored JS math continues from the same state.
          wasmBail(e);
        }
      }
      var toT = curT + 1;
      if (toT >= T) return false;
      var n = st.N,
        i;
      var hasProp = curT >= 0;

      // 1. PROPAGATE: x_t ~ N(x_{t-1}, SIG_STEP)  (first step: x_0 is the prior)
      var newS = new Array(n);
      for (i = 0; i < n; i++) {
        newS[i] = hasProp ? s[i] + FV.randn(rand) * SIG_STEP : s[i];
      }

      // 2. WEIGHT by the likelihood of the new observation
      var logw = new Array(n);
      for (i = 0; i < n; i++) {
        logw[i] =
          Math.log(w[i]) + FV.dist.normal.logpdf(ys[toT], newS[i], st.sigObs);
      }
      var logZ = FV.logsumexp(logw); // log Σ W_i · p(y_t | x_i)
      var newW = new Array(n),
        sumsq = 0,
        mean = 0;
      for (i = 0; i < n; i++) {
        newW[i] = Math.exp(logw[i] - logZ);
        sumsq += newW[i] * newW[i];
        mean += newW[i] * newS[i];
      }
      var ess = 1 / sumsq; // effective sample size
      var essN = ess / n;
      // weighted variance of the filtering distribution at this step
      var vari = 0;
      for (i = 0; i < n; i++) {
        var dm = newS[i] - mean;
        vari += newW[i] * dm * dm;
      }

      // 3. RESAMPLE (adaptive: only when ESS/N drops below 0.5)
      var doResample = st.adaptive && essN < 0.5;
      var parents = null,
        postS = newS,
        postW = newW,
        jitter = null,
        chosen = null;
      if (doResample) {
        parents = systematic(newW);
        chosen = {};
        for (i = 0; i < n; i++) chosen[parents[i]] = (chosen[parents[i]] || 0) + 1;
        // deterministic small vertical fan so duplicated children separate
        jitter = new Array(n);
        var seen = {};
        var span = 0.28 * (yDomain[1] - yDomain[0]) * 0.06;
        for (i = 0; i < n; i++) {
          var p = parents[i];
          var k = seen[p] || 0;
          var cnt = chosen[p];
          jitter[i] = cnt > 1 ? (k - (cnt - 1) / 2) * span : 0;
          seen[p] = k + 1;
        }
        postS = new Array(n);
        for (i = 0; i < n; i++) postS[i] = newS[parents[i]];
        postW = new Array(n);
        for (i = 0; i < n; i++) postW[i] = 1 / n;
      }

      anim.stp = {
        fromT: curT,
        toT: toT,
        hasProp: hasProp,
        prevS: s.slice(),
        newS: newS,
        newW: newW,
        doResample: doResample,
        parents: parents,
        chosen: chosen,
        jitter: jitter,
        postS: postS,
        postW: postW,
        essN: essN,
        logZ: logZ,
        estMean: mean,
        estVar: vari,
      };
      anim.phases = [];
      anim.durs = [];
      if (hasProp) {
        anim.phases.push("propagate");
        anim.durs.push(DUR.propagate);
      }
      anim.phases.push("weight");
      anim.durs.push(DUR.weight);
      if (doResample) {
        anim.phases.push("resample");
        anim.durs.push(DUR.resample);
      }
      anim.time = 0;
      anim.active = true;
      return true;
    }

    function finalizeStep() {
      var stp = anim.stp;
      if (!stp) return;
      s = stp.postS.slice();
      w = stp.postW.slice();
      curT = stp.toT;
      est[stp.toT] = stp.estMean;
      sd[stp.toT] = Math.sqrt(Math.max(stp.estVar, 0));
      logEv += stp.logZ;
      lastEss = stp.essN;
      anim.active = false;
      anim.stp = null;
      updateReadouts();
    }

    // Which phase are we in, and 0..1 progress within it?
    function phaseState() {
      var acc = 0;
      for (var i = 0; i < anim.phases.length; i++) {
        if (anim.time < acc + anim.durs[i] || i === anim.phases.length - 1) {
          var p = (anim.time - acc) / anim.durs[i];
          return { name: anim.phases[i], p: Math.max(0, Math.min(1, p)) };
        }
        acc += anim.durs[i];
      }
      return { name: anim.phases[anim.phases.length - 1], p: 1 };
    }

    function totalDur() {
      var s2 = 0;
      for (var i = 0; i < anim.durs.length; i++) s2 += anim.durs[i];
      return s2;
    }

    function ease(t) {
      return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    }

    // -----------------------------------------------------------------------
    function tick(dt) {
      if (!anim.active) {
        if (mode === "playing") {
          if (curT >= T - 1) {
            pause();
            return;
          }
          beginStep();
        } else {
          return;
        }
      }
      anim.time += dt;
      if (anim.time >= totalDur()) {
        finalizeStep();
        if (mode === "stepping") {
          mode = "idle";
          loopApi.pause();
        }
      }
      draw();
    }

    // ---- rendering ---------------------------------------------------------
    function layout() {
      var padL = 44,
        padR = 14,
        padT = 14,
        padB = 30;
      return {
        x: padL,
        y: padT,
        w: Math.max(10, cv.w - padL - padR),
        h: Math.max(10, cv.h - padT - padB),
      };
    }

    function circle(ctx, x, y, r, fill, alpha) {
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 6.28318530718);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    function draw() {
      if (!cv) return; // canvas() fires onResize before cv is assigned
      var ctx = cv.ctx;
      cv.clear();
      colors = FV.theme().colors;
      var L = layout();
      var colX = FV.scale([0, T - 1], [L.x, L.x + L.w]);
      var stateY = FV.scale(yDomain, [L.y + L.h, L.y]);

      FV.axes(ctx, {
        x: L.x,
        y: L.y,
        w: L.w,
        h: L.h,
        xscale: colX,
        yscale: stateY,
        xlabel: "time  t",
        ylabel: "state  x",
        theme: FV.theme(),
      });

      // true latent path (ink, dashed) — the ground truth the filter chases
      var tp = [];
      for (var t = 0; t < T; t++) tp.push([colX(t), stateY(truth[t])]);
      FV.curve(ctx, tp, { color: colors.ink, width: 1, dash: [4, 4] });

      // how far time has been revealed (mid-weight the new obs fades in)
      var ps = anim.active ? phaseState() : null;
      var revealT = curT;
      var obsAlphaNew = 1;
      if (anim.active) {
        var toT = anim.stp.toT;
        if (ps.name === "weight") {
          revealT = toT;
          obsAlphaNew = ease(ps.p);
        } else if (ps.name === "resample") {
          revealT = toT;
        } else {
          revealT = anim.stp.fromT; // propagate: obs not yet shown
        }
      }

      // observations up to the revealed time (yellow data dots)
      for (t = 0; t <= revealT && t < T; t++) {
        var a = t === (anim.active ? anim.stp.toT : -99) ? obsAlphaNew : 1;
        circle(ctx, colX(t), stateY(ys[t]), 3.2, colors.data, 0.9 * a);
      }

      // filtered mean ±1σ band over time — the answer the swarm computes,
      // drawn faint and behind the mean line
      var top = [],
        bot = [];
      for (t = 0; t < est.length; t++) {
        if (est[t] !== undefined && sd[t] !== undefined) {
          top.push([colX(t), stateY(est[t] + sd[t])]);
          bot.push([colX(t), stateY(est[t] - sd[t])]);
        }
      }
      if (anim.active && ps.name !== "propagate" && anim.stp.estVar != null) {
        var sdNow = Math.sqrt(Math.max(anim.stp.estVar, 0));
        top.push([colX(anim.stp.toT), stateY(anim.stp.estMean + sdNow)]);
        bot.push([colX(anim.stp.toT), stateY(anim.stp.estMean - sdNow)]);
      }
      if (top.length > 1) {
        ctx.save();
        ctx.globalAlpha = 0.13;
        ctx.fillStyle = colors.post;
        ctx.beginPath();
        ctx.moveTo(top[0][0], top[0][1]);
        for (var i = 1; i < top.length; i++) ctx.lineTo(top[i][0], top[i][1]);
        for (i = bot.length - 1; i >= 0; i--) ctx.lineTo(bot[i][0], bot[i][1]);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      }

      // posterior-mean estimate (green) through completed steps
      var ep = [];
      for (t = 0; t < est.length; t++) {
        if (est[t] !== undefined) ep.push([colX(t), stateY(est[t])]);
      }
      if (anim.active && ps.name !== "propagate")
        ep.push([colX(anim.stp.toT), stateY(anim.stp.estMean)]);
      if (ep.length > 1) FV.curve(ctx, ep, { color: colors.post, width: 2 });

      // particle cloud
      if (!anim.active) {
        drawRest(ctx, colX, stateY);
      } else {
        drawAnim(ctx, colX, stateY, ps);
      }

      // filtering distribution — a live weighted-particle KDE violin at the
      // current column, translucent green over the cloud that produced it
      if (!anim.active) {
        if (curT >= 0) drawRibbon(ctx, colX, stateY, curT, s, w, 0.22);
      } else if (ps.name === "weight") {
        drawRibbon(ctx, colX, stateY, anim.stp.toT, anim.stp.newS, anim.stp.newW, 0.22 * obsAlphaNew);
      } else if (ps.name === "resample") {
        drawRibbon(ctx, colX, stateY, anim.stp.toT, anim.stp.newS, anim.stp.newW, 0.22);
      } else if (anim.stp.fromT >= 0) {
        drawRibbon(ctx, colX, stateY, anim.stp.fromT, anim.stp.prevS, w, 0.15);
      }
    }

    // Weighted-particle kernel density (Gaussian kernel, Silverman-ish
    // bandwidth) drawn as a symmetric violin centered on time column `tIdx`.
    function drawRibbon(ctx, colX, stateY, tIdx, states, weights, alpha) {
      var n = states.length;
      if (n === 0) return;
      var i,
        sw = 0,
        m = 0;
      for (i = 0; i < n; i++) {
        sw += weights[i];
        m += weights[i] * states[i];
      }
      if (!(sw > 0)) return;
      m /= sw;
      var v = 0;
      for (i = 0; i < n; i++) {
        var d = states[i] - m;
        v += weights[i] * d * d;
      }
      v /= sw;
      var span = yDomain[1] - yDomain[0];
      var h = 1.06 * Math.sqrt(Math.max(v, 0)) * Math.pow(n, -0.2);
      if (!(h > 0.02 * span)) h = 0.02 * span; // bandwidth floor
      var M = 64,
        dens = new Array(M),
        maxD = 0,
        k,
        y;
      for (k = 0; k < M; k++) {
        y = yDomain[0] + (span * k) / (M - 1);
        var ds = 0;
        for (i = 0; i < n; i++) {
          var z = (y - states[i]) / h;
          ds += weights[i] * Math.exp(-0.5 * z * z);
        }
        dens[k] = ds;
        if (ds > maxD) maxD = ds;
      }
      if (!(maxD > 0)) return;
      var cx = colX(tIdx);
      var maxHalf = Math.min(38, 0.44 * (colX(1) - colX(0)));
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.fillStyle = colors.post;
      ctx.beginPath();
      for (k = 0; k < M; k++) {
        var yp = stateY(yDomain[0] + (span * k) / (M - 1));
        var half = (dens[k] / maxD) * maxHalf;
        if (k === 0) ctx.moveTo(cx + half, yp);
        else ctx.lineTo(cx + half, yp);
      }
      for (k = M - 1; k >= 0; k--) {
        var yp2 = stateY(yDomain[0] + (span * k) / (M - 1));
        var half2 = (dens[k] / maxD) * maxHalf;
        ctx.lineTo(cx - half2, yp2);
      }
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    function drawRest(ctx, colX, stateY) {
      var cx = colX(curT < 0 ? 0 : curT);
      for (var i = 0; i < st.N; i++) {
        circle(ctx, cx, stateY(s[i]), radius(w[i]), colors.prior, 0.6);
      }
    }

    function drawAnim(ctx, colX, stateY, ps) {
      var stp = anim.stp,
        n = st.N,
        i;
      var fromX = colX(stp.fromT < 0 ? 0 : stp.fromT);
      var toX = colX(stp.toT);
      var uR = uniformR();

      if (ps.name === "propagate") {
        var e = ease(ps.p);
        for (i = 0; i < n; i++) {
          var x0 = fromX,
            y0 = stateY(stp.prevS[i]);
          var x1 = toX,
            y1 = stateY(stp.newS[i]);
          var cx = x0 + (x1 - x0) * e,
            cy = y0 + (y1 - y0) * e;
          // faint drift line: the proposal being drawn
          ctx.globalAlpha = 0.18;
          ctx.strokeStyle = colors.prior;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.lineTo(cx, cy);
          ctx.stroke();
          ctx.globalAlpha = 1;
          circle(ctx, cx, cy, uR, colors.prior, 0.6);
        }
      } else if (ps.name === "weight") {
        var e2 = ease(ps.p);
        var oy = stateY(ys[stp.toT]);
        for (i = 0; i < n; i++) {
          var py = stateY(stp.newS[i]);
          var r = uR + (radius(stp.newW[i]) - uR) * e2;
          // thin yellow tie to the observation, brighter for better fits
          var lik = Math.min(1, stp.newW[i] * n);
          ctx.globalAlpha = 0.12 + 0.25 * lik * e2;
          ctx.strokeStyle = colors.data;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(toX, py);
          ctx.lineTo(toX, oy);
          ctx.stroke();
          ctx.globalAlpha = 1;
          circle(ctx, toX, py, r, colors.prior, 0.62);
        }
      } else {
        // resample: lineage lines converge; extinct fade coral, survivors fan
        var e3 = ease(ps.p);
        // dying particles (never chosen) flash coral and fade
        for (i = 0; i < n; i++) {
          if (!stp.chosen[i]) {
            circle(
              ctx,
              toX,
              stateY(stp.newS[i]),
              radius(stp.newW[i]),
              colors.hot,
              0.5 * (1 - e3)
            );
          }
        }
        // survivors + duplicates emerge from their parent with a violet thread
        for (i = 0; i < n; i++) {
          var parent = stp.parents[i];
          var sy = stateY(stp.newS[parent]);
          var ty = stateY(stp.postS[i]) + stp.jitter[i];
          var yy = sy + (ty - sy) * e3;
          var rr = radius(stp.newW[parent]) + (uR - radius(stp.newW[parent])) * e3;
          ctx.globalAlpha = 0.3 * (1 - e3);
          ctx.strokeStyle = colors.flow;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(toX, sy);
          ctx.lineTo(toX, yy);
          ctx.stroke();
          ctx.globalAlpha = 1;
          circle(ctx, toX, yy, rr, colors.prior, 0.6);
        }
      }
    }

    // ---- readouts ----------------------------------------------------------
    var essRead = FV.readout(readouts, { label: "ESS / N" });
    var evRead = FV.readout(readouts, { label: "log-evidence" });
    var stepRead = FV.readout(readouts, { label: "step" });

    function updateReadouts() {
      essRead.set(lastEss.toFixed(2), lastEss > 0.5 ? "post" : "hot");
      evRead.set(curT < 0 ? "—" : logEv.toFixed(2), "flow");
      stepRead.set((curT + 1) + " / " + T);
    }

    // ---- controls ----------------------------------------------------------
    FV.slider(controls, {
      label: "PARTICLES",
      min: 10,
      max: 500,
      step: 10,
      value: st.N,
      fmt: function (v) {
        return String(v | 0);
      },
      onInput: function (v) {
        st.N = v | 0;
        reset();
      },
    });

    FV.slider(controls, {
      label: "OBS NOISE",
      min: 0.2,
      max: 3,
      step: 0.05,
      value: st.sigObs,
      fmt: function (v) {
        return v.toFixed(2);
      },
      onInput: function (v) {
        st.sigObs = v;
        if (pf) {
          // wasm mode: retune the live filter mid-run (no reset needed);
          // any rebuild via reset() picks up st.sigObs anyway.
          try {
            pf.set_sig_obs(v);
          } catch (e) {
            wasmBail(e);
            reset();
          }
        } else {
          reset();
        }
      },
    });

    FV.toggle(controls, {
      label: "ADAPTIVE RESAMPLE",
      value: st.adaptive,
      onChange: function (v) {
        st.adaptive = v;
        reset();
      },
    });

    var btns = FV.buttons(controls, [
      { label: "Play", primary: true, title: "Advance the filter", onClick: onPlay },
      { label: "Step", title: "One observation", onClick: onStep },
      { label: "Reset", title: "Replay from the seed", onClick: reset },
    ]);
    var playBtn = btns.fvButtons["Play"];

    function setPlayLabel(playing) {
      playBtn.textContent = playing ? "Pause" : "Play";
    }

    function onPlay() {
      if (loopApi.reduced) return; // reduced motion: use Step
      if (mode === "playing") {
        pause();
        return;
      }
      if (curT >= T - 1 && !anim.active) reset();
      mode = "playing";
      setPlayLabel(true);
      loopApi.play();
    }

    function pause() {
      mode = "idle";
      setPlayLabel(false);
      loopApi.pause();
    }

    function onStep() {
      if (mode === "playing") {
        pause();
        return;
      }
      if (anim.active) return;
      if (curT >= T - 1) return;
      if (loopApi.reduced) {
        // instant: no tween
        if (beginStep()) finalizeStep();
        draw();
        return;
      }
      mode = "stepping";
      beginStep();
      loopApi.play();
    }

    // seed scrub (a seeded run is a replayable trace)
    var seedRow = document.createElement("div");
    seedRow.className = "fv-readouts";
    var seedLbl = document.createElement("span");
    seedLbl.className = "fv-readout-label";
    seedLbl.textContent = "SEED";
    var seedSpan = document.createElement("span");
    seedSpan.className = "fv-scrub";
    seedRow.appendChild(seedLbl);
    seedRow.appendChild(seedSpan);
    readouts.appendChild(seedRow);
    FV.scrub(seedSpan, {
      min: 1,
      max: 9999,
      step: 1,
      value: st.seed,
      fmt: function (v) {
        return String(v | 0);
      },
      onInput: function (v) {
        st.seed = v >>> 0;
        reset();
      },
    });

    // ---- loop + theme ------------------------------------------------------
    var loopApi = FV.loop(root, tick);
    if (loopApi.reduced) playBtn.style.display = "none";

    FV.onThemeChange(function () {
      colors = FV.theme().colors;
      draw();
    });

    reset();
    // Pre-warm: run 3 filter steps with no tween so the swarm is already partway
    // through its story at first paint — and so the reduced-motion frame is rich
    // (three columns of particles + a filtered mean), never an empty axis.
    for (var pw = 0; pw < 3; pw++) { if (beginStep()) finalizeStep(); }
    draw();
    // autoplay the filter (onPlay is a no-op under reduced motion, which keeps the
    // pre-warmed frame and leaves stepping to the Step button).
    onPlay();
  }
})();
