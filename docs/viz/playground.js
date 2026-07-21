/* Playground — edit a fugue model and run REAL fugue inference in the
 * browser (WebAssembly build of the actual crate; see fugue-wasm-loader.js).
 *
 * The editor speaks the prob!-subset DSL compiled by fugue-wasm: the text is
 * parsed in Rust and folded into real Model combinators, then MH/HMC/SMC run
 * the crate's actual kernels one animation frame at a time. Nothing on this
 * page is mirrored JS math; without the wasm package the widget explains
 * itself instead of pretending.
 */
(function () {
  "use strict";
  if (!window.FugueViz) return;
  var FV = window.FugueViz;

  var PRESETS = [
    {
      name: "Coin flip (Beta–Bernoulli)",
      site: "p",
      source:
        '// Is this coin fair? A prior belief times ten flips.\n' +
        'let p <- sample(addr!("p"), Beta(2.0, 2.0));\n' +
        'for i in 0..data.len() {\n' +
        '    observe(addr!("flip", i), Bernoulli(p), data[i]);\n' +
        '}\n' +
        'pure(p)',
      data: '[1, 0, 1, 1, 0, 1, 1, 0, 1, 1]'
    },
    {
      name: "Gaussian mean",
      site: "mu",
      source:
        '// Where is the mean? Five observations, known noise.\n' +
        'let mu <- sample(addr!("mu"), Normal(0.0, 2.0));\n' +
        'for i in 0..data.len() {\n' +
        '    observe(addr!("y", i), Normal(mu, 1.0), data[i]);\n' +
        '}\n' +
        'pure(mu)',
      data: '[1.3, 0.7, 2.1, 0.4, 1.5]'
    },
    {
      name: "Linear regression",
      site: "a",
      source:
        '// Slope and intercept, both unknown.\n' +
        'let a <- sample(addr!("a"), Normal(0.0, 2.5));\n' +
        'let b <- sample(addr!("b"), Normal(0.0, 2.5));\n' +
        'for i in 0..x.len() {\n' +
        '    observe(addr!("y", i), Normal(a * x[i] + b, 0.8), y[i]);\n' +
        '}\n' +
        'pure(a)',
      data:
        '{"x": [-3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3],\n' +
        ' "y": [-2.9, -2.2, -2.1, -1.7, -1.4, -0.9, 0.2, 0.3, 0.9, 1.2, 1.7, 2.1]}'
    },
    {
      name: "Hierarchical (eight schools)",
      site: "mu",
      source:
        '// Partial pooling: eight schools share a population mean.\n' +
        'let mu <- sample(addr!("mu"), Normal(0.0, 5.0));\n' +
        'let tau <- sample(addr!("tau"), LogNormal(1.0, 0.5));\n' +
        'for j in 0..y.len() {\n' +
        '    let theta <- sample(addr!("theta", j), Normal(mu, tau));\n' +
        '    observe(addr!("obs", j), Normal(theta, se[j]), y[j]);\n' +
        '}\n' +
        'pure(mu)',
      data:
        '{"y": [28, 8, -3, 7, -1, 1, 18, 12],\n' +
        ' "se": [15, 10, 16, 11, 9, 11, 10, 18]}'
    }
  ];

  var CHAIN_ROLES = ["post", "hot", "flow", "data"];

  FV.register("playground", function (root) {
    (FV.wasmReady || Promise.resolve(null)).then(function (W) {
      init(root, W);
    });
  });

  function init(root, W) {
    root.setAttribute("data-fugue-backend", W ? "wasm" : "none");
    if (!W) {
      var note = document.createElement("div");
      note.className = "fv-pg-notice";
      note.innerHTML =
        "The playground runs the <em>real</em> fugue crate compiled to " +
        "WebAssembly, and that package isn’t available here. On the " +
        "deployed site it loads automatically; for a local build, run " +
        "<code>wasm-pack build crates/fugue-wasm --target web --release</code> " +
        "and copy <code>pkg/</code> next to the book (see the Docs workflow).";
      root.appendChild(note);
      return;
    }

    // ---- Editor ----------------------------------------------------------
    var editorWrap = mk("div", "fv-pg-editors", root);
    var presetRow = mk("div", "fv-controls", editorWrap);
    var presetSel = selectControl(presetRow, "preset", PRESETS.map(function (p) { return p.name; }));
    var srcTa = mk("textarea", "fv-pg-editor", editorWrap);
    srcTa.spellcheck = false;
    srcTa.rows = 8;
    srcTa.setAttribute("aria-label", "fugue model source");
    var dataTa = mk("textarea", "fv-pg-editor fv-pg-data", editorWrap);
    dataTa.spellcheck = false;
    dataTa.rows = 2;
    dataTa.setAttribute("aria-label", "observed data (JSON)");
    var feedback = mk("div", "fv-pg-feedback", editorWrap);

    // ---- Controls --------------------------------------------------------
    var controls = mk("div", "fv-controls", root);
    var samplerSel = selectControl(controls, "sampler", ["MH", "HMC", "SMC"]);
    var chainsSlider = FV.slider(controls, {
      label: "chains", min: 1, max: 4, step: 1, value: 3,
      fmt: function (v) { return String(v); },
      onInput: function () { markDirty(); }
    });
    var seedSlider = FV.slider(controls, {
      label: "seed", min: 1, max: 99, step: 1, value: 11,
      fmt: function (v) { return String(v); },
      onInput: function () { markDirty(); }
    });
    var siteSel = selectControl(controls, "plot site", []);
    var btns = FV.buttons(controls, [
      { label: "Run", primary: true, onClick: onRun },
      { label: "Pause", onClick: function () { anim.pause(); } },
      { label: "Reset", onClick: onReset }
    ]);

    // ---- Plots -----------------------------------------------------------
    var traceCv = FV.canvas(root, { height: 150 });
    var histCv = FV.canvas(root, { height: 210 });

    var readouts = mk("div", "fv-readouts", root);
    var roSteps = FV.readout(readouts, { label: "steps" });
    var roAcc = FV.readout(readouts, { label: "accept" });
    var roRhat = FV.readout(readouts, { label: "R̂" });
    var roEss = FV.readout(readouts, { label: "ESS" });
    var roZ = FV.readout(readouts, { label: "log Z" });
    var badge = mk("div", "fv-hint fv-pg-badge", root);
    badge.textContent =
      "every draw on this page is computed by the actual fugue crate " +
      "(fugue-wasm " + safeVersion(W) + ") — same kernels, same seeds, in your browser";

    // ---- State -----------------------------------------------------------
    var S = {
      sampler: "MH",       // MH | HMC | SMC
      mh: null,
      hmc: null,
      dirty: true,
      sites: [],
      site: null,
      chains: [],          // per-chain draw arrays for the plotted site
      warm: 0,             // HMC warmup remaining
      smc: null,           // parsed SMC result
      lastError: ""
    };

    function markDirty() {
      S.dirty = true;
      anim.pause();
    }

    presetSel.onChange(function (idx) {
      var p = PRESETS[idx];
      srcTa.value = p.source;
      dataTa.value = p.data;
      checkNow();
      markDirty();
    });
    samplerSel.onChange(function () { markDirty(); });
    siteSel.onChange(function () {
      S.site = siteSel.value();
      S.chains = [];
      if (S.mh) pullAll();
      draw();
    });

    var checkTimer = null;
    function checkSoon() {
      if (checkTimer) clearTimeout(checkTimer);
      checkTimer = setTimeout(checkNow, 300);
      markDirty();
    }
    function checkNow() {
      var err = "";
      try {
        err = W.check_model(srcTa.value, dataTa.value);
      } catch (e) {
        err = String(e && e.message ? e.message : e);
      }
      S.lastError = err;
      feedback.textContent = err ? "✗ " + err : "✓ model compiles";
      feedback.className = "fv-pg-feedback" + (err ? " fv-pg-err" : " fv-pg-ok");
    }
    srcTa.addEventListener("input", checkSoon);
    dataTa.addEventListener("input", checkSoon);

    function rebuild() {
      S.sampler = samplerSel.value();
      S.mh = null;
      S.hmc = null;
      S.smc = null;
      S.chains = [];
      S.warm = 0;
      var seed = BigInt(Math.round(seedSlider.fvGet()));
      var src = srcTa.value;
      var dj = dataTa.value;
      try {
        if (S.sampler === "MH") {
          S.mh = new W.WasmMh(src, dj, Math.round(chainsSlider.fvGet()), seed);
          setSites(toArray(S.mh.site_names()));
        } else if (S.sampler === "HMC") {
          S.hmc = new W.WasmHmc(src, dj, seed, 200, 16, 0.0);
          S.warm = 200;
          setSites(toArray(S.hmc.site_names()));
        } else {
          var json = W.wasm_smc_run(src, dj, 1000, 2, seed);
          S.smc = JSON.parse(json);
          setSites(S.smc.sites);
        }
        S.dirty = false;
        return true;
      } catch (e) {
        S.lastError = String(e && e.message ? e.message : e);
        feedback.textContent = "✗ " + S.lastError;
        feedback.className = "fv-pg-feedback fv-pg-err";
        return false;
      }
    }

    function setSites(names) {
      S.sites = names;
      siteSel.setOptions(names);
      // Prefer the preset's headline site when present.
      var want = PRESETS[presetSel.index()].site;
      S.site = names.indexOf(want) >= 0 ? want : names[0] || null;
      siteSel.set(S.site);
    }

    function onRun() {
      checkNow();
      if (S.lastError) return;
      if (S.dirty && !rebuild()) return;
      if (S.sampler === "SMC") {
        // One-shot: adaptive tempered SMC already ran in rebuild().
        drawSmc();
        return;
      }
      if (anim.reduced) {
        // Reduced motion: one explicit batch, one final frame.
        advance(S.sampler === "MH" ? 1500 : 300);
        draw();
        return;
      }
      anim.play();
    }

    function onReset() {
      anim.pause();
      S.dirty = true;
      S.mh = null;
      S.hmc = null;
      S.smc = null;
      S.chains = [];
      clearCanvas(traceCv);
      clearCanvas(histCv);
      setReadouts("—", "—", "—", "—", "—");
    }

    function advance(n) {
      if (S.sampler === "MH" && S.mh) {
        S.mh.step(n);
        pullAll();
      } else if (S.sampler === "HMC" && S.hmc) {
        if (S.warm > 0) {
          var w = Math.min(S.warm, n);
          S.hmc.step(w);
          S.warm -= w;
        } else {
          S.hmc.step(n);
        }
        S.chains = [toArray(S.hmc.values(S.site))];
      }
    }

    function pullAll() {
      var nc = S.mh.n_chains();
      for (var c = 0; c < nc; c++) {
        if (!S.chains[c]) S.chains[c] = [];
        var got = toArray(S.mh.values_since(S.site, c, S.chains[c].length));
        for (var i = 0; i < got.length; i++) S.chains[c].push(got[i]);
      }
    }

    var anim = FV.loop(root, function () {
      if (S.sampler === "MH") advance(24);
      else advance(3);
      draw();
    });

    // ---- Rendering -------------------------------------------------------
    function clearCanvas(cv) {
      cv.clear();
    }

    function setReadouts(steps, acc, rhat, ess, z) {
      roSteps.set(steps);
      roAcc.set(acc);
      roRhat.set(rhat, rhat !== "—" && parseFloat(rhat) > 1.1 ? "hot" : "post");
      roEss.set(ess);
      roZ.set(z);
    }

    function pooled() {
      var all = [];
      for (var c = 0; c < S.chains.length; c++) {
        var v = S.chains[c];
        for (var i = Math.floor(v.length * 0.25); i < v.length; i++) all.push(v[i]);
      }
      return all;
    }

    function draw() {
      var t = FV.theme();
      drawTrace(t);
      drawHist(t, pooled(), null);
      updateReadouts();
    }

    function drawTrace(t) {
      var ctx = traceCv.ctx;
      traceCv.clear();
      var pad = { l: 44, r: 10, t: 8, b: 18 };
      var w = traceCv.w - pad.l - pad.r, h = traceCv.h - pad.t - pad.b;
      var win = 600;
      var lo = Infinity, hi = -Infinity, maxLen = 0, c, i;
      for (c = 0; c < S.chains.length; c++) {
        var v = S.chains[c];
        maxLen = Math.max(maxLen, v.length);
        for (i = Math.max(0, v.length - win); i < v.length; i++) {
          if (v[i] < lo) lo = v[i];
          if (v[i] > hi) hi = v[i];
        }
      }
      if (!isFinite(lo)) { lo = -1; hi = 1; }
      if (hi - lo < 1e-9) { hi = lo + 1; }
      var pad2 = (hi - lo) * 0.1;
      var xs = FV.scale([Math.max(0, maxLen - win), Math.max(win, maxLen)], [pad.l, pad.l + w]);
      var ys = FV.scale([lo - pad2, hi + pad2], [pad.t + h, pad.t]);
      FV.axes(ctx, { x: pad.l, y: pad.t, w: w, h: h, xscale: xs, yscale: ys, theme: t });
      for (c = 0; c < S.chains.length; c++) {
        var vv = S.chains[c];
        var pts = [];
        for (i = Math.max(0, vv.length - win); i < vv.length; i++) {
          pts.push([xs(i), ys(vv[i])]);
        }
        FV.curve(ctx, pts, { color: t.colors[CHAIN_ROLES[c % CHAIN_ROLES.length]], width: 1.4 });
      }
      label(ctx, t, traceCv, "draws of " + S.site + " (last " + win + ")");
    }

    function drawHist(t, samples, weights) {
      var ctx = histCv.ctx;
      histCv.clear();
      if (!samples || samples.length === 0) return;
      var pad = { l: 44, r: 10, t: 10, b: 22 };
      var w = histCv.w - pad.l - pad.r, h = histCv.h - pad.t - pad.b;
      var lo = Infinity, hi = -Infinity, i;
      for (i = 0; i < samples.length; i++) {
        if (samples[i] < lo) lo = samples[i];
        if (samples[i] > hi) hi = samples[i];
      }
      if (!isFinite(lo) || hi - lo < 1e-9) { lo -= 1; hi += 1; }
      var span = hi - lo;
      lo -= span * 0.08;
      hi += span * 0.08;
      var bins = 36;
      var bw = (hi - lo) / bins;
      var dens = new Array(bins);
      var wsum = 0;
      for (i = 0; i < bins; i++) dens[i] = 0;
      for (i = 0; i < samples.length; i++) {
        var b = Math.floor((samples[i] - lo) / bw);
        if (b < 0 || b >= bins) continue;
        var wt = weights ? weights[i] : 1;
        dens[b] += wt;
        wsum += wt;
      }
      var maxD = 0;
      for (i = 0; i < bins; i++) {
        dens[i] = dens[i] / (wsum * bw || 1);
        if (dens[i] > maxD) maxD = dens[i];
      }
      var xs = FV.scale([lo, hi], [pad.l, pad.l + w]);
      var ys = FV.scale([0, maxD * 1.12 || 1], [pad.t + h, pad.t]);
      FV.axes(ctx, { x: pad.l, y: pad.t, w: w, h: h, xscale: xs, yscale: ys, theme: t });
      ctx.save();
      ctx.globalAlpha = 0.6;
      ctx.fillStyle = t.colors.post;
      for (i = 0; i < bins; i++) {
        var xa = xs(lo + i * bw), xb = xs(lo + (i + 1) * bw);
        ctx.fillRect(xa, ys(dens[i]), xb - xa, ys(0) - ys(dens[i]));
      }
      ctx.restore();
      label(ctx, t, histCv, weights ? "posterior of " + S.site + " (weighted particles)" : "posterior of " + S.site);
    }

    function label(ctx, t, cv, txt) {
      ctx.save();
      ctx.fillStyle = t.colors.ink;
      ctx.globalAlpha = 0.75;
      ctx.font = "11px var(--mono-font, monospace)";
      ctx.textAlign = "right";
      ctx.textBaseline = "top";
      ctx.fillText(txt, cv.w - 12, 6);
      ctx.restore();
    }

    function updateReadouts() {
      if (S.sampler === "MH" && S.mh) {
        var rh = S.mh.r_hat(S.site, 0);
        setReadouts(
          String(S.mh.total_steps()),
          fmtP(S.mh.recent_acceptance(500)),
          isFinite(rh) ? rh.toFixed(3) : "—",
          Math.round(S.mh.ess(S.site)),
          "—"
        );
      } else if (S.sampler === "HMC" && S.hmc) {
        setReadouts(
          (S.chains[0] ? S.chains[0].length : 0) + (S.warm > 0 ? " (warming)" : ""),
          "ε " + fmtNum(S.hmc.step_size()),
          "—",
          S.warm > 0 ? "—" : Math.round(S.hmc.ess(S.site)),
          "—"
        );
      }
    }

    function drawSmc() {
      var t = FV.theme();
      var idx = S.smc.sites.indexOf(S.site);
      if (idx < 0) idx = 0;
      clearCanvas(traceCv);
      drawHist(t, S.smc.values[idx], S.smc.weights);
      setReadouts(
        S.smc.values[idx].length + " particles",
        "—", "—",
        Math.round(S.smc.ess),
        S.smc.log_evidence.toFixed(2)
      );
    }

    // ---- boot ------------------------------------------------------------
    presetSel.set(PRESETS[0].name);
    srcTa.value = PRESETS[0].source;
    dataTa.value = PRESETS[0].data;
    checkNow();
    setReadouts("—", "—", "—", "—", "—");
  }

  // ---- small helpers -----------------------------------------------------
  function mk(tag, cls, parent) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (parent) parent.appendChild(e);
    return e;
  }

  function toArray(v) {
    return Array.prototype.slice.call(v || []);
  }

  function fmtP(v) {
    return isFinite(v) ? Math.round(v * 100) + "%" : "—";
  }

  function fmtNum(v) {
    return isFinite(v) ? (v >= 0.01 ? v.toFixed(3) : v.toExponential(1)) : "—";
  }

  function safeVersion(W) {
    try { return W.fugue_version(); } catch (e) { return ""; }
  }

  // Native <select> wrapped in the fv-control shell so it themes with the
  // sliders. Returns {el, value(), set(v), onChange(fn), setOptions(list),
  // index()}.
  function selectControl(parent, labelTxt, options) {
    var wrap = mk("label", "fv-control", parent);
    var lab = mk("span", "fv-control-label", wrap);
    lab.textContent = labelTxt;
    var sel = mk("select", "fv-select", wrap);
    var handler = null;
    fill(options);
    function fill(list) {
      sel.innerHTML = "";
      for (var i = 0; i < list.length; i++) {
        var o = document.createElement("option");
        o.value = list[i];
        o.textContent = list[i];
        sel.appendChild(o);
      }
    }
    sel.addEventListener("change", function () {
      if (handler) handler(sel.selectedIndex);
    });
    return {
      el: wrap,
      value: function () { return sel.value; },
      index: function () { return sel.selectedIndex < 0 ? 0 : sel.selectedIndex; },
      set: function (v) { sel.value = v; },
      setOptions: fill,
      onChange: function (fn) { handler = fn; }
    };
  }
})();
