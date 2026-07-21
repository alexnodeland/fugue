// docs/viz/distributions.js — "A Field Guide to Distributions" explorable.
// Self-contained IIFE. Consumes window.FugueViz (loaded first via book.toml).
// Every distribution here maps 1:1 onto a fugue constructor; the pdf/pmf
// and sampler come straight from FugueViz.dist so the widget and the crate
// agree by construction.
(function () {
  "use strict";
  if (typeof window === "undefined" || !window.FugueViz) return;
  var FV = window.FugueViz;

  // Gamma function via the library's lgamma (needed for Weibull moments).
  function G(z) { return Math.exp(FV.lgamma(z)); }

  // --------------------------------------------------------------------------
  // The field guide. group: 'cont' | 'disc' | 'new'. kind: 'cont' | 'disc'.
  // Every `logpdf`/`sample` delegates to FV.dist (fugue's parameterization).
  // stats() returns {mean, variance, median, mode}; null = undefined/no-simple.
  // --------------------------------------------------------------------------
  var D = {
    normal: {
      name: "Normal(μ, σ)", group: "cont", kind: "cont", ret: "f64", isNew: false,
      support: "ℝ = (−∞, ∞)", reach: "a symmetric bell around a known center — noise, measurement error, CLT limits.",
      params: [
        { key: "mu", label: "μ (mean)", min: -5, max: 5, step: 0.1, value: 0 },
        { key: "sigma", label: "σ (std dev > 0)", min: 0.2, max: 3, step: 0.05, value: 1 }
      ],
      domain: function (p) { return [p.mu - 4 * p.sigma, p.mu + 4 * p.sigma]; },
      logpdf: function (x, p) { return FV.dist.normal.logpdf(x, p.mu, p.sigma); },
      sample: function (r, p) { return FV.dist.normal.sample(r, p.mu, p.sigma); },
      stats: function (p) { return { mean: p.mu, variance: p.sigma * p.sigma, median: p.mu, mode: p.mu }; }
    },
    uniform: {
      name: "Uniform(low, high)", group: "cont", kind: "cont", ret: "f64", isNew: false,
      support: "[low, high)", reach: "total ignorance on a bounded interval — a flat prior.",
      params: [
        { key: "low", label: "low", min: -5, max: 4, step: 0.25, value: -1 },
        { key: "high", label: "high", min: -4, max: 5, step: 0.25, value: 2 }
      ],
      validate: function (p) { return p.low < p.high ? null : "Uniform::new requires low < high"; },
      domain: function (p) { var pad = (p.high - p.low) * 0.15 || 0.5; return [p.low - pad, p.high + pad]; },
      logpdf: function (x, p) { return FV.dist.uniform.logpdf(x, p.low, p.high); },
      sample: function (r, p) { return FV.dist.uniform.sample(r, p.low, p.high); },
      stats: function (p) {
        var w = p.high - p.low, m = (p.low + p.high) / 2;
        return { mean: m, variance: w * w / 12, median: m, mode: null };
      }
    },
    lognormal: {
      name: "LogNormal(μ, σ)", group: "cont", kind: "cont", ret: "f64", isNew: false,
      support: "(0, ∞)", reach: "positive, right-skewed quantities whose logarithm is Normal — incomes, concentrations.",
      params: [
        { key: "mu", label: "μ of ln X", min: -1.5, max: 1.5, step: 0.1, value: 0 },
        { key: "sigma", label: "σ of ln X > 0", min: 0.2, max: 1.6, step: 0.05, value: 0.5 }
      ],
      domain: function (p) { return [0, Math.exp(p.mu + 3 * p.sigma)]; },
      logpdf: function (x, p) { return FV.dist.lognormal.logpdf(x, p.mu, p.sigma); },
      sample: function (r, p) { return FV.dist.lognormal.sample(r, p.mu, p.sigma); },
      stats: function (p) {
        var s2 = p.sigma * p.sigma;
        return {
          mean: Math.exp(p.mu + s2 / 2),
          variance: (Math.exp(s2) - 1) * Math.exp(2 * p.mu + s2),
          median: Math.exp(p.mu),
          mode: Math.exp(p.mu - s2)
        };
      }
    },
    exponential: {
      name: "Exponential(rate)", group: "cont", kind: "cont", ret: "f64", isNew: false,
      support: "[0, ∞)", reach: "waiting time until the next event of a memoryless process; mean = 1/rate.",
      params: [
        { key: "rate", label: "rate λ > 0", min: 0.2, max: 3, step: 0.05, value: 1 }
      ],
      domain: function (p) { return [0, 6 / p.rate]; },
      logpdf: function (x, p) { return FV.dist.exponential.logpdf(x, p.rate); },
      sample: function (r, p) { return FV.dist.exponential.sample(r, p.rate); },
      stats: function (p) {
        return { mean: 1 / p.rate, variance: 1 / (p.rate * p.rate), median: Math.LN2 / p.rate, mode: 0 };
      }
    },
    beta: {
      name: "Beta(α, β)", group: "cont", kind: "cont", ret: "f64", isNew: false,
      support: "[0, 1]", reach: "a probability about a probability — the conjugate prior for a coin's bias.",
      params: [
        { key: "a", label: "α > 0", min: 0.5, max: 6, step: 0.1, value: 2 },
        { key: "b", label: "β > 0", min: 0.5, max: 6, step: 0.1, value: 2 }
      ],
      domain: function () { return [0, 1]; },
      logpdf: function (x, p) { return FV.dist.beta.logpdf(x, p.a, p.b); },
      sample: function (r, p) { return FV.dist.beta.sample(r, p.a, p.b); },
      stats: function (p) {
        var s = p.a + p.b;
        return {
          mean: p.a / s,
          variance: (p.a * p.b) / (s * s * (s + 1)),
          median: null,
          mode: (p.a > 1 && p.b > 1) ? (p.a - 1) / (s - 2) : null
        };
      }
    },
    gamma: {
      name: "Gamma(shape, rate)", group: "cont", kind: "cont", ret: "f64", isNew: false,
      support: "(0, ∞)", reach: "positive quantities and waiting times for `shape` events; RATE-parameterized (mean = shape/rate).",
      params: [
        { key: "shape", label: "shape k > 0", min: 0.5, max: 6, step: 0.1, value: 2 },
        { key: "rate", label: "rate λ > 0", min: 0.3, max: 3, step: 0.05, value: 1 }
      ],
      domain: function (p) {
        var m = p.shape / p.rate, sd = Math.sqrt(p.shape) / p.rate;
        return [0, m + 5 * sd];
      },
      logpdf: function (x, p) { return FV.dist.gamma.logpdf(x, p.shape, p.rate); },
      sample: function (r, p) { return FV.dist.gamma.sample(r, p.shape, p.rate); },
      stats: function (p) {
        return {
          mean: p.shape / p.rate,
          variance: p.shape / (p.rate * p.rate),
          median: null,
          mode: p.shape >= 1 ? (p.shape - 1) / p.rate : 0
        };
      }
    },
    // ---- discrete -----------------------------------------------------------
    bernoulli: {
      name: "Bernoulli(p)", group: "disc", kind: "disc", ret: "bool", isNew: false,
      support: "{0, 1} → bool", reach: "a single yes/no trial; fugue hands you a real `bool`, no `== 1.0` dance.",
      params: [{ key: "p", label: "p ∈ [0, 1]", min: 0, max: 1, step: 0.02, value: 0.5 }],
      ints: function () { return [0, 1]; },
      logpmf: function (k, p) { return FV.dist.bernoulli.logpmf(k, p.p); },
      sample: function (r, p) { return FV.dist.bernoulli.sample(r, p.p) ? 1 : 0; },
      stats: function (p) { return { mean: p.p, variance: p.p * (1 - p.p), median: null, mode: p.p >= 0.5 ? 1 : 0 }; }
    },
    categorical: {
      name: "Categorical(probs)", group: "disc", kind: "disc", ret: "usize", isNew: false,
      support: "{0 … K−1} → usize", reach: "picking one of K labeled outcomes; returns a `usize` you index arrays with safely.",
      params: [
        { key: "w0", label: "weight 0", min: 0, max: 5, step: 0.5, value: 3 },
        { key: "w1", label: "weight 1", min: 0, max: 5, step: 0.5, value: 5 },
        { key: "w2", label: "weight 2", min: 0, max: 5, step: 0.5, value: 2 },
        { key: "w3", label: "weight 3", min: 0, max: 5, step: 0.5, value: 1 }
      ],
      probs: function (p) {
        var w = [p.w0, p.w1, p.w2, p.w3], s = w[0] + w[1] + w[2] + w[3];
        if (s <= 0) return null;
        return [w[0] / s, w[1] / s, w[2] / s, w[3] / s];
      },
      validate: function (p) { return this.probs(p) ? null : "Categorical::new needs weights that sum to > 0"; },
      ints: function () { return [0, 1, 2, 3]; },
      logpmf: function (k, p) { var ps = this.probs(p); return ps ? FV.dist.categorical.logpmf(k, ps) : -Infinity; },
      sample: function (r, p) { var ps = this.probs(p); return ps ? FV.dist.categorical.sample(r, ps) : 0; },
      stats: function (p) {
        var ps = this.probs(p); if (!ps) return { mean: null, variance: null, median: null, mode: null };
        var m = 0, m2 = 0, best = 0, bi = 0;
        for (var i = 0; i < ps.length; i++) { m += i * ps[i]; m2 += i * i * ps[i]; if (ps[i] > best) { best = ps[i]; bi = i; } }
        return { mean: m, variance: m2 - m * m, median: null, mode: bi };
      }
    },
    binomial: {
      name: "Binomial(n, p)", group: "disc", kind: "disc", ret: "u64", isNew: false,
      support: "{0 … n} → u64", reach: "the count of successes in n independent trials.",
      params: [
        { key: "n", label: "n (trials)", min: 1, max: 40, step: 1, value: 15, int: true },
        { key: "p", label: "p ∈ [0, 1]", min: 0, max: 1, step: 0.02, value: 0.4 }
      ],
      ints: function (p) { var a = [], n = Math.round(p.n); for (var i = 0; i <= n; i++) a.push(i); return a; },
      logpmf: function (k, p) { return FV.dist.binomial.logpmf(k, Math.round(p.n), p.p); },
      sample: function (r, p) { return FV.dist.binomial.sample(r, Math.round(p.n), p.p); },
      stats: function (p) {
        var n = Math.round(p.n);
        return { mean: n * p.p, variance: n * p.p * (1 - p.p), median: null, mode: Math.floor((n + 1) * p.p) };
      }
    },
    poisson: {
      name: "Poisson(λ)", group: "disc", kind: "disc", ret: "u64", isNew: false,
      support: "{0, 1, 2, …} → u64", reach: "the count of rare events in a fixed window; mean = variance = λ.",
      params: [{ key: "lambda", label: "λ > 0", min: 0.3, max: 15, step: 0.1, value: 4 }],
      ints: function (p) {
        var hi = Math.ceil(p.lambda + 4 * Math.sqrt(p.lambda) + 5), a = [];
        for (var i = 0; i <= hi; i++) a.push(i); return a;
      },
      logpmf: function (k, p) { return FV.dist.poisson.logpmf(k, p.lambda); },
      sample: function (r, p) { return FV.dist.poisson.sample(r, p.lambda); },
      stats: function (p) { return { mean: p.lambda, variance: p.lambda, median: null, mode: Math.floor(p.lambda) }; }
    },
    // ---- additional families ------------------------------------------------
    studentt: {
      name: "StudentT(df, loc, scale)", group: "cont", kind: "cont", ret: "f64", isNew: true,
      support: "ℝ", reach: "a heavier-tailed Normal; small df tolerates outliers. Robust regression noise.",
      params: [
        { key: "df", label: "df ν > 0", min: 1, max: 30, step: 0.5, value: 3 },
        { key: "loc", label: "loc", min: -3, max: 3, step: 0.1, value: 0 },
        { key: "scale", label: "scale > 0", min: 0.3, max: 3, step: 0.05, value: 1 }
      ],
      domain: function (p) { return [p.loc - 7 * p.scale, p.loc + 7 * p.scale]; },
      logpdf: function (x, p) { return FV.dist.studentt.logpdf(x, p.df, p.loc, p.scale); },
      sample: function (r, p) { return FV.dist.studentt.sample(r, p.df, p.loc, p.scale); },
      stats: function (p) {
        return {
          mean: p.df > 1 ? p.loc : null,
          variance: p.df > 2 ? p.scale * p.scale * p.df / (p.df - 2) : null,
          median: p.loc,
          mode: p.loc
        };
      }
    },
    cauchy: {
      name: "Cauchy(loc, scale)", group: "cont", kind: "cont", ret: "f64", isNew: true,
      support: "ℝ", reach: "pathologically heavy tails — no mean, no variance. StudentT with df = 1.",
      params: [
        { key: "loc", label: "loc (median)", min: -3, max: 3, step: 0.1, value: 0 },
        { key: "scale", label: "scale > 0", min: 0.3, max: 3, step: 0.05, value: 1 }
      ],
      domain: function (p) { return [p.loc - 10 * p.scale, p.loc + 10 * p.scale]; },
      logpdf: function (x, p) { return FV.dist.cauchy.logpdf(x, p.loc, p.scale); },
      sample: function (r, p) { return FV.dist.cauchy.sample(r, p.loc, p.scale); },
      stats: function (p) { return { mean: null, variance: null, median: p.loc, mode: p.loc }; }
    },
    laplace: {
      name: "Laplace(loc, scale)", group: "cont", kind: "cont", ret: "f64", isNew: true,
      support: "ℝ", reach: "a sharp peak with exponential tails; the prior behind L1 / lasso shrinkage.",
      params: [
        { key: "loc", label: "loc", min: -3, max: 3, step: 0.1, value: 0 },
        { key: "scale", label: "scale b > 0", min: 0.3, max: 3, step: 0.05, value: 1 }
      ],
      domain: function (p) { return [p.loc - 8 * p.scale, p.loc + 8 * p.scale]; },
      logpdf: function (x, p) { return FV.dist.laplace.logpdf(x, p.loc, p.scale); },
      sample: function (r, p) { return FV.dist.laplace.sample(r, p.loc, p.scale); },
      stats: function (p) { return { mean: p.loc, variance: 2 * p.scale * p.scale, median: p.loc, mode: p.loc }; }
    },
    weibull: {
      name: "Weibull(shape, scale)", group: "cont", kind: "cont", ret: "f64", isNew: true,
      support: "[0, ∞)", reach: "time-to-failure and survival; shape < 1 ages out early, shape > 1 wears out late.",
      params: [
        { key: "shape", label: "shape k > 0", min: 0.5, max: 5, step: 0.1, value: 1.5 },
        { key: "scale", label: "scale λ > 0", min: 0.3, max: 3, step: 0.05, value: 1 }
      ],
      domain: function (p) {
        return [0, p.scale * Math.pow(-Math.log(0.004), 1 / p.shape) * 1.05];
      },
      logpdf: function (x, p) { return FV.dist.weibull.logpdf(x, p.shape, p.scale); },
      sample: function (r, p) { return FV.dist.weibull.sample(r, p.shape, p.scale); },
      stats: function (p) {
        var k = p.shape, l = p.scale;
        var g1 = G(1 + 1 / k), g2 = G(1 + 2 / k);
        return {
          mean: l * g1,
          variance: l * l * (g2 - g1 * g1),
          median: l * Math.pow(Math.LN2, 1 / k),
          mode: k > 1 ? l * Math.pow((k - 1) / k, 1 / k) : 0
        };
      }
    },
    chisquared: {
      name: "ChiSquared(k)", group: "cont", kind: "cont", ret: "f64", isNew: true,
      support: "(0, ∞)", reach: "sums of k squared standard Normals; goodness-of-fit and variance tests. = Gamma(k/2, ½).",
      params: [{ key: "k", label: "df k > 0", min: 1, max: 15, step: 0.5, value: 4 }],
      domain: function (p) { return [0, p.k + 4 * Math.sqrt(2 * p.k) + 2]; },
      logpdf: function (x, p) { return FV.dist.chisquared.logpdf(x, p.k); },
      sample: function (r, p) { return FV.dist.chisquared.sample(r, p.k); },
      stats: function (p) {
        var med = p.k * Math.pow(1 - 2 / (9 * p.k), 3);
        return { mean: p.k, variance: 2 * p.k, median: med > 0 ? med : null, mode: Math.max(p.k - 2, 0) };
      }
    },
    inversegamma: {
      name: "InverseGamma(shape, rate)", group: "cont", kind: "cont", ret: "f64", isNew: true,
      support: "(0, ∞)", reach: "the conjugate prior for a Normal's variance; α = shape, β = rate.",
      params: [
        { key: "shape", label: "shape α > 0", min: 1.5, max: 6, step: 0.1, value: 3 },
        { key: "rate", label: "rate β > 0", min: 0.5, max: 4, step: 0.1, value: 2 }
      ],
      domain: function (p) {
        var hi = p.shape > 1 ? p.rate / (p.shape - 1) : p.rate / (p.shape + 1);
        return [0, hi * 6];
      },
      logpdf: function (x, p) { return FV.dist.inversegamma.logpdf(x, p.shape, p.rate); },
      sample: function (r, p) { return FV.dist.inversegamma.sample(r, p.shape, p.rate); },
      stats: function (p) {
        return {
          mean: p.shape > 1 ? p.rate / (p.shape - 1) : null,
          variance: p.shape > 2 ? (p.rate * p.rate) / ((p.shape - 1) * (p.shape - 1) * (p.shape - 2)) : null,
          median: null,
          mode: p.rate / (p.shape + 1)
        };
      }
    },
    discreteuniform: {
      name: "DiscreteUniform(low, high)", group: "disc", kind: "disc", ret: "i64", isNew: true,
      support: "{low … high} inclusive → i64", reach: "a fair die over an integer range; every value equally likely.",
      params: [
        { key: "low", label: "low", min: 0, max: 6, step: 1, value: 1, int: true },
        { key: "high", label: "high", min: 1, max: 12, step: 1, value: 6, int: true }
      ],
      validate: function (p) { return Math.round(p.low) <= Math.round(p.high) ? null : "DiscreteUniform::new requires low ≤ high"; },
      ints: function (p) {
        var lo = Math.round(p.low), hi = Math.round(p.high), a = [];
        for (var i = lo; i <= hi; i++) a.push(i); return a;
      },
      logpmf: function (k, p) { return FV.dist.discreteuniform.logpmf(k, Math.round(p.low), Math.round(p.high)); },
      sample: function (r, p) { return FV.dist.discreteuniform.sample(r, Math.round(p.low), Math.round(p.high)); },
      stats: function (p) {
        var lo = Math.round(p.low), hi = Math.round(p.high), n = hi - lo + 1;
        return { mean: (lo + hi) / 2, variance: (n * n - 1) / 12, median: (lo + hi) / 2, mode: null };
      }
    }
  };

  // Menu order, grouped for the <optgroup>s.
  var GROUPS = [
    { label: "Continuous", keys: ["normal", "uniform", "lognormal", "exponential", "beta", "gamma", "studentt", "cauchy", "laplace", "weibull", "chisquared", "inversegamma"] },
    { label: "Discrete", keys: ["bernoulli", "categorical", "binomial", "poisson", "discreteuniform"] }
  ];

  var MAX_SAMPLES = 60000; // cap the continuous sample buffer

  FV.register("distributions", function (root, FV) {
    // ---- shell ------------------------------------------------------------
    var controls = document.createElement("div");
    controls.className = "fv-controls";
    root.appendChild(controls);

    // selector + meta line
    var selWrap = document.createElement("label");
    selWrap.className = "fv-control";
    var selLab = document.createElement("span");
    selLab.className = "fv-control-label";
    selLab.textContent = "distribution";
    selWrap.appendChild(selLab);
    var sel = document.createElement("select");
    sel.className = "fv-select";
    sel.style.marginTop = "4px";
    for (var gi = 0; gi < GROUPS.length; gi++) {
      var og = document.createElement("optgroup");
      og.label = GROUPS[gi].label;
      for (var ki = 0; ki < GROUPS[gi].keys.length; ki++) {
        var key = GROUPS[gi].keys[ki];
        var opt = document.createElement("option");
        opt.value = key;
        opt.textContent = D[key].name;
        og.appendChild(opt);
      }
      sel.appendChild(og);
    }
    selWrap.appendChild(sel);
    controls.appendChild(selWrap);

    // dynamic parameter sliders live here
    var paramBox = document.createElement("div");
    paramBox.className = "fv-controls fv-param-box";
    paramBox.style.margin = "0";
    root.appendChild(paramBox);

    // transport + seed
    var transport = document.createElement("div");
    transport.className = "fv-controls";
    root.appendChild(transport);

    var playBtns = FV.buttons(transport, [
      { label: "Play", primary: true, title: "Stream samples", onClick: togglePlay },
      { label: "Step", title: "Draw one batch of samples", onClick: function () { lp.step(); } },
      { label: "Reset", title: "Clear samples, keep params", onClick: function () { resetSamples(); render(); } }
    ]);
    var playBtn = playBtns.fvButtons.Play;

    var seedWrap = document.createElement("label");
    seedWrap.className = "fv-control";
    var seedLab = document.createElement("span");
    seedLab.className = "fv-control-label";
    seedLab.textContent = "seed";
    seedWrap.appendChild(seedLab);
    var seedSpan = document.createElement("span");
    seedWrap.appendChild(seedSpan);
    transport.appendChild(seedWrap);

    var seed = parseInt(root.getAttribute("data-seed"), 10);
    if (!(seed >= 0)) seed = 11;
    FV.scrub(seedSpan, {
      min: 1, max: 9999, step: 1, value: seed,
      fmt: function (v) { return String(v); },
      onInput: function (v) { seed = v; resetSamples(); render(); }
    });

    // canvas
    var cv = FV.canvas(root, { height: 340, onResize: function () { render(); } });

    var instr = document.createElement("div");
    instr.className = "fv-instruction";
    instr.textContent = "Drag across the canvas to query the log-density at any point.";
    root.appendChild(instr);

    // readouts
    var readBox = document.createElement("div");
    readBox.className = "fv-readouts";
    root.appendChild(readBox);
    var rLogf = FV.readout(readBox, { label: "log f(x)" });
    var rMean = FV.readout(readBox, { label: "mean" });
    var rVar = FV.readout(readBox, { label: "variance" });
    var rRet = FV.readout(readBox, { label: "sample →" });
    var rN = FV.readout(readBox, { label: "samples" });

    var hint = document.createElement("div");
    hint.className = "fv-hint";
    hint.textContent = "watch the green samples race to cover the blue law, then drop the seed back to replay the exact same run.";
    root.appendChild(hint);

    // ---- state ------------------------------------------------------------
    var defKey = "normal";
    var def = D[defKey];
    var params = {};
    var rand = FV.rng(seed);
    var samples = [];      // continuous buffer
    var counts = {};       // discrete: integer -> count
    var total = 0;
    var qx = 0;            // query-x for the coral line
    var qxSet = false;

    function currentParams() {
      var p = {};
      for (var i = 0; i < def.params.length; i++) {
        var sp = def.params[i];
        p[sp.key] = sp._slider ? sp._slider.fvGet() : sp.value;
      }
      return p;
    }

    function resetSamples() {
      rand = FV.rng(seed);
      samples = [];
      counts = {};
      total = 0;
    }

    function buildParamSliders() {
      // clear
      while (paramBox.firstChild) paramBox.removeChild(paramBox.firstChild);
      for (var i = 0; i < def.params.length; i++) {
        (function (sp) {
          var decimals = (String(sp.step).split(".")[1] || "").length;
          sp._slider = FV.slider(paramBox, {
            label: sp.label, min: sp.min, max: sp.max, step: sp.step, value: sp.value,
            fmt: function (v) { return sp.int ? String(Math.round(v)) : v.toFixed(Math.max(decimals, 0)); },
            onInput: function () { params = currentParams(); resetSamples(); qxSet = false; render(); }
          });
        })(def.params[i]);
      }
    }

    function selectDist(key) {
      def = D[key];
      defKey = key;
      buildParamSliders();
      params = currentParams();
      resetSamples();
      qxSet = false;
      render();
    }

    sel.addEventListener("change", function () { selectDist(sel.value); });

    // ---- sampling ---------------------------------------------------------
    function addBatch(n) {
      if (def.validate && def.validate(params)) return;
      for (var i = 0; i < n; i++) {
        var v = def.sample(rand, params);
        if (def.kind === "disc") {
          var iv = Math.round(v);
          counts[iv] = (counts[iv] || 0) + 1;
          total++;
        } else {
          if (samples.length < MAX_SAMPLES) samples.push(v);
          total++;
        }
      }
    }

    // ---- drawing ----------------------------------------------------------
    var ML = 46, MR = 16, MT = 16, MB = 34;

    function fmtNum(v) {
      if (v == null || !isFinite(v)) return "—";
      var a = Math.abs(v);
      if (a !== 0 && (a >= 1e4 || a < 1e-3)) return v.toExponential(2);
      return (Math.round(v * 1000) / 1000).toString();
    }

    function clampY(py, top, bot) { return py < top ? top : py > bot ? bot : py; }

    function render() {
      if (!def || !params) return; // canvas() fires onResize before state is ready
      var t = FV.theme();
      var c = t.colors;
      var ctx = cv.ctx;
      cv.clear();

      var W = cv.w, H = cv.h;
      var plotL = ML, plotR = W - MR, plotT = MT, plotB = H - MB;
      var plotW = plotR - plotL, plotH = plotB - plotT;

      // return-type badge (fugue's natural sample type)
      rRet.set(def.ret);

      // invalid parameters: show fugue's validation story, no draw
      if (def.validate) {
        var msg = def.validate(params);
        if (msg) {
          ctx.save();
          ctx.fillStyle = c.hot;
          ctx.font = "13px var(--mono-font, monospace)";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(msg, W / 2, H / 2 - 8);
          ctx.globalAlpha = 0.7;
          ctx.font = "11px var(--mono-font, monospace)";
          ctx.fillStyle = c.ink;
          ctx.fillText("(the constructor returns Err — nothing to sample)", W / 2, H / 2 + 12);
          ctx.restore();
          rLogf.set("—"); rMean.set("—"); rVar.set("—"); rN.set(String(total));
          return;
        }
      }

      var st = def.stats(params);
      var dom = def.kind === "disc"
        ? (function () { var ks = def.ints(params); return [ks[0] - 0.5, ks[ks.length - 1] + 0.5]; })()
        : def.domain(params);
      var xlo = dom[0], xhi = dom[1];
      var xscale = FV.scale([xlo, xhi], [plotL, plotR]);

      // default query-x at the mean (or domain centre)
      if (!qxSet) {
        qx = (st && st.mean != null && isFinite(st.mean)) ? st.mean : (xlo + xhi) / 2;
        if (qx < xlo) qx = xlo; if (qx > xhi) qx = xhi;
        qxSet = true;
      }

      // ---- theoretical values + ymax ----
      var ymax, theo;
      if (def.kind === "disc") {
        var ks = def.ints(params);
        theo = [];
        ymax = 1e-9;
        for (var i = 0; i < ks.length; i++) {
          var pm = Math.exp(def.logpmf(ks[i], params));
          theo.push(pm);
          if (pm > ymax) ymax = pm;
        }
        // empirical bars may exceed theory a touch
        for (var kk in counts) { var e = counts[kk] / (total || 1); if (e > ymax) ymax = e; }
        ymax *= 1.2;
      } else {
        var N = 240, dens = [];
        var eps = (xhi - xlo) * 1e-4;
        theo = [];
        for (var g = 0; g <= N; g++) {
          var x = xlo + (xhi - xlo) * (g / N);
          // nudge off exact support edges (Beta/Weibull can be +∞ there)
          if (g === 0) x += eps; if (g === N) x -= eps;
          var d = Math.exp(def.logpdf(x, params));
          theo.push([x, d]);
          if (isFinite(d)) dens.push(d);
        }
        // robust ymax: 97th percentile so a spike doesn't flatten the rest
        dens.sort(function (a, b) { return a - b; });
        var q = dens.length ? dens[Math.min(dens.length - 1, Math.floor(dens.length * 0.97))] : 1;
        ymax = (q > 0 ? q : 1) * 1.3;
      }
      var yscale = FV.scale([0, ymax], [plotB, plotT]);

      // ---- axes ----
      FV.axes(ctx, {
        x: plotL, y: plotT, w: plotW, h: plotH,
        xscale: xscale, yscale: yscale,
        xlabel: "x", ylabel: def.kind === "disc" ? "P(x)" : "density", theme: t
      });

      // support baseline (ink) along y = 0
      ctx.save();
      ctx.strokeStyle = c.ink;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(plotL, yscale(0));
      ctx.lineTo(plotR, yscale(0));
      ctx.stroke();
      ctx.restore();

      // ---- empirical samples (green) ----
      if (def.kind === "disc") {
        var ks2 = def.ints(params);
        ctx.save();
        ctx.globalAlpha = 0.5;
        ctx.fillStyle = c.post;
        for (var b = 0; b < ks2.length; b++) {
          var kv = ks2[b];
          var emp = (counts[kv] || 0) / (total || 1);
          if (emp <= 0) continue;
          var bx = xscale(kv);
          var halfw = Math.min(14, (xscale(1) - xscale(0)) * 0.34);
          ctx.fillRect(bx - halfw, yscale(emp), halfw * 2, yscale(0) - yscale(emp));
        }
        ctx.restore();
      } else {
        FV.histogram(ctx, samples, { bins: 46, xscale: xscale, yscale: yscale, color: c.post, alpha: 0.5 });
      }

      // ---- theoretical law (blue) ----
      if (def.kind === "disc") {
        var ks3 = def.ints(params);
        ctx.save();
        ctx.strokeStyle = c.prior;
        ctx.fillStyle = c.prior;
        ctx.lineWidth = 2;
        for (var s = 0; s < ks3.length; s++) {
          var sx = xscale(ks3[s]);
          var sy = yscale(theo[s]);
          ctx.globalAlpha = 0.9;
          ctx.beginPath(); ctx.moveTo(sx, yscale(0)); ctx.lineTo(sx, sy); ctx.stroke();
          ctx.globalAlpha = 1;
          ctx.beginPath(); ctx.arc(sx, sy, 3.2, 0, 2 * Math.PI); ctx.fill();
        }
        ctx.restore();
      } else {
        var pts = [];
        for (var pj = 0; pj < theo.length; pj++) {
          var xv = theo[pj][0], dv = theo[pj][1];
          if (!isFinite(dv)) { pts.push(null); continue; }
          pts.push([xscale(xv), clampY(yscale(dv), plotT, plotB)]);
        }
        FV.curve(ctx, pts, { color: c.prior, width: 2.2 });
      }

      // ---- mean / median / mode markers ----
      function marker(val, color, dash, label, labY) {
        if (val == null || !isFinite(val) || val < xlo || val > xhi) return;
        var mx = xscale(val);
        ctx.save();
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.9;
        ctx.lineWidth = 1.5;
        if (dash) ctx.setLineDash(dash);
        ctx.beginPath(); ctx.moveTo(mx, plotB); ctx.lineTo(mx, plotT + 2); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = color;
        ctx.font = "10px var(--mono-font, monospace)";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(label, mx, labY);
        ctx.restore();
      }
      if (st) {
        marker(st.mean, c.flow, null, "mean", plotT + 3);
        marker(st.median, c.ink, [3, 3], "med", plotT + 15);
        marker(st.mode, c.data, [1, 3], "mode", plotT + 27);
      }

      // ---- coral query line ----
      var qxc = qx;
      var logv;
      if (def.kind === "disc") {
        var ksq = def.ints(params);
        var nearest = ksq[0];
        for (var qi = 0; qi < ksq.length; qi++) if (Math.abs(ksq[qi] - qx) < Math.abs(nearest - qx)) nearest = ksq[qi];
        qxc = nearest;
        logv = def.logpmf(nearest, params);
      } else {
        logv = def.logpdf(qx, params);
      }
      var lx = xscale(qxc);
      ctx.save();
      ctx.strokeStyle = c.hot;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(lx, plotT); ctx.lineTo(lx, plotB); ctx.stroke();
      // dot on the law
      var dot = def.kind === "disc" ? Math.exp(def.logpmf(qxc, params)) : Math.exp(def.logpdf(qxc, params));
      if (isFinite(dot)) {
        var dotY = clampY(yscale(dot), plotT, plotB);
        // §A.2: soft grab-halo on the queried point while actively dragging.
        if (dragging && FV.halo) FV.halo(ctx, lx, dotY, 11, c.hot, 0.3);
        ctx.fillStyle = c.hot;
        ctx.beginPath(); ctx.arc(lx, dotY, 4, 0, 2 * Math.PI); ctx.fill();
      }
      ctx.restore();

      // ---- readouts ----
      var xlabel = def.kind === "disc" ? String(qxc) : (Math.round(qxc * 100) / 100).toString();
      rLogf.set(fmtNum(logv) + "  @ x=" + xlabel, "hot");
      rMean.set(st ? fmtNum(st.mean) : "—", "flow");
      rVar.set(st ? fmtNum(st.variance) : "—");
      rN.set(String(total));
    }

    // ---- pointer: drag the query line (horizontal scrub) -----------------
    // §A.1 scroll-fight fix: the query line is a purely HORIZONTAL scrub, so
    // the canvas gets touch-action:pan-y — a vertical thumb-swipe scrolls the
    // PAGE, a horizontal drag queries the density. We never preventDefault
    // (the old code did, unconditionally, on every touchmove — that ate page
    // scroll over this 340px canvas, the phone "bugginess"). Pointer capture
    // keeps a fast drag tracking even off the canvas edge without any window
    // listeners; if the browser claims the gesture for a vertical scroll it
    // fires pointercancel and we stop cleanly.
    cv.el.style.touchAction = "pan-y";
    cv.el.style.cursor = "ew-resize";

    function pointerX(clientX) {
      var rect = cv.el.getBoundingClientRect();
      var cx = clientX - rect.left;
      var plotL = ML, plotR = cv.w - MR;
      var frac = (cx - plotL) / (plotR - plotL || 1);
      if (frac < 0) frac = 0; if (frac > 1) frac = 1;
      var dom = def.kind === "disc"
        ? (function () { var ks = def.ints(params); return [ks[0] - 0.5, ks[ks.length - 1] + 0.5]; })()
        : def.domain(params);
      return dom[0] + frac * (dom[1] - dom[0]);
    }
    var dragging = false;
    function beginQuery(clientX) { dragging = true; qx = pointerX(clientX); qxSet = true; render(); }
    function moveQuery(clientX) { if (!dragging) return; qx = pointerX(clientX); render(); }
    function endQuery() { if (!dragging) return; dragging = false; render(); }

    if (typeof window.PointerEvent !== "undefined") {
      cv.el.addEventListener("pointerdown", function (ev) {
        if (ev.pointerType === "mouse" && ev.button !== 0) return;
        try { cv.el.setPointerCapture(ev.pointerId); } catch (e) {}
        beginQuery(ev.clientX);
      });
      cv.el.addEventListener("pointermove", function (ev) { moveQuery(ev.clientX); });
      cv.el.addEventListener("pointerup", endQuery);
      cv.el.addEventListener("pointercancel", endQuery);
    } else {
      // Legacy fallback (no Pointer Events): touch-action:pan-y still blocks
      // horizontal page-pan, so no preventDefault is needed to scrub.
      cv.el.addEventListener("mousedown", function (ev) { beginQuery(ev.clientX); });
      window.addEventListener("mousemove", function (ev) { moveQuery(ev.clientX); });
      window.addEventListener("mouseup", endQuery);
      cv.el.addEventListener("touchstart", function (ev) { if (ev.touches[0]) beginQuery(ev.touches[0].clientX); });
      cv.el.addEventListener("touchmove", function (ev) { if (ev.touches[0]) moveQuery(ev.touches[0].clientX); });
      window.addEventListener("touchend", endQuery);
    }

    // ---- animation --------------------------------------------------------
    // autoplay: samples start streaming the moment the widget scrolls into view.
    // Under reduced motion the loop no-ops and the pre-warmed histogram (below) stands in.
    var lp = FV.loop(root, function () {
      var batch = def.kind === "disc" ? 24 : 40;
      addBatch(batch);
      render();
    }, { autoplay: true });
    function togglePlay() {
      if (lp.playing) { lp.pause(); playBtn.textContent = "Play"; }
      else { lp.play(); playBtn.textContent = lp.playing ? "Pause" : "Play"; }
    }

    FV.onThemeChange(function () { render(); });

    // ---- go ---------------------------------------------------------------
    sel.value = defKey;
    selectDist(defKey);
    // Pre-warm ~200 samples so the green histogram is already forming at first paint
    // (and so the reduced-motion frame shows a partly-built histogram, not a bare curve).
    addBatch(200);
    render();
    // The loop autoplays itself (FV.loop {autoplay:true}); reflect that on the button.
    if (lp.reduced) { playBtn.textContent = "Play"; hint.textContent = "reduced-motion is on — tap Step to draw a batch of samples and watch them accumulate."; }
    else { playBtn.textContent = "Pause"; }
  });
})();
