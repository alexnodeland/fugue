/*
 * fugue-viz.js — shared infrastructure for the Fugue Explorables.
 *
 * Attaches a single global `window.FugueViz`. ES5-compatible IIFE: no modules,
 * no build step, no external dependencies, works from file://. Widget scripts
 * (docs/viz/*.js) call FugueViz.register("name", fn) and consume this API; they
 * MUST NOT duplicate what lives here.
 *
 * The distribution math mirrors fugue's src/core/distribution.rs EXACTLY
 * (parameterizations, support, boundary limits). See the RETURN contract in the
 * foundation agent's report for the full per-distribution parameter list.
 */
(function () {
  "use strict";

  if (typeof window !== "undefined" && window.FugueViz) {
    return; // already loaded (guard against double-inclusion)
  }

  // ==========================================================================
  // Special functions
  // ==========================================================================

  // Lanczos approximation to ln Γ(x) (g = 7, n = 9). Matches libm::lgamma to
  // ~1e-13 relative over the range the explorables use. Reflection for x < 0.5.
  var LANCZOS_G = 7;
  var LANCZOS_C = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
  ];
  var LN_2PI = 1.8378770664093454589; // ln(2π)
  var LN_PI = 1.1447298858494001742; // ln(π)
  var LN_2 = 0.6931471805599453094; // ln(2)

  function lgamma(x) {
    if (x !== x) return NaN;
    if (x < 0.5) {
      // Reflection: Γ(x)Γ(1-x) = π / sin(πx)
      var sinpix = Math.sin(Math.PI * x);
      if (sinpix === 0) return Infinity;
      return LN_PI - Math.log(Math.abs(sinpix)) - lgamma(1.0 - x);
    }
    x -= 1.0;
    var a = LANCZOS_C[0];
    var t = x + LANCZOS_G + 0.5;
    for (var i = 1; i < LANCZOS_G + 2; i++) {
      a += LANCZOS_C[i] / (x + i);
    }
    return 0.5 * LN_2PI + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }

  function logsumexp(arr) {
    var n = arr.length;
    if (n === 0) return -Infinity;
    var m = -Infinity;
    for (var i = 0; i < n; i++) {
      if (arr[i] > m) m = arr[i];
    }
    if (m === -Infinity) return -Infinity;
    if (m === Infinity) return Infinity;
    var s = 0.0;
    for (var j = 0; j < n; j++) {
      s += Math.exp(arr[j] - m);
    }
    return m + Math.log(s);
  }

  // log(1 + x), accurate for small x (Math.log1p may be absent under ES5).
  function log1p(x) {
    if (Math.log1p) return Math.log1p(x);
    if (x <= -1) return x === -1 ? -Infinity : NaN;
    var u = 1 + x;
    if (u === 1) return x;
    return Math.log(u) * (x / (u - 1));
  }

  // ==========================================================================
  // Deterministic RNG (reproducibility is a fugue value)
  // ==========================================================================

  // mulberry32: seed (uint32) -> function returning float in [0, 1).
  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Standard normal via Box–Muller, consuming an rng fn. No caching, so a given
  // rng stream deterministically yields the same normal stream.
  function randn(rand) {
    var u1 = rand();
    var u2 = rand();
    if (u1 < 1e-300) u1 = 1e-300;
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }

  // Marsaglia–Tsang gamma with unit rate (shape > 0). Used by gamma/beta/
  // chi-squared/student-t/inverse-gamma samplers.
  function gammaStd(rand, shape) {
    if (shape < 1.0) {
      var u = rand();
      if (u < 1e-300) u = 1e-300;
      return gammaStd(rand, shape + 1.0) * Math.pow(u, 1.0 / shape);
    }
    var d = shape - 1.0 / 3.0;
    var c = 1.0 / Math.sqrt(9.0 * d);
    for (;;) {
      var x, v;
      do {
        x = randn(rand);
        v = 1.0 + c * x;
      } while (v <= 0.0);
      v = v * v * v;
      var uu = rand();
      var x2 = x * x;
      if (uu < 1.0 - 0.0331 * x2 * x2) return d * v;
      if (Math.log(uu) < 0.5 * x2 + d * (1.0 - v + Math.log(v))) return d * v;
    }
  }

  // ==========================================================================
  // Distribution math — parameterizations match fugue exactly.
  // logpdf/logpmf are pure log-space (no exp), finite for every finite input.
  // ==========================================================================

  var dist = {
    // Normal(mu, sigma): sigma is the standard deviation (> 0). Returns f64.
    normal: {
      logpdf: function (x, mu, sigma) {
        if (!(sigma > 0) || !isFinite(sigma) || !isFinite(mu) || !isFinite(x)) return -Infinity;
        var z = (x - mu) / sigma;
        return -0.5 * z * z - Math.log(sigma) - 0.5 * LN_2PI;
      },
      sample: function (rand, mu, sigma) {
        return mu + sigma * randn(rand);
      }
    },

    // Uniform(low, high): support [low, high). Returns f64.
    uniform: {
      logpdf: function (x, low, high) {
        if (!(low < high) || !isFinite(low) || !isFinite(high) || !isFinite(x)) return -Infinity;
        if (x < low || x >= high) return -Infinity;
        return -Math.log(high - low);
      },
      sample: function (rand, low, high) {
        return low + rand() * (high - low);
      }
    },

    // LogNormal(mu, sigma): mu, sigma are the mean/sd of ln(X). Support (0, ∞).
    lognormal: {
      logpdf: function (x, mu, sigma) {
        if (!(sigma > 0) || !isFinite(sigma) || !isFinite(mu)) return -Infinity;
        if (!(x > 0) || !isFinite(x)) return -Infinity;
        var lx = Math.log(x);
        var z = (lx - mu) / sigma;
        return -0.5 * z * z - lx - Math.log(sigma) - 0.5 * LN_2PI;
      },
      sample: function (rand, mu, sigma) {
        return Math.exp(mu + sigma * randn(rand));
      }
    },

    // Exponential(rate): rate = λ (> 0). Support [0, ∞). Mean 1/λ. Returns f64.
    exponential: {
      logpdf: function (x, rate) {
        if (!(rate > 0) || !isFinite(rate) || !isFinite(x)) return -Infinity;
        if (x < 0) return -Infinity;
        return Math.log(rate) - rate * x;
      },
      sample: function (rand, rate) {
        var u = rand();
        if (u < 1e-300) u = 1e-300;
        return -Math.log(u) / rate;
      }
    },

    // Bernoulli(p): logpmf takes a boolean OR 0/1. sample returns a boolean.
    bernoulli: {
      logpmf: function (k, p) {
        if (!(p >= 0) || !(p <= 1) || !isFinite(p)) return -Infinity;
        var t = k === true || k === 1;
        if (t) return p <= 0 ? -Infinity : Math.log(p);
        return p >= 1 ? -Infinity : Math.log(1 - p);
      },
      sample: function (rand, p) {
        return rand() < p;
      }
    },

    // Categorical(ps): support {0..k-1}. logpmf(index, ps). sample returns usize.
    categorical: {
      logpmf: function (k, ps) {
        if (k < 0 || k >= ps.length) return -Infinity;
        var p = ps[k];
        return p > 0 ? Math.log(p) : -Infinity;
      },
      sample: function (rand, ps) {
        var u = rand();
        var acc = 0.0;
        for (var i = 0; i < ps.length; i++) {
          acc += ps[i];
          if (u < acc) return i;
        }
        return ps.length - 1;
      }
    },

    // Beta(a, b): shape params (> 0). Support [0, 1]. logpdf matches scipy
    // boundary limits (±∞ at the edges depending on the shape).
    beta: {
      logpdf: function (x, a, b) {
        if (!(a > 0) || !(b > 0) || !isFinite(a) || !isFinite(b) || !isFinite(x)) return -Infinity;
        if (x < 0 || x > 1) return -Infinity;
        var logB = lgamma(a) + lgamma(b) - lgamma(a + b);
        if (x === 0) {
          if (a > 1) return -Infinity;
          if (a < 1) return Infinity;
          return -logB; // a == 1 -> ln(b)
        }
        if (x === 1) {
          if (b > 1) return -Infinity;
          if (b < 1) return Infinity;
          return -logB; // b == 1 -> ln(a)
        }
        return (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - logB;
      },
      sample: function (rand, a, b) {
        var ga = gammaStd(rand, a);
        var gb = gammaStd(rand, b);
        var s = ga + gb;
        return s > 0 ? ga / s : 0.5;
      }
    },

    // Gamma(shape, rate): fugue is RATE-parameterized (2nd arg = λ, NOT scale).
    // Mean = shape/rate. Support (0, ∞). Returns f64.
    gamma: {
      logpdf: function (x, shape, rate) {
        if (!(shape > 0) || !(rate > 0) || !isFinite(shape) || !isFinite(rate) || !isFinite(x)) return -Infinity;
        if (x <= 0) return -Infinity;
        return shape * Math.log(rate) + (shape - 1) * Math.log(x) - rate * x - lgamma(shape);
      },
      sample: function (rand, shape, rate) {
        return gammaStd(rand, shape) / rate;
      }
    },

    // Binomial(n, p): support {0..n}. logpmf(k, n, p). sample returns u64.
    binomial: {
      logpmf: function (k, n, p) {
        if (!(p >= 0) || !(p <= 1) || !isFinite(p)) return -Infinity;
        if (k < 0 || k > n) return -Infinity;
        if (p === 0) return k === 0 ? 0.0 : -Infinity;
        if (p === 1) return k === n ? 0.0 : -Infinity;
        var logC = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
        return logC + k * Math.log(p) + (n - k) * Math.log(1 - p);
      },
      sample: function (rand, n, p) {
        var c = 0;
        for (var i = 0; i < n; i++) {
          if (rand() < p) c++;
        }
        return c;
      }
    },

    // Poisson(lambda): rate λ (> 0). Support {0,1,...}. logpmf(k, lambda).
    // sample returns u64 (Knuth; adequate for the widgets' modest λ).
    poisson: {
      logpmf: function (k, lambda) {
        if (!(lambda > 0) || !isFinite(lambda)) return -Infinity;
        if (k < 0) return -Infinity;
        if (lambda > 700 && k === 0) return -lambda;
        return k * Math.log(lambda) - lambda - lgamma(k + 1);
      },
      sample: function (rand, lambda) {
        if (lambda < 30) {
          var L = Math.exp(-lambda);
          var k = 0;
          var pp = 1.0;
          do {
            k++;
            pp *= rand();
          } while (pp > L);
          return k - 1;
        }
        // Normal approximation for large λ (rounded, clamped ≥ 0).
        var g = Math.round(lambda + Math.sqrt(lambda) * randn(rand));
        return g < 0 ? 0 : g;
      }
    },

    // StudentT(df, loc, scale): ν = df (> 0), location-scale. Support ℝ.
    studentt: {
      logpdf: function (x, df, loc, scale) {
        if (!(df > 0) || !(scale > 0) || !isFinite(df) || !isFinite(scale) || !isFinite(loc) || !isFinite(x)) return -Infinity;
        var z = (x - loc) / scale;
        return lgamma((df + 1) / 2) - lgamma(df / 2) - 0.5 * (Math.log(df) + LN_PI) - Math.log(scale) - 0.5 * (df + 1) * log1p((z * z) / df);
      },
      sample: function (rand, df, loc, scale) {
        var z = randn(rand);
        var g = gammaStd(rand, df / 2) * 2.0; // chi-squared(df)
        return loc + scale * (z / Math.sqrt(g / df));
      }
    },

    // Cauchy(loc, scale): median loc, half-width scale (> 0). Support ℝ.
    cauchy: {
      logpdf: function (x, loc, scale) {
        if (!(scale > 0) || !isFinite(scale) || !isFinite(loc) || !isFinite(x)) return -Infinity;
        var z = (x - loc) / scale;
        return -LN_PI - Math.log(scale) - log1p(z * z);
      },
      sample: function (rand, loc, scale) {
        return loc + scale * Math.tan(Math.PI * (rand() - 0.5));
      }
    },

    // Laplace(loc, scale): mean loc, scale b (> 0). Support ℝ. Variance 2b².
    laplace: {
      logpdf: function (x, loc, scale) {
        if (!(scale > 0) || !isFinite(scale) || !isFinite(loc) || !isFinite(x)) return -Infinity;
        return -Math.log(2 * scale) - Math.abs(x - loc) / scale;
      },
      sample: function (rand, loc, scale) {
        var u = rand() - 0.5;
        var s = u < 0 ? -1 : u > 0 ? 1 : 0;
        return loc - scale * s * Math.log(1 - 2 * Math.abs(u));
      }
    },

    // Weibull(shape, scale): k = shape (> 0), λ = scale (> 0). Support [0, ∞).
    weibull: {
      logpdf: function (x, shape, scale) {
        if (!(shape > 0) || !(scale > 0) || !isFinite(shape) || !isFinite(scale) || !isFinite(x)) return -Infinity;
        if (x < 0) return -Infinity;
        if (x === 0) {
          if (shape > 1) return -Infinity;
          if (shape < 1) return Infinity;
          return -Math.log(scale);
        }
        return Math.log(shape) - shape * Math.log(scale) + (shape - 1) * Math.log(x) - Math.pow(x / scale, shape);
      },
      sample: function (rand, shape, scale) {
        var u = rand();
        if (u < 1e-300) u = 1e-300;
        return scale * Math.pow(-Math.log(u), 1.0 / shape);
      }
    },

    // ChiSquared(k): df k (> 0). Special case Gamma(k/2, rate 1/2). Support (0,∞).
    chisquared: {
      logpdf: function (x, k) {
        if (!(k > 0) || !isFinite(k) || !isFinite(x)) return -Infinity;
        if (x <= 0) return -Infinity;
        var hk = k / 2;
        return -hk * LN_2 - lgamma(hk) + (hk - 1) * Math.log(x) - x / 2;
      },
      sample: function (rand, k) {
        return gammaStd(rand, k / 2) * 2.0;
      }
    },

    // InverseGamma(shape, rate): α = shape, β = rate (> 0). 1/X ~ Gamma(α, β).
    // Support (0, ∞). Matches scipy invgamma(a=α, scale=β).
    inversegamma: {
      logpdf: function (x, shape, rate) {
        if (!(shape > 0) || !(rate > 0) || !isFinite(shape) || !isFinite(rate) || !isFinite(x)) return -Infinity;
        if (x <= 0) return -Infinity;
        return shape * Math.log(rate) - lgamma(shape) - (shape + 1) * Math.log(x) - rate / x;
      },
      sample: function (rand, shape, rate) {
        return 1.0 / (gammaStd(rand, shape) / rate);
      }
    },

    // DiscreteUniform(low, high): inclusive integer range [low, high]. i64.
    discreteuniform: {
      logpmf: function (k, low, high) {
        if (high < low) return -Infinity;
        if (k < low || k > high) return -Infinity;
        return -Math.log(high - low + 1);
      },
      sample: function (rand, low, high) {
        return low + Math.floor(rand() * (high - low + 1));
      }
    }
  };

  // ==========================================================================
  // Theming
  // ==========================================================================

  var LIGHT_THEMES = { light: 1, rust: 1 };

  var DARK_COLORS = {
    prior: "#58A6FF",
    data: "#F2CC60",
    post: "#56D364",
    hot: "#FF7B72",
    flow: "#BC8CFF",
    ink: "rgba(230,237,243,0.9)",
    grid: "rgba(230,237,243,0.08)",
    panel: "rgba(110,118,129,0.08)"
  };
  var LIGHT_COLORS = {
    prior: "#0969DA",
    data: "#9A6700",
    post: "#1A7F37",
    hot: "#CF222E",
    flow: "#8250DF",
    ink: "rgba(31,35,40,0.9)",
    grid: "rgba(31,35,40,0.08)",
    panel: "rgba(175,184,193,0.12)"
  };

  function isDark() {
    if (typeof document === "undefined") return true;
    // Trust the page's actual rendered ground, not the theme class name:
    // mdbook stamps whatever default-theme names, valid or not (a bogus name
    // falls back to light CSS while the class still says otherwise), so class
    // whitelists mislabel exactly the broken case. Luminance can't.
    try {
      // mdbook's own ground token; hex like #ffffff (light) / #161923 (navy).
      var bg = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim();
      var m = bg.match(/^#([0-9a-f]{6})$/i);
      if (m) {
        var n = parseInt(m[1], 16);
        var lum = 0.2126 * (n >> 16 & 255) + 0.7152 * (n >> 8 & 255) + 0.0722 * (n & 255);
        return lum < 128;
      }
      m = bg.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
      if (m) {
        return 0.2126 * m[1] + 0.7152 * m[2] + 0.0722 * m[3] < 128;
      }
      m = bg.match(/hsla?\(\s*[\d.]+\s*,\s*[\d.]+%\s*,\s*([\d.]+)%/);
      if (m) {
        return parseFloat(m[1]) < 50;
      }
    } catch (e) { /* fall through to class heuristic */ }
    var cls = document.documentElement.className || "";
    var names = cls.split(/\s+/);
    for (var i = 0; i < names.length; i++) {
      if (LIGHT_THEMES[names[i]]) return false;
    }
    return true;
  }

  function readColor(name, fallback) {
    if (typeof getComputedStyle === "undefined") return fallback;
    try {
      var v = getComputedStyle(document.documentElement).getPropertyValue("--fv-" + name);
      v = v && v.trim();
      return v || fallback;
    } catch (e) {
      return fallback;
    }
  }

  function theme() {
    var dark = isDark();
    var base = dark ? DARK_COLORS : LIGHT_COLORS;
    return {
      dark: dark,
      colors: {
        prior: readColor("prior", base.prior),
        data: readColor("data", base.data),
        post: readColor("post", base.post),
        hot: readColor("hot", base.hot),
        flow: readColor("flow", base.flow),
        ink: readColor("ink", base.ink),
        grid: readColor("grid", base.grid),
        panel: readColor("panel", base.panel)
      }
    };
  }

  var themeListeners = [];
  var themeObserver = null;
  function onThemeChange(fn) {
    themeListeners.push(fn);
    if (!themeObserver && typeof MutationObserver !== "undefined" && typeof document !== "undefined") {
      var last = isDark();
      themeObserver = new MutationObserver(function () {
        var now = isDark();
        if (now !== last) {
          last = now;
          var t = theme();
          for (var i = 0; i < themeListeners.length; i++) {
            try {
              themeListeners[i](t);
            } catch (e) {}
          }
        }
      });
      themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    }
  }

  // ==========================================================================
  // Canvas scaffolding
  // ==========================================================================

  function canvas(parentEl, opts) {
    opts = opts || {};
    var height = opts.height || 300;
    var el = document.createElement("canvas");
    el.className = "fv-canvas";
    el.style.display = "block";
    el.style.width = "100%";
    el.style.height = height + "px";
    parentEl.appendChild(el);
    var ctx = el.getContext("2d");

    var api = { ctx: ctx, el: el, w: 0, h: 0, dpr: 1, clear: clear };

    function resize() {
      var dpr = window.devicePixelRatio || 1;
      var rect = el.getBoundingClientRect();
      var cssW = Math.max(1, rect.width || parentEl.clientWidth || 300);
      var cssH = height;
      el.width = Math.round(cssW * dpr);
      el.height = Math.round(cssH * dpr);
      api.w = cssW;
      api.h = cssH;
      api.dpr = dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (opts.onResize) opts.onResize(api);
    }

    function clear() {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, el.width, el.height);
      ctx.setTransform(api.dpr, 0, 0, api.dpr, 0, 0);
    }

    resize();
    if (typeof ResizeObserver !== "undefined") {
      var ro = new ResizeObserver(function () {
        resize();
      });
      ro.observe(parentEl);
      api._ro = ro;
    } else if (typeof window !== "undefined") {
      window.addEventListener("resize", resize);
    }
    return api;
  }

  // Linear scale: maps domain [d0,d1] -> range [r0,r1]. Returns fn with
  // .invert, .domain, .range.
  function scale(domain, range) {
    var d0 = domain[0], d1 = domain[1], r0 = range[0], r1 = range[1];
    var dspan = d1 - d0 || 1;
    var f = function (x) {
      return r0 + ((x - d0) / dspan) * (r1 - r0);
    };
    f.invert = function (y) {
      return d0 + ((y - r0) / (r1 - r0 || 1)) * dspan;
    };
    f.domain = domain;
    f.range = range;
    return f;
  }

  function niceTicks(lo, hi, count) {
    count = count || 5;
    var span = hi - lo;
    if (span <= 0 || !isFinite(span)) return [lo];
    var step = Math.pow(10, Math.floor(Math.log(span / count) / Math.LN10));
    var err = (span / count) / step;
    if (err >= 7.5) step *= 10;
    else if (err >= 3.5) step *= 5;
    else if (err >= 1.5) step *= 2;
    var start = Math.ceil(lo / step) * step;
    var out = [];
    for (var v = start; v <= hi + step * 1e-6; v += step) {
      out.push(Math.abs(v) < step * 1e-6 ? 0 : v);
    }
    return out;
  }

  function fmtTick(v) {
    if (v === 0) return "0";
    var a = Math.abs(v);
    if (a >= 1e5 || a < 1e-3) return v.toExponential(0);
    return String(Math.round(v * 1000) / 1000);
  }

  // Draws axes/gridlines/labels. opts: {x, y, w, h, xscale, yscale,
  // xlabel, ylabel, theme}. x,y = pixel origin of the plot's bottom-left area
  // (defaults to a sensible inset). If xscale/yscale given, ticks are drawn.
  function axes(ctx, opts) {
    var t = opts.theme || theme();
    var c = t.colors;
    var x0 = opts.x != null ? opts.x : 0;
    var y0 = opts.y != null ? opts.y : 0;
    var w = opts.w, h = opts.h;
    ctx.save();
    ctx.lineWidth = 1;
    ctx.strokeStyle = c.grid;
    ctx.fillStyle = c.ink;
    ctx.font = "11px var(--mono-font, monospace)";
    ctx.textBaseline = "top";
    ctx.textAlign = "center";

    if (opts.xscale) {
      var xt = niceTicks(opts.xscale.domain[0], opts.xscale.domain[1], 6);
      for (var i = 0; i < xt.length; i++) {
        var px = opts.xscale(xt[i]);
        ctx.beginPath();
        ctx.moveTo(px, y0);
        ctx.lineTo(px, y0 + h);
        ctx.stroke();
        ctx.fillText(fmtTick(xt[i]), px, y0 + h + 4);
      }
    }
    if (opts.yscale) {
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      var yt = niceTicks(opts.yscale.domain[0], opts.yscale.domain[1], 5);
      for (var j = 0; j < yt.length; j++) {
        var py = opts.yscale(yt[j]);
        ctx.beginPath();
        ctx.moveTo(x0, py);
        ctx.lineTo(x0 + w, py);
        ctx.stroke();
        ctx.fillText(fmtTick(yt[j]), x0 - 4, py);
      }
    }
    // Axis frame
    ctx.strokeStyle = c.ink;
    ctx.globalAlpha = 0.35;
    ctx.strokeRect(x0, y0, w, h);
    ctx.globalAlpha = 1;

    if (opts.xlabel) {
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(opts.xlabel, x0 + w / 2, y0 + h + 24);
    }
    if (opts.ylabel) {
      ctx.save();
      ctx.translate(x0 - 30, y0 + h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(opts.ylabel, 0, 0);
      ctx.restore();
    }
    ctx.restore();
  }

  // Polyline through pts (array of [px, py] in PIXEL coords). opts: {color,
  // width, dash}.
  function curve(ctx, pts, opts) {
    opts = opts || {};
    if (!pts || pts.length === 0) return;
    ctx.save();
    ctx.strokeStyle = opts.color || (theme().colors.ink);
    ctx.lineWidth = opts.width || 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    if (opts.dash) ctx.setLineDash(opts.dash);
    ctx.beginPath();
    var started = false;
    for (var i = 0; i < pts.length; i++) {
      var p = pts[i];
      if (!p || !isFinite(p[0]) || !isFinite(p[1])) {
        started = false;
        continue;
      }
      if (!started) {
        ctx.moveTo(p[0], p[1]);
        started = true;
      } else {
        ctx.lineTo(p[0], p[1]);
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  // Histogram of `samples` (numbers). opts: {bins, xscale, yscale, color,
  // alpha}. Bars are drawn as a DENSITY (area = 1) so they compare directly to
  // a pdf drawn with the same yscale. Bin edges span xscale.domain.
  function histogram(ctx, samples, opts) {
    opts = opts || {};
    var bins = opts.bins || 30;
    var xs = opts.xscale, ys = opts.yscale;
    if (!xs || !ys || !samples || samples.length === 0) return;
    var lo = xs.domain[0], hi = xs.domain[1];
    var width = (hi - lo) / bins;
    if (width <= 0) return;
    var counts = new Array(bins);
    for (var b = 0; b < bins; b++) counts[b] = 0;
    var n = 0;
    for (var i = 0; i < samples.length; i++) {
      var v = samples[i];
      if (v < lo || v >= hi || !isFinite(v)) continue;
      var idx = Math.floor((v - lo) / width);
      if (idx < 0) idx = 0;
      if (idx >= bins) idx = bins - 1;
      counts[idx]++;
      n++;
    }
    if (n === 0) return;
    var baseline = ys(0);
    ctx.save();
    ctx.globalAlpha = opts.alpha != null ? opts.alpha : 0.55;
    ctx.fillStyle = opts.color || theme().colors.post;
    for (var k = 0; k < bins; k++) {
      var density = counts[k] / (n * width);
      var xa = xs(lo + k * width);
      var xb = xs(lo + (k + 1) * width);
      var yTop = ys(density);
      ctx.fillRect(xa, yTop, xb - xa, baseline - yTop);
    }
    ctx.restore();
  }

  function hexToRgb(hex) {
    hex = (hex || "").trim();
    var m = /^#?([0-9a-f]{6})$/i.exec(hex);
    if (m) {
      var n = parseInt(m[1], 16);
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }
    var rgb = /rgba?\(([^)]+)\)/.exec(hex);
    if (rgb) {
      var parts = rgb[1].split(",");
      return [parseInt(parts[0], 10), parseInt(parts[1], 10), parseInt(parts[2], 10)];
    }
    return [128, 128, 128];
  }

  // Heatmap of scalar field f(x, y) (data coords). opts: {xscale, yscale, w, h,
  // colormap}. colormap 'post'|'flow' (or a hex): transparent -> color ramp,
  // normalized to the field's max over the sampled grid.
  function heatmap(ctx, f, opts) {
    opts = opts || {};
    var xs = opts.xscale, ys = opts.yscale;
    var w = opts.w, h = opts.h;
    var t = theme();
    var colHex = opts.colormap === "flow" ? t.colors.flow : opts.colormap === "post" ? t.colors.post : (opts.colormap || t.colors.post);
    var rgb = hexToRgb(colHex);
    var step = opts.step || 4; // pixel block size
    var cols = Math.ceil(w / step);
    var rows = Math.ceil(h / step);
    // sample field
    var vals = new Array(cols * rows);
    var maxv = -Infinity;
    for (var iy = 0; iy < rows; iy++) {
      for (var ix = 0; ix < cols; ix++) {
        var dx = xs.invert(ix * step + step / 2);
        var dy = ys.invert(iy * step + step / 2);
        var v = f(dx, dy);
        if (!isFinite(v)) v = 0;
        vals[iy * cols + ix] = v;
        if (v > maxv) maxv = v;
      }
    }
    if (!isFinite(maxv) || maxv <= 0) maxv = 1;
    ctx.save();
    for (var jy = 0; jy < rows; jy++) {
      for (var jx = 0; jx < cols; jx++) {
        var a = vals[jy * cols + jx] / maxv;
        if (a <= 0.002) continue;
        if (a > 1) a = 1;
        ctx.fillStyle = "rgba(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + "," + a + ")";
        ctx.fillRect(jx * step, jy * step, step, step);
      }
    }
    ctx.restore();
  }

  // ==========================================================================
  // Controls (all keyboard-accessible; return the root element)
  // ==========================================================================

  function el(tag, cls, parent) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (parent) parent.appendChild(e);
    return e;
  }

  function slider(parentEl, o) {
    o = o || {};
    var root = el("label", "fv-control", parentEl);
    var lab = el("span", "fv-control-label", root);
    lab.textContent = o.label || "";
    var input = el("input", "fv-range", root);
    input.type = "range";
    input.min = o.min;
    input.max = o.max;
    input.step = o.step != null ? o.step : "any";
    input.value = o.value != null ? o.value : o.min;
    var out = el("span", "fv-control-value", root);
    var fmt = o.fmt || function (v) { return String(v); };
    function render(v) {
      out.textContent = fmt(v);
    }
    render(parseFloat(input.value));
    input.addEventListener("input", function () {
      var v = parseFloat(input.value);
      render(v);
      if (o.onInput) o.onInput(v);
    });
    root.fvSet = function (v) {
      input.value = v;
      render(parseFloat(input.value));
    };
    root.fvGet = function () {
      return parseFloat(input.value);
    };
    return root;
  }

  function buttons(parentEl, specs) {
    var root = el("div", "fv-buttons", parentEl);
    root.fvButtons = {};
    for (var i = 0; i < specs.length; i++) {
      (function (spec) {
        var b = el("button", "fv-btn" + (spec.primary ? " fv-primary" : ""), root);
        b.type = "button";
        b.textContent = spec.label;
        if (spec.title) b.title = spec.title;
        b.addEventListener("click", function () {
          if (spec.onClick) spec.onClick();
        });
        root.fvButtons[spec.label] = b;
      })(specs[i]);
    }
    return root;
  }

  function toggle(parentEl, o) {
    o = o || {};
    var root = el("label", "fv-control fv-toggle", parentEl);
    var input = el("input", "fv-checkbox", root);
    input.type = "checkbox";
    input.checked = !!o.value;
    var lab = el("span", "fv-control-label", root);
    lab.textContent = o.label || "";
    input.addEventListener("change", function () {
      if (o.onChange) o.onChange(input.checked);
    });
    root.fvSet = function (v) {
      input.checked = !!v;
    };
    root.fvGet = function () {
      return input.checked;
    };
    return root;
  }

  function readout(parentEl, o) {
    o = o || {};
    var root = el("div", "fv-readout", parentEl);
    var lab = el("span", "fv-readout-label", root);
    lab.textContent = o.label || "";
    var val = el("span", "fv-readout-value", root);
    val.textContent = "—";
    return {
      el: root,
      set: function (txt, colorRole) {
        val.textContent = txt;
        val.style.color = colorRole ? "var(--fv-" + colorRole + ")" : "";
      }
    };
  }

  // Victor-style draggable number. Binds to a <span>. Drag horizontally to
  // change (ew-resize cursor, coral while active); also arrow-key steppable when
  // focused. onInput(value) fires on change. Returns the span, with .fvSet.
  function scrub(spanEl, o) {
    o = o || {};
    var min = o.min != null ? o.min : parseFloat(spanEl.getAttribute("data-min"));
    var max = o.max != null ? o.max : parseFloat(spanEl.getAttribute("data-max"));
    var step = o.step != null ? o.step : (parseFloat(spanEl.getAttribute("data-step")) || 1);
    var value = o.value != null ? o.value : (parseFloat(spanEl.getAttribute("data-value")) || min || 0);
    var fmt = o.fmt || function (v) { return String(v); };
    var decimals = (String(step).split(".")[1] || "").length;

    spanEl.className = (spanEl.className ? spanEl.className + " " : "") + "fv-scrub";
    spanEl.setAttribute("tabindex", "0");
    spanEl.setAttribute("role", "slider");
    spanEl.setAttribute("aria-valuemin", min);
    spanEl.setAttribute("aria-valuemax", max);

    function clamp(v) {
      if (min != null && v < min) v = min;
      if (max != null && v > max) v = max;
      var q = Math.round(v / step) * step;
      return decimals ? parseFloat(q.toFixed(decimals)) : q;
    }
    function render() {
      spanEl.textContent = fmt(value);
      spanEl.setAttribute("aria-valuenow", value);
    }
    function emit() {
      render();
      if (o.onInput) o.onInput(value);
    }

    var dragging = false, startX = 0, startVal = 0, pid = null;
    var range = (max - min) || 1;
    // Prefer Pointer Events: setPointerCapture keeps move/up on the span itself
    // (no window listeners), and `.fv-scrub` sets touch-action:none, so a thumb
    // drag scrubs cleanly and never scroll-fights the page. Legacy fallback keeps
    // the old mouse+touch path for browsers without PointerEvent.
    var usePointer = typeof window !== "undefined" && !!window.PointerEvent;

    function coordX(e) {
      if (e.touches && e.touches[0]) return e.touches[0].clientX;
      if (e.changedTouches && e.changedTouches[0]) return e.changedTouches[0].clientX;
      return e.clientX;
    }
    function onDown(e) {
      dragging = true;
      startX = coordX(e);
      startVal = value;
      spanEl.classList.add("fv-scrub-active");
      if (usePointer) {
        pid = e.pointerId;
        if (spanEl.setPointerCapture && pid != null) {
          try { spanEl.setPointerCapture(pid); } catch (err) {}
        }
      } else {
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
        window.addEventListener("touchmove", onMove, { passive: false });
        window.addEventListener("touchend", onUp);
      }
      if (e.cancelable) e.preventDefault();
    }
    function onMove(e) {
      if (!dragging) return;
      if (usePointer && pid != null && e.pointerId != null && e.pointerId !== pid) return;
      var dx = coordX(e) - startX;
      // ~200px of drag traverses the full range
      value = clamp(startVal + (dx / 200) * range);
      emit();
      if (e.cancelable) e.preventDefault();
    }
    function onUp(e) {
      if (!dragging) return;
      dragging = false;
      spanEl.classList.remove("fv-scrub-active");
      if (usePointer) {
        if (spanEl.releasePointerCapture && pid != null) {
          try { spanEl.releasePointerCapture(pid); } catch (err) {}
        }
        pid = null;
      } else {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        window.removeEventListener("touchmove", onMove);
        window.removeEventListener("touchend", onUp);
      }
    }
    if (usePointer) {
      spanEl.addEventListener("pointerdown", onDown);
      spanEl.addEventListener("pointermove", onMove);
      spanEl.addEventListener("pointerup", onUp);
      spanEl.addEventListener("pointercancel", onUp);
    } else {
      spanEl.addEventListener("mousedown", onDown);
      spanEl.addEventListener("touchstart", onDown, { passive: false });
    }
    spanEl.addEventListener("keydown", function (e) {
      var d = 0;
      if (e.key === "ArrowLeft" || e.key === "ArrowDown") d = -1;
      else if (e.key === "ArrowRight" || e.key === "ArrowUp") d = 1;
      else if (e.key === "Home") { value = clamp(min); emit(); e.preventDefault(); return; }
      else if (e.key === "End") { value = clamp(max); emit(); e.preventDefault(); return; }
      if (d !== 0) {
        var m = e.shiftKey ? 10 : 1;
        value = clamp(value + d * step * m);
        emit();
        e.preventDefault();
      }
    });

    value = clamp(value);
    render();
    spanEl.fvSet = function (v) {
      value = clamp(v);
      render();
    };
    spanEl.fvGet = function () {
      return value;
    };
    return spanEl;
  }

  // ==========================================================================
  // Animation loop
  // ==========================================================================

  var reduceMotion = false;
  if (typeof window !== "undefined" && window.matchMedia) {
    try {
      var mq = window.matchMedia("(prefers-reduced-motion: reduce)");
      reduceMotion = mq.matches;
      if (mq.addEventListener) mq.addEventListener("change", function (e) { reduceMotion = e.matches; });
    } catch (e) {}
  }

  // loop(widgetRootEl, tickFn, opts) -> {play(), pause(), step(), playing, reduced}.
  //
  // Drives a rAF animation of tickFn(dt). Auto-pauses when the widget scrolls
  // offscreen (IntersectionObserver) or the tab is hidden, and resumes on return.
  //
  // opts.autoplay (boolean): when true, the widget begins playing the moment it
  // initializes — init is already lazy on scroll-into-view, so nobody lands on a
  // dead canvas. Autoplay routes through play(), which is a no-op under
  // prefers-reduced-motion and under the offscreen/hidden guards; so a
  // reduced-motion visitor never gets autoplaying animation (the widget should
  // render a fully-formed static frame instead), and an offscreen autoplay simply
  // resumes once scrolled into view. The returned API is identical with or without
  // opts — autoplay only changes whether play() is invoked once at the end of setup.
  function loop(widgetRootEl, tickFn, opts) {
    opts = opts || {};
    var raf = null;
    var playing = false;
    var onscreen = true;
    var lastT = 0;
    var api = { play: play, pause: pause, step: step, playing: false, get reduced() { return reduceMotion; } };

    function frame(now) {
      if (!playing) return;
      var dt = lastT ? (now - lastT) / 1000 : 0;
      lastT = now;
      try {
        tickFn(dt);
      } catch (e) {
        pause();
        throw e;
      }
      raf = window.requestAnimationFrame(frame);
    }
    function play() {
      if (reduceMotion) return; // no autoplaying animation under reduced-motion
      if (playing) return;
      if (!onscreen || (typeof document !== "undefined" && document.hidden)) return;
      playing = true;
      api.playing = true;
      lastT = 0;
      raf = window.requestAnimationFrame(frame);
    }
    function pause() {
      playing = false;
      api.playing = false;
      if (raf) window.cancelAnimationFrame(raf);
      raf = null;
    }
    function step() {
      try {
        tickFn(0);
      } catch (e) {
        throw e;
      }
    }

    // Auto-pause offscreen.
    if (typeof IntersectionObserver !== "undefined" && widgetRootEl) {
      var io = new IntersectionObserver(function (entries) {
        for (var i = 0; i < entries.length; i++) {
          onscreen = entries[i].isIntersecting;
          if (!onscreen && playing) {
            var wasPlaying = true;
            pause();
            api._wasPlaying = wasPlaying;
          } else if (onscreen && api._wasPlaying && !reduceMotion) {
            api._wasPlaying = false;
            play();
          }
        }
      }, { threshold: 0.01 });
      io.observe(widgetRootEl);
    }
    // Auto-pause when tab hidden.
    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", function () {
        if (document.hidden && playing) {
          pause();
          api._wasPlaying = true;
        } else if (!document.hidden && api._wasPlaying && onscreen && !reduceMotion) {
          api._wasPlaying = false;
          play();
        }
      });
    }
    // Autoplay on init (respects reduced-motion / offscreen guards inside play()).
    if (opts.autoplay) play();
    return api;
  }

  // ==========================================================================
  // Touch & smoothness helpers (see §A of the explorables spec)
  //
  // These exist so every widget handles a thumb the same, correct way:
  //  - a canvas drag NEVER scroll-fights the page (claim the gesture only on a
  //    real hit; otherwise let the page scroll),
  //  - coarse pointers get inflated hit targets (>=22 CSS px),
  //  - state advances on its own clock while render tweens between states.
  // ==========================================================================

  // True on touch/stylus-primary devices. Checked live (not cached) so hybrid
  // laptops that gain/lose a touchscreen answer correctly per gesture.
  function isCoarsePointer() {
    if (typeof window === "undefined" || !window.matchMedia) return false;
    try {
      return window.matchMedia("(pointer: coarse)").matches;
    } catch (e) {
      return false;
    }
  }

  // drag(canvasEl, opts) -> handle. An OPT-IN pointer drag manager for a canvas.
  //
  // The whole point is scroll-fight avoidance: on pointerdown it runs your
  // hitTest; ONLY when that returns a target does it claim the gesture
  // (setPointerCapture + preventDefault) so the drag can never scroll the page.
  // A miss is ignored entirely, so a thumb on empty canvas still scrolls.
  //
  // opts:
  //   hitTest(x, y, slop) -> target   REQUIRED. x,y are CSS px from the canvas
  //       top-left; `slop` is the current hit-inflation (>=22 on coarse pointers,
  //       else opts.inflate). Return any truthy target (an index, an object) the
  //       pointer is over, or a "miss" sentinel: null / undefined / false / -1.
  //   onStart(target, x, y, ev)       optional; once when a grab begins.
  //   onDrag(target, x, y, ev)        called on every move while grabbing.
  //   onEnd(target, ev)               optional; when the grab releases/cancels.
  //   inflate  (number, default 0)    base hit slop on FINE pointers; coarse
  //                                   pointers always get at least 22.
  //   fullCapture (bool, default true) true adds `.fv-touch-none` to the canvas
  //       (touch-action:none) — the whole canvas is treated as interactive, so a
  //       drag is perfectly smooth but a plain swipe over it won't scroll the
  //       page. Set false for a mostly-ambient canvas that should still scroll on
  //       a swipe (a hit is still claimed best-effort). Ambient-only micros
  //       should simply NOT call drag(): their canvas stays pan-y and scrolls.
  //
  // Returns { grabbed, target, slop, isCoarse, destroy() } where `grabbed` (bool)
  // and `target` are live getters — draw your grab halo (see halo()) while
  // `grabbed` is true. destroy() removes every listener and drops the class.
  function drag(canvasEl, opts) {
    opts = opts || {};
    var hitTest = opts.hitTest || function () { return null; };
    var baseInflate = opts.inflate || 0;
    var fullCapture = opts.fullCapture !== false; // default true
    var usePointer = typeof window !== "undefined" && !!window.PointerEvent;

    if (fullCapture) canvasEl.classList.add("fv-touch-none");

    var state = { grabbed: false, target: null, slop: baseInflate, isCoarse: false };
    var activeId = null;

    function pt(e) {
      var t = (e.touches && e.touches[0]) || (e.changedTouches && e.changedTouches[0]);
      if (t) return { x: t.clientX, y: t.clientY, id: t.identifier };
      return { x: e.clientX, y: e.clientY, id: e.pointerId != null ? e.pointerId : 0 };
    }
    function local(p) {
      var r = canvasEl.getBoundingClientRect();
      return [p.x - r.left, p.y - r.top];
    }
    function isHit(t) {
      return t !== null && t !== undefined && t !== false && t !== -1;
    }
    function slopFor() {
      var c = isCoarsePointer();
      state.isCoarse = c;
      return c ? Math.max(22, baseInflate) : baseInflate;
    }
    // For a pointer event, only the captured pointer drives move/end.
    function otherPointer(e) {
      return usePointer && e.pointerId != null && activeId != null && e.pointerId !== activeId;
    }

    function begin(e) {
      if (state.grabbed) return;
      var slop = slopFor();
      state.slop = slop;
      var p = pt(e), xy = local(p);
      var target = hitTest(xy[0], xy[1], slop);
      if (!isHit(target)) return; // miss: don't claim — page scrolls, others run
      state.grabbed = true;
      state.target = target;
      activeId = p.id;
      canvasEl.classList.add("fv-grabbing");
      if (usePointer && canvasEl.setPointerCapture && e.pointerId != null) {
        try { canvasEl.setPointerCapture(e.pointerId); } catch (err) {}
      } else if (!usePointer) {
        window.addEventListener("mousemove", move);
        window.addEventListener("mouseup", end);
        window.addEventListener("touchmove", move, { passive: false });
        window.addEventListener("touchend", end);
        window.addEventListener("touchcancel", end);
      }
      if (e.cancelable) e.preventDefault();
      if (opts.onStart) opts.onStart(target, xy[0], xy[1], e);
    }
    function move(e) {
      if (!state.grabbed || otherPointer(e)) return;
      var xy = local(pt(e));
      if (opts.onDrag) opts.onDrag(state.target, xy[0], xy[1], e);
      if (e.cancelable) e.preventDefault();
    }
    function end(e) {
      if (!state.grabbed || otherPointer(e)) return;
      state.grabbed = false;
      canvasEl.classList.remove("fv-grabbing");
      if (usePointer && canvasEl.releasePointerCapture && e.pointerId != null) {
        try { canvasEl.releasePointerCapture(e.pointerId); } catch (err) {}
      } else if (!usePointer) {
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", end);
        window.removeEventListener("touchmove", move);
        window.removeEventListener("touchend", end);
        window.removeEventListener("touchcancel", end);
      }
      var t = state.target;
      state.target = null;
      activeId = null;
      if (opts.onEnd) opts.onEnd(t, e);
    }
    function hover(e) {
      if (state.grabbed) return;
      var xy = local(pt(e));
      canvasEl.style.cursor = isHit(hitTest(xy[0], xy[1], state.slop || baseInflate)) ? "grab" : "";
    }

    if (usePointer) {
      canvasEl.addEventListener("pointerdown", begin);
      canvasEl.addEventListener("pointermove", move);
      canvasEl.addEventListener("pointerup", end);
      canvasEl.addEventListener("pointercancel", end);
      canvasEl.addEventListener("pointermove", hover);
    } else {
      canvasEl.addEventListener("mousedown", begin);
      canvasEl.addEventListener("touchstart", begin, { passive: false });
      canvasEl.addEventListener("mousemove", hover);
    }

    return {
      get grabbed() { return state.grabbed; },
      get target() { return state.target; },
      get slop() { return state.slop; },
      get isCoarse() { return state.isCoarse; },
      destroy: function () {
        if (usePointer) {
          canvasEl.removeEventListener("pointerdown", begin);
          canvasEl.removeEventListener("pointermove", move);
          canvasEl.removeEventListener("pointerup", end);
          canvasEl.removeEventListener("pointercancel", end);
          canvasEl.removeEventListener("pointermove", hover);
        } else {
          canvasEl.removeEventListener("mousedown", begin);
          canvasEl.removeEventListener("touchstart", begin);
          canvasEl.removeEventListener("mousemove", hover);
          window.removeEventListener("mousemove", move);
          window.removeEventListener("mouseup", end);
          window.removeEventListener("touchmove", move);
          window.removeEventListener("touchend", end);
          window.removeEventListener("touchcancel", end);
        }
        if (fullCapture) canvasEl.classList.remove("fv-touch-none");
      }
    };
  }

  // halo(ctx, x, y, r, color, alpha) — draw a soft grab-halo ring at pixel (x,y).
  // Call from render while a drag handle's api.grabbed is true (§A.2: "show a
  // subtle halo on the grabbed point while dragging"). Defaults to the hot color.
  function halo(ctx, x, y, r, color, alpha) {
    if (!isFinite(x) || !isFinite(y)) return;
    var col = color || theme().colors.hot;
    var a = alpha != null ? alpha : 0.35;
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.globalAlpha = a * 0.4;
    ctx.fillStyle = col;
    ctx.fill();
    ctx.globalAlpha = a;
    ctx.lineWidth = 2;
    ctx.strokeStyle = col;
    ctx.stroke();
    ctx.restore();
  }

  // pace(hz, maxPerFrame) -> step(dt) -> integer count.
  //
  // Tick/render decoupling (§A.3): advance logical state at a FIXED `hz`
  // steps/sec regardless of the render frame rate. Each frame call step(dt) with
  // that frame's dt (seconds); it returns how many logical ticks to run now,
  // carrying the sub-tick remainder to the next frame — so no stutter when rAF
  // drops frames. After a long stall (backgrounded tab) the backlog is capped at
  // maxPerFrame (default 5) and the remainder dropped, avoiding a spiral of death.
  //
  //   var pacer = FV.pace(60);
  //   ... inside loop tick(dt): for (var n = pacer(dt); n-- > 0; ) advance();
  function pace(hz, maxPerFrame) {
    var acc = 0;
    var cap = maxPerFrame > 0 ? maxPerFrame : 5;
    return function (dt) {
      if (!(dt > 0) || !(hz > 0)) return 0;
      acc += dt * hz;
      var n = Math.floor(acc);
      if (n > cap) { n = cap; acc = 0; }
      else acc -= n;
      return n;
    };
  }

  // ==========================================================================
  // Widget lifecycle — lazy init via IntersectionObserver
  // ==========================================================================

  var registry = {};
  var pending = [];
  var lifecycleObserver = null;

  function ensureObserver() {
    if (lifecycleObserver || typeof IntersectionObserver === "undefined") return;
    lifecycleObserver = new IntersectionObserver(function (entries) {
      for (var i = 0; i < entries.length; i++) {
        var entry = entries[i];
        if (entry.isIntersecting) {
          maybeInit(entry.target);
        }
      }
    }, { rootMargin: "200px" });
  }

  function maybeInit(root) {
    if (!root || root._fvInited) return;
    var name = root.getAttribute("data-viz");
    var initFn = registry[name];
    if (!initFn) return; // widget script not loaded yet
    root._fvInited = true;
    if (lifecycleObserver) lifecycleObserver.unobserve(root);
    try {
      initFn(root, FugueViz);
    } catch (e) {
      root._fvInited = false;
      if (typeof console !== "undefined") console.error("[FugueViz] init failed for '" + name + "'", e);
    }
  }

  function scan() {
    if (typeof document === "undefined") return;
    var nodes = document.querySelectorAll(".fugue-explorable[data-viz]");
    ensureObserver();
    for (var i = 0; i < nodes.length; i++) {
      var node = nodes[i];
      if (node._fvInited || node._fvObserved) continue;
      if (lifecycleObserver) {
        node._fvObserved = true;
        lifecycleObserver.observe(node);
      } else {
        maybeInit(node); // no IO support: init eagerly
      }
    }
  }

  function register(name, initFn) {
    registry[name] = initFn;
    // A matching element may already be scrolled into view; try to init it.
    if (typeof document !== "undefined") {
      var nodes = document.querySelectorAll('.fugue-explorable[data-viz="' + name + '"]');
      for (var i = 0; i < nodes.length; i++) {
        var node = nodes[i];
        if (node._fvObserved || node._fvInited) {
          // already tracked; the observer will fire, but init now if visible
          maybeInit(node);
        } else {
          ensureObserver();
          if (lifecycleObserver) {
            node._fvObserved = true;
            lifecycleObserver.observe(node);
          } else {
            maybeInit(node);
          }
        }
      }
    }
  }

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", scan);
    } else {
      scan();
    }
    // Re-scan on load in case widget scripts registered after first scan.
    if (typeof window !== "undefined") {
      window.addEventListener("load", scan);
    }
  }

  // ==========================================================================
  // Public API
  // ==========================================================================

  var FugueViz = {
    register: register,
    theme: theme,
    onThemeChange: onThemeChange,
    rng: rng,
    randn: randn,
    dist: dist,
    lgamma: lgamma,
    logsumexp: logsumexp,
    log1p: log1p,
    gammaStd: gammaStd,
    canvas: canvas,
    scale: scale,
    axes: axes,
    curve: curve,
    histogram: histogram,
    heatmap: heatmap,
    slider: slider,
    buttons: buttons,
    toggle: toggle,
    readout: readout,
    scrub: scrub,
    loop: loop,
    drag: drag,
    halo: halo,
    pace: pace,
    isCoarsePointer: isCoarsePointer,
    _registry: registry,
    _scan: scan
  };

  if (typeof window !== "undefined") window.FugueViz = FugueViz;
  if (typeof module !== "undefined" && module.exports) module.exports = FugueViz;
})();
