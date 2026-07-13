//! Statistical goodness-of-fit validation for EVERY exported sampler.
//!
//! Covers finding FG-14: before this file, only `Normal` had any
//! moment-matching or distributional goodness-of-fit test anywhere in the
//! crate (`src/inference/validation.rs`'s `ks_test_distribution` was called
//! exactly 3 times, always on `Normal(0,1)`). A parameterization bug (e.g.
//! `Gamma::new(shape, rate)` silently treated as `Gamma(shape, scale)`, or
//! `Poisson` sampling with variance instead of rate) would have been
//! invisible to the test suite as long as sampled values stayed
//! finite/in-support. This file exercises all 17 distributions exported from
//! `src/core/distribution.rs`:
//!
//! - **Continuous** (12): a one-sample Kolmogorov-Smirnov test of `n = 5000`
//!   seeded draws against the distribution's own analytic CDF, at
//!   `alpha = 0.001`.
//! - **Discrete** (5): a chi-square goodness-of-fit test of `n = 5000` seeded
//!   draws against the analytic PMF, at `alpha = 0.001`.
//! - **All 17**: a standardized-moment check `|sample_mean - mu| / SE < 5`
//!   (SE = sigma / sqrt(n), a ~5-sigma band under the CLT so the false-positive
//!   rate is astronomically small while still catching a mean computed with
//!   the wrong parameterization).
//!
//! ## Special functions
//!
//! No `statrs` dependency exists in this crate, and this sandbox has no
//! network access to fetch one, so the analytic CDFs are hand-implemented
//! here from the two special functions that generate all of them: the
//! regularized lower incomplete gamma `P(a,x)` (Numerical Recipes
//! `gser`/`gcf`) and the regularized incomplete beta `I_x(a,b)` (Numerical
//! Recipes `betacf`). `erf` itself is *not* hand-approximated separately —
//! it is obtained exactly from the identity `erf(x) = P(1/2, x^2)` (since
//! `gamma(1/2, x^2) = sqrt(pi)*erf(x)` and `Gamma(1/2) = sqrt(pi)`), so Normal
//! and LogNormal ultimately route through the same verified `gammp`.
//!
//! Both `gammp` and `betai` were independently cross-checked (offline, via
//! `tests/gen_refs.py`, pure-stdlib Simpson's-rule numerical integration of
//! the raw PDFs plus closed-form special cases: `Gamma(1,r) == Exponential(r)`,
//! `ChiSquared(2) == Exponential(0.5)`, `Beta(1,1) == Uniform(0,1)`,
//! `StudentT(1) == Cauchy`) before being ported here; see that script for the
//! derivation. All match to >= 1e-8.
//!
//! KS and chi-square critical values use the standard asymptotic formulas,
//! confirmed against the exact Kolmogorov distribution / an independent
//! `gammp`-based inverse-chi-square in the same script.

use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

const N: usize = 5000;
const ALPHA: f64 = 0.001;

// ===========================================================================
// Special functions (Numerical Recipes gser/gcf/betacf), verified offline.
// ===========================================================================

/// Regularized lower incomplete gamma `P(a,x) = gamma(a,x)/Gamma(a)`.
fn gammp(a: f64, x: f64) -> f64 {
    assert!(x >= 0.0 && a > 0.0);
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gser(a, x)
    } else {
        1.0 - gcf(a, x)
    }
}

fn gser(a: f64, x: f64) -> f64 {
    let gln = libm::lgamma(a);
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..500 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-15 {
            break;
        }
    }
    sum * (-x + a * x.ln() - gln).exp()
}

fn gcf(a: f64, x: f64) -> f64 {
    let gln = libm::lgamma(a);
    let fpmin = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..500 {
        let fi = i as f64;
        let an = -fi * (fi - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    (-x + a * x.ln() - gln).exp() * h
}

/// Regularized incomplete beta `I_x(a,b)`.
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt =
        (libm::lgamma(a + b) - libm::lgamma(a) - libm::lgamma(b) + a * x.ln() + b * (1.0 - x).ln())
            .exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let fpmin = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < fpmin {
        d = fpmin;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..500 {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa2 = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa2 * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa2 / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}

/// `erf(x)` via the exact identity `erf(x) = sign(x) * P(1/2, x^2)`.
fn erf(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    x.signum() * gammp(0.5, x * x)
}

fn ln_gamma_fn(x: f64) -> f64 {
    libm::lgamma(x)
}

// ===========================================================================
// Analytic CDFs, one per continuous distribution.
// ===========================================================================

fn normal_cdf(mu: f64, sigma: f64, x: f64) -> f64 {
    0.5 * (1.0 + erf((x - mu) / (sigma * std::f64::consts::SQRT_2)))
}
fn uniform_cdf(low: f64, high: f64, x: f64) -> f64 {
    ((x - low) / (high - low)).clamp(0.0, 1.0)
}
fn lognormal_cdf(mu: f64, sigma: f64, x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        normal_cdf(mu, sigma, x.ln())
    }
}
fn exponential_cdf(rate: f64, x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        1.0 - (-rate * x).exp()
    }
}
fn cauchy_cdf(loc: f64, scale: f64, x: f64) -> f64 {
    0.5 + ((x - loc) / scale).atan() / std::f64::consts::PI
}
fn laplace_cdf(loc: f64, scale: f64, x: f64) -> f64 {
    let z = (x - loc) / scale;
    if x < loc {
        0.5 * z.exp()
    } else {
        1.0 - 0.5 * (-z).exp()
    }
}
fn weibull_cdf(shape: f64, scale: f64, x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        1.0 - (-(x / scale).powf(shape)).exp()
    }
}
fn beta_cdf(a: f64, b: f64, x: f64) -> f64 {
    betai(a, b, x)
}
fn gamma_cdf(shape: f64, rate: f64, x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        gammp(shape, rate * x)
    }
}
fn studentt_cdf(df: f64, loc: f64, scale: f64, x: f64) -> f64 {
    let z = (x - loc) / scale;
    let xarg = df / (df + z * z);
    if z >= 0.0 {
        1.0 - 0.5 * betai(df / 2.0, 0.5, xarg)
    } else {
        0.5 * betai(df / 2.0, 0.5, xarg)
    }
}
fn chi2_cdf(k: f64, x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        gammp(k / 2.0, x / 2.0)
    }
}
fn invgamma_cdf(shape: f64, rate: f64, x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        1.0 - gammp(shape, rate / x)
    }
}

// ===========================================================================
// Generic test machinery.
// ===========================================================================

/// One-sample KS statistic of `samples` against analytic `cdf`.
fn ks_one_sample(samples: &mut [f64], cdf: impl Fn(f64) -> f64) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len() as f64;
    let mut d: f64 = 0.0;
    for (i, &x) in samples.iter().enumerate() {
        let f = cdf(x);
        let d_plus = (i as f64 + 1.0) / n - f;
        let d_minus = f - (i as f64) / n;
        d = d.max(d_plus).max(d_minus);
    }
    d
}

/// Asymptotic two-sided Kolmogorov critical value for `D_n` at `alpha`:
/// `D_crit = sqrt(-0.5*ln(alpha/2)) / sqrt(n)`. Verified in `gen_refs.py`
/// against the exact Kolmogorov CDF (agrees to 1.6e-8 at alpha=0.001).
fn ks_critical(alpha: f64, n: usize) -> f64 {
    (-0.5 * (alpha / 2.0).ln()).sqrt() / (n as f64).sqrt()
}

fn assert_ks_ok(name: &str, mut samples: Vec<f64>, cdf: impl Fn(f64) -> f64) {
    let n = samples.len();
    let d = ks_one_sample(&mut samples, cdf);
    let crit = ks_critical(ALPHA, n);
    assert!(
        d < crit,
        "FG-14: {name} failed one-sample KS test: D={d:.5} >= critical {crit:.5} (n={n}, alpha={ALPHA})"
    );
}

/// `|sample_mean - mu_theory| / (sigma_theory/sqrt(n)) < 5`: a ~5-SE band
/// under the CLT. For n=5000 the false-positive rate of a correct
/// implementation failing this is astronomically small (~5.7e-7 two-sided
/// per test), while a parameterization bug (e.g. rate vs scale, doubled
/// variance) typically shifts the mean or inflates its variance by a
/// constant factor and is caught easily.
fn assert_moment_ok(name: &str, sample_mean: f64, mu_theory: f64, sigma_theory: f64, n: usize) {
    let se = sigma_theory / (n as f64).sqrt();
    let z = (sample_mean - mu_theory).abs() / se;
    assert!(
        z < 5.0,
        "FG-14: {name} mean check failed: sample_mean={sample_mean:.5}, theory={mu_theory:.5}, z={z:.3}"
    );
}

fn chi_square_statistic(observed: &[u64], expected_probs: &[f64], n: u64) -> f64 {
    observed
        .iter()
        .zip(expected_probs)
        .map(|(&o, &p)| {
            let e = p * n as f64;
            (o as f64 - e).powi(2) / e
        })
        .sum()
}

/// Chi-square critical values at alpha=0.001 for the specific degrees of
/// freedom used below, cross-checked in `gen_refs.py` via bisection on the
/// (independently verified) `gammp`-based chi-square CDF against the
/// standard published table (df=1: 10.828, df=3: 16.266, df=5: 20.515,
/// df=10: 29.588, df=13: 34.528).
fn chi2_critical_0_001(df: usize) -> f64 {
    match df {
        1 => 10.827566170662625,
        3 => 16.26623619623801,
        5 => 20.515005652432748,
        10 => 29.588298445074273,
        13 => 34.52817897487073,
        _ => panic!("no tabulated critical value for df={df}, add one via gen_refs.py"),
    }
}

// ===========================================================================
// Continuous distributions: KS + moment checks.
// ===========================================================================

#[test]
fn fg14_normal_ks_and_moments() {
    let mut rng = StdRng::seed_from_u64(1001);
    let (mu, sigma) = (2.0, 3.0);
    let d = Normal::new(mu, sigma).unwrap();
    let samples: Vec<f64> = (0..N).map(|_| d.sample(&mut rng)).collect();
    let mean = samples.iter().sum::<f64>() / N as f64;
    assert_moment_ok("Normal", mean, mu, sigma, N);
    assert_ks_ok("Normal", samples, |x| normal_cdf(mu, sigma, x));
}

#[test]
fn fg14_uniform_ks_and_moments() {
    let mut rng = StdRng::seed_from_u64(1002);
    let (low, high) = (-3.0, 5.0);
    let d = Uniform::new(low, high).unwrap();
    let samples: Vec<f64> = (0..N).map(|_| d.sample(&mut rng)).collect();
    let mean = samples.iter().sum::<f64>() / N as f64;
    let mu_theory = (low + high) / 2.0;
    let sigma_theory = ((high - low).powi(2) / 12.0).sqrt();
    assert_moment_ok("Uniform", mean, mu_theory, sigma_theory, N);
    assert_ks_ok("Uniform", samples, |x| uniform_cdf(low, high, x));
}

#[test]
fn fg14_lognormal_ks_and_moments() {
    let mut rng = StdRng::seed_from_u64(1003);
    let (mu, sigma) = (0.2, 0.5);
    let d = LogNormal::new(mu, sigma).unwrap();
    let samples: Vec<f64> = (0..N).map(|_| d.sample(&mut rng)).collect();
    let mean = samples.iter().sum::<f64>() / N as f64;
    let mu_theory = (mu + sigma * sigma / 2.0).exp();
    let var_theory = ((sigma * sigma).exp() - 1.0) * (2.0 * mu + sigma * sigma).exp();
    assert_moment_ok("LogNormal", mean, mu_theory, var_theory.sqrt(), N);
    assert_ks_ok("LogNormal", samples, |x| lognormal_cdf(mu, sigma, x));
}

#[test]
fn fg14_exponential_ks_and_moments() {
    let mut rng = StdRng::seed_from_u64(1004);
    let rate = 2.0;
    let d = Exponential::new(rate).unwrap();
    let samples: Vec<f64> = (0..N).map(|_| d.sample(&mut rng)).collect();
    let mean = samples.iter().sum::<f64>() / N as f64;
    assert_moment_ok("Exponential", mean, 1.0 / rate, 1.0 / rate, N);
    assert_ks_ok("Exponential", samples, |x| exponential_cdf(rate, x));
}

#[test]
fn fg14_beta_ks_and_moments() {
    let mut rng = StdRng::seed_from_u64(1005);
    let (a, b) = (2.0, 5.0);
    let d = Beta::new(a, b).unwrap();
    let samples: Vec<f64> = (0..N).map(|_| d.sample(&mut rng)).collect();
    let mean = samples.iter().sum::<f64>() / N as f64;
    let mu_theory = a / (a + b);
    let var_theory = (a * b) / ((a + b).powi(2) * (a + b + 1.0));
    assert_moment_ok("Beta", mean, mu_theory, var_theory.sqrt(), N);
    assert_ks_ok("Beta", samples, |x| beta_cdf(a, b, x));
}

#[test]
fn fg14_gamma_ks_and_moments() {
    // FG-14: this is exactly the kind of case the missing coverage let
    // through — Gamma::new(shape, rate) uses a RATE parameterization
    // (mean = shape/rate). A scale-parameterization bug (mean = shape*rate)
    // would move the mean from 1.5 to 6.0 here and this test would catch it.
    let mut rng = StdRng::seed_from_u64(1006);
    let (shape, rate) = (3.0, 2.0);
    let d = Gamma::new(shape, rate).unwrap();
    let samples: Vec<f64> = (0..N).map(|_| d.sample(&mut rng)).collect();
    let mean = samples.iter().sum::<f64>() / N as f64;
    let mu_theory = shape / rate;
    let var_theory = shape / (rate * rate);
    assert_moment_ok("Gamma", mean, mu_theory, var_theory.sqrt(), N);
    assert_ks_ok("Gamma", samples, |x| gamma_cdf(shape, rate, x));
}

#[test]
fn fg14_studentt_ks_and_moments() {
    let mut rng = StdRng::seed_from_u64(1007);
    let (df, loc, scale) = (8.0, 1.0, 1.5);
    let d = StudentT::new(df, loc, scale).unwrap();
    let samples: Vec<f64> = (0..N).map(|_| d.sample(&mut rng)).collect();
    let mean = samples.iter().sum::<f64>() / N as f64;
    let var_theory = scale * scale * df / (df - 2.0); // finite since df=8 > 2
    assert_moment_ok("StudentT", mean, loc, var_theory.sqrt(), N);
    assert_ks_ok("StudentT", samples, |x| studentt_cdf(df, loc, scale, x));
}

#[test]
fn fg14_cauchy_ks_only() {
    // Cauchy has no finite mean/variance, so no moment check; KS alone.
    let mut rng = StdRng::seed_from_u64(1008);
    let (loc, scale) = (0.5, 1.2);
    let d = Cauchy::new(loc, scale).unwrap();
    let samples: Vec<f64> = (0..N).map(|_| d.sample(&mut rng)).collect();
    assert_ks_ok("Cauchy", samples, |x| cauchy_cdf(loc, scale, x));
}

#[test]
fn fg14_laplace_ks_and_moments() {
    let mut rng = StdRng::seed_from_u64(1009);
    let (loc, scale) = (-1.0, 2.0);
    let d = Laplace::new(loc, scale).unwrap();
    let samples: Vec<f64> = (0..N).map(|_| d.sample(&mut rng)).collect();
    let mean = samples.iter().sum::<f64>() / N as f64;
    let var_theory = 2.0 * scale * scale;
    assert_moment_ok("Laplace", mean, loc, var_theory.sqrt(), N);
    assert_ks_ok("Laplace", samples, |x| laplace_cdf(loc, scale, x));
}

#[test]
fn fg14_weibull_ks_and_moments() {
    let mut rng = StdRng::seed_from_u64(1010);
    let (shape, scale) = (1.5, 2.0);
    let d = Weibull::new(shape, scale).unwrap();
    let samples: Vec<f64> = (0..N).map(|_| d.sample(&mut rng)).collect();
    let mean = samples.iter().sum::<f64>() / N as f64;
    let g1 = (ln_gamma_fn(1.0 + 1.0 / shape)).exp();
    let g2 = (ln_gamma_fn(1.0 + 2.0 / shape)).exp();
    let mu_theory = scale * g1;
    let var_theory = scale * scale * (g2 - g1 * g1);
    assert_moment_ok("Weibull", mean, mu_theory, var_theory.sqrt(), N);
    assert_ks_ok("Weibull", samples, |x| weibull_cdf(shape, scale, x));
}

#[test]
fn fg14_chi_squared_ks_and_moments() {
    let mut rng = StdRng::seed_from_u64(1011);
    let k = 6.0;
    let d = ChiSquared::new(k).unwrap();
    let samples: Vec<f64> = (0..N).map(|_| d.sample(&mut rng)).collect();
    let mean = samples.iter().sum::<f64>() / N as f64;
    assert_moment_ok("ChiSquared", mean, k, (2.0 * k).sqrt(), N);
    assert_ks_ok("ChiSquared", samples, |x| chi2_cdf(k, x));
}

#[test]
fn fg14_inverse_gamma_ks_and_moments() {
    let mut rng = StdRng::seed_from_u64(1012);
    // shape > 2 so both the mean and variance are finite.
    let (shape, rate) = (4.0, 3.0);
    let d = InverseGamma::new(shape, rate).unwrap();
    let samples: Vec<f64> = (0..N).map(|_| d.sample(&mut rng)).collect();
    let mean = samples.iter().sum::<f64>() / N as f64;
    let mu_theory = rate / (shape - 1.0);
    let var_theory = (rate * rate) / ((shape - 1.0).powi(2) * (shape - 2.0));
    assert_moment_ok("InverseGamma", mean, mu_theory, var_theory.sqrt(), N);
    assert_ks_ok("InverseGamma", samples, |x| invgamma_cdf(shape, rate, x));
}

// ===========================================================================
// Discrete distributions: chi-square goodness-of-fit + moment checks.
// ===========================================================================

#[test]
fn fg14_bernoulli_chi_square_and_moments() {
    let mut rng = StdRng::seed_from_u64(2001);
    let p = 0.3;
    let d = Bernoulli::new(p).unwrap();
    let samples: Vec<bool> = (0..N).map(|_| d.sample(&mut rng)).collect();
    let successes = samples.iter().filter(|&&b| b).count() as u64;
    let observed = [N as u64 - successes, successes]; // [false, true]
    let expected_probs = [1.0 - p, p];
    let stat = chi_square_statistic(&observed, &expected_probs, N as u64);
    let crit = chi2_critical_0_001(1);
    assert!(
        stat < crit,
        "FG-14: Bernoulli failed chi-square GOF: stat={stat:.4} >= crit={crit:.4}"
    );
    let mean = successes as f64 / N as f64;
    assert_moment_ok("Bernoulli", mean, p, (p * (1.0 - p)).sqrt(), N);
}

#[test]
fn fg14_categorical_chi_square_and_moments() {
    let mut rng = StdRng::seed_from_u64(2002);
    let probs = vec![0.1, 0.2, 0.3, 0.4];
    let d = Categorical::new(probs.clone()).unwrap();
    let mut counts = [0u64; 4];
    let mut sum_idx = 0.0;
    for _ in 0..N {
        let i = d.sample(&mut rng);
        counts[i] += 1;
        sum_idx += i as f64;
    }
    let stat = chi_square_statistic(&counts, &probs, N as u64);
    let crit = chi2_critical_0_001(3);
    assert!(
        stat < crit,
        "FG-14: Categorical failed chi-square GOF: stat={stat:.4} >= crit={crit:.4}"
    );
    let mu_theory: f64 = probs.iter().enumerate().map(|(i, &p)| i as f64 * p).sum();
    let e_x2: f64 = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| (i as f64).powi(2) * p)
        .sum();
    let var_theory = e_x2 - mu_theory * mu_theory;
    let mean = sum_idx / N as f64;
    assert_moment_ok("Categorical", mean, mu_theory, var_theory.sqrt(), N);
}

#[test]
fn fg14_binomial_chi_square_and_moments() {
    let mut rng = StdRng::seed_from_u64(2003);
    let (n_trials, p) = (10u64, 0.4);
    let d = Binomial::new(n_trials, p).unwrap();
    let mut counts = [0u64; 11]; // k = 0..=10
    let mut sum_k = 0.0;
    for _ in 0..N {
        let k = d.sample(&mut rng);
        counts[k as usize] += 1;
        sum_k += k as f64;
    }
    // Expected PMF via the same closed form as distribution.rs's log_prob.
    let expected_probs: Vec<f64> = (0..=n_trials)
        .map(|k| {
            let log_binom = ln_gamma_fn(n_trials as f64 + 1.0)
                - ln_gamma_fn(k as f64 + 1.0)
                - ln_gamma_fn((n_trials - k) as f64 + 1.0);
            (log_binom + (k as f64) * p.ln() + ((n_trials - k) as f64) * (1.0 - p).ln()).exp()
        })
        .collect();
    let stat = chi_square_statistic(&counts, &expected_probs, N as u64);
    let crit = chi2_critical_0_001(10);
    assert!(
        stat < crit,
        "FG-14: Binomial failed chi-square GOF: stat={stat:.4} >= crit={crit:.4}"
    );
    let mean = sum_k / N as f64;
    let mu_theory = n_trials as f64 * p;
    let sigma_theory = (n_trials as f64 * p * (1.0 - p)).sqrt();
    assert_moment_ok("Binomial", mean, mu_theory, sigma_theory, N);
}

#[test]
fn fg14_poisson_chi_square_and_moments() {
    // FG-14: this is exactly the "variance instead of rate" scenario the
    // finding warns about — Poisson's mean AND variance both equal lambda,
    // so a mean-only check with a loose tolerance could miss a sampler that
    // draws from the wrong lambda but by coincidence matches the mean; the
    // chi-square test checks the whole shape of the distribution, not just
    // its first moment.
    let mut rng = StdRng::seed_from_u64(2004);
    let lambda = 4.0;
    let d = Poisson::new(lambda).unwrap();
    // Bin 0..=12 individually, and "13+" as an overflow bin (df = 13).
    const MAXK: usize = 12;
    let mut counts = [0u64; MAXK + 2];
    let mut sum_k = 0.0;
    for _ in 0..N {
        let k = d.sample(&mut rng);
        sum_k += k as f64;
        let bin = (k as usize).min(MAXK + 1);
        counts[bin] += 1;
    }
    let mut expected_probs = vec![0.0; MAXK + 2];
    let mut cum = 0.0;
    for (k, p) in expected_probs.iter_mut().enumerate().take(MAXK + 1) {
        let logp = (k as f64) * lambda.ln() - lambda - ln_gamma_fn(k as f64 + 1.0);
        *p = logp.exp();
        cum += *p;
    }
    expected_probs[MAXK + 1] = 1.0 - cum; // tail mass for k >= MAXK+1
    let stat = chi_square_statistic(&counts, &expected_probs, N as u64);
    let crit = chi2_critical_0_001(13);
    assert!(
        stat < crit,
        "FG-14: Poisson failed chi-square GOF: stat={stat:.4} >= crit={crit:.4}"
    );
    let mean = sum_k / N as f64;
    assert_moment_ok("Poisson", mean, lambda, lambda.sqrt(), N);
}

#[test]
fn fg14_discrete_uniform_chi_square_and_moments() {
    let mut rng = StdRng::seed_from_u64(2005);
    let (low, high) = (1i64, 6i64); // fair die
    let d = DiscreteUniform::new(low, high).unwrap();
    let n_bins = (high - low + 1) as usize;
    let mut counts = vec![0u64; n_bins];
    let mut sum_k = 0.0;
    for _ in 0..N {
        let k = d.sample(&mut rng);
        counts[(k - low) as usize] += 1;
        sum_k += k as f64;
    }
    let expected_probs = vec![1.0 / n_bins as f64; n_bins];
    let stat = chi_square_statistic(&counts, &expected_probs, N as u64);
    let crit = chi2_critical_0_001(5);
    assert!(
        stat < crit,
        "FG-14: DiscreteUniform failed chi-square GOF: stat={stat:.4} >= crit={crit:.4}"
    );
    let mean = sum_k / N as f64;
    let mu_theory = (low + high) as f64 / 2.0;
    let span = (high - low + 1) as f64;
    let var_theory = (span * span - 1.0) / 12.0;
    assert_moment_ok("DiscreteUniform", mean, mu_theory, var_theory.sqrt(), N);
}
