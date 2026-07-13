//! FG-31: HMC posterior-correctness tests (seeded).
//!
//! (a) 2-D correlated Gaussian: sample mean within 3·SE, sample covariance
//!     within 15% of the truth.
//! (b) conjugate Normal-Normal: posterior mean/variance vs the analytic values.
//! (c) bounded-support site: HMC stays inside the support and recovers the mean.

use fugue::inference::hmc::{hmc_chain, HMCConfig};
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64], m: f64) -> f64 {
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64
}

fn covariance(xs: &[f64], ys: &[f64], mx: f64, my: f64) -> f64 {
    xs.iter()
        .zip(ys)
        .map(|(x, y)| (x - mx) * (y - my))
        .sum::<f64>()
        / xs.len() as f64
}

// -------------------------------------------------------------------------
// (a) 2-D correlated Gaussian posterior.
// -------------------------------------------------------------------------
#[test]
fn fg31_hmc_correlated_gaussian_mean_and_covariance() {
    const RHO: f64 = 0.8;
    let cond_sd = (1.0 - RHO * RHO).sqrt();
    // Joint N(0, [[1, RHO],[RHO, 1]]).
    let model_fn = || {
        sample(addr!("x"), Normal::new(0.0, 1.0).unwrap()).bind(move |x| {
            sample(addr!("y"), Normal::new(RHO * x, cond_sd).unwrap()).map(move |y| (x, y))
        })
    };

    let mut rng = StdRng::seed_from_u64(2026_0711);
    let samples = hmc_chain(&mut rng, model_fn, 4000, 1000, HMCConfig::default());

    let xs: Vec<f64> = samples.iter().map(|((x, _), _)| *x).collect();
    let ys: Vec<f64> = samples.iter().map(|((_, y), _)| *y).collect();

    let mx = mean(&xs);
    let my = mean(&ys);
    let vx = variance(&xs, mx);
    let vy = variance(&ys, my);
    let cxy = covariance(&xs, &ys, mx, my);

    // Sample mean within 3 standard errors, where SE = sample_sd / sqrt(ESS)
    // (accounts for the chain's autocorrelation).
    let ess_x = effective_sample_size_mcmc(&xs);
    let ess_y = effective_sample_size_mcmc(&ys);
    let se_x = vx.sqrt() / ess_x.sqrt();
    let se_y = vy.sqrt() / ess_y.sqrt();
    assert!(
        mx.abs() < 3.0 * se_x,
        "x mean {mx} exceeds 3·SE ({:.4}); ess={ess_x:.1}",
        3.0 * se_x
    );
    assert!(
        my.abs() < 3.0 * se_y,
        "y mean {my} exceeds 3·SE ({:.4}); ess={ess_y:.1}",
        3.0 * se_y
    );

    // Sample covariance entries within 15% of the true values (variances 1.0,
    // off-diagonal RHO).
    assert!((vx - 1.0).abs() < 0.15, "var(x)={vx} not within 15% of 1.0");
    assert!((vy - 1.0).abs() < 0.15, "var(y)={vy} not within 15% of 1.0");
    assert!(
        (cxy - RHO).abs() < 0.15 * RHO,
        "cov(x,y)={cxy} not within 15% of {RHO}"
    );
}

// -------------------------------------------------------------------------
// (b) Conjugate Normal-Normal: posterior mean/variance vs analytic.
// -------------------------------------------------------------------------
#[test]
fn fg31_hmc_conjugate_normal_normal_matches_analytic() {
    // Prior mu ~ N(mu0, sigma0), likelihood y_i ~ N(mu, sigma).
    const MU0: f64 = 0.0;
    const SIGMA0: f64 = 2.0;
    const SIGMA: f64 = 1.0;
    let data = [1.0_f64, 2.0, 3.0, 1.5, 2.5];
    let n = data.len() as f64;
    let sum_y: f64 = data.iter().sum();

    // Analytic posterior for a Normal mean with known variance.
    let post_var = 1.0 / (1.0 / (SIGMA0 * SIGMA0) + n / (SIGMA * SIGMA));
    let post_mean = post_var * (MU0 / (SIGMA0 * SIGMA0) + sum_y / (SIGMA * SIGMA));
    // post_var = 1/(1/4 + 5) = 0.19047619..., post_mean = 10/5.25 = 1.90476190...

    let model_fn = move || {
        sample(addr!("mu"), Normal::new(MU0, SIGMA0).unwrap()).bind(move |mu| {
            let obs = (0..data.len()).fold(pure(()), move |acc, i| {
                let yi = data[i];
                acc.bind(move |_| observe(addr!("y", i), Normal::new(mu, SIGMA).unwrap(), yi))
            });
            obs.map(move |_| mu)
        })
    };

    let mut rng = StdRng::seed_from_u64(4242);
    let samples = hmc_chain(&mut rng, model_fn, 4000, 1000, HMCConfig::default());
    let mus: Vec<f64> = samples.iter().map(|(mu, _)| *mu).collect();

    let m = mean(&mus);
    let v = variance(&mus, m);
    let ess = effective_sample_size_mcmc(&mus);
    let se = v.sqrt() / ess.sqrt();

    // Posterior mean within 3·SE of the analytic value.
    assert!(
        (m - post_mean).abs() < 3.0 * se,
        "posterior mean {m} vs analytic {post_mean} (3·SE = {:.4}, ess={ess:.1})",
        3.0 * se
    );
    // Posterior variance within 12% of the analytic value.
    assert!(
        (v - post_var).abs() < 0.12 * post_var,
        "posterior var {v} vs analytic {post_var}"
    );
}

// -------------------------------------------------------------------------
// (c) Bounded-support site: HMC stays inside the support (proposals that leave
//     it are rejected) and still recovers the target mean. Documents the
//     efficiency caveat that bounded sites can reject near a hard boundary.
// -------------------------------------------------------------------------
#[test]
fn fg31_hmc_bounded_support_stays_in_support() {
    // Gamma(3, 1): support (0, ∞), mean 3, mode 2 — the bulk sits away from the
    // boundary and the 1/x force repels the trajectory from 0.
    let model_fn = || sample(addr!("g"), Gamma::new(3.0, 1.0).unwrap());
    let mut rng = StdRng::seed_from_u64(99);
    let samples = hmc_chain(&mut rng, model_fn, 3000, 1000, HMCConfig::default());
    let gs: Vec<f64> = samples.iter().map(|(g, _)| *g).collect();

    // Every accepted sample is strictly inside the support.
    assert!(
        gs.iter().all(|&g| g > 0.0 && g.is_finite()),
        "HMC produced a sample outside the Gamma support"
    );
    // Recovers the mean (3) within a loose Monte-Carlo tolerance.
    let m = mean(&gs);
    assert!(
        (m - 3.0).abs() < 0.3,
        "Gamma(3,1) mean estimate {m} off from 3"
    );
}
