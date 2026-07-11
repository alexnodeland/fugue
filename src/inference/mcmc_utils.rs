//! Utilities for robust MCMC implementation.
//!
//! This module provides helper functions and improved algorithms for
//! Metropolis-Hastings and related MCMC methods with proper theoretical
//! guarantees and numerical stability.
//!
//! ## Effective sample size (FG-01 / FG-37)
//!
//! [`effective_sample_size_mcmc`] and [`effective_sample_size_multichain`] are
//! the crate's single canonical ESS estimators. They compute the integrated
//! autocorrelation time from the *normalized* autocorrelations
//! `rho_k = gamma_k / gamma_0` (a dimensionless quantity), so the resulting ESS
//! is invariant to rescaling the input series — as an effective sample size must
//! be. The multi-chain estimator follows Vehtari, Gelman, Simpson, Carpenter &
//! Bürkner (2021): the autocorrelations are combined across chains through the
//! pooled `W + B` variance normalization used by R-hat, and the sum is truncated
//! with Geyer's initial positive/monotone sequence. `diagnostics::effective_sample_size`
//! is a thin wrapper over [`effective_sample_size_mcmc`] so every ESS path in the
//! crate routes through this normalized estimator.

use crate::core::address::Address;
use std::collections::HashMap;

/// Diminishing adaptation schedule that preserves ergodicity.
///
/// Implements the adaptation schedule recommended by Roberts & Rosenthal (2007)
/// that ensures the adapted chain remains ergodic and converges to the correct
/// stationary distribution.
#[derive(Debug, Clone)]
pub struct DiminishingAdaptation {
    /// Current proposal scales and their cached logarithms for each site
    /// Stored as (scale, log_scale) to avoid expensive ln() computations
    pub scales: HashMap<Address, (f64, f64)>,
    /// Acceptance counts for each site
    pub accept_counts: HashMap<Address, usize>,
    /// Total proposal counts for each site
    pub total_counts: HashMap<Address, usize>,
    /// Target acceptance rate
    pub target_rate: f64,
    /// Adaptation strength parameter (should be in (0.5, 1])
    pub gamma: f64,
}

impl DiminishingAdaptation {
    /// Create a new diminishing adaptation scheduler.
    ///
    /// # Arguments
    ///
    /// * `target_rate` - Target acceptance rate (0.234 for optimal scaling, 0.44 for random walk)
    /// * `gamma` - Adaptation rate parameter (0.7 is a good default)
    pub fn new(target_rate: f64, gamma: f64) -> Self {
        assert!(target_rate > 0.0 && target_rate < 1.0);
        assert!(gamma > 0.5 && gamma <= 1.0); // Required for ergodicity

        Self {
            scales: HashMap::new(),
            accept_counts: HashMap::new(),
            total_counts: HashMap::new(),
            target_rate,
            gamma,
        }
    }

    /// Get current scale for a site, initializing if necessary.
    ///
    /// FG-38: on a cache hit this reads through a shared borrow and clones
    /// nothing; the `Address` key is only cloned on the (one-time) miss that
    /// first inserts the site. The old `entry(addr.clone())` form allocated a
    /// fresh `String` on *every* call regardless of hit/miss.
    pub fn get_scale(&mut self, addr: &Address) -> f64 {
        if let Some(&(scale, _)) = self.scales.get(addr) {
            scale
        } else {
            self.scales.insert(addr.clone(), (1.0, 0.0));
            1.0
        }
    }

    /// Update adaptation based on acceptance outcome.
    ///
    /// Uses diminishing step sizes that ensure the adaptation eventually stops,
    /// preserving the ergodic properties of the chain.
    ///
    /// FG-38: every map is touched with `get_mut`-then-`insert`, so the
    /// `Address` key is cloned only when a site is seen for the first time. On
    /// the steady-state hot path (every site already present) this method
    /// performs zero string allocations.
    pub fn update(&mut self, addr: &Address, accepted: bool) {
        // Update total counter (clone the key only on first insertion).
        let total_count = match self.total_counts.get_mut(addr) {
            Some(t) => {
                *t += 1;
                *t
            }
            None => {
                self.total_counts.insert(addr.clone(), 1);
                1
            }
        };

        if accepted {
            match self.accept_counts.get_mut(addr) {
                Some(a) => *a += 1,
                None => {
                    self.accept_counts.insert(addr.clone(), 1);
                }
            }
        }

        let accept_count = *self.accept_counts.get(addr).unwrap_or(&0);

        if total_count < 10 {
            return; // Need some samples before adapting
        }

        let accept_rate = accept_count as f64 / total_count as f64;

        // Diminishing step size: α_n = 1/n^γ
        let step_size = 1.0 / (total_count as f64).powf(self.gamma);

        // Update scale using stochastic approximation with cached log scale.
        let entry = match self.scales.get_mut(addr) {
            Some(e) => e,
            None => {
                self.scales.insert(addr.clone(), (1.0, 0.0));
                self.scales
                    .get_mut(addr)
                    .expect("just inserted the scale entry")
            }
        };
        let (ref mut scale, ref mut log_scale) = *entry;

        // Update: log(scale_{n+1}) = log(scale_n) + α_n * (accept_rate - target_rate)
        *log_scale += step_size * (accept_rate - self.target_rate);

        // Keep scale in reasonable bounds and ensure positivity
        let new_scale = log_scale.exp();
        *scale = if new_scale.is_finite() && new_scale > 0.0 {
            new_scale.clamp(0.001, 100.0)
        } else {
            1.0 // Reset to default if numerical issues
        };

        // Update cached log scale to match the clamped scale
        if *scale == 1.0 {
            *log_scale = 0.0; // ln(1.0) = 0.0
        } else {
            *log_scale = scale.ln(); // Recompute only when we had to clamp
        }
    }

    /// Check if adaptation should continue.
    ///
    /// Returns true if any site has had fewer than a minimum number of updates.
    pub fn should_continue_adaptation(&self, min_updates: usize) -> bool {
        self.total_counts.values().any(|&count| count < min_updates)
    }

    /// Get adaptation statistics for diagnostics.
    pub fn get_stats(&self) -> Vec<(Address, f64, f64, usize)> {
        self.scales
            .iter()
            .map(|(addr, &(scale, _log_scale))| {
                let accepts = *self.accept_counts.get(addr).unwrap_or(&0);
                let total = *self.total_counts.get(addr).unwrap_or(&0);
                let rate = if total > 0 {
                    accepts as f64 / total as f64
                } else {
                    0.0
                };
                (addr.clone(), scale, rate, total)
            })
            .collect()
    }
}

/// Effective sample size for a single MCMC chain.
///
/// Computes `ESS = m·n / tau_hat` where `tau_hat` is the integrated
/// autocorrelation time estimated from the *normalized* autocorrelations. This
/// is the single-chain special case of [`effective_sample_size_multichain`].
///
/// Because the estimator normalizes by the lag-0 autocovariance (the variance),
/// the result is invariant to rescaling the input by a constant — the property
/// that the pre-FG-01 `diagnostics::effective_sample_size` violated by summing
/// raw autocovariances.
///
/// # Arguments
///
/// * `samples` - Vector of scalar samples from an MCMC chain
///
/// # Returns
///
/// Effective sample size (between 1 and `samples.len()`)
pub fn effective_sample_size_mcmc(samples: &[f64]) -> f64 {
    let n = samples.len();
    if n < 4 {
        return n as f64; // Can't estimate autocorrelation with too few samples
    }
    ess_from_chains(&[samples])
}

/// Multi-chain effective sample size (Vehtari et al. 2021).
///
/// Combines the per-chain autocorrelations through the pooled `W + B` variance
/// normalization (`var_plus = (n-1)/n · W + B/n`, the same quantity used by
/// R-hat) and truncates the autocorrelation sum with Geyer's initial
/// positive/monotone sequence. All chains contribute, so the reported ESS is
/// consistent with the pooled mean/quantiles rather than reflecting a single
/// chain (FG-37).
///
/// Chains of unequal length (or fewer than 4 draws) fall back to the total draw
/// count, matching the small-sample behavior of the single-chain estimator.
pub fn effective_sample_size_multichain(chains: &[Vec<f64>]) -> f64 {
    if chains.is_empty() {
        return 0.0;
    }
    let refs: Vec<&[f64]> = chains.iter().map(|c| c.as_slice()).collect();
    let n = refs[0].len();
    if n < 4 || refs.iter().any(|c| c.len() != n) {
        return chains.iter().map(|c| c.len()).sum::<usize>().max(1) as f64;
    }
    ess_from_chains(&refs)
}

/// Autocovariances (biased, denominator `n`) for lags `0..=max_lag`.
///
/// Uses the denominator-`n` (biased) estimator recommended by Geyer/Stan for
/// autocorrelation-time estimation: it damps the noisy high-lag terms and keeps
/// the resulting spectral sum well-behaved.
fn autocovariances(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = x.iter().map(|&v| v - mean).collect();
    let mut acov = Vec::with_capacity(max_lag + 1);
    for lag in 0..=max_lag {
        let mut s = 0.0;
        for i in 0..(n - lag) {
            s += centered[i] * centered[i + lag];
        }
        acov.push(s / n as f64);
    }
    acov
}

/// Core ESS estimator shared by the single- and multi-chain entry points.
///
/// Implements the Vehtari et al. (2021) / Stan multi-chain effective sample
/// size: per-chain autocovariances are pooled, normalized by the between+within
/// variance `var_plus`, and summed via Geyer's initial positive sequence made
/// monotone. Returns `m·n / tau_hat` with `tau_hat >= 1` (so ESS never exceeds
/// the total number of draws).
fn ess_from_chains(chains: &[&[f64]]) -> f64 {
    let m = chains.len();
    if m == 0 {
        return 0.0;
    }
    let n = chains[0].len();
    if n < 4 || chains.iter().any(|c| c.len() != n) {
        return chains.iter().map(|c| c.len()).sum::<usize>().max(1) as f64;
    }

    // All lags Geyer might need; capped so a single very long chain stays O(n·cap)
    // rather than O(n^2). The initial-positive-sequence truncation almost always
    // stops far earlier than this cap for any usefully-mixing chain.
    let max_lag = (n - 1).min(2048);
    let acovs: Vec<Vec<f64>> = chains.iter().map(|c| autocovariances(c, max_lag)).collect();

    let nf = n as f64;
    let mf = m as f64;
    let chain_means: Vec<f64> = chains.iter().map(|c| c.iter().sum::<f64>() / nf).collect();
    // Unbiased within-chain variance: acov0 * n/(n-1).
    let chain_vars: Vec<f64> = acovs.iter().map(|a| a[0] * nf / (nf - 1.0)).collect();
    let mean_var = chain_vars.iter().sum::<f64>() / mf; // W

    if mean_var <= 0.0 {
        // Every chain is constant: treat every draw as independent.
        return (m * n) as f64;
    }

    // var_plus = (n-1)/n · W + B/n  (identical to the R-hat pooled variance).
    let mut var_plus = mean_var * (nf - 1.0) / nf;
    if m > 1 {
        let overall = chain_means.iter().sum::<f64>() / mf;
        let between = chain_means
            .iter()
            .map(|&mu| (mu - overall).powi(2))
            .sum::<f64>()
            / (mf - 1.0);
        var_plus += between;
    }

    // Combined normalized autocorrelation at lag t (Vehtari 2021):
    // rho_t = 1 - (W - mean_over_chains(acov_t)) / var_plus.
    let rho = |t: usize| -> f64 {
        let acov_t = acovs.iter().map(|a| a[t]).sum::<f64>() / mf;
        1.0 - (mean_var - acov_t) / var_plus
    };

    let mut rho_hat = vec![0.0f64; max_lag + 1];
    rho_hat[0] = 1.0;
    if max_lag >= 1 {
        rho_hat[1] = rho(1);
    }

    // Geyer initial positive sequence: sum autocorrelations in pairs and stop as
    // soon as a pair sum turns negative.
    let mut t = 1usize;
    let mut max_t = 1usize.min(max_lag);
    while t + 2 <= max_lag {
        let rho_even = rho(t + 1);
        let rho_odd = rho(t + 2);
        if rho_even + rho_odd < 0.0 {
            break;
        }
        rho_hat[t + 1] = rho_even;
        rho_hat[t + 2] = rho_odd;
        max_t = t + 2;
        t += 2;
    }

    // Make the sequence of pair sums monotone non-increasing (reduces variance).
    let mut k = 1usize;
    while k + 2 <= max_t {
        let prev = rho_hat[k - 1] + rho_hat[k];
        let cur = rho_hat[k + 1] + rho_hat[k + 2];
        if cur > prev {
            let avg = prev / 2.0;
            rho_hat[k + 1] = avg;
            rho_hat[k + 2] = avg;
        }
        k += 2;
    }

    // tau = 1 + 2·sum_{k>=1} rho_k = -1 + 2·sum_{k>=0} rho_hat_k.
    let sum_rho: f64 = rho_hat[0..=max_t].iter().sum();
    let tau = (-1.0 + 2.0 * sum_rho).max(1.0);
    (m * n) as f64 / tau
}

/// Geweke convergence diagnostic for a single chain.
///
/// Compares the mean of the first 10% and last 50% of the chain. Under
/// stationarity the returned z-score is asymptotically standard normal;
/// `|z| > 2` suggests non-convergence.
///
/// FG-39: the standard error uses each segment's spectral density at frequency
/// zero — `var(mean) = s^2 · tau / n` with `tau` the integrated autocorrelation
/// time — rather than the iid formula `s^2 / n`. Using the raw sample variance
/// (which assumes independent draws) understates the SE of an autocorrelated
/// segment by a factor of `sqrt(tau)` and inflates `|z|` by the same factor,
/// producing spurious "non-convergence" flags for perfectly stationary but
/// correlated chains.
pub fn geweke_diagnostic(chain: &[f64]) -> f64 {
    let n = chain.len();
    if n < 20 {
        return f64::NAN; // Too few samples
    }

    let first_end = n / 10;
    let last_start = n / 2;

    let first_part = &chain[0..first_end];
    let last_part = &chain[last_start..];

    if first_part.len() < 2 || last_part.len() < 2 {
        return f64::NAN;
    }

    let mean1 = first_part.iter().sum::<f64>() / first_part.len() as f64;
    let mean2 = last_part.iter().sum::<f64>() / last_part.len() as f64;

    // Autocorrelation-consistent variance of each segment mean.
    let varmean1 = spectral_variance_of_mean(first_part);
    let varmean2 = spectral_variance_of_mean(last_part);

    let se = (varmean1 + varmean2).sqrt();

    if se == 0.0 {
        return 0.0; // Constant chain
    }

    (mean1 - mean2) / se
}

/// Variance of the mean of an autocorrelated segment, `s^2 · tau / n`.
///
/// `tau = 1 + 2·sum_k rho_k` is the integrated autocorrelation time estimated
/// from the same normalized-autocovariance machinery used for ESS (the initial
/// positive sequence: sum `rho_k` until it turns non-positive). This is the
/// spectral density at zero divided by `n`, i.e. the correct asymptotic variance
/// of a correlated sample mean.
fn spectral_variance_of_mean(seg: &[f64]) -> f64 {
    let n = seg.len();
    if n < 2 {
        return 0.0;
    }
    let mean = seg.iter().sum::<f64>() / n as f64;
    let s2 = seg.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    if s2 == 0.0 {
        return 0.0;
    }

    let max_lag = (n - 1).min(1024);
    let acov = autocovariances(seg, max_lag);
    let var0 = acov[0];
    if var0 <= 0.0 {
        return 0.0;
    }

    let mut tau = 1.0;
    for &cov in acov.iter().skip(1) {
        let rho_k = cov / var0;
        if rho_k <= 0.0 {
            break;
        }
        tau += 2.0 * rho_k;
    }

    s2 * tau / n as f64
}

#[cfg(test)]
mod mcmc_tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_diminishing_adaptation() {
        let mut adapter = DiminishingAdaptation::new(0.44, 0.7);
        let addr = Address("test".to_string());

        // Initial scale should be 1.0
        assert_eq!(adapter.get_scale(&addr), 1.0);

        // After many acceptances, scale should increase gradually
        for _ in 0..500 {
            adapter.update(&addr, true);
        }
        assert!(adapter.get_scale(&addr) > 1.0);

        // After many rejections, scale should decrease gradually
        for _ in 0..500 {
            adapter.update(&addr, false);
        }
        // Due to diminishing adaptation, scale changes become very small
        // Just check that the algorithm doesn't crash and produces reasonable values
        let final_scale = adapter.get_scale(&addr);
        assert!(final_scale > 0.0 && final_scale.is_finite()); // Sanity bounds
    }

    #[test]
    fn test_effective_sample_size() {
        // Random chain (low correlation)
        let random: Vec<f64> = (0..100)
            .map(|i| (i as f64).sin() * (i as f64).cos())
            .collect();
        let ess = effective_sample_size_mcmc(&random);
        assert!(ess > 1.0 && ess <= 100.0); // Basic sanity check

        // Highly correlated chain
        let correlated: Vec<f64> = (0..100).map(|i| (i / 10) as f64).collect();
        let ess_corr = effective_sample_size_mcmc(&correlated);
        assert!(ess_corr > 0.0 && ess_corr <= 100.0); // Basic bounds check
    }

    // FG-01: ESS must be invariant to rescaling the input series. The pre-fix
    // diagnostics estimator summed raw autocovariances (never dividing by the
    // variance), so scaling the series by c scaled tau — and hence ESS — with
    // c^2. The normalized estimator here divides by gamma_0, so ESS is unchanged.
    #[test]
    fn ess_is_scale_invariant() {
        let mut rng = StdRng::seed_from_u64(20260710);
        // AR(1) with phi = 0.6.
        let phi = 0.6;
        let n = 3000;
        let mut x = 0.0;
        let mut series = Vec::with_capacity(n);
        for _ in 0..n {
            let z: f64 = {
                // Box-Muller standard normal
                let u1: f64 = rng.gen::<f64>().max(1e-12);
                let u2: f64 = rng.gen();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            };
            x = phi * x + z;
            series.push(x);
        }
        let ess_base = effective_sample_size_mcmc(&series);
        let scaled: Vec<f64> = series.iter().map(|&v| v * 1000.0).collect();
        let ess_scaled = effective_sample_size_mcmc(&scaled);
        // Identical up to floating-point: the two runs do the same arithmetic on
        // proportional inputs, so agreement is tight.
        let rel = (ess_base - ess_scaled).abs() / ess_base;
        assert!(
            rel < 1e-9,
            "ESS not scale-invariant: base={ess_base}, scaled={ess_scaled}"
        );
    }

    // FG-01 / FG-35 known answer: an AR(1) chain with autocorrelation phi has
    // ESS/n -> (1 - phi)/(1 + phi). For phi = 0.9 that limit is 0.1/1.9 ≈ 0.0526.
    #[test]
    fn ess_matches_ar1_known_answer() {
        let mut rng = StdRng::seed_from_u64(424242);
        let phi = 0.9_f64;
        let n = 8000;
        let mut x = 0.0;
        let mut series = Vec::with_capacity(n);
        for _ in 0..n {
            let u1: f64 = rng.gen::<f64>().max(1e-12);
            let u2: f64 = rng.gen();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            x = phi * x + z;
            series.push(x);
        }
        let ess = effective_sample_size_mcmc(&series);
        let ratio = ess / n as f64;
        let expected = (1.0 - phi) / (1.0 + phi); // 0.05263...
                                                  // Tolerance 15% of the target per the audit design decision; the Geyer
                                                  // estimator on 8000 draws is comfortably inside this band.
        let rel = (ratio - expected).abs() / expected;
        assert!(
            rel < 0.15,
            "AR(1) ESS/n = {ratio:.4}, expected ≈ {expected:.4} (rel err {rel:.3})"
        );
    }

    // FG-39: on a stationary but autocorrelated chain the Geweke z-score must
    // stay small. The old raw-variance SE inflated |z| by sqrt(tau); with tau≈19
    // for phi=0.9 that is a ~4.4x inflation that would routinely exceed the
    // |z|>2 flag on a perfectly stationary chain.
    #[test]
    fn geweke_stationary_is_small() {
        let mut rng = StdRng::seed_from_u64(9001);
        let phi = 0.9_f64;
        let n = 6000;
        let mut x = 0.0;
        let mut series = Vec::with_capacity(n);
        for _ in 0..n {
            let u1: f64 = rng.gen::<f64>().max(1e-12);
            let u2: f64 = rng.gen();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            x = phi * x + z;
            series.push(x);
        }
        let z = geweke_diagnostic(&series);
        assert!(
            z.abs() < 3.0,
            "stationary Geweke |z| = {z:.3} should be < 3"
        );
    }

    // FG-39: a drifting (non-stationary) chain must be flagged.
    #[test]
    fn geweke_drift_is_flagged() {
        let mut rng = StdRng::seed_from_u64(9002);
        let n = 6000;
        let mut series = Vec::with_capacity(n);
        for i in 0..n {
            let u1: f64 = rng.gen::<f64>().max(1e-12);
            let u2: f64 = rng.gen();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            // Linear drift dominates the noise.
            series.push(i as f64 * 0.01 + 0.5 * z);
        }
        let z = geweke_diagnostic(&series);
        assert!(z.abs() > 4.0, "drifting Geweke |z| = {z:.3} should be > 4");
    }
}
