//! Shared test utilities for the Fugue test suite.
//!
//! This module provides common helper functions, fixtures, and utilities
//! that can be reused across different test files to reduce duplication
//! and ensure consistency.

use fugue::*;
use rand::{rngs::StdRng, SeedableRng};

/// Standard RNG seed for deterministic tests
pub const TEST_SEED: u64 = 42;

/// Create a deterministic RNG for testing
pub fn test_rng() -> StdRng {
    StdRng::seed_from_u64(TEST_SEED)
}

/// Create a seeded RNG with a specific seed
pub fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// Test tolerance for floating point comparisons
pub const TEST_TOLERANCE: f64 = 1e-9;

/// Test tolerance for probabilistic tests (less strict)
pub const PROB_TOLERANCE: f64 = 1e-6;

/// Assert that two f64 values are approximately equal within tolerance
pub fn assert_approx_eq(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() < tolerance,
        "Values not approximately equal: {} vs {} (tolerance: {})",
        a,
        b,
        tolerance
    );
}

/// Assert that a value is finite (not NaN or infinite)
pub fn assert_finite(value: f64) {
    assert!(
        value.is_finite(),
        "Value should be finite but got: {}",
        value
    );
}

/// Common test models for reuse across test files
pub mod models {
    use super::*;

    /// Simple Gaussian mean model: mu ~ N(0, 1); y ~ N(mu, 1)
    pub fn gaussian_mean(obs: f64) -> Model<f64> {
        sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap()).bind(move |mu| {
            observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), obs).bind(move |_| pure(mu))
        })
    }

    /// Gaussian mean with prior parameters
    pub fn gaussian_mean_with_prior(obs: f64, prior_mu: f64, prior_sigma: f64) -> Model<f64> {
        sample(addr!("mu"), Normal::new(prior_mu, prior_sigma).unwrap()).bind(move |mu| {
            observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), obs).bind(move |_| pure(mu))
        })
    }

    /// Simple coin flip model
    pub fn coin_flip(p: f64) -> Model<bool> {
        sample(addr!("coin"), Bernoulli::new(p).unwrap())
    }

    /// Mixed type model for testing type safety
    pub fn mixed_model() -> Model<(f64, bool)> {
        sample(addr!("continuous"), Normal::new(0.0, 1.0).unwrap()).bind(|x| {
            sample(addr!("discrete"), Bernoulli::new(0.5).unwrap()).bind(move |b| pure((x, b)))
        })
    }
}

/// Test fixtures and data generators
pub mod fixtures {
    /// Generate synthetic regression data
    pub fn linear_regression_data(
        n: usize,
        slope: f64,
        intercept: f64,
        noise: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>) {
        use rand::Rng;
        let mut rng = super::seeded_rng(seed);

        let x_data: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y_data: Vec<f64> = x_data
            .iter()
            .map(|&x| slope * x + intercept + rng.gen::<f64>() * noise - noise / 2.0)
            .collect();

        (x_data, y_data)
    }
}

/// Statistical test helpers
pub mod stats {
    use super::*;

    /// Calculate sample mean
    pub fn mean(values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len() as f64
    }

    /// Calculate sample variance (with Bessel's correction)
    pub fn variance(values: &[f64]) -> f64 {
        let m = mean(values);
        let sum_sq_dev: f64 = values.iter().map(|x| (x - m).powi(2)).sum();
        sum_sq_dev / (values.len() - 1) as f64
    }

    /// Calculate standard deviation
    pub fn std_dev(values: &[f64]) -> f64 {
        variance(values).sqrt()
    }
}

/// Trace analysis utilities
pub mod trace_utils {
    use super::*;

    /// Extract all f64 values from a collection of traces
    pub fn extract_f64_samples(traces: &[(f64, Trace)], addr: &Address) -> Vec<f64> {
        traces
            .iter()
            .filter_map(|(_, trace)| trace.get_f64(addr))
            .collect()
    }

    /// Validate that all traces contain expected addresses
    pub fn validate_trace_addresses(traces: &[Trace], expected_addrs: &[Address]) -> bool {
        traces.iter().all(|trace| {
            expected_addrs
                .iter()
                .all(|addr| trace.choices.contains_key(addr))
        })
    }
}

/// Performance testing utilities
pub mod perf {
    use std::time::{Duration, Instant};

    /// Time a function execution
    pub fn time_fn<F, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }
}
