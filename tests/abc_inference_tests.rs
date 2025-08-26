//! Tests for ABC (Approximate Bayesian Computation) inference methods.
//!
//! This module tests the ABC functionality which currently has 0% coverage.

mod test_utils;

use fugue::inference::abc::*;
use fugue::*;
use test_utils::*;

#[test]
fn test_abc_scalar_summary_basic() {
    let mut rng = test_rng();

    // Simple model for ABC
    let model = || models::gaussian_mean(2.0);

    // Extract mu from trace as summary statistic
    let summary_fn = |trace: &Trace| -> f64 { trace.get_f64(&addr!("mu")).unwrap_or(0.0) };

    let samples = abc_scalar_summary(
        &mut rng, model, summary_fn, 2.0, // observed summary (mu should be around 2.0)
        0.5, // tolerance
        100, // max samples
    );

    assert!(!samples.is_empty());

    // Extract mu values and check they're within tolerance
    for trace in &samples {
        let mu_sample = trace.get_f64(&addr!("mu")).unwrap();
        assert!((mu_sample - 2.0).abs() <= 0.5);
        assert_finite(mu_sample);
    }
}

#[test]
fn test_abc_rejection_with_distance_function() {
    let mut rng = seeded_rng(123);

    let model_fn = || models::gaussian_mean(1.5);

    // Simulator extracts mu value as the "data"
    let simulator = |trace: &Trace| -> f64 { trace.get_f64(&addr!("mu")).unwrap_or(0.0) };

    let observed_data = 1.5;
    let distance_fn = ScalarDistance;
    let tolerance = 0.3;
    let max_samples = 50;

    let samples = abc_rejection(
        &mut rng,
        &model_fn,
        &simulator,
        &observed_data,
        &distance_fn,
        tolerance,
        max_samples,
    );

    assert!(!samples.is_empty());
    assert!(samples.len() <= max_samples);

    // All samples should meet distance criterion
    for trace in &samples {
        let simulated = simulator(trace);
        let dist = distance_fn.distance(&observed_data, &simulated);
        assert!(dist <= tolerance);
    }
}

#[test]
fn test_abc_with_multivariate_data() {
    let mut rng = seeded_rng(456);

    // Model that generates a 2D observation
    let model_fn = || {
        sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
            .bind(|x| sample(addr!("y"), Normal::new(x, 0.5).unwrap()).bind(move |y| pure((x, y))))
    };

    // Simulator returns both x and y as a vector
    let simulator = |trace: &Trace| -> Vec<f64> {
        let x = trace.get_f64(&addr!("x")).unwrap_or(0.0);
        let y = trace.get_f64(&addr!("y")).unwrap_or(0.0);
        vec![x, y]
    };

    let observed_data = vec![1.0, 1.2]; // x≈1, y≈1.2
    let distance_fn = EuclideanDistance;
    let tolerance = 0.8;
    let max_samples = 30;

    let samples = abc_rejection(
        &mut rng,
        &model_fn,
        &simulator,
        &observed_data,
        &distance_fn,
        tolerance,
        max_samples,
    );

    assert!(!samples.is_empty());

    // Verify all samples meet distance criterion
    for trace in &samples {
        let simulated = simulator(trace);
        let dist = distance_fn.distance(&observed_data, &simulated);
        assert!(dist <= tolerance);
        assert!(simulated.iter().all(|&v| v.is_finite()));
    }
}

#[test]
fn test_abc_smc_basic() {
    let mut rng = seeded_rng(789);

    let model_fn = || models::gaussian_mean(0.5);

    let simulator =
        |trace: &Trace| -> Vec<f64> { vec![trace.get_f64(&addr!("mu")).unwrap_or(0.0)] };

    let observed_data = vec![0.8];
    let distance_fn = EuclideanDistance;

    let config = ABCSMCConfig {
        initial_tolerance: 1.0,
        tolerance_schedule: vec![0.5, 0.2],
        particles_per_round: 20,
    };

    let final_samples = abc_smc(
        &mut rng,
        model_fn,
        simulator,
        &observed_data,
        &distance_fn,
        config,
    );

    assert_eq!(final_samples.len(), 20);

    // Final samples should be close to observed data
    for trace in &final_samples {
        let simulated = simulator(trace);
        let dist = distance_fn.distance(&observed_data, &simulated);
        assert!(dist <= 0.2); // Within final tolerance
    }
}

#[test]
fn test_distance_functions() {
    let euclidean = EuclideanDistance;
    let manhattan = ManhattanDistance;

    let vec1 = vec![1.0, 2.0, 3.0];
    let vec2 = vec![1.0, 2.0, 3.0];
    let vec3 = vec![2.0, 3.0, 4.0];

    // Test identical vectors
    assert_approx_eq(euclidean.distance(&vec1, &vec2), 0.0, TEST_TOLERANCE);
    assert_approx_eq(manhattan.distance(&vec1, &vec2), 0.0, TEST_TOLERANCE);

    // Test different vectors
    let euclidean_dist = euclidean.distance(&vec1, &vec3);
    let manhattan_dist = manhattan.distance(&vec1, &vec3);

    assert!(euclidean_dist > 0.0);
    assert!(manhattan_dist > 0.0);
    assert_finite(euclidean_dist);
    assert_finite(manhattan_dist);

    // Manhattan distance should be sum of absolute differences: |1-2| + |2-3| + |3-4| = 3
    assert_approx_eq(manhattan_dist, 3.0, TEST_TOLERANCE);

    // Euclidean distance should be sqrt(1^2 + 1^2 + 1^2) = sqrt(3)
    assert_approx_eq(euclidean_dist, 3.0_f64.sqrt(), TEST_TOLERANCE);
}

#[test]
fn test_summary_statistics_distance() {
    let weights = vec![1.0, 1.0, 1.0]; // Equal weights for mean, std, median
    let summary_dist = SummaryStatsDistance::new(weights);

    let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let data2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let data3 = vec![2.0, 3.0, 4.0, 5.0, 6.0]; // Shifted by 1

    // Identical datasets should have zero distance
    assert_approx_eq(summary_dist.distance(&data1, &data2), 0.0, TEST_TOLERANCE);

    // Different datasets should have positive distance
    let dist = summary_dist.distance(&data1, &data3);
    assert!(dist > 0.0);
    assert_finite(dist);
}

#[test]
fn test_abc_with_discrete_model() {
    let mut rng = seeded_rng(999);

    // Bernoulli model
    let model_fn = || {
        sample(addr!("p"), Beta::new(2.0, 2.0).unwrap())
            .bind(|p| sample(addr!("coin"), Bernoulli::new(p).unwrap()))
    };

    // Convert boolean to scalar for distance computation
    let simulator = |trace: &Trace| -> f64 {
        if trace.get_bool(&addr!("coin")).unwrap_or(false) {
            1.0
        } else {
            0.0
        }
    };

    let observed_data = 1.0; // Observed heads
    let distance_fn = ScalarDistance;
    let tolerance = 0.1; // Must be exact match for discrete data
    let max_samples = 100;

    let samples = abc_rejection(
        &mut rng,
        &model_fn,
        &simulator,
        &observed_data,
        &distance_fn,
        tolerance,
        max_samples,
    );

    assert!(!samples.is_empty());

    // All accepted samples should have coin=true (heads)
    for trace in &samples {
        assert_eq!(trace.get_bool(&addr!("coin")), Some(true));

        // p values should be biased toward higher values (since we observed heads)
        let p = trace.get_f64(&addr!("p")).unwrap();
        assert!(p >= 0.0 && p <= 1.0);
    }

    // Check that p values are biased upward
    let p_values: Vec<f64> = samples
        .iter()
        .filter_map(|t| t.get_f64(&addr!("p")))
        .collect();

    let mean_p = stats::mean(&p_values);
    assert!(mean_p > 0.5); // Should be biased toward heads
}

#[test]
fn test_abc_tolerance_effect() {
    let mut rng = seeded_rng(1111);

    let model_fn = || models::gaussian_mean(0.0);
    let summary_fn = |trace: &Trace| -> f64 { trace.get_f64(&addr!("mu")).unwrap_or(0.0) };

    // Test with loose tolerance
    let samples_loose = abc_scalar_summary(
        &mut rng,
        &model_fn,
        &summary_fn,
        0.0, // target
        1.0, // loose tolerance
        50,
    );

    // Test with tight tolerance
    let samples_tight = abc_scalar_summary(
        &mut rng,
        &model_fn,
        &summary_fn,
        0.0, // target
        0.1, // tight tolerance
        50,
    );

    // Loose tolerance should accept more samples
    assert!(samples_loose.len() >= samples_tight.len());

    // All tight samples should be within tight tolerance
    for trace in &samples_tight {
        let sample = trace.get_f64(&addr!("mu")).unwrap();
        assert!((sample - 0.0).abs() <= 0.1);
    }

    // All loose samples should be within loose tolerance
    for trace in &samples_loose {
        let sample = trace.get_f64(&addr!("mu")).unwrap();
        assert!((sample - 0.0).abs() <= 1.0);
    }
}

// Custom distance function for testing
#[derive(Clone, Copy)]
struct ScalarDistance;

impl DistanceFunction<f64> for ScalarDistance {
    fn distance(&self, observed: &f64, simulated: &f64) -> f64 {
        (observed - simulated).abs()
    }
}

#[test]
fn test_custom_distance_function() {
    let dist_fn = ScalarDistance;

    assert_approx_eq(dist_fn.distance(&2.0, &2.0), 0.0, TEST_TOLERANCE);
    assert_approx_eq(dist_fn.distance(&1.0, &3.0), 2.0, TEST_TOLERANCE);
    assert_approx_eq(dist_fn.distance(&-1.0, &1.0), 2.0, TEST_TOLERANCE);
}

#[test]
fn test_abc_efficiency_comparison() {
    let mut rng = seeded_rng(2222);

    let easy_model = || models::gaussian_mean(0.0);
    let summary_fn = |trace: &Trace| -> f64 { trace.get_f64(&addr!("mu")).unwrap_or(0.0) };

    // Easy case: target at prior mean with generous tolerance
    let samples_easy = abc_scalar_summary(
        &mut rng,
        &easy_model,
        &summary_fn,
        0.0, // at prior mean
        0.5, // generous tolerance
        50,
    );

    // Hard case: target away from prior mean with tight tolerance
    let samples_hard = abc_scalar_summary(
        &mut rng,
        &easy_model,
        &summary_fn,
        3.0, // far from prior mean
        0.1, // tight tolerance
        50,
    );

    // Easy case should accept more samples
    assert!(samples_easy.len() > samples_hard.len());

    // Verify samples meet their respective criteria
    for trace in &samples_easy {
        let sample = trace.get_f64(&addr!("mu")).unwrap();
        assert!((sample - 0.0).abs() <= 0.5);
    }

    for trace in &samples_hard {
        let sample = trace.get_f64(&addr!("mu")).unwrap();
        assert!((sample - 3.0).abs() <= 0.1);
    }
}
