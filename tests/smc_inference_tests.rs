//! Tests for SMC (Sequential Monte Carlo) inference methods.

mod test_utils;

use fugue::inference::smc::*;
use fugue::*;
use test_utils::*;

#[test]
fn test_smc_prior_particles_basic() {
    let mut rng = test_rng();

    let model = || models::gaussian_mean(0.5);
    let particles = smc_prior_particles(&mut rng, 100, model);

    assert_eq!(particles.len(), 100);

    // Weights should be normalized
    let total_weight: f64 = particles.iter().map(|p| p.weight).sum();
    assert_approx_eq(total_weight, 1.0, TEST_TOLERANCE);

    // All particles should have valid traces
    for particle in &particles {
        assert!(particle.trace.choices.contains_key(&addr!("mu")));
        assert_finite(particle.trace.total_log_weight());
        assert_finite(particle.weight);
        assert_finite(particle.log_weight);
        assert!(particle.weight > 0.0);
    }
}

#[test]
fn test_effective_sample_size() {
    // Test with uniform weights
    let uniform_particles: Vec<Particle> = (0..4)
        .map(|_| Particle {
            trace: Trace::default(),
            weight: 0.25,
            log_weight: 0.25_f64.ln(),
        })
        .collect();

    let ess_uniform = effective_sample_size(&uniform_particles);
    assert_approx_eq(ess_uniform, 4.0, TEST_TOLERANCE);

    // Test with one dominant weight
    let weights = vec![0.97, 0.01, 0.01, 0.01];
    let skewed_particles: Vec<Particle> = weights
        .iter()
        .map(|&w| Particle {
            trace: Trace::default(),
            weight: w,
            log_weight: w.ln(),
        })
        .collect();

    let ess_skewed = effective_sample_size(&skewed_particles);
    assert!(ess_skewed < 2.0);
    assert!(ess_skewed >= 1.0);
}

#[test]
fn test_adaptive_smc_basic() {
    let mut rng = seeded_rng(456);

    let model = || models::gaussian_mean(2.0);

    let config = SMCConfig {
        ess_threshold: 0.5,
        resampling_method: ResamplingMethod::Systematic,
        rejuvenation_steps: 3,
    };

    let particles = adaptive_smc(&mut rng, 50, model, config);

    assert_eq!(particles.len(), 50);

    // Should be properly normalized
    let total_weight: f64 = particles.iter().map(|p| p.weight).sum();
    assert_approx_eq(total_weight, 1.0, TEST_TOLERANCE);

    // ESS should be reasonable
    let ess = effective_sample_size(&particles);
    assert!(ess > 5.0);

    // All particles should have valid traces
    for particle in &particles {
        assert!(particle.trace.choices.contains_key(&addr!("mu")));
        assert_finite(particle.weight);
        assert!(particle.weight > 0.0);
        assert_finite(particle.log_weight);
    }
}

#[test]
fn test_smc_resampling_methods() {
    let mut rng = seeded_rng(123);

    // Create particles with uneven weights
    let weights = vec![0.6, 0.3, 0.08, 0.02];
    let particles: Vec<Particle> = weights
        .iter()
        .enumerate()
        .map(|(i, &w)| Particle {
            trace: {
                let mut t = Trace::default();
                t.insert_choice(addr!("x", i), ChoiceValue::F64(i as f64), -0.5);
                t
            },
            weight: w,
            log_weight: w.ln(),
        })
        .collect();

    // Test systematic resampling
    let resampled = resample_particles(&mut rng, &particles, ResamplingMethod::Systematic);
    assert_eq!(resampled.len(), particles.len());

    let total_weight: f64 = resampled.iter().map(|p| p.weight).sum();
    assert_approx_eq(total_weight, 1.0, TEST_TOLERANCE);

    // After resampling, weights should be uniform
    let uniform_weight = 1.0 / particles.len() as f64;
    for particle in &resampled {
        assert_approx_eq(particle.weight, uniform_weight, TEST_TOLERANCE);
    }
}

#[test]
fn test_normalize_particles() {
    let mut particles = vec![
        Particle {
            trace: Trace::default(),
            weight: 2.0,
            log_weight: 2.0_f64.ln(),
        },
        Particle {
            trace: Trace::default(),
            weight: 4.0,
            log_weight: 4.0_f64.ln(),
        },
    ];

    normalize_particles(&mut particles);

    // After normalization, weights should sum to 1
    let total_weight: f64 = particles.iter().map(|p| p.weight).sum();
    assert_approx_eq(total_weight, 1.0, TEST_TOLERANCE);

    // Relative weights should be preserved: 2:4 -> 1/3:2/3
    assert_approx_eq(particles[0].weight, 1.0 / 3.0, TEST_TOLERANCE);
    assert_approx_eq(particles[1].weight, 2.0 / 3.0, TEST_TOLERANCE);
}
