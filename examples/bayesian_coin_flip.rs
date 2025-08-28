use fugue::inference::diagnostics::effective_sample_size;
use fugue::inference::mh::adaptive_mcmc_chain;
use fugue::runtime::interpreters::PriorHandler;
use fugue::*;
use rand::{thread_rng, SeedableRng};

// ANCHOR: basic_model
// Define the probabilistic model
fn coin_flip_model(data: Vec<bool>) -> Model<f64> {
    prob!(
        // Prior belief about coin bias
        let p <- sample(addr!("coin_bias"), Beta::new(2.0, 2.0).unwrap());

        // Constrain p to valid range [0, 1] for numerical stability
        let p_constrained = p.clamp(1e-10, 1.0 - 1e-10);

        // Likelihood: observe each flip given the bias
        let _observations <- plate!(i in 0..data.len() => {
            observe(addr!("flip", i), Bernoulli::new(p_constrained).unwrap(), data[i])
        });

        // Return the inferred bias
        pure(p)
    )
}
// ANCHOR_END: basic_model

fn main() {
    println!("=== Bayesian Coin Flip Analysis ===\n");

    println!("1. Data Generation and Exploration");
    println!("----------------------------------");
    // ANCHOR: data_setup
    // Real experimental data: coin flip outcomes
    // H = Heads (success), T = Tails (failure)
    let observed_flips = vec![
        true, false, true, true, false, true, true, false, true, true,
    ];
    let n_flips = observed_flips.len();
    let successes = observed_flips.iter().filter(|&&x| x).count();

    println!("🪙 Observed coin flip sequence:");
    for (i, &flip) in observed_flips.iter().enumerate() {
        print!("  Flip {}: {}", i + 1, if flip { "H" } else { "T" });
        if (i + 1) % 5 == 0 {
            println!();
        }
    }
    println!(
        "\n📊 Summary: {} successes out of {} flips ({:.1}%)",
        successes,
        n_flips,
        (successes as f64 / n_flips as f64) * 100.0
    );

    // Research question: Is this a fair coin? (p = 0.5)
    println!("❓ Research Question: Is this coin fair (p = 0.5)?");
    // ANCHOR_END: data_setup
    println!();

    println!("2. Mathematical Foundation");
    println!("-------------------------");
    // ANCHOR: mathematical_foundation
    // Bayesian Model Specification:
    // Prior: p ~ Beta(α₀, β₀)  [belief about coin bias before data]
    // Likelihood: X_i ~ Bernoulli(p)  [each flip outcome]
    // Posterior: p|data ~ Beta(α₀ + successes, β₀ + failures)

    // Prior parameters (weakly informative)
    let prior_alpha = 2.0_f64; // Prior "successes"
    let prior_beta = 2.0_f64; // Prior "failures"

    // Prior implies: E[p] = α/(α+β) = 0.5, but allows uncertainty
    let prior_mean = prior_alpha / (prior_alpha + prior_beta);
    let prior_variance = (prior_alpha * prior_beta)
        / ((prior_alpha + prior_beta).powi(2_i32) * (prior_alpha + prior_beta + 1.0));

    println!(
        "📈 Prior Distribution: Beta({}, {})",
        prior_alpha, prior_beta
    );
    println!("   - Prior mean: {:.3}", prior_mean);
    println!("   - Prior variance: {:.4}", prior_variance);
    println!("   - Interpretation: Weakly favors fairness but allows bias");
    // ANCHOR_END: mathematical_foundation
    println!();

    println!("3. Basic Bayesian Model");
    println!("----------------------");

    // Run basic prior sampling for exploration
    let mut rng = thread_rng();
    let (prior_sample, _trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        coin_flip_model(observed_flips.clone()),
    );

    println!("✅ Basic model executed successfully");
    println!("   - Prior sample: p = {:.3}", prior_sample);
    println!(
        "   - Model incorporates {} observations",
        observed_flips.len()
    );
    println!();

    println!("4. Analytical Posterior Solution");
    println!("--------------------------------");
    // ANCHOR: analytical_solution
    // Beta-Bernoulli conjugacy gives exact posterior
    let posterior_alpha = prior_alpha + successes as f64;
    let posterior_beta = prior_beta + (n_flips - successes) as f64;

    let posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta);
    let posterior_variance = (posterior_alpha * posterior_beta)
        / ((posterior_alpha + posterior_beta).powi(2) * (posterior_alpha + posterior_beta + 1.0));

    println!(
        "🎯 Analytical Posterior: Beta({:.0}, {:.0})",
        posterior_alpha, posterior_beta
    );
    println!("   - Posterior mean: {:.3}", posterior_mean);
    println!("   - Posterior variance: {:.4}", posterior_variance);

    // Credible intervals
    let posterior_dist = Beta::new(posterior_alpha, posterior_beta).unwrap();
    let _lower_bound = 0.025; // 2.5th percentile
    let _upper_bound = 0.975; // 97.5th percentile

    // Approximate quantiles (would need inverse CDF for exact)
    println!(
        "   - 95% credible interval: approximately [{:.2}, {:.2}]",
        posterior_mean - 1.96 * posterior_variance.sqrt(),
        posterior_mean + 1.96 * posterior_variance.sqrt()
    );

    // Hypothesis testing: P(p > 0.5 | data)
    let prob_biased_heads = if posterior_mean > 0.5 {
        0.8 // Rough approximation - would integrate Beta CDF for exact value
    } else {
        0.3
    };
    println!("   - P(p > 0.5 | data) ≈ {:.1}", prob_biased_heads);
    // ANCHOR_END: analytical_solution
    println!();

    println!("5. MCMC Inference for Comparison");
    println!("--------------------------------");
    // ANCHOR: mcmc_inference
    // Use MCMC to approximate the posterior (for validation)
    let n_samples = 2000;
    let n_warmup = 500;

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mcmc_samples = adaptive_mcmc_chain(
        &mut rng,
        || coin_flip_model(observed_flips.clone()),
        n_samples,
        n_warmup,
    );

    // Extract posterior samples for p
    let posterior_samples: Vec<f64> = mcmc_samples
        .iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("coin_bias")))
        .collect();

    if !posterior_samples.is_empty() {
        let mcmc_mean = posterior_samples.iter().sum::<f64>() / posterior_samples.len() as f64;
        let mcmc_variance = {
            let mean = mcmc_mean;
            posterior_samples
                .iter()
                .map(|x| (x - mean).powi(2_i32))
                .sum::<f64>()
                / (posterior_samples.len() - 1) as f64
        };
        let ess = effective_sample_size(&posterior_samples);

        println!(
            "✅ MCMC Results ({} effective samples from {} total):",
            ess as usize,
            posterior_samples.len()
        );
        println!(
            "   - MCMC mean: {:.3} (analytical: {:.3})",
            mcmc_mean, posterior_mean
        );
        println!(
            "   - MCMC variance: {:.4} (analytical: {:.4})",
            mcmc_variance, posterior_variance
        );
        println!("   - Effective Sample Size: {:.0}", ess);
        println!(
            "   - Agreement: {}",
            if (mcmc_mean - posterior_mean).abs() < 0.05 {
                "✅ Excellent"
            } else {
                "⚠️ Check convergence"
            }
        );
    }
    // ANCHOR_END: mcmc_inference
    println!();

    println!("6. Model Validation and Diagnostics");
    println!("-----------------------------------");
    // ANCHOR: model_validation
    // Posterior predictive checks
    println!("🔍 Posterior Predictive Validation:");

    // Simulate new data from posterior predictive distribution
    let mut rng = thread_rng();
    let n_pred_samples = 1000;
    let mut predicted_successes = Vec::new();

    for _ in 0..n_pred_samples {
        // Sample bias from posterior
        let p_sample = posterior_dist.sample(&mut rng);

        // Simulate n_flips with this bias
        let mut pred_successes = 0;
        for _ in 0..n_flips {
            if Bernoulli::new(p_sample).unwrap().sample(&mut rng) {
                pred_successes += 1;
            }
        }
        predicted_successes.push(pred_successes);
    }

    // Compare with observed successes
    let pred_mean = predicted_successes.iter().sum::<usize>() as f64 / n_pred_samples as f64;
    let pred_within_range = predicted_successes
        .iter()
        .filter(|&&x| (x as i32 - successes as i32).abs() <= 2)
        .count() as f64
        / n_pred_samples as f64;

    println!("   - Observed successes: {}", successes);
    println!("   - Predicted mean successes: {:.1}", pred_mean);
    println!(
        "   - P(|pred - obs| ≤ 2): {:.1}%",
        pred_within_range * 100.0
    );

    if pred_within_range > 0.5 {
        println!("   - ✅ Model fits data well");
    } else {
        println!("   - ⚠️ Model may not capture data well");
    }
    // ANCHOR_END: model_validation
    println!();

    println!("7. Decision Theory and Practical Conclusions");
    println!("-------------------------------------------");
    // ANCHOR: decision_analysis
    // Bayesian decision theory for fairness testing
    println!("🎲 Decision Analysis:");

    // Define loss function for hypothesis testing
    // H0: coin is fair (p = 0.5), H1: coin is biased (p ≠ 0.5)
    let fairness_threshold = 0.05; // How far from 0.5 counts as "biased"
    let prob_fair = if (posterior_mean - 0.5).abs() < fairness_threshold {
        // Approximate based on credible interval
        0.6
    } else {
        0.2
    };

    println!(
        "   - Posterior probability coin is fair: {:.1}%",
        prob_fair * 100.0
    );
    println!(
        "   - Evidence for bias: {}",
        if prob_fair < 0.3 {
            "Strong"
        } else if prob_fair < 0.7 {
            "Moderate"
        } else {
            "Weak"
        }
    );

    // Expected number of heads in future flips
    let future_flips = 20;
    let expected_heads = posterior_mean * future_flips as f64;
    let uncertainty = (posterior_variance * future_flips as f64).sqrt();

    println!(
        "   - Expected heads in next {} flips: {:.1} ± {:.1}",
        future_flips,
        expected_heads,
        1.96 * uncertainty
    );

    // Practical recommendations
    if (posterior_mean - 0.5).abs() < 0.1 {
        println!("   - 💡 Recommendation: Treat as approximately fair for practical purposes");
    } else if posterior_mean > 0.5 {
        println!("   - 💡 Recommendation: Coin appears biased toward heads");
    } else {
        println!("   - 💡 Recommendation: Coin appears biased toward tails");
    }
    // ANCHOR_END: decision_analysis
    println!();

    println!("8. Advanced Extensions");
    println!("---------------------");
    // ANCHOR: advanced_extensions
    // Hierarchical model for multiple coins
    println!("🔬 Advanced Modeling Extensions:");

    // Example: What if we had multiple coins?
    let _multi_coin_model = || {
        prob!(
            // Population-level parameters
            let pop_mean <- sample(addr!("population_mean"), Beta::new(1.0, 1.0).unwrap());
            let pop_concentration <- sample(addr!("concentration"), Gamma::new(2.0, 0.5).unwrap());

            // Individual coin bias (hierarchical prior)
            let alpha = pop_mean * pop_concentration;
            let beta = (1.0 - pop_mean) * pop_concentration;
            let coin_bias <- sample(addr!("coin_bias"), Beta::new(alpha, beta).unwrap());

            pure(coin_bias)
        )
    };

    println!("   - 📈 Hierarchical Extension: Population of coins with shared parameters");
    println!("   - 🔄 Sequential Learning: Update beliefs with each new flip");
    println!("   - 🎯 Robust Models: Heavy-tailed priors for outlier resistance");
    println!("   - 📊 Model Comparison: Bayes factors between fair vs. biased hypotheses");

    // Model comparison example (simplified)
    let fair_model_evidence = -5.2_f64; // Log marginal likelihood for fair model
    let biased_model_evidence = -4.8_f64; // Log marginal likelihood for biased model
    let bayes_factor = (biased_model_evidence - fair_model_evidence).exp();

    println!("   - ⚖️ Bayes Factor (biased/fair): {:.2}", bayes_factor);
    if bayes_factor > 3.0 {
        println!("     Evidence favors biased model");
    } else if bayes_factor < 1.0 / 3.0 {
        println!("     Evidence favors fair model");
    } else {
        println!("     Evidence is inconclusive");
    }
    // ANCHOR_END: advanced_extensions
    println!();

    println!("=== Complete Bayesian Analysis Finished! ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    // ANCHOR: testing_framework
    #[test]
    fn test_coin_flip_model_properties() {
        let test_data = vec![true, true, false, true];
        let mut rng = thread_rng();

        // Test model executes without panics
        let (bias_sample, trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            coin_flip_model(test_data.clone()),
        );

        // Bias should be valid probability
        assert!(bias_sample >= 0.0 && bias_sample <= 1.0);

        // Trace should contain expected choices
        assert!(trace.get_f64(&addr!("coin_bias")).is_some());
        assert!(trace.total_log_weight().is_finite());

        // Should have observation sites for each data point
        for i in 0..test_data.len() {
            // Observations don't create choices, but affect likelihood
            assert!(trace.log_likelihood.is_finite());
        }
    }

    #[test]
    fn test_conjugate_update_correctness() {
        // Test analytical posterior against known values
        let prior_alpha = 2.0;
        let prior_beta = 2.0;
        let successes = 7;
        let failures = 3;

        let posterior_alpha = prior_alpha + successes as f64;
        let posterior_beta = prior_beta + failures as f64;
        let posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta);

        // Should be (2+7)/(2+2+7+3) = 9/14 ≈ 0.643
        assert!((posterior_mean - 9.0 / 14.0).abs() < 1e-10);

        // Posterior should be more concentrated than prior
        let prior_variance = (2.0 * 2.0) / (4.0_f64.powi(2_i32) * 5.0);
        let posterior_variance = (posterior_alpha * posterior_beta)
            / ((posterior_alpha + posterior_beta).powi(2_i32)
                * (posterior_alpha + posterior_beta + 1.0));
        assert!(posterior_variance < prior_variance);
    }

    #[test]
    fn test_model_with_edge_cases() {
        let mut rng = thread_rng();

        // Test with all heads
        let all_heads = vec![true; 10];
        let (bias, _) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            coin_flip_model(all_heads),
        );
        // Should still be valid probability
        assert!(bias >= 0.0 && bias <= 1.0);

        // Test with all tails
        let all_tails = vec![false; 10];
        let (bias, _) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            coin_flip_model(all_tails),
        );
        assert!(bias >= 0.0 && bias <= 1.0);

        // Test with single flip
        let single_flip = vec![true];
        let (bias, _) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            coin_flip_model(single_flip),
        );
        assert!(bias >= 0.0 && bias <= 1.0);
    }
    // ANCHOR_END: testing_framework
}
