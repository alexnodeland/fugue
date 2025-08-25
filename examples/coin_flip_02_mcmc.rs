//! # MCMC Inference for Coin Flip
//! 
//! **Tutorial**: [Bayesian Coin Flip Tutorial](../docs/src/tutorials/bayesian-coin-flip.md)  
//! **Section**: Step 2: MCMC Inference  
//! **Level**: Beginner  
//! **Concepts**: MCMC, Metropolis-Hastings, Trace analysis, Burn-in, Acceptance rates
//! 
//! This example demonstrates proper MCMC inference for the coin flip model.
//! Prior sampling doesn't give posterior samples - we need MCMC for the true posterior.

use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn coin_model(observed_heads: u64, total_flips: u64) -> Model<f64> {
    prob! {
        let bias <- sample(addr!("bias"), Beta::new(1.0, 1.0).unwrap());
        observe(addr!("heads"), Binomial::new(total_flips, bias).unwrap(), observed_heads);
        pure(bias)
    }
}

fn run_mcmc_analysis() {
    let heads = 7;
    let flips = 10;
    let model = || coin_model(heads, flips);

    println!("🔗 Running MCMC...");

    let mut rng = StdRng::seed_from_u64(42);

    // Collect MCMC samples manually to understand the process
    let mut samples = Vec::new();

    // Start with a sample from the prior
    let (mut current_bias, mut current_trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model(),
    );

    let n_samples = 2000;
    let mut n_accepted = 0;

    for i in 0..n_samples {
        // Propose new state using Metropolis-Hastings
        let (new_bias, new_trace) = inference::mh::single_site_random_walk_mh(
            &mut rng,
            0.1,  // Step size - smaller for better acceptance
            model,
            &current_trace,
        );

        // The MH algorithm automatically handles acceptance/rejection
        // We just need to check if the trace changed
        if (new_trace.total_log_weight() - current_trace.total_log_weight()).abs() > 1e-10
           || (new_bias - current_bias).abs() > 1e-10 {
            n_accepted += 1;
        }

        current_bias = new_bias;
        current_trace = new_trace;
        samples.push(current_bias);

        if i % 400 == 0 {
            println!("  Iteration {}: bias = {:.3}", i, current_bias);
        }
    }

    let acceptance_rate = n_accepted as f64 / n_samples as f64;
    println!("  Acceptance rate: {:.1}%", acceptance_rate * 100.0);

    analyze_samples(&samples, heads, flips);
}

fn analyze_samples(samples: &[f64], heads: u64, flips: u64) {
    // Remove burn-in (first 25% of samples)
    let burnin = samples.len() / 4;
    let posterior_samples = &samples[burnin..];

    // Compute posterior statistics
    let mean = posterior_samples.iter().sum::<f64>() / posterior_samples.len() as f64;
    let variance = posterior_samples.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (posterior_samples.len() - 1) as f64;
    let std_dev = variance.sqrt();

    // Compute credible interval (central 95%)
    let mut sorted_samples = posterior_samples.to_vec();
    sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ci_lower = sorted_samples[(0.025 * sorted_samples.len() as f64) as usize];
    let ci_upper = sorted_samples[(0.975 * sorted_samples.len() as f64) as usize];

    println!("\n📈 Posterior Analysis:");
    println!("  Data: {} heads out of {} flips", heads, flips);
    println!("  Sample size: {} (after burn-in)", posterior_samples.len());
    println!("  Posterior mean: {:.3}", mean);
    println!("  Posterior std: {:.3}", std_dev);
    println!("  95% credible interval: [{:.3}, {:.3}]", ci_lower, ci_upper);

    // Answer specific questions
    let prob_fair = posterior_samples.iter()
        .filter(|&&x| (x - 0.5).abs() < 0.05)  // Within 5% of fair
        .count() as f64 / posterior_samples.len() as f64;

    let prob_biased_towards_heads = posterior_samples.iter()
        .filter(|&&x| x > 0.5)
        .count() as f64 / posterior_samples.len() as f64;

    println!("\n🤔 Questions Answered:");
    println!("  Probability coin is fair (45%-55%): {:.1}%", prob_fair * 100.0);
    println!("  Probability biased toward heads: {:.1}%", prob_biased_towards_heads * 100.0);
}

fn main() {
    println!("🪙 Bayesian Coin Flip Analysis");
    println!("================================");
    run_mcmc_analysis();
}
