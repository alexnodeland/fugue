use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn analytical_solution(heads: u64, flips: u64, prior_alpha: f64, prior_beta: f64) {
    // Conjugate update: Beta(α, β) + Binomial(n, k) → Beta(α + k, β + n - k)
    let posterior_alpha = prior_alpha + heads as f64;
    let posterior_beta = prior_beta + (flips - heads) as f64;

    // Beta distribution mean and variance
    let mean = posterior_alpha / (posterior_alpha + posterior_beta);
    let variance = (posterior_alpha * posterior_beta) /
                   ((posterior_alpha + posterior_beta).powi(2) * (posterior_alpha + posterior_beta + 1.0));
    let std_dev = variance.sqrt();

    println!("\n🧮 Analytical Solution:");
    println!("  Prior: Beta({}, {})", prior_alpha, prior_beta);
    println!("  Posterior: Beta({}, {})", posterior_alpha, posterior_beta);
    println!("  Posterior mean: {:.3}", mean);
    println!("  Posterior std: {:.3}", std_dev);

    // Credible interval using Beta quantiles (approximation)
    // For exact values, you'd use a proper Beta quantile function
    let alpha = 0.05;  // 95% CI
    println!("  95% credible interval: approximately [{:.3}, {:.3}]",
             mean - 1.96 * std_dev, mean + 1.96 * std_dev);
}

fn validation_experiment() {
    let heads = 7;
    let flips = 10;

    println!("🔬 Validation: MCMC vs Analytical");
    println!("==================================");

    // Analytical solution
    analytical_solution(heads, flips, 1.0, 1.0);

    // MCMC solution
    let model = || coin_model(heads, flips);
    let mut rng = StdRng::seed_from_u64(42);

    // Use Fugue's built-in adaptive MCMC for better results
    let samples = inference::mcmc::adaptive_mcmc_chain(
        &mut rng,
        model,
        5000,  // More samples for better accuracy
        2000,  // Longer warmup
    );

    // Extract bias values from traces
    let bias_samples: Vec<f64> = samples
        .iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("bias")))
        .collect();

    let mcmc_mean = bias_samples.iter().sum::<f64>() / bias_samples.len() as f64;
    let mcmc_variance = bias_samples.iter()
        .map(|x| (x - mcmc_mean).powi(2))
        .sum::<f64>() / (bias_samples.len() - 1) as f64;

    println!("\n📊 MCMC Results:");
    println!("  Sample size: {}", bias_samples.len());
    println!("  MCMC mean: {:.3}", mcmc_mean);
    println!("  MCMC std: {:.3}", mcmc_variance.sqrt());

    // Compare with analytical
    let analytical_mean = 8.0 / 12.0;  // (1+7) / (1+1+10)
    let difference = (mcmc_mean - analytical_mean).abs();

    println!("\n✅ Validation:");
    println!("  Analytical mean: {:.3}", analytical_mean);
    println!("  MCMC mean: {:.3}", mcmc_mean);
    println!("  Absolute difference: {:.4}", difference);

    if difference < 0.01 {
        println!("  🎉 MCMC matches analytical solution!");
    } else {
        println!("  ⚠️ MCMC differs from analytical solution");
    }
}

fn coin_model(observed_heads: u64, total_flips: u64) -> Model<f64> {
    prob! {
        let bias <- sample(addr!("bias"), Beta::new(1.0, 1.0).unwrap());
        observe(addr!("heads"), Binomial::new(total_flips, bias).unwrap(), observed_heads);
        pure(bias)
    }
}

fn main() {
    validation_experiment();
}