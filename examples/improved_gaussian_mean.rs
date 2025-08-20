//! Improved Gaussian mean estimation example with comprehensive diagnostics.
//!
//! This example demonstrates the enhanced numerical stability and diagnostic
//! capabilities of the improved Fugue implementation.

use clap::Parser;
use fugue::*;
// removed unused import
use rand::{rngs::StdRng, SeedableRng};

/// Model for Gaussian mean estimation with known variance.
fn gaussian_mean_model(obs: f64) -> Model<f64> {
    sample(addr!("mu"), Normal::new(0.0, 5.0).unwrap()).bind(move |mu| {
        observe(addr!("y"), Normal::new(mu, 1.0).unwrap(), obs).bind(move |_| pure(mu))
    })
}

#[derive(Parser, Debug)]
#[command(
    name = "improved_gaussian_mean",
    about = "Improved Gaussian mean estimation with diagnostics"
)]
struct Args {
    /// Observation value
    #[arg(long, default_value_t = 2.7)]
    obs: f64,

    /// Number of MCMC samples
    #[arg(long, default_value_t = 1000)]
    n_samples: usize,

    /// Number of warmup samples
    #[arg(long, default_value_t = 500)]
    n_warmup: usize,

    /// Number of chains for diagnostics
    #[arg(long, default_value_t = 4)]
    n_chains: usize,

    /// Random seed
    #[arg(long)]
    seed: Option<u64>,

    /// Run validation tests
    #[arg(long)]
    validate: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Validate model parameters
    let prior = Normal::new(0.0, 5.0).unwrap();
    let likelihood = Normal::new(args.obs, 1.0).unwrap();
    prior.validate()?;
    likelihood.validate()?;

    println!("=== Improved Fugue: Gaussian Mean Estimation ===");
    println!("Observation: {}", args.obs);
    println!("Prior: N(0, 5²)");
    println!("Likelihood: N(μ, 1²)");

    // Analytical solution for comparison
    let prior_precision = 1.0 / 25.0; // 1/σ²
    let likelihood_precision = 1.0; // 1/1²
    let posterior_precision = prior_precision + likelihood_precision;
    let posterior_variance = 1.0 / posterior_precision;
    let posterior_mean =
        posterior_variance * (prior_precision * 0.0 + likelihood_precision * args.obs);

    println!(
        "\nAnalytical posterior: N({:.4}, {:.4}²)",
        posterior_mean,
        posterior_variance.sqrt()
    );

    // Run multiple chains for diagnostics
    let mut all_samples = Vec::new();
    let mut chains = Vec::new();

    for chain_id in 0..args.n_chains {
        let seed = args.seed.unwrap_or(42) + chain_id as u64;
        let mut rng = StdRng::seed_from_u64(seed);

        println!("\nRunning chain {} with seed {}...", chain_id + 1, seed);

        let chain_samples = adaptive_mcmc_chain(
            &mut rng,
            || gaussian_mean_model(args.obs),
            args.n_samples,
            args.n_warmup,
        );

        // Extract mu values
        let mu_samples: Vec<f64> = chain_samples
            .iter()
            .filter_map(|(_, trace)| trace.choices.get(&addr!("mu")))
            .filter_map(|choice| match choice.value {
                ChoiceValue::F64(mu) => Some(mu),
                _ => None,
            })
            .collect();

        if mu_samples.is_empty() {
            eprintln!("Warning: Chain {} produced no samples", chain_id + 1);
            continue;
        }

        // Compute chain statistics
        let chain_mean = mu_samples.iter().sum::<f64>() / mu_samples.len() as f64;
        let chain_var = mu_samples
            .iter()
            .map(|&x| (x - chain_mean).powi(2))
            .sum::<f64>()
            / (mu_samples.len() - 1) as f64;
        let ess = effective_sample_size_mcmc(&mu_samples);
        let geweke = geweke_diagnostic(&mu_samples);

        println!(
            "  Chain {}: mean={:.4}, sd={:.4}, ESS={:.1}, Geweke={:.2}",
            chain_id + 1,
            chain_mean,
            chain_var.sqrt(),
            ess,
            geweke
        );

        chains.push(chain_samples.into_iter().map(|(_, trace)| trace).collect());
        all_samples.extend(mu_samples);
    }

    if all_samples.is_empty() {
        return Err("No valid samples produced".into());
    }

    // Overall posterior statistics
    let sample_mean = all_samples.iter().sum::<f64>() / all_samples.len() as f64;
    let sample_var = all_samples
        .iter()
        .map(|&x| (x - sample_mean).powi(2))
        .sum::<f64>()
        / (all_samples.len() - 1) as f64;

    println!("\n=== Posterior Summary ===");
    println!(
        "Sample mean: {:.4} (true: {:.4})",
        sample_mean, posterior_mean
    );
    println!(
        "Sample sd: {:.4} (true: {:.4})",
        sample_var.sqrt(),
        posterior_variance.sqrt()
    );
    println!("Mean error: {:.6}", (sample_mean - posterior_mean).abs());
    println!(
        "SD error: {:.6}",
        (sample_var.sqrt() - posterior_variance.sqrt()).abs()
    );

    // Convergence diagnostics
    if chains.len() >= 2 {
        let r_hat = r_hat_f64(&chains, &addr!("mu"));
        println!("\n=== Convergence Diagnostics ===");
        println!("R-hat: {:.4} (should be < 1.1)", r_hat);

        if r_hat > 1.1 {
            println!("WARNING: R-hat > 1.1 suggests poor convergence");
        } else {
            println!("✓ Chains appear to have converged");
        }
    }

    // Validation against analytical solution
    if args.validate {
        println!("\n=== Validation ===");

        let mcmc_test = |rng: &mut StdRng, n_samples: usize, n_warmup: usize| {
            adaptive_mcmc_chain(rng, || gaussian_mean_model(args.obs), n_samples, n_warmup)
        };

        let validation = test_conjugate_normal_model(
            &mut StdRng::seed_from_u64(args.seed.unwrap_or(42)),
            mcmc_test,
            0.0,      // prior_mu
            5.0,      // prior_sigma
            1.0,      // likelihood_sigma
            args.obs, // observation
            args.n_samples,
            args.n_warmup,
        );

        validation.print_summary();

        if !validation.is_valid() {
            println!("WARNING: Validation failed - MCMC may not be working correctly");
        } else {
            println!("✓ MCMC validation passed");
        }
    }

    // Test different inference methods for comparison
    println!("\n=== Method Comparison ===");

    // Prior sampling (baseline)
    let mut rng = StdRng::seed_from_u64(args.seed.unwrap_or(42));
    let (prior_sample, prior_trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        gaussian_mean_model(args.obs),
    );
    println!(
        "Prior sample: μ={:.4}, log_weight={:.4}",
        prior_sample,
        prior_trace.total_log_weight()
    );

    // SMC
    let smc_particles = adaptive_smc(
        &mut rng,
        100,
        || gaussian_mean_model(args.obs),
        SMCConfig::default(),
    );
    let smc_mean =
        smc_particles
            .iter()
            .filter_map(|p| p.trace.choices.get(&addr!("mu")))
            .filter_map(|choice| match choice.value {
                ChoiceValue::F64(mu) => Some(mu),
                _ => None,
            })
            .map(|mu| {
                mu * smc_particles
                    .iter()
                    .find(|p| {
                        p.trace.choices.get(&addr!("mu"))
                .map(|c| matches!(c.value, ChoiceValue::F64(x) if (x - mu).abs() < 1e-10))
                .unwrap_or(false)
                    })
                    .unwrap()
                    .weight
            })
            .sum::<f64>();

    println!(
        "SMC estimate: μ={:.4}, ESS={:.1}",
        smc_mean,
        effective_sample_size(&smc_particles)
    );

    Ok(())
}
