use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn linear_regression_model(advertising: Vec<f64>, revenue: Vec<f64>) -> Model<(f64, f64, f64)> {
    prob! {
        // Priors
        let alpha <- sample(addr!("alpha"), Normal::new(0.0, 100.0).unwrap());  // Intercept
        let beta <- sample(addr!("beta"), Normal::new(0.0, 10.0).unwrap());    // Slope
        let sigma <- sample(addr!("sigma"), Exponential::new(1.0).unwrap());   // Noise

        // Likelihood: observe all data points
        for (i, (&ad, &rev)) in advertising.iter().zip(revenue.iter()).enumerate() {
            let predicted_mean = alpha + beta * ad;
            observe(
                addr!("revenue", i),
                Normal::new(predicted_mean, sigma).unwrap(),
                rev
            );
        }

        pure((alpha, beta, sigma))
    }
}

fn run_basic_regression() {
    let (advertising, revenue) = generate_data();

    println!("🔗 Running Bayesian Linear Regression...");

    let model = || linear_regression_model(advertising.clone(), revenue.clone());
    let mut rng = StdRng::seed_from_u64(42);

    // Run MCMC
    let samples = inference::mcmc::adaptive_mcmc_chain(
        &mut rng,
        model,
        3000,  // Total samples
        1500,  // Burn-in
    );

    // Extract parameter samples
    let alpha_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("alpha")))
        .collect();

    let beta_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("beta")))
        .collect();

    let sigma_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("sigma")))
        .collect();

    // Posterior summaries
    let alpha_mean = alpha_samples.iter().sum::<f64>() / alpha_samples.len() as f64;
    let beta_mean = beta_samples.iter().sum::<f64>() / beta_samples.len() as f64;
    let sigma_mean = sigma_samples.iter().sum::<f64>() / sigma_samples.len() as f64;

    println!("\n📊 Posterior Estimates:");
    println!("  Intercept (α): {:.2} ± {:.2}", alpha_mean,
             (alpha_samples.iter().map(|x| (x - alpha_mean).powi(2)).sum::<f64>()
              / alpha_samples.len() as f64).sqrt());
    println!("  Slope (β): {:.3} ± {:.3}", beta_mean,
             (beta_samples.iter().map(|x| (x - beta_mean).powi(2)).sum::<f64>()
              / beta_samples.len() as f64).sqrt());
    println!("  Noise (σ): {:.2} ± {:.2}", sigma_mean,
             (sigma_samples.iter().map(|x| (x - sigma_mean).powi(2)).sum::<f64>()
              / sigma_samples.len() as f64).sqrt());

    println!("\n📝 Interpretation:");
    println!("  Base revenue: ${:.0}K", alpha_mean);
    println!("  Revenue per $1K ad spend: ${:.0}", beta_mean * 1000.0);

    // True values for comparison (since we generated the data)
    println!("\n🎯 True Values (for validation):");
    println!("  True α: 50.0, True β: 3.5, True σ: 8.0");
}

// Include the generate_data function from Step 1

fn main() {
    run_basic_regression();
}