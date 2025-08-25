use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn diagnostic_analysis() {
    let (advertising, revenue) = generate_data();

    println!("🔍 Model Diagnostics");
    println!("===================");

    let model = || linear_regression_model(advertising.clone(), revenue.clone());
    let mut rng = StdRng::seed_from_u64(42);

    let samples = inference::mcmc::adaptive_mcmc_chain(
        &mut rng,
        model,
        4000,
        2000,
    );

    // Extract parameters
    let alpha_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("alpha")))
        .collect();
    let beta_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("beta")))
        .collect();
    let sigma_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("sigma")))
        .collect();

    // 1. Parameter correlation
    let n = alpha_samples.len() as f64;
    let alpha_mean = alpha_samples.iter().sum::<f64>() / n;
    let beta_mean = beta_samples.iter().sum::<f64>() / n;

    let correlation = alpha_samples.iter()
        .zip(beta_samples.iter())
        .map(|(&a, &b)| (a - alpha_mean) * (b - beta_mean))
        .sum::<f64>() / n;

    let alpha_var = alpha_samples.iter()
        .map(|&a| (a - alpha_mean).powi(2))
        .sum::<f64>() / n;
    let beta_var = beta_samples.iter()
        .map(|&b| (b - beta_mean).powi(2))
        .sum::<f64>() / n;

    let corr_coeff = correlation / (alpha_var.sqrt() * beta_var.sqrt());

    println!("📈 Parameter Diagnostics:");
    println!("  α-β correlation: {:.3}", corr_coeff);
    if corr_coeff.abs() > 0.8 {
        println!("  ⚠️ High correlation - consider centering predictors");
    } else {
        println!("  ✅ Reasonable parameter correlation");
    }

    // 2. Posterior predictive checking
    println!("\n🎯 Posterior Predictive Checks:");

    let mut residuals = Vec::new();
    for (i, (&ad, &obs_rev)) in advertising.iter().zip(revenue.iter()).enumerate() {
        let predicted_means: Vec<f64> = alpha_samples.iter()
            .zip(beta_samples.iter())
            .map(|(&a, &b)| a + b * ad)
            .collect();

        let mean_prediction = predicted_means.iter().sum::<f64>() / predicted_means.len() as f64;
        residuals.push(obs_rev - mean_prediction);
    }

    // Check residual patterns
    let residual_mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
    let residual_std = (residuals.iter()
        .map(|r| (r - residual_mean).powi(2))
        .sum::<f64>() / residuals.len() as f64).sqrt();

    println!("  Residual mean: {:.3} (should be ≈0)", residual_mean);
    println!("  Residual std: {:.2}", residual_std);

    // 3. R-squared calculation
    let revenue_mean = revenue.iter().sum::<f64>() / revenue.len() as f64;
    let total_sum_squares: f64 = revenue.iter()
        .map(|&r| (r - revenue_mean).powi(2))
        .sum();
    let residual_sum_squares: f64 = residuals.iter()
        .map(|&r| r.powi(2))
        .sum();

    let r_squared = 1.0 - (residual_sum_squares / total_sum_squares);
    println!("  R²: {:.3}", r_squared);

    if r_squared > 0.8 {
        println!("  ✅ Strong linear relationship");
    } else if r_squared > 0.5 {
        println!("  ⚠️ Moderate relationship - consider non-linear terms");
    } else {
        println!("  🔴 Weak linear relationship");
    }

    // 4. Leave-one-out cross-validation (simplified)
    println!("\n🔄 Model Validation:");
    let mut loo_errors = Vec::new();

    for holdout_idx in 0..advertising.len() {
        let train_ad: Vec<f64> = advertising.iter()
            .enumerate()
            .filter(|(i, _)| *i != holdout_idx)
            .map(|(_, &ad)| ad)
            .collect();
        let train_rev: Vec<f64> = revenue.iter()
            .enumerate()
            .filter(|(i, _)| *i != holdout_idx)
            .map(|(_, &rev)| rev)
            .collect();

        // Quick prediction using posterior means (simplified for tutorial)
        let test_ad = advertising[holdout_idx];
        let test_rev = revenue[holdout_idx];
        let predicted = alpha_mean + beta_mean * test_ad;
        loo_errors.push((test_rev - predicted).abs());
    }

    let mean_absolute_error = loo_errors.iter().sum::<f64>() / loo_errors.len() as f64;
    println!("  Leave-one-out MAE: ${:.1}K", mean_absolute_error);

    if mean_absolute_error < sigma_samples.iter().sum::<f64>() / sigma_samples.len() as f64 {
        println!("  ✅ Good predictive performance");
    } else {
        println!("  ⚠️ Consider model improvements");
    }
}

// Include generate_data and linear_regression_model functions

fn main() {
    diagnostic_analysis();
}