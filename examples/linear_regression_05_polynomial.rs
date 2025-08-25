use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Generate data with quadratic relationship
fn generate_quadratic_data() -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(9999);

    let true_alpha = 20.0;   // Intercept
    let true_beta1 = 2.0;    // Linear term
    let true_beta2 = 0.08;   // Quadratic term (diminishing returns)
    let true_sigma = 6.0;

    let advertising: Vec<f64> = (1..=25)
        .map(|i| i as f64 * 2.0)  // $2K to $50K
        .collect();

    let revenue: Vec<f64> = advertising
        .iter()
        .map(|&ad| {
            let mean = true_alpha + true_beta1 * ad + true_beta2 * ad * ad;
            Normal::new(mean, true_sigma).unwrap().sample(&mut rng)
        })
        .collect();

    (advertising, revenue)
}

fn quadratic_regression_model(
    advertising: Vec<f64>,
    revenue: Vec<f64>
) -> Model<(f64, f64, f64, f64)> {
    prob! {
        // Priors for polynomial regression
        let alpha <- sample(addr!("alpha"), Normal::new(0.0, 100.0).unwrap());
        let beta1 <- sample(addr!("beta1"), Normal::new(0.0, 10.0).unwrap());  // Linear
        let beta2 <- sample(addr!("beta2"), Normal::new(0.0, 1.0).unwrap());   // Quadratic (smaller scale)
        let sigma <- sample(addr!("sigma"), Exponential::new(1.0).unwrap());

        // Likelihood with quadratic terms
        for (i, (&ad, &rev)) in advertising.iter().zip(revenue.iter()).enumerate() {
            let predicted_mean = alpha + beta1 * ad + beta2 * ad * ad;
            observe(
                addr!("revenue", i),
                Normal::new(predicted_mean, sigma).unwrap(),
                rev
            );
        }

        pure((alpha, beta1, beta2, sigma))
    }
}

fn polynomial_analysis() {
    let (advertising, revenue) = generate_quadratic_data();

    println!("📈 Polynomial Regression Analysis");
    println!("=================================");

    // Compare linear vs quadratic models
    println!("📊 Comparing Models...");

    // Linear model
    let linear_model = || linear_regression_model(advertising.clone(), revenue.clone());
    let mut rng = StdRng::seed_from_u64(42);

    let linear_samples = inference::mcmc::adaptive_mcmc_chain(
        &mut rng,
        linear_model,
        2000,
        1000,
    );

    // Quadratic model
    let quad_model = || quadratic_regression_model(advertising.clone(), revenue.clone());
    let mut rng = StdRng::seed_from_u64(42);

    let quad_samples = inference::mcmc::adaptive_mcmc_chain(
        &mut rng,
        quad_model,
        2000,
        1000,
    );

    // Model comparison via log-likelihood
    let linear_log_likelihood = linear_samples.iter()
        .map(|(_, trace)| trace.total_log_weight())
        .sum::<f64>() / linear_samples.len() as f64;

    let quad_log_likelihood = quad_samples.iter()
        .map(|(_, trace)| trace.total_log_weight())
        .sum::<f64>() / quad_samples.len() as f64;

    println!("\n🏆 Model Comparison:");
    println!("  Linear model avg log-likelihood: {:.1}", linear_log_likelihood);
    println!("  Quadratic model avg log-likelihood: {:.1}", quad_log_likelihood);

    if quad_log_likelihood > linear_log_likelihood + 5.0 {
        println!("  ✅ Quadratic model strongly preferred");
    } else if quad_log_likelihood > linear_log_likelihood {
        println!("  📊 Quadratic model slightly preferred");
    } else {
        println!("  📊 Linear model adequate");
    }

    // Quadratic model parameters
    let alpha_samples: Vec<f64> = quad_samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("alpha")))
        .collect();
    let beta1_samples: Vec<f64> = quad_samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("beta1")))
        .collect();
    let beta2_samples: Vec<f64> = quad_samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("beta2")))
        .collect();

    let alpha_mean = alpha_samples.iter().sum::<f64>() / alpha_samples.len() as f64;
    let beta1_mean = beta1_samples.iter().sum::<f64>() / beta1_samples.len() as f64;
    let beta2_mean = beta2_samples.iter().sum::<f64>() / beta2_samples.len() as f64;

    println!("\n📊 Quadratic Model Estimates:");
    println!("  Intercept (α): {:.2}", alpha_mean);
    println!("  Linear term (β₁): {:.3}", beta1_mean);
    println!("  Quadratic term (β₂): {:.4}", beta2_mean);

    // Economic interpretation
    println!("\n💡 Economic Insights:");
    if beta2_mean < 0.0 {
        println!("  📉 Diminishing returns detected (β₂ < 0)");
        // Optimal advertising spend (where derivative = 0)
        let optimal_ad = -beta1_mean / (2.0 * beta2_mean);
        if optimal_ad > 0.0 && optimal_ad < 100.0 {
            println!("  🎯 Optimal ad spend: ${:.0}K", optimal_ad);
        }
    } else if beta2_mean > 0.0 {
        println!("  📈 Accelerating returns detected (β₂ > 0)");
    }

    // Prediction comparison
    println!("\n🔮 Prediction Comparison (Ad Spend: $40K):");
    let test_ad = 40.0;

    // Linear prediction
    let linear_alpha: Vec<f64> = linear_samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("alpha")))
        .collect();
    let linear_beta: Vec<f64> = linear_samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("beta")))
        .collect();

    let linear_preds: Vec<f64> = linear_alpha.iter()
        .zip(linear_beta.iter())
        .map(|(&a, &b)| a + b * test_ad)
        .collect();
    let linear_mean = linear_preds.iter().sum::<f64>() / linear_preds.len() as f64;

    // Quadratic prediction
    let quad_preds: Vec<f64> = alpha_samples.iter()
        .zip(beta1_samples.iter())
        .zip(beta2_samples.iter())
        .map(|((&a, &b1), &b2)| a + b1 * test_ad + b2 * test_ad * test_ad)
        .collect();
    let quad_mean = quad_preds.iter().sum::<f64>() / quad_preds.len() as f64;

    println!("  Linear model: ${:.1}K", linear_mean);
    println!("  Quadratic model: ${:.1}K", quad_mean);
    println!("  Difference: ${:.1}K", (quad_mean - linear_mean).abs());
}

// Include previous functions
fn main() {
    polynomial_analysis();
}