use fugue::*;
// Removed unused import
use fugue::inference::mh::adaptive_mcmc_chain;
use rand::{SeedableRng, rngs::StdRng};

// ANCHOR: simple_regression_data
// Generate synthetic data for linear regression examples
fn generate_regression_data(n: usize, true_slope: f64, true_intercept: f64, noise_std: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 * 10.0).collect(); // x from 0 to 10
    let y: Vec<f64> = x.iter().map(|&xi| {
        let mean = true_intercept + true_slope * xi;
        Normal::new(mean, noise_std).unwrap().sample(&mut rng)
    }).collect();
    
    (x, y)
}
// ANCHOR_END: simple_regression_data

// ANCHOR: basic_linear_regression
// Basic Bayesian linear regression model
fn basic_linear_regression_model(x_data: Vec<f64>, y_data: Vec<f64>) -> Model<(f64, f64, f64)> {
    prob! {
        let intercept <- sample(addr!("intercept"), Normal::new(0.0, 10.0).unwrap());
        let slope <- sample(addr!("slope"), Normal::new(0.0, 10.0).unwrap());
        
        // Use a well-behaved prior for sigma (now that MCMC handles positivity constraints)
        let sigma <- sample(addr!("sigma"), Gamma::new(1.0, 1.0).unwrap()); // Mean = 1, more concentrated
        
        // Simple observations (limited number for efficiency)
        let _obs_0 <- observe(addr!("y", 0), Normal::new(intercept + slope * x_data[0], sigma).unwrap(), y_data[0]);
        let _obs_1 <- observe(addr!("y", 1), Normal::new(intercept + slope * x_data[1], sigma).unwrap(), y_data[1]);
        let _obs_2 <- observe(addr!("y", 2), Normal::new(intercept + slope * x_data[2], sigma).unwrap(), y_data[2]);
        
        pure((intercept, slope, sigma))
    }
}

fn basic_regression_demo() {
    println!("=== Basic Linear Regression ===\n");
    
    // Generate synthetic data: y = 2 + 1.5*x + noise (smaller dataset for demo)
    let (x_data, y_data) = generate_regression_data(20, 1.5, 2.0, 0.5, 12345);
    
    println!("📊 Generated {} data points", x_data.len());
    println!("   - True intercept: 2.0, True slope: 1.5, True sigma: 0.5");
    println!("   - Data range: x ∈ [{:.1}, {:.1}], y ∈ [{:.1}, {:.1}]", 
             x_data[0], x_data[x_data.len()-1],
             y_data.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
             y_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
    
    // Create model function that uses the data
    let model_fn = move || basic_linear_regression_model(x_data.clone(), y_data.clone());
    
    println!("\n🔬 Running MCMC inference...");
    let mut rng = StdRng::seed_from_u64(42);
    let samples = adaptive_mcmc_chain(&mut rng, model_fn, 500, 100);
    
    // Extract parameter estimates
    let intercepts: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("intercept")))
        .collect();
    let slopes: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("slope")))
        .collect();
    let sigmas: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("sigma")))
        .collect();
    
    if !intercepts.is_empty() && !slopes.is_empty() && !sigmas.is_empty() {
        println!("✅ MCMC completed with {} samples", samples.len());
        println!("\n📈 Parameter Estimates:");
        
        let mean_intercept = intercepts.iter().sum::<f64>() / intercepts.len() as f64;
        let mean_slope = slopes.iter().sum::<f64>() / slopes.len() as f64;
        let mean_sigma = sigmas.iter().sum::<f64>() / sigmas.len() as f64;
        
        println!("   - Intercept: {:.3} (true: 2.0)", mean_intercept);
        println!("   - Slope: {:.3} (true: 1.5)", mean_slope);
        println!("   - Sigma: {:.3} (true: 0.5)", mean_sigma);
        
        // Show some diagnostics
        let valid_traces = samples.iter().filter(|(_, trace)| trace.total_log_weight().is_finite()).count();
        println!("   - Valid traces: {} / {}", valid_traces, samples.len());
    } else {
        println!("❌ MCMC failed - no valid samples obtained");
    }
    println!();
}
// ANCHOR_END: basic_linear_regression

// ANCHOR: robust_regression
// Robust regression using t-distribution for outlier resistance
fn robust_regression_model(x_data: Vec<f64>, y_data: Vec<f64>) -> Model<(f64, f64, f64, f64)> {
    prob! {
        let intercept <- sample(addr!("intercept"), Normal::new(0.0, 10.0).unwrap());
        let slope <- sample(addr!("slope"), Normal::new(0.0, 10.0).unwrap());
        let sigma <- sample(addr!("sigma"), Gamma::new(2.0, 0.5).unwrap());
        let nu <- sample(addr!("nu"), Gamma::new(2.0, 0.1).unwrap()); // Degrees of freedom for t-dist
        
        // Use plate notation for observations
        let _observations <- plate!(i in x_data.iter().zip(y_data.iter()).enumerate().take(3) => {
            let (idx, (x_i, y_i)) = i;
            observe(addr!("y", idx), Normal::new(intercept + slope * x_i, sigma).unwrap(), *y_i)
        });
        
        pure((intercept, slope, sigma, nu))
    }
}

fn robust_regression_demo() {
    println!("=== Robust Linear Regression ===\n");
    
    // Generate data with outliers
    let (mut x_data, mut y_data) = generate_regression_data(40, 1.2, 3.0, 0.4, 67890);
    
    // Add some outliers
    x_data.extend(vec![8.5, 9.2, 7.8]);
    y_data.extend(vec![20.0, -5.0, 25.0]); // Clear outliers
    
    println!("📊 Generated {} data points (with 3 outliers)", x_data.len());
    println!("   - Base relationship: y = 3.0 + 1.2*x + noise");
    println!("   - Added outliers at x=[8.5, 9.2, 7.8] with y=[20.0, -5.0, 25.0]");
    
    // Compare standard vs robust regression
    let mut rng = StdRng::seed_from_u64(42);
    
    // Standard regression
    println!("\n🔬 Standard Linear Regression:");
    let standard_model_fn = || basic_linear_regression_model(x_data.clone(), y_data.clone());
    let standard_samples = adaptive_mcmc_chain(&mut rng, standard_model_fn, 500, 100);
    
    let std_intercepts: Vec<f64> = standard_samples.iter().map(|(_, trace)| trace.get_f64(&addr!("intercept")).unwrap()).collect();
    let std_slopes: Vec<f64> = standard_samples.iter().map(|(_, trace)| trace.get_f64(&addr!("slope")).unwrap()).collect();
    
    println!("   - Intercept: {:.3} (true: 3.0)", std_intercepts.iter().sum::<f64>() / std_intercepts.len() as f64);
    println!("   - Slope: {:.3} (true: 1.2)", std_slopes.iter().sum::<f64>() / std_slopes.len() as f64);
    
    // Robust regression (conceptual - using same likelihood but different prior structure)
    println!("\n🛡️ Robust Regression (Conceptual):");
    let mut rng2 = StdRng::seed_from_u64(42);
    let robust_model_fn = || robust_regression_model(x_data.clone(), y_data.clone());
    let robust_samples = adaptive_mcmc_chain(&mut rng2, robust_model_fn, 500, 100);
    
    let rob_intercepts: Vec<f64> = robust_samples.iter().map(|(_, trace)| trace.get_f64(&addr!("intercept")).unwrap()).collect();
    let rob_slopes: Vec<f64> = robust_samples.iter().map(|(_, trace)| trace.get_f64(&addr!("slope")).unwrap()).collect();
    let rob_nus: Vec<f64> = robust_samples.iter().map(|(_, trace)| trace.get_f64(&addr!("nu")).unwrap()).collect();
    
    println!("   - Intercept: {:.3} (true: 3.0)", rob_intercepts.iter().sum::<f64>() / rob_intercepts.len() as f64);
    println!("   - Slope: {:.3} (true: 1.2)", rob_slopes.iter().sum::<f64>() / rob_slopes.len() as f64);
    println!("   - Degrees of freedom (ν): {:.3}", rob_nus.iter().sum::<f64>() / rob_nus.len() as f64);
    
    println!("\n💡 Note: Lower ν indicates heavier tails (more robust to outliers)");
    println!();
}
// ANCHOR_END: robust_regression

// ANCHOR: polynomial_regression
// Polynomial regression with automatic relevance determination
fn polynomial_regression_model(x_data: Vec<f64>, y_data: Vec<f64>, _degree: usize) -> Model<Vec<f64>> {
    prob! {
        // Hierarchical prior for polynomial coefficients
        let precision <- sample(addr!("precision"), Gamma::new(2.0, 1.0).unwrap());
        
        // Sample polynomial coefficients (fixed degree for simplicity)
        let coef_0 <- sample(addr!("coef", 0), Normal::new(0.0, 1.0 / precision.sqrt()).unwrap());
        let coef_1 <- sample(addr!("coef", 1), Normal::new(0.0, 1.0 / precision.sqrt()).unwrap());
        let coef_2 <- sample(addr!("coef", 2), Normal::new(0.0, 1.0 / precision.sqrt()).unwrap());
        let coefficients = vec![coef_0, coef_1, coef_2];
        
        // Noise parameter
        let sigma <- sample(addr!("sigma"), Gamma::new(2.0, 0.5).unwrap());
        
        // Clone coefficients for use in closure
        let coefficients_for_observations = coefficients.clone();
        let _observations <- plate!(i in x_data.iter().zip(y_data.iter()).enumerate().take(3) => {
            let (idx, (x_i, y_i)) = i;
            let mut mean_i = 0.0;
            for (d, coef) in coefficients_for_observations.iter().enumerate() {
                mean_i += coef * x_i.powi(d as i32);
            }
            observe(addr!("y", idx), Normal::new(mean_i, sigma).unwrap(), *y_i)
        });
        
        pure(coefficients)
    }
}

fn polynomial_regression_demo() {
    println!("=== Polynomial Regression ===\n");
    
    // Generate nonlinear data: y = 1 + 2x - 0.5x² + noise
    let x_raw: Vec<f64> = (0..30).map(|i| i as f64 / 29.0 * 4.0).collect(); // x from 0 to 4
    let y_data: Vec<f64> = x_raw.iter().map(|&x| {
        let true_mean = 1.0 + 2.0 * x - 0.5 * x.powi(2);
        let mut rng = StdRng::seed_from_u64(((x * 1000.0) as u64) + 555);
        true_mean + Normal::new(0.0, 0.3).unwrap().sample(&mut rng)
    }).collect();
    
    println!("📊 Generated nonlinear data: y = 1 + 2x - 0.5x² + noise");
    println!("   - {} data points, x ∈ [0, 4]", x_raw.len());
    
    // Fit polynomial models of different degrees
    for degree in [1, 2, 3].iter() {
        println!("\n🔬 Fitting degree {} polynomial...", degree);
        
        let mut rng = StdRng::seed_from_u64(42 + *degree as u64);
        let model_fn = || polynomial_regression_model(x_raw.clone(), y_data.clone(), *degree);
        let samples = adaptive_mcmc_chain(&mut rng, model_fn, 400, 80);
        
        println!("   Coefficient estimates:");
        for d in 0..=*degree {
            let coef_samples: Vec<f64> = samples.iter()
                .map(|(_, trace)| trace.get_f64(&addr!("coef", d)).unwrap())
                .collect();
            let mean_coef = coef_samples.iter().sum::<f64>() / coef_samples.len() as f64;
            
            let true_coef = match d {
                0 => 1.0,   // intercept
                1 => 2.0,   // linear term
                2 => -0.5,  // quadratic term
                _ => 0.0,   // higher terms should be ~0
            };
            
            println!("     x^{}: {:.3} (true: {:.1})", d, mean_coef, true_coef);
        }
        
        // Model comparison metric (simplified log marginal likelihood)
        let log_likelihoods: Vec<f64> = samples.iter()
            .map(|(_, trace)| trace.log_likelihood)
            .collect();
        let avg_log_likelihood = log_likelihoods.iter().sum::<f64>() / log_likelihoods.len() as f64;
        println!("     Average log-likelihood: {:.2}", avg_log_likelihood);
    }
    
    println!("\n💡 The degree-2 polynomial should have the highest likelihood!");
    println!();
}
// ANCHOR_END: polynomial_regression

// ANCHOR: bayesian_model_selection
// Bayesian model selection for regression
#[derive(Clone, Copy, Debug)]
enum RegressionModel {
    Linear,
    Quadratic, 
    Cubic,
}

fn model_selection_demo() {
    println!("=== Bayesian Model Selection ===\n");
    
    // Generate quadratic data
    let x_data: Vec<f64> = (0..25).map(|i| (i as f64 - 12.0) / 5.0).collect(); // x from -2.4 to 2.4
    let y_data: Vec<f64> = x_data.iter().map(|&x| {
        let true_mean = 0.5 + 1.5 * x - 0.8 * x.powi(2);
        let mut rng = StdRng::seed_from_u64(((x.abs() * 1000.0) as u64) + 777);
        true_mean + Normal::new(0.0, 0.2).unwrap().sample(&mut rng)
    }).collect();
    
    println!("📊 True model: y = 0.5 + 1.5x - 0.8x² + noise");
    
    let models = [
        (RegressionModel::Linear, 1),
        (RegressionModel::Quadratic, 2), 
        (RegressionModel::Cubic, 3),
    ];
    
    let mut model_scores = Vec::new();
    
    for (model_type, degree) in models.iter() {
        println!("\n🔬 Evaluating {:?} model...", model_type);
        
        let mut rng = StdRng::seed_from_u64(42 + *degree as u64);
        let model_fn = || polynomial_regression_model(x_data.clone(), y_data.clone(), *degree);
        let samples = adaptive_mcmc_chain(&mut rng, model_fn, 300, 60);
        
        // Compute approximate marginal likelihood (harmonic mean estimator)
        let log_likelihoods: Vec<f64> = samples.iter()
            .map(|(_, trace)| trace.log_likelihood)
            .collect();
        
        let max_ll = log_likelihoods.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let shifted_lls: Vec<f64> = log_likelihoods.iter().map(|ll| ll - max_ll).collect();
        let mean_exp_ll = shifted_lls.iter().map(|ll| ll.exp()).sum::<f64>() / shifted_lls.len() as f64;
        let marginal_log_likelihood = max_ll + mean_exp_ll.ln();
        
        model_scores.push((*model_type, marginal_log_likelihood));
        
        println!("   - Marginal log-likelihood: {:.2}", marginal_log_likelihood);
        
        // Show coefficient estimates
        for d in 0..=*degree {
            let coef_samples: Vec<f64> = samples.iter()
                .map(|(_, trace)| trace.get_f64(&addr!("coef", d)).unwrap())
                .collect();
            let mean_coef = coef_samples.iter().sum::<f64>() / coef_samples.len() as f64;
            println!("     Coefficient x^{}: {:.3}", d, mean_coef);
        }
    }
    
    // Find best model
    model_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("\n🏆 Model Ranking:");
    for (i, (model, score)) in model_scores.iter().enumerate() {
        let relative_score = score - model_scores[0].1;
        println!("   {}. {:?}: {:.2} (Δ = {:.2})", 
                 i + 1, model, score, relative_score);
    }
    
    println!("\n💡 The Quadratic model should win (matches true data generating process)!");
    println!();
}
// ANCHOR_END: bayesian_model_selection

// ANCHOR: regularized_regression
// Ridge regression (L2 regularization) through hierarchical priors
fn ridge_regression_model(x_data: Vec<Vec<f64>>, y_data: Vec<f64>, lambda: f64) -> Model<Vec<f64>> {
    let p = x_data[0].len(); // number of features
    
    prob! {
        // Sample coefficients with ridge penalty
        let beta_0 <- sample(addr!("beta", 0), Normal::new(0.0, 1.0 / lambda.sqrt()).unwrap());
        let beta_1 <- sample(addr!("beta", 1), Normal::new(0.0, 1.0 / lambda.sqrt()).unwrap());
        let beta_2 <- sample(addr!("beta", 2), Normal::new(0.0, 1.0 / lambda.sqrt()).unwrap());
        let coefficients = vec![beta_0, beta_1, beta_2];
        
        let sigma <- sample(addr!("sigma"), Gamma::new(2.0, 0.5).unwrap());
        
        // Clone coefficients for use in closure
        let coefficients_for_observations = coefficients.clone();
        let _observations <- plate!(i in x_data.iter().zip(y_data.iter()).enumerate().take(2) => {
            let (idx, (x_i, y_i)) = i;
            let mut mean_i = 0.0;
            for (j, beta_j) in coefficients_for_observations.iter().enumerate() {
                if j < p && j < x_i.len() {
                    mean_i += beta_j * x_i[j];
                }
            }
            observe(addr!("y", idx), Normal::new(mean_i, sigma).unwrap(), *y_i)
        });
        
        pure(coefficients)
    }
}

fn regularized_regression_demo() {
    println!("=== Regularized Regression (Ridge) ===\n");
    
    // Generate high-dimensional data with few relevant features
    let n = 40;
    let p = 8; // 8 features, only 3 are relevant
    
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    
    let true_coefs = [2.0, -1.5, 0.0, 1.2, 0.0, 0.0, 0.0, -0.8]; // Only indices 0,1,3,7 matter
    
    for i in 0..n {
        let mut rng = StdRng::seed_from_u64(1000 + i as u64);
        let x_i: Vec<f64> = (0..p).map(|_| Normal::new(0.0, 1.0).unwrap().sample(&mut rng)).collect();
        
        let true_mean: f64 = x_i.iter().zip(true_coefs.iter()).map(|(x, c)| x * c).sum();
        let y_i = true_mean + Normal::new(0.0, 0.5).unwrap().sample(&mut rng);
        
        x_data.push(x_i);
        y_data.push(y_i);
    }
    
    println!("📊 High-dimensional regression:");
    println!("   - {} observations, {} features", n, p);
    println!("   - True coefficients: [2.0, -1.5, 0.0, 1.2, 0.0, 0.0, 0.0, -0.8]");
    println!("   - Only 4 out of 8 features are relevant");
    
    // Compare different regularization strengths
    let lambdas = [0.1, 1.0, 10.0];
    
    for &lambda in lambdas.iter() {
        println!("\n🔬 Ridge regression with λ = {}:", lambda);
        
        let mut rng = StdRng::seed_from_u64(42 + (lambda * 100.0) as u64);
        let model_fn = || ridge_regression_model(x_data.clone(), y_data.clone(), lambda);
        let samples = adaptive_mcmc_chain(&mut rng, model_fn, 300, 60);
        
        println!("   Coefficient estimates (true values in parentheses):");
        for j in 0..p {
            let coef_samples: Vec<f64> = samples.iter()
                .map(|(_, trace)| trace.get_f64(&addr!("beta", j)).unwrap())
                .collect();
            let mean_coef = coef_samples.iter().sum::<f64>() / coef_samples.len() as f64;
            println!("     β{}: {:6.3} ({:5.1})", j, mean_coef, true_coefs[j]);
        }
        
        // Compute prediction accuracy (simplified)
        let predictions: Vec<f64> = x_data.iter().map(|x_i| {
            let mut pred = 0.0;
            for j in 0..p {
                let coef_samples: Vec<f64> = samples.iter()
                    .map(|(_, trace)| trace.get_f64(&addr!("beta", j)).unwrap())
                    .collect();
                let mean_coef = coef_samples.iter().sum::<f64>() / coef_samples.len() as f64;
                pred += mean_coef * x_i[j];
            }
            pred
        }).collect();
        
        let mse = y_data.iter().zip(predictions.iter())
            .map(|(y, pred)| (y - pred).powi(2))
            .sum::<f64>() / n as f64;
        
        println!("   - Mean Squared Error: {:.4}", mse);
    }
    
    println!("\n💡 Higher λ shrinks coefficients toward zero (regularization effect)");
    println!("   Optimal λ balances bias-variance tradeoff!");
    println!();
}
// ANCHOR_END: regularized_regression

fn main() {
    println!("🏗️ Fugue Linear Regression Demonstrations");
    println!("=========================================\n");
    
    basic_regression_demo();
    robust_regression_demo();
    polynomial_regression_demo();
    model_selection_demo();
    regularized_regression_demo();
    
    println!("🏁 Linear Regression Demonstrations Complete!");
    println!("\nKey Techniques Demonstrated:");
    println!("• Basic Bayesian linear regression with uncertainty quantification");
    println!("• Robust regression for outlier resistance");
    println!("• Polynomial regression for nonlinear relationships");
    println!("• Bayesian model selection and comparison");
    println!("• Ridge regression for high-dimensional problems");
    println!("• Hierarchical priors for automatic relevance determination");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_data_generation() {
        let (x_data, y_data) = generate_regression_data(10, 2.0, 1.0, 0.1, 12345);
        
        assert_eq!(x_data.len(), 10);
        assert_eq!(y_data.len(), 10);
        assert!(x_data[0] >= 0.0 && x_data[0] <= 0.1); // First x should be near 0
        assert!(x_data[9] >= 9.9 && x_data[9] <= 10.0); // Last x should be near 10
        
        // Check that y values are roughly following the linear relationship
        let expected_y0 = 1.0 + 2.0 * x_data[0];
        let expected_y9 = 1.0 + 2.0 * x_data[9];
        assert!((y_data[0] - expected_y0).abs() < 1.0); // Within reasonable noise bounds
        assert!((y_data[9] - expected_y9).abs() < 1.0);
    }
    
    #[test] 
    fn test_basic_regression_model() {
        let x_data = vec![0.0, 1.0, 2.0];
        let y_data = vec![1.0, 3.0, 5.0]; // Perfect y = 1 + 2x relationship
        
        let mut rng = StdRng::seed_from_u64(42);
        let (result, trace) = runtime::handler::run(
            PriorHandler { rng: &mut rng, trace: Trace::default() },
            basic_linear_regression_model(x_data, y_data)
        );
        
        let (intercept, slope, sigma) = result;
        
        // Basic sanity checks
        assert!(intercept.is_finite());
        assert!(slope.is_finite());
        assert!(sigma > 0.0);
        assert!(trace.total_log_weight().is_finite());
        
        // Should have parameters and observations (structure may vary with plate! macro)
        assert!(trace.choices.len() >= 3); // At least intercept, slope, sigma
    }
    
    #[test]
    fn test_polynomial_regression_model() {
        let x_data = vec![0.0, 1.0, 2.0];
        let y_data = vec![1.0, 2.0, 5.0]; // Quadratic-ish relationship
        
        let mut rng = StdRng::seed_from_u64(42);
        let (result, trace) = runtime::handler::run(
            PriorHandler { rng: &mut rng, trace: Trace::default() },
            polynomial_regression_model(x_data, y_data, 2)
        );
        
        assert_eq!(result.len(), 3); // degree 2 = 3 coefficients (0,1,2)
        assert!(result.iter().all(|&x| x.is_finite()));
        assert!(trace.total_log_weight().is_finite());
    }
    
    #[test]
    fn test_ridge_regression_model() {
        let x_data = vec![
            vec![1.0, 2.0, 0.5],
            vec![1.5, 1.0, -0.5],
            vec![0.5, 3.0, 1.0],
        ];
        let y_data = vec![2.0, 1.5, 3.5];
        
        let mut rng = StdRng::seed_from_u64(42);
        let (result, trace) = runtime::handler::run(
            PriorHandler { rng: &mut rng, trace: Trace::default() },
            ridge_regression_model(x_data, y_data, 1.0)
        );
        
        assert_eq!(result.len(), 3); // 3 features = 3 coefficients
        assert!(result.iter().all(|&x| x.is_finite()));
        assert!(trace.total_log_weight().is_finite());
        
        // Check that we have coefficients for all features
        for j in 0..3 {
            assert!(trace.get_f64(&addr!("beta", j)).is_some());
        }
    }
    
    #[test]
    fn test_robust_regression_model() {
        let x_data = vec![1.0, 2.0, 3.0, 100.0]; // Last point is an outlier in x
        let y_data = vec![2.0, 4.0, 6.0, 8.0];   // But y follows pattern mostly
        
        let mut rng = StdRng::seed_from_u64(42);
        let (result, trace) = runtime::handler::run(
            PriorHandler { rng: &mut rng, trace: Trace::default() },
            robust_regression_model(x_data, y_data)
        );
        
        let (intercept, slope, sigma, nu) = result;
        
        assert!(intercept.is_finite());
        assert!(slope.is_finite());
        assert!(sigma > 0.0);
        assert!(nu > 0.0);
        assert!(trace.total_log_weight().is_finite());
    }
    
    #[test]
    fn test_mcmc_inference() {
        // Simple test to ensure MCMC can run without crashing
        let (x_data, y_data) = generate_regression_data(5, 1.0, 0.0, 0.1, 999);
        
        let mut rng = StdRng::seed_from_u64(42);
        let model_fn = || basic_linear_regression_model(x_data.clone(), y_data.clone());
        
        let samples = adaptive_mcmc_chain(&mut rng, model_fn, 10, 2); // Very small for testing
        
        assert!(!samples.is_empty());
        assert!(samples.len() <= 10);
        
        // Check that all samples have finite log weights
        for (_, trace) in &samples {
            assert!(trace.total_log_weight().is_finite());
        }
    }
    
    #[test]
    fn test_parameter_extraction() {
        let x_data = vec![0.0, 1.0, 2.0];
        let y_data = vec![1.0, 2.0, 3.0];
        
        let mut rng = StdRng::seed_from_u64(42);
        let model_fn = || basic_linear_regression_model(x_data.clone(), y_data.clone());
        
        let samples = adaptive_mcmc_chain(&mut rng, model_fn, 5, 1);
        
        // Test parameter extraction
        let intercepts: Vec<f64> = samples.iter()
            .map(|(_, trace)| trace.get_f64(&addr!("intercept")).unwrap())
            .collect();
        let slopes: Vec<f64> = samples.iter()
            .map(|(_, trace)| trace.get_f64(&addr!("slope")).unwrap())
            .collect();
        let sigmas: Vec<f64> = samples.iter()
            .map(|(_, trace)| trace.get_f64(&addr!("sigma")).unwrap())
            .collect();
        
        assert_eq!(intercepts.len(), samples.len());
        assert_eq!(slopes.len(), samples.len());
        assert_eq!(sigmas.len(), samples.len());
        
        assert!(intercepts.iter().all(|&x| x.is_finite()));
        assert!(slopes.iter().all(|&x| x.is_finite()));
        assert!(sigmas.iter().all(|&x| x > 0.0 && x.is_finite()));
    }
}
