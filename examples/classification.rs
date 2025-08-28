use fugue::inference::mh::adaptive_mcmc_chain;
use fugue::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

// ANCHOR: synthetic_classification_data
// Generate synthetic binary classification data
fn generate_classification_data(n: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<bool>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut features = Vec::new();
    let mut labels = Vec::new();

    // True coefficients: intercept=-1.0, feature1=2.0, feature2=-1.5
    let true_intercept = -1.0;
    let true_coef1 = 2.0;
    let true_coef2 = -1.5;

    for _ in 0..n {
        // Generate features from standard normal
        let x1: f64 = StandardNormal.sample(&mut rng);
        let x2: f64 = StandardNormal.sample(&mut rng);

        // Compute true log-odds and probability
        let log_odds = true_intercept + true_coef1 * x1 + true_coef2 * x2;
        let prob = 1.0 / (1.0 + (-log_odds as f64).exp());

        // Sample binary outcome
        let y = rng.gen::<f64>() < prob;

        features.push(vec![1.0, x1, x2]); // Include intercept column
        labels.push(y);
    }

    (features, labels)
}

// Generate multi-class classification data
fn generate_multiclass_data(n: usize, n_classes: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut features = Vec::new();
    let mut labels = Vec::new();

    // Create distinct clusters for each class
    for _ in 0..n {
        // Randomly assign to a class
        let true_class = rng.gen_range(0..n_classes);

        // Generate features centered around class-specific means
        let class_center_x = (true_class as f64 - (n_classes as f64 - 1.0) / 2.0) * 2.0;
        let class_center_y = if true_class % 2 == 0 { 1.0 } else { -1.0 };

        let noise1: f64 = StandardNormal.sample(&mut rng);
        let noise2: f64 = StandardNormal.sample(&mut rng);
        let x1 = class_center_x + noise1 * 0.8;
        let x2 = class_center_y + noise2 * 0.8;

        features.push(vec![1.0, x1, x2]); // Include intercept
        labels.push(true_class);
    }

    (features, labels)
}

// Generate hierarchical data (groups within population)
fn generate_hierarchical_data(
    n_groups: usize,
    n_per_group: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<bool>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut features = Vec::new();
    let mut labels = Vec::new();
    let mut groups = Vec::new();

    // Generate group-specific intercepts
    let global_intercept = 0.0;
    let group_sd = 1.0;

    for group_id in 0..n_groups {
        // Group-specific intercept
        let group_noise: f64 = StandardNormal.sample(&mut rng);
        let group_intercept = global_intercept + group_noise * group_sd;
        let slope = 1.5; // Common slope across groups

        for _ in 0..n_per_group {
            let x: f64 = StandardNormal.sample(&mut rng);

            let log_odds: f64 = group_intercept + slope * x;
            let prob = 1.0 / (1.0 + (-log_odds as f64).exp());
            let y = rng.gen::<f64>() < prob;

            features.push(vec![1.0, x]); // Intercept + one feature
            labels.push(y);
            groups.push(group_id);
        }
    }

    (features, labels, groups)
}
// ANCHOR_END: synthetic_classification_data

// ANCHOR: basic_logistic_regression
// Basic Bayesian logistic regression model
fn logistic_regression_model(features: Vec<Vec<f64>>, labels: Vec<bool>) -> Model<Vec<f64>> {
    let n_features = features[0].len();

    prob! {
        // Sample coefficients with regularizing priors - build using plate
        let coefficients <- plate!(i in 0..n_features => {
            sample(addr!("beta", i), fugue::Normal::new(0.0, 2.0).unwrap())
        });

        // Clone coefficients for use in closure
        let coefficients_for_obs = coefficients.clone();
        let _observations <- plate!(obs_idx in features.iter().zip(labels.iter()).enumerate() => {
            let (idx, (x_vec, &y)) = obs_idx;
            // Compute linear predictor (log-odds)
            let mut linear_pred = 0.0;
            for (coef, &x_val) in coefficients_for_obs.iter().zip(x_vec.iter()) {
                linear_pred += coef * x_val;
            }

            // Convert to probability using logistic function
            let prob = 1.0 / (1.0 + (-linear_pred as f64).exp());

            // Ensure probability is in valid range
            let bounded_prob = prob.max(1e-10).min(1.0 - 1e-10);

            // Observe the binary outcome
            observe(addr!("y", idx), Bernoulli::new(bounded_prob).unwrap(), y)
        });

        pure(coefficients)
    }
}

fn binary_classification_demo() {
    println!("=== Binary Classification with Logistic Regression ===\n");

    // Generate synthetic data
    let (features, labels) = generate_classification_data(100, 42);
    let positive_cases = labels.iter().filter(|&&x| x).count();

    println!("📊 Generated {} data points", features.len());
    println!("   - Features: {} dimensions", features[0].len());
    println!(
        "   - Positive cases: {} / {} ({:.1}%)",
        positive_cases,
        labels.len(),
        100.0 * positive_cases as f64 / labels.len() as f64
    );
    println!("   - True coefficients: intercept=-1.0, β₁=2.0, β₂=-1.5");

    // Run MCMC inference
    let model_fn = move || logistic_regression_model(features.clone(), labels.clone());
    let mut rng = StdRng::seed_from_u64(12345);

    println!("\n🔬 Running MCMC inference...");
    let samples = adaptive_mcmc_chain(&mut rng, model_fn, 800, 200);

    // Extract coefficient estimates
    let valid_samples: Vec<_> = samples
        .iter()
        .filter_map(|(coeffs, trace)| {
            if trace.total_log_weight().is_finite() {
                Some(coeffs)
            } else {
                None
            }
        })
        .collect();

    if !valid_samples.is_empty() {
        println!(
            "✅ MCMC completed with {} valid samples",
            valid_samples.len()
        );
        println!("\n📈 Coefficient Estimates:");

        let coef_names = ["Intercept", "β₁ (feature 1)", "β₂ (feature 2)"];
        let true_coefs = [-1.0, 2.0, -1.5];

        for (i, (name, true_val)) in coef_names.iter().zip(true_coefs.iter()).enumerate() {
            let coef_samples: Vec<f64> = valid_samples.iter().map(|coeffs| coeffs[i]).collect();

            let mean_coef = coef_samples.iter().sum::<f64>() / coef_samples.len() as f64;
            let std_coef = {
                let variance = coef_samples
                    .iter()
                    .map(|c| (c - mean_coef).powi(2))
                    .sum::<f64>()
                    / (coef_samples.len() - 1) as f64;
                variance.sqrt()
            };

            println!(
                "   - {}: {:.3} ± {:.3} (true: {:.1})",
                name, mean_coef, std_coef, true_val
            );
        }

        // Model diagnostics
        let avg_log_weight = samples
            .iter()
            .map(|(_, trace)| trace.total_log_weight())
            .filter(|w| w.is_finite())
            .sum::<f64>()
            / valid_samples.len() as f64;

        println!("   - Average log-likelihood: {:.2}", avg_log_weight);

        // Make predictions on new data
        println!("\n🔮 Prediction Example:");
        let test_features = vec![1.0, 0.5, -0.8]; // New observation
        let mut predicted_probs = Vec::new();

        for coeffs in valid_samples.iter().take(50) {
            // Use subset for speed
            let mut linear_pred = 0.0;
            for (coef, &x_val) in coeffs.iter().zip(test_features.iter()) {
                linear_pred += coef * x_val;
            }
            let prob = 1.0 / (1.0 + (-linear_pred).exp());
            predicted_probs.push(prob);
        }

        let mean_prob = predicted_probs.iter().sum::<f64>() / predicted_probs.len() as f64;
        let std_prob = {
            let variance = predicted_probs
                .iter()
                .map(|p| (p - mean_prob).powi(2))
                .sum::<f64>()
                / (predicted_probs.len() - 1) as f64;
            variance.sqrt()
        };

        println!(
            "   - Test point [0.5, -0.8]: P(y=1) = {:.3} ± {:.3}",
            mean_prob, std_prob
        );
        if mean_prob > 0.5 {
            println!("   - Prediction: Class 1 (probability > 0.5)");
        } else {
            println!("   - Prediction: Class 0 (probability < 0.5)");
        }
    } else {
        println!("❌ No valid MCMC samples obtained");
    }

    println!();
}
// ANCHOR_END: basic_logistic_regression

// ANCHOR: multinomial_classification
// Multinomial logistic regression for multi-class classification
// Note: This is a simplified version - full multinomial requires more complex implementation
fn multiclass_classification_demo() {
    println!("=== Multi-class Classification (Conceptual) ===\n");

    let (features, labels) = generate_multiclass_data(150, 3, 1337);

    println!("📊 Generated {} data points", features.len());
    println!("   - {} classes", 3);
    println!("   - Features: {} dimensions", features[0].len());

    // Count class distribution
    let mut class_counts = vec![0; 3];
    for &label in &labels {
        class_counts[label] += 1;
    }

    for (class_id, count) in class_counts.iter().enumerate() {
        println!(
            "   - Class {}: {} samples ({:.1}%)",
            class_id,
            count,
            100.0 * *count as f64 / labels.len() as f64
        );
    }

    println!("\n💡 Multinomial Classification Concepts:");
    println!("   - Uses K-1 sets of coefficients (reference category approach)");
    println!("   - Each coefficient set models log(P(class_k) / P(class_reference))");
    println!("   - Probabilities sum to 1 via softmax transformation");
    println!("   - More complex to implement but follows same Bayesian principles");

    // For now, demonstrate the concept with binary classification on each class
    println!("\n🔬 One-vs-Rest Classification (simplified approach):");

    for target_class in 0..3 {
        // Convert to binary problem: target_class vs. all others
        let binary_labels: Vec<bool> = labels.iter().map(|&label| label == target_class).collect();

        let positive_cases = binary_labels.iter().filter(|&&x| x).count();

        println!("\n   Class {} vs Rest:", target_class);
        println!(
            "   - Positive cases: {} / {}",
            positive_cases,
            binary_labels.len()
        );

        // Clone data for each iteration to avoid move issues
        let features_copy = features.clone();
        let model_fn =
            move || logistic_regression_model(features_copy.clone(), binary_labels.clone());
        let mut rng = StdRng::seed_from_u64(1000 + target_class as u64);

        let samples = adaptive_mcmc_chain(&mut rng, model_fn, 300, 60);
        let valid_samples = samples.len();

        if valid_samples > 0 {
            println!("   - MCMC: {} samples obtained", valid_samples);
        }
    }

    println!("\n💭 Note: Full multinomial logistic regression requires implementing");
    println!("   the softmax link function and careful handling of identifiability constraints.");
    println!();
}
// ANCHOR_END: multinomial_classification

// ANCHOR: hierarchical_classification
// Hierarchical logistic regression with group-level effects
fn hierarchical_classification_model(
    features: Vec<Vec<f64>>,
    labels: Vec<bool>,
    groups: Vec<usize>,
) -> Model<(f64, f64, Vec<f64>)> {
    let n_groups = groups.iter().max().unwrap_or(&0) + 1;

    prob! {
        // Global parameters
        let global_intercept <- sample(addr!("global_intercept"), fugue::Normal::new(0.0, 2.0).unwrap());
        let slope <- sample(addr!("slope"), fugue::Normal::new(0.0, 2.0).unwrap());

        // Group-level variance
        let group_sigma <- sample(addr!("group_sigma"), Gamma::new(1.0, 1.0).unwrap());

        // Group-specific intercepts using plate notation
        let group_intercepts <- plate!(g in 0..n_groups => {
            sample(addr!("group_intercept", g), fugue::Normal::new(global_intercept, group_sigma).unwrap())
        });

        // Clone group_intercepts for use in closure
        let group_intercepts_for_obs = group_intercepts.clone();
        let _observations <- plate!(data in features.iter()
            .map(|f| f[1]) // Extract the single feature (after intercept)
            .zip(labels.iter())
            .zip(groups.iter())
            .enumerate() => {
            let (obs_idx, ((x_val, &y), &group_id)) = data;
            let linear_pred = group_intercepts_for_obs[group_id] + slope * x_val;
            let prob = 1.0 / (1.0 + (-linear_pred as f64).exp());
            let bounded_prob = prob.max(1e-10).min(1.0 - 1e-10);

            observe(addr!("obs", obs_idx), Bernoulli::new(bounded_prob).unwrap(), y)
        });

        pure((global_intercept, slope, group_intercepts))
    }
}

fn hierarchical_classification_demo() {
    println!("=== Hierarchical Classification ===\n");

    let (features, labels, groups) = generate_hierarchical_data(4, 25, 5678);
    let n_groups = groups.iter().max().unwrap() + 1;

    println!("📊 Generated hierarchical data:");
    println!(
        "   - {} groups with {} observations each",
        n_groups,
        features.len() / n_groups
    );
    println!("   - Total: {} data points", features.len());

    // Show group-wise statistics
    for group_id in 0..n_groups {
        let group_labels: Vec<bool> = groups
            .iter()
            .zip(labels.iter())
            .filter_map(|(&g, &y)| if g == group_id { Some(y) } else { None })
            .collect();

        let positive_rate =
            group_labels.iter().filter(|&&x| x).count() as f64 / group_labels.len() as f64;
        println!(
            "   - Group {}: {:.1}% positive cases",
            group_id,
            positive_rate * 100.0
        );
    }

    println!("\n🔬 Running hierarchical MCMC...");
    let model_fn =
        move || hierarchical_classification_model(features.clone(), labels.clone(), groups.clone());
    let mut rng = StdRng::seed_from_u64(9999);
    let samples = adaptive_mcmc_chain(&mut rng, model_fn, 600, 150);

    let valid_samples: Vec<_> = samples
        .iter()
        .filter(|(_, trace)| trace.total_log_weight().is_finite())
        .collect();

    if !valid_samples.is_empty() {
        println!(
            "✅ Hierarchical MCMC completed with {} valid samples",
            valid_samples.len()
        );

        // Extract global parameters
        let global_intercepts: Vec<f64> =
            valid_samples.iter().map(|(params, _)| params.0).collect();
        let slopes: Vec<f64> = valid_samples.iter().map(|(params, _)| params.1).collect();

        let mean_global_int =
            global_intercepts.iter().sum::<f64>() / global_intercepts.len() as f64;
        let mean_slope = slopes.iter().sum::<f64>() / slopes.len() as f64;

        println!("\n📈 Global Parameter Estimates:");
        println!("   - Global intercept: {:.3} (true: ~0.0)", mean_global_int);
        println!("   - Slope: {:.3} (true: 1.5)", mean_slope);

        // Extract group-specific intercepts
        println!("\n🏘️  Group-Specific Intercepts:");
        for group_id in 0..n_groups {
            let group_intercepts: Vec<f64> = valid_samples
                .iter()
                .map(|(params, _)| params.2[group_id])
                .collect();

            let mean_group_int =
                group_intercepts.iter().sum::<f64>() / group_intercepts.len() as f64;
            println!("   - Group {}: {:.3}", group_id, mean_group_int);
        }

        println!("\n💡 Hierarchical Benefits:");
        println!("   - Groups share information through global parameters");
        println!("   - Individual groups can have their own intercepts");
        println!("   - Better predictions for groups with less data");
        println!("   - Automatic regularization through group-level priors");
    } else {
        println!("❌ No valid hierarchical samples obtained");
    }

    println!();
}
// ANCHOR_END: hierarchical_classification

// ANCHOR: model_comparison
// Simple model comparison using log-likelihood
fn model_comparison_demo() {
    println!("=== Model Comparison ===\n");

    let (features, labels) = generate_classification_data(80, 2021);
    let _features_ref = &features;
    let _labels_ref = &labels;

    println!("📊 Comparing different logistic regression models:");
    println!("   - Model 1: Intercept only");
    println!("   - Model 2: Intercept + Feature 1");
    println!("   - Model 3: Full model (Intercept + Feature 1 + Feature 2)");

    struct ModelResult {
        name: String,
        n_params: usize,
        log_likelihood: f64,
        samples: usize,
    }

    let mut results = Vec::new();

    // Model 1: Intercept only
    {
        let intercept_features: Vec<Vec<f64>> = features
            .iter()
            .map(|f| vec![f[0]]) // Just intercept
            .collect();
        let labels_clone = labels.clone();

        let model_fn =
            move || logistic_regression_model(intercept_features.clone(), labels_clone.clone());
        let mut rng = StdRng::seed_from_u64(1111);
        let samples = adaptive_mcmc_chain(&mut rng, model_fn, 300, 80);

        let valid_samples: Vec<_> = samples
            .iter()
            .filter(|(_, trace)| trace.total_log_weight().is_finite())
            .collect();

        if !valid_samples.is_empty() {
            let avg_log_lik = valid_samples
                .iter()
                .map(|(_, trace)| trace.total_log_weight())
                .sum::<f64>()
                / valid_samples.len() as f64;

            results.push(ModelResult {
                name: "Intercept only".to_string(),
                n_params: 1,
                log_likelihood: avg_log_lik,
                samples: valid_samples.len(),
            });
        }
    }

    // Model 2: Intercept + Feature 1
    {
        let reduced_features: Vec<Vec<f64>> = features
            .iter()
            .map(|f| vec![f[0], f[1]]) // Intercept + first feature
            .collect();
        let labels_clone = labels.clone();

        let model_fn =
            move || logistic_regression_model(reduced_features.clone(), labels_clone.clone());
        let mut rng = StdRng::seed_from_u64(2222);
        let samples = adaptive_mcmc_chain(&mut rng, model_fn, 300, 80);

        let valid_samples: Vec<_> = samples
            .iter()
            .filter(|(_, trace)| trace.total_log_weight().is_finite())
            .collect();

        if !valid_samples.is_empty() {
            let avg_log_lik = valid_samples
                .iter()
                .map(|(_, trace)| trace.total_log_weight())
                .sum::<f64>()
                / valid_samples.len() as f64;

            results.push(ModelResult {
                name: "Intercept + Feature 1".to_string(),
                n_params: 2,
                log_likelihood: avg_log_lik,
                samples: valid_samples.len(),
            });
        }
    }

    // Model 3: Full model
    {
        let labels_clone = labels.clone();
        let model_fn = move || logistic_regression_model(features.clone(), labels_clone.clone());
        let mut rng = StdRng::seed_from_u64(3333);
        let samples = adaptive_mcmc_chain(&mut rng, model_fn, 300, 80);

        let valid_samples: Vec<_> = samples
            .iter()
            .filter(|(_, trace)| trace.total_log_weight().is_finite())
            .collect();

        if !valid_samples.is_empty() {
            let avg_log_lik = valid_samples
                .iter()
                .map(|(_, trace)| trace.total_log_weight())
                .sum::<f64>()
                / valid_samples.len() as f64;

            results.push(ModelResult {
                name: "Full model".to_string(),
                n_params: 3,
                log_likelihood: avg_log_lik,
                samples: valid_samples.len(),
            });
        }
    }

    if !results.is_empty() {
        println!("\n🏆 Model Comparison Results:");
        println!("   Model                    | Params | Log-Likelihood | Samples");
        println!("   -------------------------|--------|----------------|--------");

        for result in &results {
            println!(
                "   {:24} | {:6} | {:14.2} | {:7}",
                result.name, result.n_params, result.log_likelihood, result.samples
            );
        }

        // Find best model
        if let Some(best) = results
            .iter()
            .max_by(|a, b| a.log_likelihood.partial_cmp(&b.log_likelihood).unwrap())
        {
            println!("\n🥇 Best Model: {} (highest log-likelihood)", best.name);
        }

        println!("\n💡 Model Selection Notes:");
        println!("   - Higher log-likelihood indicates better fit to data");
        println!("   - In practice, use information criteria (AIC, BIC, WAIC)");
        println!("   - These account for model complexity to prevent overfitting");
        println!("   - Cross-validation provides robust model comparison");
    } else {
        println!("❌ Model comparison failed - no valid samples obtained");
    }

    println!();
}
// ANCHOR_END: model_comparison

fn main() {
    println!("🧠 Fugue Classification Demonstrations");
    println!("=====================================\n");

    binary_classification_demo();
    multiclass_classification_demo();
    hierarchical_classification_demo();
    model_comparison_demo();

    println!("✨ Classification demonstrations completed!");
    println!("   Key advantages of Bayesian classification:");
    println!("   • Automatic uncertainty quantification");
    println!("   • Principled regularization through priors");
    println!("   • Natural handling of hierarchical structure");
    println!("   • Robust model comparison and selection");
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_generation() {
        let (features, labels) = generate_classification_data(50, 123);
        assert_eq!(features.len(), 50);
        assert_eq!(labels.len(), 50);
        assert_eq!(features[0].len(), 3); // Intercept + 2 features

        let (mc_features, mc_labels) = generate_multiclass_data(30, 3, 456);
        assert_eq!(mc_features.len(), 30);
        assert_eq!(mc_labels.len(), 30);
        assert!(mc_labels.iter().all(|&l| l < 3));

        let (h_features, h_labels, h_groups) = generate_hierarchical_data(3, 10, 789);
        assert_eq!(h_features.len(), 30);
        assert_eq!(h_labels.len(), 30);
        assert_eq!(h_groups.len(), 30);
        assert!(h_groups.iter().all(|&g| g < 3));
    }

    #[test]
    fn test_logistic_regression_model() {
        let features = vec![
            vec![1.0, 0.5, -0.2],
            vec![1.0, -0.3, 0.8],
            vec![1.0, 1.2, -1.1],
        ];
        let labels = vec![true, false, true];

        // Test that model compiles and runs
        let mut rng = StdRng::seed_from_u64(42);
        let (coefficients, trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            logistic_regression_model(features, labels),
        );

        assert_eq!(coefficients.len(), 3); // Three coefficients
        assert!(trace.choices.len() >= 3); // At least the coefficients
    }

    #[test]
    fn test_hierarchical_model() {
        let features = vec![
            vec![1.0, 0.5],
            vec![1.0, -0.3], // Group 0
            vec![1.0, 1.2],
            vec![1.0, -0.7], // Group 1
        ];
        let labels = vec![true, false, true, false];
        let groups = vec![0, 0, 1, 1];

        let mut rng = StdRng::seed_from_u64(42);
        let (params, trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            hierarchical_classification_model(features, labels, groups),
        );

        assert_eq!(params.2.len(), 2); // Two group intercepts
        assert!(trace.choices.len() >= 4); // Global params + group intercepts
    }

    #[test]
    fn test_classification_mcmc() {
        let (features, labels) = generate_classification_data(20, 999);
        let model_fn = move || logistic_regression_model(features.clone(), labels.clone());
        let mut rng = StdRng::seed_from_u64(1234);

        let samples = adaptive_mcmc_chain(&mut rng, model_fn, 10, 5);
        assert_eq!(samples.len(), 10);

        // Check that we get some valid samples
        let valid_samples = samples
            .iter()
            .filter(|(_, trace)| trace.total_log_weight().is_finite())
            .count();
        assert!(valid_samples > 0, "Should have at least some valid samples");
    }
}
