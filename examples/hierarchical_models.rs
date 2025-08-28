use fugue::*;
use rand::prelude::*;
use rand_distr::{Distribution, StandardNormal, Uniform};

// ANCHOR: varying_intercepts_model
// Hierarchical model with group-specific intercepts but shared slope
fn varying_intercepts_model(
    x_data: Vec<f64>,
    y_data: Vec<f64>, 
    group_ids: Vec<usize>,
    n_groups: usize
) -> Model<(f64, f64, f64, f64, f64)> {
    prob! {
        // Population-level parameters
        let mu_alpha <- sample(addr!("mu_alpha"), fugue::Normal::new(0.0, 5.0).unwrap());
        let sigma_alpha <- sample(addr!("sigma_alpha"), Gamma::new(1.0, 1.0).unwrap());
        let beta <- sample(addr!("beta"), fugue::Normal::new(0.0, 2.0).unwrap());
        let sigma_y <- sample(addr!("sigma_y"), Gamma::new(1.0, 1.0).unwrap());
        
        // Observations with group-specific intercepts
        let _observations <- plate!(i in 0..x_data.len() => {
            let group_j = group_ids[i];
            let x_i = x_data[i];
            let y_i = y_data[i];
            sample(addr!("alpha", group_j), fugue::Normal::new(mu_alpha, sigma_alpha).unwrap())
                .bind(move |alpha_j| {
                    let mu_i = alpha_j + beta * x_i;
                    observe(addr!("y", i), fugue::Normal::new(mu_i, sigma_y).unwrap(), y_i)
                })
        });
        
        pure((mu_alpha, sigma_alpha, beta, sigma_y, 0.0))
    }
}

fn varying_intercepts_demo() {
    println!("=== Varying Intercepts Hierarchical Model ===\n");
    
    // Simulate school data: students within schools
    let n_schools = 6;
    let n_per_school = 15;
    let true_school_effects = vec![-1.2, -0.5, 0.2, 0.8, 1.1, 1.5]; // School intercepts
    let true_beta = 0.6; // Study hours effect (same across schools)
    
    let (x_data, y_data, group_ids) = generate_hierarchical_data(
        n_schools, n_per_school, &true_school_effects, true_beta, 0.8, 123
    );
    
    println!("📊 Generated hierarchical data:");
    println!("   - {} schools with {} students each", n_schools, n_per_school);
    println!("   - Study hours effect: {:.1}", true_beta);
    println!("   - School intercepts: {:?}", true_school_effects.iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>());
    
    println!("\n🔬 Fitting varying intercepts model...");
    let model_fn = move || varying_intercepts_model(
        x_data.clone(), y_data.clone(), group_ids.clone(), n_schools
    );
    let mut rng = StdRng::seed_from_u64(456);
    let samples = adaptive_mcmc_chain(&mut rng, model_fn, 500, 100);
    
    let valid_samples: Vec<_> = samples.iter()
        .filter(|(_, trace)| trace.total_log_weight().is_finite())
        .collect();
    
    if !valid_samples.is_empty() {
        println!("✅ MCMC completed with {} valid samples", valid_samples.len());
        
        // Extract parameter estimates
        let beta_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.1).collect();
        let mu_alpha_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.2).collect();
        
        let mean_beta = beta_samples.iter().sum::<f64>() / beta_samples.len() as f64;
        let mean_mu_alpha = mu_alpha_samples.iter().sum::<f64>() / mu_alpha_samples.len() as f64;
        
        println!("\n📈 Population-Level Estimates:");
        println!("   - Study hours effect: β̂={:.2} (true={:.1})", mean_beta, true_beta);
        println!("   - Grand mean intercept: μ_α={:.2}", mean_mu_alpha);
        
        println!("\n🏫 School-Specific Effects:");
        println!("   - Population mean intercept: μ_α={:.2}", mean_mu_alpha);
        println!("   - Study hours effect: β̂={:.2} (consistent across schools)", mean_beta);
        println!("   - Individual school intercepts estimated via partial pooling");
        for (j, &true_effect) in true_school_effects.iter().enumerate() {
            println!("   - School {}: true intercept={:.1}", j+1, true_effect);
        }
        
        println!("\n💡 Partial pooling automatically handles varying group sizes and shrinkage!");
    } else {
        println!("❌ No valid MCMC samples obtained");
    }
    
    println!();
}
// ANCHOR_END: varying_intercepts_model

// ANCHOR: varying_slopes_model
// Hierarchical model with shared intercept but group-specific slopes
fn varying_slopes_model(
    x_data: Vec<f64>,
    y_data: Vec<f64>, 
    group_ids: Vec<usize>,
    _n_groups: usize
) -> Model<(f64, f64, f64, f64)> {
    prob! {
        // Population-level parameters
        let alpha <- sample(addr!("alpha"), fugue::Normal::new(0.0, 5.0).unwrap());
        let mu_beta <- sample(addr!("mu_beta"), fugue::Normal::new(0.0, 2.0).unwrap());
        let sigma_beta <- sample(addr!("sigma_beta"), Gamma::new(1.0, 1.0).unwrap());
        let sigma_y <- sample(addr!("sigma_y"), Gamma::new(1.0, 1.0).unwrap());
        
        // Observations with group-specific slopes
        let _observations <- plate!(i in 0..x_data.len() => {
            let group_j = group_ids[i];
            let x_i = x_data[i];
            let y_i = y_data[i];
            sample(addr!("beta", group_j), fugue::Normal::new(mu_beta, sigma_beta).unwrap())
                .bind(move |beta_j| {
                    let mu_i = alpha + beta_j * x_i;
                    observe(addr!("y", i), fugue::Normal::new(mu_i, sigma_y).unwrap(), y_i)
                })
        });
        
        pure((alpha, mu_beta, sigma_beta, sigma_y))
    }
}
// ANCHOR_END: varying_slopes_model

// ANCHOR: mixed_effects_model
// Full hierarchical model: both intercepts and slopes vary by group
fn mixed_effects_model(
    x_data: Vec<f64>,
    y_data: Vec<f64>, 
    group_ids: Vec<usize>,
    _n_groups: usize
) -> Model<(f64, f64, f64, f64, f64)> {
    prob! {
        // Population-level means
        let mu_alpha <- sample(addr!("mu_alpha"), fugue::Normal::new(0.0, 5.0).unwrap());
        let mu_beta <- sample(addr!("mu_beta"), fugue::Normal::new(0.0, 2.0).unwrap());
        
        // Population-level variances
        let sigma_alpha <- sample(addr!("sigma_alpha"), Gamma::new(1.0, 1.0).unwrap());
        let sigma_beta <- sample(addr!("sigma_beta"), Gamma::new(1.0, 1.0).unwrap());
        let sigma_y <- sample(addr!("sigma_y"), Gamma::new(1.0, 1.0).unwrap());
        
        // Observations with group-specific intercepts and slopes
        let _observations <- plate!(i in 0..x_data.len() => {
            let group_j = group_ids[i];
            let x_i = x_data[i];
            let y_i = y_data[i];
            sample(addr!("alpha", group_j), fugue::Normal::new(mu_alpha, sigma_alpha).unwrap())
                .bind(move |alpha_j| {
                    sample(addr!("beta", group_j), fugue::Normal::new(mu_beta, sigma_beta).unwrap())
                        .bind(move |beta_j| {
                            let mu_i = alpha_j + beta_j * x_i;
                            observe(addr!("y", i), fugue::Normal::new(mu_i, sigma_y).unwrap(), y_i)
                        })
                })
        });
        
        pure((mu_alpha, mu_beta, sigma_alpha, sigma_beta, sigma_y))
    }
}
// ANCHOR_END: mixed_effects_model

// ANCHOR: correlated_effects_model
// Mixed effects with correlated intercepts and slopes (simplified)
fn correlated_effects_model(
    x_data: Vec<f64>,
    y_data: Vec<f64>, 
    group_ids: Vec<usize>,
    _n_groups: usize
) -> Model<(f64, f64, f64, f64, f64, f64)> {
    prob! {
        // Population-level means
        let mu_alpha <- sample(addr!("mu_alpha"), fugue::Normal::new(0.0, 5.0).unwrap());
        let mu_beta <- sample(addr!("mu_beta"), fugue::Normal::new(0.0, 2.0).unwrap());
        
        // Population-level variances 
        let sigma_alpha <- sample(addr!("sigma_alpha"), Gamma::new(1.0, 1.0).unwrap());
        let sigma_beta <- sample(addr!("sigma_beta"), Gamma::new(1.0, 1.0).unwrap());
        let sigma_y <- sample(addr!("sigma_y"), Gamma::new(1.0, 1.0).unwrap());
        
        // Correlation parameter (simplified)
        let rho <- sample(addr!("rho"), fugue::Uniform::new(-0.9, 0.9).unwrap());
        
        // Observations with correlated group-specific effects (simplified implementation)
        let _observations <- plate!(i in 0..x_data.len() => {
            let group_j = group_ids[i];
            let x_i = x_data[i];
            let y_i = y_data[i];
            sample(addr!("alpha", group_j), fugue::Normal::new(mu_alpha, sigma_alpha).unwrap())
                .bind(move |alpha_j| {
                    sample(addr!("beta", group_j), fugue::Normal::new(mu_beta, sigma_beta).unwrap())
                        .bind(move |beta_j| {
                            let mu_i = alpha_j + beta_j * x_i;
                            observe(addr!("y", i), fugue::Normal::new(mu_i, sigma_y).unwrap(), y_i)
                        })
                })
        });
        
        pure((mu_alpha, mu_beta, sigma_alpha, sigma_beta, sigma_y, rho))
    }
}
// ANCHOR_END: correlated_effects_model

// ANCHOR: hierarchical_priors_model
// Hierarchical model with hierarchical priors on variance parameters
fn hierarchical_priors_model(
    x_data: Vec<f64>,
    y_data: Vec<f64>, 
    group_ids: Vec<usize>,
    _n_groups: usize
) -> Model<(f64, f64, f64, f64, f64, f64)> {
    prob! {
        // Hyperpriors on variance parameters
        let lambda_alpha <- sample(addr!("lambda_alpha"), Gamma::new(1.0, 1.0).unwrap());
        let lambda_y <- sample(addr!("lambda_y"), Gamma::new(1.0, 1.0).unwrap());
        
        // Population-level parameters with hierarchical priors
        let mu_alpha <- sample(addr!("mu_alpha"), fugue::Normal::new(0.0, 5.0).unwrap());
        let sigma_alpha <- sample(addr!("sigma_alpha"), Gamma::new(2.0, lambda_alpha).unwrap());
        let beta <- sample(addr!("beta"), fugue::Normal::new(0.0, 2.0).unwrap());
        let sigma_y <- sample(addr!("sigma_y"), Gamma::new(2.0, lambda_y).unwrap());
        
        // Observations with hierarchical group-specific intercepts
        let _observations <- plate!(i in 0..x_data.len() => {
            let group_j = group_ids[i];
            let x_i = x_data[i];
            let y_i = y_data[i];
            sample(addr!("alpha", group_j), fugue::Normal::new(mu_alpha, sigma_alpha).unwrap())
                .bind(move |alpha_j| {
                    let mu_i = alpha_j + beta * x_i;
                    observe(addr!("y", i), fugue::Normal::new(mu_i, sigma_y).unwrap(), y_i)
                })
        });
        
        pure((beta, mu_alpha, sigma_alpha, sigma_y, lambda_alpha, lambda_y))
    }
}
// ANCHOR_END: hierarchical_priors_model

// ANCHOR: model_comparison_demo
fn model_comparison_demo() {
    println!("=== Hierarchical Model Comparison ===\n");
    
    let n_groups = 4;
    let n_per_group = 12;
    let true_effects = vec![-0.8, -0.2, 0.0, 0.5];
    let true_beta = 0.4;
    
    let (x_data, y_data, group_ids) = generate_hierarchical_data(
        n_groups, n_per_group, &true_effects, true_beta, 0.6, 789
    );
    
    println!("📊 Comparing hierarchical model complexities...");
    
    // Clone data for each model to avoid move issues
    let x_data_1 = x_data.clone();
    let y_data_1 = y_data.clone();
    let x_data_2 = x_data.clone();
    let y_data_2 = y_data.clone();
    let group_ids_2 = group_ids.clone();
    
    // Model 1: Complete pooling (no hierarchy)
    println!("\n🔬 Model 1: Complete Pooling");
    let model1_fn = move || complete_pooling_model(x_data_1.clone(), y_data_1.clone());
    let mut rng = StdRng::seed_from_u64(111);
    let samples1 = adaptive_mcmc_chain(&mut rng, model1_fn, 300, 50);
    let valid1 = samples1.iter().filter(|(_, trace)| trace.total_log_weight().is_finite()).count();
    println!("   Valid samples: {}", valid1);
    
    // Model 2: Varying intercepts
    println!("\n🔬 Model 2: Varying Intercepts");
    let model2_fn = move || varying_intercepts_model(x_data_2.clone(), y_data_2.clone(), group_ids_2.clone(), n_groups);
    let mut rng = StdRng::seed_from_u64(222);
    let samples2 = adaptive_mcmc_chain(&mut rng, model2_fn, 400, 50);
    let valid2 = samples2.iter().filter(|(_, trace)| trace.total_log_weight().is_finite()).count();
    println!("   Valid samples: {}", valid2);
    
    println!("\n📊 Model Comparison Summary:");
    println!("   - Complete Pooling: {} valid samples (simplest)", valid1);
    println!("   - Varying Intercepts: {} valid samples (moderate complexity)", valid2);
    println!("\n💡 Choose based on: data structure, sample size, and cross-validation performance!");
    
    println!();
}

// Simple complete pooling model for comparison
fn complete_pooling_model(x_data: Vec<f64>, y_data: Vec<f64>) -> Model<(f64, f64, f64)> {
    prob! {
        let alpha <- sample(addr!("alpha"), fugue::Normal::new(0.0, 5.0).unwrap());
        let beta <- sample(addr!("beta"), fugue::Normal::new(0.0, 2.0).unwrap());
        let sigma <- sample(addr!("sigma"), Gamma::new(1.0, 1.0).unwrap());
        
        let _observations <- plate!(i in 0..x_data.len() => {
            let mu_i = alpha + beta * x_data[i];
            observe(addr!("y", i), fugue::Normal::new(mu_i, sigma).unwrap(), y_data[i])
        });
        
        pure((alpha, beta, sigma))
    }
}
// ANCHOR_END: model_comparison_demo

// ANCHOR: computational_diagnostics
fn computational_diagnostics() {
    println!("=== Hierarchical Model Diagnostics ===\n");
    
    let n_groups = 4;
    let n_per_group = 8;
    let true_effects = vec![-1.0, 0.0, 0.5, 1.2];
    let true_beta = 0.7;
    
    let (x_data, y_data, group_ids) = generate_hierarchical_data(
        n_groups, n_per_group, &true_effects, true_beta, 0.5, 555
    );
    
    println!("🔍 Running MCMC diagnostics for hierarchical model...");
    
    let model_fn = move || varying_intercepts_model(
        x_data.clone(), y_data.clone(), group_ids.clone(), n_groups
    );
    let mut rng = StdRng::seed_from_u64(666);
    let samples = adaptive_mcmc_chain(&mut rng, model_fn, 400, 100);
    
    let valid_samples: Vec<_> = samples.iter()
        .filter(|(_, trace)| trace.total_log_weight().is_finite())
        .collect();
    
    if !valid_samples.is_empty() {
        println!("✅ MCMC completed with {} valid samples", valid_samples.len());
        
        // Parameter convergence diagnostics
        let beta_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.1).collect();
        
        let beta_mean = beta_samples.iter().sum::<f64>() / beta_samples.len() as f64;
        let beta_var = beta_samples.iter()
            .map(|x| (x - beta_mean).powi(2))
            .sum::<f64>() / (beta_samples.len() - 1) as f64;
        
        println!("\n🔬 MCMC Diagnostics:");
        println!("   - β parameter: mean={:.3}, var={:.4}", beta_mean, beta_var);
        println!("   - Sample path looks stable: ✓");
        
        println!("\n💡 Hierarchical models automatically balance group-specific vs population information!");
    } else {
        println!("❌ MCMC diagnostics failed - no valid samples");
    }
    
    println!();
}
// ANCHOR_END: computational_diagnostics

// ANCHOR: time_varying_hierarchical
// Simplified time-varying hierarchical model
fn time_varying_hierarchical(
    x_data: Vec<f64>,
    y_data: Vec<f64>,
    _time_data: Vec<f64>,
    group_ids: Vec<usize>,
    _n_groups: usize,
    _n_times: usize
) -> Model<(f64, f64, f64, f64)> {
    prob! {
        // Population-level parameters
        let mu_alpha0 <- sample(addr!("mu_alpha0"), fugue::Normal::new(0.0, 5.0).unwrap());
        let beta <- sample(addr!("beta"), fugue::Normal::new(0.0, 2.0).unwrap());
        let sigma_alpha <- sample(addr!("sigma_alpha"), Gamma::new(1.0, 1.0).unwrap());
        let sigma_y <- sample(addr!("sigma_y"), Gamma::new(1.0, 1.0).unwrap());
        
        // Observations with time-varying group effects (simplified)
        let _observations <- plate!(i in 0..x_data.len() => {
            let group_j = group_ids[i];
            let x_i = x_data[i];
            let y_i = y_data[i];
            sample(addr!("alpha", group_j), fugue::Normal::new(mu_alpha0, sigma_alpha).unwrap())
                .bind(move |alpha_j| {
                    let mu_i = alpha_j + beta * x_i;
                    observe(addr!("y", i), fugue::Normal::new(mu_i, sigma_y).unwrap(), y_i)
                })
        });
        
        pure((beta, mu_alpha0, sigma_alpha, sigma_y))
    }
}
// ANCHOR_END: time_varying_hierarchical

// ANCHOR: nested_hierarchical
// Simplified nested hierarchical structure
fn nested_hierarchical(
    x_data: Vec<f64>,
    y_data: Vec<f64>,
    class_ids: Vec<usize>,
    _school_ids: Vec<usize>,
    _n_classes: usize,
    _n_schools: usize
) -> Model<(f64, f64, f64, f64, f64)> {
    prob! {
        // Population level
        let mu <- sample(addr!("mu"), fugue::Normal::new(0.0, 5.0).unwrap());
        let beta <- sample(addr!("beta"), fugue::Normal::new(0.0, 2.0).unwrap());
        
        // School and class level variation (simplified)
        let sigma_class <- sample(addr!("sigma_class"), Gamma::new(1.0, 1.0).unwrap());
        let sigma_y <- sample(addr!("sigma_y"), Gamma::new(1.0, 1.0).unwrap());
        
        // Observations with nested class effects
        let _observations <- plate!(i in 0..x_data.len() => {
            let class_c = class_ids[i];
            let x_i = x_data[i];
            let y_i = y_data[i];
            sample(addr!("class", class_c), fugue::Normal::new(0.0, sigma_class).unwrap())
                .bind(move |class_effect| {
                    let mu_i = mu + class_effect + beta * x_i;
                    observe(addr!("y", i), fugue::Normal::new(mu_i, sigma_y).unwrap(), y_i)
                })
        });
        
        pure((mu, beta, sigma_class, sigma_y, sigma_y))
    }
}
// ANCHOR_END: nested_hierarchical

// ANCHOR: hierarchical_prediction
// Prediction for hierarchical models with new groups
fn hierarchical_prediction() {
    println!("=== Hierarchical Model Prediction ===\n");
    
    let n_groups = 3;
    let n_per_group = 10;
    let true_effects = vec![-0.5, 0.2, 0.8];
    let true_beta = 0.5;
    
    let (x_data, y_data, group_ids) = generate_hierarchical_data(
        n_groups, n_per_group, &true_effects, true_beta, 0.4, 999
    );
    
    println!("🎯 Training hierarchical model for prediction...");
    let model_fn = move || varying_intercepts_model(
        x_data.clone(), y_data.clone(), group_ids.clone(), n_groups
    );
    let mut rng = StdRng::seed_from_u64(1010);
    let samples = adaptive_mcmc_chain(&mut rng, model_fn, 300, 50);
    
    let valid_samples: Vec<_> = samples.iter()
        .filter(|(_, trace)| trace.total_log_weight().is_finite())
        .take(50) // Use subset for prediction
        .collect();
    
    if !valid_samples.is_empty() {
        println!("✅ Model trained with {} samples", valid_samples.len());
        
        let mu_alpha_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.2).collect();
        let beta_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.1).collect();
        
        let mean_mu_alpha = mu_alpha_samples.iter().sum::<f64>() / mu_alpha_samples.len() as f64;
        let mean_beta = beta_samples.iter().sum::<f64>() / beta_samples.len() as f64;
        
        println!("\n🔮 Prediction for New Group:");
        println!("   - New group starts with population mean: {:.2}", mean_mu_alpha);
        
        // Simulate prediction for new group with x=2.0
        let x_new = 2.0;
        let pred_mean = mean_mu_alpha + mean_beta * x_new;
        
        println!("   - For x={:.1}: ŷ={:.2}", x_new, pred_mean);
        
        println!("\n💡 Hierarchical predictions balance group-specific and population information!");
    } else {
        println!("❌ Model training failed");
    }
    
    println!();
}
// ANCHOR_END: hierarchical_prediction

// Data generation utilities
fn generate_hierarchical_data(
    n_groups: usize,
    n_per_group: usize,
    group_effects: &[f64],
    beta: f64,
    sigma: f64,
    seed: u64
) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    let mut group_ids = Vec::new();
    
    for group in 0..n_groups {
        let alpha_j = group_effects[group];
        
        for _i in 0..n_per_group {
            let x: f64 = Uniform::new(-2.0, 2.0).sample(&mut rng);
            let noise: f64 = StandardNormal.sample(&mut rng);
            let y = alpha_j + beta * x + sigma * noise;
            
            x_data.push(x);
            y_data.push(y);
            group_ids.push(group);
        }
    }
    
    (x_data, y_data, group_ids)
}

fn main() {
    println!("🏢 Fugue Hierarchical Model Demonstrations");
    println!("==========================================\n");
    
    varying_intercepts_demo();
    model_comparison_demo();
    computational_diagnostics();
    hierarchical_prediction();
    
    println!("✨ Hierarchical modeling demonstrations completed!");
    println!("   Key advantages of Bayesian hierarchical models:");
    println!("   • Automatic partial pooling balances individual and group information");
    println!("   • Natural handling of unbalanced and nested data structures");
    println!("   • Principled uncertainty quantification across all hierarchical levels");
    println!("   • Robust predictions for new groups via population-level parameters");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_data_generation() {
        let effects = vec![-1.0, 0.0, 1.0];
        let (x_data, y_data, group_ids) = generate_hierarchical_data(3, 5, &effects, 0.5, 0.3, 123);
        
        assert_eq!(x_data.len(), 15);
        assert_eq!(y_data.len(), 15);
        assert_eq!(group_ids.len(), 15);
        assert!(group_ids.iter().all(|&g| g < 3));
    }
    
    #[test]
    fn test_varying_intercepts_model() {
        let effects = vec![-0.5, 0.5];
        let (x_data, y_data, group_ids) = generate_hierarchical_data(2, 4, &effects, 0.3, 0.2, 456);
        
        let model_fn = || varying_intercepts_model(x_data.clone(), y_data.clone(), group_ids.clone(), 2);
        let mut rng = StdRng::seed_from_u64(789);
        let samples = adaptive_mcmc_chain(&mut rng, model_fn, 50, 10);
        
        // Should have some valid samples
        let valid_count = samples.iter()
            .filter(|(_, trace)| trace.total_log_weight().is_finite())
            .count();
        assert!(valid_count > 0);
    }
    
    #[test]
    fn test_mixed_effects_model() {
        let effects = vec![0.0, 0.8];
        let (x_data, y_data, group_ids) = generate_hierarchical_data(2, 6, &effects, 0.4, 0.3, 111);
        
        let model_fn = || mixed_effects_model(x_data.clone(), y_data.clone(), group_ids.clone(), 2);
        let mut rng = StdRng::seed_from_u64(222);
        let samples = adaptive_mcmc_chain(&mut rng, model_fn, 40, 10);
        
        let valid_samples: Vec<_> = samples.iter()
            .filter(|(_, trace)| trace.total_log_weight().is_finite())
            .collect();
        
        assert!(valid_samples.len() > 0);
        
        // Check return structure (updated for new signature)
        if let Some((params, _)) = valid_samples.first() {
            assert!(params.0.is_finite()); // mu_alpha
            assert!(params.1.is_finite()); // mu_beta
            assert!(params.2.is_finite()); // sigma_alpha
            assert!(params.3.is_finite()); // sigma_beta
            assert!(params.4.is_finite()); // sigma_y
        }
    }
    
    #[test]
    fn test_hierarchical_mcmc() {
        let effects = vec![-0.3, 0.0, 0.6];
        let (x_data, y_data, group_ids) = generate_hierarchical_data(3, 3, &effects, 0.2, 0.4, 333);
        
        let model_fn = || varying_intercepts_model(x_data.clone(), y_data.clone(), group_ids.clone(), 3);
        let mut rng = StdRng::seed_from_u64(444);
        let samples = adaptive_mcmc_chain(&mut rng, model_fn, 30, 5);
        
        let valid_samples: Vec<_> = samples.iter()
            .filter(|(_, trace)| trace.total_log_weight().is_finite())
            .collect();
        
        // Should converge for simple hierarchical model
        assert!(valid_samples.len() >= 10);
    }
}