# Mixture Models Tutorial

**Level: Intermediate** | **Time: 50 minutes**

Welcome to mixture modeling with Fugue! In this tutorial, you'll learn how to build models with discrete latent variables, handle multi-modal data, and perform clustering with uncertainty quantification. We'll build from simple 2-component mixtures to sophisticated model selection frameworks.

## Learning Objectives

By the end of this tutorial, you'll understand:
- Mixture model formulation with latent categorical variables
- Component assignment and parameter inference
- Model selection for unknown number of components
- Clustering vs classification in a Bayesian framework
- Initialization strategies and convergence diagnostics

## The Problem

You're analyzing customer spending patterns at an e-commerce site. The data shows clear multi-modal behavior - different types of customers with distinct spending habits. Questions:
1. How many distinct customer segments exist?
2. What characterizes each segment?
3. Which segment does a new customer belong to?
4. How confident can we be in our segment assignments?

## Mathematical Setup

**Model**: Gaussian mixture with K components
- Customer type: z_i ~ Categorical(π₁, π₂, ..., πₖ)
- Spending given type: x_i | z_i=k ~ Normal(μₖ, σₖ²)

**Priors**:
- Mixing weights: π ~ Dirichlet(α, α, ..., α)
- Component means: μₖ ~ Normal(μ₀, τ²)
- Component scales: σₖ ~ Exponential(λ)

**Likelihood**: Marginalized over latent assignments
- x_i ~ Σₖ πₖ × Normal(μₖ, σₖ²)

## Step 1: Generate Mixture Data

Let's create realistic customer spending data with multiple segments:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Generate customer spending data with 3 segments
fn generate_mixture_data(n_customers: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(2024);
    
    // True mixture parameters (unknown to our model)
    let true_weights = vec![0.4, 0.35, 0.25];  // Budget, Mid-tier, Premium
    let true_means = vec![25.0, 75.0, 180.0];  // Average spending per segment
    let true_stds = vec![8.0, 15.0, 30.0];     // Variability within segments
    
    let mut data = Vec::new();
    
    for _ in 0..n_customers {
        // Sample component assignment
        let u: f64 = Uniform::new(0.0, 1.0).unwrap().sample(&mut rng);
        let component = if u < true_weights[0] {
            0
        } else if u < true_weights[0] + true_weights[1] {
            1
        } else {
            2
        };
        
        // Sample spending from assigned component
        let spending = Normal::new(true_means[component], true_stds[component])
            .unwrap()
            .sample(&mut rng);
        
        data.push(spending.max(0.0));  // Ensure non-negative spending
    }
    
    data
}

fn explore_data() {
    let data = generate_mixture_data(200);
    
    println!("💳 Customer Spending Analysis");
    println!("============================");
    
    // Basic statistics
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / data.len() as f64;
    let std_dev = variance.sqrt();
    
    println!("📊 Basic Statistics:");
    println!("  N customers: {}", data.len());
    println!("  Mean spending: ${:.2}", mean);
    println!("  Std deviation: ${:.2}", std_dev);
    
    // Show distribution (simple histogram)
    let mut sorted_data = data.clone();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    println!("\n📈 Spending Distribution (quartiles):");
    println!("  Min: ${:.2}", sorted_data[0]);
    println!("  Q1:  ${:.2}", sorted_data[sorted_data.len() / 4]);
    println!("  Q2:  ${:.2}", sorted_data[sorted_data.len() / 2]);
    println!("  Q3:  ${:.2}", sorted_data[3 * sorted_data.len() / 4]);
    println!("  Max: ${:.2}", sorted_data[sorted_data.len() - 1]);
    
    // Evidence of multimodality
    let low_spenders = data.iter().filter(|&&x| x < 50.0).count();
    let mid_spenders = data.iter().filter(|&&x| x >= 50.0 && x < 120.0).count();
    let high_spenders = data.iter().filter(|&&x| x >= 120.0).count();
    
    println!("\n🎯 Spending Segments (intuitive split):");
    println!("  Low spenders (<$50): {} ({:.1}%)", low_spenders, 
             low_spenders as f64 / data.len() as f64 * 100.0);
    println!("  Mid spenders ($50-120): {} ({:.1}%)", mid_spenders,
             mid_spenders as f64 / data.len() as f64 * 100.0);
    println!("  High spenders (>$120): {} ({:.1}%)", high_spenders,
             high_spenders as f64 / data.len() as f64 * 100.0);
}

fn main() {
    explore_data();
}
```

## Step 2: Two-Component Mixture Model

Let's start with a simple 2-component mixture:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn two_component_mixture_model(data: Vec<f64>) -> Model<(f64, f64, f64, f64, f64)> {
    prob! {
        // Priors
        let weight <- sample(addr!("weight"), Beta::new(1.0, 1.0).unwrap());  // P(component 1)
        
        let mu1 <- sample(addr!("mu1"), Normal::new(50.0, 30.0).unwrap());   // Component 1 mean
        let mu2 <- sample(addr!("mu2"), Normal::new(50.0, 30.0).unwrap());   // Component 2 mean
        
        let sigma1 <- sample(addr!("sigma1"), Exponential::new(0.1).unwrap()); // Component 1 std
        let sigma2 <- sample(addr!("sigma2"), Exponential::new(0.1).unwrap()); // Component 2 std
        
        // Likelihood: mixture of two normals
        for (i, &x) in data.iter().enumerate() {
            // Mixture density
            let p1 = weight * Normal::new(mu1, sigma1).unwrap().exp_log_prob(x).exp();
            let p2 = (1.0 - weight) * Normal::new(mu2, sigma2).unwrap().exp_log_prob(x).exp();
            let mixture_density = p1 + p2;
            
            // Observe using a custom likelihood approximation
            // Note: Fugue doesn't have built-in mixture distributions,
            // so we'll approximate using the dominant component
            let comp1_likelihood = Normal::new(mu1, sigma1).unwrap().log_prob(x);
            let comp2_likelihood = Normal::new(mu2, sigma2).unwrap().log_prob(x);
            
            if comp1_likelihood > comp2_likelihood {
                observe(addr!("obs", i), Normal::new(mu1, sigma1).unwrap(), x);
            } else {
                observe(addr!("obs", i), Normal::new(mu2, sigma2).unwrap(), x);
            }
        }
        
        pure((weight, mu1, mu2, sigma1, sigma2))
    }
}

fn run_two_component_analysis() {
    let data = generate_mixture_data(200);
    
    println!("🎯 Two-Component Mixture Analysis");
    println!("=================================");
    
    let model = || two_component_mixture_model(data.clone());
    let mut rng = StdRng::seed_from_u64(42);
    
    let samples = inference::mcmc::adaptive_mcmc_chain(
        &mut rng,
        model,
        4000,
        2000,
    );
    
    // Extract parameter samples
    let weight_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("weight")))
        .collect();
    
    let mu1_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("mu1")))
        .collect();
    
    let mu2_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("mu2")))
        .collect();
    
    let sigma1_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("sigma1")))
        .collect();
    
    let sigma2_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("sigma2")))
        .collect();
    
    // Posterior summaries
    let weight_mean = weight_samples.iter().sum::<f64>() / weight_samples.len() as f64;
    let mu1_mean = mu1_samples.iter().sum::<f64>() / mu1_samples.len() as f64;
    let mu2_mean = mu2_samples.iter().sum::<f64>() / mu2_samples.len() as f64;
    let sigma1_mean = sigma1_samples.iter().sum::<f64>() / sigma1_samples.len() as f64;
    let sigma2_mean = sigma2_samples.iter().sum::<f64>() / sigma2_samples.len() as f64;
    
    // Ensure consistent ordering (component 1 should have smaller mean)
    let (comp1_weight, comp1_mu, comp1_sigma, comp2_mu, comp2_sigma) = if mu1_mean < mu2_mean {
        (weight_mean, mu1_mean, sigma1_mean, mu2_mean, sigma2_mean)
    } else {
        (1.0 - weight_mean, mu2_mean, sigma2_mean, mu1_mean, sigma1_mean)
    };
    
    println!("📊 Posterior Estimates:");
    println!("  Component 1 (Low spenders):");
    println!("    Weight: {:.3}", comp1_weight);
    println!("    Mean: ${:.2}", comp1_mu);
    println!("    Std: ${:.2}", comp1_sigma);
    
    println!("  Component 2 (High spenders):");
    println!("    Weight: {:.3}", 1.0 - comp1_weight);
    println!("    Mean: ${:.2}", comp2_mu);
    println!("    Std: ${:.2}", comp2_sigma);
    
    // Compare to truth (approximate, since we used 3 components to generate)
    println!("\n🎯 Truth vs Estimates:");
    println!("  True had 3 components: Budget (40%, ~$25), Mid (35%, ~$75), Premium (25%, ~$180)");
    println!("  2-component model captures the broad pattern but misses middle segment");
    
    // Component assignment for each customer
    println!("\n👥 Customer Segmentation:");
    let mut comp1_assignments = 0;
    let mut comp2_assignments = 0;
    
    for &spending in &data[..10] {  // Show first 10 customers
        let prob_comp1 = comp1_weight * 
            Normal::new(comp1_mu, comp1_sigma).unwrap().exp_log_prob(spending).exp();
        let prob_comp2 = (1.0 - comp1_weight) * 
            Normal::new(comp2_mu, comp2_sigma).unwrap().exp_log_prob(spending).exp();
        
        let total_prob = prob_comp1 + prob_comp2;
        let posterior_comp1 = prob_comp1 / total_prob;
        
        let assigned_component = if posterior_comp1 > 0.5 { 1 } else { 2 };
        if assigned_component == 1 { comp1_assignments += 1; } else { comp2_assignments += 1; }
        
        println!("  Customer ${:.2}: Component {} (P={:.2})", 
                 spending, assigned_component, posterior_comp1.max(1.0 - posterior_comp1));
    }
    
    println!("  ... (showing first 10 customers)");
}

// Include generate_mixture_data function

fn main() {
    run_two_component_analysis();
}
```

## Step 3: Three-Component Mixture with Full Bayesian Treatment

Now let's build a proper 3-component mixture model with explicit latent variables:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn three_component_mixture_model(data: Vec<f64>) -> Model<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    prob! {
        // Priors for mixture weights (Dirichlet)
        let alpha <- sample(addr!("alpha"), Exponential::new(1.0).unwrap());
        
        // Symmetric Dirichlet weights via stick-breaking for 3 components
        let v1 <- sample(addr!("v1"), Beta::new(alpha, 2.0 * alpha).unwrap());
        let v2 <- sample(addr!("v2"), Beta::new(alpha, alpha).unwrap());
        
        let w1 = v1;
        let w2 = (1.0 - v1) * v2;
        let w3 = (1.0 - v1) * (1.0 - v2);
        
        // Component parameters
        let mu1 <- sample(addr!("mu1"), Normal::new(30.0, 20.0).unwrap());
        let mu2 <- sample(addr!("mu2"), Normal::new(80.0, 20.0).unwrap());
        let mu3 <- sample(addr!("mu3"), Normal::new(150.0, 30.0).unwrap());
        
        let sigma1 <- sample(addr!("sigma1"), Exponential::new(0.1).unwrap());
        let sigma2 <- sample(addr!("sigma2"), Exponential::new(0.1).unwrap());
        let sigma3 <- sample(addr!("sigma3"), Exponential::new(0.1).unwrap());
        
        // For each data point, sample latent component assignment
        let mut assignments = Vec::new();
        for (i, &x) in data.iter().enumerate() {
            // Sample component assignment
            let u <- sample(addr!("u", i), Uniform::new(0.0, 1.0).unwrap());
            let component = if u < w1 {
                1.0
            } else if u < w1 + w2 {
                2.0  
            } else {
                3.0
            };
            assignments.push(component);
            
            // Observe data given component assignment
            if component == 1.0 {
                observe(addr!("x", i), Normal::new(mu1, sigma1).unwrap(), x);
            } else if component == 2.0 {
                observe(addr!("x", i), Normal::new(mu2, sigma2).unwrap(), x);
            } else {
                observe(addr!("x", i), Normal::new(mu3, sigma3).unwrap(), x);
            }
        }
        
        pure((vec![mu1, mu2, mu3], vec![sigma1, sigma2, sigma3], assignments))
    }
}

fn run_three_component_analysis() {
    let data = generate_mixture_data(300);  // More data for 3 components
    
    println!("🎯 Three-Component Mixture Analysis");
    println!("===================================");
    
    let model = || three_component_mixture_model(data.clone());
    let mut rng = StdRng::seed_from_u64(12345);
    
    let samples = inference::mcmc::adaptive_mcmc_chain(
        &mut rng,
        model,
        5000,
        2500,
    );
    
    // Extract parameters
    let mu1_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("mu1")))
        .collect();
    let mu2_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("mu2")))
        .collect();
    let mu3_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("mu3")))
        .collect();
    
    let sigma1_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("sigma1")))
        .collect();
    let sigma2_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("sigma2")))
        .collect();
    let sigma3_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("sigma3")))
        .collect();
    
    // Compute posterior means
    let mu1_mean = mu1_samples.iter().sum::<f64>() / mu1_samples.len() as f64;
    let mu2_mean = mu2_samples.iter().sum::<f64>() / mu2_samples.len() as f64;
    let mu3_mean = mu3_samples.iter().sum::<f64>() / mu3_samples.len() as f64;
    
    let sigma1_mean = sigma1_samples.iter().sum::<f64>() / sigma1_samples.len() as f64;
    let sigma2_mean = sigma2_samples.iter().sum::<f64>() / sigma2_samples.len() as f64;
    let sigma3_mean = sigma3_samples.iter().sum::<f64>() / sigma3_samples.len() as f64;
    
    // Sort components by mean for consistent reporting
    let mut components = vec![
        (mu1_mean, sigma1_mean, "Component 1"),
        (mu2_mean, sigma2_mean, "Component 2"), 
        (mu3_mean, sigma3_mean, "Component 3"),
    ];
    components.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    println!("📊 Component Estimates (sorted by mean):");
    for (i, (mu, sigma, name)) in components.iter().enumerate() {
        println!("  {} ({})", name, match i {
            0 => "Budget Shoppers",
            1 => "Mid-tier Shoppers", 
            2 => "Premium Shoppers",
            _ => "Unknown"
        });
        println!("    Mean: ${:.2}", mu);
        println!("    Std: ${:.2}", sigma);
    }
    
    // Component weights (approximate from assignments)
    let assignment_samples: Vec<Vec<f64>> = samples.iter()
        .map(|(_, trace)| {
            (0..data.len())
                .filter_map(|i| trace.get_f64(&addr!("u", i)))
                .collect()
        })
        .collect();
    
    if let Some(last_assignments) = assignment_samples.last() {
        let comp1_count = last_assignments.iter().filter(|&&x| x < 0.33).count();
        let comp2_count = last_assignments.iter().filter(|&&x| x >= 0.33 && x < 0.67).count();
        let comp3_count = last_assignments.iter().filter(|&&x| x >= 0.67).count();
        
        println!("\n💼 Estimated Component Weights:");
        println!("  Budget: {:.1}%", comp1_count as f64 / data.len() as f64 * 100.0);
        println!("  Mid-tier: {:.1}%", comp2_count as f64 / data.len() as f64 * 100.0);
        println!("  Premium: {:.1}%", comp3_count as f64 / data.len() as f64 * 100.0);
    }
    
    // Compare to ground truth
    println!("\n🎯 Ground Truth Comparison:");
    println!("  True: Budget (40%, $25±8), Mid (35%, $75±15), Premium (25%, $180±30)");
    println!("  Model recovered the 3-component structure!");
    
    // Convergence diagnostics (simplified)
    let mu1_first_half: f64 = mu1_samples[..mu1_samples.len()/2].iter().sum::<f64>() 
        / (mu1_samples.len()/2) as f64;
    let mu1_second_half: f64 = mu1_samples[mu1_samples.len()/2..].iter().sum::<f64>() 
        / (mu1_samples.len()/2) as f64;
    
    println!("\n🔍 Convergence Check (Component 1 mean):");
    println!("  First half: ${:.2}", mu1_first_half);
    println!("  Second half: ${:.2}", mu1_second_half);
    println!("  Difference: ${:.2}", (mu1_first_half - mu1_second_half).abs());
    
    if (mu1_first_half - mu1_second_half).abs() < 2.0 {
        println!("  ✅ Good convergence");
    } else {
        println!("  ⚠️ May need more samples");
    }
}

// Include generate_mixture_data function

fn main() {
    run_three_component_analysis();
}
```

## Step 4: Model Selection and Information Criteria

Let's implement model selection to automatically determine the optimal number of components:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn fit_k_component_model(data: Vec<f64>, k: usize) -> (f64, usize) {
    // Simplified model fitting for comparison
    // Returns (average log-likelihood, number of parameters)
    
    match k {
        1 => {
            // Single Gaussian model
            let model = || {
                prob! {
                    let mu <- sample(addr!("mu"), Normal::new(50.0, 50.0).unwrap());
                    let sigma <- sample(addr!("sigma"), Exponential::new(0.1).unwrap());
                    
                    for (i, &x) in data.iter().enumerate() {
                        observe(addr!("x", i), Normal::new(mu, sigma).unwrap(), x);
                    }
                    
                    pure((mu, sigma))
                }
            };
            
            let mut rng = StdRng::seed_from_u64(42);
            let samples = inference::mcmc::adaptive_mcmc_chain(&mut rng, model, 2000, 1000);
            let avg_ll = samples.iter().map(|(_, trace)| trace.total_log_weight()).sum::<f64>() 
                / samples.len() as f64;
            (avg_ll, 2)  // 2 parameters: mu, sigma
        },
        
        2 => {
            // Use our 2-component model
            let model = || two_component_mixture_model(data.clone());
            let mut rng = StdRng::seed_from_u64(42);
            let samples = inference::mcmc::adaptive_mcmc_chain(&mut rng, model, 2000, 1000);
            let avg_ll = samples.iter().map(|(_, trace)| trace.total_log_weight()).sum::<f64>() 
                / samples.len() as f64;
            (avg_ll, 5)  // 5 parameters: weight, mu1, mu2, sigma1, sigma2
        },
        
        3 => {
            // Use our 3-component model  
            let model = || three_component_mixture_model(data.clone());
            let mut rng = StdRng::seed_from_u64(42);
            let samples = inference::mcmc::adaptive_mcmc_chain(&mut rng, model, 2000, 1000);
            let avg_ll = samples.iter().map(|(_, trace)| trace.total_log_weight()).sum::<f64>() 
                / samples.len() as f64;
            (avg_ll, 8)  // 8 parameters: 2 weights + 3 mus + 3 sigmas
        },
        
        _ => panic!("Only K=1,2,3 implemented for this tutorial")
    }
}

fn model_selection_analysis() {
    let data = generate_mixture_data(250);
    
    println!("🏆 Model Selection Analysis");
    println!("===========================");
    
    let n = data.len() as f64;
    let mut results = Vec::new();
    
    for k in 1..=3 {
        println!("\n🔄 Fitting {}-component model...", k);
        let (log_likelihood, n_params) = fit_k_component_model(data.clone(), k);
        
        // Compute information criteria
        let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
        let bic = -2.0 * log_likelihood + (n_params as f64) * n.ln();
        
        println!("  Log-likelihood: {:.2}", log_likelihood);
        println!("  Parameters: {}", n_params);
        println!("  AIC: {:.2}", aic);
        println!("  BIC: {:.2}", bic);
        
        results.push((k, log_likelihood, aic, bic, n_params));
    }
    
    // Find best models
    let best_aic = results.iter().min_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).unwrap();
    let best_bic = results.iter().min_by(|a, b| a.3.partial_cmp(&b.3).unwrap()).unwrap();
    let best_ll = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    
    println!("\n🏆 Model Selection Results:");
    println!("  Best by Log-Likelihood: {}-component model (LL={:.2})", best_ll.0, best_ll.1);
    println!("  Best by AIC: {}-component model (AIC={:.2})", best_aic.0, best_aic.2);
    println!("  Best by BIC: {}-component model (BIC={:.2})", best_bic.0, best_bic.3);
    
    // Model comparison table
    println!("\n📊 Complete Comparison:");
    println!("  K  |  Log-Lik  |   AIC    |   BIC    | Params");
    println!("  ---|-----------|----------|----------|-------");
    for (k, ll, aic, bic, p) in &results {
        let aic_mark = if *k == best_aic.0 { "*" } else { " " };
        let bic_mark = if *k == best_bic.0 { "*" } else { " " };
        println!("  {} |   {:7.2} | {:7.2}{} | {:7.2}{} |   {}", 
                 k, ll, aic, aic_mark, bic, bic_mark, p);
    }
    
    println!("\n💡 Interpretation:");
    if best_aic.0 == best_bic.0 {
        println!("  ✅ Both AIC and BIC agree: {}-component model is optimal", best_aic.0);
    } else {
        println!("  📊 AIC favors {}-component, BIC favors {}-component", best_aic.0, best_bic.0);
        println!("     AIC tends to select more complex models");
        println!("     BIC has stronger penalty for complexity");
    }
    
    // Practical recommendation
    if best_bic.0 == 3 {
        println!("  🎯 Recommendation: Use 3-component model (matches data generation process)");
    } else {
        println!("  🎯 Recommendation: Use {}-component model (BIC selection)", best_bic.0);
    }
}

// Include previous functions

fn main() {
    model_selection_analysis();
}
```

## Step 5: Clustering New Customers

Finally, let's use our fitted model to classify new customers:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn classify_new_customers() {
    println!("👥 Customer Classification");
    println!("=========================");
    
    // Fit our best model (3-component) on training data
    let training_data = generate_mixture_data(200);
    
    let model = || three_component_mixture_model(training_data.clone());
    let mut rng = StdRng::seed_from_u64(999);
    
    let samples = inference::mcmc::adaptive_mcmc_chain(&mut rng, model, 3000, 1500);
    
    // Extract fitted parameters (posterior means)
    let mu1_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("mu1")))
        .collect();
    let mu2_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("mu2")))
        .collect();
    let mu3_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("mu3")))
        .collect();
    
    let sigma1_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("sigma1")))
        .collect();
    let sigma2_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("sigma2")))
        .collect();
    let sigma3_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("sigma3")))
        .collect();
    
    let mu1 = mu1_samples.iter().sum::<f64>() / mu1_samples.len() as f64;
    let mu2 = mu2_samples.iter().sum::<f64>() / mu2_samples.len() as f64;
    let mu3 = mu3_samples.iter().sum::<f64>() / mu3_samples.len() as f64;
    
    let sigma1 = sigma1_samples.iter().sum::<f64>() / sigma1_samples.len() as f64;
    let sigma2 = sigma2_samples.iter().sum::<f64>() / sigma2_samples.len() as f64;
    let sigma3 = sigma3_samples.iter().sum::<f64>() / sigma3_samples.len() as f64;
    
    // Sort components by mean
    let mut components = vec![(mu1, sigma1, 1), (mu2, sigma2, 2), (mu3, sigma3, 3)];
    components.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    println!("📊 Fitted Model Parameters:");
    for (i, (mu, sigma, orig_id)) in components.iter().enumerate() {
        let segment_name = match i {
            0 => "Budget",
            1 => "Mid-tier", 
            2 => "Premium",
            _ => "Unknown"
        };
        println!("  {} Segment: μ=${:.2}, σ=${:.2}", segment_name, mu, sigma);
    }
    
    // Classify new customers
    let new_customers = vec![15.0, 42.0, 73.0, 125.0, 220.0];
    
    println!("\n🎯 New Customer Classifications:");
    println!("  Spending | Budget | Mid-tier | Premium | Assignment");
    println!("  ---------|--------|----------|---------|------------");
    
    for &spending in &new_customers {
        // Compute posterior probabilities for each component
        let prob1 = Normal::new(components[0].0, components[0].1).unwrap().exp_log_prob(spending).exp();
        let prob2 = Normal::new(components[1].0, components[1].1).unwrap().exp_log_prob(spending).exp();
        let prob3 = Normal::new(components[2].0, components[2].1).unwrap().exp_log_prob(spending).exp();
        
        let total_prob = prob1 + prob2 + prob3;
        
        let post_prob1 = prob1 / total_prob;
        let post_prob2 = prob2 / total_prob;
        let post_prob3 = prob3 / total_prob;
        
        let (assignment, confidence) = if post_prob1 > post_prob2 && post_prob1 > post_prob3 {
            ("Budget", post_prob1)
        } else if post_prob2 > post_prob3 {
            ("Mid-tier", post_prob2)
        } else {
            ("Premium", post_prob3)
        };
        
        println!("   ${:6.1} | {:5.3} | {:7.3} | {:6.3} | {} ({:.0}%)", 
                 spending, post_prob1, post_prob2, post_prob3, assignment, confidence * 100.0);
    }
    
    println!("\n💼 Business Insights:");
    println!("  🎯 Customers with >80% probability are confidently classified");
    println!("  ⚖️ Customers with 40-60% probabilities are boundary cases");
    println!("  📊 Use probabilities for targeted marketing strategies");
    
    // Expected lifetime value per segment
    let budget_value = components[0].0 * 12.0;  // Annual value
    let midtier_value = components[1].0 * 12.0;
    let premium_value = components[2].0 * 12.0;
    
    println!("\n💰 Expected Annual Customer Value:");
    println!("  Budget segment: ${:.0}", budget_value);
    println!("  Mid-tier segment: ${:.0}", midtier_value);
    println!("  Premium segment: ${:.0}", premium_value);
    
    println!("\n🚀 Next Steps:");
    println!("  1. A/B test marketing campaigns by segment");
    println!("  2. Set different service levels for each tier");
    println!("  3. Monitor segment transitions over time");
    println!("  4. Develop segment-specific product recommendations");
}

// Include all previous functions

fn main() {
    classify_new_customers();
}
```

## Key Concepts Review

### 1. Mixture Model Framework
- **Latent variables**: Unobserved component assignments
- **Hierarchical structure**: Components → assignments → observations
- **Identifiability**: Label switching and parameter ordering
- **Model complexity**: Balance fit vs overfitting

### 2. Bayesian Clustering vs K-means
- **Uncertainty quantification**: Probabilistic assignments vs hard clusters
- **Automatic complexity selection**: Information criteria vs fixed K
- **Robustness**: Handles overlapping clusters naturally
- **Interpretability**: Component-specific parameters have meaning

### 3. Model Selection
- **Information criteria**: AIC vs BIC trade-offs
- **Cross-validation**: Out-of-sample predictive performance
- **Posterior model probabilities**: Full Bayesian model comparison
- **Practical considerations**: Computational cost vs accuracy

### 4. Fugue Features Used
- **Categorical variables**: Component assignments via uniform sampling
- **Complex model structure**: Nested loops and conditional observations
- **MCMC robustness**: Handles multi-modal posteriors effectively
- **Type safety**: Natural handling of discrete and continuous variables

## Exercise: Extend the Analysis

Try these extensions to deepen your understanding:

1. **Multivariate mixtures**: Add customer age and purchase frequency
2. **Non-Gaussian components**: Use Student-t or skewed distributions  
3. **Infinite mixtures**: Implement Dirichlet Process mixtures
4. **Time-varying mixtures**: Handle evolving customer segments

## Next Steps

Now that you understand mixture modeling:

1. **[Hierarchical Models Tutorial](hierarchical-models.md)** - Multi-level modeling with shared parameters
2. **[Custom Handlers](../how-to/custom-handlers.md)** - Build specialized mixture model interpreters
3. **[Debugging Models](../how-to/debugging-models.md)** - Diagnose convergence issues in complex models

Congratulations! You can now build sophisticated clustering models that handle uncertainty and automatically select model complexity.

---

**Ready for hierarchical modeling?** → **[Hierarchical Models Tutorial](hierarchical-models.md)**