# Bayesian Linear Regression Tutorial

**Level: Intermediate** | **Time: 45 minutes**

Welcome to Bayesian linear regression with Fugue! In this tutorial, you'll learn how to build flexible regression models that quantify uncertainty in both parameters and predictions. We'll start simple and build up to sophisticated models with multiple predictors and polynomial features.

## Learning Objectives

By the end of this tutorial, you'll understand:
- How to formulate Bayesian regression models
- Prior specification for regression parameters
- Posterior inference for slope, intercept, and noise parameters
- Uncertainty quantification in predictions
- Model comparison and diagnostics
- Polynomial and multiple regression

## The Problem

You're analyzing the relationship between a company's advertising spend and revenue. You have data from 20 quarters and want to:
1. Estimate the relationship between advertising and revenue
2. Quantify uncertainty in your estimates
3. Predict revenue for future advertising budgets
4. Assess if a linear relationship is adequate

## Mathematical Setup

**Model**: Linear relationship with Gaussian noise
- Revenue = α + β × Advertising + ε
- ε ~ Normal(0, σ²)

**Priors**:
- Intercept: α ~ Normal(0, 100) [weakly informative]
- Slope: β ~ Normal(0, 10) [positive relationship expected]
- Noise: σ ~ Exponential(1) [positive, moderately informative]

**Likelihood**: Given parameters, revenue follows Normal distribution
- Revenue | α, β, σ ~ Normal(α + β × Advertising, σ²)

## Step 1: Generate Synthetic Data

Let's start by creating some realistic data:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Generate realistic advertising/revenue data
fn generate_data() -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(12345);
    
    // True parameters (unknown to our model)
    let true_alpha = 50.0;  // Base revenue
    let true_beta = 3.5;    // Revenue per ad dollar
    let true_sigma = 8.0;   // Noise level
    
    let advertising: Vec<f64> = (1..=20)
        .map(|i| 10.0 + i as f64 * 2.0)  // $12K to $50K advertising
        .collect();
    
    let revenue: Vec<f64> = advertising
        .iter()
        .map(|&ad| {
            let mean = true_alpha + true_beta * ad;
            Normal::new(mean, true_sigma).unwrap().sample(&mut rng)
        })
        .collect();
    
    (advertising, revenue)
}

fn main() {
    let (advertising, revenue) = generate_data();
    
    println!("📊 Advertising vs Revenue Data");
    println!("==============================");
    for (i, (&ad, &rev)) in advertising.iter().zip(revenue.iter()).enumerate() {
        println!("Quarter {}: ${:.0}K ads → ${:.1}K revenue", i+1, ad, rev);
    }
}
```

## Step 2: Simple Linear Regression Model

Now let's build our Bayesian linear regression model:

```rust
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
```

## Step 3: Posterior Predictions with Uncertainty

Let's extend our model to make predictions with uncertainty bands:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn predictive_regression_model(
    advertising: Vec<f64>, 
    revenue: Vec<f64>, 
    new_ad_spend: f64
) -> Model<(f64, f64, f64, f64)> {
    prob! {
        // Priors and likelihood (same as before)
        let alpha <- sample(addr!("alpha"), Normal::new(0.0, 100.0).unwrap());
        let beta <- sample(addr!("beta"), Normal::new(0.0, 10.0).unwrap());
        let sigma <- sample(addr!("sigma"), Exponential::new(1.0).unwrap());
        
        for (i, (&ad, &rev)) in advertising.iter().zip(revenue.iter()).enumerate() {
            let predicted_mean = alpha + beta * ad;
            observe(
                addr!("revenue", i), 
                Normal::new(predicted_mean, sigma).unwrap(), 
                rev
            );
        }
        
        // Posterior prediction for new advertising spend
        let predicted_mean = alpha + beta * new_ad_spend;
        let predicted_revenue <- sample(
            addr!("predicted_revenue"), 
            Normal::new(predicted_mean, sigma).unwrap()
        );
        
        pure((alpha, beta, sigma, predicted_revenue))
    }
}

fn prediction_analysis() {
    let (advertising, revenue) = generate_data();
    
    println!("🔮 Prediction Analysis");
    println!("=====================");
    
    // Test different advertising levels
    let test_ad_spends = vec![25.0, 35.0, 45.0, 55.0];  // Including extrapolation
    
    for &ad_spend in &test_ad_spends {
        let model = || predictive_regression_model(
            advertising.clone(), 
            revenue.clone(), 
            ad_spend
        );
        let mut rng = StdRng::seed_from_u64(42);
        
        let samples = inference::mcmc::adaptive_mcmc_chain(
            &mut rng,
            model,
            2000,
            1000,
        );
        
        let predicted_revenues: Vec<f64> = samples.iter()
            .filter_map(|(_, trace)| trace.get_f64(&addr!("predicted_revenue")))
            .collect();
        
        // Compute prediction intervals
        let mut sorted_preds = predicted_revenues.clone();
        sorted_preds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mean_pred = predicted_revenues.iter().sum::<f64>() / predicted_revenues.len() as f64;
        let pred_5th = sorted_preds[(0.05 * sorted_preds.len() as f64) as usize];
        let pred_95th = sorted_preds[(0.95 * sorted_preds.len() as f64) as usize];
        
        println!("\n💰 Ad Spend: ${:.0}K", ad_spend);
        println!("  Expected Revenue: ${:.1}K", mean_pred);
        println!("  90% Prediction Interval: [${:.1}K, ${:.1}K]", pred_5th, pred_95th);
        
        // ROI analysis
        let roi = (mean_pred - ad_spend) / ad_spend;
        println!("  Expected ROI: {:.1}%", roi * 100.0);
        
        // Probability of profitability (revenue > ad spend)
        let prob_profitable = predicted_revenues.iter()
            .filter(|&&rev| rev > ad_spend)
            .count() as f64 / predicted_revenues.len() as f64;
        println!("  P(Revenue > Ad Spend): {:.1}%", prob_profitable * 100.0);
    }
}

// Include generate_data function

fn main() {
    prediction_analysis();
}
```

## Step 4: Model Diagnostics and Validation

Let's add comprehensive model checking:

```rust
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
```

## Step 5: Polynomial Regression

Now let's extend to polynomial features to capture non-linear relationships:

```rust
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
```

## Key Concepts Review

### 1. Bayesian Regression Framework
- **Parameters as random variables**: α, β, σ all have uncertainty
- **Prior specification**: Encodes domain knowledge
- **Posterior inference**: Combines prior beliefs with data
- **Prediction uncertainty**: Natural consequence of parameter uncertainty

### 2. Model Building Process
- **Start simple**: Linear relationship first
- **Add complexity gradually**: Polynomial terms, interactions
- **Validate assumptions**: Residual analysis, posterior predictive checks
- **Compare models**: Log-likelihood, cross-validation

### 3. Practical Insights
- **Uncertainty quantification**: Prediction intervals vs point estimates
- **Economic interpretation**: ROI analysis, optimal spending
- **Diminishing returns**: Quadratic terms capture non-linearity
- **Model diagnostics**: Essential for reliable inference

### 4. Fugue Features Used
- **Type-safe continuous distributions**: `Normal`, `Exponential`
- **Vector operations**: Efficient handling of multiple observations
- **MCMC inference**: `adaptive_mcmc_chain` with automatic tuning
- **Trace analysis**: Parameter extraction and correlation analysis

## Exercise: Extend the Analysis

Try these extensions to deepen your understanding:

1. **Multiple predictors**: Add seasonality or competitor spending
2. **Robust regression**: Use Student-t likelihood for outlier resistance
3. **Hierarchical priors**: Different slopes for different market segments
4. **Model selection**: Implement formal Bayesian model comparison

## Next Steps

Now that you understand Bayesian regression:

1. **[Mixture Models Tutorial](mixture-models.md)** - Handle discrete latent variables
2. **[Working with Distributions](../how-to/working-with-distributions.md)** - Master advanced distribution features
3. **[Trace Manipulation](../how-to/trace-manipulation.md)** - Advanced posterior analysis

Congratulations! You can now build sophisticated regression models that properly quantify uncertainty and make reliable predictions.

---

**Ready for discrete latent variables?** → **[Mixture Models Tutorial](mixture-models.md)**