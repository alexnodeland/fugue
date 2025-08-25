# Advanced Inference Tutorial

**Level: Advanced** | **Time: 50 minutes**

Master sophisticated inference techniques in Fugue! This tutorial covers advanced diagnostics, specialized distributions, multi-chain analysis, and production-ready inference pipelines.

## Learning Objectives

By the end of this tutorial, you'll understand:

- Comprehensive convergence diagnostics (R-hat, ESS, Geweke)
- Multi-chain analysis and validation
- Specialized distributions (exponential, log-normal, hazard models)
- Production-ready inference pipelines
- Advanced numerical stability techniques
- Model validation against analytical solutions

## What Makes Inference "Advanced"?

Advanced inference goes beyond basic MCMC:

- **Rigorous diagnostics** - Ensure convergence and reliability
- **Multiple inference methods** - SMC, variational inference, ensemble methods
- **Numerical stability** - Handle challenging parameter spaces
- **Production deployment** - Robust, scalable, validated pipelines
- **Specialized domains** - Survival analysis, reliability, finance

## Part 1: Comprehensive Diagnostics

**Try it**: Run with `cargo run --example improved_gaussian_mean`

```rust
{{#include ../../../examples/improved_gaussian_mean.rs}}
```

### Advanced Diagnostic Techniques

The example demonstrates several sophisticated techniques:

#### 1. Multi-Chain Convergence (R-hat)
```rust
let r_hat = r_hat_f64(&chains, &addr!("mu"));
println!("R-hat: {:.4} (should be < 1.1)", r_hat);
```

**Interpretation**:
- R-hat < 1.01: Excellent convergence
- R-hat < 1.1: Good convergence  
- R-hat > 1.1: Poor convergence, run longer

#### 2. Effective Sample Size (ESS)
```rust
let ess = effective_sample_size_mcmc(&samples);
println!("ESS: {:.1}", ess);
```

**Rule of thumb**: ESS should be > 100 × number of chains for reliable inference.

#### 3. Geweke Diagnostic
```rust
let geweke = geweke_diagnostic(&samples);
println!("Geweke: {:.2}", geweke);
```

**Interpretation**: |Geweke| > 2.0 suggests convergence issues.

### Validation Framework

```rust
let validation = test_conjugate_normal_model(
    &mut rng,
    mcmc_test,
    ConjugateNormalConfig {
        prior_mu: 0.0,
        prior_sigma: 5.0,
        likelihood_sigma: 1.0,
        observation: 2.7,
        n_samples: 1000,
        n_warmup: 500,
    },
);

validation.print_summary();
if !validation.is_valid() {
    println!("WARNING: Validation failed");
}
```

**Key insight**: Always validate against known solutions when possible.

## Part 2: Specialized Distributions

### Exponential and Hazard Models

**Try it**: Run with `cargo run --example exponential_hazard`  

```rust
{{#include ../../../examples/exponential_hazard.rs}}
```

### Understanding Hazard Models

**Hazard function**: Instantaneous failure rate at time t
- λ(t) = λ (constant hazard - exponential distribution)
- λ(t) = λt (increasing hazard - Weibull with shape > 1)
- λ(t) = λ/t (decreasing hazard - Weibull with shape < 1)

### Applications

**Survival analysis**: Time until event (death, failure, churn)
```rust
fn survival_model(event_times: Vec<f64>, censored: Vec<bool>) -> Model<f64> {
    prob! {
        let rate <- sample(addr!("rate"), LogNormal::new(0.0, 1.0).unwrap());
        
        for (i, (&time, &is_censored)) in event_times.iter().zip(censored.iter()).enumerate() {
            if is_censored {
                // Right-censored: survived past time
                let survival_prob = (-rate * time).exp();
                factor(addr!("survival", i), survival_prob.ln());
            } else {
                // Observed event at time
                observe(addr!("event", i), Exponential::new(rate).unwrap(), time);
            }
        }
        
        pure(rate)
    }
}
```

**Reliability engineering**: Component lifetime analysis
```rust
fn reliability_model(failure_times: Vec<f64>) -> Model<(f64, f64)> {
    prob! {
        // Weibull parameters
        let shape <- sample(addr!("shape"), LogNormal::new(0.0, 0.5).unwrap());
        let scale <- sample(addr!("scale"), LogNormal::new(2.0, 0.5).unwrap());
        
        for (i, &time) in failure_times.iter().enumerate() {
            // Weibull likelihood
            let logp = (shape - 1.0) * time.ln() - (time / scale).powf(shape) + shape.ln() - shape * scale.ln();
            factor(addr!("failure", i), logp);
        }
        
        pure((shape, scale))
    }
}
```

## Part 3: Multi-Method Inference Comparison

### SMC vs MCMC vs Variational Inference

```rust
fn comprehensive_inference_comparison(model_fn: ModelFn) {
    println!("🔬 Inference Method Comparison");
    println!("=============================");
    
    let mut rng = StdRng::seed_from_u64(42);
    
    // 1. MCMC
    let start_time = std::time::Instant::now();
    let mcmc_samples = adaptive_mcmc_chain(&mut rng, model_fn, 2000, 1000);
    let mcmc_time = start_time.elapsed();
    
    let mcmc_estimates = extract_parameter_estimates(&mcmc_samples);
    
    // 2. SMC  
    let start_time = std::time::Instant::now();
    let smc_particles = adaptive_smc(&mut rng, 1000, model_fn, SMCConfig::default());
    let smc_time = start_time.elapsed();
    
    let smc_estimates = extract_particle_estimates(&smc_particles);
    
    // 3. Variational Inference (if available)
    let start_time = std::time::Instant::now();
    let vi_result = mean_field_vi(&mut rng, model_fn, VIConfig::default());
    let vi_time = start_time.elapsed();
    
    // Compare results
    println!("\n📊 Method Comparison:");
    println!("  MCMC: μ={:.3}, time={:.2}s", mcmc_estimates.mean, mcmc_time.as_secs_f64());
    println!("  SMC:  μ={:.3}, time={:.2}s", smc_estimates.mean, smc_time.as_secs_f64());
    println!("  VI:   μ={:.3}, time={:.2}s", vi_result.mean, vi_time.as_secs_f64());
    
    // Accuracy vs speed trade-offs
    if mcmc_time < smc_time && (mcmc_estimates.mean - smc_estimates.mean).abs() < 0.01 {
        println!("  ✅ MCMC is both faster and accurate for this model");
    }
}
```

## Part 4: Production-Ready Pipelines

### Robust Inference Pipeline

```rust
pub struct InferencePipeline {
    config: InferenceConfig,
    validators: Vec<Box<dyn ModelValidator>>,
    diagnostics: Vec<Box<dyn DiagnosticCheck>>,
}

impl InferencePipeline {
    pub fn run<T>(&self, model_fn: impl Fn() -> Model<T>) -> InferenceResult<T> {
        // 1. Pre-flight validation
        self.validate_model(&model_fn)?;
        
        // 2. Run inference with multiple chains
        let chains = self.run_multiple_chains(&model_fn)?;
        
        // 3. Convergence diagnostics
        let diagnostics = self.run_diagnostics(&chains)?;
        
        // 4. Post-processing
        let summary = self.summarize_results(&chains)?;
        
        Ok(InferenceResult {
            summary,
            diagnostics,
            chains,
            metadata: self.create_metadata(),
        })
    }
    
    fn validate_model<T>(&self, model_fn: &impl Fn() -> Model<T>) -> Result<(), InferenceError> {
        for validator in &self.validators {
            validator.validate(model_fn)?;
        }
        Ok(())
    }
    
    fn run_multiple_chains<T>(&self, model_fn: &impl Fn() -> Model<T>) -> Result<Vec<Chain<T>>, InferenceError> {
        let mut chains = Vec::new();
        
        for chain_id in 0..self.config.n_chains {
            let mut rng = StdRng::seed_from_u64(self.config.base_seed + chain_id as u64);
            
            let samples = adaptive_mcmc_chain(
                &mut rng,
                model_fn,
                self.config.n_samples,
                self.config.n_warmup,
            );
            
            chains.push(Chain::new(chain_id, samples));
        }
        
        Ok(chains)
    }
}
```

### Configuration Management

```rust
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    // Sampling parameters
    pub n_samples: usize,
    pub n_warmup: usize,
    pub n_chains: usize,
    pub base_seed: u64,
    
    // Convergence criteria
    pub r_hat_threshold: f64,
    pub min_ess: f64,
    pub max_runtime: std::time::Duration,
    
    // Diagnostic options
    pub check_convergence: bool,
    pub validate_against_prior: bool,
    pub posterior_predictive_checks: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            n_samples: 2000,
            n_warmup: 1000,
            n_chains: 4,
            base_seed: 42,
            r_hat_threshold: 1.1,
            min_ess: 400.0,  // 100 × n_chains
            max_runtime: std::time::Duration::from_secs(300),
            check_convergence: true,
            validate_against_prior: true,
            posterior_predictive_checks: true,
        }
    }
}
```

## Part 5: Numerical Stability Techniques

### Handling Challenging Parameter Spaces

```rust
fn numerically_stable_model(data: Vec<f64>) -> Model<(f64, f64)> {
    prob! {
        // Use log-normal for positive parameters to avoid boundary issues
        let log_sigma <- sample(addr!("log_sigma"), Normal::new(0.0, 1.0).unwrap());
        let sigma = log_sigma.exp();  // Always positive
        
        // Center and scale data for better numerical properties
        let data_mean = data.iter().sum::<f64>() / data.len() as f64;
        let centered_data: Vec<f64> = data.iter().map(|&x| x - data_mean).collect();
        
        let mu_centered <- sample(addr!("mu_centered"), Normal::new(0.0, 2.0).unwrap());
        
        // Likelihood with centered parameterization
        for (i, &x) in centered_data.iter().enumerate() {
            observe(addr!("obs", i), Normal::new(mu_centered, sigma).unwrap(), x);
        }
        
        // Transform back to original scale
        let mu = mu_centered + data_mean;
        
        pure((mu, sigma))
    }
}
```

### Advanced Parameterization Tricks

```rust
// Non-centered parameterization for hierarchical models
fn non_centered_hierarchical_model(groups: Vec<Vec<f64>>) -> Model<(f64, f64, Vec<f64>)> {
    prob! {
        // Hyperparameters
        let mu <- sample(addr!("mu"), Normal::new(0.0, 10.0).unwrap());
        let log_tau <- sample(addr!("log_tau"), Normal::new(0.0, 1.0).unwrap());
        let tau = log_tau.exp();
        
        // Non-centered group effects
        let mut group_effects = Vec::new();
        for (g, group_data) in groups.iter().enumerate() {
            let z <- sample(addr!("z", g), Normal::new(0.0, 1.0).unwrap());
            let group_effect = mu + tau * z;  // Non-centered transformation
            group_effects.push(group_effect);
            
            // Likelihood
            for (i, &y) in group_data.iter().enumerate() {
                observe(addr!("y", g, i), Normal::new(group_effect, 1.0).unwrap(), y);
            }
        }
        
        pure((mu, tau, group_effects))
    }
}
```

**Why non-centered helps**: Reduces correlation between hyperparameters and group effects, improving MCMC efficiency.

## Part 6: Advanced Model Comparison

### Information Criteria with Corrections

```rust
fn advanced_model_comparison(models: Vec<ModelFn>, data: Vec<f64>) -> ModelComparisonResult {
    let mut results = Vec::new();
    
    for (i, model_fn) in models.iter().enumerate() {
        // Fit model
        let mut rng = StdRng::seed_from_u64(42);
        let samples = adaptive_mcmc_chain(&mut rng, model_fn, 3000, 1500);
        
        // Compute WAIC (Watanabe-Akaike Information Criterion)
        let waic = compute_waic(&samples, &data);
        
        // Compute LOO-CV (Leave-One-Out Cross-Validation)
        let loo_cv = compute_loo_cv(&samples, model_fn, &data);
        
        // Posterior predictive p-value
        let pp_pvalue = posterior_predictive_check(&samples, model_fn, &data);
        
        results.push(ModelResult {
            model_id: i,
            waic,
            loo_cv,
            pp_pvalue,
            n_parameters: count_parameters(&samples),
        });
    }
    
    // Rank models
    results.sort_by(|a, b| a.waic.partial_cmp(&b.waic).unwrap());
    
    ModelComparisonResult { results }
}
```

### Cross-Validation for Model Selection

```rust
fn k_fold_cross_validation(model_fn: ModelFn, data: Vec<f64>, k: usize) -> f64 {
    let fold_size = data.len() / k;
    let mut total_log_score = 0.0;
    
    for fold in 0..k {
        // Create train/test split
        let test_start = fold * fold_size;
        let test_end = if fold == k - 1 { data.len() } else { (fold + 1) * fold_size };
        
        let train_data: Vec<f64> = data.iter()
            .enumerate()
            .filter(|(i, _)| *i < test_start || *i >= test_end)
            .map(|(_, &x)| x)
            .collect();
        
        let test_data = &data[test_start..test_end];
        
        // Fit on training data
        let mut rng = StdRng::seed_from_u64(42 + fold as u64);
        let train_model = || model_fn_with_data(model_fn, train_data.clone());
        let samples = adaptive_mcmc_chain(&mut rng, train_model, 1000, 500);
        
        // Evaluate on test data
        for &test_point in test_data {
            let log_predictive_density = compute_log_predictive_density(&samples, test_point);
            total_log_score += log_predictive_density;
        }
    }
    
    total_log_score / data.len() as f64
}
```

## Part 7: Specialized Inference Techniques

### Hamiltonian Monte Carlo (HMC)

```rust
// If HMC is available in Fugue
fn hmc_inference_example(model_fn: ModelFn) {
    let mut rng = StdRng::seed_from_u64(42);
    
    let hmc_config = HMCConfig {
        n_samples: 2000,
        n_warmup: 1000,
        step_size: 0.1,
        n_leapfrog_steps: 10,
        adapt_step_size: true,
        max_tree_depth: 10,
    };
    
    let samples = hamiltonian_monte_carlo(&mut rng, model_fn, hmc_config);
    
    // HMC typically has better convergence properties
    let acceptance_rate = compute_acceptance_rate(&samples);
    println!("HMC acceptance rate: {:.1}%", acceptance_rate * 100.0);
    
    // Should be much higher than random-walk Metropolis
    assert!(acceptance_rate > 0.6);
}
```

### Parallel Tempering

```rust
fn parallel_tempering_inference(model_fn: ModelFn, temperatures: Vec<f64>) {
    let mut chains = Vec::new();
    
    // Initialize chains at different temperatures
    for &temp in &temperatures {
        let tempered_model = || tempered_model_fn(model_fn, temp);
        chains.push(ChainState::new(tempered_model));
    }
    
    for iteration in 0..10000 {
        // Update each chain
        for chain in &mut chains {
            chain.update();
        }
        
        // Propose swaps between adjacent temperatures
        if iteration % 10 == 0 {
            for i in 0..chains.len() - 1 {
                let swap_prob = compute_swap_probability(&chains[i], &chains[i + 1]);
                if rand::random::<f64>() < swap_prob {
                    chains.swap(i, i + 1);
                }
            }
        }
    }
    
    // Use samples from the cold chain (temperature = 1.0)
    let cold_samples = &chains[0].samples;
}
```

## Best Practices for Advanced Inference

### 1. Always Run Multiple Chains
```rust
// ✅ Good - multiple independent chains
let config = InferenceConfig {
    n_chains: 4,
    n_samples: 2000,
    n_warmup: 1000,
    ..Default::default()
};

// ❌ Bad - single chain
let samples = single_chain_mcmc(model, 8000, 4000);
```

### 2. Use Appropriate Convergence Criteria  
```rust
// ✅ Good - comprehensive diagnostics
if r_hat < 1.01 && ess > 400.0 && geweke.abs() < 2.0 {
    println!("✅ Excellent convergence");
} else {
    println!("⚠️ Consider running longer");
}

// ❌ Bad - no diagnostics
let samples = mcmc_chain(model, 1000, 500);  // Hope it converged!
```

### 3. Validate Against Known Solutions
```rust
// ✅ Good - test against analytical solutions
let analytical_mean = compute_analytical_posterior_mean();
let mcmc_mean = compute_sample_mean(&samples);
assert!((analytical_mean - mcmc_mean).abs() < 0.01);

// ❌ Bad - no validation
let samples = mcmc_chain(model, n_samples, n_warmup);  // Trust blindly
```

### 4. Handle Numerical Issues Proactively
```rust
// ✅ Good - numerical stability
let log_sigma <- sample(addr!("log_sigma"), Normal::new(0.0, 1.0).unwrap());
let sigma = log_sigma.exp();  // Always positive

// ❌ Bad - can hit boundaries  
let sigma <- sample(addr!("sigma"), Exponential::new(1.0).unwrap());
// Can become very small, causing numerical issues
```

## Production Deployment Checklist

- ✅ **Multi-chain inference** with convergence diagnostics
- ✅ **Robust error handling** for numerical issues
- ✅ **Comprehensive logging** for debugging
- ✅ **Performance monitoring** and resource limits
- ✅ **Model validation** against test cases
- ✅ **Configurable parameters** for different use cases
- ✅ **Graceful degradation** when inference fails

## Next Steps

Now that you understand advanced inference:

1. **[Hierarchical Models](hierarchical-models.md)** - Apply advanced techniques to multi-level models
2. **[Mixture Models](mixture-models.md)** - Handle complex latent variable models
3. **Production deployment** - Scale inference to real applications

## Key Takeaways

- **Diagnostics are essential** - R-hat, ESS, and Geweke tests ensure reliability
- **Multi-method comparison** - Different algorithms have different strengths
- **Numerical stability matters** - Parameterization affects convergence
- **Production requires robustness** - Error handling, monitoring, validation
- **Validate when possible** - Compare against analytical solutions
- **Multiple chains always** - Essential for convergence assessment

Advanced inference techniques enable reliable, production-ready probabilistic programming!

---

**Ready to apply these techniques?** → **[Hierarchical Models Tutorial](hierarchical-models.md)**
