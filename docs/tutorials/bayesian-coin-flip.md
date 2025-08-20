# Bayesian Coin Flip Tutorial

**Level: Beginner** | **Time: 30 minutes**

Welcome to your first complete Bayesian analysis with Fugue! In this tutorial, we'll estimate the bias of a coin using Bayesian inference. You'll learn fundamental concepts while building a complete, working probabilistic program.

## Learning Objectives

By the end of this tutorial, you'll understand:
- How to formulate a Bayesian model
- The relationship between priors, likelihoods, and posteriors
- How to use conjugate priors for exact solutions
- How to run MCMC inference in Fugue
- How to analyze and interpret results

## The Problem

You have a coin that might be biased. You flip it 10 times and observe 7 heads. Questions:
1. What's the most likely bias of the coin?
2. How confident can we be in our estimate?
3. What's the probability the coin is fair (50% heads)?

We'll answer these questions using Bayesian inference.

## Mathematical Setup

**Prior**: We start with a uniform belief about the coin's bias
- Bias ~ Beta(1, 1) [uniform on [0, 1]]

**Likelihood**: Given the bias, heads follow a binomial distribution
- Heads | Bias ~ Binomial(n=10, p=bias)

**Posterior**: After observing 7 heads out of 10 flips
- Bias | Heads=7 ~ Beta(1+7, 1+10-7) = Beta(8, 4)

This is a **conjugate analysis** - the Beta prior combined with a Binomial likelihood gives a Beta posterior.

## Step 1: Basic Model Implementation

Let's start with the simplest possible model:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn simple_coin_model(observed_heads: u64, total_flips: u64) -> Model<f64> {
    prob! {
        // Prior: uniform belief about coin bias
        let bias <- sample(addr!("bias"), Beta::new(1.0, 1.0).unwrap());
        
        // Likelihood: observe the data
        observe(
            addr!("heads"), 
            Binomial::new(total_flips, bias).unwrap(), 
            observed_heads
        );
        
        // Return the parameter we're interested in
        pure(bias)
    }
}

fn main() {
    println!("🪙 Bayesian Coin Flip Analysis");
    println!("================================");
    
    // Our data: 7 heads out of 10 flips
    let heads = 7;
    let flips = 10;
    
    let model = simple_coin_model(heads, flips);
    
    // Sample from the posterior
    println!("\n📊 Posterior Samples:");
    for i in 0..5 {
        let mut rng = StdRng::seed_from_u64(i);
        let (bias, trace) = runtime::handler::run(
            runtime::interpreters::PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model.clone(),
        );
        
        println!("Sample {}: bias = {:.3}, log prob = {:.4}", 
                 i + 1, bias, trace.total_log_weight());
    }
}
```

**Try it**: Save this as `src/main.rs` and run `cargo run`.

## Step 2: MCMC Inference

Prior sampling doesn't give us proper posterior samples. Let's use MCMC to get the true posterior:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn coin_model(observed_heads: u64, total_flips: u64) -> Model<f64> {
    prob! {
        let bias <- sample(addr!("bias"), Beta::new(1.0, 1.0).unwrap());
        observe(addr!("heads"), Binomial::new(total_flips, bias).unwrap(), observed_heads);
        pure(bias)
    }
}

fn run_mcmc_analysis() {
    let heads = 7;
    let flips = 10;
    let model = || coin_model(heads, flips);
    
    println!("🔗 Running MCMC...");
    
    let mut rng = StdRng::seed_from_u64(42);
    
    // Collect MCMC samples manually to understand the process
    let mut samples = Vec::new();
    
    // Start with a sample from the prior
    let (mut current_bias, mut current_trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model(),
    );
    
    let n_samples = 2000;
    let mut n_accepted = 0;
    
    for i in 0..n_samples {
        // Propose new state using Metropolis-Hastings
        let (new_bias, new_trace) = inference::mh::single_site_random_walk_mh(
            &mut rng,
            0.1,  // Step size - smaller for better acceptance
            model,
            &current_trace,
        );
        
        // The MH algorithm automatically handles acceptance/rejection
        // We just need to check if the trace changed
        if (new_trace.total_log_weight() - current_trace.total_log_weight()).abs() > 1e-10 
           || (new_bias - current_bias).abs() > 1e-10 {
            n_accepted += 1;
        }
        
        current_bias = new_bias;
        current_trace = new_trace;
        samples.push(current_bias);
        
        if i % 400 == 0 {
            println!("  Iteration {}: bias = {:.3}", i, current_bias);
        }
    }
    
    let acceptance_rate = n_accepted as f64 / n_samples as f64;
    println!("  Acceptance rate: {:.1}%", acceptance_rate * 100.0);
    
    analyze_samples(&samples, heads, flips);
}

fn analyze_samples(samples: &[f64], heads: u64, flips: u64) {
    // Remove burn-in (first 25% of samples)
    let burnin = samples.len() / 4;
    let posterior_samples = &samples[burnin..];
    
    // Compute posterior statistics
    let mean = posterior_samples.iter().sum::<f64>() / posterior_samples.len() as f64;
    let variance = posterior_samples.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (posterior_samples.len() - 1) as f64;
    let std_dev = variance.sqrt();
    
    // Compute credible interval (central 95%)
    let mut sorted_samples = posterior_samples.to_vec();
    sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ci_lower = sorted_samples[(0.025 * sorted_samples.len() as f64) as usize];
    let ci_upper = sorted_samples[(0.975 * sorted_samples.len() as f64) as usize];
    
    println!("\n📈 Posterior Analysis:");
    println!("  Data: {} heads out of {} flips", heads, flips);
    println!("  Sample size: {} (after burn-in)", posterior_samples.len());
    println!("  Posterior mean: {:.3}", mean);
    println!("  Posterior std: {:.3}", std_dev);
    println!("  95% credible interval: [{:.3}, {:.3}]", ci_lower, ci_upper);
    
    // Answer specific questions
    let prob_fair = posterior_samples.iter()
        .filter(|&&x| (x - 0.5).abs() < 0.05)  // Within 5% of fair
        .count() as f64 / posterior_samples.len() as f64;
    
    let prob_biased_towards_heads = posterior_samples.iter()
        .filter(|&&x| x > 0.5)
        .count() as f64 / posterior_samples.len() as f64;
    
    println!("\n🤔 Questions Answered:");
    println!("  Probability coin is fair (45%-55%): {:.1}%", prob_fair * 100.0);
    println!("  Probability biased toward heads: {:.1}%", prob_biased_towards_heads * 100.0);
}

fn main() {
    println!("🪙 Bayesian Coin Flip Analysis");
    println!("================================");
    run_mcmc_analysis();
}
```

## Step 3: Analytical Comparison

Since we're using conjugate priors, we can compute the exact posterior analytically. Let's compare:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn analytical_solution(heads: u64, flips: u64, prior_alpha: f64, prior_beta: f64) {
    // Conjugate update: Beta(α, β) + Binomial(n, k) → Beta(α + k, β + n - k)
    let posterior_alpha = prior_alpha + heads as f64;
    let posterior_beta = prior_beta + (flips - heads) as f64;
    
    // Beta distribution mean and variance
    let mean = posterior_alpha / (posterior_alpha + posterior_beta);
    let variance = (posterior_alpha * posterior_beta) / 
                   ((posterior_alpha + posterior_beta).powi(2) * (posterior_alpha + posterior_beta + 1.0));
    let std_dev = variance.sqrt();
    
    println!("\n🧮 Analytical Solution:");
    println!("  Prior: Beta({}, {})", prior_alpha, prior_beta);
    println!("  Posterior: Beta({}, {})", posterior_alpha, posterior_beta);
    println!("  Posterior mean: {:.3}", mean);
    println!("  Posterior std: {:.3}", std_dev);
    
    // Credible interval using Beta quantiles (approximation)
    // For exact values, you'd use a proper Beta quantile function
    let alpha = 0.05;  // 95% CI
    println!("  95% credible interval: approximately [{:.3}, {:.3}]", 
             mean - 1.96 * std_dev, mean + 1.96 * std_dev);
}

fn validation_experiment() {
    let heads = 7;
    let flips = 10;
    
    println!("🔬 Validation: MCMC vs Analytical");
    println!("==================================");
    
    // Analytical solution
    analytical_solution(heads, flips, 1.0, 1.0);
    
    // MCMC solution
    let model = || coin_model(heads, flips);
    let mut rng = StdRng::seed_from_u64(42);
    
    // Use Fugue's built-in adaptive MCMC for better results
    let samples = inference::mcmc::adaptive_mcmc_chain(
        &mut rng,
        model,
        5000,  // More samples for better accuracy
        2000,  // Longer warmup
    );
    
    // Extract bias values from traces
    let bias_samples: Vec<f64> = samples
        .iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("bias")))
        .collect();
    
    let mcmc_mean = bias_samples.iter().sum::<f64>() / bias_samples.len() as f64;
    let mcmc_variance = bias_samples.iter()
        .map(|x| (x - mcmc_mean).powi(2))
        .sum::<f64>() / (bias_samples.len() - 1) as f64;
    
    println!("\n📊 MCMC Results:");
    println!("  Sample size: {}", bias_samples.len());
    println!("  MCMC mean: {:.3}", mcmc_mean);
    println!("  MCMC std: {:.3}", mcmc_variance.sqrt());
    
    // Compare with analytical
    let analytical_mean = 8.0 / 12.0;  // (1+7) / (1+1+10)
    let difference = (mcmc_mean - analytical_mean).abs();
    
    println!("\n✅ Validation:");
    println!("  Analytical mean: {:.3}", analytical_mean);
    println!("  MCMC mean: {:.3}", mcmc_mean);
    println!("  Absolute difference: {:.4}", difference);
    
    if difference < 0.01 {
        println!("  🎉 MCMC matches analytical solution!");
    } else {
        println!("  ⚠️ MCMC differs from analytical solution");
    }
}

fn coin_model(observed_heads: u64, total_flips: u64) -> Model<f64> {
    prob! {
        let bias <- sample(addr!("bias"), Beta::new(1.0, 1.0).unwrap());
        observe(addr!("heads"), Binomial::new(total_flips, bias).unwrap(), observed_heads);
        pure(bias)
    }
}

fn main() {
    validation_experiment();
}
```

## Step 4: Exploring Different Scenarios

Let's explore how different data affects our conclusions:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn scenario_analysis() {
    println!("🎭 Scenario Analysis");
    println!("====================");
    
    let scenarios = vec![
        (0, 10, "Never heads"),
        (1, 10, "Rarely heads"),
        (3, 10, "Sometimes heads"),
        (5, 10, "Half heads (fair?)"),
        (7, 10, "Often heads"),
        (9, 10, "Almost always heads"),
        (10, 10, "Always heads"),
        (20, 40, "Many flips: half heads"),
        (28, 40, "Many flips: often heads"),
    ];
    
    for (heads, flips, description) in scenarios {
        analyze_scenario(heads, flips, description);
    }
}

fn analyze_scenario(heads: u64, flips: u64, description: &str) {
    let model = || coin_model(heads, flips);
    let mut rng = StdRng::seed_from_u64(42);
    
    // Quick MCMC run
    let samples = inference::mcmc::adaptive_mcmc_chain(
        &mut rng,
        model,
        1000,
        500,
    );
    
    let bias_samples: Vec<f64> = samples
        .iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("bias")))
        .collect();
    
    let mean = bias_samples.iter().sum::<f64>() / bias_samples.len() as f64;
    
    // Probability assessments
    let prob_fair = bias_samples.iter()
        .filter(|&&x| (x - 0.5).abs() < 0.1)
        .count() as f64 / bias_samples.len() as f64;
    
    let prob_heads_favored = bias_samples.iter()
        .filter(|&&x| x > 0.6)
        .count() as f64 / bias_samples.len() as f64;
    
    let prob_tails_favored = bias_samples.iter()
        .filter(|&&x| x < 0.4)
        .count() as f64 / bias_samples.len() as f64;
    
    println!("\n📋 {}: {}/{} heads", description, heads, flips);
    println!("   Estimated bias: {:.3}", mean);
    println!("   P(fair): {:.0}%, P(heads-biased): {:.0}%, P(tails-biased): {:.0}%", 
             prob_fair * 100.0, prob_heads_favored * 100.0, prob_tails_favored * 100.0);
}

fn coin_model(observed_heads: u64, total_flips: u64) -> Model<f64> {
    prob! {
        let bias <- sample(addr!("bias"), Beta::new(1.0, 1.0).unwrap());
        observe(addr!("heads"), Binomial::new(total_flips, bias).unwrap(), observed_heads);
        pure(bias)
    }
}

fn main() {
    scenario_analysis();
}
```

## Step 5: Advanced Analysis with Multiple Questions

Let's extend our model to answer more sophisticated questions:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Extended model that predicts future flips
fn predictive_coin_model(observed_heads: u64, total_flips: u64, future_flips: u64) -> Model<(f64, u64)> {
    prob! {
        // Infer bias from observed data
        let bias <- sample(addr!("bias"), Beta::new(1.0, 1.0).unwrap());
        observe(addr!("observed_heads"), Binomial::new(total_flips, bias).unwrap(), observed_heads);
        
        // Predict future outcomes
        let future_heads <- sample(addr!("future_heads"), Binomial::new(future_flips, bias).unwrap());
        
        pure((bias, future_heads))
    }
}

fn comprehensive_analysis() {
    println!("🔮 Comprehensive Coin Analysis");
    println!("==============================");
    
    let observed_heads = 7;
    let observed_flips = 10;
    let future_flips = 20;
    
    println!("📋 Setup:");
    println!("  Observed: {} heads in {} flips", observed_heads, observed_flips);
    println!("  Predicting: {} future flips", future_flips);
    
    let model = || predictive_coin_model(observed_heads, observed_flips, future_flips);
    let mut rng = StdRng::seed_from_u64(42);
    
    let samples = inference::mcmc::adaptive_mcmc_chain(
        &mut rng,
        model,
        3000,
        1500,
    );
    
    // Extract results
    let bias_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("bias")))
        .collect();
    
    let future_heads_samples: Vec<u64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_u64(&addr!("future_heads")))
        .collect();
    
    // Bias analysis
    let bias_mean = bias_samples.iter().sum::<f64>() / bias_samples.len() as f64;
    let mut sorted_bias = bias_samples.clone();
    sorted_bias.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let bias_median = sorted_bias[sorted_bias.len() / 2];
    
    println!("\n🎯 Bias Estimation:");
    println!("  Posterior mean: {:.3}", bias_mean);
    println!("  Posterior median: {:.3}", bias_median);
    
    // Future predictions
    let future_mean = future_heads_samples.iter().sum::<u64>() as f64 / future_heads_samples.len() as f64;
    
    // Prediction intervals
    let mut sorted_future = future_heads_samples.clone();
    sorted_future.sort();
    let pred_5th = sorted_future[(0.05 * sorted_future.len() as f64) as usize];
    let pred_95th = sorted_future[(0.95 * sorted_future.len() as f64) as usize];
    
    println!("\n🔮 Future Predictions ({} flips):", future_flips);
    println!("  Expected heads: {:.1}", future_mean);
    println!("  90% prediction interval: [{}, {}] heads", pred_5th, pred_95th);
    
    // Probability questions
    let prob_future_majority_heads = future_heads_samples.iter()
        .filter(|&&x| x > future_flips / 2)
        .count() as f64 / future_heads_samples.len() as f64;
    
    let prob_future_all_heads = future_heads_samples.iter()
        .filter(|&&x| x == future_flips)
        .count() as f64 / future_heads_samples.len() as f64;
    
    let prob_future_no_heads = future_heads_samples.iter()
        .filter(|&&x| x == 0)
        .count() as f64 / future_heads_samples.len() as f64;
    
    println!("\n❓ Prediction Probabilities:");
    println!("  P(majority heads): {:.1}%", prob_future_majority_heads * 100.0);
    println!("  P(all heads): {:.2}%", prob_future_all_heads * 100.0);
    println!("  P(no heads): {:.2}%", prob_future_no_heads * 100.0);
    
    // Model criticism: check if observed data is typical
    let simulated_heads: Vec<u64> = bias_samples.iter()
        .zip(samples.iter())
        .filter_map(|(&bias, (_, trace))| {
            // Simulate what we might have observed given this bias
            let mut sim_rng = StdRng::seed_from_u64(42);  // Deterministic for reproducibility
            let heads = Binomial::new(observed_flips, bias).unwrap().sample(&mut sim_rng);
            Some(heads)
        })
        .collect();
    
    let prob_observed_or_more_extreme = simulated_heads.iter()
        .filter(|&&x| (x as i64 - observed_heads as i64).abs() >= 
                     (observed_heads as i64 - (observed_flips as f64 * 0.5) as i64).abs())
        .count() as f64 / simulated_heads.len() as f64;
    
    println!("\n🔍 Model Criticism:");
    println!("  P(observing {} or more extreme | model): {:.2}%", 
             observed_heads, prob_observed_or_more_extreme * 100.0);
    
    if prob_observed_or_more_extreme > 0.05 {
        println!("  ✅ Observed data is consistent with model");
    } else {
        println!("  ⚠️ Observed data is unusual under this model");
    }
}

fn main() {
    comprehensive_analysis();
}
```

## Key Concepts Review

Let's solidify the key concepts from this tutorial:

### 1. Bayesian Framework
- **Prior**: What we believe before seeing data
- **Likelihood**: How probable the data is given our hypothesis
- **Posterior**: What we believe after seeing data (prior × likelihood)

### 2. Conjugate Analysis
- Beta + Binomial = Beta (convenient mathematical property)
- Allows exact analytical solutions
- MCMC confirms these analytical results

### 3. Practical Insights
- **More data = more precision**: 7/10 vs 28/40 heads
- **Extreme observations**: 0/10 or 10/10 heads strongly suggest bias
- **Fair coin hypothesis**: Can be tested probabilistically

### 4. Fugue Features Used
- **Type-safe distributions**: `Bernoulli` returns `bool`, `Binomial` returns `u64`
- **`prob!` macro**: Clean, readable model specification
- **MCMC inference**: `adaptive_mcmc_chain` for automatic tuning
- **Trace analysis**: Extract and analyze parameter samples

## Exercise: Extend the Analysis

Try these extensions to deepen your understanding:

1. **Different priors**: Use `Beta(2, 2)` (weakly favors fair) or `Beta(0.5, 0.5)` (Jeffrey's prior)
2. **Model comparison**: Compare fair coin vs biased coin models using model evidence
3. **Sequential updating**: Update beliefs as you observe more flips one by one
4. **Multiple coins**: Analyze several coins simultaneously with hierarchical modeling

## Next Steps

Now that you understand basic Bayesian inference:

1. **[Linear Regression Tutorial](linear-regression.md)** - Continuous parameters and multiple variables
2. **[Understanding Models](../getting-started/understanding-models.md)** - Deepen your model composition skills
3. **[Working with Distributions](../how-to/working-with-distributions.md)** - Master all distribution types

Congratulations! You've completed your first Bayesian analysis with Fugue. You now understand the core concepts of probabilistic programming and can apply them to real problems.

---

**Ready for more complex models?** → **[Linear Regression Tutorial](linear-regression.md)**