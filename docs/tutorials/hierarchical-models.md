# Hierarchical Models Tutorial

**Level: Advanced** | **Time: 60 minutes**

Welcome to hierarchical modeling with Fugue! In this tutorial, you'll learn how to build multi-level models with shared parameters, understand partial pooling and shrinkage effects, and handle complex grouped data structures. This is the most sophisticated modeling approach in the tutorial series.

## Learning Objectives

By the end of this tutorial, you'll understand:
- Hierarchical model structure and shared hyperparameters
- Partial pooling vs complete pooling vs no pooling
- Shrinkage effects and when they occur
- Group-level and population-level inference
- Hierarchical model diagnostics and interpretation

## The Problem

You're analyzing student test scores across multiple schools in a district. Each school has different numbers of students, teacher quality, and resources. You want to:

1. Estimate the true performance of each school
2. Account for varying sample sizes across schools
3. Borrow strength between similar schools
4. Predict performance for new schools
5. Identify schools that are truly exceptional vs lucky

This is a classic hierarchical modeling problem where schools are exchangeable units with shared population-level parameters.

## Mathematical Setup

**Hierarchical Structure**:
- Population level: μ, τ (overall mean and between-school variance)
- School level: θⱼ ~ Normal(μ, τ) for j = 1, ..., J
- Student level: yᵢⱼ ~ Normal(θⱼ, σ) for student i in school j

**Priors**:
- Population mean: μ ~ Normal(75, 15) [reasonable test score range]
- Between-school std: τ ~ Exponential(0.1) [allows substantial variation]
- Within-school std: σ ~ Exponential(0.1) [student-level noise]

**Key Insight**: Schools with small sample sizes will shrink toward the population mean more than schools with large samples.

## Step 1: Generate Hierarchical Data

Let's create realistic school test score data:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Debug, Clone)]
struct SchoolData {
    school_id: usize,
    name: String,
    n_students: usize,
    scores: Vec<f64>,
    sample_mean: f64,
}

fn generate_school_data() -> Vec<SchoolData> {
    let mut rng = StdRng::seed_from_u64(2024);
    
    // True population parameters (unknown to our model)
    let true_pop_mean = 78.0;     // Population average
    let true_between_std = 6.0;   // Between-school variation
    let true_within_std = 12.0;   // Within-school variation (student-level)
    
    let school_names = vec![
        "Lincoln Elementary", "Washington Middle", "Roosevelt High", 
        "Jefferson Academy", "Madison Prep", "Monroe Charter",
        "Adams Elementary", "Hamilton High", "Franklin Middle",
        "Wilson Academy", "Garfield Elementary", "Kennedy High"
    ];
    
    let mut schools = Vec::new();
    
    for (id, name) in school_names.into_iter().enumerate() {
        // Sample true school effect
        let true_school_effect = Normal::new(true_pop_mean, true_between_std)
            .unwrap()
            .sample(&mut rng);
        
        // Varying sample sizes (realistic for different schools)
        let n_students = match id {
            0..=2 => 15 + (id * 3),      // Small schools: 15-21 students
            3..=6 => 30 + (id * 5),      // Medium schools: 45-60 students  
            _ => 80 + (id * 10),         // Large schools: 80+ students
        };
        
        // Generate student scores for this school
        let scores: Vec<f64> = (0..n_students)
            .map(|_| {
                Normal::new(true_school_effect, true_within_std)
                    .unwrap()
                    .sample(&mut rng)
                    .max(0.0)  // Ensure non-negative scores
                    .min(100.0) // Cap at 100
            })
            .collect();
        
        let sample_mean = scores.iter().sum::<f64>() / scores.len() as f64;
        
        schools.push(SchoolData {
            school_id: id,
            name: name.to_string(),
            n_students,
            scores,
            sample_mean,
        });
    }
    
    schools
}

fn explore_school_data() {
    let schools = generate_school_data();
    
    println!("🏫 School District Test Score Analysis");
    println!("=====================================");
    
    let total_students: usize = schools.iter().map(|s| s.n_students).sum();
    let overall_mean = schools.iter()
        .flat_map(|s| &s.scores)
        .sum::<f64>() / total_students as f64;
    
    println!("📊 District Overview:");
    println!("  Number of schools: {}", schools.len());
    println!("  Total students: {}", total_students);
    println!("  Overall mean score: {:.2}", overall_mean);
    
    println!("\n🏫 School-by-School Breakdown:");
    println!("  School                | N   | Mean  | Min   | Max   ");
    println!("  ---------------------|-----|-------|-------|-------");
    
    for school in &schools {
        let min_score = school.scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_score = school.scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        println!("  {:20} | {:3} | {:5.1} | {:5.1} | {:5.1}", 
                 school.name, school.n_students, school.sample_mean, min_score, max_score);
    }
    
    // Show sampling variability issue
    let small_schools: Vec<&SchoolData> = schools.iter().filter(|s| s.n_students < 30).collect();
    let large_schools: Vec<&SchoolData> = schools.iter().filter(|s| s.n_students > 60).collect();
    
    if !small_schools.is_empty() && !large_schools.is_empty() {
        let small_var = small_schools.iter()
            .map(|s| s.sample_mean)
            .map(|x| (x - overall_mean).powi(2))
            .sum::<f64>() / small_schools.len() as f64;
        
        let large_var = large_schools.iter()
            .map(|s| s.sample_mean)
            .map(|x| (x - overall_mean).powi(2))
            .sum::<f64>() / large_schools.len() as f64;
        
        println!("\n📏 Sampling Variability:");
        println!("  Small schools (<30 students) variance: {:.2}", small_var);
        println!("  Large schools (>60 students) variance: {:.2}", large_var);
        
        if small_var > large_var * 1.5 {
            println!("  🎯 Small schools show higher variability - perfect case for hierarchical modeling!");
        }
    }
}

fn main() {
    explore_school_data();
}
```

## Step 2: No Pooling vs Complete Pooling Models

Let's start by comparing naive approaches to see why hierarchical modeling is needed:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// No pooling: each school analyzed independently
fn no_pooling_model(school_scores: Vec<f64>) -> Model<(f64, f64)> {
    prob! {
        let mu <- sample(addr!("mu"), Normal::new(75.0, 15.0).unwrap());
        let sigma <- sample(addr!("sigma"), Exponential::new(0.1).unwrap());
        
        for (i, &score) in school_scores.iter().enumerate() {
            observe(addr!("score", i), Normal::new(mu, sigma).unwrap(), score);
        }
        
        pure((mu, sigma))
    }
}

// Complete pooling: all schools treated as one population
fn complete_pooling_model(all_scores: Vec<f64>) -> Model<(f64, f64)> {
    prob! {
        let mu <- sample(addr!("mu"), Normal::new(75.0, 15.0).unwrap());
        let sigma <- sample(addr!("sigma"), Exponential::new(0.1).unwrap());
        
        for (i, &score) in all_scores.iter().enumerate() {
            observe(addr!("score", i), Normal::new(mu, sigma).unwrap(), score);
        }
        
        pure((mu, sigma))
    }
}

fn compare_pooling_approaches() {
    let schools = generate_school_data();
    
    println!("🔍 Pooling Approaches Comparison");
    println!("================================");
    
    // Complete pooling: treat all students as from same population
    let all_scores: Vec<f64> = schools.iter().flat_map(|s| &s.scores).cloned().collect();
    
    let pooled_model = || complete_pooling_model(all_scores.clone());
    let mut rng = StdRng::seed_from_u64(42);
    
    let pooled_samples = inference::mcmc::adaptive_mcmc_chain(&mut rng, pooled_model, 2000, 1000);
    
    let pooled_mu: Vec<f64> = pooled_samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("mu")))
        .collect();
    let pooled_mean = pooled_mu.iter().sum::<f64>() / pooled_mu.len() as f64;
    
    println!("🏊 Complete Pooling Results:");
    println!("  Estimated overall mean: {:.2}", pooled_mean);
    println!("  Treats all schools identically");
    
    // No pooling: analyze each school separately
    println!("\n🚫 No Pooling Results:");
    println!("  School               | Sample | No-Pool | Diff  | N");
    println!("  --------------------|--------|---------|-------|----");
    
    let mut no_pool_estimates = Vec::new();
    
    for school in &schools[..6] {  // Show first 6 schools
        let school_model = || no_pooling_model(school.scores.clone());
        let mut rng = StdRng::seed_from_u64(42 + school.school_id as u64);
        
        let school_samples = inference::mcmc::adaptive_mcmc_chain(&mut rng, school_model, 1500, 750);
        
        let school_mu: Vec<f64> = school_samples.iter()
            .filter_map(|(_, trace)| trace.get_f64(&addr!("mu")))
            .collect();
        let school_mean = school_mu.iter().sum::<f64>() / school_mu.len() as f64;
        
        no_pool_estimates.push(school_mean);
        
        let diff = school_mean - school.sample_mean;
        println!("  {:19} | {:6.1} | {:7.1} | {:5.1} | {:3}",
                 school.name, school.sample_mean, school_mean, diff, school.n_students);
    }
    
    println!("  ... (showing first 6 schools)");
    
    // Compare variability
    let sample_means: Vec<f64> = schools.iter().map(|s| s.sample_mean).collect();
    let sample_var = sample_means.iter()
        .map(|&x| (x - pooled_mean).powi(2))
        .sum::<f64>() / sample_means.len() as f64;
    
    let no_pool_var = no_pool_estimates.iter()
        .map(|&x| (x - pooled_mean).powi(2))
        .sum::<f64>() / no_pool_estimates.len() as f64;
    
    println!("\n📊 Variability Comparison:");
    println!("  Sample means variance: {:.2}", sample_var);
    println!("  No-pooling variance: {:.2}", no_pool_var);
    println!("  Complete pooling variance: 0.00 (by design)");
    
    println!("\n🎯 Key Issues:");
    println!("  ❌ Complete pooling ignores school differences");
    println!("  ❌ No pooling doesn't share information between schools");
    println!("  ❌ Small schools have very uncertain estimates");
    println!("  ✅ Need hierarchical modeling for optimal balance!");
}

// Include generate_school_data function

fn main() {
    compare_pooling_approaches();
}
```

## Step 3: Hierarchical Model Implementation

Now let's build the full hierarchical model that balances between these extremes:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn hierarchical_model(schools: Vec<SchoolData>) -> Model<(f64, f64, f64, Vec<f64>)> {
    prob! {
        // Population-level parameters (hyperpriors)
        let mu <- sample(addr!("mu"), Normal::new(75.0, 15.0).unwrap());        // Population mean
        let tau <- sample(addr!("tau"), Exponential::new(0.1).unwrap());        // Between-school std
        let sigma <- sample(addr!("sigma"), Exponential::new(0.1).unwrap());    // Within-school std
        
        // School-level parameters
        let mut school_means = Vec::new();
        for school in &schools {
            let theta_j <- sample(
                addr!("theta", school.school_id), 
                Normal::new(mu, tau).unwrap()
            );
            school_means.push(theta_j);
        }
        
        // Student-level observations
        for (j, school) in schools.iter().enumerate() {
            for (i, &score) in school.scores.iter().enumerate() {
                observe(
                    addr!("score", school.school_id, i),
                    Normal::new(school_means[j], sigma).unwrap(),
                    score
                );
            }
        }
        
        pure((mu, tau, sigma, school_means))
    }
}

fn run_hierarchical_analysis() {
    let schools = generate_school_data();
    
    println!("🏗️ Hierarchical Model Analysis");
    println!("==============================");
    
    let model = || hierarchical_model(schools.clone());
    let mut rng = StdRng::seed_from_u64(12345);
    
    println!("🔄 Running MCMC (this may take a moment for {} schools)...", schools.len());
    
    let samples = inference::mcmc::adaptive_mcmc_chain(&mut rng, model, 4000, 2000);
    
    // Extract population-level parameters
    let mu_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("mu")))
        .collect();
    let tau_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("tau")))
        .collect();
    let sigma_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("sigma")))
        .collect();
    
    let mu_mean = mu_samples.iter().sum::<f64>() / mu_samples.len() as f64;
    let tau_mean = tau_samples.iter().sum::<f64>() / tau_samples.len() as f64;
    let sigma_mean = sigma_samples.iter().sum::<f64>() / sigma_samples.len() as f64;
    
    println!("\n📊 Population-Level Estimates:");
    println!("  Population mean (μ): {:.2} ± {:.2}", mu_mean, 
             (mu_samples.iter().map(|x| (x - mu_mean).powi(2)).sum::<f64>() / mu_samples.len() as f64).sqrt());
    println!("  Between-school std (τ): {:.2} ± {:.2}", tau_mean,
             (tau_samples.iter().map(|x| (x - tau_mean).powi(2)).sum::<f64>() / tau_samples.len() as f64).sqrt());
    println!("  Within-school std (σ): {:.2} ± {:.2}", sigma_mean,
             (sigma_samples.iter().map(|x| (x - sigma_mean).powi(2)).sum::<f64>() / sigma_samples.len() as f64).sqrt());
    
    // Extract school-level parameters
    println!("\n🏫 School-Level Estimates (Partial Pooling):");
    println!("  School               | N   | Sample | Hier.  | Shrink | CI Width");
    println!("  --------------------|-----|--------|--------|--------|----------");
    
    for school in &schools {
        let theta_samples: Vec<f64> = samples.iter()
            .filter_map(|(_, trace)| trace.get_f64(&addr!("theta", school.school_id)))
            .collect();
        
        if !theta_samples.is_empty() {
            let theta_mean = theta_samples.iter().sum::<f64>() / theta_samples.len() as f64;
            let theta_std = (theta_samples.iter().map(|x| (x - theta_mean).powi(2)).sum::<f64>() 
                / theta_samples.len() as f64).sqrt();
            
            // Calculate shrinkage toward population mean
            let shrinkage = (school.sample_mean - theta_mean) / (school.sample_mean - mu_mean);
            let shrinkage_pct = shrinkage * 100.0;
            
            // 95% credible interval width
            let mut sorted_theta = theta_samples.clone();
            sorted_theta.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let ci_lower = sorted_theta[(0.025 * sorted_theta.len() as f64) as usize];
            let ci_upper = sorted_theta[(0.975 * sorted_theta.len() as f64) as usize];
            let ci_width = ci_upper - ci_lower;
            
            println!("  {:19} | {:3} | {:6.1} | {:6.1} | {:5.1}% | {:7.1}",
                     school.name, school.n_students, school.sample_mean, 
                     theta_mean, shrinkage_pct, ci_width);
        }
    }
    
    // Shrinkage analysis
    println!("\n🎯 Shrinkage Analysis:");
    
    let small_schools: Vec<&SchoolData> = schools.iter().filter(|s| s.n_students < 30).collect();
    let large_schools: Vec<&SchoolData> = schools.iter().filter(|s| s.n_students > 60).collect();
    
    if !small_schools.is_empty() && !large_schools.is_empty() {
        let avg_shrinkage_small: f64 = small_schools.iter()
            .filter_map(|school| {
                let theta_samples: Vec<f64> = samples.iter()
                    .filter_map(|(_, trace)| trace.get_f64(&addr!("theta", school.school_id)))
                    .collect();
                if !theta_samples.is_empty() {
                    let theta_mean = theta_samples.iter().sum::<f64>() / theta_samples.len() as f64;
                    let shrinkage = ((school.sample_mean - theta_mean) / (school.sample_mean - mu_mean)).abs();
                    Some(shrinkage)
                } else {
                    None
                }
            })
            .sum::<f64>() / small_schools.len() as f64;
        
        let avg_shrinkage_large: f64 = large_schools.iter()
            .filter_map(|school| {
                let theta_samples: Vec<f64> = samples.iter()
                    .filter_map(|(_, trace)| trace.get_f64(&addr!("theta", school.school_id)))
                    .collect();
                if !theta_samples.is_empty() {
                    let theta_mean = theta_samples.iter().sum::<f64>() / theta_samples.len() as f64;
                    let shrinkage = ((school.sample_mean - theta_mean) / (school.sample_mean - mu_mean)).abs();
                    Some(shrinkage)
                } else {
                    None
                }
            })
            .sum::<f64>() / large_schools.len() as f64;
        
        println!("  Small schools avg shrinkage: {:.1}%", avg_shrinkage_small * 100.0);
        println!("  Large schools avg shrinkage: {:.1}%", avg_shrinkage_large * 100.0);
        
        if avg_shrinkage_small > avg_shrinkage_large * 1.5 {
            println!("  ✅ Appropriate shrinkage: small schools shrink more!");
        }
    }
    
    // Model comparison summary
    let total_students: usize = schools.iter().map(|s| s.n_students).sum();
    println!("\n📈 Model Comparison Summary:");
    println!("  Data: {} students across {} schools", total_students, schools.len());
    println!("  Hierarchical model provides:");
    println!("    🎯 School-specific estimates (vs complete pooling)");
    println!("    📊 Shared information across schools (vs no pooling)");
    println!("    🎲 Appropriate uncertainty quantification");
    println!("    📏 Automatic shrinkage based on sample size");
}

// Include previous functions

fn main() {
    run_hierarchical_analysis();
}
```

## Step 4: Model Diagnostics and Validation

Let's add comprehensive diagnostics for hierarchical models:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn hierarchical_diagnostics() {
    let schools = generate_school_data();
    
    println!("🔍 Hierarchical Model Diagnostics");
    println!("=================================");
    
    let model = || hierarchical_model(schools.clone());
    let mut rng = StdRng::seed_from_u64(42);
    
    let samples = inference::mcmc::adaptive_mcmc_chain(&mut rng, model, 3000, 1500);
    
    // 1. Convergence diagnostics
    let mu_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("mu")))
        .collect();
    
    let tau_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("tau")))
        .collect();
    
    // Split-chain diagnostics (simplified R-hat)
    let n_half = mu_samples.len() / 2;
    let mu_first_half: f64 = mu_samples[..n_half].iter().sum::<f64>() / n_half as f64;
    let mu_second_half: f64 = mu_samples[n_half..].iter().sum::<f64>() / n_half as f64;
    
    let tau_first_half: f64 = tau_samples[..n_half].iter().sum::<f64>() / n_half as f64;
    let tau_second_half: f64 = tau_samples[n_half..].iter().sum::<f64>() / n_half as f64;
    
    println!("🔄 Convergence Diagnostics:");
    println!("  μ first half: {:.3}, second half: {:.3}, diff: {:.3}", 
             mu_first_half, mu_second_half, (mu_first_half - mu_second_half).abs());
    println!("  τ first half: {:.3}, second half: {:.3}, diff: {:.3}", 
             tau_first_half, tau_second_half, (tau_first_half - tau_second_half).abs());
    
    if (mu_first_half - mu_second_half).abs() < 0.5 && (tau_first_half - tau_second_half).abs() < 0.5 {
        println!("  ✅ Good convergence for population parameters");
    } else {
        println!("  ⚠️ May need more samples for convergence");
    }
    
    // 2. Posterior predictive checking
    println!("\n🎯 Posterior Predictive Checks:");
    
    let mu_mean = mu_samples.iter().sum::<f64>() / mu_samples.len() as f64;
    let tau_mean = tau_samples.iter().sum::<f64>() / tau_samples.len() as f64;
    let sigma_mean = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("sigma")))
        .sum::<f64>() / samples.len() as f64;
    
    // Simulate new schools from fitted model
    let mut sim_rng = StdRng::seed_from_u64(999);
    let mut sim_school_means = Vec::new();
    
    for _ in 0..schools.len() {
        let sim_theta = Normal::new(mu_mean, tau_mean).unwrap().sample(&mut sim_rng);
        sim_school_means.push(sim_theta);
    }
    
    // Compare observed vs simulated school mean distribution
    let observed_school_means: Vec<f64> = schools.iter().map(|s| s.sample_mean).collect();
    
    let obs_mean = observed_school_means.iter().sum::<f64>() / observed_school_means.len() as f64;
    let sim_mean = sim_school_means.iter().sum::<f64>() / sim_school_means.len() as f64;
    
    let obs_var = observed_school_means.iter()
        .map(|&x| (x - obs_mean).powi(2))
        .sum::<f64>() / observed_school_means.len() as f64;
    let sim_var = sim_school_means.iter()
        .map(|&x| (x - sim_mean).powi(2))
        .sum::<f64>() / sim_school_means.len() as f64;
    
    println!("  Observed school means: μ={:.2}, σ²={:.2}", obs_mean, obs_var);
    println!("  Simulated school means: μ={:.2}, σ²={:.2}", sim_mean, sim_var);
    
    let var_ratio = obs_var / sim_var;
    if var_ratio > 0.5 && var_ratio < 2.0 {
        println!("  ✅ Model captures between-school variability well");
    } else {
        println!("  ⚠️ Model may not capture variability correctly (ratio: {:.2})", var_ratio);
    }
    
    // 3. Effective sample size analysis
    println!("\n📊 Effective Sample Size Analysis:");
    
    // For each school, compute effective sample size
    for school in &schools[..5] {  // Show first 5
        let n_obs = school.n_students as f64;
        let pooling_factor = n_obs / (n_obs + (sigma_mean / tau_mean).powi(2));
        let effective_n = pooling_factor * n_obs + (1.0 - pooling_factor) * schools.len() as f64;
        
        println!("  {}: N={}, Effective N={:.1}, Pooling={:.2}", 
                 school.name, n_obs, effective_n, pooling_factor);
    }
    
    println!("  ... (showing first 5 schools)");
    println!("  📝 Higher pooling factor = more influence from own data");
    println!("  📝 Lower pooling factor = more shrinkage toward population mean");
    
    // 4. Outlier detection
    println!("\n🚨 Outlier Detection:");
    
    let population_mean = mu_mean;
    let between_school_std = tau_mean;
    
    let mut potential_outliers = Vec::new();
    
    for school in &schools {
        let z_score = (school.sample_mean - population_mean) / between_school_std;
        if z_score.abs() > 2.0 {
            potential_outliers.push((school.name.clone(), school.sample_mean, z_score));
        }
    }
    
    if potential_outliers.is_empty() {
        println!("  ✅ No obvious outlier schools detected");
    } else {
        println!("  Schools with |z-score| > 2.0:");
        for (name, mean, z) in potential_outliers {
            println!("    {}: mean={:.1}, z-score={:.2}", name, mean, z);
        }
    }
    
    // 5. Intraclass Correlation Coefficient (ICC)
    let icc = tau_mean.powi(2) / (tau_mean.powi(2) + sigma_mean.powi(2));
    
    println!("\n📏 Intraclass Correlation (ICC):");
    println!("  ICC = τ²/(τ² + σ²) = {:.3}", icc);
    
    if icc > 0.1 {
        println!("  ✅ Substantial clustering - hierarchical model justified");
    } else if icc > 0.05 {
        println!("  📊 Moderate clustering - hierarchical model beneficial");
    } else {
        println!("  ⚠️ Low clustering - simple pooled model might suffice");
    }
    
    println!("  📝 ICC = proportion of total variance due to school differences");
}

// Include all previous functions

fn main() {
    hierarchical_diagnostics();
}
```

## Step 5: Predictions for New Schools

Finally, let's use our hierarchical model to make predictions for new schools:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn predict_new_schools() {
    println!("🔮 Predictions for New Schools");
    println!("=============================");
    
    // Fit model on existing schools
    let existing_schools = generate_school_data();
    
    let model = || hierarchical_model(existing_schools.clone());
    let mut rng = StdRng::seed_from_u64(42);
    
    println!("🏫 Training on {} existing schools...", existing_schools.len());
    let samples = inference::mcmc::adaptive_mcmc_chain(&mut rng, model, 3000, 1500);
    
    // Extract fitted hyperparameters
    let mu_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("mu")))
        .collect();
    let tau_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("tau")))
        .collect();
    let sigma_samples: Vec<f64> = samples.iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("sigma")))
        .collect();
    
    let mu_mean = mu_samples.iter().sum::<f64>() / mu_samples.len() as f64;
    let tau_mean = tau_samples.iter().sum::<f64>() / tau_samples.len() as f64;
    let sigma_mean = sigma_samples.iter().sum::<f64>() / sigma_samples.len() as f64;
    
    println!("✅ Model fitted. Population parameters:");
    println!("  μ = {:.2} (population mean)", mu_mean);
    println!("  τ = {:.2} (between-school std)", tau_mean);
    println!("  σ = {:.2} (within-school std)", sigma_mean);
    
    // Scenario 1: Completely new school (no data yet)
    println!("\n🏫 Scenario 1: Brand New School (No Data)");
    println!("==========================================");
    
    // For a completely new school, our best guess is the population distribution
    let new_school_samples: Vec<f64> = (0..1000)
        .map(|_| {
            let idx = (rand::random::<f64>() * mu_samples.len() as f64) as usize;
            Normal::new(mu_samples[idx], tau_samples[idx]).unwrap().sample(&mut rng)
        })
        .collect();
    
    let mut sorted_new = new_school_samples.clone();
    sorted_new.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let new_mean = new_school_samples.iter().sum::<f64>() / new_school_samples.len() as f64;
    let new_5th = sorted_new[(0.05 * sorted_new.len() as f64) as usize];
    let new_95th = sorted_new[(0.95 * sorted_new.len() as f64) as usize];
    
    println!("  Expected school mean: {:.1}", new_mean);
    println!("  90% prediction interval: [{:.1}, {:.1}]", new_5th, new_95th);
    println!("  Probability of above-average school: {:.0}%", 
             new_school_samples.iter().filter(|&&x| x > mu_mean).count() as f64 / new_school_samples.len() as f64 * 100.0);
    
    // Scenario 2: New school with limited data
    println!("\n🏫 Scenario 2: New School with Limited Data");
    println!("===========================================");
    
    let new_school_scores = vec![82.0, 79.0, 88.0, 75.0, 91.0];  // 5 students
    let new_sample_mean = new_school_scores.iter().sum::<f64>() / new_school_scores.len() as f64;
    let new_n = new_school_scores.len() as f64;
    
    println!("  Observed: {} students, sample mean = {:.1}", new_n, new_sample_mean);
    
    // Bayesian updating: combine prior (population) with likelihood (data)
    let precision_prior = 1.0 / tau_mean.powi(2);
    let precision_likelihood = new_n / sigma_mean.powi(2);
    let precision_posterior = precision_prior + precision_likelihood;
    
    let mean_posterior = (precision_prior * mu_mean + precision_likelihood * new_sample_mean) / precision_posterior;
    let var_posterior = 1.0 / precision_posterior;
    let std_posterior = var_posterior.sqrt();
    
    println!("  Hierarchical estimate: {:.1} ± {:.1}", mean_posterior, std_posterior);
    println!("  Shrinkage toward population: {:.1}%", 
             ((new_sample_mean - mean_posterior) / (new_sample_mean - mu_mean)).abs() * 100.0);
    
    // Compare to naive estimate (just sample mean)
    let naive_std = sigma_mean / (new_n.sqrt());
    println!("  Naive estimate: {:.1} ± {:.1}", new_sample_mean, naive_std);
    println!("  Hierarchical model is {:.1}x more precise", naive_std / std_posterior);
    
    // Scenario 3: Different school sizes
    println!("\n🏫 Scenario 3: Impact of School Size");
    println!("====================================");
    
    let school_sizes = vec![5, 15, 30, 60, 120];
    let test_sample_mean = 85.0;  // Same observed mean for all
    
    println!("  All schools observe sample mean = {:.1}", test_sample_mean);
    println!("  N   | Hier. Est. | Uncertainty | Shrinkage");
    println!("  ----|------------|-------------|----------");
    
    for &n in &school_sizes {
        let n_f = n as f64;
        let precision_prior = 1.0 / tau_mean.powi(2);
        let precision_likelihood = n_f / sigma_mean.powi(2);
        let precision_posterior = precision_prior + precision_likelihood;
        
        let mean_posterior = (precision_prior * mu_mean + precision_likelihood * test_sample_mean) / precision_posterior;
        let std_posterior = (1.0 / precision_posterior).sqrt();
        let shrinkage = ((test_sample_mean - mean_posterior) / (test_sample_mean - mu_mean)).abs() * 100.0;
        
        println!("  {:3} | {:10.1} | {:11.2} | {:8.1}%", n, mean_posterior, std_posterior, shrinkage);
    }
    
    println!("\n💡 Key Insights:");
    println!("  📏 Larger schools: less shrinkage, more precise estimates");
    println!("  📏 Smaller schools: more shrinkage toward population mean");
    println!("  🎯 Hierarchical model automatically balances individual vs population information");
    
    // Scenario 4: School ranking with uncertainty
    println!("\n🏆 School Performance Ranking");
    println!("============================");
    
    // Get posterior estimates for existing schools
    let mut school_estimates = Vec::new();
    
    for school in &existing_schools[..8] {  // Top 8 schools
        let theta_samples: Vec<f64> = samples.iter()
            .filter_map(|(_, trace)| trace.get_f64(&addr!("theta", school.school_id)))
            .collect();
        
        if !theta_samples.is_empty() {
            let theta_mean = theta_samples.iter().sum::<f64>() / theta_samples.len() as f64;
            let theta_std = (theta_samples.iter().map(|x| (x - theta_mean).powi(2)).sum::<f64>() 
                / theta_samples.len() as f64).sqrt();
            
            school_estimates.push((school.name.clone(), school.sample_mean, theta_mean, theta_std, school.n_students));
        }
    }
    
    // Sort by hierarchical estimate
    school_estimates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    
    println!("  Rank | School               | Sample | Hier.  | ±95%  | N   ");
    println!("  -----|---------------------|--------|--------|-------|-----");
    
    for (i, (name, sample, hier, std, n)) in school_estimates.iter().enumerate() {
        let ci_width = 1.96 * std;
        println!("   {:2}  | {:19} | {:6.1} | {:6.1} | {:5.1} | {:3}", 
                 i + 1, name, sample, hier, ci_width, n);
    }
    
    println!("\n🎯 Ranking Insights:");
    println!("  📊 Small schools with extreme sample means get 'adjusted' rankings");
    println!("  📏 Uncertainty intervals help identify truly exceptional vs lucky schools");
    println!("  🎲 Consider overlap in confidence intervals when making decisions");
}

// Include all previous functions

fn main() {
    predict_new_schools();
}
```

## Key Concepts Review

### 1. Hierarchical Model Structure
- **Multiple levels**: Population → Groups → Individuals
- **Partial pooling**: Optimal balance between complete and no pooling
- **Exchangeability**: Groups are similar but not identical
- **Shrinkage**: Automatic regularization based on data quality

### 2. Bayesian Advantages
- **Uncertainty propagation**: From hyperparameters to group estimates
- **Automatic complexity control**: No manual tuning of shrinkage
- **Natural handling of unbalanced data**: Different group sizes
- **Principled inference**: Full posterior distributions at all levels

### 3. Practical Applications
- **Education**: School effects, teacher effectiveness
- **Medicine**: Hospital quality, treatment effects by clinic
- **Marketing**: Customer segments, regional preferences
- **A/B Testing**: Treatment effects across user segments

### 4. Model Diagnostics
- **Convergence**: Split-chain R-hat for population parameters
- **Model fit**: Posterior predictive checking of group-level variance
- **Outlier detection**: Schools far from population distribution
- **ICC**: Quantifies clustering and model necessity

## Exercise: Extend the Analysis

Try these extensions to deepen your understanding:

1. **Three-level hierarchy**: Students → Teachers → Schools → District
2. **Regression coefficients**: School-specific slopes for covariates
3. **Non-normal data**: Hierarchical logistic regression for test pass rates
4. **Time series**: Longitudinal hierarchical models for school improvement

## Next Steps

Congratulations! You've mastered the most sophisticated modeling approach in this tutorial series. You now understand:

- **Complete PPL workflow**: From simple models to complex hierarchies
- **Bayesian inference**: Principled uncertainty quantification
- **Model comparison**: Information criteria and cross-validation
- **Real-world applications**: Practical probabilistic programming

### Continue Your Journey

1. **Advanced topics**: Explore non-parametric Bayesian methods
2. **Computational methods**: Learn about advanced MCMC techniques
3. **Model checking**: Develop sophisticated diagnostic workflows
4. **Production deployment**: Scale Bayesian models to real applications

### Fugue Mastery Checklist
- ✅ **Basic models**: Coin flips, simple inference
- ✅ **Regression**: Linear, polynomial, multiple predictors
- ✅ **Clustering**: Mixture models, model selection
- ✅ **Hierarchical**: Multi-level, partial pooling, shrinkage
- ✅ **Diagnostics**: Convergence, model fit, outlier detection
- ✅ **Predictions**: New observations, uncertainty quantification

You're now equipped to tackle sophisticated probabilistic modeling challenges with Fugue!

---

**Ready to build production systems?** → **[Custom Handlers Guide](../how-to/custom-handlers.md)**