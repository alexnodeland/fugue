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