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