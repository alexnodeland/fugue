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