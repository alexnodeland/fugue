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