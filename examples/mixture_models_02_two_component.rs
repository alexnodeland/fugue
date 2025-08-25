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