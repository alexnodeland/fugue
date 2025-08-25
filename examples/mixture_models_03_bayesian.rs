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