use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn classify_new_customers() {
    println!("👥 Customer Classification");
    println!("=========================");

    // Fit our best model (3-component) on training data
    let training_data = generate_mixture_data(200);

    let model = || three_component_mixture_model(training_data.clone());
    let mut rng = StdRng::seed_from_u64(999);

    let samples = inference::mcmc::adaptive_mcmc_chain(&mut rng, model, 3000, 1500);

    // Extract fitted parameters (posterior means)
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

    let mu1 = mu1_samples.iter().sum::<f64>() / mu1_samples.len() as f64;
    let mu2 = mu2_samples.iter().sum::<f64>() / mu2_samples.len() as f64;
    let mu3 = mu3_samples.iter().sum::<f64>() / mu3_samples.len() as f64;

    let sigma1 = sigma1_samples.iter().sum::<f64>() / sigma1_samples.len() as f64;
    let sigma2 = sigma2_samples.iter().sum::<f64>() / sigma2_samples.len() as f64;
    let sigma3 = sigma3_samples.iter().sum::<f64>() / sigma3_samples.len() as f64;

    // Sort components by mean
    let mut components = vec![(mu1, sigma1, 1), (mu2, sigma2, 2), (mu3, sigma3, 3)];
    components.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    println!("📊 Fitted Model Parameters:");
    for (i, (mu, sigma, orig_id)) in components.iter().enumerate() {
        let segment_name = match i {
            0 => "Budget",
            1 => "Mid-tier",
            2 => "Premium",
            _ => "Unknown"
        };
        println!("  {} Segment: μ=${:.2}, σ=${:.2}", segment_name, mu, sigma);
    }

    // Classify new customers
    let new_customers = vec![15.0, 42.0, 73.0, 125.0, 220.0];

    println!("\n🎯 New Customer Classifications:");
    println!("  Spending | Budget | Mid-tier | Premium | Assignment");
    println!("  ---------|--------|----------|---------|------------");

    for &spending in &new_customers {
        // Compute posterior probabilities for each component
        let prob1 = Normal::new(components[0].0, components[0].1).unwrap().exp_log_prob(spending).exp();
        let prob2 = Normal::new(components[1].0, components[1].1).unwrap().exp_log_prob(spending).exp();
        let prob3 = Normal::new(components[2].0, components[2].1).unwrap().exp_log_prob(spending).exp();

        let total_prob = prob1 + prob2 + prob3;

        let post_prob1 = prob1 / total_prob;
        let post_prob2 = prob2 / total_prob;
        let post_prob3 = prob3 / total_prob;

        let (assignment, confidence) = if post_prob1 > post_prob2 && post_prob1 > post_prob3 {
            ("Budget", post_prob1)
        } else if post_prob2 > post_prob3 {
            ("Mid-tier", post_prob2)
        } else {
            ("Premium", post_prob3)
        };

        println!("   ${:6.1} | {:5.3} | {:7.3} | {:6.3} | {} ({:.0}%)",
                 spending, post_prob1, post_prob2, post_prob3, assignment, confidence * 100.0);
    }

    println!("\n💼 Business Insights:");
    println!("  🎯 Customers with >80% probability are confidently classified");
    println!("  ⚖️ Customers with 40-60% probabilities are boundary cases");
    println!("  📊 Use probabilities for targeted marketing strategies");

    // Expected lifetime value per segment
    let budget_value = components[0].0 * 12.0;  // Annual value
    let midtier_value = components[1].0 * 12.0;
    let premium_value = components[2].0 * 12.0;

    println!("\n💰 Expected Annual Customer Value:");
    println!("  Budget segment: ${:.0}", budget_value);
    println!("  Mid-tier segment: ${:.0}", midtier_value);
    println!("  Premium segment: ${:.0}", premium_value);

    println!("\n🚀 Next Steps:");
    println!("  1. A/B test marketing campaigns by segment");
    println!("  2. Set different service levels for each tier");
    println!("  3. Monitor segment transitions over time");
    println!("  4. Develop segment-specific product recommendations");
}

// Include all previous functions

fn main() {
    classify_new_customers();
}