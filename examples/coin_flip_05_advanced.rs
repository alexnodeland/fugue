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