use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn predictive_regression_model(
    advertising: Vec<f64>,
    revenue: Vec<f64>,
    new_ad_spend: f64
) -> Model<(f64, f64, f64, f64)> {
    prob! {
        // Priors and likelihood (same as before)
        let alpha <- sample(addr!("alpha"), Normal::new(0.0, 100.0).unwrap());
        let beta <- sample(addr!("beta"), Normal::new(0.0, 10.0).unwrap());
        let sigma <- sample(addr!("sigma"), Exponential::new(1.0).unwrap());

        for (i, (&ad, &rev)) in advertising.iter().zip(revenue.iter()).enumerate() {
            let predicted_mean = alpha + beta * ad;
            observe(
                addr!("revenue", i),
                Normal::new(predicted_mean, sigma).unwrap(),
                rev
            );
        }

        // Posterior prediction for new advertising spend
        let predicted_mean = alpha + beta * new_ad_spend;
        let predicted_revenue <- sample(
            addr!("predicted_revenue"),
            Normal::new(predicted_mean, sigma).unwrap()
        );

        pure((alpha, beta, sigma, predicted_revenue))
    }
}

fn prediction_analysis() {
    let (advertising, revenue) = generate_data();

    println!("🔮 Prediction Analysis");
    println!("=====================");

    // Test different advertising levels
    let test_ad_spends = vec![25.0, 35.0, 45.0, 55.0];  // Including extrapolation

    for &ad_spend in &test_ad_spends {
        let model = || predictive_regression_model(
            advertising.clone(),
            revenue.clone(),
            ad_spend
        );
        let mut rng = StdRng::seed_from_u64(42);

        let samples = inference::mcmc::adaptive_mcmc_chain(
            &mut rng,
            model,
            2000,
            1000,
        );

        let predicted_revenues: Vec<f64> = samples.iter()
            .filter_map(|(_, trace)| trace.get_f64(&addr!("predicted_revenue")))
            .collect();

        // Compute prediction intervals
        let mut sorted_preds = predicted_revenues.clone();
        sorted_preds.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean_pred = predicted_revenues.iter().sum::<f64>() / predicted_revenues.len() as f64;
        let pred_5th = sorted_preds[(0.05 * sorted_preds.len() as f64) as usize];
        let pred_95th = sorted_preds[(0.95 * sorted_preds.len() as f64) as usize];

        println!("\n💰 Ad Spend: ${:.0}K", ad_spend);
        println!("  Expected Revenue: ${:.1}K", mean_pred);
        println!("  90% Prediction Interval: [${:.1}K, ${:.1}K]", pred_5th, pred_95th);

        // ROI analysis
        let roi = (mean_pred - ad_spend) / ad_spend;
        println!("  Expected ROI: {:.1}%", roi * 100.0);

        // Probability of profitability (revenue > ad spend)
        let prob_profitable = predicted_revenues.iter()
            .filter(|&&rev| rev > ad_spend)
            .count() as f64 / predicted_revenues.len() as f64;
        println!("  P(Revenue > Ad Spend): {:.1}%", prob_profitable * 100.0);
    }
}

// Include generate_data function

fn main() {
    prediction_analysis();
}