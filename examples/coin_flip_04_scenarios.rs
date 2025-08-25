use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn scenario_analysis() {
    println!("🎭 Scenario Analysis");
    println!("====================");

    let scenarios = vec![
        (0, 10, "Never heads"),
        (1, 10, "Rarely heads"),
        (3, 10, "Sometimes heads"),
        (5, 10, "Half heads (fair?)"),
        (7, 10, "Often heads"),
        (9, 10, "Almost always heads"),
        (10, 10, "Always heads"),
        (20, 40, "Many flips: half heads"),
        (28, 40, "Many flips: often heads"),
    ];

    for (heads, flips, description) in scenarios {
        analyze_scenario(heads, flips, description);
    }
}

fn analyze_scenario(heads: u64, flips: u64, description: &str) {
    let model = || coin_model(heads, flips);
    let mut rng = StdRng::seed_from_u64(42);

    // Quick MCMC run
    let samples = inference::mcmc::adaptive_mcmc_chain(
        &mut rng,
        model,
        1000,
        500,
    );

    let bias_samples: Vec<f64> = samples
        .iter()
        .filter_map(|(_, trace)| trace.get_f64(&addr!("bias")))
        .collect();

    let mean = bias_samples.iter().sum::<f64>() / bias_samples.len() as f64;

    // Probability assessments
    let prob_fair = bias_samples.iter()
        .filter(|&&x| (x - 0.5).abs() < 0.1)
        .count() as f64 / bias_samples.len() as f64;

    let prob_heads_favored = bias_samples.iter()
        .filter(|&&x| x > 0.6)
        .count() as f64 / bias_samples.len() as f64;

    let prob_tails_favored = bias_samples.iter()
        .filter(|&&x| x < 0.4)
        .count() as f64 / bias_samples.len() as f64;

    println!("\n📋 {}: {}/{} heads", description, heads, flips);
    println!("   Estimated bias: {:.3}", mean);
    println!("   P(fair): {:.0}%, P(heads-biased): {:.0}%, P(tails-biased): {:.0}%",
             prob_fair * 100.0, prob_heads_favored * 100.0, prob_tails_favored * 100.0);
}

fn coin_model(observed_heads: u64, total_flips: u64) -> Model<f64> {
    prob! {
        let bias <- sample(addr!("bias"), Beta::new(1.0, 1.0).unwrap());
        observe(addr!("heads"), Binomial::new(total_flips, bias).unwrap(), observed_heads);
        pure(bias)
    }
}

fn main() {
    scenario_analysis();
}