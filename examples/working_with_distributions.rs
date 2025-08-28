use fugue::*;
use rand::thread_rng;

fn main() {
    let mut rng = thread_rng();

    println!("=== Fugue Distribution Examples ===\n");

    println!("1. Type Safety Demo");
    println!("-------------------");
    // ANCHOR: type_safety_demo
    // Demonstrate natural return types
    let coin = Bernoulli::new(0.5).unwrap();
    let flip: bool = coin.sample(&mut rng); // Natural boolean

    if flip {
        println!("🪙 Coin flip result: Heads!");
    } else {
        println!("🪙 Coin flip result: Tails!");
    }
    // ANCHOR_END: type_safety_demo
    println!("✓ Natural boolean type - no casting needed!");
    println!();

    println!("2. Continuous Distributions");
    println!("---------------------------");
    // ANCHOR: continuous_distributions
    // Working with continuous distributions
    let standard_normal = Normal::new(0.0, 1.0).unwrap();
    let sample: f64 = standard_normal.sample(&mut rng);
    println!("📊 Standard normal sample: {:.3}", sample);

    // Compute log-probability density
    let log_density = standard_normal.log_prob(&0.0); // Peak of standard normal
    println!(
        "📈 Log-density at x=0: {:.3} (peak of standard normal)",
        log_density
    );

    // Custom parameters
    let measurement_model = Normal::new(10.0, 0.5).unwrap();
    let measurement = measurement_model.sample(&mut rng);
    println!("🔬 Sensor measurement (μ=10.0, σ=0.5): {:.3}", measurement);
    // ANCHOR_END: continuous_distributions
    println!("✓ Direct f64 arithmetic - no type conversion overhead");
    println!();

    println!("3. Discrete Distributions");
    println!("-------------------------");
    // ANCHOR: discrete_distributions
    // Working with discrete distributions

    // Count data
    let event_rate = Poisson::new(3.0).unwrap();
    let count: u64 = event_rate.sample(&mut rng);
    println!("📅 Event count (λ=3.0): {} events", count);

    // Log-probability mass
    let prob_3_events = event_rate.log_prob(&3);
    println!(
        "🎯 Log-probability of exactly 3 events: {:.3}",
        prob_3_events
    );

    // Use counts directly in calculations
    let total_cost = count * 50; // Direct arithmetic with u64
    println!("💰 Total cost ({} events × $50): ${}", count, total_cost);
    // ANCHOR_END: discrete_distributions
    println!("✓ Natural u64 counts - no precision loss from floats");
    println!();

    println!("4. Safe Categorical Sampling");
    println!("----------------------------");
    // ANCHOR: categorical_usage
    // Safe categorical sampling
    let choices = vec![0.3, 0.5, 0.2]; // Three categories
    let categorical = Categorical::new(choices).unwrap();
    let selected: usize = categorical.sample(&mut rng);

    // Safe array indexing (no bounds checking needed)
    let options = ["Option A", "Option B", "Option C"];
    println!(
        "🎲 Categorical choice (weights: 0.3, 0.5, 0.2): {}",
        options[selected]
    );

    // Uniform categorical
    let uniform_choice = Categorical::uniform(5).unwrap();
    let idx: usize = uniform_choice.sample(&mut rng);
    println!("🎯 Uniform random index (0-4): {}", idx);
    // ANCHOR_END: categorical_usage
    println!("✓ usize return guarantees valid array indexing");
    println!();

    println!("5. Parameter Validation");
    println!("----------------------");
    // ANCHOR: parameter_validation
    // Distribution parameter validation

    // This will return an error
    match Normal::new(0.0, -1.0) {
        Ok(_) => println!("✅ Normal(μ=0.0, σ=-1.0) created successfully"),
        Err(e) => println!("❌ Normal(μ=0.0, σ=-1.0) failed: {:?}", e),
    }

    // Beta distribution parameters must be positive
    match Beta::new(0.0, 1.0) {
        Ok(_) => println!("✅ Beta(α=0.0, β=1.0) created successfully"),
        Err(e) => println!("❌ Beta(α=0.0, β=1.0) failed: {:?}", e),
    }

    // Poisson rate must be non-negative
    match Poisson::new(-1.0) {
        Ok(_) => println!("✅ Poisson(λ=-1.0) created successfully"),
        Err(e) => println!("❌ Poisson(λ=-1.0) failed: {:?}", e),
    }
    // ANCHOR_END: parameter_validation
    println!("✓ All parameter validation happens at construction time");
    println!();

    println!("6. Distribution Collections");
    println!("--------------------------");
    // ANCHOR: distribution_composition
    // Storing different distributions together
    let mut continuous_dists: Vec<Box<dyn Distribution<f64>>> = vec![];
    continuous_dists.push(Normal::new(0.0, 1.0).unwrap().clone_box());
    continuous_dists.push(Beta::new(2.0, 5.0).unwrap().clone_box());
    continuous_dists.push(Uniform::new(-1.0, 1.0).unwrap().clone_box());

    // Sample from each
    for (i, dist) in continuous_dists.iter().enumerate() {
        let sample = dist.sample(&mut rng);
        let dist_name = match i {
            0 => "Normal(0,1)",
            1 => "Beta(2,5)",
            2 => "Uniform(-1,1)",
            _ => "Unknown",
        };
        println!("📦 {} sample: {:.3}", dist_name, sample);
    }
    // ANCHOR_END: distribution_composition
    println!("✓ Trait objects enable dynamic distribution selection");
    println!();

    println!("7. Practical Modeling Examples");
    println!("------------------------------");
    // ANCHOR: practical_modeling
    // Practical modeling examples

    // Model a sensor with noise
    let true_temperature = 20.5; // True value
    let sensor_noise = Normal::new(0.0, 0.2).unwrap(); // Measurement error
    let measured_temp = true_temperature + sensor_noise.sample(&mut rng);
    println!(
        "🌡️  True temperature: {:.2}°C → Measured: {:.2}°C",
        true_temperature, measured_temp
    );

    // Count model for arrivals
    let arrival_rate = Poisson::new(2.5).unwrap(); // 2.5 arrivals per hour
    let hourly_arrivals = arrival_rate.sample(&mut rng);
    println!(
        "🚪 Expected arrivals: 2.5/hour → Actual: {} arrivals",
        hourly_arrivals
    );

    // Decision model
    let decision_prob = 0.7;
    let decision = Bernoulli::new(decision_prob).unwrap();
    let will_buy = decision.sample(&mut rng);
    if will_buy {
        println!("🛒 Customer decision (p=0.7): Will make a purchase");
    } else {
        println!("🚶 Customer decision (p=0.7): Will not purchase");
    }
    // ANCHOR_END: practical_modeling
    println!("✓ Each distribution serves its natural domain");
    println!();

    println!("8. Log-Probability Calculations");
    println!("-------------------------------");
    // ANCHOR: probability_calculations
    // Working with log-probabilities

    let normal = Normal::new(100.0, 15.0).unwrap();

    // Multiple observations
    let observations = vec![98.5, 102.1, 99.8, 101.5, 97.2];
    let mut total_log_prob = 0.0;

    println!(
        "📋 Evaluating {} observations under Normal(μ=100.0, σ=15.0):",
        observations.len()
    );
    for obs in &observations {
        let log_p = normal.log_prob(obs);
        total_log_prob += log_p;
        println!("   x={}: log P(x) = {:.3}", obs, log_p);
    }

    println!("🔢 Joint log-probability: {:.3}", total_log_prob);

    // Convert back to probability (be careful with underflow!)
    if total_log_prob > -700.0 {
        // Avoid underflow
        let probability = total_log_prob.exp();
        println!("📊 Joint probability: {:.2e}", probability);
    } else {
        println!("⚠️  Joint probability too small to represent as f64");
    }
    // ANCHOR_END: probability_calculations
    println!("✓ Log-space arithmetic prevents numerical underflow");
    println!();

    println!("=== All examples completed successfully! ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    // ANCHOR: distribution_testing
    #[test]
    fn test_distribution_properties() {
        let mut rng = thread_rng();

        // Test type safety
        let coin = Bernoulli::new(0.5).unwrap();
        let flip: bool = coin.sample(&mut rng);
        assert!(flip == true || flip == false); // Must be boolean

        // Test parameter validation
        assert!(Normal::new(0.0, -1.0).is_err());
        assert!(Beta::new(0.0, 1.0).is_err());
        assert!(Poisson::new(-1.0).is_err());

        // Test valid distributions
        let normal = Normal::new(0.0, 1.0).unwrap();
        let sample = normal.sample(&mut rng);
        let log_prob = normal.log_prob(&sample);
        assert!(log_prob.is_finite());
    }
    // ANCHOR_END: distribution_testing
}
