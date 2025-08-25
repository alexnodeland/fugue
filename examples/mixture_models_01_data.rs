use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Generate customer spending data with 3 segments
fn generate_mixture_data(n_customers: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(2024);

    // True mixture parameters (unknown to our model)
    let true_weights = vec![0.4, 0.35, 0.25];  // Budget, Mid-tier, Premium
    let true_means = vec![25.0, 75.0, 180.0];  // Average spending per segment
    let true_stds = vec![8.0, 15.0, 30.0];     // Variability within segments

    let mut data = Vec::new();

    for _ in 0..n_customers {
        // Sample component assignment
        let u: f64 = Uniform::new(0.0, 1.0).unwrap().sample(&mut rng);
        let component = if u < true_weights[0] {
            0
        } else if u < true_weights[0] + true_weights[1] {
            1
        } else {
            2
        };

        // Sample spending from assigned component
        let spending = Normal::new(true_means[component], true_stds[component])
            .unwrap()
            .sample(&mut rng);

        data.push(spending.max(0.0));  // Ensure non-negative spending
    }

    data
}

fn explore_data() {
    let data = generate_mixture_data(200);

    println!("💳 Customer Spending Analysis");
    println!("============================");

    // Basic statistics
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / data.len() as f64;
    let std_dev = variance.sqrt();

    println!("📊 Basic Statistics:");
    println!("  N customers: {}", data.len());
    println!("  Mean spending: ${:.2}", mean);
    println!("  Std deviation: ${:.2}", std_dev);

    // Show distribution (simple histogram)
    let mut sorted_data = data.clone();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!("\n📈 Spending Distribution (quartiles):");
    println!("  Min: ${:.2}", sorted_data[0]);
    println!("  Q1:  ${:.2}", sorted_data[sorted_data.len() / 4]);
    println!("  Q2:  ${:.2}", sorted_data[sorted_data.len() / 2]);
    println!("  Q3:  ${:.2}", sorted_data[3 * sorted_data.len() / 4]);
    println!("  Max: ${:.2}", sorted_data[sorted_data.len() - 1]);

    // Evidence of multimodality
    let low_spenders = data.iter().filter(|&&x| x < 50.0).count();
    let mid_spenders = data.iter().filter(|&&x| x >= 50.0 && x < 120.0).count();
    let high_spenders = data.iter().filter(|&&x| x >= 120.0).count();

    println!("\n🎯 Spending Segments (intuitive split):");
    println!("  Low spenders (<$50): {} ({:.1}%)", low_spenders,
             low_spenders as f64 / data.len() as f64 * 100.0);
    println!("  Mid spenders ($50-120): {} ({:.1}%)", mid_spenders,
             mid_spenders as f64 / data.len() as f64 * 100.0);
    println!("  High spenders (>$120): {} ({:.1}%)", high_spenders,
             high_spenders as f64 / data.len() as f64 * 100.0);
}

fn main() {
    explore_data();
}