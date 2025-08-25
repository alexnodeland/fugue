use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Generate realistic advertising/revenue data
fn generate_data() -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(12345);

    // True parameters (unknown to our model)
    let true_alpha = 50.0;  // Base revenue
    let true_beta = 3.5;    // Revenue per ad dollar
    let true_sigma = 8.0;   // Noise level

    let advertising: Vec<f64> = (1..=20)
        .map(|i| 10.0 + i as f64 * 2.0)  // $12K to $50K advertising
        .collect();

    let revenue: Vec<f64> = advertising
        .iter()
        .map(|&ad| {
            let mean = true_alpha + true_beta * ad;
            Normal::new(mean, true_sigma).unwrap().sample(&mut rng)
        })
        .collect();

    (advertising, revenue)
}

fn main() {
    let (advertising, revenue) = generate_data();

    println!("📊 Advertising vs Revenue Data");
    println!("==============================");
    for (i, (&ad, &rev)) in advertising.iter().zip(revenue.iter()).enumerate() {
        println!("Quarter {}: ${:.0}K ads → ${:.1}K revenue", i+1, ad, rev);
    }
}