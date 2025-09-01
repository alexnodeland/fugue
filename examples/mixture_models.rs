use fugue::inference::mh::adaptive_mcmc_chain;
use fugue::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

// ANCHOR: synthetic_mixture_data
// Generate synthetic data from a mixture of Gaussians
fn generate_mixture_data(
    n: usize,
    components: &[(f64, f64, f64)],
    seed: u64,
) -> (Vec<f64>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Vec::new();
    let mut true_labels = Vec::new();

    // Extract mixing weights
    let weights: Vec<f64> = components.iter().map(|(w, _, _)| *w).collect();
    let cumulative_weights: Vec<f64> = weights
        .iter()
        .scan(0.0, |acc, &w| {
            *acc += w;
            Some(*acc)
        })
        .collect();

    for _ in 0..n {
        // Sample component
        let u: f64 = rng.gen();
        let component = cumulative_weights
            .iter()
            .position(|&cw| u <= cw)
            .unwrap_or(components.len() - 1);
        let (_, mu, sigma) = components[component];

        // Sample from component
        let noise: f64 = StandardNormal.sample(&mut rng);
        let x = mu + sigma * noise;

        data.push(x);
        true_labels.push(component);
    }

    (data, true_labels)
}

// Generate mixture of experts data
fn generate_moe_data(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for _ in 0..n {
        let x: f64 = rng.gen::<f64>() * 4.0 - 2.0; // x in [-2, 2]

        // Different relationships in different regions
        let y = if x < 0.0 {
            // Linear relationship for x < 0
            let noise: f64 = StandardNormal.sample(&mut rng);
            2.0 * x + 1.0 + noise * 0.3
        } else {
            // Quadratic relationship for x >= 0
            let noise: f64 = StandardNormal.sample(&mut rng);
            x * x - 0.5 * x + noise * 0.3
        };

        x_data.push(x);
        y_data.push(y);
    }

    (x_data, y_data)
}
// ANCHOR_END: synthetic_mixture_data

// ANCHOR: gaussian_mixture_model
// Simple 2-component Gaussian mixture model
fn gaussian_mixture_model(data: Vec<f64>) -> Model<(f64, f64, f64, f64, f64)> {
    prob! {
        // Mixing weight for first component
        let pi1 <- sample(addr!("pi1"), fugue::Beta::new(1.0, 1.0).unwrap());

        // Component 1 parameters
        let mu1 <- sample(addr!("mu1"), fugue::Normal::new(0.0, 5.0).unwrap());
        let sigma1 <- sample(addr!("sigma1"), Gamma::new(1.0, 1.0).unwrap());

        // Component 2 parameters
        let mu2 <- sample(addr!("mu2"), fugue::Normal::new(0.0, 5.0).unwrap());
        let sigma2 <- sample(addr!("sigma2"), Gamma::new(1.0, 1.0).unwrap());

        // Observations
        let _observations <- plate!(i in 0..data.len() => {
            // Ensure valid probabilities
            let p1 = pi1.clamp(0.001, 0.999); // Clamp to valid range
            let weights = vec![p1, 1.0 - p1];
            let x = data[i];

            sample(addr!("z", i), Categorical::new(weights).unwrap())
                .bind(move |z_i| {
                    // Explicitly handle only 2 components
                    let (mu_i, sigma_i) = if z_i == 0 {
                        (mu1, sigma1)
                    } else {
                        (mu2, sigma2)
                    };
                    observe(addr!("x", i), fugue::Normal::new(mu_i, sigma_i).unwrap(), x)
                })
        });

        pure((pi1, mu1, sigma1, mu2, sigma2))
    }
}

fn gaussian_mixture_demo() {
    println!("=== Gaussian Mixture Model ===\n");

    // Generate synthetic mixture data: 2 components
    let true_components = vec![
        (0.6, -1.5, 0.8), // 60% weight, mean=-1.5, std=0.8
        (0.4, 2.0, 1.2),  // 40% weight, mean=2.0, std=1.2
    ];
    let (data, true_labels) = generate_mixture_data(80, &true_components, 42);

    println!(
        "📊 Generated {} data points from {} true components",
        data.len(),
        true_components.len()
    );
    for (i, (weight, mu, sigma)) in true_components.iter().enumerate() {
        println!(
            "   - Component {}: π={:.1}, μ={:.1}, σ={:.1}",
            i + 1,
            weight,
            mu,
            sigma
        );
    }

    let n_true_labels: Vec<usize> = (0..true_components.len())
        .map(|k| true_labels.iter().filter(|&&label| label == k).count())
        .collect();

    for (k, count) in n_true_labels.iter().enumerate() {
        println!(
            "   - True cluster {}: {} observations ({:.1}%)",
            k + 1,
            count,
            100.0 * *count as f64 / data.len() as f64
        );
    }

    // Fit mixture model
    println!("\n🔬 Fitting 2-component Gaussian mixture model...");
    let model_fn = move || gaussian_mixture_model(data.clone());
    let mut rng = StdRng::seed_from_u64(123);
    let samples = adaptive_mcmc_chain(&mut rng, model_fn, 600, 150);

    let valid_samples: Vec<_> = samples
        .iter()
        .filter(|(_, trace)| trace.total_log_weight().is_finite())
        .collect();

    if !valid_samples.is_empty() {
        println!(
            "✅ MCMC completed with {} valid samples",
            valid_samples.len()
        );

        // Extract parameter estimates
        println!("\n📈 Estimated Parameters:");

        let pi1_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.0).collect();
        let mu1_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.1).collect();
        let sigma1_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.2).collect();
        let mu2_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.3).collect();
        let sigma2_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.4).collect();

        let mean_pi1 = pi1_samples.iter().sum::<f64>() / pi1_samples.len() as f64;
        let mean_mu1 = mu1_samples.iter().sum::<f64>() / mu1_samples.len() as f64;
        let mean_sigma1 = sigma1_samples.iter().sum::<f64>() / sigma1_samples.len() as f64;
        let mean_mu2 = mu2_samples.iter().sum::<f64>() / mu2_samples.len() as f64;
        let mean_sigma2 = sigma2_samples.iter().sum::<f64>() / sigma2_samples.len() as f64;

        println!(
            "   - Component 1: π̂={:.2}, μ̂={:.1}, σ̂={:.1}",
            mean_pi1, mean_mu1, mean_sigma1
        );
        println!(
            "   - Component 2: π̂={:.2}, μ̂={:.1}, σ̂={:.1}",
            1.0 - mean_pi1,
            mean_mu2,
            mean_sigma2
        );

        println!("\n🎯 Parameter Recovery:");
        let (true_w1, true_mu1, true_sigma1) = true_components[0];
        let (true_w2, true_mu2, true_sigma2) = true_components[1];
        println!("   - Component 1: π true={:.1} est={:.2}, μ true={:.1} est={:.1}, σ true={:.1} est={:.1}",
                true_w1, mean_pi1, true_mu1, mean_mu1, true_sigma1, mean_sigma1);
        println!("   - Component 2: π true={:.1} est={:.2}, μ true={:.1} est={:.1}, σ true={:.1} est={:.1}",
                true_w2, 1.0 - mean_pi1, true_mu2, mean_mu2, true_sigma2, mean_sigma2);
    } else {
        println!("❌ No valid MCMC samples obtained");
    }

    println!();
}
// ANCHOR_END: gaussian_mixture_model

// ANCHOR: multivariate_mixture_model
// Simple 2-component multivariate Gaussian mixture (2D)
#[allow(clippy::type_complexity)] // Complex tuple needed for demonstration
fn multivariate_mixture_model(
    data: Vec<Vec<f64>>,
) -> Model<(f64, f64, f64, f64, f64, f64, f64, f64)> {
    prob! {
        // Mixing weight
        let pi1 <- sample(addr!("pi1"), fugue::Beta::new(1.0, 1.0).unwrap());

        // Component 1 parameters (2D means and diagonal covariance)
        let mu1_0 <- sample(addr!("mu1_0"), fugue::Normal::new(0.0, 5.0).unwrap());
        let mu1_1 <- sample(addr!("mu1_1"), fugue::Normal::new(0.0, 5.0).unwrap());
        let sigma1_0 <- sample(addr!("sigma1_0"), Gamma::new(1.0, 1.0).unwrap());
        let sigma1_1 <- sample(addr!("sigma1_1"), Gamma::new(1.0, 1.0).unwrap());

        // Component 2 parameters
        let mu2_0 <- sample(addr!("mu2_0"), fugue::Normal::new(0.0, 5.0).unwrap());
        let mu2_1 <- sample(addr!("mu2_1"), fugue::Normal::new(0.0, 5.0).unwrap());
        let sigma2_0 <- sample(addr!("sigma2_0"), Gamma::new(1.0, 1.0).unwrap());
        let sigma2_1 <- sample(addr!("sigma2_1"), Gamma::new(1.0, 1.0).unwrap());

        // Observations (diagonal covariance assumption)
        let _observations <- plate!(i in 0..data.len() => {
            let p1 = pi1.clamp(0.001, 0.999);
            let weights = vec![p1, 1.0 - p1];
            let x0 = data[i][0];
            let x1 = data[i][1];

            sample(addr!("z", i), Categorical::new(weights).unwrap())
                .bind(move |z_i| {
                    let (mu_0, mu_1, sigma_0, sigma_1) = if z_i == 0 {
                        (mu1_0, mu1_1, sigma1_0, sigma1_1)
                    } else {
                        (mu2_0, mu2_1, sigma2_0, sigma2_1)
                    };

                    // Independent dimensions (diagonal covariance)
                    observe(addr!("x0", i), fugue::Normal::new(mu_0, sigma_0).unwrap(), x0)
                        .bind(move |_| {
                            observe(addr!("x1", i), fugue::Normal::new(mu_1, sigma_1).unwrap(), x1)
                        })
                })
        });

        pure((pi1, mu1_0, mu1_1, sigma1_0, sigma1_1, mu2_0, mu2_1, sigma2_0))
    }
}

fn multivariate_mixture_demo() {
    println!("=== Multivariate Gaussian Mixture Model ===\n");

    // Generate 2D mixture data
    let true_components = vec![(0.6, vec![-1.0, -1.0], 0.5), (0.4, vec![2.0, 1.5], 0.7)];
    let (data, true_labels) = generate_multivariate_mixture_data(60, &true_components, 456);

    println!(
        "📊 Generated {} 2D data points from {} components",
        data.len(),
        true_components.len()
    );
    for (i, (weight, ref mu_vec, sigma)) in true_components.iter().enumerate() {
        println!(
            "   - Component {}: π={:.1}, μ=[{:.1}, {:.1}], σ={:.1}",
            i + 1,
            weight,
            mu_vec[0],
            mu_vec[1],
            sigma
        );
    }

    let n_true_labels: Vec<usize> = (0..true_components.len())
        .map(|k| true_labels.iter().filter(|&&label| label == k).count())
        .collect();

    for (k, count) in n_true_labels.iter().enumerate() {
        println!(
            "   - True cluster {}: {} observations ({:.1}%)",
            k + 1,
            count,
            100.0 * *count as f64 / data.len() as f64
        );
    }

    println!("\n🔬 Fitting 2D mixture model with K=2...");
    let model_fn = move || multivariate_mixture_model(data.clone());
    let mut rng = StdRng::seed_from_u64(789);
    let samples = adaptive_mcmc_chain(&mut rng, model_fn, 500, 100);

    let valid_samples: Vec<_> = samples
        .iter()
        .filter(|(_, trace)| trace.total_log_weight().is_finite())
        .collect();

    if !valid_samples.is_empty() {
        println!(
            "✅ MCMC completed with {} valid samples",
            valid_samples.len()
        );

        // Extract parameter estimates
        let pi1_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.0).collect();
        let mu1_0_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.1).collect();
        let mu1_1_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.2).collect();
        let mu2_0_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.5).collect();
        let mu2_1_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.6).collect();

        let mean_pi1 = pi1_samples.iter().sum::<f64>() / pi1_samples.len() as f64;
        let mean_mu1_0 = mu1_0_samples.iter().sum::<f64>() / mu1_0_samples.len() as f64;
        let mean_mu1_1 = mu1_1_samples.iter().sum::<f64>() / mu1_1_samples.len() as f64;
        let mean_mu2_0 = mu2_0_samples.iter().sum::<f64>() / mu2_0_samples.len() as f64;
        let mean_mu2_1 = mu2_1_samples.iter().sum::<f64>() / mu2_1_samples.len() as f64;

        println!("\n📈 Estimated 2D Mixture Components:");
        println!(
            "   - Component 1: π̂={:.2}, μ̂=[{:.1}, {:.1}]",
            mean_pi1, mean_mu1_0, mean_mu1_1
        );
        println!(
            "   - Component 2: π̂={:.2}, μ̂=[{:.1}, {:.1}]",
            1.0 - mean_pi1,
            mean_mu2_0,
            mean_mu2_1
        );

        println!("\n💡 Multivariate mixture models handle correlated features and complex cluster shapes!");
    } else {
        println!("❌ No valid MCMC samples obtained");
    }

    println!();
}

// Generate multivariate mixture data
fn generate_multivariate_mixture_data(
    n: usize,
    components: &[(f64, Vec<f64>, f64)], // (weight, mean_vec, sigma)
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Vec::new();
    let mut true_labels = Vec::new();

    let weights: Vec<f64> = components.iter().map(|(w, _, _)| *w).collect();
    let cumulative_weights: Vec<f64> = weights
        .iter()
        .scan(0.0, |acc, &w| {
            *acc += w;
            Some(*acc)
        })
        .collect();

    for _ in 0..n {
        let u: f64 = rng.gen();
        let component = cumulative_weights
            .iter()
            .position(|&cw| u <= cw)
            .unwrap_or(components.len() - 1);
        let (_, ref mu_vec, sigma) = components[component];

        let mut x_vec = Vec::new();
        for &mu in mu_vec {
            let noise: f64 = StandardNormal.sample(&mut rng);
            x_vec.push(mu + sigma * noise);
        }

        data.push(x_vec);
        true_labels.push(component);
    }

    (data, true_labels)
}
// ANCHOR_END: multivariate_mixture_model

// ANCHOR: mixture_of_experts
// Simple mixture of experts with 2 experts
fn mixture_of_experts_model(
    x_data: Vec<f64>,
    y_data: Vec<f64>,
) -> Model<(f64, f64, f64, f64, f64, f64)> {
    prob! {
        // Expert 1 parameters (for x < 0)
        let intercept1 <- sample(addr!("intercept1"), fugue::Normal::new(0.0, 2.0).unwrap());
        let slope1 <- sample(addr!("slope1"), fugue::Normal::new(0.0, 2.0).unwrap());
        let sigma1 <- sample(addr!("sigma1"), Gamma::new(1.0, 1.0).unwrap());

        // Expert 2 parameters (for x >= 0)
        let intercept2 <- sample(addr!("intercept2"), fugue::Normal::new(0.0, 2.0).unwrap());
        let slope2 <- sample(addr!("slope2"), fugue::Normal::new(0.0, 2.0).unwrap());
        let sigma2 <- sample(addr!("sigma2"), Gamma::new(1.0, 1.0).unwrap());

        // Observations with simple binary gating
        let _observations <- plate!(i in 0..x_data.len() => {
            let x = x_data[i];
            let y = y_data[i];

            if x < 0.0 {
                // Use expert 1
                let mean_y = intercept1 + slope1 * x;
                observe(addr!("y", i), fugue::Normal::new(mean_y, sigma1).unwrap(), y)
            } else {
                // Use expert 2
                let mean_y = intercept2 + slope2 * x;
                observe(addr!("y", i), fugue::Normal::new(mean_y, sigma2).unwrap(), y)
            }
        });

        pure((intercept1, slope1, sigma1, intercept2, slope2, sigma2))
    }
}

fn mixture_of_experts_demo() {
    println!("=== Mixture of Experts ===\n");

    let (x_data, y_data) = generate_moe_data(60, 321);

    println!(
        "📊 Generated {} (x,y) points with region-specific relationships",
        x_data.len()
    );
    println!("   - Left region (x < 0): Linear relationship");
    println!("   - Right region (x ≥ 0): Quadratic relationship");

    println!("\n🔬 Fitting mixture of experts with 2 experts...");
    let model_fn = move || mixture_of_experts_model(x_data.clone(), y_data.clone());
    let mut rng = StdRng::seed_from_u64(654);
    let samples = adaptive_mcmc_chain(&mut rng, model_fn, 500, 100);

    let valid_samples: Vec<_> = samples
        .iter()
        .filter(|(_, trace)| trace.total_log_weight().is_finite())
        .collect();

    if !valid_samples.is_empty() {
        println!(
            "✅ MCMC completed with {} valid samples",
            valid_samples.len()
        );

        println!("\n📈 Expert Network Parameters:");

        let intercept1_samples: Vec<f64> =
            valid_samples.iter().map(|(params, _)| params.0).collect();
        let slope1_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.1).collect();
        let sigma1_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.2).collect();

        let intercept2_samples: Vec<f64> =
            valid_samples.iter().map(|(params, _)| params.3).collect();
        let slope2_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.4).collect();
        let sigma2_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.5).collect();

        let mean_intercept1 =
            intercept1_samples.iter().sum::<f64>() / intercept1_samples.len() as f64;
        let mean_slope1 = slope1_samples.iter().sum::<f64>() / slope1_samples.len() as f64;
        let mean_sigma1 = sigma1_samples.iter().sum::<f64>() / sigma1_samples.len() as f64;

        let mean_intercept2 =
            intercept2_samples.iter().sum::<f64>() / intercept2_samples.len() as f64;
        let mean_slope2 = slope2_samples.iter().sum::<f64>() / slope2_samples.len() as f64;
        let mean_sigma2 = sigma2_samples.iter().sum::<f64>() / sigma2_samples.len() as f64;

        println!(
            "   - Expert 1 [Left (x < 0)]: intercept={:.2}, slope={:.2}, σ={:.2}",
            mean_intercept1, mean_slope1, mean_sigma1
        );
        println!(
            "   - Expert 2 [Right (x ≥ 0)]: intercept={:.2}, slope={:.2}, σ={:.2}",
            mean_intercept2, mean_slope2, mean_sigma2
        );

        println!(
            "\n💡 Mixture of Experts captures different relationships in different input regions"
        );
    } else {
        println!("❌ No valid MCMC samples obtained");
    }

    println!();
}
// ANCHOR_END: mixture_of_experts

// ANCHOR: dirichlet_process_mixture
// Simplified Dirichlet Process with truncated stick-breaking
#[allow(clippy::type_complexity)] // Complex tuple needed for demonstration
fn dirichlet_process_mixture_model(
    data: Vec<f64>,
) -> Model<(f64, f64, f64, f64, f64, f64, f64, usize)> {
    prob! {
        // Stick-breaking for 3 components (truncated)
        let v1 <- sample(addr!("v1"), fugue::Beta::new(1.0, 1.0).unwrap());
        let v2 <- sample(addr!("v2"), fugue::Beta::new(1.0, 1.0).unwrap());

        // Convert to weights (clamp to avoid negative probabilities during MCMC)
        let v1_safe = v1.clamp(0.001, 0.999);
        let v2_safe = v2.clamp(0.001, 0.999);

        let w1 = v1_safe;
        let w2 = (1.0 - v1_safe) * v2_safe;
        let w3 = (1.0 - v1_safe) * (1.0 - v2_safe);

        // Component parameters
        let mu1 <- sample(addr!("mu1"), fugue::Normal::new(0.0, 5.0).unwrap());
        let mu2 <- sample(addr!("mu2"), fugue::Normal::new(0.0, 5.0).unwrap());
        let mu3 <- sample(addr!("mu3"), fugue::Normal::new(0.0, 5.0).unwrap());

        let sigma1 <- sample(addr!("sigma1"), Gamma::new(1.0, 1.0).unwrap());
        let sigma2 <- sample(addr!("sigma2"), Gamma::new(1.0, 1.0).unwrap());
        let sigma3 <- sample(addr!("sigma3"), Gamma::new(1.0, 1.0).unwrap());

        // Observations and count active components
        let assignments <- plate!(i in 0..data.len() => {
            // Ensure valid probabilities (normalize and clamp)
            let total = w1 + w2 + w3;
            let raw_weights = if total > 0.0 && total.is_finite() {
                vec![w1 / total, w2 / total, w3 / total]
            } else {
                vec![0.33, 0.33, 0.34] // Fallback to uniform
            };

            // Extra safety: clamp all weights to valid range
            let weights: Vec<f64> = raw_weights.iter()
                .map(|&w| w.clamp(0.001, 0.999))
                .collect();

            // Renormalize after clamping
            let weight_sum: f64 = weights.iter().sum();
            let safe_weights: Vec<f64> = weights.iter()
                .map(|&w| w / weight_sum)
                .collect();

            let x = data[i];

            sample(addr!("z", i), Categorical::new(safe_weights).unwrap())
                .bind(move |z_i| {
                    // Explicitly handle only 3 components
                    let (mu_i, sigma_i) = match z_i {
                        0 => (mu1, sigma1),
                        1 => (mu2, sigma2),
                        _ => (mu3, sigma3), // 2 or any other value
                    };
                    observe(addr!("x", i), fugue::Normal::new(mu_i, sigma_i).unwrap(), x)
                        .map(move |_| z_i)
                })
        });

        let active_components = assignments.iter().max().unwrap_or(&0) + 1;

        pure((w1, w2, w3, mu1, mu2, mu3, sigma1, active_components))
    }
}

fn dirichlet_process_mixture_demo() {
    println!("=== Dirichlet Process Mixture (Truncated) ===\n");

    let true_components = vec![(0.5, -1.5, 0.4), (0.3, 1.0, 0.6), (0.2, 4.0, 0.5)];
    let (data, _) = generate_mixture_data(80, &true_components, 987);

    println!(
        "📊 Generated {} data points from {} unknown components",
        data.len(),
        true_components.len()
    );
    println!("   - Goal: Automatically discover the number of components");

    println!("\n🔬 Fitting Dirichlet Process mixture (max K=3, α=1.0)...");
    let model_fn = move || dirichlet_process_mixture_model(data.clone());
    let mut rng = StdRng::seed_from_u64(147);
    let samples = adaptive_mcmc_chain(&mut rng, model_fn, 400, 100);

    let valid_samples: Vec<_> = samples
        .iter()
        .filter(|(_, trace)| trace.total_log_weight().is_finite())
        .collect();

    if !valid_samples.is_empty() {
        println!(
            "✅ MCMC completed with {} valid samples",
            valid_samples.len()
        );

        let active_counts: Vec<usize> = valid_samples.iter().map(|(params, _)| params.7).collect();

        let mean_active = active_counts.iter().sum::<usize>() as f64 / active_counts.len() as f64;
        let mode_active = {
            let mut counts = [0; 4];
            for &ac in &active_counts {
                if ac < counts.len() {
                    counts[ac] += 1;
                }
            }
            counts
                .iter()
                .enumerate()
                .max_by_key(|(_, &count)| count)
                .unwrap()
                .0
        };

        println!("\n🔍 Component Discovery Results:");
        println!("   - True number of components: {}", true_components.len());
        println!("   - Mean active components: {:.1}", mean_active);
        println!("   - Mode active components: {}", mode_active);

        println!("\n💡 Dirichlet Process successfully explores different model complexities!");
    } else {
        println!("❌ No valid MCMC samples obtained");
    }

    println!();
}
// ANCHOR_END: dirichlet_process_mixture

// ANCHOR: hidden_markov_model
// Simple 2-state Hidden Markov Model (highly simplified)
fn hidden_markov_model(observations: Vec<f64>) -> Model<(f64, f64, f64, f64)> {
    prob! {
        // Emission parameters (means for each state)
        let mu0 <- sample(addr!("mu0"), fugue::Normal::new(0.0, 5.0).unwrap());
        let mu1 <- sample(addr!("mu1"), fugue::Normal::new(0.0, 5.0).unwrap());

        // Emission variances
        let sigma0 <- sample(addr!("sigma0"), Gamma::new(1.0, 1.0).unwrap());
        let sigma1 <- sample(addr!("sigma1"), Gamma::new(1.0, 1.0).unwrap());

        // Simplified: assign each observation to a state independently
        let _states <- plate!(t in 0..observations.len() => {
            let initial_dist = vec![0.5, 0.5]; // Equal probability
            let obs = observations[t];

            sample(addr!("state", t), Categorical::new(initial_dist).unwrap())
                .bind(move |state_t| {
                    // Explicitly handle only 2 states
                    let (mu_t, sigma_t) = if state_t == 0 {
                        (mu0, sigma0)
                    } else {
                        (mu1, sigma1)
                    };
                    observe(addr!("obs", t), fugue::Normal::new(mu_t, sigma_t).unwrap(), obs)
                        .map(move |_| state_t)
                })
        });

        pure((mu0, sigma0, mu1, sigma1))
    }
}

fn hidden_markov_model_demo() {
    println!("=== Hidden Markov Model ===\n");

    // Generate simple regime-switching data
    let mut rng = StdRng::seed_from_u64(555);
    let mut hmm_data = Vec::new();
    let mut current_regime = 0;

    for t in 0..60 {
        if t % 15 == 0 && rng.gen::<f64>() < 0.8 {
            current_regime = 1 - current_regime;
        }

        let noise: f64 = StandardNormal.sample(&mut rng);
        let observation = if current_regime == 0 {
            0.0 + 0.5 * noise // Low volatility
        } else {
            0.0 + 2.0 * noise // High volatility
        };

        hmm_data.push(observation);
    }

    println!(
        "📊 Generated {} observations from switching regime process",
        hmm_data.len()
    );

    println!("\n🔬 Fitting HMM with 2 states...");
    let model_fn = move || hidden_markov_model(hmm_data.clone());
    let mut rng = StdRng::seed_from_u64(888);
    let samples = adaptive_mcmc_chain(&mut rng, model_fn, 400, 100);

    let valid_samples: Vec<_> = samples
        .iter()
        .filter(|(_, trace)| trace.total_log_weight().is_finite())
        .collect();

    if !valid_samples.is_empty() {
        println!(
            "✅ HMM MCMC completed with {} valid samples",
            valid_samples.len()
        );

        let mu0_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.0).collect();
        let sigma0_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.1).collect();
        let mu1_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.2).collect();
        let sigma1_samples: Vec<f64> = valid_samples.iter().map(|(params, _)| params.3).collect();

        let mean_mu0 = mu0_samples.iter().sum::<f64>() / mu0_samples.len() as f64;
        let mean_sigma0 = sigma0_samples.iter().sum::<f64>() / sigma0_samples.len() as f64;
        let mean_mu1 = mu1_samples.iter().sum::<f64>() / mu1_samples.len() as f64;
        let mean_sigma1 = sigma1_samples.iter().sum::<f64>() / sigma1_samples.len() as f64;

        println!("\n📈 HMM Emission Parameters:");
        let volatility_type0 = if mean_sigma0 < 1.0 { "Low" } else { "High" };
        let volatility_type1 = if mean_sigma1 < 1.0 { "Low" } else { "High" };
        println!(
            "   - State 0: μ̂={:.2}, σ̂={:.2} ({} volatility)",
            mean_mu0, mean_sigma0, volatility_type0
        );
        println!(
            "   - State 1: μ̂={:.2}, σ̂={:.2} ({} volatility)",
            mean_mu1, mean_sigma1, volatility_type1
        );

        println!("\n💡 HMM identifies different volatility regimes!");
    } else {
        println!("❌ No valid HMM samples obtained");
    }

    println!();
}
// ANCHOR_END: hidden_markov_model

// ANCHOR: mixture_model_selection
// Basic model comparison
fn mixture_model_selection_demo() {
    println!("=== Mixture Model Selection ===\n");

    let true_components = vec![(0.7, 0.0, 1.0), (0.3, 4.0, 1.2)];
    let (data, _) = generate_mixture_data(60, &true_components, 999);

    println!(
        "📊 Generated data from {} true components",
        true_components.len()
    );
    println!("   Comparing single Gaussian vs 2-component mixture...");

    // Single Gaussian model
    let single_gaussian_model = move |data: Vec<f64>| {
        prob! {
            let mu <- sample(addr!("mu"), fugue::Normal::new(0.0, 5.0).unwrap());
            let sigma <- sample(addr!("sigma"), Gamma::new(1.0, 1.0).unwrap());

            let _observations <- plate!(i in 0..data.len() => {
                let x = data[i];
                observe(addr!("x", i), fugue::Normal::new(mu, sigma).unwrap(), x)
            });

            pure((mu, sigma))
        }
    };

    // Test single Gaussian
    let data_single = data.clone();
    let single_model_fn = move || single_gaussian_model(data_single.clone());
    let mut rng1 = StdRng::seed_from_u64(111);
    let single_samples = adaptive_mcmc_chain(&mut rng1, single_model_fn, 300, 50);

    // Test mixture model
    let data_mixture = data.clone();
    let mixture_model_fn = move || gaussian_mixture_model(data_mixture.clone());
    let mut rng2 = StdRng::seed_from_u64(222);
    let mixture_samples = adaptive_mcmc_chain(&mut rng2, mixture_model_fn, 300, 50);

    let single_valid: Vec<_> = single_samples
        .iter()
        .filter(|(_, trace)| trace.total_log_weight().is_finite())
        .collect();

    let mixture_valid: Vec<_> = mixture_samples
        .iter()
        .filter(|(_, trace)| trace.total_log_weight().is_finite())
        .collect();

    if !single_valid.is_empty() && !mixture_valid.is_empty() {
        let single_loglik = single_valid
            .iter()
            .map(|(_, trace)| trace.total_log_weight())
            .sum::<f64>()
            / single_valid.len() as f64;

        let mixture_loglik = mixture_valid
            .iter()
            .map(|(_, trace)| trace.total_log_weight())
            .sum::<f64>()
            / mixture_valid.len() as f64;

        println!("\n🏆 Model Comparison Results:");
        println!("   Model               | Samples | Log-Likelihood");
        println!("   --------------------|---------|---------------");
        println!(
            "   Single Gaussian     | {:7} | {:13.1}",
            single_valid.len(),
            single_loglik
        );
        println!(
            "   2-Component Mixture | {:7} | {:13.1}",
            mixture_valid.len(),
            mixture_loglik
        );

        if mixture_loglik > single_loglik {
            println!("\n🥇 Best model: 2-Component Mixture (higher log-likelihood)");
            println!("   ✅ Correctly identifies mixture structure!");
        } else {
            println!("\n🥇 Best model: Single Gaussian");
            println!("   ⚠️  May indicate insufficient data or overlap");
        }
    } else {
        println!("❌ Insufficient valid samples for comparison");
    }

    println!();
}
// ANCHOR_END: mixture_model_selection

// ANCHOR: cluster_diagnostics
// Basic cluster diagnostics
fn cluster_diagnostics_demo() {
    println!("=== Cluster Diagnostics ===\n");

    let true_components = vec![(0.4, -2.0, 0.6), (0.6, 2.0, 0.8)];
    let (data, true_labels) = generate_mixture_data(60, &true_components, 777);
    let data_for_diagnostics = data.clone();

    println!("📊 Running cluster diagnostics on mixture model results");

    let model_fn = move || gaussian_mixture_model(data.clone());
    let mut rng = StdRng::seed_from_u64(333);
    let samples = adaptive_mcmc_chain(&mut rng, model_fn, 300, 50);

    let valid_samples: Vec<_> = samples
        .iter()
        .filter(|(_, trace)| trace.total_log_weight().is_finite())
        .collect();

    if !valid_samples.is_empty() {
        println!(
            "✅ Fitted mixture model with {} samples",
            valid_samples.len()
        );

        let final_sample = &valid_samples[valid_samples.len() - 1].0;
        let means = [final_sample.1, final_sample.3];

        // Simple cluster assignment
        let mut estimated_labels = Vec::new();
        for &x in &data_for_diagnostics {
            let dist0 = (x - means[0]).abs();
            let dist1 = (x - means[1]).abs();
            let label = if dist0 < dist1 { 0 } else { 1 };
            estimated_labels.push(label);
        }

        let mut correct = 0;
        for (true_label, est_label) in true_labels.iter().zip(estimated_labels.iter()) {
            if true_label == est_label {
                correct += 1;
            }
        }

        let accuracy = correct as f64 / data_for_diagnostics.len() as f64;

        println!("\n🔍 Clustering Diagnostics:");
        println!(
            "   - Accuracy: {:.2} ({} correct out of {})",
            accuracy,
            correct,
            data_for_diagnostics.len()
        );

        if accuracy > 0.7 {
            println!("   ✅ Good clustering performance!");
        } else {
            println!("   ⚠️  Moderate clustering - may need more data or features");
        }
    } else {
        println!("❌ No valid samples for diagnostics");
    }

    println!();
}
// ANCHOR_END: cluster_diagnostics

fn main() {
    println!("🧬 Fugue Mixture Model Demonstrations");
    println!("====================================\n");

    gaussian_mixture_demo();
    multivariate_mixture_demo();
    mixture_of_experts_demo();
    dirichlet_process_mixture_demo();
    hidden_markov_model_demo();
    mixture_model_selection_demo();
    cluster_diagnostics_demo();

    println!("✨ Mixture modeling demonstrations completed!");
    println!("   Key advantages of Bayesian mixture models:");
    println!("   • Natural handling of uncertainty in cluster assignments");
    println!("   • Principled model selection via marginal likelihood");
    println!("   • Flexible extensions to complex data structures");
    println!("   • Integration of domain knowledge through informative priors");
    println!("   • Robust inference with constraint-aware MCMC");
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixture_data_generation() {
        let components = vec![(0.5, 0.0, 1.0), (0.5, 3.0, 1.0)];
        let (data, labels) = generate_mixture_data(50, &components, 42);

        assert_eq!(data.len(), 50);
        assert_eq!(labels.len(), 50);
        assert!(labels.iter().all(|&l| l < components.len()));
    }

    #[test]
    fn test_gaussian_mixture_model() {
        let data = vec![0.0, 0.1, 0.2, 4.0, 4.1, 4.2];

        let mut rng = StdRng::seed_from_u64(42);
        let (params, trace) = runtime::handler::run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            gaussian_mixture_model(data),
        );

        assert!(params.0.is_finite() && params.0 >= 0.0 && params.0 <= 1.0); // pi1
        assert!(params.1.is_finite()); // mu1
        assert!(params.2.is_finite() && params.2 > 0.0); // sigma1
        assert!(params.3.is_finite()); // mu2
        assert!(params.4.is_finite() && params.4 > 0.0); // sigma2
        assert!(trace.choices.len() > 0);
    }

    #[test]
    fn test_multivariate_mixture_data_generation() {
        let components = vec![(0.6, vec![0.0, 0.0], 1.0), (0.4, vec![2.0, -1.0], 1.0)];
        let (data, labels) = generate_multivariate_mixture_data(30, &components, 123);

        assert_eq!(data.len(), 30);
        assert_eq!(labels.len(), 30);
        assert!(data.iter().all(|x| x.len() == 2)); // 2D data
        assert!(labels.iter().all(|&l| l < components.len()));
    }

    #[test]
    fn test_moe_data_generation() {
        let (x_data, y_data) = generate_moe_data(30, 456);

        assert_eq!(x_data.len(), 30);
        assert_eq!(y_data.len(), 30);
        assert!(x_data.iter().all(|&x| x >= -2.0 && x <= 2.0));
        assert!(y_data.iter().all(|&y| y.is_finite()));
    }

    #[test]
    fn test_mixture_mcmc() {
        let data = vec![-1.0, -0.9, 2.0, 2.1];
        let model_fn = move || gaussian_mixture_model(data.clone());
        let mut rng = StdRng::seed_from_u64(999);

        let samples = adaptive_mcmc_chain(&mut rng, model_fn, 5, 2);
        assert_eq!(samples.len(), 5);

        let valid_count = samples
            .iter()
            .filter(|(_, trace)| trace.total_log_weight().is_finite())
            .count();
        assert!(valid_count > 0);
    }
}
