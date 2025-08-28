use fugue::*;
use rand::thread_rng;

fn main() {
    let mut rng = thread_rng();
    
    println!("=== Advanced Distribution Patterns ===\n");
    
    println!("1. Hierarchical Priors");
    println!("----------------------");
    // ANCHOR: hierarchical_priors  
    // Hierarchical prior structure
    let global_mean = Normal::new(0.0, 10.0).unwrap();
    let mu = global_mean.sample(&mut rng);
    
    let group_precision = Gamma::new(2.0, 0.5).unwrap();
    let tau = group_precision.sample(&mut rng);
    let sigma = (1.0 / tau).sqrt(); // Convert precision to std dev
    
    // Individual observations from hierarchical model
    let individual = Normal::new(mu, sigma).unwrap();
    let observation = individual.sample(&mut rng);
    
    println!("🌐 Global mean: {:.3}", mu);
    println!("📊 Group std dev: {:.3}", sigma); 
    println!("👤 Individual observation: {:.3}", observation);
    // ANCHOR_END: hierarchical_priors
    println!("✓ Hierarchical structure allows sharing information across groups");
    println!();
    
    println!("2. Mixture Model Components");
    println!("---------------------------");
    // ANCHOR: mixture_components
    // Mixture model components
    let mixture_weights = vec![0.6, 0.3, 0.1];
    let component_selector = Categorical::new(mixture_weights).unwrap();
    let selected_component: usize = component_selector.sample(&mut rng);
    
    // Different components
    let components = vec![
        Normal::new(-2.0, 0.5).unwrap(),
        Normal::new(0.0, 1.0).unwrap(), 
        Normal::new(3.0, 0.8).unwrap(),
    ];
    
    let sample = components[selected_component].sample(&mut rng);
    println!("🎯 Selected component {}: sample = {:.3}", selected_component, sample);
    // ANCHOR_END: mixture_components
    println!("✓ Mixture models capture multi-modal distributions");
    println!();
    
    println!("3. Conjugate Prior Updates");
    println!("--------------------------");
    // ANCHOR: conjugate_pairs
    // Beta-Bernoulli conjugacy
    let prior_alpha = 2.0;
    let prior_beta = 8.0;
    let prior = Beta::new(prior_alpha, prior_beta).unwrap();
    let p: f64 = prior.sample(&mut rng);
    
    // Simulate some trials
    let trials = 20;
    let mut successes = 0;
    let bernoulli = Bernoulli::new(p).unwrap();
    
    for _ in 0..trials {
        if bernoulli.sample(&mut rng) {
            successes += 1;
        }
    }
    
    // Posterior parameters (conjugate update)
    let posterior_alpha = prior_alpha + successes as f64;
    let posterior_beta = prior_beta + (trials - successes) as f64;
    let posterior = Beta::new(posterior_alpha, posterior_beta).unwrap();
    let updated_p = posterior.sample(&mut rng);
    
    println!("🎲 Prior p: {:.3}", p);
    println!("📈 Observed: {}/{} successes", successes, trials);
    println!("🔄 Posterior p: {:.3}", updated_p);
    // ANCHOR_END: conjugate_pairs
    println!("✓ Conjugate priors enable exact Bayesian updates");
    println!();
    
    println!("4. Robust Modeling with Heavy Tails");
    println!("-----------------------------------");
    // ANCHOR: robust_modeling
    // Robust modeling with heavy tails
    
    // Compare normal vs robust alternatives
    let normal_model = Normal::new(0.0, 1.0).unwrap();
    let normal_sample = normal_model.sample(&mut rng);
    
    // Student-t approximation using mixture
    let df = 3.0; // Degrees of freedom
    let scale_mixture = Gamma::new(df / 2.0, df / 2.0).unwrap();
    let precision = scale_mixture.sample(&mut rng);
    let robust_model = Normal::new(0.0, (1.0 / precision).sqrt()).unwrap();
    let robust_sample = robust_model.sample(&mut rng);
    
    println!("📏 Normal sample: {:.3}", normal_sample);
    println!("🛡️  Robust sample: {:.3}", robust_sample);
    // ANCHOR_END: robust_modeling
    println!("✓ Heavy-tailed distributions are less sensitive to outliers");
    println!();
    
    println!("5. Count Data Regression");
    println!("------------------------");
    // ANCHOR: count_regression
    // Poisson regression structure
    let baseline_rate: f64 = 2.0;
    let covariate_effect = Normal::new(0.0, 0.5).unwrap().sample(&mut rng);
    
    // Simulate covariates
    let x = Normal::new(0.0, 1.0).unwrap().sample(&mut rng);
    
    // Link function: log-linear model
    let log_rate = baseline_rate.ln() + covariate_effect * x;
    let rate = log_rate.exp();
    
    let count_model = Poisson::new(rate).unwrap();
    let observed_count = count_model.sample(&mut rng);
    
    println!("📊 Covariate: {:.3}", x);
    println!("⚡ Rate: {:.3}", rate);
    println!("🔢 Count: {}", observed_count);
    // ANCHOR_END: count_regression
    println!("✓ Log-linear models ensure positive rates for count data");
    println!();
    
    println!("6. Time Series with Innovations");
    println!("-------------------------------");
    // ANCHOR: time_series_innovations
    // AR(1) with normal innovations
    let phi = 0.8; // Autoregressive coefficient
    let innovation_std = 0.3;
    let innovation_dist = Normal::new(0.0, innovation_std).unwrap();
    
    // Simulate AR(1) series
    let mut series = vec![0.0]; // Initial value
    
    for t in 1..10 {
        let innovation = innovation_dist.sample(&mut rng);
        let next_value = phi * series[t - 1] + innovation;
        series.push(next_value);
    }
    
    println!("📈 AR(1) series: {:?}", series.iter()
        .map(|x| format!("{:.2}", x))
        .collect::<Vec<_>>());
    // ANCHOR_END: time_series_innovations
    println!("✓ Autoregressive models capture temporal dependencies");
    println!();
    
    println!("7. Distribution Transformations");
    println!("-------------------------------");
    // ANCHOR: transformation_techniques
    // Log-normal via transformation
    let log_normal_base = Normal::new(2.0, 0.5).unwrap();
    let log_sample = log_normal_base.sample(&mut rng);
    let lognormal_sample = log_sample.exp();
    
    println!("💰 Log-normal sample: {:.3}", lognormal_sample);
    
    // Logit transformation for probabilities
    let logit_normal = Normal::new(0.0, 1.0).unwrap();
    let logit_sample = logit_normal.sample(&mut rng);
    let prob_sample = 1.0 / (1.0 + (-logit_sample).exp());
    
    println!("🎯 Probability via logit: {:.3}", prob_sample);
    // ANCHOR_END: transformation_techniques
    println!("✓ Transformations create new distributions from existing ones");
    println!();
    
    println!("=== All advanced patterns demonstrated successfully! ===");
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    
    // ANCHOR: advanced_testing
    #[test] 
    fn test_conjugate_updates() {
        let mut rng = StdRng::seed_from_u64(42);
        
        // Test Beta-Bernoulli conjugacy
        let prior = Beta::new(1.0, 1.0).unwrap(); // Uniform prior
        let p = 0.7; // True parameter
        
        // Simulate data
        let bernoulli = Bernoulli::new(p).unwrap();
        let mut successes = 0;
        let trials = 100;
        
        for _ in 0..trials {
            if bernoulli.sample(&mut rng) {
                successes += 1;
            }
        }
        
        // Posterior should concentrate around true value
        let posterior = Beta::new(1.0 + successes as f64, 1.0 + (trials - successes) as f64).unwrap();
        let posterior_mean = (1.0 + successes as f64) / (2.0 + trials as f64);
        
        // Should be close to true value with high probability
        assert!((posterior_mean - p).abs() < 0.1);
    }
    // ANCHOR_END: advanced_testing
}
