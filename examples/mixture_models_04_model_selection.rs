use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn fit_k_component_model(data: Vec<f64>, k: usize) -> (f64, usize) {
    // Simplified model fitting for comparison
    // Returns (average log-likelihood, number of parameters)

    match k {
        1 => {
            // Single Gaussian model
            let model = || {
                prob! {
                    let mu <- sample(addr!("mu"), Normal::new(50.0, 50.0).unwrap());
                    let sigma <- sample(addr!("sigma"), Exponential::new(0.1).unwrap());

                    for (i, &x) in data.iter().enumerate() {
                        observe(addr!("x", i), Normal::new(mu, sigma).unwrap(), x);
                    }

                    pure((mu, sigma))
                }
            };

            let mut rng = StdRng::seed_from_u64(42);
            let samples = inference::mcmc::adaptive_mcmc_chain(&mut rng, model, 2000, 1000);
            let avg_ll = samples.iter().map(|(_, trace)| trace.total_log_weight()).sum::<f64>()
                / samples.len() as f64;
            (avg_ll, 2)  // 2 parameters: mu, sigma
        },

        2 => {
            // Use our 2-component model
            let model = || two_component_mixture_model(data.clone());
            let mut rng = StdRng::seed_from_u64(42);
            let samples = inference::mcmc::adaptive_mcmc_chain(&mut rng, model, 2000, 1000);
            let avg_ll = samples.iter().map(|(_, trace)| trace.total_log_weight()).sum::<f64>()
                / samples.len() as f64;
            (avg_ll, 5)  // 5 parameters: weight, mu1, mu2, sigma1, sigma2
        },

        3 => {
            // Use our 3-component model
            let model = || three_component_mixture_model(data.clone());
            let mut rng = StdRng::seed_from_u64(42);
            let samples = inference::mcmc::adaptive_mcmc_chain(&mut rng, model, 2000, 1000);
            let avg_ll = samples.iter().map(|(_, trace)| trace.total_log_weight()).sum::<f64>()
                / samples.len() as f64;
            (avg_ll, 8)  // 8 parameters: 2 weights + 3 mus + 3 sigmas
        },

        _ => panic!("Only K=1,2,3 implemented for this tutorial")
    }
}

fn model_selection_analysis() {
    let data = generate_mixture_data(250);

    println!("🏆 Model Selection Analysis");
    println!("===========================");

    let n = data.len() as f64;
    let mut results = Vec::new();

    for k in 1..=3 {
        println!("\n🔄 Fitting {}-component model...", k);
        let (log_likelihood, n_params) = fit_k_component_model(data.clone(), k);

        // Compute information criteria
        let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
        let bic = -2.0 * log_likelihood + (n_params as f64) * n.ln();

        println!("  Log-likelihood: {:.2}", log_likelihood);
        println!("  Parameters: {}", n_params);
        println!("  AIC: {:.2}", aic);
        println!("  BIC: {:.2}", bic);

        results.push((k, log_likelihood, aic, bic, n_params));
    }

    // Find best models
    let best_aic = results.iter().min_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).unwrap();
    let best_bic = results.iter().min_by(|a, b| a.3.partial_cmp(&b.3).unwrap()).unwrap();
    let best_ll = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();

    println!("\n🏆 Model Selection Results:");
    println!("  Best by Log-Likelihood: {}-component model (LL={:.2})", best_ll.0, best_ll.1);
    println!("  Best by AIC: {}-component model (AIC={:.2})", best_aic.0, best_aic.2);
    println!("  Best by BIC: {}-component model (BIC={:.2})", best_bic.0, best_bic.3);

    // Model comparison table
    println!("\n📊 Complete Comparison:");
    println!("  K  |  Log-Lik  |   AIC    |   BIC    | Params");
    println!("  ---|-----------|----------|----------|-------");
    for (k, ll, aic, bic, p) in &results {
        let aic_mark = if *k == best_aic.0 { "*" } else { " " };
        let bic_mark = if *k == best_bic.0 { "*" } else { " " };
        println!("  {} |   {:7.2} | {:7.2}{} | {:7.2}{} |   {}",
                 k, ll, aic, aic_mark, bic, bic_mark, p);
    }

    println!("\n💡 Interpretation:");
    if best_aic.0 == best_bic.0 {
        println!("  ✅ Both AIC and BIC agree: {}-component model is optimal", best_aic.0);
    } else {
        println!("  📊 AIC favors {}-component, BIC favors {}-component", best_aic.0, best_bic.0);
        println!("     AIC tends to select more complex models");
        println!("     BIC has stronger penalty for complexity");
    }

    // Practical recommendation
    if best_bic.0 == 3 {
        println!("  🎯 Recommendation: Use 3-component model (matches data generation process)");
    } else {
        println!("  🎯 Recommendation: Use {}-component model (BIC selection)", best_bic.0);
    }
}

// Include previous functions

fn main() {
    model_selection_analysis();
}