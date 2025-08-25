//! # Basic Bayesian Coin Flip
//! 
//! **Tutorial**: [Bayesian Coin Flip Tutorial](../docs/src/tutorials/bayesian-coin-flip.md)  
//! **Section**: Step 1: Basic Model Implementation  
//! **Level**: Beginner  
//! **Concepts**: Prior, Likelihood, Posterior sampling
//! 
//! This example demonstrates the simplest possible Bayesian coin flip model.
//! You flip a coin 10 times and observe 7 heads. What's the coin's bias?

use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn simple_coin_model(observed_heads: u64, total_flips: u64) -> Model<f64> {
    prob! {
        // Prior: uniform belief about coin bias
        let bias <- sample(addr!("bias"), Beta::new(1.0, 1.0).unwrap());
        
        // Likelihood: observe the data
        observe(
            addr!("heads"), 
            Binomial::new(total_flips, bias).unwrap(), 
            observed_heads
        );
        
        // Return the parameter we're interested in
        pure(bias)
    }
}

fn main() {
    println!("🪙 Bayesian Coin Flip Analysis");
    println!("================================");
    
    // Our data: 7 heads out of 10 flips
    let heads = 7;
    let flips = 10;
    
    // Sample from the posterior
    println!("\n📊 Posterior Samples:");
    for i in 0..5 {
        let mut rng = StdRng::seed_from_u64(i);
        let (bias, trace) = runtime::handler::run(
            runtime::interpreters::PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            simple_coin_model(heads, flips),
        );
        
        println!("Sample {}: bias = {:.3}, log prob = {:.4}", 
                 i + 1, bias, trace.total_log_weight());
    }
}
