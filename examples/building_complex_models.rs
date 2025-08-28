use fugue::*;
use rand::thread_rng;

fn main() {
    let mut rng = thread_rng();
    
    println!("=== Building Complex Models with Fugue ===\n");
    
    println!("1. Basic prob! Macro Usage");
    println!("-------------------------");
    // ANCHOR: basic_prob_macro
    // Simple do-notation style probabilistic program
    let simple_model = prob!(
        let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        let y <- sample(addr!("y"), Normal::new(x, 0.5).unwrap());
        let sum = x + y;  // Regular variable assignment
        pure(sum)
    );
    println!("✅ Created simple model with prob! macro");
    // ANCHOR_END: basic_prob_macro
    println!("   - Uses <- for probabilistic binding");
    println!("   - Uses = for regular assignments");
    println!("   - Returns final value with pure()");
    println!();
    
    println!("2. Plate Notation for Independent Samples");
    println!("------------------------------------------");
    // ANCHOR: plate_notation_basic
    // Independent samples using plate notation
    let vector_model = plate!(i in 0..5 => {
        sample(addr!("sample", i), Normal::new(0.0, 1.0).unwrap())
    });
    println!("✅ Created vectorized model with {} samples", 5);
    
    // Plate with observations
    let observations = vec![1.2, -0.5, 2.1, 0.8, -1.0];
    let n_obs = observations.len();
    let observed_model = plate!(i in 0..n_obs => {
        observe(addr!("obs", i), Normal::new(0.0, 1.0).unwrap(), observations[i])
    });
    println!("✅ Created observation model for {} data points", n_obs);
    // ANCHOR_END: plate_notation_basic
    println!("   - plate! automatically handles indexing");
    println!("   - Each iteration gets unique address");
    println!();
    
    println!("3. Hierarchical Models with Scoped Addresses");
    println!("--------------------------------------------");
    // ANCHOR: hierarchical_scoping
    // Hierarchical model using scoped addresses
    let hierarchical_model = prob!(
        let global_mu <- sample(addr!("global_mu"), Normal::new(0.0, 10.0).unwrap());
        let group_mu <- sample(scoped_addr!("group", "mu", "{}", 0),
                              Normal::new(global_mu, 1.0).unwrap());
        pure((global_mu, group_mu))
    );
    println!("✅ Created hierarchical model with scoped addresses");
    // ANCHOR_END: hierarchical_scoping
    println!("   - scoped_addr! creates organized parameter names");
    println!("   - Hierarchical parameter structure");
    println!();
    
    println!("4. Model Composition with Functions");
    println!("----------------------------------");
    // ANCHOR: model_composition
    // Helper function to create a component model
    fn create_normal_component(name: &str, mean: f64, std: f64) -> Model<f64> {
        sample(addr!(name), Normal::new(mean, std).unwrap())
    }
    
    // Compose multiple components
    let composition_model = prob! {
        let param1 <- create_normal_component("param1", 0.0, 1.0);
        let param2 <- create_normal_component("param2", 2.0, 0.5);
        let combined = param1 * param2;
        pure(combined)
    };
    println!("✅ Created composed model with reusable components");
    // ANCHOR_END: model_composition
    println!("   - Functions return Model<T> for reuse");
    println!("   - Clean separation of concerns");
    println!();
    
    println!("5. Sequential Dependencies");
    println!("-------------------------");
    // ANCHOR: sequential_dependencies
    // Sequential model with dependencies
    let sequential_model = prob! {
        let states <- plate!(t in 0..3 => {
            sample(addr!("x", t), Normal::new(0.0, 1.0).unwrap())
                .bind(move |x_t| {
                    observe(addr!("y", t), Normal::new(x_t, 0.5).unwrap(), 1.0 + t as f64)
                        .map(move |_| x_t)
                })
        });
        
        pure(states)
    };
    println!("✅ Created sequential model with observations");
    // ANCHOR_END: sequential_dependencies
    println!("   - Each time step depends on previous");
    println!("   - Observations condition the model");
    println!();
    
    println!("6. Mixture Models");
    println!("----------------");
    // ANCHOR: mixture_models
    // Mixture model with component selection
    let mixture_model = prob! {
        let component <- sample(addr!("component"), Bernoulli::new(0.3).unwrap());
        let mu = if component { -2.0 } else { 2.0 };
        let x <- sample(addr!("x"), Normal::new(mu, 1.0).unwrap());
        pure((component, x))
    };
    println!("✅ Created mixture model with 2 components");
    // ANCHOR_END: mixture_models
    println!("   - Boolean component selection");
    println!("   - Natural if/else branching");
    println!();
    
    println!("7. Advanced Address Management");
    println!("-----------------------------");
    // ANCHOR: address_management
    // Complex addressing for large models
    let neural_layer_model = plate!(layer in 0..3 => {
        let layer_size = match layer {
            0 => 4,
            1 => 8,
            2 => 1,
            _ => 1,
        };
        
        plate!(i in 0..layer_size => {
            sample(
                scoped_addr!("layer", "weight", "{}_{}", layer, i),
                Normal::new(0.0, 0.1).unwrap()
            )
        })
    });
    println!("✅ Created neural network parameter structure");
    // ANCHOR_END: address_management
    println!("   - Systematic parameter organization");
    println!("   - Hierarchical scoping prevents conflicts");
    println!();
    
    println!("8. Bayesian Linear Regression");
    println!("----------------------------");
    // ANCHOR: bayesian_regression
    // Complete Bayesian linear regression
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = vec![2.1, 3.9, 6.2, 8.1, 9.8];
    let n = x_data.len();
    
    let regression_model = prob! {
        let intercept <- sample(addr!("intercept"), Normal::new(0.0, 10.0).unwrap());
        let slope <- sample(addr!("slope"), Normal::new(0.0, 10.0).unwrap());
        let precision <- sample(addr!("precision"), Gamma::new(1.0, 1.0).unwrap());
        let sigma = (1.0 / precision).sqrt();
        
        let likelihood <- plate!(i in 0..n => {
            let predicted = intercept + slope * x_data[i];
            observe(addr!("y", i), Normal::new(predicted, sigma).unwrap(), y_data[i])
        });
        
        pure((intercept, slope, sigma))
    };
    println!("✅ Created Bayesian linear regression model");
    // ANCHOR_END: bayesian_regression
    println!("   - Proper priors for all parameters");
    println!("   - Vectorized likelihood computation");
    println!();
    
    println!("9. Multi-level Hierarchy");
    println!("-----------------------");
    // ANCHOR: multilevel_hierarchy
    // Simplified hierarchy to avoid nested macro issues
    let multilevel_model = prob!(
        let pop_mean <- sample(addr!("pop_mean"), Normal::new(0.0, 10.0).unwrap());
        let pop_precision <- sample(addr!("pop_precision"), Gamma::new(2.0, 0.5).unwrap());
        let group_mean <- sample(scoped_addr!("group", "mean", "{}", 0), 
                                Normal::new(pop_mean, 1.0).unwrap());
        pure((pop_mean, group_mean))
    );
    println!("✅ Created hierarchical model structure");
    // ANCHOR_END: multilevel_hierarchy
    println!("   - Population -> Groups hierarchy");
    println!("   - Demonstrates scoped addressing");
    println!();
    
    println!("=== All model composition patterns demonstrated! ===");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // ANCHOR: composition_testing
    #[test]
    fn test_model_composition() {
        // Test that models construct without errors
        let simple = prob! {
            let x <- sample(addr!("test_x"), Normal::new(0.0, 1.0).unwrap());
            pure(x)
        };
        
        // Test plate notation
        let plate_model = plate!(i in 0..3 => {
            sample(addr!("plate_test", i), Normal::new(0.0, 1.0).unwrap())
        });
        
        // Test scoped addresses
        let addr1 = scoped_addr!("test", "param");
        let addr2 = scoped_addr!("test", "param", "{}", 42);
        
        // Addresses should be different
        assert_ne!(addr1.0, addr2.0);
        assert!(addr2.0.contains("42"));
        
        // Test hierarchical model construction
        let hierarchical = prob! {
            let global <- sample(addr!("global"), Normal::new(0.0, 1.0).unwrap());
            let locals <- plate!(i in 0..2 => {
                sample(scoped_addr!("local", "param", "{}", i), 
                       Normal::new(global, 0.1).unwrap())
            });
            pure((global, locals))
        };
        
        // All models should construct successfully
        // (Actual execution would require handlers)
    }
    // ANCHOR_END: composition_testing
}
