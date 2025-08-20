# Custom Handlers

Handlers are the heart of Fugue's extensible architecture. They define how models are interpreted and executed. This guide shows you how to create custom handlers for specialized inference algorithms, debugging, and novel probabilistic programming patterns.

## Understanding Handlers

A `Handler` is a trait that defines how to interpret the four fundamental model operations:
- `sample` - Drawing values from distributions
- `observe` - Conditioning on observed data
- `factor` - Adding log-weight factors
- Finishing and returning a trace

```rust
use fugue::*;

pub trait Handler {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64;
    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool;
    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64;
    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize;
    
    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64);
    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool);
    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64);
    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize);
    
    fn on_factor(&mut self, logw: f64);
    fn finish(self) -> Trace where Self: Sized;
}
```

## Built-in Handlers Review

Before creating custom handlers, let's understand the built-in ones:

### PriorHandler - Forward Sampling

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn explore_prior_handler() {
    let model = prob! {
        let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        let y <- sample(addr!("y"), Normal::new(x, 0.5).unwrap());
        observe(addr!("z"), Normal::new(x + y, 0.1).unwrap(), 2.0);
        pure((x, y))
    };
    
    let mut rng = StdRng::seed_from_u64(42);
    let ((x, y), trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    );
    
    println!("🎯 PriorHandler:");
    println!("  Sampled: x={:.3}, y={:.3}", x, y);
    println!("  Log weight: {:.4}", trace.total_log_weight());
    println!("  Purpose: Forward sampling + likelihood computation");
}
```

### ReplayHandler - Deterministic Replay

```rust
fn explore_replay_handler() {
    let model = prob! {
        let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        let y <- sample(addr!("y"), Normal::new(x, 0.5).unwrap());
        pure((x, y))
    };
    
    // Create base trace
    let mut rng = StdRng::seed_from_u64(42);
    let (_, base_trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model.clone(),
    );
    
    // Replay with same choices
    let ((x2, y2), replay_trace) = runtime::handler::run(
        runtime::interpreters::ReplayHandler {
            rng: &mut rng,  // RNG state doesn't matter for replay
            base: base_trace.clone(),
            trace: Trace::default(),
        },
        model,
    );
    
    println!("\n🔄 ReplayHandler:");
    println!("  Original x: {:.3}", base_trace.get_f64(&addr!("x")).unwrap());
    println!("  Replayed x: {:.3}", x2);
    println!("  Same values: {}", base_trace.get_f64(&addr!("x")).unwrap() == x2);
    println!("  Purpose: Deterministic replay for MCMC");
}
```

### ScoreGivenTrace - Exact Scoring

```rust
fn explore_score_handler() {
    let model = prob! {
        let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        observe(addr!("y"), Normal::new(x, 0.1).unwrap(), 2.0);
        pure(x)
    };
    
    // Create a trace with specific values
    let mut fixed_trace = Trace::default();
    fixed_trace.insert_choice(addr!("x"), ChoiceValue::F64(1.5), 0.0);  // Log prob will be recomputed
    
    let (x, scored_trace) = runtime::handler::run(
        runtime::interpreters::ScoreGivenTrace {
            base: fixed_trace,
            trace: Trace::default(),
        },
        model,
    );
    
    println!("\n⚖️ ScoreGivenTrace:");
    println!("  Fixed x: {:.3}", x);
    println!("  Computed log weight: {:.4}", scored_trace.total_log_weight());
    println!("  Purpose: Exact probability computation");
}
```

## Creating Custom Handlers

### Example 1: Logging Handler

Let's create a handler that logs all operations for debugging:

```rust
use fugue::*;
use std::collections::BTreeMap;

pub struct LoggingHandler<R: rand::Rng> {
    rng: R,
    trace: Trace,
    log_level: LogLevel,
}

#[derive(Debug, Clone, Copy)]
pub enum LogLevel {
    Silent,
    Operations,
    Verbose,
}

impl<R: rand::Rng> LoggingHandler<R> {
    pub fn new(rng: R, log_level: LogLevel) -> Self {
        Self {
            rng,
            trace: Trace::default(),
            log_level,
        }
    }
    
    fn log(&self, level: LogLevel, message: &str) {
        if matches!((self.log_level, level), 
                   (LogLevel::Verbose, _) | 
                   (LogLevel::Operations, LogLevel::Operations)) {
            println!("  🔍 {}", message);
        }
    }
}

impl<R: rand::Rng> Handler for LoggingHandler<R> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_pdf(value);
        
        self.log(LogLevel::Operations, &format!("SAMPLE {} ~ {:?} = {:.4}", addr, dist, value));
        self.log(LogLevel::Verbose, &format!("  log_prob = {:.4}", log_prob));
        
        self.trace.insert_choice(addr.clone(), ChoiceValue::F64(value), log_prob);
        self.trace.log_prior += log_prob;
        
        value
    }
    
    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_pmf(value);
        
        self.log(LogLevel::Operations, &format!("SAMPLE {} ~ {:?} = {}", addr, dist, value));
        self.log(LogLevel::Verbose, &format!("  log_prob = {:.4}", log_prob));
        
        self.trace.insert_choice(addr.clone(), ChoiceValue::Bool(value), log_prob);
        self.trace.log_prior += log_prob;
        
        value
    }
    
    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_pmf(value);
        
        self.log(LogLevel::Operations, &format!("SAMPLE {} ~ {:?} = {}", addr, dist, value));
        self.log(LogLevel::Verbose, &format!("  log_prob = {:.4}", log_prob));
        
        self.trace.insert_choice(addr.clone(), ChoiceValue::U64(value), log_prob);
        self.trace.log_prior += log_prob;
        
        value
    }
    
    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_pmf(value);
        
        self.log(LogLevel::Operations, &format!("SAMPLE {} ~ {:?} = {}", addr, dist, value));
        self.log(LogLevel::Verbose, &format!("  log_prob = {:.4}", log_prob));
        
        self.trace.insert_choice(addr.clone(), ChoiceValue::Usize(value), log_prob);
        self.trace.log_prior += log_prob;
        
        value
    }
    
    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        let log_prob = dist.log_pdf(value);
        
        self.log(LogLevel::Operations, &format!("OBSERVE {} = {:.4} ~ {:?}", addr, value, dist));
        self.log(LogLevel::Verbose, &format!("  log_likelihood = {:.4}", log_prob));
        
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        let log_prob = dist.log_pmf(value);
        
        self.log(LogLevel::Operations, &format!("OBSERVE {} = {} ~ {:?}", addr, value, dist));
        self.log(LogLevel::Verbose, &format!("  log_likelihood = {:.4}", log_prob));
        
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        let log_prob = dist.log_pmf(value);
        
        self.log(LogLevel::Operations, &format!("OBSERVE {} = {} ~ {:?}", addr, value, dist));
        self.log(LogLevel::Verbose, &format!("  log_likelihood = {:.4}", log_prob));
        
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize) {
        let log_prob = dist.log_pmf(value);
        
        self.log(LogLevel::Operations, &format!("OBSERVE {} = {} ~ {:?}", addr, value, dist));
        self.log(LogLevel::Verbose, &format!("  log_likelihood = {:.4}", log_prob));
        
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_factor(&mut self, logw: f64) {
        self.log(LogLevel::Operations, &format!("FACTOR {:.4}", logw));
        self.trace.log_factors += logw;
    }
    
    fn finish(self) -> Trace {
        self.log(LogLevel::Operations, &format!("FINISH: total_log_weight = {:.4}", self.trace.total_log_weight()));
        self.trace
    }
}

// Usage example
fn test_logging_handler() {
    let model = prob! {
        let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        let coin <- sample(addr!("coin"), Bernoulli::new(0.7).unwrap());
        observe(addr!("y"), Normal::new(x, 0.1).unwrap(), 1.5);
        factor(if coin { 0.0 } else { -1.0 });
        pure((x, coin))
    };
    
    println!("🔍 Logging Handler Demo:");
    
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    let rng = StdRng::seed_from_u64(42);
    
    let ((x, coin), trace) = runtime::handler::run(
        LoggingHandler::new(rng, LogLevel::Operations),
        model,
    );
    
    println!("🎯 Final result: x={:.3}, coin={}, log_weight={:.4}", 
             x, coin, trace.total_log_weight());
}
```

### Example 2: Constraint Handler

A handler that automatically rejects samples outside constraints:

```rust
use std::collections::HashMap;

pub struct ConstraintHandler<R: rand::Rng> {
    rng: R,
    trace: Trace,
    constraints: HashMap<Address, Box<dyn Fn(f64) -> bool>>,
    max_rejections: usize,
    rejections: usize,
}

impl<R: rand::Rng> ConstraintHandler<R> {
    pub fn new(rng: R, max_rejections: usize) -> Self {
        Self {
            rng,
            trace: Trace::default(),
            constraints: HashMap::new(),
            max_rejections,
            rejections: 0,
        }
    }
    
    pub fn add_constraint<F>(mut self, addr: Address, constraint: F) -> Self 
    where
        F: Fn(f64) -> bool + 'static
    {
        self.constraints.insert(addr, Box::new(constraint));
        self
    }
    
    fn check_constraints(&mut self, addr: &Address, value: f64) -> bool {
        if let Some(constraint) = self.constraints.get(addr) {
            let satisfied = constraint(value);
            if !satisfied {
                self.rejections += 1;
                if self.rejections > self.max_rejections {
                    panic!("Too many constraint violations at {}: {}", addr, value);
                }
            }
            satisfied
        } else {
            true
        }
    }
}

impl<R: rand::Rng> Handler for ConstraintHandler<R> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        loop {
            let value = dist.sample(&mut self.rng);
            
            if self.check_constraints(addr, value) {
                let log_prob = dist.log_pdf(value);
                self.trace.insert_choice(addr.clone(), ChoiceValue::F64(value), log_prob);
                self.trace.log_prior += log_prob;
                return value;
            }
            // If constraint violated, try again
        }
    }
    
    // For simplicity, other sample methods don't check constraints
    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_pmf(value);
        self.trace.insert_choice(addr.clone(), ChoiceValue::Bool(value), log_prob);
        self.trace.log_prior += log_prob;
        value
    }
    
    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_pmf(value);
        self.trace.insert_choice(addr.clone(), ChoiceValue::U64(value), log_prob);
        self.trace.log_prior += log_prob;
        value
    }
    
    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_pmf(value);
        self.trace.insert_choice(addr.clone(), ChoiceValue::Usize(value), log_prob);
        self.trace.log_prior += log_prob;
        value
    }
    
    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        let log_prob = dist.log_pdf(value);
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        let log_prob = dist.log_pmf(value);
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        let log_prob = dist.log_pmf(value);
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize) {
        let log_prob = dist.log_pmf(value);
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }
    
    fn finish(self) -> Trace {
        println!("🚫 Constraint handler: {} rejections", self.rejections);
        self.trace
    }
}

// Usage example
fn test_constraint_handler() {
    let model = prob! {
        let x <- sample(addr!("x"), Normal::new(0.0, 2.0).unwrap());
        let y <- sample(addr!("y"), Normal::new(x, 1.0).unwrap());
        observe(addr!("z"), Normal::new(x + y, 0.1).unwrap(), 3.0);
        pure((x, y))
    };
    
    println!("\n🚫 Constraint Handler Demo:");
    
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    let rng = StdRng::seed_from_u64(42);
    
    // Only allow positive values for x
    let handler = ConstraintHandler::new(rng, 1000)
        .add_constraint(addr!("x"), |x| x > 0.0)
        .add_constraint(addr!("y"), |y| y.abs() < 5.0);
    
    let ((x, y), trace) = runtime::handler::run(handler, model);
    
    println!("🎯 Constrained result: x={:.3} (>0), y={:.3} (|y|<5)", x, y);
    println!("   Log weight: {:.4}", trace.total_log_weight());
}
```

### Example 3: Statistics Collection Handler

A handler that collects statistics during execution:

```rust
use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct ExecutionStats {
    pub sample_counts: HashMap<String, usize>,
    pub observe_counts: HashMap<String, usize>,
    pub factor_calls: usize,
    pub total_log_weight: f64,
    pub execution_path: Vec<String>,
}

pub struct StatsCollectorHandler<R: rand::Rng> {
    rng: R,
    trace: Trace,
    stats: ExecutionStats,
}

impl<R: rand::Rng> StatsCollectorHandler<R> {
    pub fn new(rng: R) -> Self {
        Self {
            rng,
            trace: Trace::default(),
            stats: ExecutionStats::default(),
        }
    }
    
    pub fn get_stats(&self) -> &ExecutionStats {
        &self.stats
    }
    
    fn record_sample(&mut self, addr: &Address, dist_name: &str) {
        let key = format!("{}:{}", addr, dist_name);
        *self.stats.sample_counts.entry(key.clone()).or_insert(0) += 1;
        self.stats.execution_path.push(format!("SAMPLE {}", key));
    }
    
    fn record_observe(&mut self, addr: &Address, dist_name: &str) {
        let key = format!("{}:{}", addr, dist_name);
        *self.stats.observe_counts.entry(key.clone()).or_insert(0) += 1;
        self.stats.execution_path.push(format!("OBSERVE {}", key));
    }
}

impl<R: rand::Rng> Handler for StatsCollectorHandler<R> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        self.record_sample(addr, "f64");
        
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_pdf(value);
        
        self.trace.insert_choice(addr.clone(), ChoiceValue::F64(value), log_prob);
        self.trace.log_prior += log_prob;
        
        value
    }
    
    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        self.record_sample(addr, "bool");
        
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_pmf(value);
        
        self.trace.insert_choice(addr.clone(), ChoiceValue::Bool(value), log_prob);
        self.trace.log_prior += log_prob;
        
        value
    }
    
    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        self.record_sample(addr, "u64");
        
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_pmf(value);
        
        self.trace.insert_choice(addr.clone(), ChoiceValue::U64(value), log_prob);
        self.trace.log_prior += log_prob;
        
        value
    }
    
    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        self.record_sample(addr, "usize");
        
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_pmf(value);
        
        self.trace.insert_choice(addr.clone(), ChoiceValue::Usize(value), log_prob);
        self.trace.log_prior += log_prob;
        
        value
    }
    
    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.record_observe(addr, "f64");
        
        let log_prob = dist.log_pdf(value);
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        self.record_observe(addr, "bool");
        
        let log_prob = dist.log_pmf(value);
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        self.record_observe(addr, "u64");
        
        let log_prob = dist.log_pmf(value);
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize) {
        self.record_observe(addr, "usize");
        
        let log_prob = dist.log_pmf(value);
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_factor(&mut self, logw: f64) {
        self.stats.factor_calls += 1;
        self.stats.execution_path.push(format!("FACTOR {:.4}", logw));
        self.trace.log_factors += logw;
    }
    
    fn finish(mut self) -> Trace {
        self.stats.total_log_weight = self.trace.total_log_weight();
        
        println!("📊 Execution Statistics:");
        println!("  Sample operations: {:?}", self.stats.sample_counts);
        println!("  Observe operations: {:?}", self.stats.observe_counts);
        println!("  Factor calls: {}", self.stats.factor_calls);
        println!("  Total log weight: {:.4}", self.stats.total_log_weight);
        println!("  Execution path length: {}", self.stats.execution_path.len());
        
        self.trace
    }
}

// Usage example
fn test_stats_handler() {
    let model = prob! {
        let components <- plate!(i in 0..3 => {
            sample(addr!("component", i), Normal::new(i as f64, 1.0).unwrap())
        });
        
        let choice <- sample(addr!("choice"), Categorical::new(vec![0.3, 0.5, 0.2]).unwrap());
        let selected = components[choice];
        
        observe(addr!("obs"), Normal::new(selected, 0.1).unwrap(), 1.8);
        factor(-0.5);
        
        pure((choice, selected))
    };
    
    println!("\n📊 Statistics Collection Demo:");
    
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    let rng = StdRng::seed_from_u64(42);
    
    let ((choice, selected), trace) = runtime::handler::run(
        StatsCollectorHandler::new(rng),
        model,
    );
    
    println!("🎯 Result: choice={}, selected={:.3}", choice, selected);
}
```

## Advanced Handler Patterns

### Composition: Chaining Handlers

You can create meta-handlers that compose multiple handlers:

```rust
pub struct CompositeHandler<H1, H2> {
    primary: H1,
    secondary: H2,
}

impl<H1: Handler, H2: Handler> CompositeHandler<H1, H2> {
    pub fn new(primary: H1, secondary: H2) -> Self {
        Self { primary, secondary }
    }
}

// This would require careful coordination between handlers
// Implementation details depend on specific requirements
```

### Caching Handler

For expensive model evaluations:

```rust
use std::collections::HashMap;

pub struct CachingHandler<R: rand::Rng> {
    rng: R,
    trace: Trace,
    cache: HashMap<(Address, String), (f64, f64)>,  // (addr, dist_params) -> (value, log_prob)
}

impl<R: rand::Rng> CachingHandler<R> {
    pub fn new(rng: R) -> Self {
        Self {
            rng,
            trace: Trace::default(),
            cache: HashMap::new(),
        }
    }
    
    fn cache_key(&self, addr: &Address, dist: &dyn Distribution<f64>) -> String {
        // This is a simplified cache key - in practice you'd need
        // proper serialization of distribution parameters
        format!("{:?}", dist)
    }
}

// Implementation would cache results for identical (address, distribution) pairs
```

### Sensitivity Analysis Handler

For automatic differentiation and sensitivity analysis:

```rust
pub struct SensitivityHandler<R: rand::Rng> {
    rng: R,
    trace: Trace,
    parameter_gradients: HashMap<Address, f64>,
    perturbation_scale: f64,
}

impl<R: rand::Rng> SensitivityHandler<R> {
    pub fn new(rng: R, perturbation_scale: f64) -> Self {
        Self {
            rng,
            trace: Trace::default(),
            parameter_gradients: HashMap::new(),
            perturbation_scale,
        }
    }
    
    // Would implement finite difference gradients
    fn compute_gradient(&mut self, addr: &Address, base_value: f64, log_prob_fn: impl Fn(f64) -> f64) {
        let epsilon = self.perturbation_scale;
        let log_prob_plus = log_prob_fn(base_value + epsilon);
        let log_prob_minus = log_prob_fn(base_value - epsilon);
        let gradient = (log_prob_plus - log_prob_minus) / (2.0 * epsilon);
        
        self.parameter_gradients.insert(addr.clone(), gradient);
    }
}
```

## Testing Custom Handlers

Always test your custom handlers thoroughly:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    
    #[test]
    fn test_logging_handler_consistency() {
        let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        
        // Test with logging handler
        let rng1 = StdRng::seed_from_u64(42);
        let (value1, trace1) = runtime::handler::run(
            LoggingHandler::new(rng1, LogLevel::Silent),
            model.clone(),
        );
        
        // Test with standard handler
        let rng2 = StdRng::seed_from_u64(42);
        let (value2, trace2) = runtime::handler::run(
            runtime::interpreters::PriorHandler {
                rng: &mut rng2,
                trace: Trace::default(),
            },
            model,
        );
        
        // Should produce same results with same seed
        assert_eq!(value1, value2);
        assert_eq!(trace1.total_log_weight(), trace2.total_log_weight());
    }
    
    #[test]
    fn test_constraint_handler_violations() {
        let model = sample(addr!("x"), Normal::new(-5.0, 0.1).unwrap());  // Likely negative
        
        let rng = StdRng::seed_from_u64(42);
        let handler = ConstraintHandler::new(rng, 10)
            .add_constraint(addr!("x"), |x| x > 0.0);  // Require positive
        
        let (value, _) = runtime::handler::run(handler, model);
        
        assert!(value > 0.0, "Constraint should ensure positive value");
    }
}
```

## Performance Considerations

### Efficient Handler Implementation

```rust
// ❌ Inefficient: Repeated allocations
fn inefficient_logging(addr: &Address, value: f64) {
    println!("Sample at {}: {}", addr.to_string(), value);  // String allocation
}

// ✅ Efficient: Minimize allocations
fn efficient_logging(addr: &Address, value: f64) {
    println!("Sample at {}: {}", addr, value);  // Uses Display directly
}

// ✅ Even better: Conditional logging
fn conditional_logging(enabled: bool, addr: &Address, value: f64) {
    if enabled {
        println!("Sample at {}: {}", addr, value);
    }
}
```

### Memory Management

```rust
pub struct MemoryEfficientHandler<R: rand::Rng> {
    rng: R,
    trace: Trace,
    // Use Vec instead of HashMap for better cache locality
    cached_addresses: Vec<Address>,
    // Pre-allocate commonly used structures
    temp_buffer: Vec<f64>,
}

impl<R: rand::Rng> MemoryEfficientHandler<R> {
    pub fn new(rng: R, expected_addresses: usize) -> Self {
        Self {
            rng,
            trace: Trace::default(),
            cached_addresses: Vec::with_capacity(expected_addresses),
            temp_buffer: Vec::with_capacity(1000),
        }
    }
}
```

## Complete Example: Custom MCMC Handler

Here's a complete example of a custom handler for specialized MCMC:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;

pub struct AdaptiveMCMCHandler<R: rand::Rng> {
    rng: R,
    trace: Trace,
    proposal_scales: HashMap<Address, f64>,
    acceptance_counts: HashMap<Address, usize>,
    proposal_counts: HashMap<Address, usize>,
    adaptation_rate: f64,
    target_acceptance: f64,
}

impl<R: rand::Rng> AdaptiveMCMCHandler<R> {
    pub fn new(rng: R, target_acceptance: f64) -> Self {
        Self {
            rng,
            trace: Trace::default(),
            proposal_scales: HashMap::new(),
            acceptance_counts: HashMap::new(),
            proposal_counts: HashMap::new(),
            adaptation_rate: 0.01,
            target_acceptance,
        }
    }
    
    fn get_proposal_scale(&mut self, addr: &Address) -> f64 {
        *self.proposal_scales.entry(addr.clone()).or_insert(1.0)
    }
    
    fn update_proposal_scale(&mut self, addr: &Address, accepted: bool) {
        *self.proposal_counts.entry(addr.clone()).or_insert(0) += 1;
        if accepted {
            *self.acceptance_counts.entry(addr.clone()).or_insert(0) += 1;
        }
        
        let proposals = self.proposal_counts[addr] as f64;
        let acceptances = self.acceptance_counts.get(addr).unwrap_or(&0) as f64;
        let acceptance_rate = acceptances / proposals;
        
        let current_scale = self.get_proposal_scale(addr);
        let adaptation = if acceptance_rate > self.target_acceptance {
            1.0 + self.adaptation_rate
        } else {
            1.0 - self.adaptation_rate
        };
        
        let new_scale = current_scale * adaptation;
        self.proposal_scales.insert(addr.clone(), new_scale.max(0.001).min(10.0));
    }
}

impl<R: rand::Rng> Handler for AdaptiveMCMCHandler<R> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        // For MCMC, you'd typically propose from current state + noise
        // This is simplified - real implementation would use current trace
        let proposal_scale = self.get_proposal_scale(addr);
        
        // Simplified proposal: just sample from distribution with adapted scale
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_pdf(value);
        
        // In real MCMC, you'd decide accept/reject here
        let accepted = true;  // Simplified
        self.update_proposal_scale(addr, accepted);
        
        self.trace.insert_choice(addr.clone(), ChoiceValue::F64(value), log_prob);
        self.trace.log_prior += log_prob;
        
        value
    }
    
    // Other methods similar to PriorHandler...
    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_pmf(value);
        self.trace.insert_choice(addr.clone(), ChoiceValue::Bool(value), log_prob);
        self.trace.log_prior += log_prob;
        value
    }
    
    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_pmf(value);
        self.trace.insert_choice(addr.clone(), ChoiceValue::U64(value), log_prob);
        self.trace.log_prior += log_prob;
        value
    }
    
    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_pmf(value);
        self.trace.insert_choice(addr.clone(), ChoiceValue::Usize(value), log_prob);
        self.trace.log_prior += log_prob;
        value
    }
    
    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        let log_prob = dist.log_pdf(value);
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        let log_prob = dist.log_pmf(value);
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        let log_prob = dist.log_pmf(value);
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize) {
        let log_prob = dist.log_pmf(value);
        self.trace.log_likelihood += log_prob;
    }
    
    fn on_factor(&mut self, logw: f64) {
        self.trace.log_factors += logw;
    }
    
    fn finish(self) -> Trace {
        println!("🎯 Adaptive MCMC: {} addresses adapted", self.proposal_scales.len());
        for (addr, scale) in &self.proposal_scales {
            let proposals = self.proposal_counts.get(addr).unwrap_or(&0);
            let acceptances = self.acceptance_counts.get(addr).unwrap_or(&0);
            let rate = if *proposals > 0 { *acceptances as f64 / *proposals as f64 } else { 0.0 };
            println!("  {}: scale={:.3}, acceptance={:.1}%", addr, scale, rate * 100.0);
        }
        self.trace
    }
}

fn main() {
    // Test all custom handlers
    test_logging_handler();
    test_constraint_handler();
    test_stats_handler();
    
    println!("\n🎯 All custom handler examples completed!");
}
```

## Key Takeaways

1. **Handlers define interpretation** - They control how models execute
2. **Type safety is preserved** - Handlers maintain Fugue's type-safe distribution system
3. **Composability is powerful** - Handlers can be combined and chained
4. **Testing is crucial** - Always validate custom handlers against known results
5. **Performance matters** - Efficient handlers enable scalable inference
6. **Extensibility is unlimited** - Custom handlers enable novel algorithms and analyses

## Next Steps

- **[Debugging Models](debugging-models.md)** - Advanced debugging techniques using custom handlers
- **[Mixture Models Tutorial](../tutorials/mixture-models.md)** - See custom handlers in complex models
- **[Understanding Models](../getting-started/understanding-models.md)** - Deeper model composition techniques

---

**Ready to debug like a pro?** → **[Debugging Models](debugging-models.md)**