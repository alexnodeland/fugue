use fugue::*;
use fugue::runtime::interpreters::PriorHandler;
use fugue::runtime::trace::{Choice, ChoiceValue, Trace};
use fugue::runtime::handler::Handler;
use rand::{thread_rng, Rng};
use std::collections::HashMap;

// ANCHOR: basic_custom_handler
/// Simple handler that just samples from priors (similar to PriorHandler)
struct BasicHandler<R: Rng> {
    rng: R,
    trace: Trace,
}

impl<R: Rng> Handler for BasicHandler<R> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_prob(&value);
        
        // Store in trace
        self.trace.log_prior += log_prob;
        self.trace.choices.insert(addr.clone(), Choice {
            addr: addr.clone(),
            value: ChoiceValue::F64(value),
            logp: log_prob,
        });
        
        value
    }
    
    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_prob(&value);
        
        self.trace.log_prior += log_prob;
        self.trace.choices.insert(addr.clone(), Choice {
            addr: addr.clone(),
            value: ChoiceValue::Bool(value),
            logp: log_prob,
        });
        
        value
    }
    
    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_prob(&value);
        
        self.trace.log_prior += log_prob;
        self.trace.choices.insert(addr.clone(), Choice {
            addr: addr.clone(),
            value: ChoiceValue::U64(value),
            logp: log_prob,
        });
        
        value
    }
    
    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        let value = dist.sample(&mut self.rng);
        let log_prob = dist.log_prob(&value);
        
        self.trace.log_prior += log_prob;
        self.trace.choices.insert(addr.clone(), Choice {
            addr: addr.clone(),
            value: ChoiceValue::Usize(value),
            logp: log_prob,
        });
        
        value
    }
    
    fn on_observe_f64(&mut self, _addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }
    
    fn on_observe_bool(&mut self, _addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }
    
    fn on_observe_u64(&mut self, _addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }
    
    fn on_observe_usize(&mut self, _addr: &Address, dist: &dyn Distribution<usize>, value: usize) {
        self.trace.log_likelihood += dist.log_prob(&value);
    }
    
    fn on_factor(&mut self, log_weight: f64) {
        self.trace.log_factors += log_weight;
    }
    
    fn finish(self) -> Trace {
        self.trace
    }
}
// ANCHOR_END: basic_custom_handler

// ANCHOR: logging_handler
/// Handler decorator that logs all operations
struct LoggingHandler<H: Handler> {
    inner: H,
    log: Vec<String>,
    verbose: bool,
}

impl<H: Handler> LoggingHandler<H> {
    fn new(inner: H, verbose: bool) -> Self {
        Self {
            inner,
            log: Vec::new(),
            verbose,
        }
    }
    
    fn log_operation(&mut self, operation: String) {
        if self.verbose {
            println!("LOG: {}", operation);
        }
        self.log.push(operation);
    }
}

impl<H: Handler> Handler for LoggingHandler<H> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        let value = self.inner.on_sample_f64(addr, dist);
        self.log_operation(format!("Sample f64 at {}: {:.3}", addr, value));
        value
    }
    
    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let value = self.inner.on_sample_bool(addr, dist);
        self.log_operation(format!("Sample bool at {}: {}", addr, value));
        value
    }
    
    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        let value = self.inner.on_sample_u64(addr, dist);
        self.log_operation(format!("Sample u64 at {}: {}", addr, value));
        value
    }
    
    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        let value = self.inner.on_sample_usize(addr, dist);
        self.log_operation(format!("Sample usize at {}: {}", addr, value));
        value
    }
    
    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.log_operation(format!("Observe f64 at {}: {:.3}", addr, value));
        self.inner.on_observe_f64(addr, dist, value);
    }
    
    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        self.log_operation(format!("Observe bool at {}: {}", addr, value));
        self.inner.on_observe_bool(addr, dist, value);
    }
    
    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        self.log_operation(format!("Observe u64 at {}: {}", addr, value));
        self.inner.on_observe_u64(addr, dist, value);
    }
    
    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize) {
        self.log_operation(format!("Observe usize at {}: {}", addr, value));
        self.inner.on_observe_usize(addr, dist, value);
    }
    
    fn on_factor(&mut self, log_weight: f64) {
        self.log_operation(format!("Factor: {:.3}", log_weight));
        self.inner.on_factor(log_weight);
    }
    
    fn finish(self) -> Trace {
        let trace = self.inner.finish();
        println!("✅ Logged {} operations total", self.log.len());
        trace
    }
}
// ANCHOR_END: logging_handler

// ANCHOR: statistics_handler
/// Handler that accumulates statistics about model execution
#[derive(Debug)]
struct ExecutionStats {
    sample_counts: HashMap<String, u32>, // Type -> count
    observe_counts: HashMap<String, u32>,
    factor_count: u32,
    total_log_weight: f64,
    parameter_ranges: HashMap<String, (f64, f64)>, // Address -> (min, max) for f64 params
}

impl Default for ExecutionStats {
    fn default() -> Self {
        Self {
            sample_counts: HashMap::new(),
            observe_counts: HashMap::new(),
            factor_count: 0,
            total_log_weight: 0.0,
            parameter_ranges: HashMap::new(),
        }
    }
}

struct StatisticsHandler<H: Handler> {
    inner: H,
    stats: ExecutionStats,
}

impl<H: Handler> StatisticsHandler<H> {
    fn new(inner: H) -> Self {
        Self {
            inner,
            stats: ExecutionStats::default(),
        }
    }
    
    fn update_f64_range(&mut self, addr: &Address, value: f64) {
        let key = addr.0.clone();
        self.stats.parameter_ranges
            .entry(key)
            .and_modify(|(min, max)| {
                *min = min.min(value);
                *max = max.max(value);
            })
            .or_insert((value, value));
    }
}

impl<H: Handler> Handler for StatisticsHandler<H> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        let value = self.inner.on_sample_f64(addr, dist);
        *self.stats.sample_counts.entry("f64".to_string()).or_insert(0) += 1;
        self.update_f64_range(addr, value);
        value
    }
    
    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let value = self.inner.on_sample_bool(addr, dist);
        *self.stats.sample_counts.entry("bool".to_string()).or_insert(0) += 1;
        value
    }
    
    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        let value = self.inner.on_sample_u64(addr, dist);
        *self.stats.sample_counts.entry("u64".to_string()).or_insert(0) += 1;
        value
    }
    
    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        let value = self.inner.on_sample_usize(addr, dist);
        *self.stats.sample_counts.entry("usize".to_string()).or_insert(0) += 1;
        value
    }
    
    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        *self.stats.observe_counts.entry("f64".to_string()).or_insert(0) += 1;
        self.inner.on_observe_f64(addr, dist, value);
    }
    
    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        *self.stats.observe_counts.entry("bool".to_string()).or_insert(0) += 1;
        self.inner.on_observe_bool(addr, dist, value);
    }
    
    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        *self.stats.observe_counts.entry("u64".to_string()).or_insert(0) += 1;
        self.inner.on_observe_u64(addr, dist, value);
    }
    
    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize) {
        *self.stats.observe_counts.entry("usize".to_string()).or_insert(0) += 1;
        self.inner.on_observe_usize(addr, dist, value);
    }
    
    fn on_factor(&mut self, log_weight: f64) {
        self.stats.factor_count += 1;
        self.stats.total_log_weight += log_weight;
        self.inner.on_factor(log_weight);
    }
    
    fn finish(self) -> Trace {
        println!("✅ Execution Statistics:");
        println!("   - Samples by type: {:?}", self.stats.sample_counts);
        println!("   - Observations by type: {:?}", self.stats.observe_counts);
        println!("   - Factor operations: {}", self.stats.factor_count);
        println!("   - Parameter ranges:");
        for (addr, (min, max)) in &self.stats.parameter_ranges {
            println!("     {}: [{:.3}, {:.3}]", addr, min, max);
        }
        self.inner.finish()
    }
}
// ANCHOR_END: statistics_handler

// ANCHOR: filtering_handler
/// Handler that filters/modifies values based on conditions
struct FilteringHandler<H: Handler> {
    inner: H,
    f64_clamp_range: Option<(f64, f64)>,
    bool_flip_probability: f64,
    rng: rand::rngs::ThreadRng,
}

impl<H: Handler> FilteringHandler<H> {
    fn new(inner: H, f64_clamp_range: Option<(f64, f64)>, bool_flip_probability: f64) -> Self {
        Self {
            inner,
            f64_clamp_range,
            bool_flip_probability,
            rng: thread_rng(),
        }
    }
}

impl<H: Handler> Handler for FilteringHandler<H> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        let mut value = self.inner.on_sample_f64(addr, dist);
        
        // Apply clamping if specified
        if let Some((min, max)) = self.f64_clamp_range {
            value = value.clamp(min, max);
        }
        
        value
    }
    
    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let mut value = self.inner.on_sample_bool(addr, dist);
        
        // Flip boolean with specified probability
        if self.rng.gen::<f64>() < self.bool_flip_probability {
            value = !value;
        }
        
        value
    }
    
    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        self.inner.on_sample_u64(addr, dist)
    }
    
    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        self.inner.on_sample_usize(addr, dist)
    }
    
    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.inner.on_observe_f64(addr, dist, value);
    }
    
    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        self.inner.on_observe_bool(addr, dist, value);
    }
    
    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        self.inner.on_observe_u64(addr, dist, value);
    }
    
    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize) {
        self.inner.on_observe_usize(addr, dist, value);
    }
    
    fn on_factor(&mut self, log_weight: f64) {
        self.inner.on_factor(log_weight);
    }
    
    fn finish(self) -> Trace {
        self.inner.finish()
    }
}
// ANCHOR_END: filtering_handler

// ANCHOR: performance_handler
use std::time::{Duration, Instant};

/// Handler that monitors performance characteristics
struct PerformanceHandler<H: Handler> {
    inner: H,
    start_time: Instant,
    operation_times: Vec<Duration>,
    sample_count: u32,
    observe_count: u32,
}

impl<H: Handler> PerformanceHandler<H> {
    fn new(inner: H) -> Self {
        Self {
            inner,
            start_time: Instant::now(),
            operation_times: Vec::new(),
            sample_count: 0,
            observe_count: 0,
        }
    }
    
    fn time_operation<F, R>(&mut self, operation: F) -> R
    where F: FnOnce(&mut H) -> R {
        let start = Instant::now();
        let result = operation(&mut self.inner);
        let duration = start.elapsed();
        self.operation_times.push(duration);
        result
    }
}

impl<H: Handler> Handler for PerformanceHandler<H> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        self.sample_count += 1;
        self.time_operation(|inner| inner.on_sample_f64(addr, dist))
    }
    
    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        self.sample_count += 1;
        self.time_operation(|inner| inner.on_sample_bool(addr, dist))
    }
    
    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        self.sample_count += 1;
        self.time_operation(|inner| inner.on_sample_u64(addr, dist))
    }
    
    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        self.sample_count += 1;
        self.time_operation(|inner| inner.on_sample_usize(addr, dist))
    }
    
    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.observe_count += 1;
        self.time_operation(|inner| inner.on_observe_f64(addr, dist, value))
    }
    
    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        self.observe_count += 1;
        self.time_operation(|inner| inner.on_observe_bool(addr, dist, value))
    }
    
    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        self.observe_count += 1;
        self.time_operation(|inner| inner.on_observe_u64(addr, dist, value))
    }
    
    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize) {
        self.observe_count += 1;
        self.time_operation(|inner| inner.on_observe_usize(addr, dist, value))
    }
    
    fn on_factor(&mut self, log_weight: f64) {
        self.time_operation(|inner| inner.on_factor(log_weight))
    }
    
    fn finish(self) -> Trace {
        let total_time = self.start_time.elapsed();
        let avg_op_time = if !self.operation_times.is_empty() {
            self.operation_times.iter().sum::<Duration>() / self.operation_times.len() as u32
        } else {
            Duration::ZERO
        };
        
        println!("✅ Performance Monitoring Results:");
        println!("   - Total execution time: {:?}", total_time);
        println!("   - Operations performed: {}", self.operation_times.len());
        println!("   - Sample operations: {}", self.sample_count);
        println!("   - Observe operations: {}", self.observe_count);
        println!("   - Average operation time: {:?}", avg_op_time);
        
        self.inner.finish()
    }
}
// ANCHOR_END: performance_handler

// ANCHOR: custom_inference_handler
/// Simple custom MCMC-like handler that perturbs existing values
struct SimpleMCMCHandler<R: Rng> {
    rng: R,
    base_trace: Trace,
    current_trace: Trace,
    perturbation_scale: f64,
}

impl<R: Rng> SimpleMCMCHandler<R> {
    fn new(rng: R, base_trace: Trace, perturbation_scale: f64) -> Self {
        Self {
            rng,
            base_trace,
            current_trace: Trace::default(),
            perturbation_scale,
        }
    }
}

impl<R: Rng> Handler for SimpleMCMCHandler<R> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        let value = if let Some(base_value) = self.base_trace.get_f64(addr) {
            // Perturb existing value
            let perturbation = Normal::new(0.0, self.perturbation_scale).unwrap();
            base_value + perturbation.sample(&mut self.rng)
        } else {
            // Sample fresh if not in base trace
            dist.sample(&mut self.rng)
        };
        
        let log_prob = dist.log_prob(&value);
        self.current_trace.log_prior += log_prob;
        self.current_trace.choices.insert(addr.clone(), Choice {
            addr: addr.clone(),
            value: ChoiceValue::F64(value),
            logp: log_prob,
        });
        
        value
    }
    
    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        let value = if let Some(base_value) = self.base_trace.get_bool(addr) {
            // Maybe flip the boolean with small probability
            if self.rng.gen::<f64>() < 0.1 {
                !base_value
            } else {
                base_value
            }
        } else {
            dist.sample(&mut self.rng)
        };
        
        let log_prob = dist.log_prob(&value);
        self.current_trace.log_prior += log_prob;
        self.current_trace.choices.insert(addr.clone(), Choice {
            addr: addr.clone(),
            value: ChoiceValue::Bool(value),
            logp: log_prob,
        });
        
        value
    }
    
    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        // For simplicity, just use base value or sample fresh
        let value = self.base_trace.get_u64(addr).unwrap_or_else(|| dist.sample(&mut self.rng));
        
        let log_prob = dist.log_prob(&value);
        self.current_trace.log_prior += log_prob;
        self.current_trace.choices.insert(addr.clone(), Choice {
            addr: addr.clone(),
            value: ChoiceValue::U64(value),
            logp: log_prob,
        });
        
        value
    }
    
    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        let value = self.base_trace.get_usize(addr).unwrap_or_else(|| dist.sample(&mut self.rng));
        
        let log_prob = dist.log_prob(&value);
        self.current_trace.log_prior += log_prob;
        self.current_trace.choices.insert(addr.clone(), Choice {
            addr: addr.clone(),
            value: ChoiceValue::Usize(value),
            logp: log_prob,
        });
        
        value
    }
    
    fn on_observe_f64(&mut self, _addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        self.current_trace.log_likelihood += dist.log_prob(&value);
    }
    
    fn on_observe_bool(&mut self, _addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        self.current_trace.log_likelihood += dist.log_prob(&value);
    }
    
    fn on_observe_u64(&mut self, _addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        self.current_trace.log_likelihood += dist.log_prob(&value);
    }
    
    fn on_observe_usize(&mut self, _addr: &Address, dist: &dyn Distribution<usize>, value: usize) {
        self.current_trace.log_likelihood += dist.log_prob(&value);
    }
    
    fn on_factor(&mut self, log_weight: f64) {
        self.current_trace.log_factors += log_weight;
    }
    
    fn finish(self) -> Trace {
        self.current_trace
    }
}
// ANCHOR_END: custom_inference_handler

// ANCHOR: handler_composition
fn main() {
    println!("=== Custom Handlers in Fugue ===\n");
    
    println!("1. Basic Custom Handler Implementation");
    println!("------------------------------------");
    
    // Test the basic handler
    let mut rng = thread_rng();
    let handler = BasicHandler {
        rng: &mut rng,
        trace: Trace::default(),
    };
    
    let test_model = || sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
    let (result, trace) = runtime::handler::run(handler, test_model());
    
    println!("✅ Basic handler executed");
    println!("   - Result: {:.3}", result);
    println!("   - Trace choices: {}", trace.choices.len());
    println!("   - Total log-weight: {:.3}", trace.total_log_weight());
    println!();
    
    println!("2. Logging Handler - Decorator Pattern");
    println!("-------------------------------------");
    
    // Test the logging handler
    let mut rng = thread_rng();
    let base_handler = PriorHandler {
        rng: &mut rng,
        trace: Trace::default(),
    };
    let logging_handler = LoggingHandler::new(base_handler, false); // Non-verbose
    
    let logged_model = || prob!(
        let x <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
        observe(addr!("obs"), Normal::new(x, 0.5).unwrap(), 1.2);
        factor(-0.5);
        pure(x)
    );
    
    let (result, _trace) = runtime::handler::run(logging_handler, logged_model());
    println!("   - Logged execution result: {:.3}", result);
    println!();
    
    println!("3. Statistics Accumulating Handler");
    println!("--------------------------------");
    
    // Test the statistics handler
    let mut rng = thread_rng();
    let base_handler = PriorHandler {
        rng: &mut rng,
        trace: Trace::default(),
    };
    let stats_handler = StatisticsHandler::new(base_handler);
    
    let complex_model = || prob!(
        let mu <- sample(addr!("mu"), Normal::new(0.0, 2.0).unwrap());
        let is_outlier <- sample(addr!("outlier"), Bernoulli::new(0.1).unwrap());
        let count <- sample(addr!("count"), Poisson::new(3.0).unwrap());
        let category <- sample(addr!("category"), Categorical::new(vec![0.3, 0.4, 0.3]).unwrap());
        
        observe(addr!("y1"), Normal::new(mu, 1.0).unwrap(), 1.5);
        observe(addr!("y2"), Normal::new(mu, 1.0).unwrap(), 2.1);
        factor(if is_outlier { -2.0 } else { 0.0 });
        
        pure((mu, is_outlier, count, category))
    );
    
    let (result, _trace) = runtime::handler::run(stats_handler, complex_model());
    println!("   - Complex model result: {:?}", (result.0.round(), result.1, result.2, result.3));
    println!();
    
    println!("4. Conditional Filtering Handler");
    println!("-------------------------------");
    
    // Test the filtering handler
    let mut rng = thread_rng();
    let base_handler = PriorHandler {
        rng: &mut rng,
        trace: Trace::default(),
    };
    let filtering_handler = FilteringHandler::new(
        base_handler,
        Some((-2.0, 2.0)), // Clamp f64 values to [-2, 2]
        0.1 // 10% chance to flip booleans
    );
    
    let filter_test_model = || prob!(
        let x <- sample(addr!("x"), Normal::new(0.0, 5.0).unwrap()); // Wide distribution
        let flag <- sample(addr!("flag"), Bernoulli::new(0.8).unwrap());
        pure((x, flag))
    );
    
    let (result, _trace) = runtime::handler::run(filtering_handler, filter_test_model());
    println!("✅ Filtering handler executed");
    println!("   - Clamped value: {:.3} (should be in [-2, 2])", result.0);
    println!("   - Boolean value: {} (may be flipped from original)", result.1);
    println!();
    
    println!("5. Performance Monitoring Handler");
    println!("--------------------------------");
    
    // Test the performance handler
    let mut rng = thread_rng();
    let base_handler = PriorHandler {
        rng: &mut rng,
        trace: Trace::default(),
    };
    let perf_handler = PerformanceHandler::new(base_handler);
    
    let perf_test_model = || plate!(i in 0..10 => {
        sample(addr!("param", i), Normal::new(0.0, 1.0).unwrap())
    });
    
    let (_result, _trace) = runtime::handler::run(perf_handler, perf_test_model());
    println!();
    
    println!("6. Custom Inference Handler");
    println!("---------------------------");
    
    // Test the custom inference handler
    let mut rng1 = thread_rng();
    let rng2 = thread_rng();
    
    // First get a base trace
    let base_handler = PriorHandler {
        rng: &mut rng1,
        trace: Trace::default(),
    };
    
    let inference_model = || prob!(
        let mu <- sample(addr!("mu"), Normal::new(0.0, 1.0).unwrap());
        observe(addr!("y"), Normal::new(mu, 0.5).unwrap(), 1.0);
        pure(mu)
    );
    
    let (base_result, base_trace) = runtime::handler::run(base_handler, inference_model());
    
    // Now use custom MCMC handler to perturb it
    let base_log_weight = base_trace.total_log_weight();
    let mcmc_handler = SimpleMCMCHandler::new(rng2, base_trace, 0.1);
    let (mcmc_result, mcmc_trace) = runtime::handler::run(mcmc_handler, inference_model());
    
    println!("✅ Custom MCMC-like inference:");
    println!("   - Base result: {:.3}", base_result);
    println!("   - MCMC result: {:.3}", mcmc_result);
    println!("   - Base log-weight: {:.3}", base_log_weight);
    println!("   - MCMC log-weight: {:.3}", mcmc_trace.total_log_weight());
    println!();
    
    println!("7. Handler Composition and Chaining");
    println!("----------------------------------");
    
    // Demonstrate composing multiple handler decorators
    let mut rng = thread_rng();
    let base_handler = PriorHandler {
        rng: &mut rng,
        trace: Trace::default(),
    };
    
    // Chain multiple decorators: Statistics -> Logging -> Performance -> Base
    let stats_handler = StatisticsHandler::new(base_handler);
    let logging_handler = LoggingHandler::new(stats_handler, false);
    let performance_handler = PerformanceHandler::new(logging_handler);
    
    let composition_model = || prob!(
        let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
        let y <- sample(addr!("y"), Bernoulli::new(0.7).unwrap());
        observe(addr!("obs"), Normal::new(x, 0.2).unwrap(), 0.5);
        factor(-0.3);
        pure((x, y))
    );
    
    println!("✅ Handler composition example:");
    let (_result, _trace) = runtime::handler::run(performance_handler, composition_model());
    println!("   - Multiple handler layers executed successfully");
    println!();
    
    println!("=== Custom Handler Patterns Demonstrated! ===");
}
// ANCHOR_END: handler_composition

#[cfg(test)]
mod tests {
    use super::*;
    
    // ANCHOR: handler_testing
    #[test]
    fn test_basic_custom_handler() {
        let mut rng = thread_rng();
        let handler = BasicHandler {
            rng: &mut rng,
            trace: Trace::default(),
        };
        
        let model = sample(addr!("test"), Normal::new(0.0, 1.0).unwrap());
        let (result, trace) = runtime::handler::run(handler, model);
        
        assert!(trace.choices.contains_key(&addr!("test")));
        assert!(trace.total_log_weight().is_finite());
        assert!(result.is_finite());
    }
    
    #[test]
    fn test_logging_handler() {
        let mut rng = thread_rng();
        let base_handler = PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        };
        let logging_handler = LoggingHandler::new(base_handler, false);
        
        let model = prob!(
            let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
            observe(addr!("obs"), Normal::new(x, 0.1).unwrap(), 1.0);
            pure(x)
        );
        
        let (result, trace) = runtime::handler::run(logging_handler, model);
        
        assert!(trace.choices.contains_key(&addr!("x")));
        assert!(trace.log_likelihood.is_finite());
        assert!(result.is_finite());
    }
    
    #[test]
    fn test_statistics_handler() {
        let mut rng = thread_rng();
        let base_handler = PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        };
        let stats_handler = StatisticsHandler::new(base_handler);
        
        let model = prob!(
            let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
            let flag <- sample(addr!("flag"), Bernoulli::new(0.5).unwrap());
            pure((x, flag))
        );
        
        let (result, trace) = runtime::handler::run(stats_handler, model);
        
        assert_eq!(trace.choices.len(), 2);
        assert!(result.0.is_finite());
    }
    
    #[test] 
    fn test_handler_composition() {
        let mut rng = thread_rng();
        let base_handler = PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        };
        
        // Compose multiple handlers
        let logged_handler = LoggingHandler::new(base_handler, false);
        let stats_handler = StatisticsHandler::new(logged_handler);
        
        let model = prob!(
            let x <- sample(addr!("param"), Normal::new(0.0, 1.0).unwrap());
            factor(-0.5);
            pure(x)
        );
        
        let (result, trace) = runtime::handler::run(stats_handler, model);
        
        assert!(trace.choices.contains_key(&addr!("param")));
        assert!(trace.log_factors.abs() > 0.0); // Factor was applied
        assert!(result.is_finite());
    }
    // ANCHOR_END: handler_testing
}