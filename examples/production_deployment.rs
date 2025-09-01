use fugue::runtime::handler::Handler;
use fugue::runtime::interpreters::PriorHandler;
use fugue::runtime::memory::{PooledPriorHandler, TracePool};
use fugue::runtime::trace::{ChoiceValue, Trace};
use fugue::*;
use rand::thread_rng;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ANCHOR: error_handling
/// Production-ready handler that gracefully handles failures
struct RobustProductionHandler<H: Handler> {
    inner: H,
    error_count: u32,
    max_errors: u32,
    _fallback_values: HashMap<String, ChoiceValue>,
    circuit_breaker_open: bool,
}

impl<H: Handler> RobustProductionHandler<H> {
    fn new(inner: H, max_errors: u32) -> Self {
        let mut fallback_values = HashMap::new();
        fallback_values.insert("default_f64".to_string(), ChoiceValue::F64(0.0));
        fallback_values.insert("default_bool".to_string(), ChoiceValue::Bool(false));
        fallback_values.insert("default_u64".to_string(), ChoiceValue::U64(0));
        fallback_values.insert("default_usize".to_string(), ChoiceValue::Usize(0));

        Self {
            inner,
            error_count: 0,
            max_errors,
            _fallback_values: fallback_values,
            circuit_breaker_open: false,
        }
    }

    fn handle_error(&mut self, operation: &str, addr: &Address) -> bool {
        self.error_count += 1;
        eprintln!("PRODUCTION ERROR: {} failed at address {}", operation, addr);

        if self.error_count >= self.max_errors {
            self.circuit_breaker_open = true;
            eprintln!("CIRCUIT BREAKER: Too many errors, switching to fallback mode");
        }

        self.circuit_breaker_open
    }

    fn get_fallback_f64(&self, addr: &Address) -> f64 {
        // In production, this might come from a cache, configuration, or ML model
        match addr.0.as_str() {
            s if s.contains("temperature") => 20.0,
            s if s.contains("price") => 100.0,
            s if s.contains("probability") => 0.5,
            _ => 0.0,
        }
    }
}

impl<H: Handler> Handler for RobustProductionHandler<H> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        if self.circuit_breaker_open {
            return self.get_fallback_f64(addr);
        }

        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.inner.on_sample_f64(addr, dist)
        })) {
            Ok(value) if value.is_finite() => value,
            Ok(invalid_value) => {
                eprintln!("Invalid f64 sample: {} at {}", invalid_value, addr);
                self.handle_error("sample_f64", addr);
                self.get_fallback_f64(addr)
            }
            Err(_) => {
                self.handle_error("sample_f64_panic", addr);
                self.get_fallback_f64(addr)
            }
        }
    }

    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        if self.circuit_breaker_open {
            return false; // Safe fallback
        }

        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.inner.on_sample_bool(addr, dist)
        })) {
            Ok(value) => value,
            Err(_) => {
                self.handle_error("sample_bool_panic", addr);
                false
            }
        }
    }

    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64 {
        if self.circuit_breaker_open {
            return 1; // Safe default
        }

        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.inner.on_sample_u64(addr, dist)
        })) {
            Ok(value) => value,
            Err(_) => {
                self.handle_error("sample_u64_panic", addr);
                1
            }
        }
    }

    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize {
        if self.circuit_breaker_open {
            return 0; // Safe array index
        }

        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.inner.on_sample_usize(addr, dist)
        })) {
            Ok(value) => value,
            Err(_) => {
                self.handle_error("sample_usize_panic", addr);
                0
            }
        }
    }

    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64) {
        if !self.circuit_breaker_open
            && std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.inner.on_observe_f64(addr, dist, value)
            }))
            .is_err()
        {
            self.handle_error("observe_f64_panic", addr);
        }
    }

    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool) {
        if !self.circuit_breaker_open
            && std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.inner.on_observe_bool(addr, dist, value)
            }))
            .is_err()
        {
            self.handle_error("observe_bool_panic", addr);
        }
    }

    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64) {
        if !self.circuit_breaker_open
            && std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.inner.on_observe_u64(addr, dist, value)
            }))
            .is_err()
        {
            self.handle_error("observe_u64_panic", addr);
        }
    }

    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize) {
        if !self.circuit_breaker_open
            && std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.inner.on_observe_usize(addr, dist, value)
            }))
            .is_err()
        {
            self.handle_error("observe_usize_panic", addr);
        }
    }

    fn on_factor(&mut self, log_weight: f64) {
        if !log_weight.is_finite() {
            eprintln!("Invalid factor log-weight: {}", log_weight);
            self.error_count += 1;
            return; // Skip invalid factors
        }

        if !self.circuit_breaker_open {
            self.inner.on_factor(log_weight);
        }
    }

    fn finish(self) -> Trace {
        println!("✅ Production handler statistics:");
        println!(
            "   - Errors encountered: {}/{}",
            self.error_count, self.max_errors
        );
        println!(
            "   - Circuit breaker status: {}",
            if self.circuit_breaker_open {
                "OPEN (fallback mode)"
            } else {
                "CLOSED (normal)"
            }
        );

        if self.circuit_breaker_open {
            // Return minimal valid trace in fallback mode
            Trace::default()
        } else {
            self.inner.finish()
        }
    }
}
// ANCHOR_END: error_handling

// ANCHOR: configuration_management
#[derive(Debug, Clone)]
struct ModelConfig {
    // Model parameters
    temperature_prior_mean: f64,
    temperature_prior_std: f64,
    validity_probability: f64,
    sensor_noise_std: f64,

    // Runtime configuration
    max_inference_time_ms: u64,
    memory_pool_size: usize,
    enable_circuit_breaker: bool,
    error_threshold: u32,

    // Environment settings
    environment: String, // "development", "staging", "production"
    _log_level: String,
    enable_metrics: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            temperature_prior_mean: 20.0,
            temperature_prior_std: 5.0,
            validity_probability: 0.95,
            sensor_noise_std: 1.0,
            max_inference_time_ms: 1000,
            memory_pool_size: 100,
            enable_circuit_breaker: true,
            error_threshold: 10,
            environment: "production".to_string(),
            _log_level: "info".to_string(),
            enable_metrics: true,
        }
    }
}

struct ConfigurableModelRunner {
    config: ModelConfig,
    pool: TracePool,
    metrics: ProductionMetrics,
}

impl ConfigurableModelRunner {
    fn new(config: ModelConfig) -> Self {
        Self {
            pool: TracePool::new(config.memory_pool_size),
            metrics: ProductionMetrics::new(config.enable_metrics),
            config,
        }
    }

    fn create_model(&self) -> Model<(f64, bool)> {
        let config = self.config.clone();
        prob!(
            let temp <- sample(
                addr!("temperature"),
                Normal::new(config.temperature_prior_mean, config.temperature_prior_std).unwrap()
            );
            let valid <- sample(
                addr!("valid"),
                Bernoulli::new(config.validity_probability).unwrap()
            );
            // Simulate sensor reading with configured noise
            observe(
                addr!("sensor"),
                Normal::new(temp, config.sensor_noise_std).unwrap(),
                22.0
            );
            pure((temp, valid))
        )
    }

    fn run_inference(&mut self) -> Result<(f64, bool), String> {
        let start = Instant::now();

        // Configure handler based on environment
        let mut rng = thread_rng();
        let model = self.create_model(); // Create model before borrowing
        let result = if self.config.environment == "production" {
            // Use safe, fault-tolerant execution in production
            let base_handler = PooledPriorHandler::new(&mut rng, &mut self.pool);
            let robust_handler =
                RobustProductionHandler::new(base_handler, self.config.error_threshold);

            let (result, _trace) = runtime::handler::run(robust_handler, model);
            Ok(result)
        } else {
            // Use faster, less safe execution in development
            let handler = PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            };
            let (result, _trace) = runtime::handler::run(handler, model);
            Ok(result)
        };

        let duration = start.elapsed();
        self.metrics.record_inference_time(duration);

        // Check timeout
        if duration.as_millis() > self.config.max_inference_time_ms as u128 {
            self.metrics.increment_timeout_count();
            return Err(format!(
                "Inference timeout: {}ms > {}ms",
                duration.as_millis(),
                self.config.max_inference_time_ms
            ));
        }

        result
    }
}
// ANCHOR_END: configuration_management

// ANCHOR: production_metrics
#[derive(Debug, Clone)]
struct ProductionMetrics {
    enabled: bool,
    inference_count: u64,
    error_count: u64,
    timeout_count: u64,
    total_inference_time: Duration,
    start_time: SystemTime,
}

impl ProductionMetrics {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            inference_count: 0,
            error_count: 0,
            timeout_count: 0,
            total_inference_time: Duration::ZERO,
            start_time: SystemTime::now(),
        }
    }

    fn record_inference_time(&mut self, duration: Duration) {
        if self.enabled {
            self.inference_count += 1;
            self.total_inference_time += duration;
        }
    }

    fn _increment_error_count(&mut self) {
        if self.enabled {
            self.error_count += 1;
        }
    }

    fn increment_timeout_count(&mut self) {
        if self.enabled {
            self.timeout_count += 1;
        }
    }

    fn get_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        if self.enabled {
            let uptime = self.start_time.elapsed().unwrap_or(Duration::ZERO);
            let avg_inference_time = if self.inference_count > 0 {
                self.total_inference_time.as_millis() as f64 / self.inference_count as f64
            } else {
                0.0
            };

            stats.insert("inference_count".to_string(), self.inference_count as f64);
            stats.insert("error_count".to_string(), self.error_count as f64);
            stats.insert("timeout_count".to_string(), self.timeout_count as f64);
            stats.insert(
                "error_rate".to_string(),
                if self.inference_count > 0 {
                    self.error_count as f64 / self.inference_count as f64
                } else {
                    0.0
                },
            );
            stats.insert("avg_inference_time_ms".to_string(), avg_inference_time);
            stats.insert("uptime_seconds".to_string(), uptime.as_secs() as f64);
            stats.insert(
                "throughput_per_second".to_string(),
                if uptime.as_secs() > 0 {
                    self.inference_count as f64 / uptime.as_secs() as f64
                } else {
                    0.0
                },
            );
        }
        stats
    }

    fn export_prometheus_metrics(&self) -> String {
        let mut metrics = String::new();
        let stats = self.get_stats();

        metrics.push_str("# HELP fugue_inference_total Total number of inference runs\n");
        metrics.push_str("# TYPE fugue_inference_total counter\n");
        metrics.push_str(&format!(
            "fugue_inference_total {}\n",
            stats.get("inference_count").unwrap_or(&0.0)
        ));

        metrics.push_str("# HELP fugue_errors_total Total number of errors\n");
        metrics.push_str("# TYPE fugue_errors_total counter\n");
        metrics.push_str(&format!(
            "fugue_errors_total {}\n",
            stats.get("error_count").unwrap_or(&0.0)
        ));

        metrics.push_str(
            "# HELP fugue_inference_duration_ms Average inference duration in milliseconds\n",
        );
        metrics.push_str("# TYPE fugue_inference_duration_ms gauge\n");
        metrics.push_str(&format!(
            "fugue_inference_duration_ms {}\n",
            stats.get("avg_inference_time_ms").unwrap_or(&0.0)
        ));

        metrics.push_str("# HELP fugue_error_rate Error rate (errors/total inferences)\n");
        metrics.push_str("# TYPE fugue_error_rate gauge\n");
        metrics.push_str(&format!(
            "fugue_error_rate {}\n",
            stats.get("error_rate").unwrap_or(&0.0)
        ));

        metrics
    }
}

/// Production monitoring handler that integrates with metrics systems
struct MetricsHandler<H: Handler> {
    inner: H,
    metrics: Arc<std::sync::Mutex<ProductionMetrics>>,
    _model_name: String,
}

impl<H: Handler> MetricsHandler<H> {
    fn new(
        inner: H,
        metrics: Arc<std::sync::Mutex<ProductionMetrics>>,
        model_name: String,
    ) -> Self {
        Self {
            inner,
            metrics,
            _model_name: model_name,
        }
    }
}

impl<H: Handler> Handler for MetricsHandler<H> {
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64 {
        let start = Instant::now();
        let result = self.inner.on_sample_f64(addr, dist);

        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.record_inference_time(start.elapsed());
        }

        result
    }

    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool {
        self.inner.on_sample_bool(addr, dist)
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
// ANCHOR_END: production_metrics

// ANCHOR: health_checks
#[derive(Debug, Clone)]
struct HealthCheckResult {
    status: HealthStatus,
    message: String,
    details: HashMap<String, String>,
    timestamp: SystemTime,
}

#[derive(Debug, Clone, PartialEq)]
enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

struct ProductionHealthChecker {
    model_config: ModelConfig,
    metrics: Arc<std::sync::Mutex<ProductionMetrics>>,
}

impl ProductionHealthChecker {
    fn new(config: ModelConfig, metrics: Arc<std::sync::Mutex<ProductionMetrics>>) -> Self {
        Self {
            model_config: config,
            metrics,
        }
    }

    fn run_health_check(&self) -> HealthCheckResult {
        let mut details = HashMap::new();
        let mut overall_status = HealthStatus::Healthy;
        let mut messages = Vec::new();

        // Check 1: Model execution health
        match self.check_model_execution() {
            Ok(duration) => {
                details.insert("model_execution".to_string(), "healthy".to_string());
                details.insert(
                    "execution_time_ms".to_string(),
                    format!("{:.1}", duration.as_millis()),
                );
            }
            Err(e) => {
                overall_status = HealthStatus::Unhealthy;
                messages.push(format!("Model execution failed: {}", e));
                details.insert("model_execution".to_string(), "failed".to_string());
            }
        }

        // Check 2: Memory usage
        if let Some(pool_stats) = self.check_memory_health() {
            let hit_ratio = pool_stats.hit_ratio();
            details.insert("memory_hit_ratio".to_string(), format!("{:.2}%", hit_ratio));

            if hit_ratio < 50.0 {
                overall_status = HealthStatus::Degraded;
                messages.push("Low memory pool hit ratio".to_string());
            }
        }

        // Check 3: Error rates
        if let Ok(metrics) = self.metrics.lock() {
            let stats = metrics.get_stats();
            let error_rate = stats.get("error_rate").unwrap_or(&0.0) * 100.0;
            details.insert(
                "error_rate_percent".to_string(),
                format!("{:.2}%", error_rate),
            );

            if error_rate > 5.0 {
                overall_status = HealthStatus::Degraded;
                messages.push(format!("High error rate: {:.1}%", error_rate));
            } else if error_rate > 20.0 {
                overall_status = HealthStatus::Unhealthy;
                messages.push(format!("Critical error rate: {:.1}%", error_rate));
            }

            let avg_time = stats.get("avg_inference_time_ms").unwrap_or(&0.0);
            details.insert(
                "avg_inference_time_ms".to_string(),
                format!("{:.1}", avg_time),
            );

            if *avg_time > self.model_config.max_inference_time_ms as f64 * 0.8 {
                overall_status = HealthStatus::Degraded;
                messages.push("Inference time approaching timeout threshold".to_string());
            }
        }

        // Check 4: System resources
        details.insert(
            "memory_pool_size".to_string(),
            self.model_config.memory_pool_size.to_string(),
        );
        details.insert(
            "circuit_breaker".to_string(),
            if self.model_config.enable_circuit_breaker {
                "enabled".to_string()
            } else {
                "disabled".to_string()
            },
        );

        let message = if messages.is_empty() {
            "All systems healthy".to_string()
        } else {
            messages.join("; ")
        };

        HealthCheckResult {
            status: overall_status,
            message,
            details,
            timestamp: SystemTime::now(),
        }
    }

    fn check_model_execution(&self) -> Result<Duration, String> {
        let start = Instant::now();
        let mut rng = thread_rng();

        // Run a simplified version of the model for health checking
        let health_model = || {
            prob!(
                let value <- sample(addr!("health_check"), Normal::new(0.0, 1.0).unwrap());
                pure(value)
            )
        };

        let handler = PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        };

        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime::handler::run(handler, health_model())
        })) {
            Ok((result, trace)) => {
                if result.is_finite() && trace.total_log_weight().is_finite() {
                    Ok(start.elapsed())
                } else {
                    Err("Invalid model output".to_string())
                }
            }
            Err(_) => Err("Model execution panicked".to_string()),
        }
    }

    fn check_memory_health(&self) -> Option<fugue::runtime::memory::PoolStats> {
        // In a real implementation, this would check the actual memory pool
        // For demonstration, we'll create a temporary pool
        let pool = TracePool::new(10);
        Some(pool.stats().clone())
    }
}
// ANCHOR_END: health_checks

// ANCHOR: input_validation
/// Secure input validator for production model parameters
struct InputValidator;

impl InputValidator {
    fn validate_temperature(temp: f64) -> Result<f64, String> {
        match temp {
            t if !t.is_finite() => Err("Temperature must be finite".to_string()),
            t if t < -50.0 => Err("Temperature too low (< -50°C)".to_string()),
            t if t > 100.0 => Err("Temperature too high (> 100°C)".to_string()),
            t => Ok(t),
        }
    }

    fn validate_probability(p: f64) -> Result<f64, String> {
        match p {
            p if !p.is_finite() => Err("Probability must be finite".to_string()),
            p if p < 0.0 => Err("Probability must be non-negative".to_string()),
            p if p > 1.0 => Err("Probability must not exceed 1.0".to_string()),
            p => Ok(p),
        }
    }

    fn _validate_sensor_reading(reading: f64) -> Result<f64, String> {
        match reading {
            r if !r.is_finite() => Err("Sensor reading must be finite".to_string()),
            r if r.abs() > 1000.0 => Err("Sensor reading out of reasonable range".to_string()),
            r => Ok(r),
        }
    }

    fn sanitize_address_component(component: &str) -> Result<String, String> {
        // Prevent injection attacks in address components
        if component.chars().any(|c| !(c.is_alphanumeric() || c == '_' || c == '-')) {
            return Err("Address component contains invalid characters".to_string());
        }

        if component.len() > 50 {
            Err("Address component too long".to_string())
        } else if component.is_empty() {
            Err("Address component cannot be empty".to_string())
        } else {
            Ok(component.to_string())
        }
    }
}

/// Production model with comprehensive input validation
fn create_validated_model(
    temperature_reading: f64,
    sensor_id: &str,
    prior_prob: f64,
) -> Result<Model<(f64, bool)>, String> {
    // Validate all inputs before model creation
    let validated_temp = InputValidator::validate_temperature(temperature_reading)?;
    let validated_prob = InputValidator::validate_probability(prior_prob)?;
    let sanitized_sensor_id = InputValidator::sanitize_address_component(sensor_id)?;

    // Additional business logic validation
    if sanitized_sensor_id.starts_with("test_") && validated_prob > 0.5 {
        return Err("Test sensors cannot have high prior probability".to_string());
    }

    Ok(prob!(
        let true_temp <- sample(
            addr!("temperature"),
            Normal::new(validated_temp, 2.0).unwrap()
        );
        let is_working <- sample(
            addr!("sensor_working", sanitized_sensor_id.clone()),
            Bernoulli::new(validated_prob).unwrap()
        );

        // Safe observation with validated input
        observe(
            addr!("reading", sanitized_sensor_id),
            Normal::new(true_temp, if is_working { 0.5 } else { 5.0 }).unwrap(),
            validated_temp
        );

        pure((true_temp, is_working))
    ))
}
// ANCHOR_END: input_validation

// ANCHOR: deployment_strategies
/// Production deployment manager with different strategies
#[derive(Debug, Clone)]
enum DeploymentStrategy {
    BlueGreen,
    CanaryRelease { percentage: f64 },
    RollingUpdate,
    _ImmediateSwitch,
}

struct ModelDeploymentManager {
    current_model_version: String,
    candidate_model_version: String,
    deployment_strategy: DeploymentStrategy,
    _rollback_threshold_error_rate: f64,
}

impl ModelDeploymentManager {
    fn new(strategy: DeploymentStrategy) -> Self {
        Self {
            current_model_version: "v1.0.0".to_string(),
            candidate_model_version: "v1.1.0".to_string(),
            deployment_strategy: strategy,
            _rollback_threshold_error_rate: 0.05, // 5% error rate triggers rollback
        }
    }

    fn should_use_candidate_model(&self, request_id: u64) -> bool {
        match &self.deployment_strategy {
            DeploymentStrategy::BlueGreen => {
                // In blue-green, we typically switch all traffic at once
                // For demo, we'll use request ID to simulate the switch
                request_id % 100 < 10 // 10% to candidate for testing
            }
            DeploymentStrategy::CanaryRelease { percentage } => {
                let hash = request_id % 100;
                (hash as f64) < (*percentage * 100.0)
            }
            DeploymentStrategy::RollingUpdate => {
                // Gradual rollout based on some criteria
                request_id % 10 < 3 // 30% rollout
            }
            DeploymentStrategy::_ImmediateSwitch => true,
        }
    }

    fn create_model(&self, use_candidate: bool) -> impl Fn() -> Model<f64> {
        let version = if use_candidate {
            self.candidate_model_version.clone()
        } else {
            self.current_model_version.clone()
        };

        move || {
            if version.starts_with("v1.1") {
                // Candidate model with improved parameters
                prob!(
                    let value <- sample(addr!("improved_param"), Normal::new(0.0, 0.8).unwrap());
                    factor(0.1); // Slight preference for this model
                    pure(value)
                )
            } else {
                // Current stable model
                prob!(
                    let value <- sample(addr!("stable_param"), Normal::new(0.0, 1.0).unwrap());
                    pure(value)
                )
            }
        }
    }

    fn process_request(&self, request_id: u64) -> Result<(f64, String), String> {
        let use_candidate = self.should_use_candidate_model(request_id);
        let version = if use_candidate {
            &self.candidate_model_version
        } else {
            &self.current_model_version
        };

        let model = self.create_model(use_candidate);

        // Execute with error handling
        let mut rng = thread_rng();
        let handler = PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        };

        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime::handler::run(handler, model())
        })) {
            Ok((result, trace)) => {
                if result.is_finite() && trace.total_log_weight().is_finite() {
                    Ok((result, version.clone()))
                } else {
                    Err(format!("Invalid result from model {}", version))
                }
            }
            Err(_) => Err(format!("Model {} panicked", version)),
        }
    }
}
// ANCHOR_END: deployment_strategies

fn main() {
    println!("=== Production Deployment Patterns for Fugue ===\n");

    println!("1. Error Handling and Graceful Degradation");
    println!("-----------------------------------------");

    // Test the robust handler
    let mut rng = thread_rng();
    let base_handler = PriorHandler {
        rng: &mut rng,
        trace: Trace::default(),
    };
    let robust_handler = RobustProductionHandler::new(base_handler, 5);

    let production_model = || {
        prob!(
            let temperature <- sample(addr!("temperature"), Normal::new(20.0, 5.0).unwrap());
            let is_valid <- sample(addr!("valid"), Bernoulli::new(0.95).unwrap());
            observe(addr!("sensor"), Normal::new(temperature, 1.0).unwrap(), 18.5);
            pure((temperature, is_valid))
        )
    };

    let (result, _trace) = runtime::handler::run(robust_handler, production_model());
    println!("   - Result: temp={:.1}°C, valid={}", result.0, result.1);
    println!();

    println!("2. Configuration Management");
    println!("--------------------------");

    // Test configuration management
    let config = ModelConfig {
        environment: "production".to_string(),
        temperature_prior_mean: 25.0,
        validity_probability: 0.98,
        max_inference_time_ms: 100,
        ..Default::default()
    };

    let mut runner = ConfigurableModelRunner::new(config.clone());
    match runner.run_inference() {
        Ok((temp, valid)) => {
            println!("✅ Configured inference completed");
            println!("   - Environment: {}", config.environment);
            println!("   - Result: temp={:.1}°C, valid={}", temp, valid);
            println!("   - Pool stats: {:?}", runner.pool.stats());
        }
        Err(e) => println!("❌ Inference failed: {}", e),
    }
    println!();

    println!("3. Production Metrics and Observability");
    println!("--------------------------------------");

    // Test metrics collection
    let metrics = Arc::new(std::sync::Mutex::new(ProductionMetrics::new(true)));

    // Simulate some inference runs
    for i in 0..5 {
        let mut rng = thread_rng();
        let base_handler = PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        };
        let metrics_handler =
            MetricsHandler::new(base_handler, metrics.clone(), format!("sensor_model_{}", i));

        let simple_model = || sample(addr!("value"), Normal::new(0.0, 1.0).unwrap());
        let (_result, _trace) = runtime::handler::run(metrics_handler, simple_model());

        // Simulate some processing time
        std::thread::sleep(Duration::from_millis(1));
    }

    // Export metrics
    if let Ok(metrics_guard) = metrics.lock() {
        let stats = metrics_guard.get_stats();
        println!("✅ Production metrics collected:");
        for (key, value) in stats {
            println!("   - {}: {:.3}", key, value);
        }

        println!("✅ Prometheus metrics format:");
        let prometheus = metrics_guard.export_prometheus_metrics();
        for line in prometheus.lines().take(6) {
            println!("   {}", line);
        }
    }
    println!();

    println!("4. Health Checks and System Validation");
    println!("-------------------------------------");

    // Test health checks
    let config = ModelConfig::default();
    let metrics = Arc::new(std::sync::Mutex::new(ProductionMetrics::new(true)));
    let health_checker = ProductionHealthChecker::new(config, metrics);

    let health_result = health_checker.run_health_check();

    println!("✅ Health Check Results:");
    println!("   - Status: {:?}", health_result.status);
    println!("   - Message: {}", health_result.message);
    println!("   - Details:");
    for (key, value) in &health_result.details {
        println!("     {}: {}", key, value);
    }
    println!(
        "   - Timestamp: {:?}",
        health_result
            .timestamp
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );
    println!();

    println!("5. Input Validation and Security");
    println!("-------------------------------");

    // Test input validation
    let test_cases = vec![
        (25.0, "sensor_001", 0.95), // Valid case
        (150.0, "sensor_002", 0.8), // Invalid temperature
        (20.0, "sensor_003", 1.5),  // Invalid probability
        (22.0, "test_sensor", 0.9), // Business rule violation
        (18.0, "sensor@#$%", 0.7),  // Invalid sensor ID
    ];

    for (temp, sensor, prob) in test_cases {
        match create_validated_model(temp, sensor, prob) {
            Ok(model) => {
                let mut rng = thread_rng();
                let handler = PriorHandler {
                    rng: &mut rng,
                    trace: Trace::default(),
                };
                let (result, _trace) = runtime::handler::run(handler, model);
                println!(
                    "✅ Valid input: temp={:.1}, sensor={}, prob={:.2} → result=({:.1}, {})",
                    temp, sensor, prob, result.0, result.1
                );
            }
            Err(e) => {
                println!(
                    "❌ Invalid input: temp={:.1}, sensor={}, prob={:.2} → error: {}",
                    temp, sensor, prob, e
                );
            }
        }
    }
    println!();

    println!("6. Deployment Strategies and Patterns");
    println!("------------------------------------");

    // Test different deployment strategies
    let strategies = vec![
        ("Blue-Green", DeploymentStrategy::BlueGreen),
        (
            "Canary 20%",
            DeploymentStrategy::CanaryRelease { percentage: 0.2 },
        ),
        ("Rolling Update", DeploymentStrategy::RollingUpdate),
    ];

    for (name, strategy) in strategies {
        println!("✅ Testing {} deployment:", name);
        let manager = ModelDeploymentManager::new(strategy);

        let mut v1_count = 0;
        let mut v1_1_count = 0;

        // Simulate 20 requests
        for request_id in 0..20 {
            match manager.process_request(request_id) {
                Ok((_result, version)) => {
                    if version.starts_with("v1.1") {
                        v1_1_count += 1;
                    } else {
                        v1_count += 1;
                    }
                }
                Err(e) => eprintln!("   Request {} failed: {}", request_id, e),
            }
        }

        println!("   - v1.0.0 requests: {}", v1_count);
        println!("   - v1.1.0 requests: {}", v1_1_count);
        println!(
            "   - Traffic split: {:.1}% / {:.1}%",
            (v1_count as f64 / 20.0) * 100.0,
            (v1_1_count as f64 / 20.0) * 100.0
        );
    }
    println!();

    println!("=== Production Deployment Patterns Demonstrated! ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    // ANCHOR: production_tests
    #[test]
    fn test_robust_error_handling() {
        let mut rng = thread_rng();
        let base_handler = PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        };
        let robust_handler = RobustProductionHandler::new(base_handler, 3);

        let model = prob!(
            let x <- sample(addr!("test_param"), Normal::new(0.0, 1.0).unwrap());
            pure(x)
        );

        let (result, trace) = runtime::handler::run(robust_handler, model);

        assert!(result.is_finite());
        assert!(trace.total_log_weight().is_finite());
    }

    #[test]
    fn test_input_validation() {
        // Valid inputs should succeed
        let valid_model = create_validated_model(25.0, "sensor_001", 0.95);
        assert!(valid_model.is_ok());

        // Invalid temperature should fail
        let invalid_temp = create_validated_model(150.0, "sensor_002", 0.8);
        assert!(invalid_temp.is_err());

        // Invalid probability should fail
        let invalid_prob = create_validated_model(20.0, "sensor_003", 1.5);
        assert!(invalid_prob.is_err());

        // Invalid sensor ID should fail
        let invalid_sensor = create_validated_model(22.0, "sensor@#$%", 0.7);
        assert!(invalid_sensor.is_err());
    }

    #[test]
    fn test_deployment_strategies() {
        let canary_manager =
            ModelDeploymentManager::new(DeploymentStrategy::CanaryRelease { percentage: 0.3 });

        let mut candidate_count = 0;
        let total_requests = 100;

        for request_id in 0..total_requests {
            if canary_manager.should_use_candidate_model(request_id) {
                candidate_count += 1;
            }
        }

        // Should be approximately 30% candidate usage
        let percentage = candidate_count as f64 / total_requests as f64;
        assert!(
            percentage > 0.25 && percentage < 0.35,
            "Canary percentage was {:.2}, expected ~0.30",
            percentage
        );
    }

    #[test]
    fn test_health_check_system() {
        let config = ModelConfig::default();
        let metrics = Arc::new(std::sync::Mutex::new(ProductionMetrics::new(true)));
        let health_checker = ProductionHealthChecker::new(config, metrics);

        let health_result = health_checker.run_health_check();

        // Health check should complete successfully
        assert!(!health_result.details.is_empty());
        assert!(!health_result.message.is_empty());
    }

    #[test]
    fn test_metrics_collection() {
        let mut metrics = ProductionMetrics::new(true);

        // Record some metrics
        metrics.record_inference_time(Duration::from_millis(50));
        metrics.record_inference_time(Duration::from_millis(75));
        metrics._increment_error_count();

        let stats = metrics.get_stats();

        assert_eq!(stats.get("inference_count").unwrap(), &2.0);
        assert_eq!(stats.get("error_count").unwrap(), &1.0);
        assert_eq!(stats.get("error_rate").unwrap(), &0.5); // 50% error rate

        // Test prometheus export
        let prometheus = metrics.export_prometheus_metrics();
        assert!(prometheus.contains("fugue_inference_total 2"));
        assert!(prometheus.contains("fugue_errors_total 1"));
    }

    #[test]
    fn test_configuration_management() {
        let config = ModelConfig {
            environment: "test".to_string(),
            temperature_prior_mean: 30.0,
            max_inference_time_ms: 50,
            ..Default::default()
        };

        let mut runner = ConfigurableModelRunner::new(config.clone());

        match runner.run_inference() {
            Ok((temp, _valid)) => {
                // Temperature should be influenced by the configured prior mean
                assert!(temp > -50.0 && temp < 100.0);
            }
            Err(e) => {
                // Timeout errors are acceptable in tests
                if !e.contains("timeout") {
                    panic!("Unexpected error: {}", e);
                }
            }
        }
    }
    // ANCHOR_END: production_tests
}
