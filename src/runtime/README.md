# Runtime Module

The runtime module provides interpretation and execution of probabilistic models:

## Components

### `handler.rs` - Type-Safe Handler Interface

- `Handler` trait: Defines how to interpret model effects with full type safety
- `run` function: Executes a model with a handler, returning value and trace

```rust
pub trait Handler {
    // Type-specific sampling handlers
    fn on_sample_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>) -> f64;
    fn on_sample_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>) -> bool;
    fn on_sample_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>) -> u64;
    fn on_sample_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>) -> usize;
    
    // Type-specific observation handlers
    fn on_observe_f64(&mut self, addr: &Address, dist: &dyn Distribution<f64>, value: f64);
    fn on_observe_bool(&mut self, addr: &Address, dist: &dyn Distribution<bool>, value: bool);
    fn on_observe_u64(&mut self, addr: &Address, dist: &dyn Distribution<u64>, value: u64);
    fn on_observe_usize(&mut self, addr: &Address, dist: &dyn Distribution<usize>, value: usize);
    
    fn on_factor(&mut self, logw: f64);
    fn finish(self) -> Trace where Self: Sized;
}

let (result, trace) = run(handler, model);
```

### `interpreters.rs` - Built-in Handlers

- `PriorHandler`: Samples from priors, accumulates log-densities
- `ReplayHandler`: Reuses values from a base trace, falls back to sampling
- `ScoreGivenTrace`: Scores a fixed trace under the model

```rust
// Prior sampling
let (value, trace) = run(PriorHandler{rng: &mut rng, trace: Trace::default()}, model);

// Replay with different observations
let (value2, trace2) = run(ReplayHandler{rng: &mut rng, base: trace, trace: Trace::default()}, model2);

// Score existing trace
let (value3, trace3) = run(ScoreGivenTrace{base: trace, trace: Trace::default()}, model);
```

### `trace.rs` - Execution Traces

- `Trace`: Records choices and accumulated log-weights
- `Choice`: Individual random choice with address, value, and log-probability  
- `ChoiceValue`: Type-safe value storage - supports `F64`, `Bool`, `U64`, `Usize`, `I64`

```rust
#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub choices: BTreeMap<Address, Choice>,
    pub log_prior: f64,
    pub log_likelihood: f64,
    pub log_factors: f64,
}

impl Trace {
    pub fn total_log_weight(&self) -> f64 {
        self.log_prior + self.log_likelihood + self.log_factors
    }
}
```

## Usage Patterns

### Basic Execution

```rust
let model = sample(addr!("x"), Normal{mu: 0.0, sigma: 1.0});
let mut rng = thread_rng();
let (x, trace) = run(PriorHandler{rng: &mut rng, trace: Trace::default()}, model);
```

### Trace Manipulation

```rust
// Generate base trace
let (_, base_trace) = run(PriorHandler{rng: &mut rng, trace: Trace::default()}, model);

// Replay with same random choices but different model
let (_, new_trace) = run(ReplayHandler{rng: &mut rng, base: base_trace, trace: Trace::default()}, different_model);

// Score a specific configuration
let (_, scored_trace) = run(ScoreGivenTrace{base: fixed_trace, trace: Trace::default()}, model);
```

## Design Principles

- **Effect Handlers**: Clean separation between model definition and execution
- **Trace-based**: All execution produces replayable, scorable traces
- **Composable**: Handlers can be chained and combined
- **Deterministic**: Same trace + same model = same result
- **Introspectable**: Full visibility into random choices and weights
