# `run` function

Execute a probabilistic model using the given handler.

This is the core execution engine for probabilistic models. It interprets a `Model<A>` by dispatching effects to the provided handler and returns both the model's final result and the accumulated execution trace. The execution proceeds by pattern matching on the model structure:

- `Pure` values are returned directly
- `SampleF64` operations are handled by calling `handler.on_sample_f64`
- `SampleBool` operations are handled by calling `handler.on_sample_bool`
- `SampleU64` operations are handled by calling `handler.on_sample_u64`
- `SampleUsize` operations are handled by calling `handler.on_sample_usize`
- `ObserveF64` operations are handled by calling `handler.on_observe_f64`
- `ObserveBool` operations are handled by calling `handler.on_observe_bool`
- `ObserveU64` operations are handled by calling `handler.on_observe_u64`
- `ObserveUsize` operations are handled by calling `handler.on_observe_usize`
- `Factor` operations are handled by calling `handler.on_factor`

## Arguments

- `h` - Handler that defines how to interpret effects
- `m` - Model to execute

## Returns

A tuple containing:

- The final result of type `A` produced by the model
- The execution trace recording all choices and weights

## Examples

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
// Execute a simple model
let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
    .map(|x| x * 2.0);
let mut rng = StdRng::seed_from_u64(42);
let handler = PriorHandler {
    rng: &mut rng,
    trace: Trace::default(),
};
let (result, trace) = runtime::handler::run(handler, model);
println!("Result: {}, Log-weight: {}", result, trace.total_log_weight());
```
