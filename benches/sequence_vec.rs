//! Benchmark for large sequences of simple observations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn benchmark_stack_overflow_model(c: &mut Criterion) {
    let n = 1_000usize;

    c.bench_function("stack_overflow_model/sequence_vec", |b| {
        b.iter(|| {
            let models: Vec<Model<bool>> = (0..n)
                .map(|i| sample(addr!("coin", i), Bernoulli::new(0.5).unwrap()))
                .collect();
            let model = sequence_vec(models);
            let mut rng = StdRng::seed_from_u64(42);
            let handler = runtime::interpreters::PriorHandler {
                rng: &mut rng,
                trace: runtime::trace::Trace::default(),
            };
            let (result, _trace) = runtime::handler::run(handler, model);
            black_box(result.len());
        })
    });
}

criterion_group!(stack_overflow_benches, benchmark_stack_overflow_model);
criterion_main!(stack_overflow_benches);
