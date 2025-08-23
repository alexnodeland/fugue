use fugue::*;
use proptest::prelude::*;
use rand::{rngs::StdRng, SeedableRng};

// Helper to create simple models for testing
fn make_simple_model(x: f64) -> Model<f64> {
    pure(x)
}

fn make_sample_model() -> Model<f64> {
    sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
}

proptest! {
    // Test functor laws
    #[test]
    fn functor_identity_law(x in any::<f64>()) {
        let model = make_simple_model(x);
        let mapped = model.map(|y| y);

        let mut rng = StdRng::seed_from_u64(42);
        let (result1, _) = runtime::handler::run(
            runtime::interpreters::PriorHandler{rng: &mut rng, trace: Trace::default()},
            make_simple_model(x)
        );

        let mut rng = StdRng::seed_from_u64(42);
        let (result2, _) = runtime::handler::run(
            runtime::interpreters::PriorHandler{rng: &mut rng, trace: Trace::default()},
            mapped
        );

        prop_assert!((result1 - result2).abs() < 1e-10);
    }

    #[test]
    fn functor_composition_law(x in -100.0..100.0f64, a in -10.0..10.0f64, b in -10.0..10.0f64) {
        let f = move |y: f64| y + a;
        let g = move |y: f64| y * b;

        let model1 = make_simple_model(x).map(f).map(g);
        let model2 = make_simple_model(x).map(move |y| g(f(y)));

        let mut rng = StdRng::seed_from_u64(42);
        let (result1, _) = runtime::handler::run(
            runtime::interpreters::PriorHandler{rng: &mut rng, trace: Trace::default()},
            model1
        );

        let mut rng = StdRng::seed_from_u64(42);
        let (result2, _) = runtime::handler::run(
            runtime::interpreters::PriorHandler{rng: &mut rng, trace: Trace::default()},
            model2
        );

        prop_assert!((result1 - result2).abs() < 1e-10);
    }

    // Test monad laws
    #[test]
    fn monad_left_identity_law(x in any::<f64>(), a in -5.0..5.0f64) {
        let f = move |y: f64| pure(y + a);

        let model1 = pure(x).bind(f);
        let model2 = f(x);

        let mut rng = StdRng::seed_from_u64(42);
        let (result1, _) = runtime::handler::run(
            runtime::interpreters::PriorHandler{rng: &mut rng, trace: Trace::default()},
            model1
        );

        let mut rng = StdRng::seed_from_u64(42);
        let (result2, _) = runtime::handler::run(
            runtime::interpreters::PriorHandler{rng: &mut rng, trace: Trace::default()},
            model2
        );

        prop_assert!((result1 - result2).abs() < 1e-10);
    }

    #[test]
    fn monad_right_identity_law(x in any::<f64>()) {
        let model1 = make_simple_model(x).bind(pure);
        let model2 = make_simple_model(x);

        let mut rng = StdRng::seed_from_u64(42);
        let (result1, _) = runtime::handler::run(
            runtime::interpreters::PriorHandler{rng: &mut rng, trace: Trace::default()},
            model1
        );

        let mut rng = StdRng::seed_from_u64(42);
        let (result2, _) = runtime::handler::run(
            runtime::interpreters::PriorHandler{rng: &mut rng, trace: Trace::default()},
            model2
        );

        prop_assert!((result1 - result2).abs() < 1e-10);
    }

    // Test trace invariants
    #[test]
    fn replay_preserves_choices(seed in any::<u64>()) {
        let model = make_sample_model();

        // Generate base trace
        let mut rng = StdRng::seed_from_u64(seed);
        let (_, base_trace) = runtime::handler::run(
            runtime::interpreters::PriorHandler{rng: &mut rng, trace: Trace::default()},
            model
        );

        // Replay should preserve the choice
        let mut rng = StdRng::seed_from_u64(seed + 1);
        let (_, replay_trace) = runtime::handler::run(
            runtime::interpreters::ReplayHandler{rng: &mut rng, base: base_trace.clone(), trace: Trace::default()},
            make_sample_model()
        );

        // Check that the choice value is preserved
        let base_choice = base_trace.choices.get(&addr!("x")).unwrap();
        let replay_choice = replay_trace.choices.get(&addr!("x")).unwrap();

        match (&base_choice.value, &replay_choice.value) {
            (ChoiceValue::F64(v1), ChoiceValue::F64(v2)) => {
                prop_assert!((v1 - v2).abs() < 1e-10);
            },
            _ => prop_assert!(false, "Expected F64 values"),
        }
    }

    #[test]
    fn score_matches_prior_for_no_observations(seed in any::<u64>()) {
        let model = sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());

        // Generate trace from prior
        let mut rng = StdRng::seed_from_u64(seed);
        let (_, base_trace) = runtime::handler::run(
            runtime::interpreters::PriorHandler{rng: &mut rng, trace: Trace::default()},
            model
        );

        // Score the same trace
        let (_, scored_trace) = runtime::handler::run(
            runtime::interpreters::ScoreGivenTrace{base: base_trace.clone(), trace: Trace::default()},
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
        );

        // Log weights should match (no observations = only prior terms)
        let base_weight = base_trace.total_log_weight();
        let scored_weight = scored_trace.total_log_weight();

        prop_assert!((base_weight - scored_weight).abs() < 1e-10);
    }

    // Test distribution properties
    #[test]
    fn normal_symmetry(mu in -5.0..5.0f64, sigma in 0.1..5.0f64, offset in 0.1..2.0f64) {
        let dist = Normal::new(mu, sigma).unwrap();
        let lp1 = dist.log_prob(&(mu + offset));
        let lp2 = dist.log_prob(&(mu - offset));

        prop_assert!((lp1 - lp2).abs() < 1e-10);
    }

    #[test]
    fn uniform_support(low in -10.0..0.0f64, high in 1.0..10.0f64) {
        prop_assume!(low < high);
        let dist = Uniform::new(low, high).unwrap();

        // Inside support should have finite log prob
        let inside = (low + high) / 2.0;
        let lp_inside = dist.log_prob(&inside);
        prop_assert!(lp_inside.is_finite());

        // Outside support should have -inf log prob
        let outside_low = low - 1.0;
        let outside_high = high + 1.0;
        prop_assert_eq!(dist.log_prob(&outside_low), f64::NEG_INFINITY);
        prop_assert_eq!(dist.log_prob(&outside_high), f64::NEG_INFINITY);
    }

    #[test]
    fn bernoulli_support(p in 0.01..0.99f64) {
        let dist = Bernoulli::new(p).unwrap();

        // Valid outcomes
        let lp0 = dist.log_prob(&false);  // Use natural bool values
        let lp1 = dist.log_prob(&true);
        prop_assert!(lp0.is_finite());
        prop_assert!(lp1.is_finite());

        // No invalid outcomes for bool - only true/false are valid
        // This test is no longer applicable since Bernoulli only accepts bool values
    }
}
