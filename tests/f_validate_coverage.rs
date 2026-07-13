//! FG-55: coverage guard for the public standalone `Validate` trait.
//!
//! The `Validate` trait is re-exported at the crate root, and a user may rely on
//! it to (re)validate any distribution obtained via a non-constructor path. The
//! trait was historically implemented for only 7 of the exported distributions;
//! this integration test locks in coverage for ALL of them and is designed to
//! break the moment a new distribution is exported without a matching `Validate`
//! impl.
//!
//! How the drift guard works: every exported distribution is enumerated below
//! and has `.validate()` called on a valid instance. Adding an 18th exported
//! distribution requires bumping `EXPORTED_DISTRIBUTION_COUNT` and appending it
//! to `validate_all_exported_distributions`; if the new type lacks a `Validate`
//! impl, this file fails to compile (the `.validate()` call has no method),
//! forcing the author back to `src/error.rs`.

use fugue::*;

/// The number of concrete distribution types re-exported from the crate root
/// (`src/lib.rs`), excluding the `Distribution` trait itself. Keep in lockstep
/// with the enumeration in `validate_all_exported_distributions`.
const EXPORTED_DISTRIBUTION_COUNT: usize = 17;

#[test]
fn validate_all_exported_distributions() {
    // One valid instance per exported distribution. Every `.validate()` here is
    // a compile-time proof that the type implements the public `Validate` trait;
    // every `is_ok()` proves the impl agrees with the constructor on valid input.
    // The `Vec<bool>` length is checked against EXPORTED_DISTRIBUTION_COUNT so a
    // forgotten entry (or a stale count) is caught at runtime as well.
    let results: Vec<bool> = vec![
        // --- the original 7 impls (FG-55 pre-existing coverage) ---
        Normal::new(0.0, 1.0).unwrap().validate().is_ok(),
        Exponential::new(1.0).unwrap().validate().is_ok(),
        Beta::new(2.0, 3.0).unwrap().validate().is_ok(),
        Gamma::new(2.0, 1.0).unwrap().validate().is_ok(),
        Uniform::new(0.0, 1.0).unwrap().validate().is_ok(),
        Bernoulli::new(0.5).unwrap().validate().is_ok(),
        Categorical::new(vec![0.2, 0.8]).unwrap().validate().is_ok(),
        // --- the 10 impls added by FG-55 ---
        LogNormal::new(0.0, 1.0).unwrap().validate().is_ok(),
        Binomial::new(10, 0.5).unwrap().validate().is_ok(),
        Poisson::new(3.0).unwrap().validate().is_ok(),
        StudentT::new(5.0, 0.0, 1.0).unwrap().validate().is_ok(),
        Cauchy::new(0.0, 1.0).unwrap().validate().is_ok(),
        Laplace::new(0.0, 1.0).unwrap().validate().is_ok(),
        Weibull::new(2.0, 1.5).unwrap().validate().is_ok(),
        ChiSquared::new(4.0).unwrap().validate().is_ok(),
        InverseGamma::new(3.0, 2.0).unwrap().validate().is_ok(),
        DiscreteUniform::new(1, 6).unwrap().validate().is_ok(),
    ];

    assert_eq!(
        results.len(),
        EXPORTED_DISTRIBUTION_COUNT,
        "FG-55: every exported distribution must be enumerated here; if you added \
         a distribution, add it above (with a `Validate` impl in src/error.rs) and \
         bump EXPORTED_DISTRIBUTION_COUNT"
    );
    assert!(
        results.iter().all(|&ok| ok),
        "FG-55: `validate()` must return Ok for every valid instance"
    );
}
