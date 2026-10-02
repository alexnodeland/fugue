//! Cross-target reproducibility guard.
//!
//! A seeded inference run must make the same random choices on wasm32 as on a
//! 64-bit host. This file is deliberately *not* gated to wasm32: CI runs it
//! natively (`cargo test -p fugue-wasm`) and in Node (`wasm-pack test --node
//! crates/fugue-wasm`), against the same pinned constants. If either target
//! drifts, one of the two runs fails.
//!
//! The pinned values are discrete on purpose: which site each step moved, the
//! accept count, and the next raw `u64` word of the generator after the run.
//! Floating-point results are not pinned, because `ln`/`exp` on
//! wasm32-unknown-unknown come from a different libm than on the host and may
//! differ in the last bit. The final raw word does catch the original bug: a
//! `usize` index drawn with `gen_range` consumed one `u32` per step on wasm32
//! and one `u64` natively, so the two streams parted after the first pick.

use fugue::inference::mh::adaptive_single_site_mh;
use fugue::*;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;

/// Four sites, so the single-site kernel's uniform site pick matters.
fn model() -> Model<f64> {
    sample(addr!("a"), Normal::new(0.0, 1.0).unwrap()).bind(|a| {
        sample(addr!("b"), Normal::new(0.0, 1.0).unwrap()).bind(move |b| {
            sample(addr!("c"), Normal::new(0.0, 1.0).unwrap()).bind(move |c| {
                sample(addr!("d"), Normal::new(0.0, 1.0).unwrap()).bind(move |d| {
                    observe(addr!("y"), Normal::new(a + b + c + d, 1.0).unwrap(), 1.5)
                        .map(move |_| a)
                })
            })
        })
    })
}

/// Run `steps` adaptive single-site MH transitions from `seed` and return
/// (the site each step moved, or '-' if it rejected; accepts; next raw word).
fn run(seed: u64, steps: usize) -> (String, usize, u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let (_, mut trace) = runtime::handler::run(
        PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model(),
    );
    let mut adaptation = DiminishingAdaptation::new(0.44, 0.7);
    let mut moved = String::new();
    let mut accepts = 0;
    for _ in 0..steps {
        let (_, next) = adaptive_single_site_mh(&mut rng, model, &trace, &mut adaptation);
        let changed: Vec<String> = next
            .choices
            .iter()
            .filter(|(addr, choice)| {
                trace.choices.get(*addr).map(|c| &c.value) != Some(&choice.value)
            })
            .map(|(addr, _)| addr.to_string())
            .collect();
        match changed.as_slice() {
            [] => moved.push('-'),
            [one] => {
                accepts += 1;
                moved.push_str(one);
            }
            many => panic!("single-site MH moved {} sites: {many:?}", many.len()),
        }
        trace = next;
    }
    (moved, accepts, rng.next_u64())
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn seeded_mh_picks_the_same_sites_on_every_target() {
    // Pinned from the native 64-bit run before the fix (fugue 887641d), so
    // they also prove that native output did not move.
    let (moved, accepts, word) = run(2024, 60);
    assert_eq!(
        moved,
        "---a--bd-b--d-b-cb--dda-d--bb---ca-cb-d----accdab-aac-d--d-b"
    );
    assert_eq!(accepts, 31);
    assert_eq!(word, 17_689_723_449_165_144_100);
}
