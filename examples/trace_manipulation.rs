use clap::Parser;
use fugue::*;
use rand::{rngs::StdRng, SeedableRng};

/// Simple model for trace manipulation demonstration.
fn simple_model(obs: f64) -> Model<(f64, f64)> {
    sample(
        addr!("mu"),
        Normal {
            mu: 0.0,
            sigma: 2.0,
        },
    )
    .bind(move |mu| {
        sample(
            addr!("sigma"),
            LogNormal {
                mu: 0.0,
                sigma: 0.5,
            },
        )
        .bind(move |sigma| {
            observe(addr!("y"), Normal { mu, sigma }, obs).bind(move |_| pure((mu, sigma)))
        })
    })
}

fn print_trace(trace: &Trace, label: &str) {
    println!("\n{} Trace:", label);
    println!("  Choices:");
    for (addr, choice) in &trace.choices {
        match choice.value {
            ChoiceValue::F64(v) => println!("    {}: {:.4} (logp: {:.4})", addr, v, choice.logp),
            ChoiceValue::I64(v) => println!("    {}: {} (logp: {:.4})", addr, v, choice.logp),
            ChoiceValue::U64(v) => println!("    {}: {} (logp: {:.4})", addr, v, choice.logp),
            ChoiceValue::Usize(v) => println!("    {}: {} (logp: {:.4})", addr, v, choice.logp),
            ChoiceValue::Bool(v) => println!("    {}: {} (logp: {:.4})", addr, v, choice.logp),
        }
    }
    println!("  Log prior: {:.4}", trace.log_prior);
    println!("  Log likelihood: {:.4}", trace.log_likelihood);
    println!("  Log factors: {:.4}", trace.log_factors);
    println!("  Total log weight: {:.4}", trace.total_log_weight());
}

fn demonstrate_trace_utilities(seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let obs_value = 1.5;

    println!("=== Trace Manipulation Demo ===");
    println!(
        "Model: mu ~ N(0,2), sigma ~ LogN(0,0.5), y ~ N(mu,sigma) with y = {}",
        obs_value
    );

    // 1. Generate initial trace from prior
    println!("\n1. PRIOR SAMPLING");
    let model = simple_model(obs_value);
    let ((mu1, sigma1), trace1) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        model,
    );
    print_trace(&trace1, "Prior");
    println!("  Sampled values: mu = {:.4}, sigma = {:.4}", mu1, sigma1);

    // 2. Replay the same model with different observation
    println!("\n2. TRACE REPLAY (different observation)");
    let new_obs = 3.0;
    let model2 = simple_model(new_obs);
    let ((mu2, sigma2), trace2) = runtime::handler::run(
        runtime::interpreters::ReplayHandler {
            rng: &mut rng,
            base: trace1.clone(),
            trace: Trace::default(),
        },
        model2,
    );
    print_trace(&trace2, "Replayed");
    println!("  Same random choices, new observation y = {}", new_obs);
    println!("  Values: mu = {:.4}, sigma = {:.4}", mu2, sigma2);
    println!("  Notice: mu, sigma unchanged but likelihood changed!");

    // 3. Score existing trace under original model
    println!("\n3. TRACE SCORING");
    let model3 = simple_model(obs_value);
    let ((mu3, sigma3), trace3) = runtime::handler::run(
        runtime::interpreters::ScoreGivenTrace {
            base: trace1.clone(),
            trace: Trace::default(),
        },
        model3,
    );
    print_trace(&trace3, "Scored");
    println!("  Scoring original trace under original model");
    println!("  Values: mu = {:.4}, sigma = {:.4}", mu3, sigma3);
    println!("  Should match trace1 exactly!");

    // 4. Manual trace manipulation
    println!("\n4. MANUAL TRACE MANIPULATION");
    let mut modified_trace = trace1.clone();

    // Change mu value manually
    if let Some(mu_choice) = modified_trace.choices.get_mut(&addr!("mu")) {
        let old_mu = match mu_choice.value {
            ChoiceValue::F64(v) => v,
            _ => panic!("Expected F64"),
        };
        let new_mu = 2.0;
        mu_choice.value = ChoiceValue::F64(new_mu);
        mu_choice.logp = Normal {
            mu: 0.0,
            sigma: 2.0,
        }
        .log_prob(&new_mu);
        println!("  Modified mu from {:.4} to {:.4}", old_mu, new_mu);
    }

    // Rescore with modified trace
    let model4 = simple_model(obs_value);
    let ((mu4, sigma4), trace4) = runtime::handler::run(
        runtime::interpreters::ScoreGivenTrace {
            base: modified_trace,
            trace: Trace::default(),
        },
        model4,
    );
    print_trace(&trace4, "Modified & Rescored");
    println!("  Values: mu = {:.4}, sigma = {:.4}", mu4, sigma4);

    // 5. Compare log weights
    println!("\n5. LOG WEIGHT COMPARISON");
    println!("  Original trace: {:.4}", trace1.total_log_weight());
    println!(
        "  Replayed (different obs): {:.4}",
        trace2.total_log_weight()
    );
    println!("  Rescored (same): {:.4}", trace3.total_log_weight());
    println!("  Modified trace: {:.4}", trace4.total_log_weight());

    println!("\n=== Summary ===");
    println!("Traces enable:");
    println!("  • Deterministic replay with different observations");
    println!("  • Exact scoring of parameter configurations");
    println!("  • Manual intervention and counterfactual reasoning");
    println!("  • Efficient MCMC transitions (modify + rescore)");
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 12345)]
    seed: u64,
}

fn main() {
    let args = Args::parse();
    demonstrate_trace_utilities(args.seed);
}
