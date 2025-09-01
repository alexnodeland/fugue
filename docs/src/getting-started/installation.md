# Installation

```admonish info title="Contents"
<!-- toc -->
```

Getting Fugue set up in your Rust project takes just 2 minutes. Let's get you running!

````admonish note
Prerequisites

Fugue requires **Rust 1.70+**. If you don't have Rust installed:

```bash
# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Update to latest stable
rustup update stable
```

````

## Adding Fugue to Your Project

### New Project

```bash
cargo new my_probabilistic_project
cd my_probabilistic_project
```

Add Fugue to your `Cargo.toml`:

```toml
[dependencies]
fugue-ppl = "0.1.0"
rand = "0.8"  # For random number generation
```

## Existing Project

Add Fugue to your existing `Cargo.toml`:

```toml
[dependencies]
fugue-ppl = "0.1.0"
rand = "0.8"
```

Or use `cargo add`:

```bash
cargo add fugue-ppl rand
```

## Verification: "Hello, Probabilistic World!"

Let's verify your installation with a simple example that showcases Fugue's type safety.

Create or replace `src/main.rs`:

```rust,ignore
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    println!("🎲 Hello, Probabilistic World!");

    // Create a simple model: flip a biased coin
    let coin_model = sample(addr!("coin"), Bernoulli::new(0.7).unwrap());

    // Run the model with a seeded RNG for reproducible results
    let mut rng = StdRng::seed_from_u64(42);
    let (is_heads, trace) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        coin_model,
    );

    // Print the result - notice it's a bool, not a float!
    let result = if is_heads { "Heads" } else { "Tails" };
    println!("🪙 Coin flip result: {}", result);
    println!("📊 Log probability: {:.4}", trace.total_log_weight());

    // Demonstrate type safety with different distributions
    let mut rng = StdRng::seed_from_u64(123);

    // Count events - returns u64 directly
    let (event_count, _) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        sample(addr!("events"), Poisson::new(3.5).unwrap()),
    );
    println!("🎯 Event count: {} (type: u64)", event_count);

    // Choose category - returns usize for safe indexing
    let options = vec!["red", "green", "blue"];
    let (category_idx, _) = runtime::handler::run(
        runtime::interpreters::PriorHandler {
            rng: &mut rng,
            trace: Trace::default(),
        },
        sample(addr!("color"), Categorical::uniform(3).unwrap()),
    );
    println!("🎨 Chosen color: {} (safe indexing!)", options[category_idx]);

    println!("✅ Fugue is working correctly!");
}
```

Run it to verify everything works:

```bash
cargo run
```

You should see output like:

```text
🎲 Hello, Probabilistic World!
🪙 Coin flip result: Heads
📊 Log probability: -0.3567
🎯 Event count: 4 (type: u64)
🎨 Chosen color: blue (safe indexing!)
✅ Fugue is working correctly!
```

```admonish tip
Type Safety in Action!

Notice how each distribution returns its natural type:

- `Bernoulli` → `bool` (not `f64`)
- `Poisson` → `u64` (not `f64`)
- `Categorical` → `usize` (not `f64`)

This prevents entire classes of runtime errors!
```

## IDE Setup

### VS Code

Install the **rust-analyzer** extension for the best development experience:

1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "rust-analyzer"
4. Install the official rust-analyzer extension

### Other IDEs

- **IntelliJ/CLion**: Install the Rust plugin
- **Vim/Neovim**: Use coc-rust-analyzer or native LSP
- **Emacs**: Use lsp-mode with rust-analyzer

## Optional: Running Examples

Fugue comes with comprehensive examples to explore:

```bash
# Clone the repository to access examples
git clone https://github.com/your-org/fugue-ppl
cd fugue-ppl

# List available examples
ls examples/

# Run a simple example
cargo run --example gaussian_mean -- --obs 2.5 --seed 42

# Try a more complex one
cargo run --example working_with_distributions
```

## Troubleshooting

### Common Issues

**Build fails with dependency errors:**

```bash
# Make sure you're using Rust 1.70+
rustc --version

# Update your dependencies
cargo update
```

**Examples don't run:**

```bash
# Make sure you're in the project root directory
pwd

# Check example names
ls examples/
```

**IDE doesn't provide completions:**

- Make sure rust-analyzer is installed and running
- Try restarting your IDE after installing dependencies
- Check that your `Cargo.toml` has the correct dependencies

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/alexandernodeland/fugue-ppl/issues)
2. Review the [examples](https://github.com/alexandernodeland/fugue-ppl/tree/main/examples) for working code
3. Read the [API documentation](https://docs.rs/fugue-ppl)

## Next Steps

Installation complete! 🎉

**Ready to build your first probabilistic model?**
→ **[Your First Model](your-first-model.md)**

**Want to explore examples first?**
→ **[Complete Tutorials](../tutorials/README.md)**

---

**Time**: ~2 minutes • **Next**: [Your First Model](your-first-model.md)
