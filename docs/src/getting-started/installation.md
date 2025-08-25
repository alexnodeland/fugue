# Installation

Getting Fugue set up in your Rust project is straightforward. This guide covers installation, basic setup, and verification that everything works correctly.

## Prerequisites

Fugue requires **Rust 1.70+**. If you don't have Rust installed:

```bash
# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Update to latest stable
rustup update stable
```

## Adding Fugue to Your Project

### New Project

Create a new Rust project and add Fugue:

```bash
cargo new my_probabilistic_project
cd my_probabilistic_project
```

Add Fugue to your `Cargo.toml`:

```toml
[dependencies]
fugue = "0.3.0"
rand = "0.8"  # For random number generation
```

### Existing Project

Add Fugue to your existing `Cargo.toml`:

```toml
[dependencies]
fugue = "0.3.0"
rand = "0.8"
```

Or use cargo add:

```bash
cargo add fugue rand
```

## Verification

Let's verify your installation with a simple "Hello, Probabilistic World!" program.

Create `src/main.rs`:

```rust
use fugue::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    println!("🎲 Hello, Probabilistic World!");
    
    // Create a simple model: flip a fair coin
    let coin_model = sample(addr!("coin"), Bernoulli::new(0.5).unwrap());
    
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
    
    println!("✅ Fugue is working correctly!");
}
```

Run it to verify everything works:

```bash
cargo run
```

You should see output like:
```
🎲 Hello, Probabilistic World!
🪙 Coin flip result: Heads
📊 Log probability: -0.6931
✅ Fugue is working correctly!
```

## Development Dependencies

For development and examples, you might want additional dependencies:

```toml
[dependencies]
fugue = "0.3.0"
rand = "0.8"

[dev-dependencies]
clap = { version = "4", features = ["derive"] }  # For CLI examples
proptest = "1.0"  # For property-based testing
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

## Running Examples

Fugue comes with several examples to explore:

```bash
# List available examples
ls examples/

# Run a simple example
cargo run --example gaussian_mean -- --obs 2.5 --seed 42

# Run with different parameters
cargo run --example gaussian_mixture -- --seed 123
```

## Troubleshooting

### Common Issues

**Build fails with dependency errors:**
- Make sure you're using Rust 1.70+: `rustc --version`
- Update your dependencies: `cargo update`

**Examples don't run:**
- Make sure you're in the project root directory
- Check example names: `ls examples/`

**IDE doesn't provide completions:**
- Make sure rust-analyzer is installed and running
- Try restarting your IDE after installing dependencies

### Getting Help

If you encounter issues:
1. Check the [GitHub Issues](https://github.com/alexandernodeland/fugue/issues)
2. Review the [examples](https://github.com/alexandernodeland/fugue/tree/main/examples) for working code
3. Read the [API documentation](https://docs.rs/fugue)

## Next Steps

Now that Fugue is installed and working:

1. **[Build Your First Model](your-first-model.md)** - Create and run a simple probabilistic model
2. **[Explore Examples](https://github.com/alexandernodeland/fugue/tree/main/examples)** - Look at complete working examples
3. **[Understanding Models](understanding-models.md)** - Learn the core concepts

---

**Ready for your first model?** → **[Your First Model](your-first-model.md)**