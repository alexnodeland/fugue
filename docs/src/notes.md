# mdBook Documentation Strategy

## Pedagogical Philosophy

**Target Audience**: Primary academic users with industry adoption goals
**Learning Model**: Spiral learning with example-driven tutorials
**Code Strategy**: Examples as source of truth, docs use anchors to pull sections

## Core Principles

1. **Examples-First**: Each tutorial maps to complete example files with anchor tags
2. **Academic Rigor + Practical Implementation**: Mathematical foundations with working code
3. **End-to-End Projects**: Complete modeling journeys from problem to solution
4. **Both Theory & Practice**: Academic depth with industry considerations

## Refined Structure

### Getting Started (15-20 min total)

- Installation
- Your First Model
- Understanding Models
- Running Inference

**Goal**: Quick success and core concepts, NOT comprehensive coverage

### Complete Tutorials (45-60 min each)

#### Foundation Tutorials (Understanding Fugue)

##### Bayesian Coin Flip

- Examples: `bayesian_coin_flip.rs`
- Concepts: Core Bayesian inference, prior/likelihood/posterior

##### Type Safety Features

- Examples: `type_safety.rs`
- Concepts: Fugue's unique type system advantages

##### Trace Manipulation

- Examples: `trace_manipulation.rs`
- Concepts: Runtime system, custom inference, debugging

#### Statistical Modeling (Real Applications)

##### Linear Regression

- Examples: `linear_regression.rs`
- Concepts: Continuous outcomes, uncertainty quantification, diagnostics

##### Classification

- Examples: `classification.rs`
- Concepts: Discrete outcomes, ROC analysis, multi-class

##### Mixture Models

- Examples: `mixture_models.rs`
- Concepts: Latent variables, clustering, component identification

##### Hierarchical Models

- Examples: `hierarchical_models.rs`
- Concepts: Multi-level data, partial pooling, shrinkage

#### Advanced Applications (Research/Production)

##### Time Series & Forecasting

- Examples: `time_series.rs`
- Concepts: Sequential data, state space models, forecasting

##### Model Comparison & Selection

- Examples: `model_selection.rs`
- Concepts: Information criteria, cross-validation, model averaging

##### Advanced Inference

- Examples: `advanced_inference.rs`
- Concepts: Custom algorithms, production optimization, diagnostics

### How-To Guides (10-15 min each)

- Working with Distributions
- Building Complex Models (macros, composition)
- Optimizing Performance (memory, pooling)
- Debugging Models
- Custom Handlers
- Production Deployment

## Tutorial Structure Template

Each tutorial follows this pattern:

````markdown
# Tutorial Name

## The Problem & Data

Real-world context and motivation...

**Data Generation & Exploration**

```rust
{{#include ../../examples/series.rs:data_setup}}
```
````

## Mathematical Foundation

$$mathematical formulation$$
Theory explanation with academic rigor...

## Basic Implementation

```rust
{{#include ../../examples/series_02_basic.rs:basic_model}}
```

## Advanced Techniques

```rust
{{#include ../../examples/series_03_advanced.rs:advanced_features}}
```

## Diagnostics & Validation

```rust
{{#include ../../examples/series_04_diagnostics.rs:convergence_checks}}
```

## Production Extensions

```rust
{{#include ../../examples/series_05_production.rs:production_ready}}
```

## Real-World Considerations

- Performance implications
- Common pitfalls
- Industry best practices
- Further reading

## Exercises

1. Extend the model...
2. Try different priors...
3. Implement validation...

## Real-World Focus Areas

**Academic Elements:**

- Mathematical formulation with LaTeX
- Theoretical foundations and derivations
- Convergence diagnostics and validation
- References to key papers/concepts

**Industry Elements:**

- Data preprocessing and cleaning
- Model validation workflows
- Prediction pipelines
- Performance optimization
- Production deployment patterns
- Error handling and monitoring

## Implementation Notes

### Anchor Strategy:

```rust
// examples/tutorial_data.rs

// ANCHOR: data_generation
let data = generate_synthetic_data();
// ANCHOR_END: data_generation

// ANCHOR: preprocessing
let cleaned_data = preprocess(data);
// ANCHOR_END: preprocessing
````

### Mathematical Content

- Use LaTeX for equations: `$$p(\theta|y) \propto p(y|\theta)p(\theta)$$`
- Include derivations for key results
- Explain intuition behind mathematical choices

### Mermaid Diagrams

- Model structure and dependencies
- Inference algorithm flows
- Learning progression paths
- Architecture overviews

### Admonitions

- ````admonish note` for key concepts
- ````admonish tip` for practical advice
- ````admonish warning` for common pitfalls
- ````admonish math` for mathematical insights
