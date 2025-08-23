# Fugue Documentation Style Guide

This guide ensures consistency across all documentation in the Fugue repository.

## Module README Structure

All module README files should follow this standardized structure:

````markdown
# [Module Name] Module

## Overview

Brief description of the module's purpose and place in the Fugue architecture.

## Quick Start

Minimal working example showing the most common use case.

## Components

### `filename.rs` - Component Name

**Key Types/Functions:**

- `Type/Function`: Description

**Example:**

```rust
// Focused example
```
````

## Common Patterns

### Pattern Name

When to use and why, with examples.

## Performance Considerations

- **Memory**: Memory usage notes
- **Computation**: Performance characteristics
- **Best Practices**: Optimization tips

## Integration

**Related Modules:**
Cross-references to other modules.

**See Also:**
Links to examples and documentation.

## Extension Points

How developers can extend the module.

## Design Principles

- **Principle**: Explanation

````

## Code Examples

### Formatting Rules
- Always use full `use fugue::*;` in examples for clarity
- Include necessary imports at the top of examples
- Use meaningful variable names (`mu`, `sigma` over `x`, `y`)
- Show error handling with `.unwrap()` or `?` as appropriate
- Comment non-obvious operations inline

### Type Safety Emphasis
When showing distribution usage, emphasize type safety:

```rust
// ✅ Good: Emphasize natural return types
let coin_flip: bool = sample(addr!("coin"), Bernoulli::new(0.5).unwrap());  // Returns bool!
let count: u64 = sample(addr!("events"), Poisson::new(3.0).unwrap());  // Returns u64!

// ❌ Avoid: Generic or unclear types
let result = sample(addr!("x"), some_dist);
````

### Example Categories

1. **Quick Start**: Minimal working example (2-5 lines)
2. **Component Examples**: Focused on specific functionality
3. **Common Patterns**: Real-world usage patterns
4. **Integration Examples**: Show interaction between modules

## Cross-References

### Linking Format

- **Internal modules**: `[core](../core/README.md)`
- **API docs**: `[API docs](https://docs.rs/fugue)`
- **Examples**: `[examples/gaussian_mean.rs](../../examples/gaussian_mean.rs)`
- **Source files**: `[error.rs](../error.rs)`

### Reference Consistency

- Always link to related functionality in other modules
- Mention relevant examples by name
- Reference benchmarks and tests where applicable

## Language and Tone

### Writing Style

- **Clear and concise**: Prefer shorter sentences
- **Active voice**: "The function returns..." not "The value is returned..."
- **Present tense**: "This module provides..." not "This module will provide..."
- **Consistent terminology**: Use the same terms throughout (e.g., "model" not "program")

### Technical Accuracy

- Always show working, compilable code
- Test examples before including them
- Use precise technical language
- Explain "why" not just "what" where helpful

## Function-Level Documentation

### Rustdoc Comments

Use this structure for public functions:

````rust
/// Brief one-line description.
///
/// Longer description explaining the purpose, algorithm, or important details.
/// Include information about error conditions, performance characteristics,
/// or usage patterns when relevant.
///
/// # Arguments
///
/// * `param1` - Description of parameter
/// * `param2` - Description of parameter
///
/// # Returns
///
/// Description of return value and its meaning.
///
/// # Errors
///
/// When this function returns errors and why.
///
/// # Examples
///
/// ```rust
/// use fugue::*;
///
/// // Working example that demonstrates typical usage
/// let result = function_name(arg1, arg2)?;
/// assert_eq!(result, expected_value);
/// ```
///
/// # Performance
///
/// Notes about computational complexity or memory usage (when relevant).
pub fn function_name(param1: Type1, param2: Type2) -> Result<ReturnType, FugueError> {
    // Implementation
}
````

### Documentation Priorities

1. **Public API**: All public functions must have comprehensive docs
2. **Examples**: Every public function should have at least one working example
3. **Error conditions**: Document when and why errors occur
4. **Performance notes**: Include when relevant to user decisions

## Maintenance Guidelines

### Regular Updates

- Update examples when API changes
- Verify links are still valid
- Check code examples compile with current version
- Update cross-references when modules are refactored

### Version Consistency

- Update version numbers in examples
- Ensure compatibility with current Cargo.toml
- Update feature flags if they change
- Maintain backward compatibility notes when needed

## Review Checklist

Before committing documentation changes:

- [ ] Follows the standardized README structure
- [ ] All code examples compile and run
- [ ] Cross-references are accurate and up-to-date
- [ ] Language is clear, concise, and consistent
- [ ] Performance and integration notes are included where relevant
- [ ] Examples demonstrate type safety benefits
- [ ] Related modules and examples are properly linked
