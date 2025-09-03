# Agent Context: Documentation Directory

## Purpose

The `docs/` directory contains user-facing documentation built with mdbook. This includes tutorials, how-to guides, API references, and examples for the Fugue probabilistic programming library.

## Structure

```text
docs/
├── book.toml           # mdbook configuration
├── src/                # Documentation source files
│   ├── SUMMARY.md      # Table of contents
│   ├── getting-started/ # Tutorials and quickstart
│   ├── how-to/         # Task-oriented guides
│   ├── reference/      # API and technical reference
│   └── examples/       # Comprehensive examples
└── *.html, *.css, *.js # Generated assets and custom styling
```

## Documentation Philosophy

### Target Audiences

1. **New Users**: Clear onboarding and basic concepts
2. **Practitioners**: Task-oriented guides for common workflows
3. **Advanced Users**: Deep technical reference and optimization guides  
4. **Contributors**: Development and extension patterns

### Writing Principles

- **Example-Driven**: Every concept illustrated with working code
- **Progressive Disclosure**: Start simple, build complexity gradually
- **Mathematical Rigor**: Precise notation for probabilistic concepts
- **Practical Focus**: Real-world applications and patterns

## Content Guidelines

### Code Examples

- All examples must be runnable and tested
- Include full imports and setup code
- Show both basic and production-ready patterns
- Demonstrate error handling where relevant

### Mathematical Content

- Use LaTeX notation for equations: `$$P(x|\theta) = ...$$`
- Define notation before first use
- Include intuitive explanations alongside formal definitions
- Link to relevant literature where appropriate

### Cross-References

- Link extensively between related topics
- Reference examples from the main `examples/` directory
- Point to API documentation for detailed reference
- Include "See Also" sections for related concepts

## File Organization Patterns

### Hierarchical Structure

- Start with `README.md` providing module overview
- Use descriptive filenames: `production-deployment.md` not `prod.md`
- Group related topics in subdirectories
- Maintain consistent navigation through `SUMMARY.md`

### Content Types

1. **Tutorials**: Step-by-step learning paths (`getting-started/`)
2. **How-To Guides**: Solution-oriented task documentation (`how-to/`)
3. **Reference**: Technical specifications and API docs (`reference/`)
4. **Examples**: Complete working programs (`examples/`)

## mdbook Configuration

### Essential Settings

- `title`: "Fugue Probabilistic Programming Guide"
- `language`: "en"
- `multilingual`: false
- `src`: "src" (source directory)

### Preprocessors

- **`mdbook-admonish`**: Note/warning callouts
- **`mdbook-mermaid`**: Diagrams and flowcharts
- **`mdbook-linkcheck`**: Validate internal/external links
- **`mdbook-toc`**: Auto-generate table of contents

### Custom Styling

- `mdbook-admonish.css`: Custom admonition styles
- `mermaid.min.js`: Diagram rendering
- `mermaid-init.js`: Mermaid initialization

## Common Documentation Tasks

### Adding New Content

1. Create markdown file in appropriate subdirectory
2. Add entry to `SUMMARY.md` for navigation
3. Include working code examples with proper imports
4. Test examples using `#include` directives from `examples/`
5. Add cross-references to related topics

### Mathematical Documentation

```markdown
## Bayesian Linear Regression

The posterior distribution over parameters follows:

$$P(\beta|X,y) \propto P(y|X,\beta)P(\beta)$$

```admonish note title="Implementation Note"
This uses the `Normal` distribution with precision parameterization.
```

### Including Code Examples

```markdown
```rust,ignore
{{#include ../../../examples/bayesian_regression.rs:model_definition}}
```

This pattern loads specific sections from example files.

### Admonitions

```markdown
```admonish warning title="Numerical Stability"
Log-space computations are critical for numerical stability with extreme values.
```

```admonish tip
Use `safe_ln()` instead of `.ln()` for better error handling.
```

## Documentation Build Process

### Local Development

```bash
# Install tools
make install-dev-tools

# Build and serve locally
make mdbook && mdbook serve docs

# Build static site
make mdbook
```

### Integration with Examples

- Documentation should reference working examples from `examples/`
- Use `#include` directives to embed code sections
- Ensure examples are tested as part of CI pipeline
- Keep documentation and code synchronized

## Quality Standards

### Content Review

- Technical accuracy verified by domain experts
- Code examples tested and maintained
- Mathematical notation consistent throughout
- Links validated and working

### Accessibility

- Clear heading hierarchy for screen readers
- Alternative text for diagrams and images
- Sufficient color contrast in custom styles
- Keyboard navigation support

## Contributing to Documentation

### Testing Documentation Changes

**Always test documentation builds after making changes.** Broken documentation builds can block releases and frustrate users.

```bash
# Test documentation build before committing
make docs-all

# Check for broken links
make mdbook && mdbook test docs

# Full validation pipeline
make all
```

### Workflow

1. Identify gaps in existing documentation
2. Draft content following established patterns
3. Include working examples and test them
4. Review for technical accuracy and clarity
5. Integrate with existing content structure

### Style Guidelines

- Use active voice where possible
- Define technical terms on first use
- Include motivation/context before technical details
- Provide both conceptual and practical perspectives

## Troubleshooting

### Common Build Issues

- **Missing preprocessors**: Run `make install-dev-tools`
- **Broken includes**: Verify paths to example files
- **Math rendering**: Check LaTeX syntax in equations
- **Link validation**: Use `mdbook-linkcheck` for verification

### Content Issues

- **Outdated examples**: Sync with current API
- **Missing context**: Add motivation and use cases
- **Poor navigation**: Review `SUMMARY.md` structure
- **Inconsistent style**: Follow established patterns
