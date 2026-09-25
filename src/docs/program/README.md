# `program` module

*Requires the `program` cargo feature:*
`fugue-ppl = { version = "0.2", features = ["program"] }`.

## Overview

A fugue model is ordinarily Rust closures, so it exists only inside the
binary that compiled it: it cannot be stored, shipped, diffed, or loaded at
runtime. This module makes a model *data*. A [`Program`] is a small
`prob!`-subset language with a serializable AST, two front ends — the text
syntax and JSON — and an interpreter that folds it into fugue's own
combinators (`sample`, `observe`, `factor`, `pure`, `bind`) at load time.
Only model *construction* is interpreted: the handlers, traces, MH/HMC/SMC
kernels and diagnostics that run the result are the real crate.

```text
text  ── Program::parse ─────┐
JSON  ── Program::from_json ─┴──> Program          (plain data: Clone, PartialEq, serde)
      ── compile(&Registry, &Data) ──> CompiledProgram
      ── build() ──> Model<Value>                   (for any handler or inference driver)
```

A host extends the language through the [`Registry`]: its own distributions
(any `Distribution<T>`, including wrapper types that carry site metadata)
and pure functions, callable by name from programs.

## Quick start

```rust
use fugue::program::{Data, Program, Registry};
use fugue::*;
use rand::{rngs::StdRng, SeedableRng};

let program = Program::parse(r#"
    let p <- sample(addr!("p"), Beta(2.0, 2.0));
    for i in 0..flips.len() {
        observe(addr!("flip", i), Bernoulli(p), flips[i]);
    }
    pure(p)
"#).unwrap();

// The same program as JSON, and back: structurally equal.
let json = program.to_json().unwrap();
assert_eq!(Program::from_json(&json).unwrap(), program);

let data = Data::from_json(r#"{"flips": [1, 0, 1, 1, 0, 1, 1, 0, 1, 1]}"#).unwrap();
let compiled = program.compile(&Registry::new(), &data).unwrap();

// A real Model, for any handler or inference driver.
let mut rng = StdRng::seed_from_u64(11);
let draws = adaptive_mcmc_chain(&mut rng, || compiled.build(), 2000, 500);
let mean = draws.iter().filter_map(|(_, t)| t.get_f64(&addr!("p"))).sum::<f64>()
    / draws.len() as f64;
assert!((mean - 9.0 / 14.0).abs() < 0.05); // Beta(2 + 7, 2 + 3)
```

## The text syntax

Statements end with `;`; a program ends with `pure(expr)`.

```text
program := stmt* "pure" "(" expr ")" ";"?
stmt    := "let" "mut"? NAME "<-" "sample" "(" addr "," dist ")" ";"   draw a site
         | "let" "mut"? NAME "=" expr ";"                            bind a variable
         | NAME "=" expr ";"                                         reassign one
         | "observe" "(" addr "," dist "," expr ")" ";"              condition on data
         | "factor" "(" expr ")" ";"                                 add a log-weight
         | "for" NAME "in" expr ".." expr block                      loop
         | "if" expr block ("else" (block | if-statement))?          branch
         | "break" ";"                                               leave the loop
block   := "{" stmt* "}"
addr    := "addr" "!" "(" STRING ("," expr)? ")"
dist    := NAME ("::" "new")? "(" (expr ("," expr)*)? ")" ("." "unwrap" "(" ")")?
expr    := expr BINOP expr | ("-" | "!") expr | postfix
postfix := primary ("[" expr "]" | "." "len" "(" ")")*
primary := NUMBER | "true" | "false" | NAME | NAME "(" exprs ")"
         | "[" exprs "]" | "(" expr ")"
```

Operators bind as in Rust, tightest first: prefix `-` and `!`; `*` `/`;
`+` `-`; the comparisons `==` `!=` `<` `<=` `>` `>=` (which do not chain:
`a < b < c` is a syntax error); `&&`; `||`. Binary operators associate to
the left, and `&&`/`||` short-circuit.

Lexically: `//` starts a comment; `2` is an integer literal and `2.0` or
`1e-3` a float; strings (address names) take Rust's escapes (`\\ \" \' \n
\r \t \0`). Rust spellings paste in unchanged — `Normal::new(0.0, 1.0).unwrap()`
reads as `Normal(0.0, 1.0)` and `let mut x` as `let x` (every variable is
assignable). `let`, `mut`, `observe`, `factor`, `for`, `in`, `if`, `else`,
`break`, `pure`, `true` and `false` are reserved; every other name is an
ASCII identifier.

## Semantics

**Scoping is Rust's.** A `let` (or `let .. <- sample`) binds a new variable
from that point to the end of the enclosing block, shadowing any outer
variable of the same name until then; a loop variable is scoped to the loop
body. An assignment `name = expr;` updates the binding `name` currently
refers to, so an assignment to an outer variable persists across loop
iterations and after the loop, while a `let` inside a body never leaks out.
Assigning a name that nothing binds is a compile error. Data names are bound
in an outer scope, like variables declared before the program.

```rust
# use fugue::program::{Data, Program, Registry, Value};
# use fugue::*;
# use rand::{rngs::StdRng, SeedableRng};
// Sample until the first success; carry a counter out of the loop.
let compiled = Program::parse(r#"
    let tries = 0;
    for i in 0..10 {
        let hit <- sample(addr!("hit", i), Bernoulli(0.3));
        tries = tries + 1;
        if hit { break; }
    }
    pure(tries)
"#).unwrap().compile(&Registry::new(), &Data::new()).unwrap();

let (tries, trace) = runtime::handler::run(
    PriorHandler { rng: &mut StdRng::seed_from_u64(4), trace: Trace::default() },
    compiled.build(),
);
// Only the iterations that ran have sites.
assert_eq!(Some(trace.choices.len()), tries.as_usize());
```

**Loops.** `for i in a..b` evaluates both bounds once, on entry; `i` is an
integer; `break` leaves the innermost loop.

**Values.** Every value is a [`Value`]: `F64`, `Int` (i64), `U64`, `Usize`,
`Bool`, or `Arr` (an array of numbers, from data or a literal `[a, b, c]`).
Sample sites keep their natural types and bind them: `Bernoulli` binds a
`Bool`, `Poisson`/`Binomial` a `U64`, `Categorical` a `Usize`,
`DiscreteUniform` an `Int`, continuous distributions an `F64`. The trace
records the same types (`trace.get_bool`, `get_u64`, `get_usize`, ...).

| construct | rule |
|---|---|
| `+ - *` | exact on integers (overflow is an error, not a wrap); float if either side is a float |
| `/` | always float division (`7 / 2` is `3.5`) |
| comparisons | numbers compare numerically (exactly when both are integers), bools with bools (`false < true`); `NaN` is unordered |
| `&& \|\| !`, `if` | take bools, or numbers as "nonzero"; `NaN` has no truth value |
| numeric positions | a bool reads as `0`/`1`, as in the playground (`2.0 * flag`) |
| `observe` | converts the value to the site's type: `Bernoulli` takes a bool or a number (nonzero is `true`), counts and indices must be non-negative integers |
| `a[i]`, `a.len()` | `i` an integral number in bounds; `.len()` is an `Int` |

A compiled program builds a `Model<Value>`. For a typed result, map it with
the accessors ([`Value::as_bool`], [`as_u64`](Value::as_u64),
[`as_usize`](Value::as_usize), ...) or use
[`build_f64`](CompiledProgram::build_f64), which is the playground's
`Model<f64>`.

## The JSON form

A program serializes to one JSON object: the format version, the statements,
and the return expression. Statements are objects tagged by `"stmt"`,
expressions objects tagged by `"op"`:

| `"stmt"` | fields |
|---|---|
| `"sample"` | `var`, `addr`, `dist` |
| `"let"`, `"assign"` | `var`, `value` |
| `"observe"` | `addr`, `dist`, `value` |
| `"factor"` | `logw` |
| `"for"` | `var`, `start`, `end`, `body` |
| `"if"` | `cond`, `then`, and `else` (omitted when empty) |
| `"break"` | — |

| `"op"` | fields |
|---|---|
| `"f64"`, `"int"`, `"bool"` | `value` |
| `"var"` | `name` |
| `"array"` | `items` |
| `"index"` | `array`, `index` |
| `"len"` | `array` |
| `"neg"`, `"not"` | `arg` |
| `"add"` `"sub"` `"mul"` `"div"` `"eq"` `"ne"` `"lt"` `"le"` `"gt"` `"ge"` `"and"` `"or"` | `lhs`, `rhs` |
| `"call"` | `func`, `args` |

An address is `{"name": .., "index": expr}` (`index` optional), a
distribution call `{"name": .., "args": [..]}`. The coin-flip program above,
written as JSON:

```rust
# use fugue::program::Program;
let json = r#"{
  "fugue_program": 1,
  "body": [
    {"stmt": "sample", "var": "p", "addr": {"name": "p"},
     "dist": {"name": "Beta", "args": [{"op": "f64", "value": 2.0}, {"op": "f64", "value": 2.0}]}},
    {"stmt": "for", "var": "i",
     "start": {"op": "int", "value": 0},
     "end": {"op": "len", "array": {"op": "var", "name": "flips"}},
     "body": [
       {"stmt": "observe",
        "addr": {"name": "flip", "index": {"op": "var", "name": "i"}},
        "dist": {"name": "Bernoulli", "args": [{"op": "var", "name": "p"}]},
        "value": {"op": "index", "array": {"op": "var", "name": "flips"},
                  "index": {"op": "var", "name": "i"}}}
     ]}
  ],
  "ret": {"op": "var", "name": "p"}
}"#;
let text = r#"
    let p <- sample(addr!("p"), Beta(2.0, 2.0));
    for i in 0..flips.len() {
        observe(addr!("flip", i), Bernoulli(p), flips[i]);
    }
    pure(p)
"#;
assert_eq!(Program::from_json(json).unwrap(), Program::parse(text).unwrap());
```

The two front ends are equally expressive: every program either reads can be
written by the other (`Display` on a [`Program`] prints its canonical text).
Reading is strict:

- **Versioned.** `fugue_program` is the format version ([`FORMAT_VERSION`],
  currently 1). It is checked before anything else is decoded, so a program
  from a newer format fails with [`ProgramError::UnsupportedVersion`] rather
  than with whatever node its newer syntax trips over first.
- **No unknown fields.** A misspelt key is an error, not a silently different
  program.
- **Exact numbers.** Float literals round-trip bit for bit (and must be
  finite: JSON has no `NaN`). Names are identifiers, as in the text syntax;
  address names are arbitrary strings.
- **Bounded nesting.** A program may nest at most [`MAX_NESTING`] levels,
  counted as the depth of its JSON (one per expression node, two per block).
  Both front ends and [`Program::compile`] enforce the same limit — the text
  parser as it reads, so a too-deep program is a syntax error with a line —
  so anything one front end accepts the other does too (serde_json refuses
  deeper documents), and hostile input cannot exhaust the parser's or the
  interpreter's stack. Long sums belong in a loop:
  `for i in 0..n { s = s + x[i]; }`.

## The registry

[`Registry::new`] holds the built-in distributions — `Normal`, `Uniform`,
`LogNormal`, `Exponential`, `Beta`, `Gamma`, `InverseGamma`, `StudentT`,
`Cauchy`, `Laplace`, `Weibull`, `ChiSquared`, `Bernoulli`, `Binomial`,
`Poisson`, `Categorical`, `DiscreteUniform`, with fugue's constructors and
parameterizations — and the functions `exp`, `ln`/`log`, `sqrt`, `abs`,
`floor`, `sin`, `cos`, `tanh`, `pow`, `min`, `max` (on `f64`).
`Categorical` takes its probabilities as arguments or as one array of any
length: `Categorical(0.2, 0.8)`, `Categorical(probs)`,
`Categorical([p, 1.0 - p])`.

A host registers its own. A distribution constructor receives the argument
[`Value`]s and returns a [`HostDist`] — any boxed `Distribution<T>` at one of
the five site types, so it can be a wrapper that carries whatever the host's
handler needs to know about the site. A function receives values and returns
one.

```rust
use fugue::program::{Arity, Data, HostDist, Program, Registry, SiteType, Value};
use fugue::*;
use rand::{rngs::StdRng, RngCore, SeedableRng};

/// A host's decision site: the habit's predictive at this context, plus the
/// context itself, for a harness that resolves sites.
#[derive(Clone)]
struct Habit {
    predictive: Categorical,
    last_tool: usize,
    failed: bool,
}

impl Distribution<usize> for Habit {
    fn sample(&self, rng: &mut dyn RngCore) -> usize {
        self.predictive.sample(rng)
    }
    fn log_prob(&self, x: &usize) -> f64 {
        self.predictive.log_prob(x)
    }
    fn clone_box(&self) -> Box<dyn Distribution<usize>> {
        Box::new(self.clone())
    }
}

let mut registry = Registry::new();
registry
    .register_distribution("Habit", SiteType::Usize, Arity::Exact(2), |args| {
        let last_tool = args[0].as_usize().ok_or("Habit: the last tool is an index")?;
        let failed = args[1].as_bool().ok_or("Habit: `failed` is a bool")?;
        let probs = if failed { vec![0.7, 0.2, 0.1] } else { vec![0.1, 0.3, 0.6] };
        let predictive = Categorical::new(probs).map_err(|e| e.to_string())?;
        Ok(HostDist::Usize(Box::new(Habit { predictive, last_tool, failed })))
    })
    .register_function("is_finish", Arity::Exact(1), |args| {
        Ok(Value::Bool(args[0].as_usize() == Some(2)))
    });

let flow = Program::parse(r#"
    let last = 0;
    let failed = false;
    for step in 0..10 {
        let tool <- sample(addr!("decide", step), Habit(last, failed));
        if is_finish(tool) { break; }
        let ok <- sample(addr!("outcome", step), Bernoulli(0.9));
        last = tool;
        failed = !ok;
    }
    pure(last)
"#)
.unwrap()
.compile(&registry, &Data::new())
.unwrap();

let (last, trace) = runtime::handler::run(
    PriorHandler { rng: &mut StdRng::seed_from_u64(1), trace: Trace::default() },
    flow.build(),
);
assert!(last.as_usize().is_some());
assert!(trace.get_usize(&addr!("decide", 0)).is_some());
```

Names resolve at compile time: an unknown distribution or function is a
[`ProgramError::Check`], and so is a wrong argument count ([`Arity`]). The
declared [`SiteType`] gives every site a fixed type even when its
distribution cannot be built (see below). Registered code must be
deterministic in its arguments — replay and scoring rebuild the same site
many times — and must not panic.

## Error policy

**Static errors** are [`ProgramError`]s, returned before any model exists:
syntax errors (with a line number), JSON decoding errors, an unsupported
version, and check errors — unknown variables, distributions or functions,
assignment to an unbound name, wrong arities, `break` outside a loop,
non-finite literals, non-identifier names, nesting beyond the cap — and
malformed data.

**Runtime errors are soft.** Model-building code runs inside `bind`
continuations, where an error cannot propagate as a `Result` (and wasm has
no unwinding), so evaluation is total: an index out of bounds, a type error,
integer overflow, invalid distribution parameters, a registered
constructor's or function's `Err`, an observed value outside its site's type
— each adds `factor(-inf)` ("this region of parameter space is impossible"),
records a warning ([`CompiledProgram::take_warnings`]), and lets the
execution continue:

- a failed `let` or assignment binds `NaN`; a failed `if` condition or `for`
  range skips the statement; a failed `observe` or `factor` contributes only
  the `-inf`; a failed `pure` returns `NaN`;
- a `sample` whose distribution cannot be built still records its site, at
  its address and with its declared type — a standard-normal draw for `f64`
  sites, `false`/`0` for the discrete types — so traces keep a stable shape
  for replay and scoring. (A sample whose *address* cannot be built records
  nothing.)

So a proposal into an invalid region is simply rejected by MH, weighted out
by SMC, and never crashes the host. As for any fugue model, a program that
samples the same address twice in one execution is a modeling error that
the fast handlers report by panicking.

## Addresses

`addr!("name")` builds its address with fugue's `make_name`, and
`addr!("name", i)` with `make_indexed`, formatting the index with the
`Display` of its natural type — so every address is byte-identical to what
`addr!` builds in compiled Rust, escaping included, and a trace from a
program lines up with one from the equivalent Rust model:

```rust
# use fugue::program::{Data, Program, Registry};
# use fugue::*;
# use rand::{rngs::StdRng, SeedableRng};
let compiled = Program::parse(r#"
    let k <- sample(addr!("k"), Poisson(2.0));
    let a <- sample(addr!("x", k), Normal(0.0, 1.0));
    let b <- sample(addr!("y", 0.5), Normal(0.0, 1.0));
    let c <- sample(addr!("a#1"), Normal(0.0, 1.0));
    let d <- sample(addr!("layer::w", 3), Normal(0.0, 1.0));
    pure(0)
"#).unwrap().compile(&Registry::new(), &Data::new()).unwrap();
let (_, t) = runtime::handler::run(
    PriorHandler { rng: &mut StdRng::seed_from_u64(0), trace: Trace::default() },
    compiled.build(),
);
let k = t.get_u64(&addr!("k")).unwrap();
for a in [addr!("x", k), addr!("y", 0.5), addr!("a#1"), scoped_addr!("layer", "w", "{}", 3)] {
    assert!(t.choices.contains_key(&a), "{a}");
}
```

A scope is part of the name (`addr!("layer::w", 3)` is
`scoped_addr!("layer", "w", "{}", 3)` for a scope without `#` or `\`).

## Data

[`Data`] binds named values as variables: [`Data::from_json`] reads a JSON
object of arrays and scalars (or a bare array, bound to `data`, as the
playground sends it); [`Data::new`]`().with(name, value)` builds one in
Rust.

## Where it is used

The docs site's playground and WASM-backed explorables (`crates/fugue-wasm`)
compile their editable models with this module.
