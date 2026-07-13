#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/docs/macros/README.md"))]

/// Probabilistic programming macro, used to define probabilistic programs with do-notation.
///
/// The left-hand side of a monadic bind (`let <pat> <- <model>;`) accepts any
/// irrefutable pattern, not just a bare identifier, so tuples and structs can be
/// destructured directly (FG-61):
///
/// ```rust
/// # use fugue::*;
/// let model = prob! {
///     let (a, b) <- pure((1, 2));   // tuple destructuring bind
///     let mut total <- pure(a + b); // `mut` bindings work too
///     pure(total)
/// };
/// ```
///
/// Example:
/// ```rust
/// # use fugue::*;
///
/// let model = prob! {
///     let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
///     let y <- sample(addr!("y"), Normal::new(x, 1.0).unwrap());
///     pure(y)
/// };
/// ```
#[macro_export]
macro_rules! prob {
    // ---- internal pattern muncher -----------------------------------------
    // Accumulates the tokens of a `let` binding pattern until it reaches the
    // `<-` (monadic bind) or `=` (plain let) that terminates the pattern. This
    // lets the left-hand side be an arbitrary irrefutable pattern: `$p:pat`
    // cannot be followed by `<` in a matcher (Rust's fragment follow-set
    // restriction), so we cannot write `let $p:pat <- ...` directly.

    // Pattern bind: `let <pat> <- <model>; rest`
    (@let [$($pat:tt)+] <- $expr:expr; $($rest:tt)*) => {
        $crate::core::model::ModelExt::bind($expr, move |__prob_bound| {
            let $($pat)+ = __prob_bound;
            $crate::prob!($($rest)*)
        })
    };

    // Plain let: `let <pat> = <value>; rest`
    (@let [$($pat:tt)+] = $expr:expr; $($rest:tt)*) => {
        { let $($pat)+ = $expr; $crate::prob!($($rest)*) }
    };

    // Keep munching pattern tokens one at a time.
    (@let [$($pat:tt)*] $next:tt $($rest:tt)*) => {
        $crate::prob!(@let [$($pat)* $next] $($rest)*)
    };

    // ---- public entry points ----------------------------------------------

    // Any `let` binding routes into the pattern muncher.
    (let $($rest:tt)*) => {
        $crate::prob!(@let [] $($rest)*)
    };

    // expr; rest  (discard the bound value)
    ($expr:expr; $($rest:tt)*) => {
        $crate::core::model::ModelExt::bind($expr, move |_| $crate::prob!($($rest)*))
    };

    // Final expression.
    ($e:expr) => { $e };
}

/// Plate notation for replicating models over ranges.
///
/// Example:
/// ```rust
/// # use fugue::*;
///
/// let model = plate!(i in 0..10 => {
///     sample(addr!("x", i), Normal::new(0.0, 1.0).unwrap())
/// });
/// ```
#[macro_export]
macro_rules! plate {
    ($var:ident in $range:expr => $body:expr) => {
        $crate::core::model::traverse_vec($range.collect::<Vec<_>>(), move |$var| $body)
    };
}

/// Enhanced address macro with scoping support.
///
/// The scope is joined to the name with the reserved `"::"` separator, and any
/// index is joined with the reserved `'#'` separator. Literal `'#'`/`'\'`
/// characters inside the name or index are escaped (see [`Address`](crate::Address))
/// so an indexed scoped address can never alias a differently-written one.
///
/// Example:
/// ```rust
/// # use fugue::*;
///
/// let a = scoped_addr!("scope", "name");
/// let b = scoped_addr!("scope", "name", "{}", 3);
/// ```
#[macro_export]
macro_rules! scoped_addr {
    ($scope:expr, $name:expr) => {
        $crate::core::address::Address::new(format!(
            "{}::{}",
            $scope,
            $crate::core::address::escape_addr_segment(&format!("{}", $name))
        ))
    };
    ($scope:expr, $name:expr, $($indices:expr),+) => {
        $crate::core::address::Address::new(format!(
            "{}::{}#{}",
            $scope,
            $crate::core::address::escape_addr_segment(&format!("{}", $name)),
            $crate::core::address::escape_addr_segment(&format!("{}", format_args!($($indices),+)))
        ))
    };
}

#[cfg(test)]
mod tests {

    use crate::addr;
    use crate::core::distribution::*;
    use crate::core::model::{observe, pure, sample};
    use crate::runtime::handler::run;
    use crate::runtime::interpreters::PriorHandler;
    use crate::runtime::trace::Trace;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn prob_macro_chains_computations() {
        // Equivalent to: let x <- sample(...); observe(...); pure(x)
        let model = prob!({
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
            let _x = pure(());
            observe(addr!("y"), Normal::new(0.0, 1.0).unwrap(), 0.1);
            pure(1)
        });
        let mut rng = StdRng::seed_from_u64(30);
        let (val, trace) = run(
            PriorHandler {
                rng: &mut rng,
                trace: Trace::default(),
            },
            model,
        );
        assert_eq!(val, 1);
        assert!(trace.log_prior.is_finite());
        assert!(trace.log_likelihood.is_finite());
    }

    #[test]
    fn plate_macro_traverses_range() {
        let xs = 0..5;
        let model = plate!(i in xs => pure(i));
        let (vals, _t) = run(
            PriorHandler {
                rng: &mut StdRng::seed_from_u64(31),
                trace: Trace::default(),
            },
            model,
        );
        assert_eq!(vals, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn scoped_addr_formats_with_scope_and_indices() {
        let a = scoped_addr!("scope", "name");
        assert_eq!(a.as_str(), "scope::name");
        let b = scoped_addr!("scope", "name", "{}", 3);
        assert_eq!(b.as_str(), "scope::name#3");
    }

    // Regression for FG-61: `prob!` binds accept irrefutable patterns on the
    // left of `<-`, not just bare identifiers.
    #[test]
    fn prob_macro_binds_tuple_patterns() {
        let model = prob! {
            let (a, b) <- pure((1i32, 2i32));
            let c <- pure(a + b);
            pure(c)
        };
        let (val, _t) = run(
            PriorHandler {
                rng: &mut StdRng::seed_from_u64(40),
                trace: Trace::default(),
            },
            model,
        );
        assert_eq!(val, 3);
    }

    // Regression for FG-61: struct-destructuring patterns and `mut` bindings.
    #[test]
    fn prob_macro_binds_struct_and_mut_patterns() {
        struct Point {
            x: i32,
            y: i32,
        }
        let model = prob! {
            let Point { x, y } <- pure(Point { x: 3, y: 4 });
            let mut acc <- pure(x);
            let sum = {
                acc += y;
                acc
            };
            pure(sum)
        };
        let (val, _t) = run(
            PriorHandler {
                rng: &mut StdRng::seed_from_u64(41),
                trace: Trace::default(),
            },
            model,
        );
        assert_eq!(val, 7);
    }

    // Regression for FG-61: nested tuple pattern with a real sampling bind in
    // between, to confirm the muncher composes with model effects.
    #[test]
    fn prob_macro_tuple_pattern_with_sampling() {
        let model = prob! {
            let x <- sample(addr!("x"), Normal::new(0.0, 1.0).unwrap());
            let (lo, hi) <- pure((x - 1.0, x + 1.0));
            pure(hi - lo)
        };
        let (val, _t) = run(
            PriorHandler {
                rng: &mut StdRng::seed_from_u64(42),
                trace: Trace::default(),
            },
            model,
        );
        assert!((val - 2.0).abs() < 1e-12);
    }
}
