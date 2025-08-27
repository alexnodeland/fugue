#![doc = include_str!("../../docs/api/macros/README.md")]

#[doc = include_str!("../../docs/api/macros/prob.md")]
#[macro_export]
macro_rules! prob {
    // Simple cases first
    ($e:expr) => { $e };

    // let var <- expr; rest
    (let $var:ident <- $expr:expr; $($rest:tt)*) => {
        $expr.bind(move |$var| prob!($($rest)*))
    };

    // let var = expr; rest
    (let $var:ident = $expr:expr; $($rest:tt)*) => {
        { let $var = $expr; prob!($($rest)*) }
    };

    // expr; rest
    ($expr:expr; $($rest:tt)*) => {
        $expr.bind(move |_| prob!($($rest)*))
    };
}

#[doc = include_str!("../../docs/api/macros/plate.md")]
#[macro_export]
macro_rules! plate {
    ($var:ident in $range:expr => $body:expr) => {
        $crate::core::model::traverse_vec($range.collect::<Vec<_>>(), move |$var| $body)
    };
}

#[doc = include_str!("../../docs/api/macros/scoped_addr.md")]
#[macro_export]
macro_rules! scoped_addr {
    ($scope:expr, $name:expr) => {
        $crate::core::address::Address(format!("{}::{}", $scope, $name))
    };
    ($scope:expr, $name:expr, $($indices:expr),+) => {
        $crate::core::address::Address(format!("{}::{}#{}", $scope, $name, format!("{}", format_args!($($indices),+))))
    };
}

#[cfg(test)]
mod tests {
    
    use crate::addr;
    use crate::core::distribution::*;
    use crate::core::model::{observe, sample, pure};
    use crate::runtime::handler::run;
    use crate::runtime::interpreters::PriorHandler;
    use crate::runtime::trace::Trace;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn prob_macro_chains_computations() {
        // Equivalent to: let x <- sample(...); observe(...); pure(x)
        let model = prob!({
            sample(addr!("x"), Normal::new(0.0, 1.0).unwrap())
        ;   let x = pure(())
        ;   observe(addr!("y"), Normal::new(0.0, 1.0).unwrap(), 0.1)
        ;   pure(1)
        });
        let mut rng = StdRng::seed_from_u64(30);
        let (val, trace) = run(PriorHandler { rng: &mut rng, trace: Trace::default() }, model);
        assert_eq!(val, 1);
        assert!(trace.log_prior.is_finite());
        assert!(trace.log_likelihood.is_finite());
    }

    #[test]
    fn plate_macro_traverses_range() {
        let xs = 0..5;
        let model = plate!(i in xs => pure(i));
        let (vals, _t) = run(PriorHandler { rng: &mut StdRng::seed_from_u64(31), trace: Trace::default() }, model);
        assert_eq!(vals, vec![0,1,2,3,4]);
    }

    #[test]
    fn scoped_addr_formats_with_scope_and_indices() {
        let a = scoped_addr!("scope", "name");
        assert_eq!(a.0, "scope::name");
        let b = scoped_addr!("scope", "name", "{}", 3);
        assert_eq!(b.0, "scope::name#3");
    }
}
