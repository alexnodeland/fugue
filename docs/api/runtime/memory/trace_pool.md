# `TracePool` struct

Memory pool for reusing trace allocations.

This pool maintains a collection of cleared Trace objects that can be reused to reduce allocation overhead in MCMC and other inference algorithms.
