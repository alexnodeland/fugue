# `CowTrace` struct

Copy-on-write trace for efficient MCMC operations.

Most MCMC operations only modify a small number of choices, so we can share the majority of the trace data between states.
