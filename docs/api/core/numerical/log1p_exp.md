# `log1p_exp` function

Compute log(1 + exp(x)) stably to avoid overflow.

This function is crucial for logistic regression and other applications where we need to compute log of sigmoid-like functions.

## Arguments

- `x` - Input value

## Returns

log(1 + exp(x)) computed stably
