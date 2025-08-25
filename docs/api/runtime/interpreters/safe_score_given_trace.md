# `SafeScoreGivenTrace` struct

Safe version of ScoreGivenTrace that uses type-safe trace accessors.

This handler computes log-probability of a model execution but gracefully handles missing addresses or type mismatches by returning negative infinity log-weight instead of panicking.
