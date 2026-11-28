# MonteCarloIntegration.jl

[![Tests](https://github.com/SciML/MonteCarloIntegration.jl/actions/workflows/Tests.yml/badge.svg?branch=master)](https://github.com/SciML/MonteCarloIntegration.jl/actions/workflows/Tests.yml)

`vegas` estimates a scalar integral over a multidimensional box and reports its
Monte Carlo standard error. It learns a grid on each axis to concentrate samples
where the integrand has large magnitude.

```julia
using MonteCarloIntegration

result = vegas(x -> exp(-sum(abs2, x)), zeros(2), ones(2))
result.integral_estimate, result.standard_deviation
```

By default, VEGAS uses at most 10 iterations, 10,000 samples per iteration, and
1,000 bins per axis. `rtol` and `atol` both default to `1e-4`. A reduced
chi-squared near one is consistent with iteration-to-iteration scatter of the
size predicted by the uncertainty estimates. Inspect `?vegas` for stopping
criteria and the fields of the result.

## Matrix batches and grid reuse

For a model that processes many points at once, use `batch = :matrix`. Each
column is a sample, and the integrand returns one value per column. The
`batch = true` interface supplies a vector of column views.

```julia
gaussian_batch(points) = exp.(-vec(sum(abs2, points; dims = 1)))
trained = vegas(gaussian_batch, zeros(2), ones(2); batch = :matrix)
second_moment = vegas(x -> sum(abs2, x) * exp(-sum(abs2, x)), trained)
```

`vegas(f, trained)` starts a new estimate using a copy of the adaptive grid and
the bounds stored in `trained`. See the [package manual](https://github.com/SciML/MonteCarloIntegration.jl/tree/master/docs/src)
for more examples.

The bounds determine the type of callback coordinates, grid matrices, and
samples from the adaptive grid. Floating bounds follow Julia's promotion
rules. When exact and floating bounds mix, the floating bounds select the
type and the exact bounds must fit it. BigFloat bounds select the greatest
precision recorded in their values, and a grid retains that precision for
reuse. Exact-only bounds can select BigFloat when Float64 lacks resolution.
The estimate and standard error use Float64.

VEGAS rounds non-dyadic rational and built-in irrational bounds inward on
nonzero intervals. Provide higher-precision bounds or use fewer bins when the
selected type cannot hold the requested bin geometry.

## Reference

G. Peter Lepage describes the grid adaptation, uncertainty estimates, and
stratified sampling in [*Adaptive multidimensional integration: VEGAS enhanced*](https://arxiv.org/abs/2009.05112).

## Roadmap

- Vector-valued integrands
- Additional integration algorithms
