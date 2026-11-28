# Monte Carlo Integration 

![CI](https://github.com/ranjanan/MonteCarloIntegration.jl/workflows/CI/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/ranjanan/MonteCarloIntegration.jl/badge.svg?branch=master)](https://coveralls.io/github/ranjanan/MonteCarloIntegration.jl?branch=master)

This package provides multidimensional integration 
algorithms based on monte carlo methods. The biggest
advantage of using monte carlo methods is that their
convergence rate is **independent of the dimension of
the integral**. 

Currently, this package only provides a routine 
called VEGAS: 

    vegas(f, st, en; kwargs...)

VEGAS is a Monte Carlo algorithm for 
multidimensional integration based on 
adaptive importance sampling. It divides
each dimension into bins and adaptively adjusts
bin widths so points are sampled from the
region where the function has highest magnitude. 

Arguments:
----------
- st: Array of starting values in each dimension. 
Defaults to zeros(2)
- end: Array of ending values in each dimension. 
Defaults to ones(2)

Kwargs:
------
- nbins: Number of bins in each dimension. 
Defaults to 100. 
- ncalls: Number of function calls per iteration. 
Defaults to 1000.
- maxiter: Maximum number of iterations. 
Defaults to 100.
- rtol: Relative tolerance required. 
Defaults to 1e-4.
- atol: Absolute tolerance required. 
Defaults to 1e-2.
- debug: Prints `abs(sd/I)` every 100 iterations. 
Defaults to false.
- `batch = false`: Pass one sample at a time to `f` and return a scalar for each.
- `batch = true`: Pass a vector of column views to `f` once per iteration.
- `batch = :matrix`: Pass an `ndims × ncalls` matrix to `f` once per iteration,
  with one sample per column. Return a vector with one value per column.

Output:
------
- Estimate for the integral 
- Standard deviation
- χ^2 / (numiter - 1): should be less than 1 
otherwise integral estimate should not be trusted. 

References:
-----------
- Lepage, G. Peter. "A new algorithm for adaptive 
multidimensional integration." Journal of 
Computational Physics 27.2 (1978): 192-203.

### Matrix batches and grid reuse

```julia
using MonteCarloIntegration

gaussian_batch(points) = exp.(-vec(sum(abs2, points; dims = 1)))
trained = vegas(gaussian_batch, [0.0, 0.0], [1.0, 1.0]; batch = :matrix)
second_moment = vegas(x -> sum(abs2, x) * exp(-sum(abs2, x)), trained)
```

`vegas(f, trained)` starts a new estimate using a copy of the adaptive grid and
the bounds stored in `trained`.

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

### Roadmap 
- Supporting vector valued functions
- Other integration algorithms
