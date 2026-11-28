# API Reference

## Using VEGAS

### Matrix batches

With `batch = :matrix`, `vegas` passes an `ndims × ncalls` matrix to the
integrand once per iteration. Each column is a sample. Return a vector with one
value per column in the same order. With `batch = true`, the integrand receives a
vector of column views.

```@example matrix_batch
using MonteCarloIntegration

gaussian_batch(points) = exp.(-vec(sum(abs2, points; dims = 1)))
trained = vegas(gaussian_batch, [0.0, 0.0], [1.0, 1.0]; batch = :matrix)
trained.integral_estimate
```

VEGAS rounds non-dyadic rational bounds and built-in irrational constants inward
for intervals of nonzero width. It stores the grid in `BigFloat` when bin widths
are subnormal or approach the local spacing between representable `Float64` values,
a bound or span overflows, a binary-exact bound is rounded, or rounding materially
changes the signed span. The active precision must preserve binary-exact bounds
and the signed span to `Float64` accuracy and resolve the bins. If needed, use
`setprecision(BigFloat, bits)` around the integration call. A matrix-batched
integrand can accept both coordinate types by preserving the element type of
its inputs.

With `callback_type = nothing`, VEGAS supplies `Float64` coordinates when the
integrand accepts the selected callback shape and every axis contains a finite
`Float64` value. This preference applies to scalar views, vectors of views, and
matrix batches. When the Float64 path is unavailable, VEGAS supplies `BigFloat`
coordinates if the integrand accepts that shape. Float64 coordinates are clamped
inward. Narrow intervals can produce repeated points, including a single
representable value for every sample. The grid and its weights keep their
precision. Explicit `callback_type = Float64` requires a finite `Float64` value
inside each axis.

The default preserves Float64 dispatch for a generic wrapper around a Float64-only
function on the following inexact interval.

```@example matrix_batch
float64_only(x::AbstractVector{Float64}) = 1.0
lower = BigFloat(2)^20 + BigFloat(2)^-34
upper = lower + 1
legacy = vegas(x -> float64_only(x), [lower], [upper]; nbins = 4)
legacy.integral_estimate
```

Use `callback_type = BigFloat` to receive interior points from a widened grid.
On a `Float64` grid this converts the existing coordinates at the active
`BigFloat` precision, which must preserve the grid bounds.

The function below ranges from 1 to 3 between its bounds and integrates to 4.
The widened callback coordinates let it evaluate at points between those bounds.

```@example matrix_batch
a = 1.0e16
b = nextfloat(a)
narrow = vegas(x -> 1 + (only(x) - a), [a], [b]; nbins = 2, callback_type = BigFloat)
narrow.integral_estimate
```

### Reuse the adaptive grid

Pass a previous result to start a fresh estimate on its stored bounds and grid.
For example, reuse the Gaussian's grid to compute its second radial moment.
When reusing or sampling a `BigFloat` grid, set the active precision to at
least the largest precision stored in the grid's edges or widths. The default
`callback_type` uses the stored bounds and the integrand's supported coordinate
type in the selected batch mode when reusing a grid.

```@example matrix_batch
moment(x) = sum(abs2, x) * exp(-sum(abs2, x))
reused = vegas(moment, trained)
reused.integral_estimate, reused.standard_deviation
```

### Interpreting the uncertainty

`result.standard_deviation` estimates the standard error of the integral. For a
single iteration with independent Jacobian-weighted samples, it is their sample
standard deviation divided by the square root of the sample count. VEGAS uses
inverse-variance weights when all iteration variances are positive. If any iteration
has zero variance, it keeps every estimate with equal weight and also accounts
for the scatter between iterations. Early stopping requires a positive standard
error below both tolerances. While the error stays zero, VEGAS continues drawing
samples up to the iteration limit. A rare peak can escape every sample. Increase
the budget and compare independent runs when the integral is dominated by a small
region.

## Functions and types

```@autodocs
Modules = [MonteCarloIntegration]
```
