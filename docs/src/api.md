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

The default `callback_type = nothing` passes `Float64` coordinates when all
stored bounds are exactly representable as finite `Float64` values. When a widened
grid has inexact `Float64` bounds, VEGAS looks for an applicable `BigFloat`
method for the scalar view, vector of views, or matrix in the selected batch
mode. When both coordinate types select the same method, VEGAS keeps `Float64`
if inward rounding removes at most `2eps(Float64)` of each interval's inward
span and the `Float64` spacing is at most `sqrt(eps(Float64))` of a nominal bin
width on each axis. Float64-only callbacks receive inward-clamped values if
each axis contains a finite `Float64` coordinate. They can receive repeated
values on narrow intervals, including the same value for every sample. The grid
and its weights retain their precision.

VEGAS checks which argument types the callable accepts through Julia dispatch.
For a generic wrapper around a Float64-only function, select `callback_type = Float64`
explicitly.

```@example matrix_batch
float64_only(x::AbstractVector{Float64}) = 1.0
offset = BigFloat(2)^53
lower, upper = offset + 1 // 8, offset + 63 // 8
legacy = vegas(x -> float64_only(x), [lower], [upper]; nbins = 4, callback_type = Float64)
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
