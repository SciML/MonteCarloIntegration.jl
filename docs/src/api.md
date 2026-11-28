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

The bounds select the floating-point type for callback coordinates,
`adaptive_grid`, `grid_spacing`, and samples drawn from the result. Floating
bounds follow Julia's promotion rules. When floating and exact bounds mix,
the floating bounds select the type and the exact bounds must fit its precision.
For example, Float32 paired with an integer uses Float32, and Float32 paired with
Float64 uses Float64. Exact-only inputs normally use Float64 and can select
BigFloat when the requested grid needs more precision.

If the bounds include BigFloat values, VEGAS uses the largest precision stored
in those arguments for callback coordinates and the grid, even when it differs
from the ambient precision. The result estimate and estimated standard error
use Float64.

BigFloat grid construction, adaptation, warm-start validation, and sampling
round coordinates to nearest. When the requested precision is below 64 bits,
unit sampling and bin calculations use extra precision. Callback coordinates
and stored grid values retain the requested precision. Integrand evaluation
uses the caller's BigFloat rounding mode. BigFloat grids preserve binary
Jacobian factors and their products through weighting. BigInt,
Rational{BigInt}, and BigFloat callback values round to Float64 after weighting.
Direct grid sampling promotes the coordinate type from both grid matrices.
On Julia versions with process-wide BigFloat precision or rounding, serialize
VEGAS integration and grid sampling with other BigFloat work that depends on or
changes those settings.

```@example matrix_batch
start = BigFloat(2)^53
lower = BigFloat(start; precision = 128)
upper = BigFloat(start + 2; precision = 128)
narrow = vegas(x -> 1 + (only(x) - lower), [lower], [upper]; nbins = 2)
narrow.integral_estimate, precision(first(narrow.adaptive_grid))
```

Non-dyadic rational bounds and built-in irrational constants round inward on
intervals of nonzero width. A type can represent only finitely spaced values.
If the promoted type cannot preserve the supplied bounds or hold the requested
bin widths and distinct edges, provide higher-precision bounds or reduce the
number of bins. A narrow bin can have representable edges and still have few
representable interior coordinates. The reported standard error measures
variation in the sampled integrand values.

On boxes with extreme axis lengths, VEGAS scales its Jacobian to avoid
intermediate overflow or underflow. This applies to scalar and batched
evaluations in both sampling modes.

### Reuse the adaptive grid

Pass a previous result to start a fresh estimate on its stored bounds and grid.
The stored grid also determines the point type and BigFloat precision.

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

### Adaptive stratification

Enable VEGAS+ with `stratified = true`. VEGAS+ divides the transformed unit box
into equal-volume strata and allocates more evaluations to strata with larger
estimated variation. Each iteration uses exactly `ncalls` evaluations and
reserves at least two per stratum to estimate its variance. Stratum errors
retain their scale for allocation and iteration weighting. Stratification can
help with narrow ridges and peaks aligned along a diagonal.

```@example matrix_batch
ridge(x) = max(1 - abs(x[1] - x[2]) / 0.025, 0.0)
result = vegas(ridge; stratified = true, nbins = 32, ncalls = 2053)
result.integral_estimate, result.standard_deviation
```

Each iteration uses a volume-weighted sum of the stratum means and adds the
estimated variances of the individual stratum contributions. Each iteration
after the first uses the grid and allocation learned from earlier iterations.

## Functions and types

```@autodocs
Modules = [MonteCarloIntegration]
```
