# Using VEGAS

## Matrix batches

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

VEGAS uses `BigFloat` coordinates when integration bins approach the local
spacing between representable `Float64` values or an integer bound cannot be
represented as `Float64`. A matrix-batched integrand can accept both types by
preserving the element type of its inputs.

## Reuse the adaptive grid

Pass a previous result to start a fresh estimate on the same bounds and grid.
For example, reuse the Gaussian's grid to compute its second radial moment.

```@example matrix_batch
moment(x) = sum(abs2, x) * exp(-sum(abs2, x))
reused = vegas(moment, trained)
reused.integral_estimate, reused.standard_deviation
```

## Interpreting the uncertainty

`result.standard_deviation` estimates the standard error of the integral. For a
single classic VEGAS iteration, it is the sample standard deviation of the
Jacobian-weighted samples divided by the square root of the sample count.
VEGAS uses inverse-variance weights when all iteration variances are positive.
If any iteration has zero variance, it keeps every estimate with equal weight
and also accounts for the scatter between iterations. A rare peak can escape
every sample. Increase the budget and compare independent runs when the integral
is dominated by a small region.

## Adaptive stratification

Enable VEGAS+ with `stratified = true`. VEGAS+ divides the transformed unit box
into equal-volume strata and allocates more evaluations to strata with larger
estimated variation. Each iteration uses exactly `ncalls` evaluations and
reserves at least two per stratum to estimate its variance. Stratification can
help with narrow ridges and peaks aligned along a diagonal.

```@example matrix_batch
ridge(x) = max(1 - abs(x[1] - x[2]) / 0.025, 0.0)
result = vegas(ridge; stratified = true, nbins = 32, ncalls = 2053)
result.integral_estimate, result.standard_deviation
```

Each iteration uses a volume-weighted sum of the stratum means and adds the
estimated variances of the individual stratum contributions. Each iteration
after the first uses the grid and allocation learned from earlier iterations.

Stratified runs require a positive estimated error for tolerance-based stopping.
A sampled stratum can appear flat when all of its draws miss a feature.
