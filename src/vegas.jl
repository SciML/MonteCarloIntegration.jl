using Random: rand!
using Distributions: Uniform
import QuasiMonteCarlo

struct VEGASResult{T1, T2, T3, T4, T5}
    integral_estimate::T1
    standard_deviation::T2
    chi_squared_average::T3
    adaptive_grid::T4
    grid_spacing::T5
end

include("estimates.jl")
include("grid.jl")

"""
    vegas(f, lb = zeros(2), ub = ones(2); maxiter = 10, nbins = 1000,
          ncalls = 10000, rtol = 1e-4, atol = 1e-4, debug = false, batch = false,
          alpha = 1.5, callback_type = nothing)

Estimate the integral of a real scalar function over the box from `lb` to `ub`
using adaptive importance sampling. Each of up to `maxiter` iterations draws
`ncalls` points. The map has `nbins` intervals per axis. `alpha` controls how
strongly it adapts, with `alpha = 0` holding the grid fixed. With `batch = true`,
`f` receives a vector of `ncalls` coordinate vectors. With `batch = :matrix`, it
receives the reusable `length(lb) × ncalls` matrix. Each batch returns one value per point.

Grid coordinates normally use `Float64`. Non-dyadic rational
bounds and built-in irrational constants are rounded inward for intervals of
nonzero width. VEGAS uses `BigFloat` when bin widths are subnormal or approach
the spacing between adjacent `Float64` values, a bound or span overflows, a
binary-exact bound is rounded, or rounding materially changes the signed span.
The active `BigFloat` precision must preserve binary-exact bounds and the signed
span to `Float64` accuracy and resolve the bins. Increase it around the call
with `setprecision(BigFloat, bits)` if needed. Integral estimates and standard
errors use `Float64`.

With `callback_type = nothing`, VEGAS uses `Float64` when the integrand accepts
the selected argument shape and each axis contains a finite `Float64` value.
Otherwise it uses `BigFloat` when the integrand accepts that shape. This rule
applies to a scalar coordinate view, a vector of views, and a matrix batch.
Float64 coordinates are clamped inward, while grid edges and weights retain their
precision. Narrow intervals can produce repeated coordinates, including a single
representable value. Use `callback_type = BigFloat` to receive interior coordinates
from a widened grid. On a `Float64` grid, this converts existing coordinates at
the active `BigFloat` precision, which must preserve the bounds. With default
`Float64` coordinates on widened grids, VEGAS warns once per call if coordinate
spacing exceeds `sqrt(eps(Float64))` of a nominal bin width or inward rounding
removes more than that fraction of an axis's span. The standard error estimates
sampling uncertainty and excludes coordinate-rounding bias. An explicit
`callback_type = Float64` selects those coordinates without the warning and
requires a finite `Float64` value inside every axis.

The result contains `integral_estimate`, its estimated standard error
`standard_deviation`, the reduced chi-squared `chi_squared_average`, and the
bin edges and widths in `adaptive_grid` and `grid_spacing`. Early stopping
requires a positive standard error below both `atol` and
`rtol * abs(integral_estimate)`. VEGAS keeps sampling while the error stays zero
and the budget allows.
Use `debug = true` to print the relative standard error each iteration.

Iterations use inverse-variance weights when every sample variance is positive.
If a batch has zero sample variance, all iteration estimates receive equal weight
and the reported error is the larger of the propagated within-iteration error and
the standard error across iterations. Reduced chi-squared uses the positive-variance
iterations and is zero with fewer than two. A sample that misses a narrow peak can have
zero estimated error.

See G. P. Lepage, *Adaptive multidimensional integration: VEGAS enhanced*,
[Journal of Computational Physics 439 (2021) 110386](https://doi.org/10.1016/j.jcp.2021.110386).
"""
function vegas(func, lb = [0.0, 0.0], ub = [1.0, 1.0]; kwargs...)
    return _vegas(func, lb, ub, nothing; kwargs...)
end

function _vegas(
        func, lb, ub, initial_grid; maxiter = 10, nbins = 1000, ncalls = 10000,
        rtol = 1.0e-4, atol = 1.0e-4, debug = false, batch = false, alpha = 1.5,
        callback_type = nothing
    )
    _check_vegas_batch(batch)
    maxiter >= 1 || throw(ArgumentError("maxiter must be positive"))
    nbins >= 1 || throw(ArgumentError("nbins must be positive"))
    ncalls >= 2 || throw(ArgumentError("ncalls must be at least two"))
    isfinite(alpha) && alpha >= 0 || throw(ArgumentError("alpha must be finite and nonnegative"))
    grid, widths = _initialize_vegas_grid(lb, ub, nbins, initial_grid)
    coordinates, callback_bounds = _vegas_callback_coordinates(func, grid, batch, callback_type)
    return _integrate_vegas(func, grid, widths, maxiter, ncalls, rtol, atol, debug, batch, alpha, coordinates, callback_bounds)
end

function _integrate_vegas(func, grid, widths, maxiter, ncalls, rtol, atol, debug, batch, alpha, ::Type{T}, callback_bounds) where {T}
    nbins = size(widths, 1)
    ndim = size(grid, 2)
    points = Matrix{T}(undef, ndim, ncalls)
    unit_points = T === Float64 ? points : similar(points, Float64)
    bins = Matrix{Int}(undef, ndim, ncalls)
    values = Vector{Float64}(undef, ncalls)
    proposed = similar(widths, nbins)
    exponents = _requires_scaled_jacobian(widths) ? Vector{Int}(undef, ncalls) : nothing
    estimates = _IterationMean[]
    errors = _IterationError[]
    exact_means = exact_variances = nothing
    mean = error = chi_squared = 0.0
    for iter in 1:maxiter
        rand!(unit_points)
        _map_vegas_points!(points, unit_points, bins, values, exponents, grid, widths, callback_bounds)
        _evaluate!(values, func, points, batch, exponents)
        estimate, stderr, exact_mean, exact_variance = _iteration_moments(values)
        exact_means = _record_iteration_mean!(estimates, exact_means, estimate, exact_mean)
        exact_variances = _record_iteration_variance!(errors, exact_variances, stderr, exact_variance)
        if !iszero(stderr) && !iszero(alpha) && nbins > 1
            weights = calculate_d(values, bins, nbins, alpha)
            _adapt_grid!(grid, widths, weights, proposed)
        end
        mean, error, chi_squared = _combine_estimates(estimates, errors, exact_means, exact_variances)
        debug && println("Iteration $iter, abs(sd/Itot) = $(abs(_rounded_error(error) / _rounded_mean(mean)))")
        if _meets_error_tolerances(error, mean, rtol, atol)
            println("Converged in $(iter * ncalls) evaluations")
            break
        end
        if exponents === nothing && !iszero(stderr) && !iszero(alpha) && nbins > 1 && iter < maxiter
            if _requires_scaled_jacobian(widths)
                exponents = Vector{Int}(undef, ncalls)
            end
        end
    end
    return VEGASResult(_rounded_mean(mean), _rounded_error(error), chi_squared, grid, widths)
end

function _requires_scaled_jacobian(widths)
    nbins = size(widths, 1)
    lower = upper = 1.0
    for axis in eachcol(widths)
        smallest, largest = extrema(abs, axis)
        small_factor = Float64(nbins * smallest)
        large_factor = Float64(nbins * largest)
        lower *= small_factor
        upper *= large_factor
        min(small_factor, lower) >= floatmin(Float64) && isfinite(max(large_factor, upper)) || return true
    end
    return false
end

function _map_vegas_points!(points, unit_points, bins, weights, exponents, grid, widths, callback_bounds = nothing)
    nbins = size(widths, 1)
    for j in axes(points, 2)
        jacobian, exponent = 1.0, 0
        for dim in axes(points, 1)
            y = unit_points[dim, j] * nbins
            bin = floor(Int, y) + 1
            bins[dim, j] = bin
            width = widths[bin, dim]
            point = muladd(y - (bin - 1), width, grid[bin, dim])
            points[dim, j] = _callback_grid_point(point, grid, dim, callback_bounds)
            if exponents === nothing
                jacobian *= Float64(nbins * width)
            else
                mantissa, shift = frexp(width)
                jacobian, step = frexp(jacobian * (Float64(mantissa) * nbins))
                exponent += shift + step
            end
        end
        weights[j] = jacobian
        exponents === nothing || (exponents[j] = exponent)
    end
    return nothing
end

"""
    sample_from_adaptive_grid(res, ncalls)

Sample within the bounds stored in the adaptive grid. Sample coordinates
preserve the grid's floating-point precision. Uniform grids with bins too narrow for `Float64`
are reconstructed at the active `BigFloat` precision. For a `BigFloat` grid,
the active precision must be at least the largest precision stored in its
edges or widths.
"""
function sample_from_adaptive_grid(res, ncalls)

    x, delx = _sampling_grid(res)

    nbins = size(delx, 1)
    ndim = size(x, 2)

    # Sample `ncalls` points from this grid
    # Uniform sampling in `y` space
    ymat = QuasiMonteCarlo.sample(ncalls, zeros(ndim), ones(ndim), Uniform())

    from_y_to_i(y) = floor(Int, nbins * y) + 1
    delta(y) = y * nbins + 1 - from_y_to_i(y)
    from_y_to_x(y, dim) = _bound_grid_point(x[from_y_to_i(y), dim] + delx[from_y_to_i(y), dim] * delta(y), x, dim)

    xmat = similar(ymat, _grid_coordinate_type(x, delx))
    for dim in 1:ndim
        xmat[dim, :] .= map(y -> from_y_to_x(y, dim), ymat[dim, :])
    end
    return map(x -> tuple(x...), eachcol(xmat))
end
