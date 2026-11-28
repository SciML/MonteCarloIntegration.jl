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
include("stratified.jl")

"""
    vegas(f, lb = zeros(2), ub = ones(2); maxiter = 10, nbins = 1000,
          ncalls = 10000, rtol = 1e-4, atol = 1e-4, debug = false, batch = false,
          alpha = 1.5, stratified = false)

Estimate the integral of a real scalar function over the box from `lb` to `ub`
using adaptive importance sampling. Each of up to `maxiter` iterations draws
`ncalls` points. The map has `nbins` intervals per axis. `alpha` controls how
strongly it adapts, with `alpha = 0` holding the grid fixed. With `batch = true`,
`f` receives a vector of `ncalls` coordinate vectors. With `batch = :matrix`, it
receives the reusable `length(lb) × ncalls` matrix. Each batch returns one value per point.

The bounds determine the floating-point type of callback coordinates, grid
edges, and bin widths. If any bounds are floating-point, their promoted type
also applies to exact inputs. BigFloat bounds select the largest precision
recorded in those values. Exact-only bounds can select BigFloat when Float64
cannot represent the requested grid. Non-dyadic rational and irrational bounds
round inward on intervals of nonzero width. If the selected type cannot
preserve the bounds or resolve the requested bins, provide higher-precision
bounds or use fewer bins. Estimates and standard errors use Float64.
For Float16 and Float32 grids, integration and sampling compute signed bin spans
from the stored edges in Float64.

With `stratified = true`, VEGAS+ partitions the transformed unit box into
equal-volume strata, samples at least twice per stratum, and uses the previous
iteration's variation to allocate additional samples. Each stratum contributes
its volume-weighted mean, and the iteration variance sums the variances of these
contributions.

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
        rtol = 1.0e-4, atol = 1.0e-4, debug = false, batch = false, alpha = 1.5, stratified = false,
    )
    _check_vegas_batch(batch)
    maxiter >= 1 || throw(ArgumentError("maxiter must be positive"))
    nbins >= 1 || throw(ArgumentError("nbins must be positive"))
    ncalls >= 2 || throw(ArgumentError("ncalls must be at least two"))
    isfinite(alpha) && alpha >= 0 || throw(ArgumentError("alpha must be finite and nonnegative"))
    if initial_grid === nothing
        T, bits = _vegas_bounds_type(lb, ub, nbins)
        return _with_vegas_precision(bits) do
            grid, widths = _uniform_vegas_grid(T, lb, ub, nbins)
            strata = stratified ? _VegasStrata(size(grid, 2), ncalls) : nothing
            _integrate_vegas(func, grid, widths, maxiter, ncalls, rtol, atol, debug, batch, alpha, strata, T)
        end
    end
    grid, widths = initial_grid
    strata = stratified ? _VegasStrata(size(grid, 2), ncalls) : nothing
    return _integrate_vegas(func, grid, widths, maxiter, ncalls, rtol, atol, debug, batch, alpha, strata, eltype(grid))
end

function _integrate_vegas(
        func, grid, widths, maxiter, ncalls, rtol, atol, debug, batch, alpha, strata, ::Type{T}
    ) where {T}
    nbins = size(widths, 1)
    ndim = size(grid, 2)
    points = Matrix{T}(undef, ndim, ncalls)
    unit_points = T in (Float64, BigFloat) ? points : similar(points, Float64)
    bins = Matrix{Int}(undef, ndim, ncalls)
    values = Vector{Float64}(undef, ncalls)
    jacobians = _vegas_weight_buffer(values, T)
    mass = similar(widths, Float64)
    counts = isnothing(strata) ? similar(bins, nbins, ndim) : nothing
    scratch = zeros(nbins)
    proposed = T === BigFloat ? similar(widths, nbins) : nothing
    exponents = _vegas_jacobian_exponents(grid, widths, ncalls)
    estimates = _IterationMean[]
    errors = _IterationError[]
    history = _ExactHistory()
    pending = maxiter > 1 && strata !== nothing ? _deferred_stratum_history(strata.counts, values) : nothing
    extended = false
    lowest = highest = 0.0
    mean = error = chi_squared = 0.0
    for iter in 1:maxiter
        _sample_vegas_iteration!(points, unit_points, bins, jacobians, exponents, grid, widths, strata)
        _evaluate!(values, func, points, batch, exponents, jacobians)
        estimate, stderr, exact_mean, exact_variance = isnothing(strata) ? _iteration_moments(values) : _stratified_estimate!(strata, values)
        if maxiter > 1
            push!(estimates, estimate)
            push!(errors, stderr)
            lowest, highest = min(lowest, estimate.high), max(highest, estimate.high)
            extended |= exact_mean !== nothing || exact_variance !== nothing || lowest < 0 < highest
            if strata === nothing
                if iter < maxiter || extended
                    sample_mean, sample_variance = _sample_moments(values, estimate, stderr, exact_mean, exact_variance)
                    push!(history.means, sample_mean)
                    push!(history.variances, sample_variance)
                end
            elseif extended
                _record_stratified_moments!(history, pending, strata.counts, values, iszero(stderr), exact_mean, exact_variance, true)
            end
        end
        adapt_grid = !iszero(stderr) && !iszero(alpha) && nbins > 1
        if adapt_grid
            training_weights = isnothing(strata) ? nothing : _stratified_training_weights!(strata)
            _adapt_grid!(grid, widths, bins, values, mass, counts, scratch, alpha, proposed, training_weights)
        end
        if maxiter == 1
            mean = exact_mean === nothing ? _rounded_mean(estimate) : exact_mean
            error = exact_variance === nothing ? _rounded_error(stderr) : exact_variance
        elseif extended
            mean, error, chi_squared = _combine_extended_estimates(history.means, history.variances, history)
        else
            mean, error, chi_squared = _combine_estimates(estimates, _RoundedIterationErrors(errors))
        end
        debug && println("Iteration $iter, abs(sd/Itot) = $(abs(_rounded_error(error) / _rounded_mean(mean)))")
        if _meets_error_tolerances(error, mean, rtol, atol)
            println("Converged in $(iter * ncalls) evaluations")
            break
        end
        if !isnothing(strata) && iter < maxiter
            if !extended
                _record_stratified_moments!(history, pending, strata.counts, values, iszero(stderr), exact_mean, exact_variance, false)
            end
            _redistribute_strata!(strata)
        end
        if exponents === nothing && adapt_grid && iter < maxiter
            exponents = _vegas_jacobian_exponents(grid, widths, ncalls)
        end
    end
    return VEGASResult(_rounded_mean(mean), _rounded_error(error), chi_squared, grid, widths)
end

_vegas_weight_buffer(values, ::Type) = values
_vegas_weight_buffer(values, ::Type{BigFloat}) = similar(values, BigFloat)

function _sample_vegas_iteration!(points, unit_points, bins, weights, exponents, grid, widths, strata)
    return _with_vegas_mapping(grid, widths) do working_widths, bits
        if isnothing(strata)
            rand!(unit_points)
        else
            _sample_strata!(unit_points, strata)
        end
        _map_vegas_points!(points, unit_points, bins, weights, exponents, grid, working_widths, bits)
    end
end

function _vegas_jacobian_exponents(grid, widths, ncalls)
    return _with_vegas_mapping(grid, widths) do working_widths, _
        _requires_scaled_jacobian(working_widths, grid) || return nothing
        return eltype(grid) === BigFloat ? Vector{Int128}(undef, ncalls) : Vector{Int}(undef, ncalls)
    end
end

function _requires_scaled_jacobian(widths, grid = nothing)
    nbins = size(widths, 1)
    lower = upper = 1.0
    for dim in axes(widths, 2)
        smallest, largest = _grid_width_extrema(grid, widths, dim)
        small_factor = _vegas_bin_factor(smallest, nbins)
        large_factor = _vegas_bin_factor(largest, nbins)
        lower *= small_factor
        upper *= large_factor
        min(small_factor, lower) >= floatmin(Float64) && isfinite(max(large_factor, upper)) || return true
    end
    return false
end

_grid_width_extrema(grid, widths, dim) = extrema(abs, view(widths, :, dim))

function _grid_width_extrema(grid::AbstractMatrix{T}, widths, dim) where {T <: _LowPrecisionGridFloat}
    return extrema(i -> abs(_grid_bin_width(grid, widths, i, dim)), axes(widths, 1))
end

_vegas_bin_factor(width, nbins) = Float64(nbins * width)
_vegas_bin_factor(width::BigFloat, nbins) = Float64(nbins * width, RoundNearest)
_vegas_bin_factor(width::Union{Float16, Float32}, nbins) = nbins * Float64(width)

_vegas_jacobian_one(::Type) = 1.0
_vegas_jacobian_one(::Type{BigFloat}) = one(BigFloat)

_vegas_jacobian_factor(width, nbins) = _vegas_bin_factor(width, nbins)
_vegas_jacobian_factor(width::BigFloat, nbins) = nbins * width

_float64_jacobian(value::Float64) = value
_float64_jacobian(value::BigFloat) = Float64(value, RoundNearest)

@inline function _map_vegas_coordinate!(points, unit_points, bins, grid, widths, nbins, bits, dim, j)
    y = unit_points[dim, j] * nbins
    bin = floor(Int, y) + 1
    eltype(unit_points) === BigFloat && (bin = min(nbins, bin))
    bins[dim, j] = bin
    width = _grid_bin_width(grid, widths, bin, dim)
    point = muladd(y - (bin - 1), width, grid[bin, dim])
    points[dim, j] = _round_vegas_coordinate(_bound_grid_point(point, grid, dim), bits)
    return width
end

function _map_vegas_points!(points, unit_points, bins, weights, exponents, grid, widths, bits = nothing)
    nbins = size(widths, 1)
    unit_jacobian = _vegas_jacobian_one(eltype(grid))
    for j in axes(points, 2)
        jacobian, exponent = unit_jacobian, 0
        for dim in axes(points, 1)
            width = _map_vegas_coordinate!(points, unit_points, bins, grid, widths, nbins, bits, dim, j)
            if exponents === nothing
                jacobian *= _vegas_jacobian_factor(width, nbins)
            else
                mantissa, shift = frexp(width)
                jacobian, step = frexp(jacobian * _vegas_jacobian_factor(mantissa, nbins))
                exponent += shift + step
            end
        end
        weights[j] = _float64_jacobian(jacobian)
        exponents === nothing || (exponents[j] = exponent)
    end
    return nothing
end

function _map_vegas_points!(points, unit_points, bins, weights::AbstractVector{BigFloat}, exponents, grid, widths, bits = nothing)
    nbins, ndim = size(widths)
    for j in axes(points, 2), dim in axes(points, 1)
        _map_vegas_coordinate!(points, unit_points, bins, grid, widths, nbins, bits, dim, j)
    end
    # Each binary factor adds at most its significand length to the exact product.
    product_bits = Int(widemul(ndim, precision(BigFloat) + ndigits(nbins; base = 2)))
    return _with_vegas_precision(product_bits) do
        for j in eachindex(weights)
            jacobian, exponent = one(BigFloat), Int128(0)
            for dim in axes(bins, 1)
                width = _grid_bin_width(grid, widths, bins[dim, j], dim)
                if exponents === nothing
                    jacobian *= width * nbins
                else
                    mantissa, shift = frexp(width)
                    jacobian, step = frexp(jacobian * (mantissa * nbins))
                    exponent += Int128(shift) + step
                end
            end
            weights[j] = jacobian
            exponents === nothing || (exponents[j] = exponent)
        end
        nothing
    end
end

"""
    sample_from_adaptive_grid(res, ncalls)

Sample within the bounds stored in the adaptive grid. Coordinates use the
stored coordinate type and BigFloat precision.
"""
function sample_from_adaptive_grid(res, ncalls)
    x, delx = _sampling_grid(res)
    return _with_vegas_precision(_stored_grid_precision(x, delx)) do
        _with_vegas_mapping(x, delx) do working_widths, bits
            _sample_adaptive_grid(x, working_widths, ncalls, _grid_coordinate_type(x, delx), bits)
        end
    end
end

function _sample_adaptive_grid(x, delx, ncalls, ::Type{T}, bits = nothing) where {T}
    nbins = size(delx, 1)
    ndim = size(x, 2)

    # Sample `ncalls` points from this grid
    # Uniform sampling in `y` space
    ymat = T === BigFloat ? rand(BigFloat, ndim, ncalls) : QuasiMonteCarlo.sample(ncalls, zeros(ndim), ones(ndim), Uniform())

    from_y_to_i(y) = min(nbins, floor(Int, nbins * y) + 1)
    delta(y) = y * nbins + 1 - from_y_to_i(y)
    function from_y_to_x(y, dim)
        bin = from_y_to_i(y)
        point = x[bin, dim] + _grid_bin_width(x, delx, bin, dim) * delta(y)
        return _round_vegas_coordinate(_bound_grid_point(point, x, dim), bits)
    end

    xmat = similar(ymat, T)
    for dim in 1:ndim
        xmat[dim, :] .= map(y -> from_y_to_x(y, dim), ymat[dim, :])
    end
    return map(x -> tuple(x...), eachcol(xmat))
end
