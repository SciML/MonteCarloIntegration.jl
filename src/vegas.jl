using Random: rand!

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
          alpha = 1.5)

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
            _integrate_vegas(func, grid, widths, maxiter, ncalls, rtol, atol, debug, batch, alpha, T)
        end
    end
    grid, widths = initial_grid
    return _integrate_vegas(func, grid, widths, maxiter, ncalls, rtol, atol, debug, batch, alpha, eltype(grid))
end

function _integrate_vegas(func, grid, widths, maxiter, ncalls, rtol, atol, debug, batch, alpha, ::Type{T}) where {T}
    nbins = size(widths, 1)
    ndim = size(grid, 2)
    points = Matrix{T}(undef, ndim, ncalls)
    unit_points = T in (Float64, BigFloat) ? points : similar(points, Float64)
    bins = Matrix{Int}(undef, ndim, ncalls)
    values = Vector{Float64}(undef, ncalls)
    mass = similar(widths, Float64)
    counts = similar(bins, nbins, ndim)
    scratch = zeros(nbins)
    proposed = T === BigFloat ? similar(widths, nbins) : nothing
    exponents = _vegas_jacobian_exponents(grid, widths, ncalls)
    estimates = _IterationMean[]
    errors = _IterationError[]
    history = _ExactHistory()
    extended = false
    lowest = highest = 0.0
    mean = error = chi_squared = 0.0
    for iter in 1:maxiter
        _sample_vegas_iteration!(points, unit_points, bins, values, exponents, grid, widths)
        _evaluate!(values, func, points, batch, exponents)
        estimate, stderr, exact_mean, exact_variance = _iteration_moments(values)
        if maxiter > 1
            push!(estimates, estimate)
            push!(errors, stderr)
            lowest, highest = min(lowest, estimate.high), max(highest, estimate.high)
            extended |= exact_mean !== nothing || exact_variance !== nothing || lowest < 0 < highest
            if iter < maxiter || extended
                sample_mean, sample_variance = _sample_moments(values, estimate, stderr, exact_mean, exact_variance)
                push!(history.means, sample_mean)
                push!(history.variances, sample_variance)
            end
        end
        if !iszero(stderr)
            _adapt_grid!(grid, widths, bins, values, mass, counts, scratch, alpha, proposed)
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
        if exponents === nothing && !iszero(stderr) && !iszero(alpha) && nbins > 1 && iter < maxiter
            exponents = _vegas_jacobian_exponents(grid, widths, ncalls)
        end
    end
    return VEGASResult(_rounded_mean(mean), _rounded_error(error), chi_squared, grid, widths)
end

function _sample_vegas_iteration!(points, unit_points, bins, values, exponents, grid, widths)
    return _with_vegas_mapping(grid, widths) do working_widths, bits
        rand!(unit_points)
        _map_vegas_points!(points, unit_points, bins, values, exponents, grid, working_widths, bits)
    end
end

function _vegas_jacobian_exponents(grid, widths, ncalls)
    return _with_vegas_mapping(grid, widths) do working_widths, _
        _requires_scaled_jacobian(working_widths, grid) ? Vector{Int}(undef, ncalls) : nothing
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

function _map_vegas_points!(points, unit_points, bins, weights, exponents, grid, widths, bits = nothing)
    nbins = size(widths, 1)
    unit_jacobian = _vegas_jacobian_one(eltype(grid))
    for j in axes(points, 2)
        jacobian, exponent = unit_jacobian, 0
        for dim in axes(points, 1)
            y = unit_points[dim, j] * nbins
            bin = floor(Int, y) + 1
            eltype(unit_points) === BigFloat && (bin = min(nbins, bin))
            bins[dim, j] = bin
            width = _grid_bin_width(grid, widths, bin, dim)
            point = muladd(y - (bin - 1), width, grid[bin, dim])
            points[dim, j] = _round_vegas_coordinate(_bound_grid_point(point, grid, dim), bits)
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

"""
    sample_from_adaptive_grid(res, ncalls)

Draw `ncalls` independent points inside the adaptive grid and return a vector
of coordinate tuples. Each dimension uses equal bin probabilities. Samples
use the stored coordinate type and BigFloat precision. Use `Random.seed!` to
seed the default RNG. The sample count must be nonnegative.
"""
function sample_from_adaptive_grid(res, ncalls::Integer)
    ncalls >= 0 || throw(ArgumentError("ncalls must be nonnegative"))
    edges, widths = _sampling_grid(res)
    return _with_vegas_precision(_stored_grid_precision(edges, widths)) do
        _with_vegas_mapping(edges, widths) do working_widths, bits
            _draw_grid_samples(ncalls, edges, working_widths, _grid_coordinate_type(edges, widths), bits)
        end
    end
end

function _draw_grid_samples(ncalls, edges, widths, ::Type{T}, bits) where {T}
    ndim = size(edges, 2)
    samples = Vector{NTuple{ndim, T}}(undef, ncalls)
    return _sample_grid!(samples, edges, widths, T, bits)
end

function _sample_grid!(samples::Vector{NTuple{N, T}}, edges, widths, ::Type{T}, bits) where {N, T}
    nbins = size(widths, 1)
    for i in eachindex(samples)
        samples[i] = ntuple(Val(N)) do dim
            if T === BigFloat
                scaled = nbins * rand(BigFloat)
                bin = min(nbins, floor(Int, scaled) + 1)
                fraction = scaled - (bin - 1)
            else
                scaled = nbins * rand()
                bin = floor(Int, scaled) + 1
                fraction = scaled + 1 - bin
            end
            width = _grid_bin_width(edges, widths, bin, dim)
            point = edges[bin, dim] + width * fraction
            return convert(T, _round_vegas_coordinate(_bound_grid_point(point, edges, dim), bits))
        end
    end
    return samples
end
