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

Sample coordinates and grid edges normally use `Float64`. VEGAS uses `BigFloat`
when bins approach the spacing between adjacent `Float64` values or an integer
bound cannot be represented as `Float64`. Sample coordinates, edges, and widths
then use the active `BigFloat` precision. Integral estimates and standard errors
use `Float64`.

The result contains `integral_estimate`, its estimated standard error
`standard_deviation`, the reduced chi-squared `chi_squared_average`, and the
bin edges and widths in `adaptive_grid` and `grid_spacing`. The standard error
must satisfy both `atol` and `rtol * abs(integral_estimate)` for early stopping.
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
        rtol = 1.0e-4, atol = 1.0e-4, debug = false, batch = false, alpha = 1.5
    )
    _check_vegas_batch(batch)
    maxiter >= 1 || throw(ArgumentError("maxiter must be positive"))
    nbins >= 1 || throw(ArgumentError("nbins must be positive"))
    ncalls >= 2 || throw(ArgumentError("ncalls must be at least two"))
    isfinite(alpha) && alpha >= 0 || throw(ArgumentError("alpha must be finite and nonnegative"))
    grid, widths = _initialize_vegas_grid(lb, ub, nbins, initial_grid)
    return _integrate_vegas(func, grid, widths, maxiter, ncalls, rtol, atol, debug, batch, alpha)
end

function _integrate_vegas(func, grid, widths, maxiter, ncalls, rtol, atol, debug, batch, alpha)
    nbins = size(widths, 1)
    ndim = size(grid, 2)
    points = Matrix{eltype(grid)}(undef, ndim, ncalls)
    unit_points = eltype(points) === Float64 ? points : similar(points, Float64)
    bins = Matrix{Int}(undef, ndim, ncalls)
    values = Vector{Float64}(undef, ncalls)
    mass = similar(widths)
    counts = similar(bins, nbins, ndim)
    scratch = zeros(nbins)
    estimates = Float64[]
    errors = Float64[]
    mean = error = chi_squared = 0.0
    for iter in 1:maxiter
        rand!(unit_points)
        for j in axes(points, 2)
            jacobian = 1.0
            for dim in axes(points, 1)
                y = unit_points[dim, j] * nbins
                bin = floor(Int, y) + 1
                bins[dim, j] = bin
                width = widths[bin, dim]
                points[dim, j] = muladd(y - (bin - 1), width, grid[bin, dim])
                jacobian *= Float64(nbins * width)
            end
            values[j] = jacobian
        end
        _evaluate!(values, func, points, batch)
        estimate, stderr = _iteration_estimate(values)
        push!(estimates, estimate)
        push!(errors, stderr)
        if !iszero(stderr)
            _adapt_grid!(grid, widths, bins, values, mass, counts, scratch, alpha)
        end
        mean, error, chi_squared = _combine_estimates(estimates, errors)
        debug && println("Iteration $iter, abs(sd/Itot) = $(abs(error / mean))")
        if error < atol && error < rtol * abs(mean)
            println("Converged in $(iter * ncalls) evaluations")
            break
        end
    end
    return VEGASResult(mean, error, chi_squared, grid, widths)
end

"""
    sample_from_adaptive_grid(res, ncalls)

Draw `ncalls` independent points from the adaptive grid in `res` as a vector
of coordinate tuples. Each coordinate picks a bin with equal probability
and samples uniformly within it. The coordinates preserve the grid's
floating-point precision. Use `Random.seed!` to seed the default RNG.
The sample count must be nonnegative.
"""
function sample_from_adaptive_grid(res, ncalls::Integer)
    ncalls >= 0 || throw(ArgumentError("ncalls must be nonnegative"))
    edges, widths = res.adaptive_grid, res.grid_spacing
    ndim = size(edges, 2)
    T = _grid_coordinate_type(edges, widths)
    samples = Vector{NTuple{ndim, T}}(undef, ncalls)
    return _sample_grid!(samples, edges, widths, T)
end

function _sample_grid!(samples::Vector{NTuple{N, T}}, edges, widths, ::Type{T}) where {N, T}
    nbins = size(widths, 1)
    for i in eachindex(samples)
        samples[i] = ntuple(Val(N)) do dim
            scaled = nbins * rand()
            bin = floor(Int, scaled) + 1
            fraction = scaled + 1 - bin
            return convert(T, edges[bin, dim] + widths[bin, dim] * fraction)
        end
    end
    return samples
end
