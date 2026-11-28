using Random: rand!
using Distributions: Uniform
import QuasiMonteCarlo

abstract type MonteCarloIntegrationResult end

struct VEGASResult{T1,T2,T3,T4,T5}
	integral_estimate::T1
	standard_deviation::T2
	chi_squared_average::T3
	adaptive_grid::T4
	grid_spacing::T5
end
"""
    vegas(f, lb = [0.0, 0.0], ub = [1.0, 1.0]; kwargs...)

VEGAS is a Monte Carlo algorithm for 
multidimensional integration based on 
adaptive importance sampling. It divides
each dimension into bins and adaptively adjusts
bin widths so the points sampled from the
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
Defaults to 1000. 
- ncalls: Number of function calls per iteration. 
Defaults to 10000.
- maxiter: Maximum number of iterations. 
Defaults to 10.
- rtol: Relative tolerance required. 
Defaults to 1e-4.
- atol: Absolute tolerance required. 
Defaults to 1e-2.
- debug: Prints `abs(sd/I)` every 100 iterations. 
Defaults to false.
- `batch = false`: Call `f(point)` for each sample and return a scalar per point.
- `batch = true`: Call `f(points)` once per iteration with a vector of column views.
- `batch = :matrix`: Call `f(points)` once per iteration with a matrix of size
  `length(lb) × ncalls`. Each column contains one sample. The matrix is reused
  across iterations.

For batched evaluation, return one scalar per sample in the input order. With
`batch = :matrix`, return a vector of length `ncalls`.

Output:
------
A VEGASResult object with the following fields:
- Estimate for the integral 
- Standard deviation
- χ^2 / (numiter - 1): should be less than 1 
otherwise integral estimate should not be trusted. 
- final adapted map

References:
-----------
- Lepage, G. Peter. "A new algorithm for adaptive 
multidimensional integration." Journal of 
Computational Physics 27.2 (1978): 192-203.
- Lepage, G. Peter. "Adaptive multidimensional 
integration: VEGAS enhanced." Journal of 
Computational Physics 439 (2021): 110386.
"""
function vegas(func, lb = [0.0, 0.0], ub = [1.0, 1.0]; kwargs...)
    return _vegas(func, lb, ub, nothing; kwargs...)
end

function _vegas(
        func, lb, ub, initial_grid;
        maxiter = 10, nbins = 1000, ncalls = 10000,
        rtol = 1.0e-4, atol = 1.0e-4, debug = false, batch = false, alpha = 1.5,
    )
    @assert length(lb) == length(ub)
    _check_vegas_batch(batch)

    ndim = length(lb)
    x, delx = _initialize_vegas_grid(lb, ub, nbins, initial_grid)
    integrals = Float64[]
    sigma_squares = Float64[]
    scaled_errors = Float64[]
    Itot = sd = 0.0
    iter = 1

    ymat = zeros(ndim, ncalls)
    xmat = similar(ymat)
    imat = Matrix{Int}(undef, ndim, ncalls)
    Jsf = Vector{Float64}(undef, ncalls)
    while iter <= maxiter
        rand!(ymat)
        for dim in 1:ndim
            lo, hi = minmax(x[1, dim], x[end, dim])
            for j in 1:ncalls
                y = nbins * ymat[dim, j]
                i = floor(Int, y) + 1
                imat[dim, j] = i
                xmat[dim, j] = _bounded_grid_coordinate(x[i, dim], delx[i, dim], y + 1 - i, lo, hi)
            end
        end
        for i in 1:ncalls
            Jsf[i] = prod(nbins * delx[imat[dim, i], dim] for dim in 1:ndim)
        end
        _evaluate!(Jsf, func, xmat, batch)

        scale = maximum(abs, Jsf)
        integral_mc, uncertainty, scaled = _vegas_sample_moments(Jsf, scale)
        if scaled && isempty(scaled_errors)
            append!(scaled_errors, sqrt.(sigma_squares))
        end
        push!(integrals, integral_mc)
        if scaled || !isempty(scaled_errors)
            push!(scaled_errors, scaled ? uncertainty : sqrt(uncertainty))
        else
            push!(sigma_squares, uncertainty)
        end

        if isfinite(scale) && !iszero(scale) && !iszero(alpha) && nbins > 1
            d = calculate_d(Jsf, imat, nbins, alpha, scale)
            xnew, delxnew = update_grid(x, delx, d)
            x .= xnew
            delx .= delxnew
        end

        if isempty(scaled_errors)
            Itot = sum(integrals ./ sigma_squares) / sum(1 ./ sigma_squares)
            sd = 1 / sqrt(sum(1 ./ sigma_squares))
        else
            Itot, sd = _combine_scaled_samples(integrals, scaled_errors)
        end

        if debug
            println("Iteration $iter, abs(sd/Itot) = $(abs(sd / Itot))")
        end
        if abs(sd / Itot) < rtol && abs(sd) < atol
            println("Converged in $(iter * ncalls) evaluations")
            break
        end
        iter += 1
    end

    chi_squared = isempty(scaled_errors) ?
        sum(((integrals .- Itot) .^ 2) ./ sigma_squares) / (iter - 1) :
        _scaled_chi_squared(integrals, scaled_errors, Itot)
    return VEGASResult(Itot, sd, chi_squared, x, delx)
end

_bounded_grid_coordinate(edge, width, fraction, lo, hi) = clamp(edge + width * fraction, lo, hi)

"""
	calculate_d(Jsf, imat, nbins) -> d

Calculate d matrix given J*f product and
locations of the sampled points.
"""
function calculate_d(Jsf, imat, nbins, alpha = 1.5, scale = maximum(abs, Jsf))
    ndim = size(imat, 1)
    d = zeros(nbins, ndim)
    (!isfinite(scale) || iszero(scale)) && return d
    values = 1.0e-100 <= scale <= 1.0e100 ? Jsf : Jsf ./ scale
    ni = length(values) / nbins
    for i in eachindex(values)
        squared = abs2(values[i]) / ni
        for dim in 1:ndim
            d[imat[dim, i], dim] += squared
        end
    end

    weights = similar(d)
    for dim in 1:ndim
        mass = view(d, :, dim)
        total = sum(mass)
        if !isfinite(total) || iszero(total)
            weights[:, dim] .= 0
            continue
        end
        for i in 1:nbins
            left, right = max(1, i - 1), min(nbins, i + 1)
            smoothed = if i == 1
                (7mass[i] + mass[right]) / 8
            elseif i == nbins
                (mass[left] + 7mass[i]) / 8
            else
                (mass[left] + 6mass[i] + mass[right]) / 8
            end
            p = clamp(smoothed / total, 0.0, 1.0)
            weights[i, dim] = iszero(alpha) || isone(p) ? 1.0 : iszero(p) ? 0.0 : ((1 - p) / log(1 / p))^alpha
        end
    end
    return weights
end

"""
	update_grid(x, delx, d) -> newx, newdelx

Function used to update the adaptive grid
given values of the d matrix. The intervals
are adjusted so that the `d` values are equal
in each interval.
"""
function update_grid(x, delx, d)
    nbins, ndims = size(d)
    newx, newdelx = copy(x), copy(delx)
    for dim in 1:ndims
        weights = view(d, :, dim)
        step = sum(weights) / nbins
        isfinite(step) && step > 0 || continue
        bin, mass = 1, weights[1]
        for i in 2:nbins
            while mass < step && bin < nbins
                bin += 1
                mass += weights[bin]
            end
            mass -= step
            left, right = x[bin, dim], x[bin + 1, dim]
            candidate = iszero(weights[bin]) ? right : right - (mass * (right - left)) / weights[bin]
            lo, hi = minmax(newx[i - 1, dim], x[end, dim])
            newx[i, dim] = clamp(candidate, lo, hi)
        end
        for i in 1:nbins
            newdelx[i, dim] = newx[i + 1, dim] - newx[i, dim]
        end
    end
    return newx, newdelx
end

"""
	sample_from_adaptive_grid(res, ncalls)

Sample using a finetuned adaptive grid.  
"""
function sample_from_adaptive_grid(res, ncalls)
    x, delx = res.adaptive_grid, res.grid_spacing
    nbins, ndim = size(delx, 1), size(x, 2)
    ymat = QuasiMonteCarlo.sample(ncalls, zeros(ndim), ones(ndim), Uniform())
    xmat = similar(ymat)
    for dim in 1:ndim
        lo, hi = minmax(x[1, dim], x[end, dim])
        for j in 1:ncalls
            y = nbins * ymat[dim, j]
            bin = floor(Int, y) + 1
            xmat[dim, j] = _bounded_grid_coordinate(x[bin, dim], delx[bin, dim], y + 1 - bin, lo, hi)
        end
    end
    return map(x -> tuple(x...), eachcol(xmat))
end
