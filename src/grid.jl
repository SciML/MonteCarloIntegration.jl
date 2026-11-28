function _initialize_vegas_grid(lb, ub, nbins, ::Nothing)
    length(lb) == length(ub) || throw(DimensionMismatch("bounds must have the same length"))
    isempty(lb) && throw(ArgumentError("the integration box needs at least one dimension"))
    wide_coordinates = false
    for dim in eachindex(lb, ub)
        lo, hi = lb[dim], ub[dim]
        a, b = Float64(lo), Float64(hi)
        isfinite(a) && isfinite(b) || throw(ArgumentError("bounds must be finite"))
        wide_coordinates |= _needs_wide_grid_axis(a, b, nbins, lo != hi) ||
            _lossy_float64_bounds(lo, hi, a, b)
    end
    return _uniform_vegas_grid(wide_coordinates ? BigFloat : Float64, lb, ub, nbins)
end

function _needs_wide_grid_axis(a, b, nbins, nonflat = a != b)
    span = b - a
    spacing = eps(max(abs(a), abs(b)))
    return !isfinite(span) || (nonflat && abs(span / nbins) <= max(2spacing, floatmin(Float64)))
end

function _lossy_float64_bounds(lo, hi, a, b)
    a == lo && b == hi && return false
    ((lo isa Union{Integer, BigFloat} && a != lo) || (hi isa Union{Integer, BigFloat} && b != hi)) && return true
    return !isapprox(_original_span(lo, hi), b - a; rtol = 2eps(Float64))
end

function _original_span(lo::Union{Integer, Rational, AbstractFloat}, hi::Union{Integer, Rational, AbstractFloat})
    return Float64(Rational{BigInt}(hi) - Rational{BigInt}(lo))
end

_original_span(lo, hi) = Float64(BigFloat(hi) - BigFloat(lo))

function _uniform_vegas_grid(::Type{T}, lb, ub, nbins) where {T}
    grid = Matrix{T}(undef, nbins + 1, length(lb))
    widths = Matrix{T}(undef, nbins, length(lb))
    for dim in eachindex(lb, ub)
        lo, hi = lb[dim], ub[dim]
        a, b = T(lo), T(hi)
        if T === BigFloat && a == b && lo != hi
            throw(ArgumentError("active BigFloat precision cannot distinguish the grid bounds"))
        end
        _set_uniform_grid_axis!(view(grid, :, dim), view(widths, :, dim), a, b)
    end
    return grid, widths
end

_grid_coordinate_type(edges, widths) =
    eltype(edges) === BigFloat || eltype(widths) === BigFloat ? BigFloat : Float64

_bound_grid_point(point, grid, dim) = point

function _bound_grid_point(point::BigFloat, grid, dim)
    return clamp(point, minmax(grid[1, dim], grid[end, dim])...)
end

_stored_grid_precision(grid) = 0
_stored_grid_precision(grid::AbstractArray{BigFloat}) = maximum(precision, grid; init = 0)

function _check_grid_precision(edges, widths)
    required = max(_stored_grid_precision(edges), _stored_grid_precision(widths))
    iszero(required) && return nothing
    active = precision(BigFloat)
    active >= required || throw(ArgumentError("the grid requires BigFloat precision of at least $required bits (active precision is $active bits)"))
    return nothing
end

function _adapt_grid!(grid, widths, bins, values, mass, counts, scratch, alpha, training_weights = nothing)
    nbins, ndim = size(mass)
    (iszero(alpha) || nbins == 1) && return
    scale = maximum(abs, values)
    iszero(scale) && return
    _bin_means!(mass, counts, bins, values, scale, training_weights)
    for dim in 1:ndim
        _smooth_weights!(scratch, view(mass, :, dim), alpha)
        _rebin!(view(grid, :, dim), view(widths, :, dim), scratch, view(mass, :, dim))
    end
    return nothing
end

function _bin_means!(mass, counts, bins, values, scale, training_weights = nothing)
    fill!(mass, 0)
    isnothing(training_weights) && fill!(counts, 0)
    for j in eachindex(values)
        weight = isnothing(training_weights) ? 1.0 : training_weights[j]
        squared = weight * abs2(values[j] / scale)
        for dim in axes(bins, 1)
            bin = bins[dim, j]
            mass[bin, dim] += squared
            isnothing(training_weights) && (counts[bin, dim] += 1)
        end
    end
    if isnothing(training_weights)
        for i in eachindex(mass, counts)
            mass[i] /= max(1, counts[i])
        end
    else
        expected_occupancy = length(values) / size(mass, 1)
        mass ./= expected_occupancy
    end
    return nothing
end

function _smooth_weights!(weights, histogram, alpha)
    total = sum(histogram)
    iszero(total) && return fill!(weights, 0)
    n = length(histogram)
    for i in 1:n
        smoothed = (histogram[max(i - 1, 1)] + 6histogram[i] + histogram[min(i + 1, n)]) / 8
        p = clamp(smoothed / total, 0.0, 1.0)
        weights[i] = iszero(alpha) || isone(p) ? 1.0 : iszero(p) ? 0.0 : ((p - 1) / log(p))^alpha
    end
    return weights
end

function _rebin!(edges, widths, mass, proposed)
    nbins = length(mass)
    step = sum(mass) / nbins
    iszero(step) && return
    bin = 1
    left = 0.0
    right = mass[1]
    previous = first(edges)
    ascending = last(edges) > previous
    for i in 1:(nbins - 1)
        target = i * step
        while bin < nbins && (iszero(mass[bin]) || right < target)
            bin += 1
            left = right
            right += mass[bin]
        end
        fraction = clamp((target - left) / mass[bin], 0.0, 1.0)
        edge = muladd(fraction, widths[bin], edges[bin])
        # Keep the current map if a narrower bin cannot be represented on this axis.
        (ascending ? previous < edge < last(edges) : previous > edge > last(edges)) || return
        proposed[i] = previous = edge
    end
    edges[2:nbins] .= view(proposed, 1:(nbins - 1))
    for i in eachindex(widths)
        widths[i] = edges[i + 1] - edges[i]
    end
    return nothing
end

"""
    calculate_d(values, bins, nbins, alpha = 1.5)

Calculate smoothed importance weights for each bin and integration axis from
Jacobian-weighted samples `values` and their `dimension × sample` bin indices.
"""
function calculate_d(values, bins, nbins, alpha = 1.5)
    mass = zeros(nbins, size(bins, 1))
    scale = maximum(abs, values)
    iszero(scale) && return mass
    _bin_means!(mass, zeros(Int, size(mass)), bins, values, scale)
    weights = similar(mass)
    for dim in axes(weights, 2)
        _smooth_weights!(view(weights, :, dim), view(mass, :, dim), alpha)
    end
    return weights
end

"""
    update_grid(edges, widths, weights)

Return bin edges and widths with equal integrated importance weight per bin on
each axis. Axes with zero total weight retain their grid.
"""
function update_grid(edges, widths, weights)
    new_edges = copy(edges)
    new_widths = copy(widths)
    proposed = similar(new_widths, size(weights, 1))
    for dim in axes(weights, 2)
        _rebin!(view(new_edges, :, dim), view(new_widths, :, dim), view(weights, :, dim), proposed)
    end
    return new_edges, new_widths
end
