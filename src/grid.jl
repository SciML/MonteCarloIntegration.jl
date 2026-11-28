function _initialize_vegas_grid(lb, ub, nbins, ::Nothing)
    length(lb) == length(ub) || throw(DimensionMismatch("bounds must have the same length"))
    isempty(lb) && throw(ArgumentError("the integration box needs at least one dimension"))
    wide_coordinates = false
    for dim in eachindex(lb, ub)
        lo, hi = lb[dim], ub[dim]
        a, b = _grid_bound(Float64, lo), _grid_bound(Float64, hi)
        if !isfinite(a) || !isfinite(b)
            isfinite(lo) && isfinite(hi) || throw(ArgumentError("bounds must be finite"))
            wide_coordinates = true
            continue
        end
        wide_coordinates |= _needs_wide_grid_axis(a, b, nbins, lo != hi) ||
            _lossy_float64_bounds(lo, hi, a, b)
    end
    return _uniform_vegas_grid(wide_coordinates ? BigFloat : Float64, lb, ub, nbins)
end

_grid_bound(::Type{T}, bound) where {T} = T(bound)

function _grid_bound(::Type{Float64}, bound::Rational)
    return Float64(BigFloat(bound, RoundNearest; precision = 128), RoundNearest)
end

function _needs_wide_grid_axis(a, b, nbins, nonflat = a != b)
    span = b - a
    spacing = eps(max(abs(a), abs(b)))
    return !isfinite(span) || (nonflat && abs(span / nbins) <= max(2spacing, floatmin(Float64)))
end

_is_binary_exact_bound(bound::Union{Integer, AbstractFloat}) = true
_is_binary_exact_bound(bound::Rational) = ispow2(denominator(bound))
_is_binary_exact_bound(bound) = false

_binary_bound_rounded(original, converted) =
    _is_binary_exact_bound(original) && original != converted

function _lossy_float64_bounds(lo, hi, a, b)
    a == lo && b == hi && return false
    return _binary_bound_rounded(lo, a) || _binary_bound_rounded(hi, b) || !_grid_span_matches(lo, hi, a, b, b - a, 1)
end

_original_rational_bound(original::Rational, converted) = Rational{BigInt}(original)
_original_rational_bound(original::Union{Integer, AbstractFloat}, converted) = Rational{BigInt}(converted)

function _original_rational_bound(original, converted)
    bits = max(256, precision(BigFloat) + 64)
    value = BigFloat(original; precision = bits)
    isfinite(value) || throw(ArgumentError("bounds must be finite"))
    # BigFloat's full decimal rendering uses the stored precision of the value.
    parts = split(string(value), 'e'; limit = 2)
    whole, fraction = split(first(parts), '.'; limit = 2)
    exponent = length(parts) == 2 ? parse(Int, last(parts)) : 0
    power = exponent - length(fraction)
    coefficient = parse(BigInt, string(whole, fraction))
    factor = big(10)^abs(power)
    return power < 0 ? coefficient // factor : (coefficient * factor) // 1
end

_span_within_float64_tolerance(expected, actual) =
    abs(expected - actual) * (Int64(1) << 51) <= max(abs(expected), abs(actual))

function _grid_span_matches(lo, hi, a, b, width, nbins)
    expected = _original_rational_bound(hi, b) - _original_rational_bound(lo, a)
    _span_within_float64_tolerance(expected, Rational{BigInt}(width) * nbins) || return false
    return nbins == 1 || _span_within_float64_tolerance(expected, Rational{BigInt}(width * nbins))
end

function _check_bigfloat_grid_bounds(lo, hi, a, b, nbins)
    reason = if _binary_bound_rounded(lo, a) || _binary_bound_rounded(hi, b)
        "preserve the grid bounds exactly"
    elseif !_grid_span_matches(lo, hi, a, b, (b - a) / nbins, nbins)
        "preserve the signed grid span"
    else
        return nothing
    end
    throw(ArgumentError("active BigFloat precision ($(precision(BigFloat)) bits) cannot $reason; increase it with setprecision"))
end

function _uniform_vegas_grid(::Type{T}, lb, ub, nbins) where {T}
    grid = Matrix{T}(undef, nbins + 1, length(lb))
    widths = Matrix{T}(undef, nbins, length(lb))
    for dim in eachindex(lb, ub)
        lo, hi = lb[dim], ub[dim]
        a, b = _grid_bound(T, lo), _grid_bound(T, hi)
        T === BigFloat && _check_bigfloat_grid_bounds(lo, hi, a, b, nbins)
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
