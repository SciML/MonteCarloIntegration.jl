function _initialize_vegas_grid(lb, ub, nbins, ::Nothing)
    length(lb) == length(ub) || throw(DimensionMismatch("bounds must have the same length"))
    isempty(lb) && throw(ArgumentError("the integration box needs at least one dimension"))
    wide_coordinates = false
    for dim in eachindex(lb, ub)
        lo, hi = lb[dim], ub[dim]
        a, b = _grid_bounds(Float64, lo, hi)
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

function _grid_bound(::Type{BigFloat}, bound::Rational)
    bits = precision(BigFloat)
    value = BigFloat(bound, RoundNearest; precision = bits + 16)
    return BigFloat(value, RoundNearest; precision = bits)
end

_grid_bound(T::Type, bound, rounding) = _grid_bound(T, bound)
_grid_bound(T::Type, bound::Irrational, rounding) = T(bound, rounding)

function _grid_bound(T::Type, bound::Rational, rounding)
    value = _grid_bound(T, bound)
    (ispow2(denominator(bound)) || !isfinite(value)) && return value
    # Guard bits keep the approximation within one step of the inward endpoint.
    exact_value = Rational{BigInt}(value)
    rounding === RoundUp && exact_value < bound && return nextfloat(value)
    rounding === RoundDown && exact_value > bound && return prevfloat(value)
    return value
end

function _grid_bounds(T, lo, hi)
    if lo < hi
        return _grid_bound(T, lo, RoundUp), _grid_bound(T, hi, RoundDown)
    elseif hi < lo
        return _grid_bound(T, lo, RoundDown), _grid_bound(T, hi, RoundUp)
    end
    return _grid_bound(T, lo), _grid_bound(T, hi)
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
    elseif (lo < hi) != (a < b) || (hi < lo) != (b < a) || !_grid_span_matches(lo, hi, a, b, (b - a) / nbins, nbins)
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
        a, b = _grid_bounds(T, lo, hi)
        T === BigFloat && _check_bigfloat_grid_bounds(lo, hi, a, b, nbins)
        _set_uniform_grid_axis!(view(grid, :, dim), view(widths, :, dim), a, b)
    end
    return grid, widths
end

_grid_coordinate_type(edges, widths) =
    eltype(edges) === BigFloat || eltype(widths) === BigFloat ? BigFloat : Float64

function _bound_grid_point(point, grid, dim)
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

function _adapt_grid!(edges, widths, weights, proposed)
    for dim in axes(weights, 2)
        _rebin!(view(edges, :, dim), view(widths, :, dim), view(weights, :, dim), proposed)
    end
    return nothing
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
    calculate_d(Jsf, imat, nbins, alpha = 1.5)

Calculate smoothed importance weights for each bin and integration axis from
Jacobian-weighted samples and their bin indices.
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
    update_grid(edges, widths, weights)

Return bin edges and widths with equal integrated importance weight per bin on
each axis. Axes with zero total weight retain their grid.
"""
function update_grid(edges, widths, weights)
    new_edges = copy(edges)
    new_widths = copy(widths)
    proposed = similar(new_widths, size(weights, 1))
    _adapt_grid!(new_edges, new_widths, weights, proposed)
    return new_edges, new_widths
end
