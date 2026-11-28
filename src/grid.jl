function _vegas_bounds_type(lb, ub, nbins)
    length(lb) == length(ub) || throw(DimensionMismatch("bounds must have the same length"))
    isempty(lb) && throw(ArgumentError("the integration box needs at least one dimension"))
    input = Union{}
    bits = 0
    for bounds in (lb, ub), bound in bounds
        bound isa Real && isfinite(bound) || throw(ArgumentError("bounds must be finite real numbers"))
        bound isa AbstractFloat && (input = promote_type(input, typeof(bound)))
        bound isa BigFloat && (bits = max(bits, precision(bound)))
    end
    T = input === Union{} ? Float64 : input
    T in (Float16, Float32, Float64, BigFloat) ||
        throw(ArgumentError("bounds must promote to Float16, Float32, Float64, or BigFloat"))
    if T === BigFloat || (input === Union{} && _needs_exact_grid_precision(lb, ub, nbins))
        return BigFloat, iszero(bits) ? _exact_grid_bits(lb, ub, nbins) : bits
    end
    return T, 0
end

function _needs_exact_grid_precision(lb, ub, nbins)
    for (lo, hi) in zip(lb, ub)
        a, b = _grid_bounds(Float64, lo, hi)
        isfinite(a) && isfinite(b) || return true
        span = b - a
        !isfinite(span) && return true
        lo != hi && abs(span / nbins) <= max(2eps(max(abs(a), abs(b))), floatmin(Float64)) && return true
        _grid_bounds_fit(lo, hi, a, b) || return true
    end
    return false
end

function _exact_grid_bits(lb, ub, nbins)
    bits = 128
    bins_bits = ndigits(nbins; base = 2)
    for (lo, hi) in zip(lb, ub)
        a, b = _exact_bound(lo), _exact_bound(hi)
        for x in (a, b)
            numerator_bits = ndigits(abs(numerator(x)); base = 2)
            !iszero(x) && ispow2(denominator(x)) && (bits = max(bits, numerator_bits - trailing_zeros(abs(numerator(x)))))
        end
        if a != b
            scale = _rational_exponent(max(abs(a), abs(b)))
            span = _rational_exponent(abs(b - a))
            bits = max(bits, 64 + bins_bits + scale - span)
        end
    end
    return bits
end

_exact_bound(bound::Union{AbstractFloat, Integer, Rational}) = Rational{BigInt}(bound)

function _exact_bound(bound)
    return _with_vegas_precision(256) do
        Rational{BigInt}(BigFloat(bound))
    end
end

_rational_exponent(x) = ndigits(numerator(x); base = 2) - ndigits(denominator(x); base = 2)

function _with_vegas_precision(f, bits)
    return iszero(bits) || precision(BigFloat) == bits ? f() : setprecision(f, BigFloat, bits)
end

_grid_bound(::Type{T}, bound) where {T} = T(bound)

function _grid_bound(::Type{T}, bound::Rational) where {T <: Union{Float16, Float32, Float64}}
    return T(BigFloat(bound, RoundNearest; precision = 128), RoundNearest)
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

_is_binary_exact_bound(bound::Union{Integer, AbstractFloat}) = true
_is_binary_exact_bound(bound::Rational) = ispow2(denominator(bound))
_is_binary_exact_bound(bound) = false

_binary_bound_rounded(original, converted) =
    _is_binary_exact_bound(original) && original != converted

function _grid_bounds_fit(lo, hi, a, b)
    a == lo && b == hi && return true
    _binary_bound_rounded(lo, a) && return false
    _binary_bound_rounded(hi, b) && return false
    (lo < hi) == (a < b) && (hi < lo) == (b < a) || return false
    lo == hi && return true
    expected = _exact_bound(hi) - _exact_bound(lo)
    actual = Rational{BigInt}(b) - Rational{BigInt}(a)
    scale = big(1) << max(0, min(precision(a), precision(b), 53) - 2)
    return abs(expected - actual) * scale <= max(abs(expected), abs(actual))
end

function _uniform_vegas_grid(::Type{T}, lb, ub, nbins) where {T}
    grid = Matrix{T}(undef, nbins + 1, length(lb))
    widths = Matrix{T}(undef, nbins, length(lb))
    for (dim, (lo, hi)) in enumerate(zip(lb, ub))
        a, b = _grid_bounds(T, lo, hi)
        isfinite(a) && isfinite(b) && _grid_bounds_fit(lo, hi, a, b) ||
            throw(ArgumentError("the promoted coordinate precision cannot preserve the bounds in dimension $dim"))
        _set_uniform_grid_axis!(view(grid, :, dim), view(widths, :, dim), a, b)
    end
    return grid, widths
end

_grid_coordinate_type(edges, widths) = float(promote_type(eltype(edges), eltype(widths)))

function _bound_grid_point(point, grid, dim)
    return clamp(point, minmax(grid[1, dim], grid[end, dim])...)
end

_stored_grid_precision(grid) = 0
_stored_grid_precision(grid::AbstractArray{BigFloat}) = maximum(precision, grid; init = 0)
_stored_grid_precision(edges, widths) = max(_stored_grid_precision(edges), _stored_grid_precision(widths))

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
        edge = eltype(edges)(muladd(fraction, widths[bin], edges[bin]))
        # Keep the current map if a narrower bin cannot be represented on this axis.
        (ascending ? previous < edge < last(edges) : previous > edge > last(edges)) || return
        isfinite(edge - previous) || return
        proposed[i] = previous = edge
    end
    isfinite(last(edges) - previous) || return
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
