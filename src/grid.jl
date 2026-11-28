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

_with_grid_rounding(f, ::Type) = f()

function _with_grid_rounding(f, ::Type{BigFloat})
    return rounding(BigFloat) === RoundNearest ? f() : setrounding(f, BigFloat, RoundNearest)
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
    return _with_grid_rounding(T) do
        grid = Matrix{T}(undef, nbins + 1, length(lb))
        widths = Matrix{T}(undef, nbins, length(lb))
        for (dim, (lo, hi)) in enumerate(zip(lb, ub))
            a, b = _grid_bounds(T, lo, hi)
            isfinite(a) && isfinite(b) && _grid_bounds_fit(lo, hi, a, b) ||
                throw(ArgumentError("the promoted coordinate precision cannot preserve the bounds in dimension $dim"))
            _set_uniform_grid_axis!(view(grid, :, dim), view(widths, :, dim), a, b)
        end
        grid, widths
    end
end

_grid_coordinate_type(edges, widths) = float(promote_type(eltype(edges), eltype(widths)))

const _LowPrecisionGridFloat = Union{Float16, Float32}

_grid_bin_width(edges, widths, bin, tail...) = widths[bin, tail...]

function _grid_bin_width(edges::AbstractArray{T}, widths, bin, tail...) where {T <: _LowPrecisionGridFloat}
    return Float64(edges[bin + 1, tail...]) - Float64(edges[bin, tail...])
end

function _grid_bin_width(edges::AbstractArray{T}, widths::AbstractArray{BigFloat}, bin, tail...) where {T <: _LowPrecisionGridFloat}
    return widths[bin, tail...]
end

function _bound_grid_point(point, grid, dim)
    return clamp(point, minmax(grid[1, dim], grid[end, dim])...)
end

_stored_grid_precision(grid) = 0
_stored_grid_precision(grid::AbstractArray{BigFloat}) = maximum(precision, grid; init = 0)
_stored_grid_precision(edges, widths) = max(_stored_grid_precision(edges), _stored_grid_precision(widths))

function _adapt_grid!(grid, widths, bins, values, mass, counts, scratch, alpha, proposed = nothing, training_weights = nothing)
    nbins, ndim = size(mass)
    (iszero(alpha) || nbins == 1) && return
    scale = maximum(abs, values)
    iszero(scale) && return
    _bin_means!(mass, counts, bins, values, scale, training_weights)
    for dim in 1:ndim
        _smooth_weights!(scratch, view(mass, :, dim), alpha)
        candidate = proposed === nothing ? view(mass, :, dim) : proposed
        _rebin!(view(grid, :, dim), view(widths, :, dim), scratch, candidate)
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

function _set_uniform_grid_axis!(edges, widths, a::BigFloat, b::BigFloat)
    bits = precision(BigFloat)
    nbins = length(widths)
    if a == b
        fill!(edges, a)
        fill!(widths, zero(a))
        return nothing
    end
    _with_vegas_precision(bits + ndigits(nbins; base = 2) + 32) do
        span = b - a
        edges[1] = a
        previous = a
        for i in 1:(nbins - 1)
            edge = BigFloat(fma(BigFloat(i) / nbins, span, a); precision = bits)
            (a < b ? previous < edge < b : previous > edge > b) || throw(_grid_resolution_error(a, nbins))
            edges[i + 1] = previous = edge
        end
        edges[end] = b
    end
    for i in eachindex(widths)
        width = edges[i + 1] - edges[i]
        isfinite(width) && !iszero(width) || throw(_grid_resolution_error(a, nbins))
        widths[i] = width
    end
    return nothing
end

_bigfloat_mapping_widths(edges) = diff(BigFloat.(edges); dims = 1)
_bigfloat_mapping_widths(edges::AbstractArray{BigFloat}) = diff(edges; dims = 1)

_with_vegas_mapping(f, edges, widths) = _with_vegas_mapping(f, edges, widths, _grid_coordinate_type(edges, widths))
_with_vegas_mapping(f, edges, widths, ::Type) = f(widths, nothing)

function _with_vegas_mapping(f, edges, widths, ::Type{BigFloat})
    return _with_grid_rounding(BigFloat) do
        bits = precision(BigFloat)
        if bits >= 64
            working_widths = eltype(edges) === eltype(widths) === BigFloat ? widths : _bigfloat_mapping_widths(edges)
            return f(working_widths, nothing)
        end
        _with_vegas_precision(64) do
            f(_bigfloat_mapping_widths(edges), bits)
        end
    end
end

_round_vegas_coordinate(point, ::Nothing) = point
_round_vegas_coordinate(point, bits::Int) = BigFloat(point; precision = bits)

function _rebin!(edges, widths, mass, proposed)
    return _with_grid_rounding(eltype(edges)) do
        changed = _with_vegas_mapping(edges, widths) do working_widths, bits
            _rebin_grid_axis!(edges, working_widths, mass, proposed, bits)
        end
        if changed
            for i in eachindex(widths)
                widths[i] = edges[i + 1] - edges[i]
            end
        end
        nothing
    end
end

function _rebin_grid_axis!(edges, working_widths, mass, proposed, bits)
    nbins = length(mass)
    step = sum(mass) / nbins
    iszero(step) && return false
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
        edge = _round_vegas_coordinate(eltype(edges)(muladd(fraction, _grid_bin_width(edges, working_widths, bin), edges[bin])), bits)
        # Keep the current map if a narrower bin cannot be represented on this axis.
        (ascending ? previous < edge < last(edges) : previous > edge > last(edges)) || return false
        isfinite(_round_vegas_coordinate(edge - previous, bits)) || return false
        proposed[i] = previous = edge
    end
    isfinite(_round_vegas_coordinate(last(edges) - previous, bits)) || return false
    edges[2:nbins] .= view(proposed, 1:(nbins - 1))
    return true
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
