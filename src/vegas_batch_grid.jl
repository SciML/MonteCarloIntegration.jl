function _check_vegas_batch(batch)
    (batch === false || batch === true || batch === :matrix) && return nothing
    throw(ArgumentError("batch must be false, true, or :matrix"))
end

_vegas_batch_argument(points, batch) = batch === true ? collect(eachcol(points)) : points

_apply_vegas_weight(weight, value, ::Nothing, _) = weight * value

const _WideVegasValue = Union{BigInt, Rational{BigInt}, BigFloat}
const _BinaryVegasFloat = Union{Float16, Float32, Float64}
const _BinaryOrExactVegasValue = Union{Integer, Rational, _BinaryVegasFloat, BigFloat}

_vegas_weight_significand(value::Union{Integer, Rational}) = value, 0
_vegas_weight_significand(value::_BinaryVegasFloat) = Rational{BigInt}(value), 0

function _vegas_weight_significand(value::BigFloat)
    return _with_vegas_precision(precision(value)) do
        mantissa, exponent = frexp(value)
        Rational{BigInt}(mantissa), exponent
    end
end

function _apply_exact_vegas_weight(weight, value, exponent)
    isfinite(weight) && isfinite(value) || return weight * Float64(value)
    if iszero(weight) || iszero(value)
        return xor(signbit(weight), signbit(value)) ? -0.0 : 0.0
    end
    weight_mantissa, weight_exponent = _vegas_weight_significand(weight)
    value_mantissa, value_exponent = _vegas_weight_significand(value)
    weighted = weight_mantissa * value_mantissa
    exponent = Int128(exponent) + weight_exponent + value_exponent
    num, den = numerator(weighted), denominator(weighted)
    order = exponent + ndigits(num; base = 2) - ndigits(den; base = 2)
    # Bit lengths bound the magnitude above and below by adjacent powers of two.
    order <= -1076 && return num < 0 ? -0.0 : 0.0
    order >= 1025 && return num < 0 ? -Inf : Inf
    shift = exponent + 1074
    return shift >= 0 ? _round_binary64_units(num << shift, den) : _round_binary64_units(num, den << -shift)
end

_apply_vegas_weight(weight::Float64, value::_WideVegasValue, ::Nothing, _) = _apply_exact_vegas_weight(weight, value, 0)
_apply_vegas_weight(weight::Float64, value::_WideVegasValue, exponents, j) = _apply_exact_vegas_weight(weight, value, exponents[j])
_apply_vegas_weight(weight::BigFloat, value::_BinaryOrExactVegasValue, ::Nothing, _) = _apply_exact_vegas_weight(weight, value, 0)
_apply_vegas_weight(weight::BigFloat, value::_BinaryOrExactVegasValue, exponents, j) = _apply_exact_vegas_weight(weight, value, exponents[j])

function _apply_vegas_weight(weight::BigFloat, value::_BinaryVegasFloat, ::Nothing, _)
    # The binary product needs at most the sum of the two significand lengths.
    return _with_vegas_precision(precision(weight) + precision(typeof(value))) do
        Float64(weight * value, RoundNearest)
    end
end

function _apply_vegas_weight(weight, value, exponents, j)
    mantissa, exponent = frexp(float(value))
    return ldexp(weight * mantissa, exponents[j] + exponent)
end

function _evaluate!(weighted_values, func, xmat, batch, exponents = nothing, weights = weighted_values)
    if batch === false
        for (j, point) in enumerate(eachcol(xmat))
            weighted_values[j] = _apply_vegas_weight(weights[j], func(point), exponents, j)
        end
        return weighted_values
    end

    values = func(_vegas_batch_argument(xmat, batch))
    batch === :matrix && !(values isa AbstractVector) &&
        throw(ArgumentError("batch = :matrix requires the integrand to return a vector"))
    ncalls = length(weighted_values)
    length(values) == ncalls || throw(DimensionMismatch("the integrand must return $ncalls values per batch"))
    for (j, value) in enumerate(values)
        weighted_values[j] = _apply_vegas_weight(weights[j], value, exponents, j)
    end
    return weighted_values
end

function _set_uniform_grid_axis!(edges, widths, a, b)
    nbins = length(widths)
    if a == b
        fill!(edges, a)
        fill!(widths, zero(a))
        return nothing
    end

    span = b - a
    width = isfinite(span) ? span / nbins : b / nbins - a / nbins
    edges[1] = a
    if a isa Float64 && abs(width) <= floatmin(Float64)
        _set_subnormal_grid_edges!(edges, a, b, nbins)
    else
        isfinite(width) && !iszero(width) || throw(_grid_resolution_error(a, nbins))
        previous = a
        for i in 1:(nbins - 1)
            edge = fma(i, width, a)
            (a < b ? previous < edge < b : previous > edge > b) || throw(_grid_resolution_error(a, nbins))
            edges[i + 1] = previous = edge
        end
    end
    edges[end] = b
    for i in eachindex(widths)
        widths[i] = edges[i + 1] - edges[i]
        isfinite(widths[i]) && !iszero(widths[i]) || throw(_grid_resolution_error(a, nbins))
    end
    return nothing
end

function _set_subnormal_grid_edges!(edges, a, b, nbins)
    left, right = _binary64_units(a), _binary64_units(b)
    origin, span = nbins * left, right - left
    previous = a
    for i in 1:(nbins - 1)
        edge = _round_binary64_units(origin + i * span, nbins)
        (a < b ? previous < edge < b : previous > edge > b) || throw(_grid_resolution_error(a, nbins))
        edges[i + 1] = previous = edge
    end
    return nothing
end

function _set_uniform_grid_axis!(edges, widths, a::T, b::T) where {T <: _LowPrecisionGridFloat}
    nbins = length(widths)
    if a == b
        fill!(edges, a)
        fill!(widths, zero(T))
        return nothing
    end

    origin = Float64(a)
    span = Float64(b) - origin
    edges[1] = a
    previous = a
    for i in 1:(nbins - 1)
        edge = T(muladd(i / nbins, span, origin))
        (a < b ? previous < edge < b : previous > edge > b) || throw(_grid_resolution_error(a, nbins))
        edges[i + 1] = previous = edge
    end
    edges[end] = b
    for i in eachindex(widths)
        widths[i] = edges[i + 1] - edges[i]
        isfinite(widths[i]) && !iszero(widths[i]) || throw(_grid_resolution_error(a, nbins))
    end
    return nothing
end

function _grid_resolution_error(bound, nbins)
    return ArgumentError("the selected coordinate precision ($(typeof(bound)), $(precision(bound)) bits) cannot represent $nbins grid bins; use fewer bins or higher-precision bounds")
end

"""
    vegas(f, previous::VEGASResult; kwargs...)

Integrate `f` over the stored bounds, starting from a copy of the adaptive grid.
The point type and BigFloat precision come from the stored grid. The estimate
and uncertainty use only the samples from this call. If supplied, `nbins` must
match the grid size. Other keywords have the same meaning as in [`vegas`](@ref).
"""
function vegas(func, previous::VEGASResult; nbins = nothing, kwargs...)
    edges, widths = previous.adaptive_grid, previous.grid_spacing
    return _with_vegas_precision(_stored_grid_precision(edges, widths)) do
        x, delx = _with_grid_rounding(_grid_coordinate_type(edges, widths)) do
            _copy_vegas_grid(edges, widths)
        end
        grid_bins = size(delx, 1)
        (nbins === nothing || nbins == grid_bins) || throw(ArgumentError("nbins must match the previous grid ($grid_bins)"))
        _vegas(func, view(x, 1, :), view(x, size(x, 1), :), (x, delx); nbins = grid_bins, kwargs...)
    end
end

function _copy_vegas_grid(edges, widths)
    edges isa AbstractMatrix{<:Real} && widths isa AbstractMatrix{<:Real} ||
        throw(ArgumentError("the previous grid must contain real matrices of edges and widths"))
    nbins, ndim = size(widths)
    nbins > 0 && ndim > 0 && size(edges) == (nbins + 1, ndim) ||
        throw(ArgumentError("the previous grid must have one more edge than width per dimension"))
    T = _grid_coordinate_type(edges, widths)
    x, delx = Matrix{T}(edges), Matrix{T}(widths)
    all(isfinite, x) && all(isfinite, delx) ||
        throw(ArgumentError("the previous grid must contain finite edges and widths"))
    rtol = 2eps(T)
    for storage in (eltype(edges), eltype(widths))
        storage in (Float16, Float32, Float64) && (rtol = max(rtol, 2eps(storage)))
    end
    for dim in axes(x, 2)
        original = view(edges, :, dim)
        promoted = view(x, :, dim)
        if eltype(edges) !== T
            for i in 1:nbins
                _grid_bounds_fit(original[i], original[i + 1], promoted[i], promoted[i + 1]) ||
                    throw(ArgumentError("the previous grid has edges that cannot be represented at the stored coordinate precision in dimension $dim"))
            end
        end
        _canonicalize_grid_axis!(promoted, view(delx, :, dim), original, view(widths, :, dim), rtol) ||
            throw(ArgumentError("the previous grid has inconsistent edges and widths in dimension $dim"))
    end
    if T === BigFloat
        bits = precision(BigFloat)
        for values in (x, delx), i in eachindex(values)
            precision(values[i]) == bits || (values[i] = BigFloat(values[i]; precision = bits))
        end
    end
    return x, delx
end

function _sampling_grid(result)
    edges, widths = result.adaptive_grid, result.grid_spacing
    T = eltype(edges)
    if _mixed_grid_storage(edges, widths) || T <: _GridStorageExact
        return _with_grid_rounding(() -> _copy_vegas_grid(edges, widths), _grid_coordinate_type(edges, widths))
    end
    if T === Float64
        _float64_grid_spans_match(edges, widths) || return _copy_vegas_grid(edges, widths)
    elseif T in (Float16, Float32)
        nbins = size(widths, 1)
        for dim in axes(edges, 2)
            a, b = edges[begin, dim], edges[end, dim]
            if a != b && abs((b - a) / nbins) <= 2eps(max(abs(a), abs(b))) &&
                    _is_uniform_grid_axis(view(edges, :, dim), view(widths, :, dim))
                return _copy_vegas_grid(edges, widths)
            end
        end
    end
    return edges, widths
end

function _float64_grid_spans_match(edges, widths)
    size(edges) == (size(widths, 1) + 1, size(widths, 2)) || return false
    for dim in axes(widths, 2), i in axes(widths, 1)
        edges[i + 1, dim] - edges[i, dim] == widths[i, dim] || return false
    end
    return true
end

function _canonicalize_grid_axis!(edges::AbstractVector{Float64}, widths, stored_edges, stored_widths, rtol)
    a, b = first(edges), last(edges)
    direction = cmp(b, a)
    nominal = length(widths) > 1 && _is_uniform_grid_axis(edges, widths, (stored_edges, stored_widths))
    if nominal && a != b && any(i -> edges[i + 1] == edges[i], eachindex(widths))
        _set_uniform_grid_axis!(edges, widths, a, b)
        return true
    end
    for i in eachindex(widths)
        actual = edges[i + 1] - edges[i]
        isfinite(actual) && (iszero(actual) || sign(actual) == direction) || return false
        (nominal || _grid_width_matches(actual, stored_edges[i], stored_edges[i + 1], stored_widths[i], rtol)) || return false
        widths[i] = actual
    end
    return true
end

const _GridStorageFloat = Union{Float16, Float32, Float64, BigFloat}
const _GridStorageExact = Union{Integer, Rational}

_grid_width_matches(actual, left, right, width, rtol) = isapprox(actual, width; rtol)

_grid_width_rtol(width, rtol) = rtol
_grid_width_rtol(width::BigFloat, rtol) = max(rtol, _bigfloat_width_rtol(width))

_round_grid_storage(value, reference::BigFloat) = BigFloat(value, RoundNearest; precision = precision(reference))
_round_grid_storage(value, ::T) where {T <: Union{Float16, Float32, Float64}} = T(value, RoundNearest)

function _grid_width_matches(actual, left, right, width::_GridStorageFloat, rtol)
    return _round_grid_storage(actual, width) == width || isapprox(actual, width; rtol = _grid_width_rtol(width, rtol))
end

function _grid_width_matches(actual, left::_GridStorageExact, right::_GridStorageExact, width::_GridStorageExact, rtol)
    return _exact_bound(right) - _exact_bound(left) == width
end

function _grid_width_matches(actual, left::_GridStorageExact, right::_GridStorageExact, width::_GridStorageFloat, rtol)
    span = _exact_bound(right) - _exact_bound(left)
    cmp(span, width) == 0 && return true
    lo, hi, closed_lo, closed_hi = _grid_rounding_interval(width)
    lower, upper = cmp(span, lo), cmp(span, hi)
    return (lower > 0 || (lower == 0 && closed_lo)) && (upper < 0 || (upper == 0 && closed_hi))
end

function _grid_width_matches(actual, left::_GridStorageFloat, right::_GridStorageFloat, width::_GridStorageExact, rtol)
    span = _exact_bound(width)
    lo_left, hi_left, close_lo_left, close_hi_left = _grid_rounding_interval(left)
    lo_right, hi_right, close_lo_right, close_hi_right = _grid_rounding_interval(right)
    lower = _compare_grid_span(lo_right, hi_left, span)
    (lower < 0 || (lower == 0 && close_lo_right && close_hi_left)) || return false
    upper = _compare_grid_span(hi_right, lo_left, span)
    return upper > 0 || (upper == 0 && close_hi_right && close_lo_left)
end

function _grid_width_matches(actual, left::_GridStorageFloat, right::_GridStorageFloat, width::_GridStorageFloat, rtol)
    (actual == width || _round_grid_storage(actual, width) == width) && return true
    if typeof(left) === typeof(right) === typeof(width) && precision(left) == precision(right) == precision(width)
        return isapprox(actual, width; rtol = _grid_width_rtol(width, rtol))
    end
    lo_left, hi_left, close_lo_left, close_hi_left = _grid_rounding_interval(left)
    lo_right, hi_right, close_lo_right, close_hi_right = _grid_rounding_interval(right)
    lo_width, hi_width, close_lo_width, close_hi_width = _grid_rounding_interval(width)
    close_low_span = close_lo_right && close_hi_left && close_hi_width
    close_high_span = close_hi_right && close_lo_left && close_lo_width
    return _grid_span_below(lo_right, hi_left, hi_width, close_low_span) &&
        _grid_span_above(hi_right, lo_left, lo_width, close_high_span)
end

function _grid_rounding_interval(value)
    prev, next = prevfloat(value), nextfloat(value)
    lo, hi = _with_vegas_precision(precision(value) + 2) do
        center = BigFloat(value)
        previous, following = BigFloat(prev), BigFloat(next)
        lower_step = isfinite(previous) ? center - previous : following - center
        upper_step = isfinite(following) ? following - center : center - previous
        center - lower_step / 2, center + upper_step / 2
    end
    return lo, hi, _round_grid_storage(lo, value) == value, _round_grid_storage(hi, value) == value
end

function _subtract_grid_span(right, left, bits, rounding)
    return _with_vegas_precision(bits) do
        setrounding(() -> right - left, BigFloat, rounding)
    end
end

function _grid_span_below(right, left, limit, closed)
    difference = _subtract_grid_span(right, left, precision(limit), closed ? RoundUp : RoundDown)
    return closed ? difference <= limit : difference < limit
end

function _grid_span_above(right, left, limit, closed)
    difference = _subtract_grid_span(right, left, precision(limit), closed ? RoundDown : RoundUp)
    return closed ? difference >= limit : difference > limit
end

function _compare_grid_span(right, left, exact)
    # Directed bounds and exact comparisons avoid expanding widely separated exponents.
    bits = max(precision(right), precision(left))
    while true
        lo = _subtract_grid_span(right, left, bits, RoundDown)
        hi = _subtract_grid_span(right, left, bits, RoundUp)
        lower = cmp(lo, exact)
        lo == hi && return lower
        lower >= 0 && return 1
        cmp(hi, exact) <= 0 && return -1
        bits *= 2
    end
    return
end

function _canonicalize_grid_axis!(edges, widths, stored_edges, stored_widths, rtol)
    a, b = first(edges), last(edges)
    nominal = length(widths) > 1 && _is_uniform_grid_axis(edges, widths, (stored_edges, stored_widths))
    if nominal && _needs_nominal_grid_rebuild(edges, widths, stored_edges, stored_widths)
        _set_uniform_grid_axis!(edges, widths, a, b)
        return true
    end
    direction = cmp(b, a)
    for i in eachindex(widths)
        actual = edges[i + 1] - edges[i]
        isfinite(actual) && (iszero(actual) || sign(actual) == direction) || return false
        (nominal || _grid_width_matches(actual, stored_edges[i], stored_edges[i + 1], stored_widths[i], rtol)) || return false
        widths[i] = actual
    end
    return true
end

function _canonicalize_grid_axis!(edges::AbstractVector{T}, widths, stored_edges, stored_widths, rtol) where {T <: _LowPrecisionGridFloat}
    a, b = first(edges), last(edges)
    direction = cmp(b, a)
    valid = all(eachindex(widths)) do i
        actual = edges[i + 1] - edges[i]
        return isfinite(actual) && (iszero(actual) || sign(actual) == direction) &&
            _grid_width_matches(actual, stored_edges[i], stored_edges[i + 1], stored_widths[i], rtol)
    end
    if !valid
        length(widths) > 1 && _is_uniform_grid_axis(edges, widths, (stored_edges, stored_widths)) || return false
        if _needs_nominal_grid_rebuild(edges, widths, stored_edges, stored_widths)
            _set_uniform_grid_axis!(edges, widths, a, b)
            return true
        end
    end
    for i in eachindex(widths)
        actual = edges[i + 1] - edges[i]
        isfinite(actual) && (iszero(actual) || sign(actual) == direction) || return false
        widths[i] = actual
    end
    return true
end

function _needs_nominal_grid_rebuild(edges, widths, stored_edges, stored_widths)
    return !_mixed_grid_storage(stored_edges, stored_widths) ||
        (first(edges) != last(edges) && any(i -> edges[i] == edges[i + 1], eachindex(widths)))
end

function _is_uniform_grid_axis(edges, widths, stored = nothing)
    width = first(widths)
    all(==(width), widths) || return false
    if stored !== nothing
        original_edges, original_widths = stored
        (eltype(original_edges) <: _GridStorageExact || eltype(original_widths) <: _GridStorageExact) && return false
        all(==(first(original_widths)), original_widths) || return false
    end
    check_last = stored !== nothing && _mixed_grid_storage(stored...)
    types = check_last ? (eltype(edges), Float64, Float32, Float16) : (eltype(edges), Float32, Float16)
    anchored = check_last && _matches_anchored_grid_width(stored...)
    return any(T -> _matches_uniform_grid(edges, width, T, check_last, anchored), types)
end

function _matches_anchored_grid_width(edges, widths)
    a, b = first(edges), last(edges)
    declared = eltype(widths) === BigFloat ? argmax(precision, widths) : first(widths)
    a isa _GridStorageFloat && b isa _GridStorageFloat && declared isa _GridStorageFloat || return false
    nbins = length(widths)
    lo, hi, closed_lo, closed_hi = _grid_rounding_interval(declared)
    bits = max(precision(lo), precision(hi)) + ndigits(nbins; base = 2)
    lo, hi = _with_vegas_precision(bits) do
        nbins * lo, nbins * hi
    end
    left, right = BigFloat(a; precision = precision(a)), BigFloat(b; precision = precision(b))
    return _grid_span_below(right, left, hi, closed_hi) && _grid_span_above(right, left, lo, closed_lo)
end

function _mixed_grid_storage(edges, widths)
    eltype(edges) === eltype(widths) || return true
    eltype(edges) === BigFloat || return false
    isempty(edges) && return false
    bits = precision(first(edges))
    return any(x -> precision(x) != bits, edges) || any(x -> precision(x) != bits, widths)
end

_bigfloat_width_rtol(width) = ldexp(one(width), 2 - precision(width))

function _matches_historical_grid_edge(point::BigFloat, index, width, origin)
    bits = precision(point)
    point == BigFloat(fma(index, width, origin); precision = bits) && return true
    offset = BigFloat(index * width; precision = precision(width))
    return point == BigFloat(origin + offset; precision = bits)
end

function _matches_uniform_grid(edges, width, ::Type{BigFloat}, check_last, anchored)
    a, b = first(edges), last(edges)
    nbins = length(edges) - 1
    nominal = (b - a) / nbins
    if check_last
        (anchored || width == nominal || _matches_historical_grid_edge(b, nbins, width, a)) || return false
    else
        isapprox(width, nominal; rtol = _bigfloat_width_rtol(width)) || return false
    end
    return all(i -> _matches_historical_grid_edge(edges[i], i - 1, width, a), 1:nbins)
end

function _matches_uniform_grid(edges, width, ::Type{T}, check_last, anchored) where {T}
    a, b, w = T(first(edges)), T(last(edges)), T(width)
    a == first(edges) && b == last(edges) && w == width || return false
    nbins = length(edges) - 1
    nominal = (b - a) / nbins
    if check_last
        (anchored || w == nominal || b == fma(nbins, w, a) || b == a + nbins * w) || return false
    else
        isapprox(w, nominal; rtol = 2eps(T)) || return false
    end
    for i in 1:nbins
        point = edges[i]
        point == fma(i - 1, w, a) || point == a + (i - 1) * w || return false
    end
    return true
end
