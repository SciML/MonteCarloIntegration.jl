function _check_vegas_batch(batch)
    (batch === false || batch === true || batch === :matrix) && return nothing
    throw(ArgumentError("batch must be false, true, or :matrix"))
end

_vegas_batch_argument(points, batch) = batch === true ? collect(eachcol(points)) : points

_apply_vegas_weight(weight, value, ::Nothing, _) = weight * value

const _ExactVegasValue = Union{BigInt, Rational{BigInt}}

function _apply_exact_vegas_weight(weight, value, exponent)
    isfinite(weight) && isfinite(value) || return weight * Float64(value)
    if iszero(weight) || iszero(value)
        return xor(signbit(weight), signbit(value)) ? -0.0 : 0.0
    end
    weighted = Rational{BigInt}(weight) * value
    num, den = numerator(weighted), denominator(weighted)
    order = Int128(exponent) + ndigits(num; base = 2) - ndigits(den; base = 2)
    # Bit lengths bound the magnitude above and below by adjacent powers of two.
    order <= -1076 && return num < 0 ? -0.0 : 0.0
    order >= 1025 && return num < 0 ? -Inf : Inf
    shift = Int128(exponent) + 1074
    return shift >= 0 ? _round_binary64_units(num << shift, den) : _round_binary64_units(num, den << -shift)
end

_apply_vegas_weight(weight, value::_ExactVegasValue, ::Nothing, _) = _apply_exact_vegas_weight(weight, value, 0)
_apply_vegas_weight(weight, value::_ExactVegasValue, exponents, j) = _apply_exact_vegas_weight(weight, value, exponents[j])

function _apply_vegas_weight(weight, value::BigFloat, ::Nothing, _)
    return _with_vegas_precision(max(64, precision(value))) do
        Float64(weight * value, RoundNearest)
    end
end

function _apply_vegas_weight(weight, value, exponents, j)
    mantissa, exponent = frexp(float(value))
    return ldexp(weight * mantissa, exponents[j] + exponent)
end

function _apply_vegas_weight(weight, value::BigFloat, exponents, j)
    return _with_vegas_precision(max(64, precision(value))) do
        mantissa, exponent = frexp(value)
        Float64(ldexp(weight * mantissa, exponents[j] + exponent), RoundNearest)
    end
end

function _evaluate!(weighted_values, func, xmat, batch, exponents = nothing)
    if batch === false
        for (j, point) in enumerate(eachcol(xmat))
            weighted_values[j] = _apply_vegas_weight(weighted_values[j], func(point), exponents, j)
        end
        return weighted_values
    end

    values = func(_vegas_batch_argument(xmat, batch))
    batch === :matrix && !(values isa AbstractVector) &&
        throw(ArgumentError("batch = :matrix requires the integrand to return a vector"))
    ncalls = length(weighted_values)
    length(values) == ncalls || throw(DimensionMismatch("the integrand must return $ncalls values per batch"))
    for (j, value) in enumerate(values)
        weighted_values[j] = _apply_vegas_weight(weighted_values[j], value, exponents, j)
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
    isfinite(width) && !iszero(width) || throw(_grid_resolution_error(a, nbins))
    edges[1] = a
    previous = a
    for i in 1:(nbins - 1)
        edge = fma(i, width, a)
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
    rtol = eltype(widths) in (Float16, Float32) ? 2eps(eltype(widths)) : 2eps(T)
    for dim in axes(x, 2)
        _canonicalize_grid_axis!(view(x, :, dim), view(delx, :, dim), rtol) ||
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
    T !== eltype(widths) && _grid_coordinate_type(edges, widths) === BigFloat && return edges, widths
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

function _canonicalize_grid_axis!(edges::AbstractVector{Float64}, widths, rtol)
    a, b = first(edges), last(edges)
    direction = cmp(b, a)
    nominal = _is_uniform_grid_axis(edges, widths)
    if nominal && a != b && any(i -> edges[i + 1] == edges[i], eachindex(widths))
        _set_uniform_grid_axis!(edges, widths, a, b)
        return true
    end
    for i in eachindex(widths)
        actual = edges[i + 1] - edges[i]
        isfinite(actual) && (iszero(actual) || sign(actual) == direction) || return false
        (nominal || isapprox(actual, widths[i]; rtol)) || return false
        widths[i] = actual
    end
    return true
end

function _canonicalize_grid_axis!(edges, widths, rtol)
    a, b = first(edges), last(edges)
    if _is_uniform_grid_axis(edges, widths)
        _set_uniform_grid_axis!(edges, widths, a, b)
        return true
    end
    direction = cmp(b, a)
    for i in eachindex(widths)
        actual = edges[i + 1] - edges[i]
        width = widths[i]
        width_rtol = width isa BigFloat ? _bigfloat_width_rtol(width) : rtol
        isfinite(actual) && (iszero(actual) || sign(actual) == direction) && isapprox(actual, width; rtol = width_rtol) || return false
        widths[i] = actual
    end
    return true
end

function _canonicalize_grid_axis!(edges::AbstractVector{T}, widths, rtol) where {T <: _LowPrecisionGridFloat}
    a, b = first(edges), last(edges)
    direction = cmp(b, a)
    valid = all(eachindex(widths)) do i
        actual = edges[i + 1] - edges[i]
        return isfinite(actual) && (iszero(actual) || sign(actual) == direction) && isapprox(actual, widths[i]; rtol)
    end
    if valid
        for i in eachindex(widths)
            widths[i] = edges[i + 1] - edges[i]
        end
    elseif _is_uniform_grid_axis(edges, widths)
        _set_uniform_grid_axis!(edges, widths, a, b)
    else
        return false
    end
    return true
end

function _is_uniform_grid_axis(edges, widths)
    width = first(widths)
    return all(==(width), widths) && any(T -> _matches_uniform_grid(edges, width, T), (eltype(edges), Float32, Float16))
end

_bigfloat_width_rtol(width) = ldexp(one(width), 2 - precision(width))

function _matches_uniform_grid(edges, width, ::Type{BigFloat})
    a, b = first(edges), last(edges)
    nbins = length(edges) - 1
    isapprox(width, (b - a) / nbins; rtol = _bigfloat_width_rtol(width)) || return false
    for i in 1:nbins
        point = edges[i]
        point_precision = precision(point)
        fused = BigFloat(fma(i - 1, width, a); precision = point_precision)
        if point != fused
            offset = BigFloat((i - 1) * width; precision = precision(width))
            point == BigFloat(a + offset; precision = point_precision) || return false
        end
    end
    return true
end

function _matches_uniform_grid(edges, width, ::Type{T}) where {T}
    a, b, w = T(first(edges)), T(last(edges)), T(width)
    a == first(edges) && b == last(edges) && w == width || return false
    nbins = length(edges) - 1
    isapprox(w, (b - a) / nbins; rtol = 2eps(T)) || return false
    for i in 1:nbins
        point = edges[i]
        point == fma(i - 1, w, a) || point == a + (i - 1) * w || return false
    end
    return true
end
