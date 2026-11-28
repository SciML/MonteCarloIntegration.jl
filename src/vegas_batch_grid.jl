function _check_vegas_batch(batch)
    (batch === false || batch === true || batch === :matrix) && return nothing
    throw(ArgumentError("batch must be false, true, or :matrix"))
end

function _vegas_callback_coordinates(func, grid, batch, requested)
    requested === BigFloat && return _bigfloat_callback_coordinates(grid)
    (requested === nothing || requested === Float64) ||
        throw(ArgumentError("callback_type must be nothing, Float64, or BigFloat"))
    accepts_float64 = requested === Float64 || applicable(func, _vegas_callback_argument(Float64, batch))
    if accepts_float64
        exact = eltype(grid) === Float64 || _callback_bounds_fit(Float64, grid)
        bounds = exact ? nothing : _inward_float64_callback_bounds(grid)
        if exact || bounds !== nothing
            requested === nothing && eltype(grid) === BigFloat && _warn_sparse_float64_callbacks(grid, bounds)
            return Float64, bounds
        end
    end
    if requested === nothing && applicable(func, _vegas_callback_argument(BigFloat, batch))
        return _bigfloat_callback_coordinates(grid)
    end
    accepts_float64 && throw(ArgumentError("Float64 callback coordinates require at least one finite Float64 value within every axis"))
    throw(ArgumentError("the integrand must accept Float64 or BigFloat coordinates in the selected batch mode"))
end

function _bigfloat_callback_coordinates(grid)
    (eltype(grid) === BigFloat || _callback_bounds_fit(BigFloat, grid)) && return BigFloat, nothing
    throw(ArgumentError("active BigFloat precision cannot preserve the callback bounds; increase it with setprecision"))
end

_vegas_batch_argument(points, batch) = batch === true ? collect(eachcol(points)) : points

function _vegas_callback_argument(::Type{T}, batch) where {T}
    points = Matrix{T}(undef, 0, 1)
    return batch === false ? first(eachcol(points)) : _vegas_batch_argument(points, batch)
end

function _callback_bounds_fit(::Type{T}, grid) where {T}
    for dim in axes(grid, 2), row in (firstindex(grid, 1), lastindex(grid, 1))
        bound = grid[row, dim]
        converted = T(bound)
        isfinite(converted) && bound == converted || return false
    end
    return true
end

function _inward_float64_callback_bounds(grid)
    lower = Vector{Float64}(undef, size(grid, 2))
    upper = similar(lower)
    for dim in axes(grid, 2)
        a, b = minmax(grid[begin, dim], grid[end, dim])
        lo, hi = Float64(a, RoundUp), Float64(b, RoundDown)
        isfinite(lo) && isfinite(hi) && lo <= hi || return nothing
        lower[dim], upper[dim] = lo, hi
    end
    return lower, upper
end

function _warn_sparse_float64_callbacks(grid, bounds)
    budget = sqrt(eps(Float64))
    nbins = size(grid, 1) - 1
    for dim in axes(grid, 2)
        a, b = minmax(grid[begin, dim], grid[end, dim])
        a == b && continue
        lo, hi = bounds === nothing ? (Float64(a), Float64(b)) : (bounds[1][dim], bounds[2][dim])
        span = b - a
        spacing = max(nextfloat(lo) - lo, hi - prevfloat(hi))
        inward_loss = (BigFloat(lo) - a) + (b - BigFloat(hi))
        if spacing > budget * (span / nbins) || inward_loss > budget * span
            @warn "VEGAS default Float64 callback coordinates are coarse on this grid axis. The estimated standard error excludes coordinate-rounding bias. Use callback_type = BigFloat for higher-precision samples." axis = dim
            return nothing
        end
    end
    return nothing
end

_callback_grid_point(point, grid, dim, ::Nothing) = _bound_grid_point(point, grid, dim)

function _callback_grid_point(point, grid, dim, bounds::Tuple)
    lower, upper = bounds
    return clamp(Float64(point), lower[dim], upper[dim])
end

_apply_vegas_weight(weight, value, ::Nothing, _) = weight * value

function _apply_vegas_weight(weight, value, exponents, j)
    mantissa, exponent = frexp(float(value))
    return ldexp(weight * mantissa, exponents[j] + exponent)
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

_initialize_vegas_grid(lb, ub, nbins, initial_grid) = initial_grid

function _set_uniform_grid_axis!(edges, widths, a, b)
    width = (b - a) / length(widths)
    _check_uniform_grid_resolution(a, b, width)
    for i in eachindex(widths)
        edges[i] = fma(i - 1, width, a)
        widths[i] = width
    end
    edges[end] = b
    return nothing
end

_check_uniform_grid_resolution(a, b, width) = nothing

function _check_uniform_grid_resolution(a::BigFloat, b::BigFloat, width)
    if a != b && abs(width) <= 2eps(max(abs(a), abs(b)))
        throw(ArgumentError("active BigFloat precision cannot resolve the uniform grid bins"))
    end
    return nothing
end

"""
    vegas(f, previous::VEGASResult; kwargs...)

Integrate `f` over the bounds stored in `previous`, starting from a copy of its
adaptive grid. The estimate and uncertainty use only the samples from this
call. The number of bins comes from the grid. If supplied, `nbins` must match
that number. Uniform grids with bins too narrow for `Float64` are reconstructed
at the active `BigFloat` precision. Reusing a `BigFloat` grid requires active
precision at least as high as the largest precision stored in its edges or
widths. The default `callback_type` prefers `Float64` when the integrand accepts
it and each stored axis contains a finite `Float64` value.
Other keywords have the same meaning as in [`vegas`](@ref).
"""
function vegas(func, previous::VEGASResult; nbins = nothing, kwargs...)
    x, delx = _copy_vegas_grid(previous)
    grid_bins = size(delx, 1)
    (nbins === nothing || nbins == grid_bins) || throw(ArgumentError("nbins must match the previous grid ($grid_bins)"))
    return _vegas(func, view(x, 1, :), view(x, size(x, 1), :), (x, delx); nbins = grid_bins, kwargs...)
end

function _copy_vegas_grid(previous)
    edges, widths = previous.adaptive_grid, previous.grid_spacing
    edges isa AbstractMatrix{<:Real} && widths isa AbstractMatrix{<:Real} ||
        throw(ArgumentError("the previous grid must contain real matrices of edges and widths"))
    nbins, ndim = size(widths)
    nbins > 0 && ndim > 0 && size(edges) == (nbins + 1, ndim) ||
        throw(ArgumentError("the previous grid must have one more edge than width per dimension"))
    _check_grid_precision(edges, widths)
    T = _grid_coordinate_type(edges, widths)
    x, delx = Matrix{T}(edges), Matrix{T}(widths)
    all(isfinite, x) && all(isfinite, delx) ||
        throw(ArgumentError("the previous grid must contain finite edges and widths"))
    rtol = eltype(widths) in (Float16, Float32) ? 2eps(eltype(widths)) : 2eps(eltype(delx))
    for dim in axes(x, 2)
        _canonicalize_grid_axis!(view(x, :, dim), view(delx, :, dim), rtol) ||
            throw(ArgumentError("the previous grid has inconsistent edges and widths in dimension $dim"))
    end
    if T === BigFloat
        p = precision(BigFloat)
        for i in eachindex(x)
            precision(x[i]) == p || (x[i] = BigFloat(x[i]; precision = p))
        end
    end
    _unresolved_uniform_grid(x, delx) && return _widen_uniform_grid(x, delx)
    return x, delx
end

function _sampling_grid(result)
    edges, widths = result.adaptive_grid, result.grid_spacing
    _check_grid_precision(edges, widths)
    _unresolved_uniform_grid(edges, widths) && return _copy_vegas_grid(result)
    return edges, widths
end

_unresolved_uniform_grid(edges, widths) = false

function _unresolved_uniform_grid(edges::AbstractMatrix{Float64}, widths::AbstractMatrix{Float64})
    nbins, ndim = size(widths)
    nbins > 0 && size(edges) == (nbins + 1, ndim) || return false
    for dim in axes(edges, 2)
        if _needs_wide_grid_axis(edges[1, dim], edges[end, dim], nbins) &&
                _is_uniform_grid_axis(view(edges, :, dim), view(widths, :, dim))
            return true
        end
    end
    return false
end

function _widen_uniform_grid(edges, widths)
    wide_edges = Matrix{BigFloat}(edges)
    wide_edges == edges || throw(ArgumentError("active BigFloat precision cannot represent the stored grid edges"))
    wide_widths = similar(wide_edges, size(widths))
    for dim in axes(edges, 2)
        if _is_uniform_grid_axis(view(edges, :, dim), view(widths, :, dim))
            _check_bigfloat_grid_bounds(
                edges[1, dim], edges[end, dim], wide_edges[1, dim], wide_edges[end, dim], size(widths, 1),
            )
            _set_uniform_grid_axis!(
                view(wide_edges, :, dim), view(wide_widths, :, dim), wide_edges[1, dim], wide_edges[end, dim],
            )
        else
            for i in axes(widths, 1)
                a, b = wide_edges[i, dim], wide_edges[i + 1, dim]
                _check_bigfloat_grid_bounds(edges[i, dim], edges[i + 1, dim], a, b, 1)
                wide_widths[i, dim] = b - a
            end
        end
    end
    return wide_edges, wide_widths
end

function _canonicalize_grid_axis!(edges, widths, rtol)
    a, b = first(edges), last(edges)
    span = b - a
    isfinite(span) || return false
    if _is_uniform_grid_axis(edges, widths)
        _set_uniform_grid_axis!(edges, widths, a, b)
        return true
    end
    direction = sign(span)
    for i in eachindex(widths)
        actual = edges[i + 1] - edges[i]
        width = widths[i]
        width_rtol = width isa BigFloat ? _bigfloat_width_rtol(width) : rtol
        (iszero(actual) || sign(actual) == direction) && isapprox(actual, width; rtol = width_rtol) || return false
        widths[i] = actual
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
