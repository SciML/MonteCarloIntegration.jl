function _check_vegas_batch(batch)
    (batch === false || batch === true || batch === :matrix) && return nothing
    throw(ArgumentError("batch must be false, true, or :matrix"))
end

function _evaluate!(weighted_values, func, xmat, batch)
    if batch === false
        for (j, point) in enumerate(eachcol(xmat))
            weighted_values[j] *= func(point)
        end
        return weighted_values
    end

    if batch === :matrix
        values = func(xmat)
        values isa AbstractVector || throw(ArgumentError("batch = :matrix requires the integrand to return a vector"))
    else
        values = func(collect(eachcol(xmat)))
    end
    ncalls = length(weighted_values)
    length(values) == ncalls || throw(DimensionMismatch("the integrand must return $ncalls values per batch"))
    for (j, value) in enumerate(values)
        weighted_values[j] *= value
    end
    return weighted_values
end

function _initialize_vegas_grid(lb, ub, nbins, ::Nothing)
    x = zeros(nbins + 1, length(lb))
    delx = zeros(nbins, length(lb))
    for dim in eachindex(lb, ub)
        _set_uniform_grid_axis!(view(x, :, dim), view(delx, :, dim), Float64(lb[dim]), Float64(ub[dim]))
    end
    return x, delx
end

_initialize_vegas_grid(lb, ub, nbins, initial_grid) = initial_grid

function _set_uniform_grid_axis!(edges, widths, a, b)
    width = (b - a) / length(widths)
    for i in eachindex(widths)
        edges[i] = fma(i - 1, width, a)
        widths[i] = width
    end
    edges[end] = b
    return nothing
end

"""
    vegas(f, previous::VEGASResult; kwargs...)

Integrate `f` over the same bounds as `previous`, starting from a copy of its
adaptive grid. The estimate and uncertainty use only the samples from this
call. The number of bins comes from the grid. If supplied, `nbins` must match
that number. Other keywords have the same meaning as in [`vegas`](@ref).
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
    x, delx = Matrix{Float64}(edges), Matrix{Float64}(widths)
    all(isfinite, x) && all(isfinite, delx) ||
        throw(ArgumentError("the previous grid must contain finite edges and widths"))
    rtol = eltype(widths) in (Float16, Float32) ? 2eps(eltype(widths)) : 2eps(eltype(delx))
    for dim in axes(x, 2)
        _canonicalize_grid_axis!(view(x, :, dim), view(delx, :, dim), rtol) ||
            throw(ArgumentError("the previous grid has inconsistent edges and widths in dimension $dim"))
    end
    return x, delx
end

function _canonicalize_grid_axis!(edges, widths, rtol)
    a, b = first(edges), last(edges)
    span = b - a
    isfinite(span) || return false
    width = first(widths)
    if all(==(width), widths) && any(T -> _matches_uniform_grid(edges, width, T), (eltype(edges), Float32, Float16))
        _set_uniform_grid_axis!(edges, widths, a, b)
        return true
    end
    direction = sign(span)
    for i in eachindex(widths)
        actual = edges[i + 1] - edges[i]
        (iszero(actual) || sign(actual) == direction) && isapprox(actual, widths[i]; rtol) || return false
        widths[i] = actual
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
