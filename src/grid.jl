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
