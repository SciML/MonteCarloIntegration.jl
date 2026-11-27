function _initialize_vegas_grid(lb, ub, nbins, ::Nothing)
    length(lb) == length(ub) || throw(DimensionMismatch("bounds must have the same length"))
    isempty(lb) && throw(ArgumentError("the integration box needs at least one dimension"))
    grid = Matrix{Float64}(undef, nbins + 1, length(lb))
    widths = Matrix{Float64}(undef, nbins, length(lb))
    for dim in eachindex(lb, ub)
        a, b = Float64(lb[dim]), Float64(ub[dim])
        isfinite(a) && isfinite(b) || throw(ArgumentError("bounds must be finite"))
        _set_uniform_grid_axis!(view(grid, :, dim), view(widths, :, dim), a, b)
    end
    return grid, widths
end

function _adapt_grid!(grid, widths, bins, values, mass, counts, scratch, alpha)
    nbins, ndim = size(mass)
    (iszero(alpha) || nbins == 1) && return
    scale = maximum(abs, values)
    iszero(scale) && return
    _bin_means!(mass, counts, bins, values, scale)
    for dim in 1:ndim
        _smooth_weights!(scratch, view(mass, :, dim), alpha)
        _rebin!(view(grid, :, dim), view(widths, :, dim), scratch, view(mass, :, dim))
    end
    return nothing
end

function _bin_means!(mass, counts, bins, values, scale)
    fill!(mass, 0)
    fill!(counts, 0)
    for j in eachindex(values)
        squared = abs2(values[j] / scale)
        for dim in axes(bins, 1)
            bin = bins[dim, j]
            mass[bin, dim] += squared
            counts[bin, dim] += 1
        end
    end
    for i in eachindex(mass, counts)
        mass[i] /= max(1, counts[i])
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
    proposed = zeros(size(weights, 1))
    for dim in axes(weights, 2)
        _rebin!(view(new_edges, :, dim), view(new_widths, :, dim), view(weights, :, dim), proposed)
    end
    return new_edges, new_widths
end
