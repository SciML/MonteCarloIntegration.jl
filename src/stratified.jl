struct _VegasStrata
    splits::Vector{Int}
    counts::Vector{Int}
    means::Vector{Float64}
    scores::Vector{Float64}
    training_weights::Vector{Float64}
end

function _stratum_count(splits, ndim, limit)
    count = 1
    for _ in 1:ndim
        count > limit ÷ splits && return 0
        count *= splits
    end
    return count
end

function _VegasStrata(ndim, ncalls)
    limit = max(1, ncalls ÷ 4)
    low, high = 1, limit
    while low < high
        mid = low + cld(high - low, 2)
        if _stratum_count(mid, ndim, limit) > 0
            low = mid
        else
            high = mid - 1
        end
    end

    splits = fill(low, ndim)
    count = _stratum_count(low, ndim, limit)
    for dim in eachindex(splits)
        other = count ÷ low
        other > limit ÷ (low + 1) && break
        splits[dim] += 1
        count = other * splits[dim]
    end

    strata = _VegasStrata(splits, Vector{Int}(undef, count), zeros(count), zeros(count), Vector{Float64}(undef, ncalls))
    _uniform_stratum_counts!(strata.counts, ncalls)
    return strata
end

function _uniform_stratum_counts!(counts, ncalls)
    base, extra = divrem(ncalls, length(counts))
    for i in eachindex(counts)
        counts[i] = base + (i <= extra)
    end
    return counts
end

function _sample_strata!(ymat, strata)
    rand!(ymat)
    length(strata.counts) == 1 && return ymat
    first = 1
    @inbounds for i in eachindex(strata.counts)
        last = first + strata.counts[i] - 1
        cell = i - 1
        for dim in eachindex(strata.splits)
            splits = strata.splits[dim]
            splits == 1 && continue
            cell, offset = divrem(cell, splits)
            limit = prevfloat((offset + 1) / splits)
            for j in first:last
                ymat[dim, j] = min((offset + ymat[dim, j]) / splits, limit)
            end
        end
        first = last + 1
    end
    return ymat
end

function _stratified_estimate!(strata, values)
    first = 1
    scale = 0.0
    scaled_variance = 0.0
    @inbounds for i in eachindex(strata.counts)
        count = strata.counts[i]
        last = first + count - 1
        strata.means[i], error = _iteration_estimate(@view values[first:last])
        strata.scores[i] = log(error) + 0.5 * log(count)
        if error > scale
            scaled_variance = scaled_variance * abs2(scale / error) + 1
            scale = error
        elseif error > 0
            scaled_variance += abs2(error / scale)
        end
        first = last + 1
    end

    nstrata = length(strata.counts)
    return _stratum_average(strata.means), scale * (sqrt(scaled_variance) / nstrata)
end

function _stratum_average(values)
    n = length(values)
    total = _compensated_sum(identity, values)
    isfinite(total) && return total / n

    scale = exp2(ceil(log2(n)) + 1)
    scaled_total = _compensated_sum(x -> x / scale, values)
    return clamp(scaled_total * (scale / n), extrema(values)...)
end

function _stratified_training_weights!(strata)
    weight_total = length(strata.training_weights) / length(strata.counts)
    first = 1
    @inbounds for count in strata.counts
        last = first + count - 1
        weight = weight_total / count
        for j in first:last
            strata.training_weights[j] = weight
        end
        first = last + 1
    end
    return strata.training_weights
end

function _redistribute_strata!(strata)
    ncalls = length(strata.training_weights)
    largest_score = maximum(strata.scores)
    if largest_score == -Inf
        return _uniform_stratum_counts!(strata.counts, ncalls)
    end

    # Round cumulative quotas so the reserved and adaptive samples exhaust the budget.
    cumulative = 0.0
    @inbounds for i in eachindex(strata.scores)
        cumulative += exp(0.75 * (strata.scores[i] - largest_score))
        strata.scores[i] = cumulative
    end
    available = ncalls - 2 * length(strata.counts)
    assigned = 0
    @inbounds for i in eachindex(strata.counts)
        target = round(Int, (strata.scores[i] / cumulative) * available)
        strata.counts[i] = 2 + (target - assigned)
        assigned = target
    end
    return strata.counts
end
