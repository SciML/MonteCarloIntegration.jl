struct _VegasStrata
    splits::Vector{Int}
    counts::Vector{Int}
    means::Vector{_IterationMean}
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

    strata = _VegasStrata(
        splits, Vector{Int}(undef, count), Vector{_IterationMean}(undef, count), zeros(count),
        Vector{Float64}(undef, ncalls)
    )
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
            limit = prevfloat(eltype(ymat)(offset + 1) / splits)
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
    normal_scale = normal_variance = narrow_variance = 0.0
    narrow_scale = _IterationError(0.0, 0)
    exact_mean = false
    @inbounds for i in eachindex(strata.counts)
        count = strata.counts[i]
        last = first + count - 1
        cell_values = @view values[first:last]
        mean, error, unrounded = _iteration_moments(cell_values)
        strata.means[i] = mean
        exact_mean |= unrounded !== nothing
        if iszero(error)
            strata.scores[i] = -Inf
        elseif _needs_extended_error(error)
            strata.scores[i] = log(error.mantissa) + error.exponent * log(2.0) + 0.5 * log(count)
            if iszero(narrow_scale) || error.exponent > narrow_scale.exponent ||
                    (error.exponent == narrow_scale.exponent && error.mantissa > narrow_scale.mantissa)
                narrow_variance = narrow_variance * abs2(_stratum_error_ratio(narrow_scale, error)) + 1
                narrow_scale = error
            else
                narrow_variance += abs2(_stratum_error_ratio(error, narrow_scale))
            end
        else
            stderr = _rounded_error(error)
            strata.scores[i] = log(stderr) + 0.5 * log(count)
            if stderr > normal_scale
                normal_variance = normal_variance * abs2(normal_scale / stderr) + 1
                normal_scale = stderr
            else
                normal_variance += abs2(stderr / normal_scale)
            end
        end
        first = last + 1
    end

    nstrata = length(strata.counts)
    mean, tiny_tail = _stratum_average(strata.means)
    if iszero(normal_scale)
        scale, scaled_variance = narrow_scale, narrow_variance
    else
        scale = _scaled_error(normal_scale, 0)
        scaled_variance = normal_variance + narrow_variance * abs2(_stratum_error_ratio(narrow_scale, scale))
    end
    error = _scaled_error(scale.mantissa * (sqrt(scaled_variance) / nstrata), scale.exponent)
    exact_variance = _needs_extended_error(error)
    if exact_variance || exact_mean || tiny_tail || _needs_exact_stratum_average(mean, strata.means)
        unrounded, variance = _exact_stratum_moments(strata.counts, values, exact_variance)
        return _IterationMean(_rounded_mean(unrounded), 0.0), error, unrounded, variance
    end
    return mean, error, nothing, nothing
end

_stratum_error_ratio(small, large) =
    ldexp(small.mantissa / large.mantissa, small.exponent - large.exponent)

function _needs_exact_stratum_average(mean, values)
    high = _rounded_mean(mean)
    isfinite(high) || return true
    abs(high) <= floatmin(Float64) || return false
    !iszero(high) && return true
    # A nonzero sum on a coarser lattice cannot round to zero after division by the count.
    cutoff = length(values) * floatmin(Float64)
    return any(values) do value
        0 < abs(_rounded_mean(value)) <= cutoff || !iszero(_mean_low(value))
    end
end

function _exact_stratum_moments(counts, values, exact_variance)
    exact_variance || return _exact_stratum_mean(counts, values), nothing
    total = variance_total = big(0) // 1
    first = 1
    for count in counts
        last = first + count - 1
        cell = @view values[first:last]
        mean, variance = _exact_binary64_moments(cell, count)
        total += mean
        variance_total += variance
        first = last + 1
    end
    nstrata = length(counts)
    return total / nstrata, variance_total / BigInt(nstrata)^2
end

function _exact_stratum_mean(counts, values)
    if all(==(first(counts)), counts)
        return _binary64_sum_units(values) // (BigInt(length(values)) << 1074)
    end
    # Apply the binary64 lattice spacing once after summing the cell means.
    total = big(0) // 1
    first_index = 1
    for count in counts
        last = first_index + count - 1
        total += _binary64_sum_units(@view values[first_index:last]) // count
        first_index = last + 1
    end
    return numerator(total) // ((denominator(total) * length(counts)) << 1074)
end

function _stratum_total(means, scale)
    total = correction = 0.0
    for mean in means
        total, residual = _two_sum(total, _rounded_mean(mean) / scale)
        correction += residual + _mean_low(mean) / scale
    end
    return total, correction
end

function _stratum_average(means)
    n = length(means)
    scale = 1.0
    total, correction = _stratum_total(means, scale)
    if !isfinite(total + correction)
        scale = exp2(ceil(log2(n)) + 1)
        total, correction = _stratum_total(means, scale)
    end
    average = total / n
    residual = fma(-average, Float64(n), total) + correction
    adjustment = residual / n
    high, low = _two_sum(average, adjustment)
    tail = low * scale
    tiny_tail = !iszero(residual) && abs(adjustment * scale) < floatmin(Float64)
    return _IterationMean(high * scale, tail), tiny_tail
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
