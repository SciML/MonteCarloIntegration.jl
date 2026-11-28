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
    mean, exact_average = _stratum_average(strata.means, normal_scale, length(values))
    if iszero(normal_scale)
        scale, scaled_variance = narrow_scale, narrow_variance
    else
        scale = _scaled_error(normal_scale, 0)
        scaled_variance = normal_variance + narrow_variance * abs2(_stratum_error_ratio(narrow_scale, scale))
    end
    error = _scaled_error(scale.mantissa * (sqrt(scaled_variance) / nstrata), scale.exponent)
    exact_variance = _needs_extended_error(error)
    if exact_variance || exact_mean || exact_average || _needs_exact_stratum_average(mean, strata.means)
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
    groups = Dict{Int, Tuple{BigInt, BigInt, Int}}()
    native_groups = Dict{Tuple{Int, Int}, Tuple{Int128, UInt128}}()
    nstrata = length(counts)
    first = 1
    for count in counts
        last = first + count - 1
        cell = @view values[first:last]
        # The bound leaves 106 bits for each squared binary64 significand.
        native = count <= (1 << 11) && nstrata <= (1 << 22) ÷ count^2 ? _homogeneous_stratum_sums(cell) : nothing
        if native === nothing
            total, squares, exponent = _binary64_sums(cell)
            _merge_stratum_moments!(groups, count, total, count * squares - abs2(total), exponent)
        else
            total, squares, exponent = native
            variance = UInt128(count) * squares - abs2(UInt128(abs(total)))
            key = (count, exponent)
            group_sum, group_variance = get(native_groups, key, (Int128(0), UInt128(0)))
            native_groups[key] = (group_sum + total, group_variance + variance)
        end
        first = last + 1
    end
    for ((count, exponent), (total, variance)) in native_groups
        _merge_stratum_moments!(groups, count, BigInt(total), BigInt(variance), exponent)
    end
    total = variance_total = big(0) // 1
    for (count, (group_sum, group_variance, exponent)) in groups
        total += _scaled_rational(group_sum, BigInt(count), exponent)
        variance_total += _scaled_rational(group_variance, BigInt(count)^2 * (count - 1), 2exponent)
    end
    return total / nstrata, variance_total / BigInt(nstrata)^2
end

function _homogeneous_stratum_sums(values)
    exponent = (reinterpret(UInt64, first(values)) >> 52) & 0x07ff
    leading = iszero(exponent) ? UInt64(0) : UInt64(1) << 52
    total, squares = Int128(0), UInt128(0)
    for value in values
        word = reinterpret(UInt64, value)
        (word >> 52) & 0x07ff == exponent || return nothing
        significand = (word & 0x000fffffffffffff) | leading
        total += signbit(value) ? -Int128(significand) : Int128(significand)
        squares += widemul(significand, significand)
    end
    return total, squares, max(1, Int(exponent)) - 1075
end

function _merge_stratum_moments!(groups, count, total, variance, exponent)
    previous = get(groups, count, nothing)
    if previous === nothing
        groups[count] = (total, variance, exponent)
        return
    end
    group_sum, group_variance, group_exponent = previous
    shift = exponent - group_exponent
    if shift > 0
        total <<= shift
        variance <<= 2shift
    elseif shift < 0
        group_sum <<= -shift
        group_variance <<= -2shift
    end
    groups[count] = (group_sum + total, group_variance + variance, min(group_exponent, exponent))
    return
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

struct _DeferredStratumIteration
    counts::Vector{Int}
    values::Vector{Float64}
    zero_variance::Bool
end

struct _DeferredStratumHistory
    iterations::Vector{_DeferredStratumIteration}
    capacity::Int
end

function _deferred_stratum_history(counts, values)
    capacity = (1 << 20) ÷ (sizeof(counts) + sizeof(values))
    return _DeferredStratumHistory(_DeferredStratumIteration[], capacity)
end

function _stratified_sample_moments(counts, values, zero_variance, exact_mean = nothing, exact_variance = nothing)
    exact_variance === nothing || return exact_mean, exact_variance
    zero_variance || return _exact_stratum_moments(counts, values, true)
    if exact_mean === nothing
        total = big(0)
        first = 1
        for count in counts
            total += _binary64_units(values[first])
            first += count
        end
        exact_mean = _scaled_rational(total, BigInt(length(counts)), -1074)
    end
    return exact_mean, zero(Rational{BigInt})
end

function _record_stratified_moments!(history, pending, counts, values, zero_variance, exact_mean, exact_variance, need_exact)
    if !need_exact && isempty(history.means) && length(pending.iterations) < pending.capacity
        push!(pending.iterations, _DeferredStratumIteration(copy(counts), copy(values), zero_variance))
        return
    end
    for saved in pending.iterations
        mean, variance = _stratified_sample_moments(saved.counts, saved.values, saved.zero_variance)
        push!(history.means, mean)
        push!(history.variances, variance)
    end
    empty!(pending.iterations)
    mean, variance = _stratified_sample_moments(counts, values, zero_variance, exact_mean, exact_variance)
    push!(history.means, mean)
    push!(history.variances, variance)
    return
end

function _stratum_total(means, scale)
    total = correction = magnitude = 0.0
    for mean in means
        high = _rounded_mean(mean)
        magnitude = max(magnitude, abs(high))
        total, residual = _two_sum(total, high / scale)
        correction += residual + _mean_low(mean) / scale
    end
    return total, correction, magnitude
end

function _stratum_average(means, max_stderr = 0.0, nsamples = 0)
    n = length(means)
    scale = 1.0
    total, correction, magnitude = _stratum_total(means, scale)
    if !isfinite(total + correction)
        scale = exp2(ceil(log2(n)) + 1)
        total, correction, magnitude = _stratum_total(means, scale)
    end
    average = total / n
    residual = fma(-average, Float64(n), total) + correction
    adjustment = residual / n
    high, low = _two_sum(average, adjustment)
    rounded = high * scale
    # Each sample can differ from its cell mean by at most (count - 1) standard errors.
    cutoff = max((8eps(Float64) * n) * magnitude, (8eps(Float64) * nsamples) * max_stderr)
    cancellation = 0 < cutoff && abs(rounded) <= cutoff
    tiny_tail = !iszero(residual) && abs(adjustment * scale) < floatmin(Float64)
    return _IterationMean(rounded, low * scale), cancellation || tiny_tail
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
