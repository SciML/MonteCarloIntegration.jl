function _compensated_sum(f, values)
    total = correction = 0.0
    for value in values
        term = f(value)
        next = total + term
        correction += abs(total) >= abs(term) ? (total - next) + term : (term - next) + total
        total = next
    end
    return total + correction
end

function _iteration_estimate(values)
    lo, hi = extrema(values)
    magnitude = max(abs(lo), abs(hi))
    isfinite(magnitude) || throw(DomainError(magnitude, "VEGAS requires finite weighted integrand values"))
    lo == hi && return (lo, 0.0)
    scale = ldexp(1.0, exponent(magnitude))
    mixed_signs = lo < 0 < hi
    origin = mixed_signs ? 0.0 : first(values) / scale
    n = length(values)
    shifted(v) = v / scale - origin
    offset = (mixed_signs ? _compensated_sum(shifted, values) : sum(shifted, values)) / n
    residual = sum(v -> abs2(shifted(v) - offset), values)
    mean = scale * clamp(origin + offset, lo / scale, hi / scale)
    if mixed_signs && any(v -> !iszero(v) && abs(v / scale) < floatmin(Float64), values)
        mean = _exact_binary64_mean(values)
    end
    return mean, scale * sqrt(residual / n / (n - 1))
end

function _exact_binary64_mean(values)
    total = big(0)
    for value in values
        mantissa, exponent = frexp(value)
        # Accumulate integer multiples of the smallest positive Float64.
        total += BigInt(Int64(ldexp(mantissa, 53))) << (exponent + 1074 - 53)
    end
    magnitude = abs(total)
    n = length(values)
    shift = max(0, ndigits(div(magnitude, n); base = 2) - 53)
    divisor = BigInt(n) << shift
    significand, remainder = divrem(magnitude, divisor)
    round_up = 2remainder > divisor || (2remainder == divisor && isodd(significand))
    rounded = ldexp(Float64(significand + round_up), shift - 1074)
    return total < 0 ? -rounded : rounded
end

function _combine_estimates(estimates, errors)
    n = length(errors)
    n == 1 && return first(estimates), first(errors), 0.0
    used = count(!iszero, errors)
    if used < n
        mean, scatter = _iteration_estimate(estimates)
        scale = maximum(errors)
        iszero(scale) && return mean, scatter, 0.0
        within = scale * (sqrt(sum(e -> abs2(e / scale), errors)) / n)
        _, _, chi_squared = _inverse_variance_estimate(estimates, errors, used)
        return mean, max(scatter, within), chi_squared
    end
    return _inverse_variance_estimate(estimates, errors, used)
end

function _inverse_variance_estimate(estimates, errors, used)
    min_error = minimum(e -> iszero(e) ? Inf : e, errors)
    weight_sum = sum(e -> iszero(e) ? 0.0 : abs2(min_error / e), errors)
    lo, hi = extrema(estimates)
    origin = (0 < hi / 2 <= lo || hi <= lo / 2 < 0) ? first(estimates) : 0.0
    offset = _compensated_sum(eachindex(errors)) do i
        iszero(errors[i]) && return 0.0
        ratio = min_error / errors[i]
        return ((estimates[i] - origin) * ratio) * (ratio / weight_sum)
    end
    mean = clamp(origin + offset, lo, hi)
    error = min_error / sqrt(weight_sum)
    chi_squared = used > 1 ? _reduced_chi_squared(estimates, errors, min_error, weight_sum, used) : 0.0
    return mean, error, chi_squared
end

function _reduced_chi_squared(estimates, errors, min_error, weight_sum, used)
    pivot = estimates[findfirst(==(min_error), errors)]
    function standardized(i)
        delta = estimates[i] - pivot
        return isfinite(delta) ? delta / errors[i] : estimates[i] / errors[i] - pivot / errors[i]
    end
    # Center in units of the most precise iteration's error before rounding the mean.
    center = _compensated_sum(eachindex(errors)) do i
        iszero(errors[i]) && return 0.0
        return (min_error / errors[i] / weight_sum) * standardized(i)
    end
    isfinite(center) || return Inf
    denominator = sqrt(used - 1)
    return sum(eachindex(errors)) do i
        iszero(errors[i]) && return 0.0
        return abs2((standardized(i) - (min_error / errors[i]) * center) / denominator)
    end
end
