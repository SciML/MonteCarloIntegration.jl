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

struct _IterationError
    mantissa::Float64
    exponent::Int
end

struct _IterationMean
    high::Float64
    low::Float64
end

_rounded_mean(mean::_IterationMean) = mean.high
_rounded_mean(mean::Float64) = mean
_rounded_mean(mean::Rational{BigInt}) = _rounded_rational(mean)
_mean_low(mean::_IterationMean) = mean.low
_mean_low(::Float64) = 0.0

_exact_mean(mean::_IterationMean) = Rational{BigInt}(mean.high) + Rational{BigInt}(mean.low)

Base.iszero(error::_IterationError) = iszero(error.mantissa)
_rounded_error(error::_IterationError) = ldexp(error.mantissa, error.exponent)
_rounded_error(error::Float64) = error

function _scaled_error(factor, exponent)
    mantissa, shift = frexp(factor)
    return _IterationError(mantissa, exponent + shift)
end

function _iteration_estimate(values)
    estimate, error, _ = _iteration_moments(values, false)
    return _rounded_mean(estimate), _rounded_error(error)
end

_iteration_estimate(values::Vector{_IterationMean}) = _iteration_estimate(map(_rounded_mean, values))

function _two_sum(a, b)
    high = a + b
    isfinite(high) || return high, 0.0
    rounded_b = high - a
    low = (a - (high - rounded_b)) + (b - rounded_b)
    return high, low
end

_needs_extended_error(error::_IterationError) =
    !iszero(error) && error.exponent < exponent(floatmin(Float64)) + 1

function _iteration_moments(values, preserve_mean = true)
    lo, hi = extrema(values)
    magnitude = max(abs(lo), abs(hi))
    isfinite(magnitude) || throw(DomainError(magnitude, "VEGAS requires finite weighted integrand values"))
    lo == hi && return _IterationMean(lo, 0.0), _IterationError(0.0, 0), nothing, nothing
    shift = exponent(magnitude)
    scale = ldexp(1.0, shift)
    mixed_signs = lo < 0 < hi
    origin = mixed_signs ? 0.0 : first(values) / scale
    n = length(values)
    shifted(v) = v / scale - origin
    offset = (mixed_signs ? _compensated_sum(shifted, values) : sum(shifted, values)) / n
    residual = sum(v -> abs2(shifted(v) - offset), values)
    error = _scaled_error(sqrt(residual / n / (n - 1)), shift)
    if preserve_mean && _needs_extended_error(error)
        mean, variance = _exact_binary64_moments(values, n)
        return _IterationMean(_rounded_rational(mean), 0.0), error, mean, variance
    end
    high, low = _two_sum(origin, offset)
    bounded = clamp(high, lo / scale, hi / scale)
    mean = scale * bounded
    tail = preserve_mean && bounded == high ? scale * low : 0.0
    if preserve_mean && (abs(mean) <= floatmin(Float64) || (!iszero(low) && abs(tail) < floatmin(Float64)))
        total = _binary64_sum_units(values)
        exact = total // (BigInt(n) << 1074)
        rounded = _round_binary64_units(total, n)
        exact == Rational{BigInt}(rounded) && return _IterationMean(rounded, 0.0), error, nothing, nothing
        return _IterationMean(rounded, 0.0), error, exact, nothing
    end
    if mixed_signs && any(v -> !iszero(v) && abs(v / scale) < floatmin(Float64), values)
        total = _binary64_sum_units(values)
        mean = _round_binary64_units(total, n)
        exact = preserve_mean ? total // (BigInt(n) << 1074) : nothing
        return _IterationMean(mean, 0.0), error, exact, nothing
    end
    return _IterationMean(mean, tail), error, nothing, nothing
end

function _binary64_units(value)
    mantissa, exponent = frexp(value)
    return BigInt(Int64(ldexp(mantissa, 53))) << (exponent + 1074 - 53)
end

_binary64_sum_units(values) = sum(_binary64_units, values; init = big(0))

function _exact_binary64_moments(values, n)
    # Accumulate on the lattice of the smallest positive Float64.
    total = squares = big(0)
    for value in values
        units = _binary64_units(value)
        total += units
        squares += abs2(units)
    end
    count = BigInt(n)
    mean = total // (count << 1074)
    variance = (count * squares - abs2(total)) // ((count^2 * (n - 1)) << 2148)
    return mean, variance
end

function _round_binary64_units(total, count)
    magnitude = abs(total)
    shift = max(0, ndigits(div(magnitude, count); base = 2) - 53)
    divisor = BigInt(count) << shift
    significand, remainder = divrem(magnitude, divisor)
    round_up = 2remainder > divisor || (2remainder == divisor && isodd(significand))
    rounded = ldexp(Float64(significand + round_up), shift - 1074)
    return total < 0 ? -rounded : rounded
end

_rounded_rational(value) = _round_binary64_units(numerator(value) << 1074, denominator(value))

struct _RoundedIterationErrors <: AbstractVector{Float64}
    values::Vector{_IterationError}
end

Base.IndexStyle(::Type{_RoundedIterationErrors}) = IndexLinear()
Base.size(errors::_RoundedIterationErrors) = size(errors.values)
Base.getindex(errors::_RoundedIterationErrors, i::Int) = _rounded_error(errors.values[i])

function _record_iteration_mean!(estimates, exact_means, mean, exact_mean)
    if exact_means === nothing && exact_mean !== nothing
        exact_means = map(_exact_mean, estimates)
    end
    push!(estimates, mean)
    if exact_means !== nothing
        push!(exact_means, exact_mean === nothing ? _exact_mean(mean) : exact_mean)
    end
    return exact_means
end

function _record_iteration_variance!(errors, exact_variances, error, exact_variance)
    if exact_variances === nothing && exact_variance !== nothing
        exact_variances = map(_exact_error_variance, errors)
    end
    push!(errors, error)
    if exact_variances !== nothing
        push!(exact_variances, exact_variance === nothing ? _exact_error_variance(error) : exact_variance)
    end
    return exact_variances
end

function _combine_estimates(estimates, errors::Vector{_IterationError}, exact_means = nothing, exact_variances = nothing)
    if exact_means === nothing && exact_variances === nothing && !any(_needs_extended_error, errors) &&
            !(any(iszero, errors) && any(m -> !iszero(_mean_low(m)), estimates))
        return _combine_estimates(estimates, _RoundedIterationErrors(errors))
    end
    means = exact_means === nothing ? map(_exact_mean, estimates) : exact_means
    variances = exact_variances === nothing ? map(_exact_error_variance, errors) : exact_variances
    return _combine_extended_estimates(means, variances)
end

function _exact_error_variance(error::_IterationError)
    mantissa = Rational{BigInt}(error.mantissa)
    power = big(1) << abs(error.exponent)
    return abs2(error.exponent < 0 ? mantissa / power : mantissa * power)
end

function _combine_extended_estimates(means, variances)
    weights = map(v -> iszero(v) ? v : inv(v), variances)
    used = count(!iszero, weights)
    n = length(weights)
    n == 1 && return only(means), only(variances), 0.0
    chi_squared = 0.0
    if used > 0
        total_weight = sum(weights)
        weighted_mean = sum(i -> weights[i] * means[i], eachindex(weights)) / total_weight
        if used > 1
            chi_squared = _rounded_rational(sum(i -> weights[i] * abs2(means[i] - weighted_mean), eachindex(weights)) / (used - 1))
        end
        used == n && return weighted_mean, inv(total_weight), chi_squared
    end
    mean = sum(means) / n
    scatter = sum(m -> abs2(m - mean), means) / n / (n - 1)
    within = sum(variances) / n / n
    return mean, max(scatter, within), chi_squared
end

function _rounded_error(variance::Rational{BigInt})
    # Compare with the squared midpoint on the Float64 lattice before rounding.
    scaled = numerator(variance) << 2148
    divisor = denominator(variance)
    floor_error = isqrt(div(scaled, divisor))
    shift = max(0, ndigits(floor_error; base = 2) - 53)
    significand = floor_error >> shift
    midpoint = (2significand + 1) << shift
    comparison = cmp(4scaled, divisor * midpoint^2)
    round_up = comparison > 0 || (comparison == 0 && isodd(significand))
    return ldexp(Float64(significand + round_up), shift - 1074)
end

_meets_error_tolerances(error::Float64, mean, rtol, atol) =
    0.0 < error < atol && error < rtol * abs(mean)

_rational_tolerance(x::Union{Integer, Rational, AbstractFloat}) = Rational{BigInt}(x)
_rational_tolerance(x) = Rational{BigInt}(Float64(x))

function _meets_error_tolerances(variance::Rational{BigInt}, mean, rtol, atol)
    variance > 0 && atol > 0 && rtol > 0 && !iszero(mean) || return false
    isinf(atol) || variance < abs2(_rational_tolerance(atol)) || return false
    return isinf(rtol) || variance < abs2(_rational_tolerance(rtol) * Rational{BigInt}(mean))
end

function _combine_estimates(estimates, errors)
    n = length(errors)
    n == 1 && return _rounded_mean(first(estimates)), first(errors), 0.0
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
    lo, hi = extrema(_rounded_mean, estimates)
    origin = (0 < hi / 2 <= lo || hi <= lo / 2 < 0) ? _rounded_mean(first(estimates)) : 0.0
    offset = _compensated_sum(eachindex(errors)) do i
        iszero(errors[i]) && return 0.0
        ratio = min_error / errors[i]
        return ((_rounded_mean(estimates[i]) - origin) * ratio) * (ratio / weight_sum)
    end
    low = _compensated_sum(eachindex(errors)) do i
        iszero(errors[i]) && return 0.0
        ratio = min_error / errors[i]
        return (_mean_low(estimates[i]) * ratio) * (ratio / weight_sum)
    end
    high, tail = _two_sum(origin, offset)
    mean = clamp(high + (tail + low), lo, hi)
    error = min_error / sqrt(weight_sum)
    chi_squared = used > 1 ? _reduced_chi_squared(estimates, errors, min_error, weight_sum, used) : 0.0
    return mean, error, chi_squared
end

function _reduced_chi_squared(estimates, errors, min_error, weight_sum, used)
    pivot = estimates[findfirst(==(min_error), errors)]
    function standardized(i)
        mean = estimates[i]
        high = _rounded_mean(mean) - _rounded_mean(pivot)
        low = _mean_low(mean) - _mean_low(pivot)
        delta = high + low
        return isfinite(delta) ? delta / errors[i] :
            (_rounded_mean(mean) / errors[i] - _rounded_mean(pivot) / errors[i]) + low / errors[i]
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
