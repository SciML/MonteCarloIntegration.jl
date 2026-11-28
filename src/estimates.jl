function _compensated_parts(f, values)
    total = correction = 0.0
    for value in values
        term = f(value)
        next = total + term
        correction += abs(total) >= abs(term) ? (total - next) + term : (term - next) + total
        total = next
    end
    return total, correction
end

function _compensated_sum(f::F, values) where {F}
    total, tail = _compensated_parts(f, values)
    return total + tail
end

function _compensated_average(f::F, values, n) where {F}
    sum, correction = _compensated_parts(f, values)
    total, tail = _two_sum(sum, correction)
    average = total / n
    return _two_sum(average, (fma(-average, Float64(n), total) + tail) / n)
end

# Sterbenz's lemma makes subtraction of the first value exact throughout this range.
_can_center(lo, hi) = 0 < hi / 2 <= lo || hi <= lo / 2 < 0

function _at_rounding_midpoint(value, tail)
    iszero(tail) && return false
    spacing = tail > 0 ? nextfloat(value) - value : value - prevfloat(value)
    return 2abs(tail) == spacing
end

function _exact_power_of_two_average(values, scale)
    ispow2(length(values)) || return false
    total = correction = 0.0
    for value in values
        term = value / scale
        term * scale == value || return false
        total, residual = _two_sum(total, term)
        correction, lost = _two_sum(correction, residual)
        iszero(lost) || return false
    end
    return true
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

function _iteration_estimate(values::Vector{_IterationMean})
    lo, hi = extrema(_rounded_mean, values)
    magnitude = max(abs(lo), abs(hi))
    iszero(magnitude) && return 0.0, 0.0
    scale = ldexp(1.0, exponent(magnitude))
    mixed_signs = lo < 0 < hi
    origin = _can_center(lo, hi) ? first(values).high / scale : 0.0
    shifted(value) = value.high / scale - origin
    tail(value) = value.low / scale
    n = length(values)
    offset, low = if iszero(origin)
        parts = Iterators.flatten(((shifted(v), tail(v)) for v in values))
        _compensated_average(identity, parts, n)
    else
        _compensated_sum(shifted, values) / n, _compensated_sum(tail, values) / n
    end
    high, correction = _two_sum(origin, offset)
    mean = scale * clamp(high + (correction + low), lo / scale, hi / scale)
    midpoint = iszero(origin) && _at_rounding_midpoint(high, correction + low)
    cancellation = mixed_signs && (
        abs(offset) <= 8eps(Float64) * n || any(
            v -> (!iszero(v.high) && abs(v.high / scale) < floatmin(Float64)) ||
                (!iszero(v.low) && abs(tail(v)) < floatmin(Float64)), values
        )
    )
    if midpoint || cancellation
        total = sum(v -> _binary64_units(v.high) + _binary64_units(v.low), values; init = big(0))
        mean = _round_binary64_units(total, n)
    end
    residual = sum(v -> abs2((shifted(v) - offset) + (tail(v) - low)), values)
    return mean, scale * sqrt(residual / n / (n - 1))
end

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
    origin = _can_center(lo, hi) ? first(values) / scale : 0.0
    n = length(values)
    shifted(v) = v / scale - origin
    offset, adjustment = iszero(origin) ? _compensated_average(shifted, values, n) : (sum(shifted, values) / n, 0.0)
    residual = sum(v -> abs2(shifted(v) - offset), values)
    error = _scaled_error(sqrt(residual / n / (n - 1)), shift)
    if preserve_mean && _needs_extended_error(error)
        mean, variance = _exact_binary64_moments(values, n)
        return _IterationMean(_rounded_rational(mean), 0.0), error, mean, variance
    end
    high, low = iszero(origin) ? (offset, adjustment) : _two_sum(origin, offset)
    bounded = clamp(high, lo / scale, hi / scale)
    mean = scale * bounded
    tail = preserve_mean && bounded == high ? scale * low : 0.0
    cancellation = mixed_signs && abs(offset) <= 8eps(Float64) * n
    midpoint = iszero(origin) && _at_rounding_midpoint(high, low) &&
        !_exact_power_of_two_average(values, scale)
    if midpoint || cancellation || (preserve_mean && (abs(mean) <= floatmin(Float64) || (!iszero(low) && abs(tail) < floatmin(Float64))))
        total = _binary64_sum_units(values)
        rounded = _round_binary64_units(total, n)
        preserve_mean || return _IterationMean(rounded, 0.0), error, nothing, nothing
        if midpoint && rounded == mean && total == n * (_binary64_units(mean) + _binary64_units(tail))
            return _IterationMean(mean, tail), error, nothing, nothing
        end
        exact = total // (BigInt(n) << 1074)
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

function _binary64_sums(values)
    index = findfirst(!iszero, values)
    isnothing(index) && return big(0), big(0), 0
    first_word = reinterpret(UInt64, values[index])
    first_exponent = Int((first_word >> 52) & 0x07ff)
    count_bits = 8sizeof(Int) - leading_zeros(length(values) - 1)
    max_shift = min(10, fld(22 - count_bits, 2))
    if max_shift >= 0 && first_exponent > 0
        base = max(1, first_exponent - cld(max_shift, 2))
        total, squares = Int128(0), UInt128(0)
        fits = true
        for value in values
            iszero(value) && continue
            word = reinterpret(UInt64, value)
            offset = Int((word >> 52) & 0x07ff) - base
            if !(0 <= offset <= max_shift)
                fits = false
                break
            end
            mantissa = ((word & 0x000fffffffffffff) | (UInt64(1) << 52)) << offset
            total += signbit(value) ? -Int128(mantissa) : Int128(mantissa)
            squares += widemul(mantissa, mantissa)
        end
        fits && return BigInt(total), BigInt(squares), base - 1075
    end

    sums = zeros(Int128, 33)
    squares = zeros(UInt128, 66)
    minimum_bit = 2045
    for value in values
        iszero(value) && continue
        word = reinterpret(UInt64, value)
        exponent = Int((word >> 52) & 0x07ff)
        frac = word & 0x000fffffffffffff
        mantissa = iszero(exponent) ? frac : frac | (UInt64(1) << 52)
        bit = max(0, exponent - 1)
        minimum_bit = min(minimum_bit, bit)
        i, shift = fldmod(bit, 64)
        sign = signbit(value) ? -1 : 1
        @inbounds begin
            sums[i + 1] += sign * Int128(mantissa << shift)
            sums[i + 2] += sign * Int128(mantissa >> (64 - shift))
        end

        square = widemul(mantissa, mantissa)
        low, high = square % UInt64, UInt64(square >> 64)
        j, shift = fldmod(2bit, 64)
        @inbounds begin
            squares[j + 1] += low << shift
            squares[j + 2] += (high << shift) | (low >> (64 - shift))
            squares[j + 3] += high >> (64 - shift)
        end
    end
    return _binary_limb_sum(sums, minimum_bit), _binary_limb_sum(squares, 2minimum_bit), minimum_bit - 1074
end

function _exact_binary64_moments(values, n)
    total, sum_squares, shift = _binary64_sums(values)
    count = BigInt(n)
    mean = _scaled_rational(total, count, shift)
    variance = _scaled_rational(count * sum_squares - abs2(total), count^2 * (n - 1), 2shift)
    return mean, variance
end

_scaled_rational(total, count, shift) = shift < 0 ? total // (count << -shift) : (total << shift) // count

function _binary_limb_sum(limbs, unit)
    # Each sample adds at most one 64-bit digit per limb, so an Int-sized
    # batch fits in the 128-bit accumulators without a carry in the sample loop.
    first = findfirst(!iszero, limbs)
    isnothing(first) && return big(0)
    last = findlast(!iszero, limbs)
    total = BigInt(limbs[last])
    for i in (last - 1):-1:first
        total = (total << 64) + limbs[i]
    end
    return total << (64 * (first - 1) - unit)
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

function _sample_moments(values, estimate, stderr, exact_mean, exact_variance)
    exact_variance === nothing || return exact_mean, exact_variance
    iszero(stderr) && return _exact_mean(estimate), zero(Rational{BigInt})
    return _exact_binary64_moments(values, length(values))
end

function _combine_extended_estimates(means, variances, history = nothing)
    bounds = history === nothing ? nothing : history.bounds
    if bounds !== nothing || !_small_odd_denominators(means, variances)
        return _combine_bounded_estimates(means, variances, history)
    end
    return _combine_exact_estimates(means, variances)
end

function _combine_exact_estimates(means, variances, calculate_chi_squared = true)
    used = count(!iszero, variances)
    n = length(variances)
    n == 1 && return only(means), only(variances), 0.0
    chi_squared = 0.0
    if used > 0 && (used == n || calculate_chi_squared)
        weights = map(v -> iszero(v) ? v : inv(v), variances)
        total_weight = sum(weights)
        weighted_mean = sum(i -> weights[i] * means[i], eachindex(weights)) / total_weight
        if used > 1 && calculate_chi_squared
            chi_squared = _rounded_rational(sum(i -> weights[i] * abs2(means[i] - weighted_mean), eachindex(weights)) / (used - 1))
        end
        used == n && return weighted_mean, inv(total_weight), chi_squared
    end
    mean = sum(means) / n
    scatter = sum(m -> abs2(m - mean), means) / n / (n - 1)
    within = sum(variances) / n / n
    return mean, max(scatter, within), chi_squared
end

_odd_part(x::BigInt) = x >> trailing_zeros(x)

function _small_odd_denominators(means, variances)
    denominator_lcm = big(1)
    for i in eachindex(means, variances)
        v = variances[i]
        denominator_lcm = lcm(denominator_lcm, _odd_part(denominator(means[i])), _odd_part(denominator(v)))
        iszero(v) || (denominator_lcm = lcm(denominator_lcm, _odd_part(numerator(v))))
        # Powers of two cannot introduce new denominator factors during summation.
        ndigits(denominator_lcm; base = 2) <= 256 || return false
    end
    return true
end

struct _BinarySumBounds
    lower::BigInt
    upper::BigInt
    exponent::Int
end

_BinarySumBounds() = _BinarySumBounds(big(0), big(0), typemax(Int))

function _add_binary_bounds(bounds, term, bits)
    iszero(term) && return bounds
    num, den = numerator(term), denominator(term)
    exponent = ndigits(abs(num); base = 2) - ndigits(den; base = 2) - bits
    lower, remainder = exponent < 0 ? fldmod(num << -exponent, den) : fldmod(num, den << exponent)
    upper = lower + !iszero(remainder)
    bounds.exponent == typemax(Int) && return _BinarySumBounds(lower, upper, exponent)
    common = min(exponent, bounds.exponent)
    # Align the two dyadic intervals and add their endpoints exactly.
    return _BinarySumBounds(
        (bounds.lower << (bounds.exponent - common)) + (lower << (exponent - common)),
        (bounds.upper << (bounds.exponent - common)) + (upper << (exponent - common)),
        common,
    )
end

_binary_rational(n, exponent) = exponent < 0 ? n // (big(1) << -exponent) : (n << exponent) // big(1)

function _rational_bounds(bounds::_BinarySumBounds)
    bounds.exponent == typemax(Int) && return big(0) // 1, big(0) // 1
    return _binary_rational(bounds.lower, bounds.exponent), _binary_rational(bounds.upper, bounds.exponent)
end

function _quotient_bounds(num, den)
    a, b = num
    c, d = den
    return a / (a < 0 ? c : d), b / (b < 0 ? d : c)
end

function _squared_bounds(bounds)
    a, b = bounds
    left, right = abs2(a), abs2(b)
    return a <= 0 <= b ? zero(left) : min(left, right), max(left, right)
end

struct _CenteredSums
    weights::_BinarySumBounds
    offsets::_BinarySumBounds
    squares::_BinarySumBounds
end

_CenteredSums() = _CenteredSums(_BinarySumBounds(), _BinarySumBounds(), _BinarySumBounds())

struct _BoundedHistory
    count::Int
    used::Int
    pivot::Int
    bits::Int
    weighted::_CenteredSums
    uniform::_CenteredSums
    within::_BinarySumBounds
end

mutable struct _ExactHistory
    means::Vector{Rational{BigInt}}
    variances::Vector{Rational{BigInt}}
    bounds::Union{Nothing, _BoundedHistory}
end

_ExactHistory() = _ExactHistory(Rational{BigInt}[], Rational{BigInt}[], nothing)

function _accumulate_centered(bounds, means, variances, pivot, scale, bits, start)
    (; weights, offsets, squares) = bounds
    for i in start:length(means)
        if variances === nothing
            ratio = one(pivot)
        else
            iszero(variances[i]) && continue
            ratio = scale / variances[i]
        end
        delta = means[i] - pivot
        offset = ratio * delta
        weights = _add_binary_bounds(weights, ratio, bits)
        offsets = _add_binary_bounds(offsets, offset, bits)
        squares = _add_binary_bounds(squares, offset * delta, bits)
    end
    return _CenteredSums(weights, offsets, squares)
end

function _centered_bounds(bounds::_CenteredSums, pivot)
    norm = _rational_bounds(bounds.weights)
    displacement = _rational_bounds(bounds.offsets)
    square = _rational_bounds(bounds.squares)
    center = _quotient_bounds(displacement, norm)
    correction = _quotient_bounds(_squared_bounds(displacement), norm)
    return (
        (pivot + center[1], pivot + center[2]), norm,
        (max(zero(square[1]), square[1] - correction[2]), max(zero(square[2]), square[2] - correction[1])),
    )
end

function _update_bounds(previous, means, variances, bits)
    n = length(means)
    old = previous === nothing || previous.bits != bits ?
        _BoundedHistory(0, 0, 0, bits, _CenteredSums(), _CenteredSums(), _BinarySumBounds()) : previous
    old.count == n && return old
    start = old.count + 1
    pivot = old.pivot
    used = old.used
    for i in start:n
        iszero(variances[i]) && continue
        used += 1
        # A retained pivot limits each relative weight to 16, bounding cancellation
        # in the centered scatter without rebuilding for each new minimum.
        (pivot == 0 || 16variances[i] < variances[pivot]) && (pivot = i)
    end
    weighted = old.weighted
    if used > 0
        if pivot == old.pivot
            weighted = _accumulate_centered(weighted, means, variances, means[pivot], variances[pivot], bits, start)
        else
            weighted = _accumulate_centered(_CenteredSums(), means, variances, means[pivot], variances[pivot], bits, 1)
        end
    end
    uniform, within = old.uniform, old.within
    if used < n
        first_unweighted = old.used == old.count ? 1 : start
        uniform = _accumulate_centered(uniform, means, nothing, first(means), nothing, bits, first_unweighted)
        for i in first_unweighted:n
            within = _add_binary_bounds(within, variances[i], bits)
        end
    end
    return _BoundedHistory(n, used, pivot, bits, weighted, uniform, within)
end

function _combination_bounds(bounds, means, variances)
    (; used, pivot) = bounds
    n = bounds.count
    chi_squared = (0, 0)
    if used > 0
        scale = variances[pivot]
        mean, norm, scatter = _centered_bounds(bounds.weighted, means[pivot])
        if used > 1
            chi_squared = scatter[1] / scale / (used - 1), scatter[2] / scale / (used - 1)
        end
        used == n && return mean, (scale / norm[2], scale / norm[1]), chi_squared
    end
    mean, _, scatter = _centered_bounds(bounds.uniform, first(means))
    a, b = _rational_bounds(bounds.within)
    variance = max(scatter[1] / n / (n - 1), a / n / n), max(scatter[2] / n / (n - 1), b / n / n)
    return mean, variance, chi_squared
end

_midpoint(bounds) = (bounds[1] + bounds[2]) / 2

const _BOUND_REFINEMENTS = (192, 384, 768, 1536, 3072, 6144)

struct _BoundedIterationError
    mean::NTuple{2, Rational{BigInt}}
    variance::NTuple{2, Rational{BigInt}}
    means::Vector{Rational{BigInt}}
    variances::Vector{Rational{BigInt}}
    bits::Int
end

_rounded_error(error::_BoundedIterationError) = _rounded_error(_midpoint(error.variance))

function _combine_bounded_estimates(means, variances, history)
    length(variances) == 1 && return only(means), only(variances), 0.0
    previous = history === nothing ? nothing : history.bounds
    bounds = _update_bounds(previous, means, variances, previous === nothing ? 96 : previous.bits)
    mean, variance, chi_squared = _combination_bounds(bounds, means, variances)
    # Refine only when rounding at the two ends of an interval can differ.
    for bits in _BOUND_REFINEMENTS
        bits <= bounds.bits && continue
        _rounded_rational(mean[1]) == _rounded_rational(mean[2]) &&
            _rounded_error(variance[1]) == _rounded_error(variance[2]) && break
        bounds = _update_bounds(nothing, means, variances, bits)
        mean, variance, chi_squared = _combination_bounds(bounds, means, variances)
    end
    history === nothing || (history.bounds = bounds)
    chi = bounds.used > 1 ? _rounded_rational(_midpoint(chi_squared)) : 0.0
    return _midpoint(mean), _BoundedIterationError(mean, variance, means, variances, bounds.bits), chi
end

function _rounded_error(variance::Rational{BigInt})
    # Compare with the squared midpoint on the Float64 lattice before rounding.
    scaled = numerator(variance) << 2148
    divisor = denominator(variance)
    scaled < divisor && return 4scaled <= divisor ? 0.0 : nextfloat(0.0)
    floor_error = isqrt(div(scaled, divisor))
    shift = max(0, ndigits(floor_error; base = 2) - 53)
    significand = floor_error >> shift
    midpoint = (2significand + 1) << shift
    comparison = cmp(4scaled, divisor * midpoint^2)
    round_up = comparison > 0 || (comparison == 0 && isodd(significand))
    return ldexp(Float64(significand + round_up), shift - 1074)
end

function _meets_error_tolerances(error::Float64, mean, rtol, atol)
    isfinite(error) && error > 0 && atol > 0 && rtol > 0 && !iszero(mean) || return false
    if rtol isa Float64 && atol isa Float64 && mean isa Float64
        error < atol || return false
        relative = rtol * abs(mean)
        error == relative || return error < relative
    end
    return _meets_error_tolerances(abs2(Rational{BigInt}(error)), mean, rtol, atol)
end

_rational_tolerance(x::Union{Integer, Rational, AbstractFloat}) = Rational{BigInt}(x)
_rational_tolerance(x) = Rational{BigInt}(Float64(x))

_variance_below(variance, tolerance, magnitude) =
    variance < abs2(_rational_tolerance(tolerance) * magnitude)

function _variance_below(variance, tolerance::BigFloat, magnitude)
    iszero(variance) && return true
    ratio = isone(magnitude) ? variance : variance / abs2(magnitude)
    num, den = numerator(ratio), denominator(ratio)
    exponent_bound = fld(ndigits(num; base = 2) - ndigits(den; base = 2), 2)
    binade = exponent(tolerance)
    binade > exponent_bound && return true
    binade < exponent_bound - 1 && return false
    # Find the greatest point on the tolerance's binary lattice below or at sqrt(ratio).
    bits = precision(tolerance)
    step = binade + 1 - bits
    units = step < 0 ? div(num << (-2step), den) : div(num, den << (2step))
    significand = isqrt(units)
    boundary = BigFloat(_binary_rational(significand, step), RoundNearest; precision = max(bits, ndigits(significand; base = 2)))
    return tolerance > boundary
end

function _meets_error_tolerances(variance::Rational{BigInt}, mean, rtol, atol)
    variance > 0 && atol > 0 && rtol > 0 && !iszero(mean) || return false
    isinf(rtol) || _variance_below(variance, rtol, Rational{BigInt}(mean)) || return false
    return isinf(atol) || _variance_below(variance, atol, 1)
end

function _certify_error_tolerances(variance, mean, rtol, atol)
    lower, upper = variance
    upper > 0 || return false
    largest = max(abs(mean[1]), abs(mean[2]))
    iszero(largest) && return false
    isinf(rtol) || _variance_below(lower, rtol, largest) || return false
    isinf(atol) || _variance_below(lower, atol, 1) || return false
    lower > 0 && (mean[2] < 0 || mean[1] > 0) || return nothing
    smallest = min(abs(mean[1]), abs(mean[2]))
    isinf(rtol) || _variance_below(upper, rtol, smallest) || return nothing
    isinf(atol) || _variance_below(upper, atol, 1) || return nothing
    return true
end

function _meets_error_tolerances(error::_BoundedIterationError, mean, rtol, atol)
    rtol > 0 && atol > 0 || return false
    decision = _certify_error_tolerances(error.variance, error.mean, rtol, atol)
    decision === nothing || return decision
    for bits in _BOUND_REFINEMENTS
        bits <= error.bits && continue
        bounds = _update_bounds(nothing, error.means, error.variances, bits)
        mean, variance, _ = _combination_bounds(bounds, error.means, error.variances)
        decision = _certify_error_tolerances(variance, mean, rtol, atol)
        decision === nothing || return decision
    end
    mean, variance, _ = _combine_exact_estimates(error.means, error.variances, false)
    return _meets_error_tolerances(variance, mean, rtol, atol)
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
    origin = _can_center(lo, hi) ? _rounded_mean(first(estimates)) : 0.0
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
