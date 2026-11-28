using MonteCarloIntegration
using Random: rand!, seed!
using Test

const Float64GridResult = VEGASResult{Float64, Float64, Float64, Matrix{Float64}, Matrix{Float64}}
const RESULT_OPTIONS = (; nbins = 4, ncalls = 128, maxiter = 2, alpha = 0.0)

@testset "Copies and edits of native grid fields" begin
    setprecision(BigFloat, 128) do
        offset = BigFloat(2)^53
        lo, hi = offset + 1 // 8, offset + 63 // 8
        original = vegas(Returns(1.0), [lo, 0.0], [hi, 1.0]; RESULT_OPTIONS...)
        @test getfield(original, :adaptive_grid) === original.adaptive_grid
        @test getfield(original, :grid_spacing) === original.grid_spacing
        copied = deepcopy(original)
        @test copied.adaptive_grid !== original.adaptive_grid
        @test copied.grid_spacing !== original.grid_spacing
        seed!(913)
        unedited = MonteCarloIntegration.sample_from_adaptive_grid(original, 64)
        seed!(913)
        @test unedited == MonteCarloIntegration.sample_from_adaptive_grid(copied, 64)

        copied.adaptive_grid[:, 2] .= BigFloat[10, 10.125, 10.5, 10.75, 11]
        copied.grid_spacing[:, 2] .= diff(copied.adaptive_grid[:, 2])
        seed!(913)
        edited = MonteCarloIntegration.sample_from_adaptive_grid(copied, 64)
        @test first.(edited) == first.(unedited)
        @test all(p -> 10 < last(p) < 11, edited)
        warm = vegas(Returns(1.0), copied; RESULT_OPTIONS...)
        @test warm.adaptive_grid == copied.adaptive_grid
        @test warm.grid_spacing == copied.grid_spacing
        @test isfinite(warm.integral_estimate)
        @test vegas(Returns(1.0), original; RESULT_OPTIONS...).integral_estimate == 7.75
    end
end

@testset "Native results with unusable Float64 geometry" begin
    setprecision(BigFloat, 128) do
        offset = BigFloat(2)^53
        huge = BigFloat(2)^1025
        cases = (
            (offset - 1 // 4, offset + 1 // 4, 1.0, 0.5),
            (offset + 3 // 4, offset + 5 // 4, 1.0, 0.5),
            (huge, huge + BigFloat(2)^1022, ldexp(1.0, -1022), 1.0),
        )
        for (a, b, value, estimated) in cases
            result = vegas(Returns(value), [a], [b]; RESULT_OPTIONS...)
            @test result.integral_estimate == estimated
            @test result.adaptive_grid isa Matrix{BigFloat}
            @test first(result.adaptive_grid) == a
            @test last(result.adaptive_grid) == b
            samples = MonteCarloIntegration.sample_from_adaptive_grid(result, 64)
            @test all(p -> a <= only(p) <= b, samples)
        end
    end
end

@testset "Public VEGASResult constructors" begin
    edges = reshape([0.0, 0.5, 1.0], :, 1)
    widths = reshape([0.5, 0.5], :, 1)
    for result in (VEGASResult(1.0, 0.0, 0.0, edges, widths), Float64GridResult(1, 0, 0, edges, widths))
        @test result isa Float64GridResult
        @test result.adaptive_grid === edges
        @test result.grid_spacing === widths
        @test propertynames(result) == (:integral_estimate, :standard_deviation, :chi_squared_average, :adaptive_grid, :grid_spacing)
        @test vegas(Returns(1.0), result; ncalls = 128, maxiter = 1).integral_estimate == 1
    end
    converted = Float64GridResult(1, 0, 0, Float32.(edges), Float32.(widths))
    @test converted.adaptive_grid == edges
    @test converted.grid_spacing == widths
    @test fieldcount(Float64GridResult) == 5
end

@testset "Native BigFloat grids survive changes in caller rounding" begin
    settings = (; ncalls = 64, maxiter = 1, alpha = 0.0)
    cases = (
        (128, BigFloat(2; precision = 128)^53, 1 // 8, 81 // 8),
        (192, BigFloat(2; precision = 192)^80, -1 // 8, 97 // 8),
    )
    for (bits, center, low, high) in cases, nbins in (6, 7), direction in (identity, reverse)
        setprecision(BigFloat, bits) do
            lower, upper = direction((center + low, center + high))
            expected = Float64(upper - lower)
            for create_rounding in (RoundNearest, RoundUp, RoundDown)
                original = setrounding(BigFloat, create_rounding) do
                    vegas(Returns(1.0), [lower], [upper]; settings..., nbins)
                end
                for reuse_rounding in (RoundNearest, RoundUp, RoundDown, RoundToZero)
                    @testset "$bits-bit, $nbins bins, $create_rounding → $reuse_rounding, $direction" begin
                        setrounding(BigFloat, reuse_rounding) do
                            observed = Ref(true)
                            calls = Ref(0)
                            warmed = vegas(original; settings...) do point
                                calls[] += 1
                                observed[] &= rounding(BigFloat) === reuse_rounding
                                observed[] &= min(lower, upper) <= only(point) <= max(lower, upper)
                                return 1.0
                            end
                            @test calls[] == settings.ncalls
                            @test observed[]
                            @test warmed.adaptive_grid == original.adaptive_grid
                            @test warmed.grid_spacing == original.grid_spacing
                            @test warmed.integral_estimate ≈ expected rtol = 2eps(Float64)
                            @test iszero(warmed.standard_deviation)
                            @test rounding(BigFloat) === reuse_rounding
                        end
                    end
                end
            end
        end
    end
end

@testset "Adapted BigFloat grids retain their stored geometry across modes" begin
    setprecision(BigFloat, 128) do
        offset = BigFloat(2)^53
        lo, hi = offset + 1 // 8, offset + 81 // 8
        settings = (; ncalls = 128, maxiter = 2, nbins = 7)
        trained = setrounding(BigFloat, RoundDown) do
            seed!(347)
            vegas(point -> exp(only(point) - offset), [lo], [hi]; settings...)
        end
        @test length(unique(trained.grid_spacing)) > 1
        for mode in (RoundNearest, RoundUp, RoundToZero)
            setrounding(BigFloat, mode) do
                reused = vegas(Returns(1.0), trained; settings..., alpha = 0.0)
                @test reused.adaptive_grid == trained.adaptive_grid
                @test reused.grid_spacing == trained.grid_spacing
                @test rounding(BigFloat) === mode
            end
        end
    end
end

@testset "Low-bit BigFloat grids use stored physical spans" begin
    settings = (; ncalls = 4096, maxiter = 2, alpha = 0.0, atol = 0.0, rtol = 0.0)
    for (bits, span) in ((8, 29), (11, 29), (16, 15), (24, 31)), direction in (identity, reverse)
        lo, hi = direction((BigFloat(0; precision = bits), BigFloat(span; precision = bits)))
        expected = direction === identity ? span : -span
        correctly_typed = Ref(true)
        function constant(point)
            correctly_typed[] &= precision(only(point)) == bits
            return one(only(point))
        end
        seed!(826)
        result = vegas(constant, [lo], [hi]; settings..., nbins = 7)
        seed!(826)
        warmed = vegas(constant, result; settings...)
        @test correctly_typed[]
        @test result.adaptive_grid == warmed.adaptive_grid
        @test result.grid_spacing == warmed.grid_spacing
        @test all(x -> precision(x) == bits, result.grid_spacing)
        @test result.integral_estimate == warmed.integral_estimate
        @test result.standard_deviation == warmed.standard_deviation
        @test abs(result.integral_estimate - expected) <= 6result.standard_deviation + eps(Float64(expected))
    end

    bits, nbins = 11, 2500
    lo, hi = BigFloat(-1; precision = bits), BigFloat(1; precision = bits)
    seed!(5928)
    result = vegas(Returns(1.0), [lo], [hi]; settings..., nbins)
    @test length(unique(result.adaptive_grid)) == nbins + 1
    @test first(result.adaptive_grid) == lo
    @test last(result.adaptive_grid) == hi
    @test all(w -> w > 0, result.grid_spacing)
    @test all(x -> precision(x) == bits, result.adaptive_grid)
    @test abs(result.integral_estimate - 2.0) <= 6result.standard_deviation + eps(2.0)
    seed!(5928)
    warmed = vegas(Returns(1.0), result; settings...)
    @test warmed.integral_estimate == result.integral_estimate
    @test warmed.adaptive_grid == result.adaptive_grid
    points = MonteCarloIntegration.sample_from_adaptive_grid(warmed, 128)
    @test all(point -> lo <= only(point) <= hi && precision(only(point)) == bits, points)

    setprecision(BigFloat, bits) do
        ncalls = 65_536
        coordinates = Matrix{BigFloat}(undef, 1, ncalls)
        bins = zeros(Int, 1, ncalls)
        weights = zeros(ncalls)
        grid, widths = warmed.adaptive_grid, warmed.grid_spacing
        MonteCarloIntegration._with_vegas_mapping(grid, widths) do working_widths, native_bits
            seed!(1234)
            rand!(coordinates)
            MonteCarloIntegration._map_vegas_points!(coordinates, coordinates, bins, weights, nothing, grid, working_widths, native_bits)
        end
        @test length(unique(bins)) == nbins
        @test all(x -> precision(x) == bits, coordinates)
    end
end

@testset "BigFloat Jacobian factors round once for Float64 statistics" begin
    setprecision(BigFloat, 53) do
        settings = (; nbins = 7, ncalls = 2048, maxiter = 1, alpha = 0.0, atol = 0.0, rtol = 0.0)
        for extra_scales in (BigFloat[], [BigFloat(2)^1100, BigFloat(2)^(-1100)])
            upper = vcat(ones(BigFloat, 32), extra_scales)
            lower = zeros(BigFloat, length(upper))
            for direction in (identity, reverse)
                a, b = direction((lower, upper))
                target = isodd(count(i -> b[i] < a[i], eachindex(a))) ? -1.0 : 1.0
                seed!(7714)
                result = vegas(Returns(1.0), a, b; settings...)
                @test abs(result.integral_estimate - target) <= max(2eps(Float64), 6result.standard_deviation)
                @test isfinite(result.standard_deviation)
            end
        end
    end
end

@testset "Exact callback values retain their digits on low-precision grids" begin
    scale = big(2)^1100
    cases = ((big(10001), 10001.0), (big(355) // big(113), 355 / 113), (3, 3.0), (3 // 2, 1.5))
    for bits in (8, 16), mode in (RoundNearest, RoundUp, RoundDown), batch in (false, true, :matrix)
        for (value, expected) in cases, exponent in (0, -1100)
            exponent == -1100 && !(value isa Union{BigInt, Rational{BigInt}}) && continue
            x = BigFloat(0; precision = bits)
            y = BigFloat(ldexp(big(1.0), exponent); precision = bits)
            weighted_value = exponent == 0 ? value : scale * value
            setrounding(BigFloat, mode) do
                seen = Ref(true)
                function evaluate(point)
                    seen[] &= rounding(BigFloat) === mode && precision(only(point)) == bits
                    return weighted_value
                end
                integrand = batch === false ? evaluate : (points -> map(evaluate, batch === true ? points : eachcol(points)))
                result = vegas(integrand, [x], [y]; batch, ncalls = 4, nbins = 1, maxiter = 1, alpha = 0.0)
                @test seen[]
                @test result.integral_estimate === expected
                @test iszero(result.standard_deviation)
            end
        end
    end
end

@testset "BigFloat sampling rounds coordinates independently of the callback" begin
    a = BigFloat(1; precision = 53)
    b = nextfloat(a)
    width = b - a
    ncalls = 8192
    settings = (; ncalls, nbins = 1, maxiter = 1, alpha = 0.0)
    baseline = vegas(Returns(0.0), [a], [b]; settings...)
    seed!(1)
    reference = MonteCarloIntegration.sample_from_adaptive_grid(baseline, ncalls)
    @test 0.46 < count(==((b,)), reference) / ncalls < 0.54
    for mode in (RoundNearest, RoundUp, RoundDown), batch in (false, true, :matrix)
        setrounding(BigFloat, mode) do
            seen = Ref(true)
            function evaluate(point)
                seen[] &= rounding(BigFloat) === mode && precision(only(point)) == 53
                return (only(point) - a) / width
            end
            integrand = batch === false ? evaluate : (points -> map(evaluate, batch === true ? points : eachcol(points)))
            seed!(1)
            fresh = vegas(integrand, [a], [b]; batch, settings...)
            seed!(1)
            warm = vegas(integrand, baseline; batch, settings...)
            @test seen[]
            @test abs(fresh.integral_estimate - Float64(width) / 2) < 6fresh.standard_deviation
            @test (fresh.integral_estimate, fresh.standard_deviation) == (warm.integral_estimate, warm.standard_deviation)
            @test rounding(BigFloat) === mode
            if batch === false
                seed!(1)
                @test MonteCarloIntegration.sample_from_adaptive_grid(warm, ncalls) == reference
            end
        end
    end
end

@testset "Exact integrands with extreme Jacobian exponents" begin
    setprecision(BigFloat, 64) do
        options = (; ncalls = 4, nbins = 1, maxiter = 1, alpha = 0.0)
        lo, unit = [BigFloat(0)], [BigFloat(1)]
        microscopic = [ldexp(BigFloat(1), -10_000_000)]
        for value in (big(0), big(1), big(-1), big(355) // big(113))
            f = Returns(value)
            vegas(f, lo, unit; options...)
            result = vegas(f, lo, microscopic; options...)
            ordinary_bytes = @allocated vegas(f, lo, unit; options...)
            microscopic_bytes = @allocated vegas(f, lo, microscopic; options...)
            @test microscopic_bytes <= 16ordinary_bytes + 16_384
            @test result.integral_estimate === (signbit(value) ? -0.0 : 0.0)
            @test iszero(result.standard_deviation)
        end

        half_smallest = [ldexp(BigFloat(1), -1075)]
        q = nextfloat(0.0)
        for (value, expected) in ((big(1), 0.0), (big(-1), -0.0), (big(3), 2q), (big(10001) // big(10000), q))
            result = vegas(Returns(value), lo, half_smallest; options...)
            @test result.integral_estimate === expected
        end

        large = [ldexp(BigFloat(1), 1023)]
        largest_finite = (big(2)^53 - 1) // big(2)^52
        overflow_threshold = (big(2)^54 - 1) // big(2)^53
        @test vegas(Returns(largest_finite), lo, large; options...).integral_estimate === floatmax(Float64)
        @test_throws DomainError vegas(Returns(overflow_threshold), lo, large; options...)
        @test_throws DomainError vegas(Returns(big(1)), lo, [ldexp(BigFloat(1), 10_000_000)]; options...)
    end
end

@testset "Direct sampling promotes mixed grid matrices" begin
    lo, hi = 1.0, nextfloat(1.0)
    edges = reshape([lo, hi], :, 1)
    widths = reshape([BigFloat(hi - lo; precision = 53)], :, 1)
    original = VEGASResult(0.0, 0.0, 0.0, edges, widths)
    sampler = MonteCarloIntegration.sample_from_adaptive_grid
    for mode in (RoundNearest, RoundUp, RoundDown)
        setrounding(BigFloat, mode) do
            warmed = vegas(Returns(0.0), original; ncalls = 4, maxiter = 1, alpha = 0.0)
            seed!(1)
            actual = sampler(original, 256)
            seed!(1)
            expected = sampler(warmed, 256)
            @test actual == expected
            @test 0 < count(point -> only(point) == hi, actual) < length(actual)
            @test all(point -> precision(only(point)) == 53, actual)
            @test rounding(BigFloat) === mode
        end
    end
    @test original.adaptive_grid === edges && edges == reshape([lo, hi], :, 1)
    @test original.grid_spacing === widths && only(widths) == hi - lo
end

@testset "Direct sampling accepts rounded widths in mixed grids" begin
    setprecision(BigFloat, 128) do
        large_edges = [BigFloat(0), BigFloat("0.1"), BigFloat(1)]
        small_edges = [0.0, 0.1, 1.0]
        sampler = MonteCarloIntegration.sample_from_adaptive_grid
        for (points, spacing) in ((large_edges, [0.1, 0.9]), (small_edges, diff(large_edges))), reversed in (false, true)
            edges = reshape(reversed ? reverse(points) : points, :, 1)
            widths = reshape(reversed ? -reverse(spacing) : spacing, :, 1)
            saved_edges, saved_widths = copy(edges), copy(widths)
            physical_edges = BigFloat.(edges)
            normalized = VEGASResult(0.0, 0.0, 0.0, physical_edges, diff(physical_edges; dims = 1))
            original = VEGASResult(0.0, 0.0, 0.0, edges, widths)
            seed!(1)
            expected = sampler(normalized, 256)
            for mode in (RoundNearest, RoundUp, RoundDown)
                setrounding(BigFloat, mode) do
                    seed!(1)
                    actual = sampler(original, 256)
                    @test actual == expected
                    @test all(point -> 0 <= only(point) <= 1 && precision(only(point)) == 128, actual)
                    @test rounding(BigFloat) === mode
                end
            end
            @test edges == saved_edges && widths == saved_widths
        end
    end
end

@testset "Weighted callback values round once at binary64 boundaries" begin
    power(n) = n < 0 ? big(1) // (big(1) << -n) : (big(1) << n) // 1
    near_overflow = BigFloat((1 + power(-54)) * power(1024); precision = 55)
    near_subnormal = BigFloat(3 * (1 + power(-53)) * power(-1075); precision = 55)
    integer_midpoint = (big(1) << 53) + 1
    overflow_midpoint = ((big(1) << 54) - 1) << 970
    cases = (
        ([1 - power(-53)], near_overflow, BigFloat, (0, 1100)),
        ([1 - power(-53)], near_subnormal, BigFloat, (0, 1100)),
        ([1 - power(-53)], near_overflow, Float64, (0,)),
        ([1 - power(-53)], near_subnormal, Float64, (0,)),
        ([1 + power(-60)], integer_midpoint, BigFloat, (0, 1100)),
        ([1 - power(-60)], overflow_midpoint, BigFloat, (0, 1100)),
        ([1 - power(-60)], overflow_midpoint // 1, BigFloat, (0, 1100)),
        ([1 + power(-60)], Int64(integer_midpoint), BigFloat, (0,)),
        ([1 + power(-53)], 3 // 2, BigFloat, (0,)),
        ([(1 + power(-60)) * power(-1075)], 1.0, BigFloat, (0,)),
        ([(1 - power(-60)) * power(-1075)], 1.0, BigFloat, (0,)),
        ([1 + power(-30), 1 - power(-30)], overflow_midpoint, BigFloat, (0, 1100)),
        ([1 + power(-30), 1 - power(-30), 1 + power(-58)], integer_midpoint, BigFloat, (0, 1100)),
        ([1 + power(-100), 1 - power(-100)], overflow_midpoint, BigFloat, (0, 1100)),
        ([1 + power(-100), 1 - power(-100)], integer_midpoint + 2, BigFloat, (0, 1100)),
    )
    setprecision(BigFloat, 128) do
        for (widths, value, coordinate_type, shifts) in cases, shift in shifts, direction in (-1, 1)
            physical_widths = copy(widths)
            physical_widths[1] *= power(-shift)
            lo, hi = zeros(coordinate_type, length(widths)), coordinate_type.(physical_widths)
            direction < 0 && ((lo[1], hi[1]) = (hi[1], lo[1]))
            scaled_value = if shift == 0
                value
            elseif value isa BigFloat
                BigFloat(Rational{BigInt}(value) * power(shift); precision = precision(value))
            else
                value * (big(1) << shift)
            end
            product = direction * prod(physical_widths) * Rational{BigInt}(scaled_value)
            expected = setprecision(BigFloat, 4096) do
                Float64(BigFloat(product), RoundNearest)
            end
            for mode in (RoundNearest, RoundUp, RoundDown), batch in (false, true, :matrix)
                setrounding(BigFloat, mode) do
                    observed = Ref(true)
                    function evaluate(_)
                        observed[] &= rounding(BigFloat) === mode
                        return scaled_value
                    end
                    integrand = batch === false ? evaluate : (points -> map(evaluate, batch === true ? points : eachcol(points)))
                    @test begin
                        result = vegas(integrand, lo, hi; batch, ncalls = 4, nbins = 1, maxiter = 1, alpha = 0.0)
                        (result.integral_estimate, result.standard_deviation) === (expected, 0.0)
                    end
                    @test observed[]
                end
            end
        end
    end
end

@testset "Native binary callback products straddle Float64 midpoints" begin
    power(n) = n < 0 ? big(1) // (big(1) << -n) : (big(1) << n) // 1
    setprecision(BigFloat, 128) do
        for T in (Float16, Float32, Float64)
            larger = nextfloat(one(T))
            smaller = prevfloat(one(T), 4)
            cases = (
                (larger, 1 + power(-53)),
                (smaller, power(-1022) - 3power(-1075)),
                (larger, power(1024) - power(970)),
            )
            for (magnitude, midpoint) in cases, side in (RoundDown, RoundUp)
                quotient = midpoint / Rational{BigInt}(magnitude)
                width = BigFloat(quotient, side; precision = 128)
                product = Rational{BigInt}(width) * Rational{BigInt}(magnitude)
                @test side === RoundDown ? product < midpoint : product > midpoint
                @test floatmin(Float64) <= Float64(width, RoundNearest) <= floatmax(Float64)
                for reversed in (false, true), callback_sign in (-1, 1)
                    a, b = reversed ? (width, zero(width)) : (zero(width), width)
                    value = copysign(magnitude, T(callback_sign))
                    exact = (reversed ? -product : product) * callback_sign
                    expected = Float64(BigFloat(exact; precision = 256), RoundNearest)
                    for mode in (RoundNearest, RoundUp, RoundDown)
                        setrounding(BigFloat, mode) do
                            observed = Ref(true)
                            function integrand(_)
                                observed[] &= precision(BigFloat) == 128 && rounding(BigFloat) === mode
                                return value
                            end
                            run() = vegas(integrand, [a], [b]; ncalls = 4, nbins = 1, maxiter = 1, alpha = 0.0)
                            if isfinite(expected)
                                result = run()
                                @test (result.integral_estimate, result.standard_deviation) === (expected, 0.0)
                            else
                                @test_throws DomainError run()
                            end
                            @test observed[]
                            @test precision(BigFloat) == 128 && rounding(BigFloat) === mode
                        end
                    end
                end
            end
        end
    end
end

@testset "Warm starts use the stored precision of mixed grid matrices" begin
    setprecision(BigFloat, 128) do
        for T in (Float16, Float32, Float64), reverse_axis in (false, true)
            for (points, spacing) in ((BigFloat[0, BigFloat("0.1"), 1], T[0.1, 0.9]), (T[0, 0.1, 1], [BigFloat("0.1"), BigFloat("0.9")]))
                edges = reshape(reverse_axis ? reverse(points) : points, :, 1)
                widths = reshape(reverse_axis ? -reverse(spacing) : spacing, :, 1)
                start = VEGASResult(0.0, 0.0, 0.0, edges, widths)
                saved_edges, saved_widths = copy(edges), copy(widths)
                normalized = BigFloat.(edges)
                physical = diff(normalized; dims = 1)
                reference = VEGASResult(0.0, 0.0, 0.0, normalized, physical)
                solve(grid) = vegas(only, grid; ncalls = 16, maxiter = 1, alpha = 0.0)
                seed!(114)
                target = solve(reference)
                seed!(114)
                result = solve(start)
                @test result.adaptive_grid == normalized && result.grid_spacing == physical
                @test (result.integral_estimate, result.standard_deviation) == (target.integral_estimate, target.standard_deviation)
                @test (edges, widths) == (saved_edges, saved_widths)
                wrong = copy(widths)
                wrong[1] /= 2
                @test_throws ArgumentError solve(VEGASResult(0.0, 0.0, 0.0, edges, wrong))
            end
        end
        step = ldexp(BigFloat(1), -1075)
        edges = reshape(BigFloat[0, step, 2step], :, 1)
        for sign in (-1, 1)
            box = VEGASResult(0.0, 0.0, 0.0, sign .* edges, zeros(2, 1))
            result = vegas(Returns(1), box; ncalls = 4, maxiter = 1, alpha = 0.0)
            @test result.grid_spacing == fill(sign * step, 2, 1)
            @test result.integral_estimate === sign * nextfloat(0.0)
        end
    end
end

@testset "Warm starts reject widths outside the stored edge intervals" begin
    setprecision(BigFloat, 128) do
        for (swap, direction) in ((identity, 1), (reverse, -1)), sign in (1, -1)
            edges = reshape(sign .* swap(Float16[0, 1, 2]), :, 1)
            widths = BigFloat[1 + big(7) // 4096, 1 - big(7) // 4096]
            spacing = reshape(sign * direction .* swap(widths), :, 1)
            previous = VEGASResult(0.0, 0.0, 0.0, edges, spacing)
            calls = Ref(0)
            observe(_) = (calls[] += 1; 1.0)
            seed!(785661)
            subsequent = rand(UInt64)
            seed!(785661)
            @test_throws ArgumentError vegas(observe, previous; ncalls = 4, maxiter = 1, alpha = 0.0)
            @test calls[] == 0
            @test rand(UInt64) == subsequent
        end
    end
end

@testset "Warm starts compare mixed widths at rounding midpoints" begin
    options = (; ncalls = 4, maxiter = 1, alpha = 0.0)
    setprecision(BigFloat, 128) do
        for T in (Float16, Float32, Float64), at_limit in (false, true)
            upper = at_limit ? floatmax(T) : one(T)
            lower = at_limit ? prevfloat(upper) : zero(T)
            limit = if at_limit
                2 * (BigFloat(upper) - BigFloat(lower))
            else
                (BigFloat(1) + BigFloat(nextfloat(upper)) - BigFloat(prevfloat(lower))) / 2
            end
            for direction in (1, -1)
                edges = reshape(direction .* T[lower, upper], :, 1)
                exact_edges = BigFloat.(edges)
                lo, hi = minmax(exact_edges...)
                valid = VEGASResult(0.0, 0.0, 0.0, edges, fill(direction * limit, 1, 1))
                result = vegas(Returns(1), valid; options...)
                @test result.adaptive_grid == exact_edges
                @test result.grid_spacing == diff(exact_edges; dims = 1)
                @test all(p -> lo <= only(p) <= hi, MonteCarloIntegration.sample_from_adaptive_grid(result, 4))
                invalid = VEGASResult(0.0, 0.0, 0.0, edges, fill(direction * nextfloat(limit), 1, 1))
                @test_throws ArgumentError vegas(Returns(1), invalid; options...)
            end
        end
        for direction in (1, -1)
            edges = reshape(direction .* Float16[1, nextfloat(Float16(2))], :, 1)
            valid = fill(BigFloat(direction * (1 + 12 // 4096); precision = 12), 1, 1)
            invalid = fill(BigFloat(direction * (1 + 14 // 4096); precision = 12), 1, 1)
            result = vegas(Returns(1), VEGASResult(0.0, 0.0, 0.0, edges, valid); options...)
            @test only(result.grid_spacing) == direction * (1 + 1 // 512)
            @test_throws ArgumentError vegas(Returns(1), VEGASResult(0.0, 0.0, 0.0, edges, invalid); options...)
        end
    end
end

@testset "Exact stored edges accept widths rounded to signed zero" begin
    q = big(1) // (big(1) << 1075)
    for direction in (1, -1)
        edges = reshape([zero(q), direction * q], :, 1)
        previous = VEGASResult(0.0, 0.0, 0.0, edges, fill(copysign(0.0, direction), 1, 1))
        result = vegas(Returns(1), previous; ncalls = 4, maxiter = 1, alpha = 0.0)
        @test only(result.grid_spacing) == direction * ldexp(BigFloat(1), -1075)
    end
end

@testset "Warm starts and direct draws retain declared mixed edge precision" begin
    options = (; ncalls = 4, maxiter = 1, alpha = 0.0)
    lower, upper = 1.0, nextfloat(1.0)
    pairs = (
        ([lower, upper], 8), ([lower, upper], 24), ([lower, upper], 52),
        ([1.0, nextfloat(1.0), 2.0], 8), ([1.0, nextfloat(1.25), 2.0], 8),
        ([big(2)^53 + 1, big(2)^53 + 9], 53),
        ([1 + big(1) // big(2)^55, 1 + big(1) // big(2)^54], 53),
        ([big(1) // 3, big(1) // 3 + big(1) // big(2)^80], 53),
    )
    for (axis, bits) in pairs, direction in (identity, reverse)
        edges = reshape(direction(axis), :, 1)
        widths = reshape([BigFloat(span, RoundNearest; precision = bits) for span in diff(direction(axis))], :, 1)
        previous = VEGASResult(0.0, 0.0, 0.0, edges, widths)
        calls = Ref(0)
        observe(_) = (calls[] += 1; 1.0)
        seed!(2964)
        next_draw = rand(UInt64)
        seed!(2964)
        @test_throws ArgumentError vegas(observe, previous; options...)
        @test calls[] == 0
        @test rand(UInt64) == next_draw
        seed!(2964)
        @test_throws ArgumentError MonteCarloIntegration.sample_from_adaptive_grid(previous, 4)
        @test rand(UInt64) == next_draw
    end

    for (axis, bits) in (([lower, upper], 53), ([big(1) // 3, big(2) // 3], 96), ([big(2)^53 + 1, big(2)^53 + 9], 54)), direction in (identity, reverse)
        edges = reshape(direction(axis), :, 1)
        widths = reshape([BigFloat(span, RoundNearest; precision = bits) for span in diff(direction(axis))], :, 1)
        previous = VEGASResult(0.0, 0.0, 0.0, edges, widths)
        result = vegas(Returns(1), previous; options...)
        @test all(x -> precision(x) == bits, result.adaptive_grid)
        @test vec(result.adaptive_grid) == [BigFloat(x, RoundNearest; precision = bits) for x in direction(axis)]
        @test result.integral_estimate ≈ Float64(sum(widths))
        seed!(5739)
        points = MonteCarloIntegration.sample_from_adaptive_grid(previous, 4)
        next_draw = rand(UInt64)
        seed!(5739)
        @test points == MonteCarloIntegration.sample_from_adaptive_grid(result, 4)
        @test rand(UInt64) == next_draw
    end
end

@testset "Mixed nominal widths account for the last stored edge" begin
    options = (; ncalls = 4, maxiter = 1, alpha = 0.0)
    for T in (Float16, Float32, Float64), (direction, order) in ((identity, 1), (reverse, -1)), sign in (1, -1)
        axis = direction(sign .* T[0, 1, nextfloat(T(2), 2)])
        widths = fill(BigFloat(sign * order; precision = 128), 2, 1)
        previous = VEGASResult(0.0, 0.0, 0.0, reshape(axis, :, 1), widths)
        calls = Ref(0)
        observe(_) = (calls[] += 1; 1.0)
        seed!(9685)
        next_draw = rand(UInt64)
        seed!(9685)
        @test_throws ArgumentError vegas(observe, previous; options...)
        @test calls[] == 0
        @test rand(UInt64) == next_draw
        seed!(9685)
        @test_throws ArgumentError MonteCarloIntegration.sample_from_adaptive_grid(previous, 4)
        @test rand(UInt64) == next_draw
    end

    for direction in (1, -1), anchor_end in (false, true)
        lo, hi = direction .* (Float16(0.7), Float16(2.1))
        nbins = 5
        step = (hi - lo) / nbins
        edges = reshape([lo + i * step for i in 0:nbins], :, 1)
        if anchor_end
            edges[end] = hi
        else
            @test (edges[end] - lo) / nbins != step
        end
        widths = fill(BigFloat(step; precision = 128), nbins, 1)
        previous = VEGASResult(0.0, 0.0, 0.0, edges, widths)
        result = vegas(Returns(1), previous; options...)
        @test result.adaptive_grid[1] == lo && result.adaptive_grid[end] == edges[end]
        @test result.grid_spacing == diff(result.adaptive_grid; dims = 1)
        seed!(4205)
        points = MonteCarloIntegration.sample_from_adaptive_grid(previous, 4)
        next_draw = rand(UInt64)
        seed!(4205)
        @test points == MonteCarloIntegration.sample_from_adaptive_grid(result, 4)
        @test rand(UInt64) == next_draw
    end
end

@testset "Exact widths and mixed-grid sampling" begin
    options = (; ncalls = 4, maxiter = 1, alpha = 0.0)
    draw(grid) = MonteCarloIntegration.sample_from_adaptive_grid(grid, 16)
    stored(points, widths) = VEGASResult(0.0, 0.0, 0.0, reshape(points, :, 1), reshape(widths, :, 1))
    function rejects(grid)
        calls = Ref(0)
        observe(_) = (calls[] += 1; 1.0)
        seed!(4659)
        next_draw = rand(UInt64)
        seed!(4659)
        @test_throws ArgumentError vegas(observe, grid; options...)
        @test calls[] == 0
        @test rand(UInt64) == next_draw
        seed!(4659)
        @test_throws ArgumentError draw(grid)
        @test rand(UInt64) == next_draw
    end
    function canonical(grid)
        result = vegas(Returns(1), grid; options...)
        @test result.grid_spacing == diff(result.adaptive_grid; dims = 1)
        seed!(5739)
        actual = draw(grid)
        next_draw = rand(UInt64)
        seed!(5739)
        @test actual == draw(result)
        @test rand(UInt64) == next_draw
    end
    setprecision(BigFloat, 128) do
        for (reverse_axis, direction) in ((identity, 1), (reverse, -1)), sign in (1, -1)
            sign_widths = sign * direction
            float16_axis = reverse_axis(sign .* Float16[0, 1, 2])
            for I in (Int, BigInt)
                width = Rational{I}[1 + 7 // 4096, 1 - 7 // 4096]
                rejects(stored(float16_axis, sign_widths .* reverse_axis(width)))
                allowed = Rational{I}[1 + 1 // 8192, 1 - 1 // 8192]
                canonical(stored(float16_axis, sign_widths .* reverse_axis(allowed)))
                rejects(stored(reverse_axis(sign .* Float16[1, nextfloat(Float16(1))]), [zero(I)]))
                rejects(stored(reverse_axis(sign .* Float16[0, 2048, 4096]), sign_widths .* reverse_axis(I[2051, 2045])))
                canonical(stored(float16_axis, fill(I(sign_widths), 2)))
                exact_axis = reverse_axis(sign .* Rational{I}[0, 1, 2])
                forged = Rational{I}[1 + 1 // 2^60, 1 - 1 // 2^60]
                rejects(stored(exact_axis, sign_widths .* reverse_axis(forged)))
            end

            closed_span = reverse_axis(sign .* Float16[1, 2])
            canonical(stored(closed_span, [sign_widths * (1 + 5 // 4096)]))
            open_span = reverse_axis(sign .* Float16[1, nextfloat(Float16(2))])
            rejects(stored(open_span, [sign_widths * (1 + 13 // 4096)]))

            rational_axis = reverse_axis(sign .* [0 // 1, 1 // 3])
            for width in (Float16(1 // 3), Float64(1 // 3), BigFloat(1 // 3; precision = 128))
                canonical(stored(rational_axis, [copysign(width, sign_widths)]))
                rejects(stored(rational_axis, [copysign(nextfloat(width), sign_widths)]))
            end
            for edge_precision in ((11, 11, 11), (128, 11, 128))
                points = [BigFloat(sign * i; precision = edge_precision[i + 1]) for i in 0:2]
                for (offset, check) in ((1 // 8192, canonical), (7 // 4096, rejects))
                    widths = [BigFloat(sign_widths * w; precision = 128) for w in reverse_axis((1 + offset, 1 - offset))]
                    check(stored(reverse_axis(points), widths))
                end
            end
        end
    end
end

@testset "Anchored mixed nominal widths preserve distinct stored edges" begin
    options = (; ncalls = 16, maxiter = 1, alpha = 0.0)
    sample(grid) = MonteCarloIntegration.sample_from_adaptive_grid(grid, 8)
    setprecision(BigFloat, 128) do
        nominal = Float16(1365 // 4096)
        extent = 1 + ldexp(one(BigFloat), -80)
        for sign in (1, -1), reversed in (false, true)
            lo, hi = reversed ? (sign * extent, BigFloat(0)) : (BigFloat(0), sign * extent)
            step = sign * (reversed ? -nominal : nominal)
            function make_grid(bound, width)
                edges = reshape([lo, lo + BigFloat(step), lo + 2BigFloat(step), bound], :, 1)
                return VEGASResult(0.0, 0.0, 0.0, edges, fill(width, 3, 1))
            end
            @testset "$sign, reversed=$reversed" begin
                grid = make_grid(hi, step)
                widths = diff(grid.adaptive_grid; dims = 1)
                reference = VEGASResult(0.0, 0.0, 0.0, grid.adaptive_grid, widths)
                seed!(9462)
                expected = vegas(Returns(1), reference; options...)
                seed!(9462)
                actual = vegas(Returns(1), grid; options...)
                @test actual.adaptive_grid == grid.adaptive_grid
                @test actual.grid_spacing == widths
                @test (actual.integral_estimate, actual.standard_deviation) == (expected.integral_estimate, expected.standard_deviation)
                seed!(7543)
                expected_points = sample(reference)
                next_draw = rand(UInt64)
                seed!(7543)
                @test sample(grid) == expected_points
                @test rand(UInt64) == next_draw
            end
            @testset "Wrong final span, $sign, reversed=$reversed" begin
                wrong = hi + copysign(ldexp(one(BigFloat), -9), step)
                for invalid in (make_grid(wrong, step), make_grid(hi, Rational{BigInt}(step)))
                    calls = Ref(0)
                    callback(_) = (calls[] += 1; 1.0)
                    seed!(96855)
                    next_draw = rand(UInt64)
                    seed!(96855)
                    @test_throws ArgumentError vegas(callback, invalid; options...)
                    @test calls[] == 0
                    @test rand(UInt64) == next_draw
                    seed!(96855)
                    @test_throws ArgumentError sample(invalid)
                    @test rand(UInt64) == next_draw
                end
            end
        end
    end
end

@testset "Collapsed cells accept only exact zero widths" begin
    options = (; ncalls = 16, maxiter = 1, alpha = 0.0)
    sample(grid) = MonteCarloIntegration.sample_from_adaptive_grid(grid, 8)
    for (orient, direction) in ((identity, 1), (reverse, -1)), sign in (1, -1)
        for (edges, widths) in (
                (Float16[2048, 2048], [1]),
                (Float16[2048, 2048], [-1]),
                (Float16[2048, 2048], [typemax(Int) ÷ 2]),
                (Float16[1, 1, 2], [1 // 4096, 1 // 1]),
                (Float16[1, 1, 2], [-1 // 4096, 1 // 1]),
                (Float16[1, 1, 2], [1 // 2^50, 1 // 1]),
                (Float16[1, 1, 2], [big(1) // 4096, big(1) // 1]),
            )
            points = reshape(orient(sign .* edges), :, 1)
            declared = reshape(sign * direction .* orient(widths), :, 1)
            grid = VEGASResult(0.0, 0.0, 0.0, points, declared)
            calls = Ref(0)
            callback(_) = (calls[] += 1; 1.0)
            seed!(7713)
            suffix = rand(UInt64)
            seed!(7713)
            @test_throws ArgumentError vegas(callback, grid; options...)
            @test calls[] == 0 && rand(UInt64) == suffix
            seed!(7713)
            @test_throws ArgumentError sample(grid)
            @test rand(UInt64) == suffix
        end
        for (edges, widths) in ((Float16[2048, 2048], [0]), (Float16[1, 1, 2], [0 // 1, 1 // 1]))
            points = reshape(orient(sign .* edges), :, 1)
            declared = reshape(sign * direction .* orient(widths), :, 1)
            raw = VEGASResult(0.0, 0.0, 0.0, points, declared)
            accepted = vegas(Returns(1), raw; options...)
            @test accepted.adaptive_grid == points
            @test accepted.grid_spacing == diff(points; dims = 1)
            seed!(732)
            actual = sample(raw)
            suffix = rand(UInt64)
            seed!(732)
            @test actual == sample(accepted)
            @test rand(UInt64) == suffix
        end
    end
end

@testset "Anchored nominal grids recognize native-rounded interior edges" begin
    options = (; ncalls = 16, maxiter = 1, alpha = 0.0)
    sample(grid) = MonteCarloIntegration.sample_from_adaptive_grid(grid, 8)
    setprecision(BigFloat, 192) do
        for T in (Float16, Float32), sign in (1, -1), direction in (1, -1)
            origin = sign * T(direction > 0 ? 1 : 2)
            step = sign * direction * T(0.3)
            lo, physical_step = BigFloat(origin), BigFloat(step)
            hi = lo + 3 * (physical_step + copysign(ldexp(one(BigFloat), -80), step))
            native = [BigFloat(origin + i * step) for i in 0:2]
            promoted = [lo + i * physical_step for i in 0:2]
            @test native != promoted
            @test T(hi) != hi
            for prefix in (native, promoted)
                @testset "$T, sign=$sign, direction=$direction, native=$(prefix === native)" begin
                    edges = reshape([prefix; hi], :, 1)
                    widths = diff(edges; dims = 1)
                    grid = VEGASResult(0.0, 0.0, 0.0, edges, fill(step, 3, 1))
                    reference = VEGASResult(0.0, 0.0, 0.0, edges, widths)
                    seed!(7527)
                    expected = vegas(Returns(1), reference; options...)
                    seed!(7527)
                    actual = vegas(Returns(1), grid; options...)
                    @test actual.adaptive_grid == edges
                    @test actual.grid_spacing == widths
                    @test (actual.integral_estimate, actual.standard_deviation) == (expected.integral_estimate, expected.standard_deviation)
                    seed!(6262)
                    samples = sample(reference)
                    suffix = rand(UInt64)
                    seed!(6262)
                    @test sample(grid) == samples
                    @test rand(UInt64) == suffix
                end
            end
            for points in ([native; hi + 12 * BigFloat(eps(abs(step))) * direction * sign], [lo, lo, native[3], hi])
                grid = VEGASResult(0.0, 0.0, 0.0, reshape(points, :, 1), fill(step, 3, 1))
                calls = Ref(0)
                callback(_) = (calls[] += 1; 1.0)
                seed!(168)
                suffix = rand(UInt64)
                seed!(168)
                @test_throws ArgumentError vegas(callback, grid; options...)
                @test calls[] == 0 && rand(UInt64) == suffix
                seed!(168)
                @test_throws ArgumentError sample(grid)
                @test rand(UInt64) == suffix
            end
        end
    end
end

@testset "Exact widths fit one set of rounded edges" begin
    options = (; ncalls = 16, maxiter = 1, alpha = 0.0)
    sample(grid) = MonteCarloIntegration.sample_from_adaptive_grid(grid, 16)
    signed(x, s) = s * x
    signed(x::BigFloat, s) = BigFloat(s * x; precision = precision(x))

    function rejects(grid)
        calls = Ref(0)
        observe(_) = (calls[] += 1; 1.0)
        seed!(133)
        suffix = rand(UInt64)
        seed!(133)
        @test_throws ArgumentError vegas(observe, grid; options...)
        @test calls[] == 0 && rand(UInt64) == suffix
        seed!(133)
        @test_throws ArgumentError sample(grid)
        @test rand(UInt64) == suffix
    end

    function accepts(grid)
        result = vegas(Returns(1), grid; options...)
        @test result.adaptive_grid == grid.adaptive_grid
        @test result.grid_spacing == diff(result.adaptive_grid; dims = 1)
        seed!(133)
        points = sample(grid)
        suffix = rand(UInt64)
        seed!(133)
        @test points == sample(result)
        @test rand(UInt64) == suffix
    end

    setprecision(BigFloat, 128) do
        incompatible = [1 + 1 // 4096, 1 + 1 // 1024, 1 - 5 // 4096]
        compatible = [1 + 1 // 8192, 1 - 1 // 8192, 1 // 1]
        mixed_big = Union{Int, BigFloat}[0, BigFloat(1; precision = 11), BigFloat(2; precision = 11), BigFloat(3)]
        for (orient, direction) in ((identity, 1), (reverse, -1)), sign in (1, -1)
            function stored(edges::AbstractVector{E}, widths::AbstractVector{W}) where {E, W}
                x = reshape(E[signed(edge, sign) for edge in orient(edges)], :, 1)
                w = reshape(W[sign * direction * width for width in orient(widths)], :, 1)
                return VEGASResult(0.0, 0.0, 0.0, x, w)
            end

            for edges in (Float16[0, 1, 2, 3], mixed_big), I in (Int, BigInt)
                rejects(stored(edges, Rational{I}.(incompatible)))
                accepts(stored(edges, Rational{I}.(compatible)))
            end

            exact_float_edges = Union{Int, Float64}[0, 1.0]
            rejects(stored(exact_float_edges, [1 + 1 // 2^52]))
            accepts(stored(exact_float_edges, [1 + 1 // 2^53]))
            open_float_edges = Union{Int, Float64}[0, nextfloat(1.0)]
            rejects(stored(open_float_edges, [1 + 3 // 2^53]))

            accepts(stored(Float16[1, 2, 3], [1 // 1, 1 + 5 // 4096]))
            rejects(stored(Float16[1, 2, nextfloat(Float16(3))], [1 // 1, 1 + 13 // 4096]))

            mixed_edges = Real[0, Float16(1), Float16(2), Float16(3), 4]
            rejects(stored(mixed_edges, Real[incompatible; Float16(1)]))
            accepts(stored(mixed_edges, Real[compatible; Float16(1)]))
            accepts(stored(Union{Int, Float16}[0, Float16(1), 2], compatible[1:2]))
            accepts(stored(Float16[0, 1, 2], Union{Float16, Rational{Int}}[1 // 1, Float16(1)]))

            for T in (Float64, BigFloat)
                bridge = Union{T, Rational{Int}}[incompatible[1], T(incompatible[2]), incompatible[3]]
                rejects(stored(Float16[0, 1, 2, 3], bridge))
                bridge = Union{T, Rational{Int}}[compatible[1], T(compatible[2]), compatible[3]]
                accepts(stored(Float16[0, 1, 2, 3], bridge))
            end

            for exponent in (-10_000_000, 10_000_000)
                edge = ldexp(one(BigFloat), exponent)
                rejects(stored(BigFloat[0, edge, 2edge], Union{Int, BigFloat}[edge, 1]))
                repeated = stored(BigFloat[0, edge, edge], Union{Int, BigFloat}[edge, 0])
                result = vegas(Returns(0), repeated; options...)
                @test result.grid_spacing == diff(repeated.adaptive_grid; dims = 1)
                seed!(133)
                points = sample(repeated)
                seed!(133)
                @test points == sample(result)
            end
        end
    end
end
