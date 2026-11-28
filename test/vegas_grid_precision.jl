using MonteCarloIntegration: MonteCarloIntegration, vegas
using Random: seed!
using Test: @test, @testset, @test_throws

@testset "VEGAS checks fresh BigFloat bounds at the active precision" begin
    medium = big(2)^270
    large = big(2)^1100
    dyadic = (2 * big(2)^53 + 1) // 2
    rational = big(2)^64
    wide_input, mixed_input = setprecision(BigFloat, 288) do
        (BigFloat(medium + 1), BigFloat(rational + 2))
    end
    cases = (
        (53, 64, 2^53 + 1, 2^53 + 103, 102),
        (256, 288, medium + 1, medium + 2^18 + 3, 2^18 + 2),
        (256, 288, wide_input, medium + 2^18 + 3, 2^18 + 2),
        (1088, 1152, large + 1, large + 103, 102),
        (53, 64, dyadic, dyadic + 102, 102),
        (64, 128, rational + 1 // 3, rational + 101 + 2 // 3, 304 // 3),
        (64, 128, mixed_input, rational + 103 + 1 // 3, 304 // 3),
    )
    for (low, high, lo, hi, span) in cases
        for (a, b, expected) in ((lo, hi, span), (hi, lo, -span))
            for batch in (false, true, :matrix)
                @testset "$low → $high bits, $a → $b, batch = $batch" begin
                    calls = Ref(0)
                    function integrand(points)
                        calls[] += 1
                        batch === false && return 1.0
                        return ones(batch === true ? length(points) : size(points, 2))
                    end
                    options = (; nbins = 2, ncalls = 16, maxiter = 1, alpha = 0.0, batch)
                    setprecision(BigFloat, low) do
                        seed!(522)
                        next_draw = rand(UInt64)
                        seed!(522)
                        @test_throws ArgumentError vegas(integrand, [a], [b]; options...)
                        @test calls[] == 0
                        @test rand(UInt64) == next_draw
                    end

                    setprecision(BigFloat, high) do
                        fresh = vegas(integrand, [a], [b]; options...)
                        reused = vegas(integrand, fresh; options...)
                        for result in (fresh, reused)
                            @test result.integral_estimate == Float64(expected)
                            @test result.standard_deviation == 0.0
                            for (bound, edge) in ((a, first(result.adaptive_grid)), (b, last(result.adaptive_grid)))
                                if bound isa Rational && !ispow2(denominator(bound))
                                    @test lo <= Rational{BigInt}(edge) <= hi
                                else
                                    @test edge == BigFloat(bound)
                                end
                            end
                        end
                        @test eltype(fresh.adaptive_grid) === BigFloat
                        samples = MonteCarloIntegration.sample_from_adaptive_grid(reused, 32)
                        @test samples isa Vector{Tuple{BigFloat}}
                    end
                end
            end
        end
    end
end

@testset "VEGAS rejects genuinely nonfinite input bounds" begin
    for bound in (Inf, -Inf, NaN, BigFloat(Inf))
        calls = Ref(0)
        @test_throws ArgumentError vegas(_ -> (calls[] += 1), [0.0], [bound]; nbins = 2, ncalls = 16)
        @test calls[] == 0
    end
end

@testset "Ordinary rational grids use Float64 independently of BigFloat precision" begin
    a = 9767792313524413 // 29303376940573329
    b = a + 1 // 8
    cases = (
        (1 // 3, 5 // 3, 4 / 3),
        (big(1) // 3, big(5) // 3, 4 / 3),
        (a, b, 0.125),
        (big(a), big(b), 0.125),
    )
    setprecision(BigFloat, 16) do
        for (lo, hi, span) in cases
            for (x, y, width) in ((lo, hi, span), (hi, lo, -span))
                result = vegas(Returns(1.0), [x], [y]; nbins = 2, ncalls = 16, maxiter = 1)
                @test eltype(result.adaptive_grid) === Float64
                @test result.integral_estimate ≈ width rtol = 2eps(Float64)
                @test all(p -> lo <= Rational{BigInt}(p) <= hi, result.adaptive_grid)
                samples = MonteCarloIntegration.sample_from_adaptive_grid(result, 32)
                @test all(p -> lo <= Rational{BigInt}(only(p)) <= hi, samples)
            end
        end
    end
end

@testset "VEGAS keeps a wide rational grid inside the requested interval" begin
    lo = big(2)^64 + 1 // 3
    hi = lo + 102
    options = (; nbins = 2, ncalls = 4096, maxiter = 1, alpha = 0.0)
    inside(x) = lo <= Rational{BigInt}(x) <= hi

    for (a, b, width) in ((lo, hi, 102.0), (hi, lo, -102.0))
        @testset "$a → $b" begin
            calls, outside = Ref(0), Ref(0)
            function integrand(point)
                calls[] += 1
                outside[] += !inside(only(point))
                return 1.0
            end

            setprecision(BigFloat, 64) do
                seed!(13)
                next_draw = rand(UInt64)
                seed!(13)
                @test_throws ArgumentError vegas(integrand, [a], [b]; options...)
                @test calls[] == 0
                @test rand(UInt64) == next_draw
            end

            calls[] = outside[] = 0
            fresh = setprecision(BigFloat, 128) do
                seed!(13)
                result = vegas(integrand, [a], [b]; options...)
                @test calls[] == options.ncalls
                @test outside[] == 0
                @test all(inside, result.adaptive_grid)
                @test result.integral_estimate == width
                @test result.standard_deviation == 0.0
                @test all(point -> inside(only(point)), MonteCarloIntegration.sample_from_adaptive_grid(result, options.ncalls))
                result
            end

            setprecision(BigFloat, 64) do
                calls[] = 0
                seed!(13)
                next_draw = rand(UInt64)
                seed!(13)
                @test_throws ArgumentError vegas(integrand, fresh; options...)
                @test_throws ArgumentError MonteCarloIntegration.sample_from_adaptive_grid(fresh, options.ncalls)
                @test calls[] == 0
                @test rand(UInt64) == next_draw
            end

            for bits in (128, 256)
                setprecision(BigFloat, bits) do
                    calls[] = outside[] = 0
                    seed!(13)
                    reused = vegas(integrand, fresh; options...)
                    @test calls[] == options.ncalls
                    @test outside[] == 0
                    @test all(inside, reused.adaptive_grid)
                    @test reused.integral_estimate == width
                    @test reused.standard_deviation == 0.0
                    for result in (fresh, reused)
                        @test all(point -> inside(only(point)), MonteCarloIntegration.sample_from_adaptive_grid(result, options.ncalls))
                    end
                end
            end
        end
    end
end

@testset "VEGAS bounds the final floating-point interpolation" begin
    lo, hi = 2 // 107, 3 // 1
    for (a, b) in ((lo, hi), (hi, lo))
        grid, widths = MonteCarloIntegration._initialize_vegas_grid([a], [b], 7, nothing)
        @test eltype(grid) === Float64
        points, unit_points = zeros(1, 2), reshape([0.0, prevfloat(1.0)], 1, 2)
        MonteCarloIntegration._map_vegas_points!(points, unit_points, zeros(Int, 1, 2), zeros(2), nothing, grid, widths)
        @test all(p -> lo <= Rational{BigInt}(p) <= hi, points)
    end
end

@testset "VEGAS rounds signed rationals with large denominators inward" begin
    q = 1 // (big(2)^1100 + 1)
    lo, hi = -102q, -q
    options = (; nbins = 2, ncalls = 32, maxiter = 1, alpha = 0.0)
    inside(p) = lo <= Rational{BigInt}(p) <= hi
    setprecision(BigFloat, 128) do
        for (a, b, width) in ((lo, hi, 101.0), (hi, lo, -101.0))
            outside = Ref(0)
            function integrand(p)
                outside[] += !inside(only(p))
                return ldexp(BigFloat(1), 1100)
            end
            fresh = vegas(integrand, [a], [b]; options...)
            for result in (fresh, vegas(integrand, fresh; options...))
                @test result.integral_estimate == width
                @test all(inside, result.adaptive_grid)
                @test all(p -> inside(only(p)), MonteCarloIntegration.sample_from_adaptive_grid(result, 32))
            end
            @test outside[] == 0
        end
    end
end

@testset "VEGAS checks irrational bounds without changing ambient precision" begin
    options = (; nbins = 2, ncalls = 16, maxiter = 1, alpha = 0.0)
    setprecision(BigFloat, 16) do
        result = vegas(Returns(1), [π], [2π]; options...)
        @test eltype(result.adaptive_grid) === Float64
        @test result.integral_estimate ≈ π rtol = 2eps(Float64)
        @test precision(BigFloat) == 16
    end

    hi = nextfloat(Float64(π), 1024)
    expected = setprecision(BigFloat, 256) do
        Float64(BigFloat(hi) - BigFloat(π))
    end
    setprecision(BigFloat, 64) do
        calls = Ref(0)
        @test_throws ArgumentError vegas(_ -> (calls[] += 1), [π], [hi]; options...)
        @test calls[] == 0
        @test precision(BigFloat) == 64
    end
    setprecision(BigFloat, 128) do
        result = vegas(Returns(1), [π], [hi]; options...)
        @test eltype(result.adaptive_grid) === BigFloat
        @test result.integral_estimate ≈ expected rtol = 2eps(Float64)
        @test precision(BigFloat) == 128
    end
end

@testset "VEGAS validates the effective span of subnormal grids" begin
    intervals = (
        (nextfloat(0.0), nextfloat(0.0, 1024), 2),
        (0.0, nextfloat(0.0, 149), 9),
        (nextfloat(0.0), nextfloat(0.0, 1024), 1),
    )
    for (lo, hi, nbins) in intervals
        for (a, b) in ((lo, hi), (hi, lo))
            calls = Ref(0)
            integrand(_) = (calls[] += 1; 1.0)
            options = (; nbins, ncalls = 16, maxiter = 1, alpha = 0.0)
            width = (b - a) / nbins
            edges = reshape([i == nbins ? b : fma(i, width, a) for i in 0:nbins], :, 1)
            widths = fill(width, nbins, 1)
            historical = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, edges, widths)
            setprecision(BigFloat, 8) do
                @test BigFloat(a) == a && BigFloat(b) == b
                seed!(341)
                next_draw = rand(UInt64)
                seed!(341)
                @test_throws ArgumentError vegas(integrand, [a], [b]; options...)
                @test_throws ArgumentError vegas(integrand, historical; options...)
                @test calls[] == 0
                @test rand(UInt64) == next_draw
            end
            setprecision(BigFloat, 64) do
                fresh = vegas(integrand, [a], [b]; options...)
                @test fresh.integral_estimate == b - a
                @test vegas(integrand, fresh; options...).integral_estimate == b - a
                @test vegas(integrand, historical; options...).integral_estimate == b - a
            end
        end
    end
end

@testset "VEGAS checks nonuniform widths when widening a historical grid" begin
    q = nextfloat(0.0)
    edges = [0.0 1.0; 512q 2.0; 1024q 1024.0]
    widths = [512q 1.0; 512q 1022.0]
    historical = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, edges, widths)
    calls = Ref(0)
    integrand(_) = (calls[] += 1; 1.0)
    options = (; ncalls = 16, maxiter = 1, alpha = 0.0)
    setprecision(BigFloat, 8) do
        seed!(342)
        next_draw = rand(UInt64)
        seed!(342)
        @test_throws ArgumentError vegas(integrand, historical; options...)
        @test_throws ArgumentError MonteCarloIntegration.sample_from_adaptive_grid(historical, 16)
        @test calls[] == 0
        @test rand(UInt64) == next_draw
    end
    setprecision(BigFloat, 64) do
        result = vegas(integrand, historical; options...)
        @test result.adaptive_grid == edges
        @test result.grid_spacing == widths
        points = MonteCarloIntegration.sample_from_adaptive_grid(historical, 16)
        @test all(p -> 0 <= p[1] <= 1024q && 1 <= p[2] <= 1024, points)
    end
end
