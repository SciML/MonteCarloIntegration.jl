using MonteCarloIntegration: MonteCarloIntegration, vegas
using Random: Random
using Test: @test, @testset, @test_throws

@testset "VEGAS narrow domains" begin
    a = 10^16
    options = (; nbins = 2, ncalls = 8192, maxiter = 2, rtol = 0.0, atol = 0.0)
    for (lo, hi, expected) in (
            (a, a + 1, 1.0), (a + 1, a, -1.0),
            (a + 1, a + 12, 11.0), (a + 12, a + 1, -11.0),
        )
        Random.seed!(1904)
        result = vegas(Returns(1), [lo], [hi]; options...)
        @test result.integral_estimate == expected
        @test first(result.adaptive_grid) == lo
        @test last(result.adaptive_grid) == hi
    end

    a = Float64(a)
    b = nextfloat(a)
    f(x) = 1 + (x[1] - a)
    Random.seed!(1905)
    result = vegas(f, [a], [b]; options..., alpha = 0.0, callback_type = BigFloat)
    @test result.integral_estimate ≈ 4.0 atol = 0.04
    @test result.standard_deviation > 0
    @test a < result.adaptive_grid[2] < b
    points = MonteCarloIntegration.sample_from_adaptive_grid(result, 128)
    @test points isa Vector{NTuple{1, BigFloat}}
    @test all(point -> a < only(point) < b, points)
    @test MonteCarloIntegration.sample_from_adaptive_grid(result, 0) isa Vector{NTuple{1, BigFloat}}

    grid = copy(result.adaptive_grid)
    f_batch(points) = map(f, eachcol(points))
    Random.seed!(1906)
    warmed = vegas(f_batch, result; ncalls = options.ncalls, maxiter = options.maxiter, batch = :matrix, callback_type = BigFloat)
    @test warmed.integral_estimate ≈ 4.0 atol = 0.06
    @test result.adaptive_grid == grid
    @test a < warmed.adaptive_grid[2] < b
end

@testset "VEGAS rational bounds" begin
    a = 10^16
    options = (; nbins = 2, ncalls = 32, maxiter = 1, alpha = 0.0)
    for (lo, hi, expected) in (((a + 1) // 1, (a + 12) // 1, 11.0), ((a + 12) // 1, (a + 1) // 1, -11.0))
        result = vegas(Returns(1), [lo], [hi]; options...)
        @test result.integral_estimate == expected
        @test first(result.adaptive_grid) == lo
        @test last(result.adaptive_grid) == hi
        @test vegas(Returns(1), result; options...).integral_estimate == expected
    end
end

@testset "VEGAS adapted grids require their stored precision" begin
    lo, hi, previous = setprecision(BigFloat, 256) do
        lo, hi = BigFloat(1), BigFloat(1) + BigFloat(2)^-80
        Random.seed!(2518)
        trained = vegas(x -> (x[1] - lo) / (hi - lo), [lo], [hi]; nbins = 4, ncalls = 256, maxiter = 2)
        # The widths fit in 64 bits of precision even when coordinates need more.
        spacing = (hi - lo) / 2^48
        edges = lo .+ round.((trained.adaptive_grid .- lo) ./ spacing) .* spacing
        previous = MonteCarloIntegration.VEGASResult(
            trained.integral_estimate, trained.standard_deviation, trained.chi_squared_average,
            edges, diff(edges; dims = 1),
        )
        (lo, hi, previous)
    end
    @test length(unique(previous.grid_spacing)) > 1
    setprecision(BigFloat, 64) do
        calls = Ref(0)
        @test_throws ArgumentError vegas(_ -> (calls[] += 1), previous; ncalls = 64, maxiter = 1)
        @test calls[] == 0
        @test_throws ArgumentError MonteCarloIntegration.sample_from_adaptive_grid(previous, 64)
    end
    setprecision(BigFloat, 256) do
        interior = BigFloat[]
        Random.seed!(2519)
        vegas(previous; ncalls = 64, maxiter = 1) do x
            push!(interior, only(x))
            return 1
        end
        points = MonteCarloIntegration.sample_from_adaptive_grid(previous, 64)
        for coordinates in (interior, only.(points))
            @test all(x -> lo < x < hi, coordinates)
            @test length(unique(coordinates)) > 1
        end
    end
end

@testset "VEGAS warm start across BigFloat precisions" begin
    a = 1.0e16
    b = nextfloat(a)
    for alpha in (0.0, 1.5)
        previous = setprecision(BigFloat, 64) do
            Random.seed!(42)
            vegas(x -> only(x) - a, [a], [b]; nbins = 3, ncalls = 128, maxiter = 2, alpha, callback_type = BigFloat)
        end
        setprecision(BigFloat, 256) do
            edges, widths = copy(previous.adaptive_grid), copy(previous.grid_spacing)
            Random.seed!(1907)
            resumed = vegas(Returns(1), previous; ncalls = 2048, maxiter = 1, alpha = 0.0)
            @test resumed.integral_estimate ≈ b - a rtol = 0.03
            for (stored, original, expanded) in (
                    (previous.adaptive_grid, edges, resumed.adaptive_grid),
                    (previous.grid_spacing, widths, resumed.grid_spacing),
                )
                @test all(value -> precision(value) == 256, expanded)
                @test all(value -> precision(value) == 64, stored)
                @test stored == original
            end
        end
    end
end

@testset "VEGAS historical uniform grids preserve narrow bounds" begin
    a, b = prevfloat(-2.0^53), -2.0^53
    options = (; nbins = 3, ncalls = 1000, maxiter = 1, alpha = 0.0)
    for bits in (64, 256), (lo, hi) in ((a, b), (b, a))
        @testset "$bits-bit precision, $lo → $hi" begin
            setprecision(BigFloat, bits) do
                edges = reshape(collect(range(lo, hi; length = options.nbins + 1)), :, 1)
                widths = fill((hi - lo) / options.nbins, options.nbins, 1)
                historical = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, edges, widths)
                f(x) = sqrt(b - only(x))
                Random.seed!(2771)
                fresh = vegas(f, [lo], [hi]; options...)

                Random.seed!(2772)
                points = MonteCarloIntegration.sample_from_adaptive_grid(historical, options.ncalls)
                @test all(point -> a <= only(point) <= b, points)
                @test points isa Vector{NTuple{1, BigFloat}}
                @test length(unique(points)) > 2
                Random.seed!(2772)
                @test points == MonteCarloIntegration.sample_from_adaptive_grid(fresh, options.ncalls)

                Random.seed!(2771)
                reused = vegas(f, historical; options...)
                @test reused.integral_estimate == fresh.integral_estimate
                @test reused.standard_deviation == fresh.standard_deviation
                @test reused.adaptive_grid == fresh.adaptive_grid
                @test edges == reshape(collect(range(lo, hi; length = options.nbins + 1)), :, 1)
                @test all(==((hi - lo) / options.nbins), widths)
            end
        end
    end
end

@testset "VEGAS checks precision before rebuilding rounded grids" begin
    a, b = prevfloat(-2.0^53), -2.0^53
    edges = reshape(collect(range(a, b; length = 4)), :, 1)
    historical = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, edges, fill((b - a) / 3, 3, 1))
    setprecision(BigFloat, 53) do
        @test_throws ArgumentError vegas(Returns(1), [a], [b]; nbins = 3, ncalls = 32, maxiter = 1)
        @test_throws ArgumentError vegas(Returns(1), historical; ncalls = 32, maxiter = 1)
        Random.seed!(2773)
        next_number = rand()
        Random.seed!(2773)
        @test_throws ArgumentError MonteCarloIntegration.sample_from_adaptive_grid(historical, 32)
        @test rand() == next_number
    end
end

@testset "VEGAS subnormal bin widths" begin
    options = (; nbins = 1000, ncalls = 256, maxiter = 1, alpha = 0.0)
    for b in (nextfloat(0.0), nextfloat(0.0, 2600), 1.0e-318), (lo, hi) in ((0.0, b), (b, 0.0))
        @testset "$lo → $hi" begin
            fresh = vegas(Returns(1.0), [lo], [hi]; options...)
            @test fresh.integral_estimate == hi - lo
            @test all(w -> sign(hi - lo) * w > 0, diff(vec(fresh.adaptive_grid)))

            width = (hi - lo) / options.nbins
            edges = reshape([fma(i, width, lo) for i in 0:options.nbins], :, 1)
            edges[end] = hi
            historical = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, edges, fill(width, options.nbins, 1))
            for previous in (fresh, historical)
                Random.seed!(2774)
                points = MonteCarloIntegration.sample_from_adaptive_grid(previous, options.ncalls)
                @test all(point -> 0.0 <= only(point) <= b, points)
                reused = vegas(Returns(1.0), previous; options...)
                @test reused.integral_estimate == hi - lo
            end
        end
    end
end

@testset "VEGAS remains inside the bounds at active BigFloat precision" begin
    setprecision(BigFloat, 64) do
        b = -(BigFloat(2)^53)
        a = b - 7eps(abs(b))
        for (lo, hi) in ((a, b), (b, a))
            Random.seed!(2780)
            result = vegas(x -> sqrt(b - only(x)), [lo], [hi]; nbins = 3, ncalls = 1000, maxiter = 1, alpha = 0.0, callback_type = BigFloat)
            @test isfinite(result.integral_estimate)
            Random.seed!(2781)
            points = MonteCarloIntegration.sample_from_adaptive_grid(result, 1000)
            @test all(point -> a <= only(point) <= b, points)
        end
    end
end

@testset "VEGAS finite bounds with an overflowing Float64 span" begin
    a, b = -1.0e308, 1.0e308
    options = (; nbins = 10, ncalls = 32, maxiter = 1, alpha = 0.0)
    for (lo, hi) in ((a, b), (b, a))
        valid = Ref(true)
        function scaled_integrand(x)
            valid[] &= a <= only(x) <= b
            return 1.0e-308
        end
        fresh = vegas(scaled_integrand, [lo], [hi]; options...)
        reused = vegas(scaled_integrand, fresh; options...)
        @test valid[]
        @test fresh.integral_estimate ≈ (hi > lo ? 2.0 : -2.0)
        @test reused.integral_estimate == fresh.integral_estimate
        points = MonteCarloIntegration.sample_from_adaptive_grid(fresh, options.ncalls)
        @test all(point -> a <= only(point) <= b, points)
    end
end
