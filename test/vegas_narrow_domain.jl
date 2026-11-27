using MonteCarloIntegration: MonteCarloIntegration, vegas
using Random: Random
using Test: @test, @testset

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
    result = vegas(f, [a], [b]; options..., alpha = 0.0)
    @test result.integral_estimate ≈ 4.0 atol = 0.04
    @test result.standard_deviation > 0
    @test a < result.adaptive_grid[2] < b
    points = MonteCarloIntegration.sample_from_adaptive_grid(result, 128)
    @test all(point -> a < only(point) < b, points)

    grid = copy(result.adaptive_grid)
    f_batch(points) = map(f, eachcol(points))
    Random.seed!(1906)
    warmed = vegas(f_batch, result; ncalls = options.ncalls, maxiter = options.maxiter, batch = :matrix)
    @test warmed.integral_estimate ≈ 4.0 atol = 0.06
    @test result.adaptive_grid == grid
    @test a < warmed.adaptive_grid[2] < b
end

@testset "VEGAS warm start across BigFloat precisions" begin
    a = 1.0e16
    b = nextfloat(a)
    previous = setprecision(BigFloat, 64) do
        vegas(Returns(1), [a], [b]; nbins = 3, ncalls = 32, maxiter = 1, alpha = 0.0)
    end
    @test all(width -> precision(width) == 64, previous.grid_spacing)
    setprecision(BigFloat, 256) do
        resumed = vegas(Returns(1), previous; ncalls = 32, maxiter = 1, alpha = 0.0)
        @test resumed.integral_estimate == b - a
        @test all(width -> precision(width) == 256, resumed.grid_spacing)
        @test all(width -> precision(width) == 64, previous.grid_spacing)
    end
end
