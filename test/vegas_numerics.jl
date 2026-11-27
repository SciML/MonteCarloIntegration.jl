using MonteCarloIntegration, Random, Test

@testset "Variance of the integral estimate" begin
    for scale in (1.0, 1.0e-240, 1.0e240)
        samples = scale .* [1, 2, 3, 4]
        result = vegas(Returns(samples), [0.0], [1.0]; batch = true, nbins = 2, ncalls = 4, maxiter = 1)
        @test result.integral_estimate / scale ≈ 2.5
        @test result.standard_deviation / scale ≈ sqrt(5 / 12)
        @test iszero(result.chi_squared_average)
    end
end

@testset "Constant and sparse integrands" begin
    extent = 8.366600265340756
    for (f, lo, hi, expected) in (
            (Returns(1), fill(-extent, 4), fill(extent, 4), (2extent)^4),
            (Returns(-3), [1.0, 0.0], [0.0, 2.0], 6.0),
            (Returns(0), zeros(100), ones(100), 0.0),
            (Returns(7), [1.5], [1.5], 0.0),
        )
        result = vegas(f, lo, hi; maxiter = 3)
        @test result.integral_estimate ≈ expected
        @test iszero(result.standard_deviation)
        @test iszero(result.chi_squared_average)
        @test result.adaptive_grid[1, :] == lo
        @test result.adaptive_grid[end, :] == hi
    end
    for nu in (0.001, 1.0, 1000.0)
        Random.seed!(457)
        result = vegas(x -> exp(-sum(abs2, x)^nu / (2nu)), [-5, -5], [5, 5]; maxiter = 5)
        @test isfinite(result.integral_estimate) && result.integral_estimate >= 0
        @test isfinite(result.standard_deviation) && result.standard_deviation >= 0
        @test all(>(0), diff(result.adaptive_grid; dims = 1))
    end
end

@testset "Sparse samples and iteration statistics" begin
    for nbins in (1, 8)
        batches = ([0.0, 0, 0, 0], [1.0, 2, 3, 4], [2.0, 3, 4, 5])
        step = Ref(0)
        result = vegas(_ -> batches[step[] += 1], [0], [1]; nbins, ncalls = 4, maxiter = 3, batch = true, alpha = 0, rtol = 0, atol = 0)
        @test step[] == 3
        @test result.integral_estimate ≈ (0 + 2.5 + 3.5) / 3
        @test result.standard_deviation ≈ sqrt((2^2 + 0.5^2 + 1.5^2) / 6)
        @test result.chi_squared_average ≈ 1.2
    end
end

@testset "Large offsets and grid weights" begin
    a = 1.5e250
    b = nextfloat(a)
    result = vegas(Returns([a, b]), [0], [1]; batch = true, nbins = 1, ncalls = 2, maxiter = 1)
    @test result.standard_deviation == (b - a) / 2
    @test a <= result.integral_estimate <= b
    bins = [1 1 2 2; 1 2 1 2]
    for scale in (0.0, 1.0e-200, 1.0e200)
        d = MonteCarloIntegration.calculate_d(scale .* [1, 2, 3, 4], bins, 2)
        @test all(isfinite, d)
        edges = [0.0 1; 0.5 0.5; 1 0]
        x, w = MonteCarloIntegration.update_grid(edges, diff(edges; dims = 1), d)
        @test x[[1, end], :] == edges[[1, end], :]
        @test all(>(0), w[:, 1]) && all(<(0), w[:, 2])
        @test w == diff(x; dims = 1)
    end
end

@testset "Weights at different scales" begin
    batches = ([0.0, 2.0e300], [0.0, 2.0e-300])
    step = Ref(0)
    result = vegas(_ -> batches[step[] += 1], [0], [1]; batch = true, nbins = 1, ncalls = 2, maxiter = 2, rtol = 0, atol = 0)
    @test result.integral_estimate / 1.0e-300 ≈ 1
    @test result.standard_deviation / 1.0e-300 ≈ 1
end

@testset "Zero-variance batches and centered chi-squared" begin
    Random.seed!(7)
    hits = Ref(0)
    function rare_event(x)
        hit = x[1] > 0.9
        hits[] += hit
        return hit
    end
    result = vegas(rare_event, [0], [1]; nbins = 1, ncalls = 2, maxiter = 100, rtol = 0, atol = 0)
    @test result.integral_estimate ≈ hits[] / 200
    @test abs(result.integral_estimate - 0.1) < 4result.standard_deviation
    a = 1.0e16
    batches = ([a - 2, a + 2], [a, a + 4])
    step = Ref(0)
    result = vegas(_ -> batches[step[] += 1], [0], [1]; batch = true, nbins = 1, ncalls = 2, maxiter = 2, rtol = 0, atol = 0)
    @test result.standard_deviation ≈ sqrt(2)
    @test result.chi_squared_average ≈ 0.5
end

@testset "Reduced chi-squared with differently weighted iterations" begin
    a = 1.0e16
    for (outlier, uncertainty) in ((0.0, 0.0), (0.0, 1.0e18))
        estimates = [a, a + 2, outlier]
        errors = [2.0, 2.0, uncertainty]
        measured = MonteCarloIntegration._combine_estimates(estimates, errors)[3]
        keep = findall(!iszero, errors)
        values, sigmas = BigFloat.(estimates[keep]), BigFloat.(errors[keep])
        mean = sum(values ./ sigmas .^ 2) / sum(inv.(sigmas .^ 2))
        reference = sum(abs2, (values .- mean) ./ sigmas) / (length(keep) - 1)
        @test measured ≈ Float64(reference)
    end
end
