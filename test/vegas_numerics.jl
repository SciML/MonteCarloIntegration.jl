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

@testset "Flat batches retain the evaluation budget" begin
    batches = ([1.0, 1.0], [0.0, 0.0], [0.0, 1.0])
    calls = Ref(0)
    result = vegas(
        _ -> batches[calls[] += 1], [0.0], [1.0];
        batch = :matrix, nbins = 2, ncalls = 2, maxiter = 3, alpha = 0.0,
        atol = 1.0e-10, rtol = 1.0e-10,
    )
    @test calls[] == 3
    @test result.integral_estimate == 0.5
    @test result.standard_deviation > 0.0
end

@testset "Jacobian factors with widely separated scales" begin
    small, large = 2.0^-600, 2.0^900
    for (batch, f) in ((false, Returns(1)), (true, x -> ones(length(x))), (:matrix, x -> ones(size(x, 2))))
        for ub in ([small, small, large], [small, large, small], [-small, small, large], [large, large, small, small])
            @testset "batch = $batch, bounds = $ub" begin
                expected = length(ub) == 4 ? 2.0^600 : copysign(2.0^-300, first(ub))
                result = vegas(f, zeros(length(ub)), ub; batch, nbins = 2, ncalls = 16, maxiter = 1, alpha = 0.0)
                @test result.integral_estimate == expected
                @test result.standard_deviation == 0.0
            end
        end
    end
    result = vegas(Returns(2.0^900), [0.0, 0.0], [small, small]; nbins = 2, ncalls = 16, maxiter = 1)
    @test result.integral_estimate == 2.0^-300
    @test result.standard_deviation == 0.0
    @testset "A span larger than binary64" begin
        bound = 2.0^1023
        result = vegas(Returns(1), [-bound, 0.0], [bound, 2.0^-1000]; nbins = 2, ncalls = 16, maxiter = 1)
        @test result.integral_estimate == 2.0^24
        @test result.standard_deviation == 0.0
    end
end

@testset "Cancellation across binary64 scales" begin
    large, tiny = 2.0^1000, 2.0^-500
    expected = tiny / 3
    for bits in (32, 256)
        setprecision(BigFloat, bits) do
            for values in ([large, -large, tiny], [large, tiny, -large], [tiny, -large, large])
                result = vegas(Returns(values), [0.0], [1.0]; batch = :matrix, nbins = 2, ncalls = 3, maxiter = 1)
                @test abs(result.integral_estimate - expected) <= 2eps(expected)
                @test isfinite(result.standard_deviation) && result.standard_deviation > 0.0
            end
        end
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

@testset "Refinement preserves selected coordinate precision" begin
    for (stored, importance) in (
                (Float16[1024, 1025, 1026, 1027, 1028], [1.0, 0.0, 0.0, 0.0]),
                (Float16[-65504, -32752, 0, 32752, 65504], [1.0, 0.0, 0.0, 0.0]),
            ), reversed in (false, true)
        axis = reshape(reversed ? reverse(stored) : stored, :, 1)
        mass = reshape(reversed ? reverse(importance) : importance, :, 1)
        widths = diff(axis; dims = 1)
        edges, spacing = MonteCarloIntegration.update_grid(axis, widths, mass)
        @test edges == axis
        @test spacing == widths
        @test all(x -> isfinite(x) && sign(x) == (reversed ? -1 : 1), spacing)
    end

    for T in (Float16, Float32, Float64)
        axis = reshape(T[0, 0.25, 0.5, 0.75, 1], :, 1)
        widths = diff(axis; dims = 1)
        weights = reshape([1.0, 1.0, 2.0, 4.0], :, 1)
        edges, spacing = MonteCarloIntegration.update_grid(axis, widths, weights)
        @test edges != axis
        @test eltype(edges) === eltype(spacing) === T
        @test spacing == diff(edges; dims = 1)
        @test all(>(0), spacing)
    end

    f(x) = only(x) < 0.5 ? 0.8 : 1.0
    options = (; nbins = 8, ncalls = 65_536, maxiter = 1, rtol = 0.0)
    Random.seed!(817)
    half = vegas(f, Float16[0], Float16[1]; options...)
    Random.seed!(817)
    double = vegas(f, [0.0], [1.0]; options...)
    @test eltype(half.adaptive_grid) === Float16
    # Point and edge rounding at Float16 precision can move the learned edges.
    @test maximum(abs, half.adaptive_grid .- double.adaptive_grid) <= 4eps(Float16)
    @test all(>(0), half.grid_spacing)
end

@testset "VEGAS bin selection at the upper endpoint" begin
    for T in (Float16, Float32, Float64, BigFloat)
        edges = reshape(T[0, 0.25, 0.5, 0.75, 1], :, 1)
        widths = diff(edges; dims = 1)
        units = reshape(T === BigFloat ? BigFloat[0, 1] : [0.0, prevfloat(1.0)], 1, :)
        points = Matrix{T}(undef, 1, 2)
        bins = Matrix{Int}(undef, 1, 2)
        weights = zeros(2)
        MonteCarloIntegration._map_vegas_points!(points, units, bins, weights, nothing, edges, widths)
        @test vec(bins) == [1, 4]
        @test points[1] == 0 && 0.75 <= points[2] <= 1
        @test weights == [1, 1]
    end
end
