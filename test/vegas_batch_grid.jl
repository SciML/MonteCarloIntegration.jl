using MonteCarloIntegration: MonteCarloIntegration, vegas
using Random: Random
using Test: @test, @testset, @test_throws

@testset "VEGAS batch inputs" begin
    lb = [0.0, 1.0]
    ub = [1.0, 3.0]
    ncalls = 128
    maxiter = 3
    options = (; nbins = 10, ncalls, maxiter, rtol = 0.0, atol = 0.0)

    calls = 0
    legacy_calls = 0
    matrix_buffers = Matrix{Float64}[]
    scalar(points) = (calls += 1; sum(points))
    function legacy(points)
        legacy_calls += 1
        @test points isa Vector{<:SubArray}
        @test length(points) == ncalls
        @test all(p -> length(p) == length(lb), points)
        return map(sum, points)
    end
    function matrix(points)
        @test points isa Matrix{Float64}
        @test size(points) == (length(lb), ncalls)
        push!(matrix_buffers, points)
        return vec(sum(points; dims = 1))
    end

    Random.seed!(2121)
    scalar_result = vegas(scalar, lb, ub; options...)
    Random.seed!(2121)
    legacy_result = vegas(legacy, lb, ub; options..., batch = true)
    Random.seed!(2121)
    matrix_result = vegas(matrix, lb, ub; options..., batch = :matrix)
    @test calls == ncalls * maxiter
    @test legacy_calls == length(matrix_buffers) == maxiter
    @test all(p -> p === first(matrix_buffers), matrix_buffers)
    @test scalar_result.integral_estimate ≈ 5.0 rtol = 0.05
    for result in (legacy_result, matrix_result)
        @test result.integral_estimate == scalar_result.integral_estimate
        @test result.standard_deviation == scalar_result.standard_deviation
        @test result.adaptive_grid == scalar_result.adaptive_grid
    end

    @test_throws ArgumentError vegas(error, lb, ub; options..., batch = :unknown)
    @test_throws DimensionMismatch vegas(x -> zeros(size(x, 2) - 1), lb, ub; options..., batch = :matrix)
    @test_throws ArgumentError vegas(x -> zeros(1, size(x, 2)), lb, ub; options..., batch = :matrix)
end

@testset "VEGAS adaptive-grid reuse" begin
    lb = [0.0, 1.0]
    ub = [1.0, 3.0]
    nbins = 10
    ncalls = 128
    peak(x) = exp(-25 * (x[1] - 0.1)^2) * x[2]
    Random.seed!(2116)
    trained = vegas(peak, lb, ub; nbins, ncalls, maxiter = 3, rtol = 0.0, atol = 0.0)
    @test vegas(sum, trained; ncalls, maxiter = 1).integral_estimate ≈ 5.0 rtol = 0.2
    edges = copy(trained.adaptive_grid)
    widths = copy(trained.grid_spacing)
    points = Matrix{Float64}(undef, length(lb), ncalls)
    function integrand(samples)
        points .= samples
        return samples[2, :]
    end

    Random.seed!(2117)
    fresh = vegas(integrand, lb, ub; nbins, ncalls, maxiter = 1, batch = :matrix)
    fresh_points = copy(points)
    Random.seed!(2117)
    reused = vegas(integrand, trained; ncalls, maxiter = 1, batch = :matrix)
    @test points != fresh_points
    @test reused.integral_estimate ≈ fresh.integral_estimate rtol = 0.15
    weighted_values = [
        points[2, j] * prod(
                nbins * widths[searchsortedlast(view(edges, :, dim), points[dim, j]), dim]
                for dim in eachindex(lb)
            )
            for j in 1:ncalls
    ]
    expected_mean = sum(weighted_values) / ncalls
    expected_variance = sum(v -> abs2(v - expected_mean), weighted_values) / (ncalls * (ncalls - 1))
    @test reused.integral_estimate ≈ expected_mean
    @test reused.standard_deviation ≈ sqrt(expected_variance)
    @test trained.adaptive_grid == edges
    @test trained.grid_spacing == widths
    @test reused.adaptive_grid !== trained.adaptive_grid
    @test reused.grid_spacing !== trained.grid_spacing
    @test_throws ArgumentError vegas(sum, trained; nbins = nbins + 1, ncalls)

    random_points = MonteCarloIntegration.sample_from_adaptive_grid(trained, ncalls)
    @test random_points isa Vector{NTuple{2, Float64}}
    @test length(random_points) == ncalls
    @test all(p -> all(dim -> lb[dim] <= p[dim] <= ub[dim], eachindex(lb)), random_points)

    for (bad_edges, bad_widths) in (
            (edges[1:(end - 1), :], widths),
            (edges, zeros(size(widths))),
            (reverse(edges; dims = 1), widths),
            (fill(NaN, size(edges)), widths),
        )
        malformed = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, bad_edges, bad_widths)
        @test_throws ArgumentError vegas(sum, malformed; ncalls)
    end

    Random.seed!(2118)
    reversed = vegas(x -> 1, [ub[1], lb[2]], [lb[1], ub[2]]; nbins, ncalls, maxiter = 1, alpha = 0.0)
    reversed_reuse = vegas(x -> 2, reversed; ncalls, maxiter = 1)
    @test reversed_reuse.integral_estimate ≈ -4.0

    collapsed = vegas(sum, lb, [lb[1], ub[2]]; nbins, ncalls, maxiter = 1, alpha = 0.0)
    @test iszero(vegas(sum, collapsed; ncalls, maxiter = 1, alpha = 0.0).integral_estimate)
end

@testset "VEGAS zero-valued warm start" begin
    Random.seed!(2122)
    previous = vegas(x -> 1 + x[1], [0.0], [1.0]; nbins = 2, ncalls = 128, maxiter = 2)
    result = vegas(Returns(0.0), previous; ncalls = 128, maxiter = 2)
    @test iszero(result.integral_estimate)
    @test result.adaptive_grid == previous.adaptive_grid
    @test result.grid_spacing == previous.grid_spacing
end

@testset "VEGAS grid with rounded edges" begin
    lb = 1.0e16
    ub = nextfloat(lb, 2)
    nbins = 16
    edges = reshape(collect(range(lb, ub; length = nbins + 1)), :, 1)
    widths = fill((ub - lb) / nbins, nbins, 1)
    @test any(iszero, diff(edges; dims = 1))
    previous = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, edges, widths)
    result = vegas(Returns(1), previous; ncalls = 32, maxiter = 1, alpha = 0.0)
    @test result.integral_estimate ≈ ub - lb

    inconsistent = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, edges, 2 .* widths)
    @test_throws ArgumentError vegas(Returns(1), inconsistent; ncalls = 32)

    short_edges = reshape([lb, nextfloat(lb), ub], :, 1)
    excessive_widths = fill(10.0, 2, 1)
    invalid = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, short_edges, excessive_widths)
    @test_throws ArgumentError vegas(Returns(1), invalid; ncalls = 32)

    shifted_widths = reshape([0.5, 3.5], :, 1)
    shifted = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, short_edges, shifted_widths)
    @test_throws ArgumentError vegas(Returns(1), shifted; ncalls = 32)

    snapped_edges = reshape([lb, lb, nextfloat(lb)], :, 1)
    snapped_widths = reshape([0.0, nextfloat(lb) - lb], :, 1)
    snapped = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, snapped_edges, snapped_widths)
    Random.seed!(2120)
    accepted = vegas(Returns(1), snapped; ncalls = 512, maxiter = 1, alpha = 0.0)
    @test accepted.integral_estimate ≈ nextfloat(lb) - lb rtol = 0.2
end

@testset "VEGAS adaptation of rounded edges" begin
    lower = 1.0e16
    upper = nextfloat(lower, 6)
    nbins = 9
    peak(x) = x[1] == lower + 4.0 ? 1.0e6 : 1.0
    for (lb, ub) in ((lower, upper), (upper, lower))
        edges = reshape(collect(range(lb, ub; length = nbins + 1)), :, 1)
        widths = fill((ub - lb) / nbins, nbins, 1)
        previous = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, edges, widths)
        for domain in (([lb], [ub]), (previous,))
            @testset "Reversed: $(lb > ub), reused: $(length(domain) == 1)" begin
                Random.seed!(1)
                trained = vegas(peak, domain...; nbins, ncalls = 128, maxiter = 1)
                @test issorted(vec(trained.adaptive_grid); rev = lb > ub)
                @test all(w -> sign(ub - lb) * w >= 0, trained.grid_spacing)
                Random.seed!(2)
                reused = vegas(Returns(1), trained; ncalls = 2048, maxiter = 2)
                @test reused.integral_estimate ≈ ub - lb rtol = 0.05
            end
        end
    end
end

@testset "VEGAS sampling stays within rounded bounds" begin
    lower, upper = prevfloat(-2.0^53), -2.0^53
    for (lb, ub) in ((lower, upper), (upper, lower))
        @testset "Reversed: $(lb > ub)" begin
            fresh = vegas(Returns(0.0), [lb], [ub]; nbins = 3, ncalls = 16, maxiter = 1)
            width = (ub - lb) / 3
            edges = reshape([fma(i, width, lb) for i in 0:3], :, 1)
            historical = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, edges, fill(width, 3, 1))
            function bounded(p)
                x = only(p)
                return sqrt(x - lower) + sqrt(upper - x)
            end
            for (name, previous) in (("fresh", fresh), ("historical", historical))
                @testset "$name grid" begin
                    Random.seed!(1)
                    points = MonteCarloIntegration.sample_from_adaptive_grid(previous, 1000)
                    @test all(p -> lower <= only(p) <= upper, points)
                    Random.seed!(1)
                    reused = vegas(bounded, previous; ncalls = 1000, maxiter = 1, alpha = 0.0)
                    @test isfinite(reused.integral_estimate)
                end
            end
        end
    end
end

@testset "VEGAS warm starts at extreme integrand scales" begin
    Random.seed!(1)
    trained = vegas(Returns(1.0), [0.0], [1.0]; nbins = 4, ncalls = 32, maxiter = 1, alpha = 0.0)
    for scale in (1.0e-200, 1.0e200)
        @testset "Scale: $scale" begin
            Random.seed!(1)
            result = vegas(Returns(scale), trained; ncalls = 128, maxiter = 2, rtol = 0.0, atol = 0.0)
            # With adaptation, even a constant integrand has random Jacobian weights.
            @test result.integral_estimate / scale ≈ 1.0 rtol = 0.02
            @test isfinite(result.standard_deviation)
            @test isfinite(result.chi_squared_average)
            @test all(isfinite, result.adaptive_grid)
            @test all(isfinite, result.grid_spacing)
        end
    end
end

@testset "VEGAS warm starts with low-precision bounds" begin
    options = (; nbins = 3, ncalls = 32, maxiter = 1)
    previous = vegas(Returns(0), Float32[0], Float32[1]; options...)
    @test sum(previous.grid_spacing) ≈ 1.0
    result = vegas(Returns(1), previous; ncalls = options.ncalls, maxiter = options.maxiter, alpha = 0.0)
    @test result.integral_estimate ≈ 1.0

    width = 1.0f0 / 3
    edges = reshape(Float64[i * width for i in 0:3], :, 1)
    widths = fill(Float64(width), 3, 1)
    historical = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, edges, widths)
    reused = vegas(Returns(1), historical; ncalls = options.ncalls, maxiter = options.maxiter, alpha = 0.0)
    @test reused.integral_estimate ≈ 1.0
    @test edges[2] == Float64(width)
    @test reused.grid_spacing ≈ fill(1 / 3, 3, 1) rtol = 2eps()
end
