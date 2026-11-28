using MonteCarloIntegration: MonteCarloIntegration, vegas
using Random: seed!
using Test: @test, @testset, @test_throws

@testset "VEGAS preserves integer bounds above binary64 precision" begin
    offset = 2^53
    options = (; nbins = 2, ncalls = 128, maxiter = 1, alpha = 0.0)
    for (batch, f) in ((false, Returns(1.0)), (true, x -> ones(length(x))), (:matrix, x -> ones(size(x, 2))))
        for (lb, ub) in ((offset + 1, offset + 3), (offset + 3, offset + 1), (offset + 1, offset + 2))
            @testset "batch = $batch, $lb → $ub" begin
                lo, hi = minmax(lb, ub)
                expected = ub - lb
                fresh = vegas(f, [lb], [ub]; options..., batch)
                warm = vegas(f, fresh; options..., batch)
                for result in (fresh, warm)
                    @test result.integral_estimate == expected
                    @test result.standard_deviation == 0.0
                    @test first(result.adaptive_grid) == lb
                    @test last(result.adaptive_grid) == ub
                    @test sum(result.grid_spacing) == expected
                    @test eltype(result.adaptive_grid) === BigFloat
                end
                seed!(57)
                points = MonteCarloIntegration.sample_from_adaptive_grid(warm, 32)
                @test points isa Vector{Tuple{BigFloat}}
                @test all(p -> lo < only(p) < hi, points)
            end
        end
    end
end

@testset "VEGAS preserves legacy callback coordinates when its grid widens" begin
    typed_point(::AbstractVector{Float64}) = 1.0
    typed_batch(points::Vector{<:AbstractVector{Float64}}) = ones(length(points))
    typed_matrix(points::Matrix{Float64}) = ones(size(points, 2))
    dual_point(::AbstractVector{Float64}) = 1.0
    dual_point(::AbstractVector{BigFloat}) = 2.0
    dual_batch(points::Vector{<:AbstractVector{Float64}}) = ones(length(points))
    dual_batch(points::Vector{<:AbstractVector{BigFloat}}) = fill(2.0, length(points))
    dual_matrix(points::Matrix{Float64}) = ones(size(points, 2))
    dual_matrix(points::Matrix{BigFloat}) = fill(2.0, size(points, 2))

    lo, hi = 1.0e16, 1.0e16 + 16
    options = (; nbins = 4, ncalls = 64, maxiter = 1, alpha = 0.0)
    for (batch, typed, dual) in ((false, typed_point, dual_point), (true, typed_batch, dual_batch), (:matrix, typed_matrix, dual_matrix))
        for (a, b) in ((lo, hi), (hi, lo))
            for integrand in (typed, x -> typed(x), dual)
                seed!(700)
                fresh = vegas(integrand, [a], [b]; options..., batch)
                @test fresh.integral_estimate == b - a
                @test eltype(fresh.adaptive_grid) === BigFloat
                @test vegas(integrand, fresh; options..., batch).integral_estimate == b - a
            end
            wide = vegas(dual, [a], [b]; options..., batch, callback_type = BigFloat)
            @test wide.integral_estimate == 2 * (b - a)
            @test vegas(dual, wide; options..., batch, callback_type = BigFloat).integral_estimate == 2 * (b - a)
            @test vegas(dual, wide; options..., batch, callback_type = Float64).integral_estimate == b - a
        end
    end
end

@testset "VEGAS supports wide callback coordinates explicitly" begin
    lo = 1.0e16
    hi = nextfloat(lo)
    options = (; nbins = 2, ncalls = 64, maxiter = 1, alpha = 0.0)
    for (a, b) in ((lo, hi), (hi, lo))
        legacy_points = Float64[]
        function record(point::AbstractVector{Float64})
            push!(legacy_points, only(point))
            return 1.0
        end
        seed!(704)
        legacy = vegas(record, [a], [b]; options...)
        @test legacy.integral_estimate == b - a
        @test all(p -> p == lo || p == hi, legacy_points)
        seen = BigFloat[]
        function inspect(point::AbstractVector{BigFloat})
            push!(seen, only(point))
            return 1.0
        end
        seed!(704)
        result = vegas(inspect, [a], [b]; options..., callback_type = BigFloat)
        @test result.integral_estimate == b - a
        @test all(p -> lo < p < hi, seen)
        @test length(unique(seen)) > 2
        empty!(seen)
        @test vegas(inspect, result; options..., callback_type = BigFloat).integral_estimate == b - a
        @test all(p -> lo < p < hi, seen)
        @test length(unique(seen)) > 2
    end

    typed_wide(::AbstractVector{BigFloat}) = 1.0
    ordinary = vegas(typed_wide, [0.0], [1.0]; options..., callback_type = BigFloat)
    @test ordinary.integral_estimate == 1.0
    @test eltype(ordinary.adaptive_grid) === Float64
end

@testset "VEGAS rejects unrepresentable callback bounds before drawing samples" begin
    lo, hi = big(2)^53 + 1, big(2)^53 + 3
    options = (; nbins = 2, ncalls = 64, maxiter = 1, alpha = 0.0)
    typed_wide(point::AbstractVector{BigFloat}) = lo <= only(point) <= hi
    exact = vegas(typed_wide, [lo], [hi]; options...)
    @test exact.integral_estimate == hi - lo
    @test vegas(typed_wide, exact; options...).integral_estimate == hi - lo
    calls = Ref(0)
    function integrand(_)
        calls[] += 1
        return 1.0
    end
    seed!(705)
    next_draw = rand(UInt64)
    seed!(705)
    @test_throws ArgumentError vegas(integrand, [lo], [hi]; options..., callback_type = Float64)
    @test_throws ArgumentError vegas(integrand, exact; options..., callback_type = Float64)
    @test_throws ArgumentError vegas(integrand, [0.0], [1.0]; options..., callback_type = Float32)
    setprecision(BigFloat, 16) do
        @test_throws ArgumentError vegas(integrand, [0.1], [1.0]; options..., callback_type = BigFloat)
    end
    @test iszero(calls[])
    @test rand(UInt64) == next_draw
end
