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

@testset "VEGAS preserves Float64 callbacks on resolvable inexact bounds" begin
    offset = BigFloat(2)^-60
    q = BigFloat(nextfloat(0.0))
    largest = floatmax(Float64)
    cases = (
        (BigFloat(0), BigFloat(1) + offset),
        (BigFloat(1) + offset, BigFloat(2) - offset),
        (-109 // 3, -97 // 3),
        (q / 4, 256q + q / 4),
        (BigFloat(1.0e16) + 1 // 4, BigFloat(1.0e16) + 32 + 1 // 4),
        (BigFloat(prevfloat(largest, 8)) + (largest - prevfloat(largest)) / 4, BigFloat(largest)),
    )
    for (lo, hi) in cases, (a, b) in ((lo, hi), (hi, lo)), batch in (false, true, :matrix)
        @testset "$batch, $a → $b" begin
            seen = Float64[]
            point(x::AbstractVector{Float64}) = (push!(seen, only(x)); 1.0)
            views(x::Vector{<:AbstractVector{Float64}}) = map(point, x)
            matrix(x::AbstractMatrix{Float64}) = map(point, eachcol(x))
            integrand = batch === false ? point : (batch === true ? views : matrix)
            options = (; batch, nbins = 4, ncalls = 64, maxiter = 1, alpha = 0.0)
            expected = Float64(BigFloat(b) - BigFloat(a))
            seed!(701)
            result = vegas(integrand, [a], [b]; options...)
            @test eltype(result.adaptive_grid) === BigFloat
            @test result.integral_estimate == expected
            @test vegas(integrand, result; options...).integral_estimate == expected
            @test vegas(integrand, result; options..., callback_type = Float64).integral_estimate == expected
            @test length(seen) == 3 * options.ncalls
            @test all(p -> Rational{BigInt}(lo) <= Rational{BigInt}(p) <= Rational{BigInt}(hi), seen)
        end
    end
end

@testset "VEGAS keeps callback dispatch on nonuniform warm grids" begin
    lo, hi = BigFloat(1) + BigFloat(2)^-60, BigFloat(2) - BigFloat(2)^-60
    edges = reshape(BigFloat[lo, lo + BigFloat(2)^-100, 1.5, 1.75, hi], :, 1)
    previous = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, edges, diff(edges; dims = 1))
    calls = Ref(0)
    inside = Ref(true)
    function f(x::AbstractVector{Float64})
        calls[] += 1
        inside[] &= lo <= only(x) <= hi
        return 1.0
    end
    seed!(702)
    reused = vegas(f, previous; ncalls = 64, maxiter = 1, alpha = 0.0)
    @test calls[] == 64
    @test inside[]
    @test eltype(reused.adaptive_grid) === BigFloat
    @test reused.adaptive_grid == previous.adaptive_grid

    top = 2.0
    gap = top - prevfloat(top)
    lower = BigFloat(prevfloat(top, 2)) - gap / 4
    for (a, b) in ((lower, BigFloat(top)), (BigFloat(top), lower))
        typed(::AbstractVector{Float64}) = 1.0
        result = vegas(typed, [a], [b]; nbins = 2, ncalls = 32, maxiter = 1, alpha = 0.0)
        @test result.integral_estimate == Float64(b - a)
    end
end

@testset "VEGAS keeps Float64 conversions inside mixed callback domains" begin
    offset = BigFloat(2)^-60
    a, b = BigFloat[1 + offset, 1.0e16 + 2], BigFloat[1.5 - offset, 1.0e16]
    settings = (; nbins = 4, ncalls = 128, maxiter = 1, alpha = 0.0)
    seen = NTuple{2, Float64}[]
    function f(x::AbstractVector{Float64})
        push!(seen, Tuple(x))
        return 1.0
    end
    seed!(703)
    fresh = vegas(f, a, b; settings...)
    @test fresh.integral_estimate == -1.0
    @test vegas(f, fresh; settings...).integral_estimate == -1.0
    @test length(seen) == 2 * settings.ncalls
    @test all(p -> a[1] <= p[1] <= b[1] && b[2] <= p[2] <= a[2], seen)

    for (lo, hi, expected) in ((a, b, [nextfloat(1.0), prevfloat(1.5)]), (b, a, [prevfloat(1.5), nextfloat(1.0)]))
        grid, widths = MonteCarloIntegration._initialize_vegas_grid(lo, hi, settings.nbins, nothing)
        coordinate_type, bounds = MonteCarloIntegration._vegas_callback_coordinates(grid, nothing)
        @test coordinate_type === Float64
        points = zeros(2, 2)
        unit_points = [0.0 prevfloat(1.0); 0.0 prevfloat(1.0)]
        weights = zeros(2)
        MonteCarloIntegration._map_vegas_points!(points, unit_points, zeros(Int, 2, 2), weights, nothing, grid, widths, bounds)
        @test points[1, :] == expected
        @test weights == fill(Float64(prod(hi .- lo)), 2)
    end
end

@testset "VEGAS requires useful inward Float64 coverage" begin
    a = BigFloat(2)^53
    q = BigFloat(nextfloat(0.0))
    largest = BigFloat(floatmax(Float64))
    cases = (
        (a + 1 // 8, a + 23 // 4, 1),
        (a + 1 // 8, a + 8 + 1 // 8, 4),
        (q / 4, q / 2, 1),
        (2largest, 2largest + largest / 8, 4),
    )
    for (lo, hi, nbins) in cases, (first, last) in ((lo, hi), (hi, lo))
        seen = BigFloat[]
        function inspect(x::AbstractVector{BigFloat})
            push!(seen, only(x))
            return 1.0
        end
        settings = (; nbins, ncalls = 32, maxiter = 1, alpha = 0.0)
        seed!(706)
        stored = vegas(inspect, [first], [last]; settings...)
        @test stored.integral_estimate == Float64(last - first)
        @test all(p -> lo <= p <= hi, seen)

        seed!(707)
        next_draw = rand(UInt64)
        seed!(707)
        @test_throws ArgumentError vegas(inspect, [first], [last]; settings..., callback_type = Float64)
        @test_throws ArgumentError vegas(inspect, stored; settings..., callback_type = Float64)
        @test rand(UInt64) == next_draw
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
