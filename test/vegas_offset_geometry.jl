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

@testset "VEGAS resolves a polynomial on an inexact offset interval" begin
    offset = BigFloat(2)^53
    lo, hi = offset + 1 // 8, offset + 63 // 8
    function polynomial(x)
        t = only(x) - offset - 4
        return abs2(t * (t * t - 4))
    end
    wide_polynomial(x::AbstractVector{BigFloat}) = polynomial(x)
    t = big(31) // 8
    exact = Float64(2 * (t^7 / 7 - 8t^5 / 5 + 16t^3 / 3))
    options = (; nbins = 1, ncalls = 32_768, maxiter = 1, alpha = 0.0)
    seed!(440)
    fresh = vegas(wide_polynomial, [lo], [hi]; options...)
    seed!(440)
    reused = vegas(wide_polynomial, fresh; options...)
    seed!(440)
    explicit = vegas(polynomial, [lo], [hi]; options...)
    @test fresh.standard_deviation > 0
    @test abs(fresh.integral_estimate - exact) < 4 * fresh.standard_deviation
    @test fresh.integral_estimate == reused.integral_estimate == explicit.integral_estimate
    @test fresh.standard_deviation == reused.standard_deviation == explicit.standard_deviation
end

@testset "VEGAS integrates periodic functions on precise offset grids" begin
    offset = BigFloat(2)^53
    lo = offset + 1 // 8
    options = (; nbins = 4, ncalls = 16_384, maxiter = 1, alpha = 0.0)
    for width in (10, 6), batch in (false, true, :matrix)
        hi = lo + width
        points = BigFloat[]
        function periodic(x::AbstractVector{BigFloat})
            p = only(x)
            push!(points, p)
            return abs2(sinpi((p - offset) / 2))
        end
        f = batch === false ? periodic : batch === true ? xs -> map(periodic, xs) : xs -> map(periodic, eachcol(xs))
        seed!(440)
        fresh = vegas(f, [lo], [hi]; options..., batch)
        seed!(440)
        warm = vegas(f, fresh; options..., batch)
        @test length(points) == 2options.ncalls
        @test all(p -> lo <= p <= hi, points)
        @test fresh.integral_estimate == warm.integral_estimate
        @test abs(fresh.integral_estimate - width / 2) < 6width / sqrt(8options.ncalls)
        @test fresh.standard_deviation == warm.standard_deviation > 0
    end
end

@testset "VEGAS uses promoted callback types on inexact and offset bounds" begin
    offset = BigFloat(2)^53
    q = BigFloat(nextfloat(0.0))
    top = floatmax(Float64)
    settings = (; nbins = 4, ncalls = 64, maxiter = 1, alpha = 0.0)
    cases = (
        (BigFloat(0), BigFloat(1) + BigFloat(2)^-60),
        (BigFloat(1) + BigFloat(2)^-60, BigFloat(2) - BigFloat(2)^-60),
        (offset + 1 // 8, offset + 63 // 8),
        (q / 4, 256q + q / 4),
        (BigFloat(1.0e16) + 1 // 4, BigFloat(1.0e16) + 32 + 1 // 4),
        (BigFloat(prevfloat(top, 8)) + (BigFloat(top) - prevfloat(top)) / 4, BigFloat(top)),
        (Float32(-109 / 3), Float32(-97 / 3)),
        (1.0e16, 1.0e16 + 16),
    )
    for (lo, hi) in cases, (a, b) in ((lo, hi), (hi, lo)), batch in (false, true, :matrix)
        observed = 0
        inside = true
        T = typeof(a)
        function inspect(point)
            observed += 1
            inside &= lo <= only(point) <= hi && eltype(point) === T
            return 1.0
        end
        f = batch === false ? inspect : batch === true ? xs -> map(inspect, xs) : xs -> map(inspect, eachcol(xs))
        fresh = vegas(f, [a], [b]; settings..., batch)
        warm = vegas(f, fresh; settings..., batch)
        @test observed == 2settings.ncalls
        @test inside
        @test fresh.integral_estimate == warm.integral_estimate == Float64(BigFloat(b) - BigFloat(a))
        @test fresh.adaptive_grid isa Matrix{T}
        @test warm.adaptive_grid isa Matrix{T}
    end
end

@testset "VEGAS preserves nonuniform warm grid precision" begin
    lo, hi = BigFloat(1) + BigFloat(2)^-60, BigFloat(2) - BigFloat(2)^-60
    edges = reshape(BigFloat[lo, lo + BigFloat(2)^-100, 1.5, 1.75, hi], :, 1)
    previous = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, edges, diff(edges; dims = 1))
    calls = Ref(0)
    inside = Ref(true)
    function f(x::AbstractVector{BigFloat})
        calls[] += 1
        inside[] &= lo <= only(x) <= hi
        return 1.0
    end
    setprecision(BigFloat, 64) do
        reused = vegas(f, previous; ncalls = 64, maxiter = 1, alpha = 0.0)
        @test calls[] == 64
        @test inside[]
        @test reused.adaptive_grid == edges
        @test reused.grid_spacing == previous.grid_spacing
        @test precision(first(reused.adaptive_grid)) == precision(lo)
        @test precision(BigFloat) == 64
    end
end

@testset "VEGAS rejects unresolved explicit precision before evaluating" begin
    for T in (Float16, Float32, Float64, BigFloat)
        lo, hi = one(T), nextfloat(one(T))
        calls = Ref(0)
        integrand(_) = (calls[] += 1; 1.0)
        seed!(583)
        next_draw = rand(UInt64)
        seed!(583)
        @test_throws ArgumentError vegas(integrand, [lo], [hi]; nbins = 2, ncalls = 64, maxiter = 1)
        @test calls[] == 0
        @test rand(UInt64) == next_draw
    end
end
