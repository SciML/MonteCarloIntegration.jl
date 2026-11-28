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

@testset "Float64 mapping follows the stored edge spans" begin
    for (a, steps, ncalls, seed, integrand, expected) in (
            (2.0^53, [0, 2, 5, 7], 500_000, 41, x -> only(x) - 2.0^53, 98.0),
            (2.0^53, [0, 4, 7, 11], 100_000, 92, x -> Float64(only(x) == 2.0^53 + 16), 2.0),
            (1.0, [0, 3, 5, 8], 200_000, 12, x -> ((only(x) - 1.0) / eps(1.0) - 4)^3 / (8eps(1.0)), 0.0),
            (2.0^52, [0, 4, 9, 14, 18], 100_000, 744, x -> Float64(only(x) == 2.0^52 + 4), 1.0),
        )
        nbins = length(steps) - 1
        opts = (; ncalls, maxiter = 1, alpha = 0.0)
        @testset "$a, $steps, reversed $reversed" for reversed in (false, true)
            expected_edges = reshape(a .+ eps(a) .* (reversed ? reverse(steps) : steps), :, 1)
            a0, b0 = first(expected_edges), last(expected_edges)
            expected_widths = diff(expected_edges; dims = 1)
            seed!(seed)
            fresh = vegas(integrand, [a0], [b0]; nbins, opts...)
            @test fresh.adaptive_grid == expected_edges
            @test fresh.grid_spacing == expected_widths

            unit = reshape([(i - fraction) / nbins for i in 1:nbins for fraction in (0.75, 0.25)], 1, :)
            points = similar(unit)
            bins, weights = zeros(Int, size(unit)), zeros(length(unit))
            MonteCarloIntegration._map_vegas_points!(
                points, unit, bins, weights, nothing, fresh.adaptive_grid, fresh.grid_spacing,
            )
            @test vec(bins) == repeat(1:nbins; inner = 2)
            @test weights == repeat(vec(nbins .* expected_widths); inner = 2)
            for j in eachindex(points)
                bin = bins[j]
                left, right = expected_edges[bin], expected_edges[bin + 1]
                fraction = BigFloat(unit[j] * nbins - (bin - 1))
                @test points[j] == Float64(BigFloat(left) + fraction * (BigFloat(right) - BigFloat(left)))
                @test min(left, right) <= points[j] <= max(left, right)
            end

            nominal_widths = fill((b0 - a0) / nbins, nbins, 1)
            legacy = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, copy(expected_edges), copy(nominal_widths))
            physical = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, expected_edges, expected_widths)
            seed!(seed)
            warm = vegas(integrand, legacy; opts...)
            @test warm.grid_spacing == expected_widths
            @test warm.adaptive_grid == expected_edges
            @test warm.integral_estimate == fresh.integral_estimate
            @test warm.standard_deviation == fresh.standard_deviation
            target = reversed ? -expected : expected
            for estimate in (fresh, warm)
                @test abs(estimate.integral_estimate - target) <= 6estimate.standard_deviation
            end

            seed!(seed)
            samples = MonteCarloIntegration.sample_from_adaptive_grid(legacy, 1024)
            next_number = rand(UInt64)
            seed!(seed)
            @test samples == MonteCarloIntegration.sample_from_adaptive_grid(physical, 1024)
            @test rand(UInt64) == next_number
            @test samples isa Vector{Tuple{Float64}}
            @test legacy.adaptive_grid == expected_edges
            @test legacy.grid_spacing == nominal_widths
        end
    end
end

@testset "Physical bin spans determine multidimensional Float64 Jacobians" begin
    a, b = 2.0^53, 2.0^53 + 14
    ndim, nbins = 32, 3
    lo, hi = fill(a, ndim), fill(b, ndim)
    lo[1], hi[1] = b, a
    result = vegas(Returns(0), lo, hi; nbins, ncalls = 4, maxiter = 1, alpha = 0.0)
    units = repeat([1 / 6 0.66 5 / 6], ndim, 1)
    points, bins = similar(units), zeros(Int, size(units))
    expected = Float64.(-big.([12, 18, 12]) .^ ndim)
    for exponents in (nothing, zeros(Int, nbins))
        weights = zeros(nbins)
        MonteCarloIntegration._map_vegas_points!(
            points, units, bins, weights, exponents, result.adaptive_grid, result.grid_spacing,
        )
        jacobians = exponents === nothing ? weights : ldexp.(weights, exponents)
        @test all(isapprox.(jacobians, expected; rtol = 8eps(Float64), atol = 0.0))
        @test bins == repeat([1 2 3], ndim, 1)
    end
end
