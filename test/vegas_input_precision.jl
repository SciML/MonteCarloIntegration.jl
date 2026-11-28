using MonteCarloIntegration
using Random: seed!
using Test

@testset "Coordinate and grid precision follow the bounds" begin
    @testset "VEGAS input coordinate precision" begin
        options = (; nbins = 2, ncalls = 128, maxiter = 2, alpha = 0.0)
        for (T, bounds) in (
                    (Float16, (Float16[0], Float16[1])),
                    (Float32, (Float32[0], Float32[1])),
                    (Float32, (Float32[0], [1])),
                    (Float32, (Float32[0], [big(1)])),
                    (Float64, (Float32[0], Float64[1])),
                    (Float64, ([0], [1])),
                    (Float64, ([big(0)], [big(1) // 1])),
                ), batch in (false, true, :matrix)
            seen = DataType[]
            function observe(x)
                push!(seen, eltype(x))
                return 1.0
            end
            f = batch === false ? observe : batch === true ? xs -> map(observe, xs) : xs -> map(observe, eachcol(xs))
            fresh = vegas(f, bounds...; options..., batch)
            warm = vegas(f, fresh; options..., batch)
            @test seen == fill(T, 2 * options.ncalls * options.maxiter)
            @test fresh.integral_estimate == warm.integral_estimate == 1.0
            for result in (fresh, warm)
                @test getfield(result, :adaptive_grid) isa Matrix{T}
                @test getfield(result, :grid_spacing) isa Matrix{T}
                seed!(127)
                samples = MonteCarloIntegration.sample_from_adaptive_grid(result, 64)
                @test samples isa Vector{Tuple{T}}
                @test all(p -> 0 <= only(p) <= 1, samples)
            end
        end
    end

    @testset "BigFloat bounds supply bits for fresh, warm, and direct samples" begin
        opts = (; nbins = 1, ncalls = 128, maxiter = 1, alpha = 0.0)
        lower = BigFloat(0; precision = 96)
        upper = BigFloat(1; precision = 128)
        for ambient in (64, 256), batch in (false, true, :matrix)
            setprecision(BigFloat, ambient) do
                received = Tuple{DataType, Int}[]
                wide_samples = Ref(0)
                function observe(point)
                    x = only(point)
                    push!(received, (typeof(x), precision(x)))
                    wide_samples[] += x != Float64(x)
                    return 1.0
                end
                f = batch === false ? observe : batch === true ? xs -> map(observe, xs) : xs -> map(observe, eachcol(xs))
                seed!(922)
                result = vegas(f, [lower], [upper]; opts..., batch)
                seed!(922)
                warm = vegas(f, result; opts..., batch)
                @test received == fill((BigFloat, 128), 2opts.ncalls)
                @test wide_samples[] > opts.ncalls
                @test result.integral_estimate == warm.integral_estimate == 1
                @test result.integral_estimate isa Float64
                @test all(x -> precision(x) == 128, result.adaptive_grid)
                @test all(x -> precision(x) == 128, warm.grid_spacing)
                seed!(651)
                samples = MonteCarloIntegration.sample_from_adaptive_grid(warm, 64)
                @test samples isa Vector{Tuple{BigFloat}}
                @test all(x -> precision(only(x)) == 128, samples)
                @test count(x -> only(x) != Float64(only(x)), samples) > 32
                @test precision(BigFloat) == ambient
            end
        end
    end

    @testset "Representable native grids and explicit narrow bounds" begin
        for T in (Float16, Float32, Float64)
            a, b = -floatmax(T), floatmax(T)
            scale = 1.0 / Float64(b)
            options = (; ncalls = 32, maxiter = 1, alpha = 0.0)
            result = vegas(Returns(scale), [a], [b]; options..., nbins = 8)
            # Large rounded edges can give the native bins slightly different widths.
            @test abs(result.integral_estimate - 2.0) <= 6result.standard_deviation + 2eps(T)
            @test sum(BigFloat, result.grid_spacing) == 2BigFloat(b)
            @test result.adaptive_grid isa Matrix{T}
            @test all(isfinite, result.grid_spacing)
            nearby = nextfloat(one(T))
            @test_throws ArgumentError vegas(Returns(1.0), [one(T)], [nearby]; options..., nbins = 2)
            narrow = vegas(Returns(1.0), [one(T)], [nearby]; options..., nbins = 1)
            @test narrow.integral_estimate == Float64(nearby - one(T))
            @test narrow.adaptive_grid isa Matrix{T}
        end
        wide = (big(1) << 300) + 1
        result = vegas(Returns(1.0), [wide], [wide + 2]; nbins = 4, ncalls = 32, maxiter = 1, alpha = 0)
        @test result.integral_estimate == 2.0
        @test result.adaptive_grid isa Matrix{BigFloat}
        @test first(result.adaptive_grid) == wide
        @test last(result.adaptive_grid) == wide + 2
    end

    @testset "Nested BigFloat callbacks restore the outer precision" begin
        options = (; nbins = 1, ncalls = 4, maxiter = 1, alpha = 0.0)
        setprecision(BigFloat, 64) do
            for inner_bits in (112, 160)
                called = Ref(false)
                function integrand(point)
                    @test precision(only(point)) == precision(BigFloat) == 112
                    if !called[]
                        called[] = true
                        child = @async begin
                            bounds = (BigFloat[BigFloat(0; precision = inner_bits)], BigFloat[BigFloat(1; precision = inner_bits)])
                            vegas(bounds...; options...) do coordinates
                                @test precision(only(coordinates)) == precision(BigFloat) == inner_bits
                                return 1.0
                            end
                        end
                        @test fetch(child).integral_estimate == 1.0
                        @test precision(BigFloat) == 112
                    end
                    return 1.0
                end
                bounds = ([BigFloat(0; precision = 112)], [BigFloat(1; precision = 112)])
                @test vegas(integrand, bounds...; options...).integral_estimate == 1.0
                @test called[]
                @test precision(BigFloat) == 64
            end
        end
    end

    @testset "Adaptation retains a grid when a proposed width overflows" begin
        for T in (Float16, Float32, Float64)
            bound = floatmax(T)
            edges = T[-bound, -bound / 2, 0, bound / 2, bound]
            widths = diff(edges)
            original = copy(edges)
            MonteCarloIntegration._rebin!(edges, widths, [0.001, 0.001, 0.001, 1.0], similar(widths))
            @test edges == original
            @test widths == diff(original)
        end
    end
end
