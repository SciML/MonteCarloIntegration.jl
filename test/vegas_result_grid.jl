using MonteCarloIntegration
using Random: rand!, seed!
using Test

const Float64GridResult = VEGASResult{Float64, Float64, Float64, Matrix{Float64}, Matrix{Float64}}
const RESULT_OPTIONS = (; nbins = 4, ncalls = 128, maxiter = 2, alpha = 0.0)

@testset "Copies and edits of native grid fields" begin
    setprecision(BigFloat, 128) do
        offset = BigFloat(2)^53
        lo, hi = offset + 1 // 8, offset + 63 // 8
        original = vegas(Returns(1.0), [lo, 0.0], [hi, 1.0]; RESULT_OPTIONS...)
        @test getfield(original, :adaptive_grid) === original.adaptive_grid
        @test getfield(original, :grid_spacing) === original.grid_spacing
        copied = deepcopy(original)
        @test copied.adaptive_grid !== original.adaptive_grid
        @test copied.grid_spacing !== original.grid_spacing
        seed!(913)
        unedited = MonteCarloIntegration.sample_from_adaptive_grid(original, 64)
        seed!(913)
        @test unedited == MonteCarloIntegration.sample_from_adaptive_grid(copied, 64)

        copied.adaptive_grid[:, 2] .= BigFloat[10, 10.125, 10.5, 10.75, 11]
        copied.grid_spacing[:, 2] .= diff(copied.adaptive_grid[:, 2])
        seed!(913)
        edited = MonteCarloIntegration.sample_from_adaptive_grid(copied, 64)
        @test first.(edited) == first.(unedited)
        @test all(p -> 10 < last(p) < 11, edited)
        warm = vegas(Returns(1.0), copied; RESULT_OPTIONS...)
        @test warm.adaptive_grid == copied.adaptive_grid
        @test warm.grid_spacing == copied.grid_spacing
        @test isfinite(warm.integral_estimate)
        @test vegas(Returns(1.0), original; RESULT_OPTIONS...).integral_estimate == 7.75
    end
end

@testset "Native results with unusable Float64 geometry" begin
    setprecision(BigFloat, 128) do
        offset = BigFloat(2)^53
        huge = BigFloat(2)^1025
        cases = (
            (offset - 1 // 4, offset + 1 // 4, 1.0, 0.5),
            (offset + 3 // 4, offset + 5 // 4, 1.0, 0.5),
            (huge, huge + BigFloat(2)^1022, ldexp(1.0, -1022), 1.0),
        )
        for (a, b, value, estimated) in cases
            result = vegas(Returns(value), [a], [b]; RESULT_OPTIONS...)
            @test result.integral_estimate == estimated
            @test result.adaptive_grid isa Matrix{BigFloat}
            @test first(result.adaptive_grid) == a
            @test last(result.adaptive_grid) == b
            samples = MonteCarloIntegration.sample_from_adaptive_grid(result, 64)
            @test all(p -> a <= only(p) <= b, samples)
        end
    end
end

@testset "Public VEGASResult constructors" begin
    edges = reshape([0.0, 0.5, 1.0], :, 1)
    widths = reshape([0.5, 0.5], :, 1)
    for result in (VEGASResult(1.0, 0.0, 0.0, edges, widths), Float64GridResult(1, 0, 0, edges, widths))
        @test result isa Float64GridResult
        @test result.adaptive_grid === edges
        @test result.grid_spacing === widths
        @test propertynames(result) == (:integral_estimate, :standard_deviation, :chi_squared_average, :adaptive_grid, :grid_spacing)
        @test vegas(Returns(1.0), result; ncalls = 128, maxiter = 1).integral_estimate == 1
    end
    converted = Float64GridResult(1, 0, 0, Float32.(edges), Float32.(widths))
    @test converted.adaptive_grid == edges
    @test converted.grid_spacing == widths
    @test fieldcount(Float64GridResult) == 5
end

@testset "Native BigFloat grids survive changes in caller rounding" begin
    settings = (; ncalls = 64, maxiter = 1, alpha = 0.0)
    cases = (
        (128, BigFloat(2; precision = 128)^53, 1 // 8, 81 // 8),
        (192, BigFloat(2; precision = 192)^80, -1 // 8, 97 // 8),
    )
    for (bits, center, low, high) in cases, nbins in (6, 7), direction in (identity, reverse)
        setprecision(BigFloat, bits) do
            lower, upper = direction((center + low, center + high))
            expected = Float64(upper - lower)
            for create_rounding in (RoundNearest, RoundUp, RoundDown)
                original = setrounding(BigFloat, create_rounding) do
                    vegas(Returns(1.0), [lower], [upper]; settings..., nbins)
                end
                for reuse_rounding in (RoundNearest, RoundUp, RoundDown, RoundToZero)
                    @testset "$bits-bit, $nbins bins, $create_rounding → $reuse_rounding, $direction" begin
                        setrounding(BigFloat, reuse_rounding) do
                            observed = Ref(true)
                            calls = Ref(0)
                            warmed = vegas(original; settings...) do point
                                calls[] += 1
                                observed[] &= rounding(BigFloat) === reuse_rounding
                                observed[] &= min(lower, upper) <= only(point) <= max(lower, upper)
                                return 1.0
                            end
                            @test calls[] == settings.ncalls
                            @test observed[]
                            @test warmed.adaptive_grid == original.adaptive_grid
                            @test warmed.grid_spacing == original.grid_spacing
                            @test warmed.integral_estimate ≈ expected rtol = 2eps(Float64)
                            @test iszero(warmed.standard_deviation)
                            @test rounding(BigFloat) === reuse_rounding
                        end
                    end
                end
            end
        end
    end
end

@testset "Adapted BigFloat grids retain their stored geometry across modes" begin
    setprecision(BigFloat, 128) do
        offset = BigFloat(2)^53
        lo, hi = offset + 1 // 8, offset + 81 // 8
        settings = (; ncalls = 128, maxiter = 2, nbins = 7)
        trained = setrounding(BigFloat, RoundDown) do
            seed!(347)
            vegas(point -> exp(only(point) - offset), [lo], [hi]; settings...)
        end
        @test length(unique(trained.grid_spacing)) > 1
        for mode in (RoundNearest, RoundUp, RoundToZero)
            setrounding(BigFloat, mode) do
                reused = vegas(Returns(1.0), trained; settings..., alpha = 0.0)
                @test reused.adaptive_grid == trained.adaptive_grid
                @test reused.grid_spacing == trained.grid_spacing
                @test rounding(BigFloat) === mode
            end
        end
    end
end

@testset "Low-bit BigFloat grids use stored physical spans" begin
    settings = (; ncalls = 4096, maxiter = 2, alpha = 0.0, atol = 0.0, rtol = 0.0)
    for (bits, span) in ((8, 29), (11, 29), (16, 15), (24, 31)), direction in (identity, reverse)
        lo, hi = direction((BigFloat(0; precision = bits), BigFloat(span; precision = bits)))
        expected = direction === identity ? span : -span
        correctly_typed = Ref(true)
        function constant(point)
            correctly_typed[] &= precision(only(point)) == bits
            return one(only(point))
        end
        seed!(826)
        result = vegas(constant, [lo], [hi]; settings..., nbins = 7)
        seed!(826)
        warmed = vegas(constant, result; settings...)
        @test correctly_typed[]
        @test result.adaptive_grid == warmed.adaptive_grid
        @test result.grid_spacing == warmed.grid_spacing
        @test all(x -> precision(x) == bits, result.grid_spacing)
        @test result.integral_estimate == warmed.integral_estimate
        @test result.standard_deviation == warmed.standard_deviation
        @test abs(result.integral_estimate - expected) <= 6result.standard_deviation + eps(Float64(expected))
    end

    bits, nbins = 11, 2500
    lo, hi = BigFloat(-1; precision = bits), BigFloat(1; precision = bits)
    seed!(5928)
    result = vegas(Returns(1.0), [lo], [hi]; settings..., nbins)
    @test length(unique(result.adaptive_grid)) == nbins + 1
    @test first(result.adaptive_grid) == lo
    @test last(result.adaptive_grid) == hi
    @test all(w -> w > 0, result.grid_spacing)
    @test all(x -> precision(x) == bits, result.adaptive_grid)
    @test abs(result.integral_estimate - 2.0) <= 6result.standard_deviation + eps(2.0)
    seed!(5928)
    warmed = vegas(Returns(1.0), result; settings...)
    @test warmed.integral_estimate == result.integral_estimate
    @test warmed.adaptive_grid == result.adaptive_grid
    points = MonteCarloIntegration.sample_from_adaptive_grid(warmed, 128)
    @test all(point -> lo <= only(point) <= hi && precision(only(point)) == bits, points)

    setprecision(BigFloat, bits) do
        ncalls = 65_536
        coordinates = Matrix{BigFloat}(undef, 1, ncalls)
        bins = zeros(Int, 1, ncalls)
        weights = zeros(ncalls)
        grid, widths = warmed.adaptive_grid, warmed.grid_spacing
        MonteCarloIntegration._with_vegas_mapping(grid, widths) do working_widths, native_bits
            seed!(1234)
            rand!(coordinates)
            MonteCarloIntegration._map_vegas_points!(coordinates, coordinates, bins, weights, nothing, grid, working_widths, native_bits)
        end
        @test length(unique(bins)) == nbins
        @test all(x -> precision(x) == bits, coordinates)
    end
end

@testset "BigFloat Jacobian factors round once for Float64 statistics" begin
    setprecision(BigFloat, 53) do
        settings = (; nbins = 7, ncalls = 2048, maxiter = 1, alpha = 0.0, atol = 0.0, rtol = 0.0)
        for extra_scales in (BigFloat[], [BigFloat(2)^1100, BigFloat(2)^(-1100)])
            upper = vcat(ones(BigFloat, 32), extra_scales)
            lower = zeros(BigFloat, length(upper))
            for direction in (identity, reverse)
                a, b = direction((lower, upper))
                target = isodd(count(i -> b[i] < a[i], eachindex(a))) ? -1.0 : 1.0
                seed!(7714)
                result = vegas(Returns(1.0), a, b; settings...)
                @test abs(result.integral_estimate - target) <= max(2eps(Float64), 6result.standard_deviation)
                @test isfinite(result.standard_deviation)
            end
        end
    end
end
