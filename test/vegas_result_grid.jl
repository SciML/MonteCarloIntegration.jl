using MonteCarloIntegration
using Random: seed!
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
