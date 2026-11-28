using MonteCarloIntegration
using Random: seed!
using Test

const Float64GridResult = VEGASResult{Float64, Float64, Float64, Matrix{Float64}, Matrix{Float64}}
const RESULT_OPTIONS = (; nbins = 4, ncalls = 128, maxiter = 2, alpha = 0.0)

consume_grid(edges::Matrix{Float64}, widths::Matrix{Float64}) = (size(edges), size(widths))
consume_result(result::Float64GridResult) = consume_grid(result.adaptive_grid, result.grid_spacing)

@testset "Float64 result dispatch and exact stored grid" begin
    setprecision(BigFloat, 128) do
        offset = BigFloat(2)^53
        lo, hi = offset + 1 // 8, offset + 63 // 8
        scalar(x::AbstractVector{Float64}) = 1.0
        vectors(xs::Vector{<:AbstractVector{Float64}}) = map(scalar, xs)
        matrix(xs::Matrix{Float64}) = map(scalar, eachcol(xs))
        for (batch, integrand) in ((false, scalar), (true, vectors), (:matrix, matrix)), (a, b) in ((lo, hi), (hi, lo))
            options = (; RESULT_OPTIONS..., batch, callback_type = Float64)
            rounded = vegas(integrand, [a], [b]; options...)
            @test rounded isa Float64GridResult
            @test consume_result(rounded) == ((5, 1), (4, 1))
            @test first(rounded.adaptive_grid) == Float64(a)
            @test last(rounded.adaptive_grid) == Float64(b)
            @test rounded.integral_estimate == b - a
            native = vegas(integrand, [a], [b]; options..., grid_precision = :native)
            @test native.adaptive_grid isa Matrix{BigFloat}
            @test first(native.adaptive_grid) == a
            @test last(native.adaptive_grid) == b
            @test sum(native.grid_spacing) == b - a

            seed!(917)
            sampled = MonteCarloIntegration.sample_from_adaptive_grid(rounded, 100)
            seed!(917)
            native_samples = MonteCarloIntegration.sample_from_adaptive_grid(native, 100)
            @test sampled isa Vector{Tuple{Float64}}
            @test native_samples isa Vector{Tuple{BigFloat}}
            @test sampled == map(p -> (clamp(Float64(only(p)), Float64(lo, RoundUp), Float64(hi, RoundDown)),), native_samples)
            @test all(p -> lo < only(p) < hi, sampled)
            wide(x::AbstractVector{BigFloat}) = 1 + only(x) - offset
            seed!(414)
            warm = vegas(wide, rounded; RESULT_OPTIONS..., grid_precision = :native)
            seed!(414)
            fresh = vegas(wide, native; RESULT_OPTIONS..., grid_precision = :native)
            @test warm.integral_estimate == fresh.integral_estimate
            @test warm.standard_deviation == fresh.standard_deviation
            @test warm.adaptive_grid == native.adaptive_grid
        end
    end
end

@testset "Deep copies and edits of rounded axes" begin
    setprecision(BigFloat, 128) do
        offset = BigFloat(2)^53
        lo, hi = offset + 1 // 8, offset + 63 // 8
        options = (; RESULT_OPTIONS..., callback_type = Float64)
        original = vegas(Returns(1.0), [lo, 0.0], [hi, 1.0]; options...)
        copied = deepcopy(original)
        @test copied.adaptive_grid !== original.adaptive_grid
        @test copied.grid_spacing !== original.grid_spacing
        seed!(913)
        unedited = MonteCarloIntegration.sample_from_adaptive_grid(original, 64)
        seed!(913)
        @test unedited == MonteCarloIntegration.sample_from_adaptive_grid(copied, 64)

        copied.adaptive_grid[:, 2] .+= 10
        seed!(913)
        edited = MonteCarloIntegration.sample_from_adaptive_grid(copied, 64)
        @test first.(edited) == first.(unedited)
        @test last.(edited) == last.(unedited) .+ 10
        warm = vegas(Returns(1.0), copied; options..., grid_precision = :native)
        @test warm.integral_estimate == 7.75
        @test warm.adaptive_grid[[1, 5], :] == [lo 10; hi 11]

        copied.adaptive_grid[:, 1] .= Float64(offset) .+ (64:4:80)
        copied.grid_spacing[:, 1] .= 4.0
        edited = MonteCarloIntegration.sample_from_adaptive_grid(copied, 64)
        @test all(p -> offset + 64 <= first(p) <= offset + 80, edited)
        @test all(p -> 10 < last(p) < 11, edited)
        warm = vegas(Returns(1.0), copied; options..., grid_precision = :native)
        @test warm.integral_estimate == 16
        @test warm.adaptive_grid[[1, 5], :] == [offset + 64 10; offset + 80 11]
        @test vegas(Returns(1.0), original; options...).integral_estimate == 7.75

        width_edit = deepcopy(original)
        width_edit.grid_spacing[:, 1] .= 2.0
        warm = vegas(Returns(1.0), width_edit; options..., grid_precision = :native)
        @test warm.integral_estimate == 8
        @test warm.adaptive_grid[[1, 5], :] == [offset 0; offset + 8 1]
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
            result = vegas(Returns(value), [a], [b]; RESULT_OPTIONS..., callback_type = BigFloat)
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
    @test_throws ArgumentError vegas(Returns(1.0); ncalls = 16, grid_precision = :unsupported)
end
