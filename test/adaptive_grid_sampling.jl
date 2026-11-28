using MonteCarloIntegration: VEGASResult, sample_from_adaptive_grid, vegas
using Random: seed!
using Test: @test, @testset, @test_throws

@testset "Adaptive-grid sampling" begin
    edges = [-2.0 9.0 4.0; -1.0 7.0 4.0; 3.0 1.0 4.0]
    widths = diff(edges; dims = 1)
    result = VEGASResult(0.0, 0.0, 0.0, edges, widths)

    for ncalls in (1, 100)
        seed!(2193)
        points = sample_from_adaptive_grid(result, ncalls)
        next_number = rand()

        seed!(2193)
        expected = map(1:ncalls) do _
            ntuple(3) do dim
                scaled = 2 * rand()
                bin = floor(Int, scaled) + 1
                edges[bin, dim] + widths[bin, dim] * (scaled + 1 - bin)
            end
        end
        @test points isa Vector{NTuple{3, Float64}}
        @test points == expected
        @test rand() == next_number
    end

    seed!(2194)
    sampled = sample_from_adaptive_grid(result, 10_000)
    @test all(p -> -2 <= p[1] <= 3 && 1 <= p[2] <= 9 && p[3] == 4, sampled)
    for (left, below) in Iterators.product((false, true), (false, true))
        frequency = count(p -> (p[1] < -1) == left && (p[2] < 7) == below, sampled) / length(sampled)
        @test isapprox(frequency, 0.25; atol = 0.025)
    end
    @test isapprox(sum(first, sampled) / length(sampled), -0.25; atol = 0.06)

    seed!(2195)
    next_number = rand()
    seed!(2195)
    empty_points = sample_from_adaptive_grid(result, 0)
    @test empty_points isa Vector{NTuple{3, Float64}}
    @test isempty(empty_points)
    @test rand() == next_number
    @test_throws ArgumentError sample_from_adaptive_grid(result, -1)
end

@testset "Adaptive-grid sampling uses stored coordinate types" begin
    for T in (Float16, Float32, Float64)
        edges = T[-2 9 4; -1 7 4; 3 1 4]
        widths = diff(edges; dims = 1)
        result = VEGASResult(0.0, 0.0, 0.0, edges, widths)
        for ncalls in (0, 1, 257)
            seed!(5411)
            samples = sample_from_adaptive_grid(result, ncalls)
            subsequent = rand(UInt64)
            seed!(5411)
            expected = map(1:ncalls) do _
                ntuple(3) do dim
                    scaled = 2rand()
                    bin = floor(Int, scaled) + 1
                    T(edges[bin, dim] + widths[bin, dim] * (scaled + 1 - bin))
                end
            end
            @test samples isa Vector{NTuple{3, T}}
            @test samples == expected
            @test rand(UInt64) == subsequent
            @test all(p -> -2 <= p[1] <= 3 && 1 <= p[2] <= 9 && p[3] == 4, samples)
        end
        @test result.adaptive_grid == edges
        @test result.grid_spacing == widths
    end
end

@testset "Adaptive-grid sampling uses stored BigFloat precision" begin
    original = setprecision(BigFloat, 241) do
        near = BigFloat(1) + ldexp(BigFloat(1), -80)
        edges = reshape(BigFloat[1 0; near 2], :, 2)
        VEGASResult(0.0, 0.0, 0.0, edges, diff(edges; dims = 1))
    end
    for active in (24, 512)
        setprecision(BigFloat, active) do
            seed!(2196)
            expected_next = rand(UInt64)
            seed!(2196)
            @test sample_from_adaptive_grid(original, 0) isa Vector{NTuple{2, BigFloat}}
            @test rand(UInt64) == expected_next
            seed!(1840)
            points = sample_from_adaptive_grid(original, 257)
            subsequent = rand(UInt64)
            @test points isa Vector{NTuple{2, BigFloat}}
            @test all(p -> all(x -> precision(x) == 241, p), points)
            @test all(p -> original.adaptive_grid[1, 1] <= p[1] <= original.adaptive_grid[2, 1] && 0 <= p[2] <= 2, points)
            setprecision(BigFloat, 241) do
                # A fraction drawn at Float64 precision cannot supply these low bits.
                @test any(p -> !isinteger(ldexp(p[1], 208)), points)
            end
            edited = deepcopy(original)
            setprecision(BigFloat, 241) do
                edited.adaptive_grid[:, 2] .+= 5
            end
            edge_copy, width_copy = copy(edited.adaptive_grid), copy(edited.grid_spacing)
            seed!(1840)
            moved = sample_from_adaptive_grid(edited, 257)
            @test all(zip(points, moved)) do (before, after)
                after[1] == before[1] && 5 <= after[2] <= 7
            end
            @test rand(UInt64) == subsequent
            @test isequal(edited.adaptive_grid, edge_copy)
            @test isequal(edited.grid_spacing, width_copy)
            @test precision(BigFloat) == active
        end
    end
end

@testset "Adaptive-grid sampling of an overflowing binary64 span" begin
    largest = floatmax(Float64)
    for (a, b) in ((-largest, largest), (largest, -largest))
        result = vegas(Returns(ldexp(1.0, -1022)), [a], [b]; nbins = 1000, ncalls = 32, maxiter = 1, alpha = 0.0)
        @test result.adaptive_grid isa Matrix{Float64}
        @test result.integral_estimate ≈ Float64((BigFloat(b) - BigFloat(a)) * ldexp(1.0, -1022))
        seed!(5414)
        points = sample_from_adaptive_grid(result, 1)
        @test points isa Vector{Tuple{Float64}}
        @test length(points) == 1 && -largest < only(only(points)) < largest
    end
end
