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

@testset "Direct BigFloat sampling preserves seeded coordinates across rounding modes" begin
    for bits in (8, 112)
        lo = BigFloat(1; precision = bits)
        hi = nextfloat(lo)
        edges = reshape([lo, hi], :, 1)
        widths = reshape([BigFloat(hi - lo; precision = bits)], :, 1)
        result = VEGASResult(0.0, 0.0, 0.0, edges, widths)
        for ncalls in (0, 1, 64)
            function observe(mode)
                return setrounding(BigFloat, mode) do
                    seed!(54171)
                    points = sample_from_adaptive_grid(result, ncalls)
                    @test points isa Vector{Tuple{BigFloat}}
                    @test all(p -> precision(only(p)) == bits, points)
                    @test all(p -> only(p) == lo || only(p) == hi, points)
                    @test ncalls < 2 || length(unique(points)) == 2
                    @test rounding(BigFloat) === mode
                    return points, rand(UInt64, 4)
                end
            end
            nearest = observe(RoundNearest)
            for mode in (RoundUp, RoundDown)
                @test isequal(observe(mode), nearest)
            end
        end
    end
end

@testset "Direct sampling uses nearest rounding for mixed stored matrices" begin
    lo, hi = 1.0, nextfloat(1.0)
    for (a, b) in ((lo, hi), (hi, lo)), big_edges in (false, true)
        edge_values = big_edges ? BigFloat.([a, b]; precision = 53) : [a, b]
        width_values = big_edges ? [b - a] : [BigFloat(b - a; precision = 53)]
        edges = reshape(edge_values, :, 1)
        widths = reshape(width_values, :, 1)
        result = VEGASResult(0.0, 0.0, 0.0, edges, widths)
        saved = (copy(edges), copy(widths))
        function sample(mode)
            return setrounding(BigFloat, mode) do
                seed!(1)
                points = sample_from_adaptive_grid(result, 256)
                @test points isa Vector{Tuple{BigFloat}}
                @test all(p -> precision(only(p)) == 53, points)
                @test all(p -> only(p) == a || only(p) == b, points)
                @test length(unique(points)) == 2
                @test rounding(BigFloat) === mode
                return points, rand(UInt64, 4)
            end
        end
        nearest = sample(RoundNearest)
        for mode in (RoundUp, RoundDown)
            @test isequal(sample(mode), nearest)
        end
        @test isequal((edges, widths), saved)
    end
end

@testset "Direct sampling accepts rounded mixed spacing" begin
    edges, widths = setprecision(BigFloat, 128) do
        grid = BigFloat[BigFloat(0), BigFloat("0.1"), BigFloat(1)]
        grid, diff(grid)
    end
    reversed_widths = setprecision(BigFloat, 128) do
        -reverse(widths)
    end
    for (edge_values, width_values) in ((edges, widths), (reverse(edges), reversed_widths))
        grid = reshape(edge_values, :, 1)
        nominal_widths = reshape(Float64.(width_values), :, 1)
        result = VEGASResult(0.0, 0.0, 0.0, grid, nominal_widths)
        reference = VEGASResult(0.0, 0.0, 0.0, grid, reshape(width_values, :, 1))
        original = (copy(grid), copy(nominal_widths))
        seed!(521)
        expected = sample_from_adaptive_grid(reference, 128)
        next_draw = rand(UInt64, 4)
        for mode in (RoundNearest, RoundUp, RoundDown)
            setrounding(BigFloat, mode) do
                seed!(521)
                points = sample_from_adaptive_grid(result, 128)
                @test points isa Vector{Tuple{BigFloat}}
                @test all(p -> precision(only(p)) == 128, points)
                @test all(p -> first(edges) <= only(p) <= last(edges), points)
                @test isequal(points, expected)
                @test rand(UInt64, 4) == next_draw
                @test rounding(BigFloat) === mode
            end
        end
        @test isequal((grid, nominal_widths), original)
    end
end

mutable struct ReadCountingSamplingMatrix <: AbstractMatrix{Float64}
    values::Matrix{Float64}
    reads::Int
end

Base.size(matrix::ReadCountingSamplingMatrix) = size(matrix.values)

function Base.getindex(matrix::ReadCountingSamplingMatrix, row::Int, col::Int)
    matrix.reads += 1
    return matrix.values[row, col]
end

@testset "Sparse Float64 sampling reads only the drawn bins" begin
    for (lo, hi) in ((0.0, 1.0), (1.0e9, 1.001e9))
        raw_edges = reshape(collect(range(lo, hi; length = 1_000_001)), :, 1)
        raw_widths = diff(raw_edges; dims = 1)
        edges = ReadCountingSamplingMatrix(raw_edges, 0)
        widths = ReadCountingSamplingMatrix(raw_widths, 0)
        grid = VEGASResult(0.0, 0.0, 0.0, edges, widths)

        seed!(638191)
        actual = sample_from_adaptive_grid(grid, 4)
        next_draw = rand(UInt64)
        @test edges.reads <= 4 * length(actual) + 4
        @test widths.reads <= 2

        seed!(638191)
        expected = map(1:4) do _
            scaled = size(raw_widths, 1) * rand()
            bin = floor(Int, scaled) + 1
            (raw_edges[bin] + raw_widths[bin] * (scaled + 1 - bin),)
        end
        @test actual == expected
        @test rand(UInt64) == next_draw
    end
end

@testset "Direct sampling resolves promoted collapsed legacy grids" begin
    for T in (Float16, Float32), at_zero in (false, true), direction in (identity, reverse)
        a = at_zero ? zero(T) : T === Float16 ? T(1024) : T(2^24)
        lo, hi = direction((a, nextfloat(a, at_zero ? 1 : 4)))
        nbins = 16
        nominal = (hi - lo) / nbins
        original = reshape(Float64[lo + (i - 1) * nominal for i in 1:(nbins + 1)], :, 1)
        original[end] = hi
        original_edges = copy(original)
        widths = fill(Float64(nominal), nbins, 1)
        migrated = reshape(collect(range(Float64(lo), Float64(hi); length = nbins + 1)), :, 1)
        original_result = VEGASResult(0.0, 0.0, 0.0, original, widths)
        migrated_result = VEGASResult(0.0, 0.0, 0.0, migrated, diff(migrated; dims = 1))
        @test any(iszero, diff(original; dims = 1))
        seed!(7487)
        expected = sample_from_adaptive_grid(migrated_result, 16)
        next_draw = rand(UInt64)
        seed!(7487)
        @test sample_from_adaptive_grid(original_result, 16) == expected
        @test rand(UInt64) == next_draw
        @test original_result.adaptive_grid == original_edges
        @test all(==(Float64(nominal)), original_result.grid_spacing)
    end
end

@testset "Direct sampling rejects a collapsed binary64 grid before drawing" begin
    for (lo, hi) in ((0.0, nextfloat(0.0)), (nextfloat(0.0), 0.0))
        edges = reshape([lo, lo, lo, lo, hi], :, 1)
        grid = VEGASResult(0.0, 0.0, 0.0, edges, zeros(4, 1))
        seed!(7751)
        next_draw = rand(UInt64)
        seed!(7751)
        @test_throws ArgumentError sample_from_adaptive_grid(grid, 4)
        @test rand(UInt64) == next_draw
    end
end
