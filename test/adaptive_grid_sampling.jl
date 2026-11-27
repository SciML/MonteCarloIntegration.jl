using MonteCarloIntegration: VEGASResult, sample_from_adaptive_grid
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
