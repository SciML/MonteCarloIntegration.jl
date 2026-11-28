using MonteCarloIntegration
using RecipesBase: apply_recipe
using Test: @test, @testset, @test_throws

@testset "VEGAS grid recipe" begin
    edges = [
        -2.0 10.0 1.0
        -0.5 15.0 2.0
        1.0 16.0 4.0
        3.0 20.0 9.0
    ]
    result = VEGASResult(1.0, 0.1, 1.0, edges, diff(edges; dims = 1))

    for (options, (xdim, ydim)) in ((Dict{Symbol, Any}(), (1, 2)), (Dict{Symbol, Any}(:dims => (3, 1)), (3, 1)))
        series = apply_recipe(options, result)
        @test length(series) == 2
        @test [s.plotattributes[:seriestype] for s in series] == [:hline, :vline]
        by_type = Dict(s.plotattributes[:seriestype] => s for s in series)
        @test only(by_type[:vline].args) == edges[:, xdim]
        @test only(by_type[:hline].args) == edges[:, ydim]
        @test by_type[:vline].plotattributes[:xlabel] == "Dimension $xdim"
        @test by_type[:hline].plotattributes[:ylabel] == "Dimension $ydim"
        @test by_type[:vline].plotattributes[:xlims] == extrema(edges[:, xdim])
        @test by_type[:hline].plotattributes[:ylims] == extrema(edges[:, ydim])
        @test all(s -> !haskey(s.plotattributes, :dims), series)
    end

    main_series, cross_lines = apply_recipe(Dict{Symbol, Any}(:label => "Adaptive grid", :seriescolor => :red), result)
    @test main_series.plotattributes[:label] == "Adaptive grid"
    @test get(main_series.plotattributes, :primary, true)
    @test cross_lines.plotattributes[:label] == ""
    @test !get(cross_lines.plotattributes, :primary, true)
    @test all(s -> s.plotattributes[:seriescolor] === :red, (main_series, cross_lines))
    secondary_only = apply_recipe(Dict{Symbol, Any}(:primary => false), result)
    @test all(s -> s.plotattributes[:primary] === false, secondary_only)

    one_dimensional = vegas(x -> x[1] + 3, [-2.0], [3.0]; nbins = 4, ncalls = 256, maxiter = 2)
    for (options, output, expected_edges) in (
            (Dict{Symbol, Any}(), one_dimensional, one_dimensional.adaptive_grid[:, 1]),
            (Dict{Symbol, Any}(:dims => [3]), result, edges[:, 3]),
        )
        series = only(apply_recipe(options, output))
        @test series.plotattributes[:seriestype] == :vline
        @test !haskey(series.plotattributes, :yshowaxis)
        @test !haskey(series.plotattributes, :xlims)
        @test !haskey(series.plotattributes, :ylims)
        @test !haskey(series.plotattributes, :primary)
        @test only(series.args) == expected_edges
    end

    options = Dict{Symbol, Any}(:xlims => (-4, 5), :ylims => (-10, 110), :yshowaxis => true)
    attributes = only(apply_recipe(options, one_dimensional)).plotattributes
    @test attributes[:xlims] == (-4, 5)
    @test attributes[:ylims] == (-10, 110)
    @test attributes[:yshowaxis] === true

    for dims in ((), (1, 2, 3), (1, 1), (0, 1), (1, 4), (1, 2.0), (true,), 1)
        @test_throws ArgumentError apply_recipe(Dict{Symbol, Any}(:dims => dims), result)
    end
end
