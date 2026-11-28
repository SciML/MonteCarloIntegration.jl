using MonteCarloIntegration
using Random: seed!
using Test

@testset "Adaptive stratified sampling" begin
    @testset "Public scalar and batch integration" begin
        options = (; stratified = true, nbins = 16, ncalls = 113, maxiter = 5, rtol = 0.0, atol = 0.0)
        scalar_calls = Ref(0)
        batch_calls = Int[]
        matrix_calls = Tuple{Int, Int}[]
        f(x) = x[1] * x[2]
        seed!(2718)
        scalar = vegas(; options...) do x
            scalar_calls[] += 1
            f(x)
        end
        seed!(2718)
        batched = vegas(; options..., batch = true) do points
            push!(batch_calls, length(points))
            map(f, points)
        end

        seed!(2718)
        matrix_batch = vegas(; options..., batch = :matrix) do points
            push!(matrix_calls, size(points))
            map(f, eachcol(points))
        end

        @test scalar_calls[] == 113 * 5
        @test batch_calls == fill(113, 5)
        @test matrix_calls == fill((2, 113), 5)
        for result in (batched, matrix_batch)
            @test result.integral_estimate == scalar.integral_estimate
            @test result.standard_deviation == scalar.standard_deviation
            @test result.adaptive_grid == scalar.adaptive_grid
        end
        @test abs(scalar.integral_estimate - 0.25) < 5 * scalar.standard_deviation
        @test scalar.standard_deviation > 0
        @test 0.0 <= scalar.chi_squared_average < Inf

        seed!(42)
        reused = vegas(f, scalar; options..., alpha = 0.0)
        @test reused.adaptive_grid == scalar.adaptive_grid
        @test reused.adaptive_grid !== scalar.adaptive_grid
        @test abs(reused.integral_estimate - 0.25) < 5 * reused.standard_deviation
    end

    @testset "Narrow integration domain" begin
        lb = 1.0e16
        ub = nextfloat(lb)
        first_counts = zeros(Int, 8)
        batches = Ref(0)
        seed!(1618)
        result = vegas(
            [lb], [ub]; stratified = true, batch = :matrix, nbins = 8,
            ncalls = 32, maxiter = 3, rtol = 0.0, atol = 0.0
        ) do points
            batches[] += 1
            if batches[] == 1
                for point in points
                    first_counts[floor(Int, 4 * (point - lb)) + 1] += 1
                end
            end
            map(point -> point - lb, vec(points))
        end
        @test batches[] == 3
        @test first_counts == fill(4, 8)
        @test eltype(result.adaptive_grid) === BigFloat
        @test all(>(0), result.grid_spacing)
        @test abs(result.integral_estimate - 2.0) < 5 * result.standard_deviation
    end

    @testset "Stratified warm start with rounded Float64 edges" begin
        ub = -2.0^53
        lb = prevfloat(ub)
        nbins, ncalls, maxiter = 3, 64, 2
        edges = reshape(collect(range(lb, ub; length = nbins + 1)), :, 1)
        widths = fill((ub - lb) / nbins, nbins, 1)
        @test any(iszero, diff(edges; dims = 1))
        previous = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, edges, widths)

        for batch in (false, true, :matrix)
            calls, outside = Ref(0), Ref(0)
            function observe(point)
                calls[] += 1
                outside[] += !(lb <= only(point) <= ub)
                return 1.0
            end
            f = if batch === false
                observe
            elseif batch === true
                points -> map(observe, points)
            else
                points -> map(observe, eachcol(points))
            end
            seed!(2718)
            result = vegas(f, previous; stratified = true, batch, ncalls, maxiter, alpha = 0)
            @test calls[] == ncalls * maxiter
            @test outside[] == 0
            @test result.integral_estimate ≈ ub - lb rtol = 2eps(Float64)
            @test iszero(result.standard_deviation)
            @test result.adaptive_grid[1] == lb
            @test result.adaptive_grid[end] == ub
        end
    end

    @testset "Stratified Jacobians for extreme boxes" begin
        offset = 10^16
        side_lengths = [1.0e-200, 1.0e-200, 1.0e200, 1.0e100]
        exact_lb = ((offset + 12) // 1, 0.0, 0.0, 0.0)
        exact_ub = ((offset + 1) // 1, 1.0e-200, 1.0e-200, 1.0e200)
        cases = (
            (zeros(4), side_lengths, 1.0e100, 1.0, Float64),
            (zeros(4), side_lengths[[1, 3, 2, 4]], 1.0e100, 1.0, Float64),
            (exact_lb, exact_ub, 1.0e200, -11.0, BigFloat),
        )
        ncalls, maxiter = 64, 3
        for (lb, ub, value, expected, coordinate_type) in cases, batch in (false, true, :matrix)
            buffer = Ref{AbstractMatrix}()
            reused = Ref(true)
            batch_sizes = Int[]
            function observe!(points, count)
                isassigned(buffer) || (buffer[] = points)
                reused[] &= points === buffer[]
                push!(batch_sizes, count)
                return nothing
            end
            integrand = if batch === false
                point -> begin
                    observe!(parent(point), 1)
                    value
                end
            elseif batch === true
                points -> begin
                    observe!(parent(first(points)), length(points))
                    fill(value, length(points))
                end
            else
                points -> begin
                    count = size(points, 2)
                    observe!(points, count)
                    fill(value, count)
                end
            end
            seed!(2718)
            result = vegas(integrand, lb, ub; stratified = true, batch, ncalls, maxiter, nbins = 2, alpha = 0)
            @test result.integral_estimate ≈ expected rtol = 16eps(Float64)
            @test iszero(result.standard_deviation)
            @test batch_sizes == (batch === false ? ones(Int, ncalls * maxiter) : fill(ncalls, maxiter))
            @test reused[]
            @test eltype(buffer[]) === coordinate_type
        end
    end

    @testset "Zero-error convergence" begin
        calls = Ref(0)
        options = (; stratified = true, ncalls = 100, nbins = 1, maxiter = 5)
        seed!(1)
        step = vegas([0.0], [1.0]; options...) do x
            calls[] += 1
            x[1] < 0.3
        end
        @test calls[] == 500
        @test abs(step.integral_estimate - 0.3) < 5 * step.standard_deviation
        @test step.standard_deviation > 0

        for stratified in (false, true)
            calls[] = 0
            constant = vegas([0.0], [1.0]; options..., stratified) do x
                calls[] += 1
                2.0
            end
            @test calls[] == options.ncalls * options.maxiter
            @test constant.integral_estimate == 2.0
            @test constant.standard_deviation == 0.0
        end

        calls[] = 0
        seed!(29736)
        stepped = vegas([0.0], [1.0]; stratified = true, ncalls = 8, nbins = 1, maxiter = 3) do x
            calls[] += 1
            return x[1] < 0.16 ? 1.0 : x[1] > 0.79 ? 3.0 : 2.0
        end
        @test calls[] == 24
        @test abs(stepped.integral_estimate - 2.05) < 5 * stepped.standard_deviation
        @test stepped.standard_deviation > 0

        calls[] = 0
        seed!(189)
        sparse = vegas([0.0], [1.0]; stratified = true, ncalls = 8, nbins = 1, maxiter = 10) do x
            calls[] += 1
            x[1] > 0.7
        end
        @test calls[] == 80
        @test abs(sparse.integral_estimate - 0.3) < 5 * sparse.standard_deviation
        @test sparse.standard_deviation > 0

        calls[] = 0
        hits, initial_hits = Ref(0), Ref(0)
        seed!(42)
        background = 1.0e-8
        width = 1 / 256
        ncalls, maxiter = 64, 20
        rare = vegas([0.0], [1.0]; stratified = true, ncalls, nbins = 2, maxiter, alpha = 0) do x
            calls[] += 1
            hit = x[1] < width
            hits[] += hit
            initial_hits[] += hit && calls[] <= ncalls
            return background + hit
        end
        @test calls[] == ncalls * maxiter
        @test iszero(initial_hits[])
        @test hits[] > 0
        @test rare.integral_estimate > background
        @test rare.standard_deviation > 0
        @test abs(rare.integral_estimate - (background + width)) < 5 * rare.standard_deviation
    end

    @testset "Stratum allocation and estimation" begin
        uneven = MonteCarloIntegration._VegasStrata(2, 25)
        @test uneven.splits == [3, 2]
        @test sum(uneven.counts) == 25
        @test extrema(uneven.counts) == (4, 5)
        points = zeros(2, 25)
        seed!(314)
        MonteCarloIntegration._sample_strata!(points, uneven)
        first = 1
        for (i, cell) in enumerate(CartesianIndices(Tuple(uneven.splits)))
            last = first + uneven.counts[i] - 1
            for j in first:last, dim in 1:2
                @test (cell[dim] - 1) / uneven.splits[dim] <= points[dim, j] < cell[dim] / uneven.splits[dim]
            end
            first = last + 1
        end

        for (ndim, ncalls) in ((80, 2), (80, 127), (3, 121), (1, 103))
            strata = MonteCarloIntegration._VegasStrata(ndim, ncalls)
            @test prod(strata.splits) == length(strata.counts)
            @test length(strata.counts) <= max(1, ncalls ÷ 4)
            @test minimum(strata.counts) >= 2
            @test sum(strata.counts) == ncalls
        end

        strata = MonteCarloIntegration._VegasStrata(1, 8)
        strata.counts .= [3, 5]
        estimate, error = MonteCarloIntegration._stratified_estimate!(strata, [1.0, 2, 3, 8, 9, 10, 11, 12])
        @test estimate ≈ 6.0
        @test error^2 ≈ 5 / 24
        weights = MonteCarloIntegration._stratified_training_weights!(strata)
        @test weights ≈ [fill(4 / 3, 3); fill(4 / 5, 5)]

        MonteCarloIntegration._stratified_estimate!(strata, [0.0, 10, 20, 0, 0, 0, 0, 0])
        MonteCarloIntegration._redistribute_strata!(strata)
        @test strata.counts == [6, 2]
        @test MonteCarloIntegration._stratified_estimate!(strata, [fill(10.0, 6); 2.0; 2.0]) == (6.0, 0.0)
        MonteCarloIntegration._redistribute_strata!(strata)
        @test strata.counts == [4, 4]
    end

    @testset "Density-weighted grid training" begin
        mass = zeros(2, 1)
        MonteCarloIntegration._bin_means!(
            mass, nothing, reshape([1, 1, 2], 1, :), [1.0, 3.0, 2.0], 3.0, [2.0, 0.5, 0.25]
        )
        @test vec(mass) ≈ [13 / 27, 2 / 27]
    end

    @testset "Cancellation between strata" begin
        for values in ((1.0e16, 1.0, -1.0e16, 0.0), (1.0e16, 1.0, -1.0e16 + 2.0, -2.0))
            seed!(2718)
            result = vegas([0.0], [1.0]; stratified = true, ncalls = 16, nbins = 1, maxiter = 1) do x
                values[floor(Int, 4 * x[1]) + 1]
            end
            @test result.integral_estimate == 0.25
            @test result.standard_deviation == 0.0
        end

        large = floatmax(Float64)
        @test MonteCarloIntegration._stratum_average([large, large]) == large
        @test MonteCarloIntegration._stratum_average([large, large, -large, -large, 1.0e-200]) ≈ 2.0e-201
    end

    @testset "Scaled moments" begin
        strata = MonteCarloIntegration._VegasStrata(1, 8)
        for scale in (1.0e-250, 1.0e250)
            values = [fill(scale, 4); -scale; 0.0; 0.0; scale]
            estimate, error = MonteCarloIntegration._stratified_estimate!(strata, values)
            @test estimate / scale ≈ 0.5
            @test error / scale ≈ sqrt(1 / 24)
            MonteCarloIntegration._redistribute_strata!(strata)
            @test strata.counts == [2, 6]
            strata.counts .= 4

            seed!(42)
            res = vegas(
                x -> scale * x[1], [0.0], [1.0]; stratified = true, ncalls = 2,
                maxiter = 4, nbins = 1, rtol = 0.0, atol = 0.0
            )
            @test 0 < res.integral_estimate / scale < 1
            @test 0 < res.standard_deviation / scale < 1
        end
    end

    @testset "Diagonal ridge at equal evaluation cost" begin
        width = 0.025
        diagonal(x) = max(1 - abs(x[1] - x[2]) / width, 0.0)
        exact = width - width^2 / 3
        options = (; nbins = 32, ncalls = 2053, maxiter = 8, rtol = 0.0, atol = 0.0, alpha = 0.2)
        function trials(stratified)
            return map(11:22) do seed
                seed!(seed)
                vegas(diagonal; options..., stratified)
            end
        end

        classic = trials(false)
        stratified = trials(true)
        squared_error(results) = sum(r -> abs2(r.integral_estimate - exact), results)
        total_variance(results) = sum(r -> abs2(r.standard_deviation), results)
        @test squared_error(stratified) < 0.5 * squared_error(classic)
        @test total_variance(stratified) < 0.5 * total_variance(classic)
    end
end
