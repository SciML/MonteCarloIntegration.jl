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
            [BigFloat(lb)], [BigFloat(ub)]; stratified = true, batch = :matrix, nbins = 8,
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

    @testset "Stratified warm start retains native narrow bounds" begin
        ub = -2.0^53
        lb = prevfloat(ub)
        nbins, ncalls, maxiter = 3, 64, 2
        edges = reshape(collect(range(lb, ub; length = nbins + 1)), :, 1)
        widths = fill((ub - lb) / nbins, nbins, 1)
        @test any(iszero, diff(edges; dims = 1))
        rounded = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, edges, widths)
        @test_throws ArgumentError vegas(Returns(1.0), rounded; stratified = true, ncalls, maxiter)
        previous = vegas(Returns(0.0), [BigFloat(lb)], [BigFloat(ub)]; stratified = true, nbins, ncalls, maxiter)

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
        exact_lb = (BigFloat(offset + 12), 0.0, 0.0, 0.0)
        exact_ub = (BigFloat(offset + 1), 1.0e-200, 1.0e-200, 1.0e200)
        denominator = big(1) << 58
        narrow_lb = (BigFloat((denominator + 1) // denominator), 0.0, 0.0, 0.0)
        narrow_ub = (BigFloat((denominator + 2) // denominator), 1.0e-200, 1.0e-200, 1.0e200)
        cases = (
            (zeros(4), side_lengths, 1.0e100, 1.0, Float64),
            (zeros(4), side_lengths[[1, 3, 2, 4]], 1.0e100, 1.0, Float64),
            (exact_lb, exact_ub, 1.0e200, -11.0, BigFloat),
            ((Float64(offset + 12), 0.0, 0.0, 0.0), (Float64(offset + 2), 1.0e-200, 1.0e-200, 1.0e200), 1.0e200, -10.0, Float64),
            (narrow_lb, narrow_ub, 1.0e200, exp2(-58), BigFloat),
        )
        function typed_callback(integrand, batch, ::Type{T}) where {T}
            scalar(point::AbstractVector{T}) = integrand(point)
            vectors(points::Vector{<:AbstractVector{T}}) = integrand(points)
            matrix(points::Matrix{T}) = integrand(points)
            return batch === false ? scalar : batch === true ? vectors : matrix
        end
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
            callback = typed_callback(integrand, batch, coordinate_type)
            seed!(2718)
            result = vegas(callback, lb, ub; stratified = true, batch, ncalls, maxiter, nbins = 2, alpha = 0)
            @test result.integral_estimate ≈ expected rtol = 16eps(Float64)
            @test iszero(result.standard_deviation)
            @test batch_sizes == (batch === false ? ones(Int, ncalls * maxiter) : fill(ncalls, maxiter))
            @test reused[]
            @test eltype(buffer[]) === coordinate_type
        end
    end

    @testset "Native coordinate precision in all stratified callback layouts" begin
        options = (; stratified = true, maxiter = 1, ncalls = 128, nbins = 1, alpha = 0.0)
        for (T, bounds, ambient) in (
                    (Float32, (Float32[0], Float32[1]), 64),
                    (Float64, (Float64[0], Float64[1]), 64),
                    (BigFloat, ([BigFloat(0; precision = 96)], [BigFloat(1; precision = 128)]), 64),
                ), batch in (false, true, :matrix)
            setprecision(BigFloat, ambient) do
                calls, wide = Ref(0), Ref(0)
                right_type, right_precision = Ref(true), Ref(true)
                function observe(point)
                    right_type[] &= eltype(point) === T
                    x = only(point)
                    if T === BigFloat
                        right_precision[] &= precision(x) == 128
                        wide[] += x != Float64(x)
                    end
                    calls[] += 1
                    return 1.0
                end
                integrand = batch === false ? observe : batch === true ? xs -> map(observe, xs) : xs -> map(observe, eachcol(xs))
                seed!(734)
                fresh = vegas(integrand, bounds...; options..., batch)
                seed!(734)
                warm = vegas(integrand, fresh; options..., batch)
                @test calls[] == 2options.ncalls
                @test right_type[] && right_precision[]
                @test fresh.integral_estimate == warm.integral_estimate == 1
                @test getfield(fresh, :adaptive_grid) isa Matrix{T}
                @test getfield(warm, :grid_spacing) isa Matrix{T}
                @test precision(BigFloat) == ambient
                seed!(912)
                points = MonteCarloIntegration.sample_from_adaptive_grid(warm, options.ncalls)
                @test points isa Vector{Tuple{T}}
                @test all(p -> 0 <= only(p) <= 1, points)
                if T === BigFloat
                    @test wide[] > options.ncalls
                    @test all(p -> precision(only(p)) == 128, points)
                    @test count(p -> only(p) != Float64(only(p)), points) > options.ncalls ÷ 2
                end
            end
        end
    end

    @testset "Low-bit BigFloat stratified bins retain physical widths" begin
        bits = 11
        options = (; stratified = true, ncalls = 4096, nbins = 2500, maxiter = 1, alpha = 0.0)
        lo, hi = BigFloat(-1; precision = bits), BigFloat(1; precision = bits)
        for batch in (false, true, :matrix)
            kept_precision = Ref(true)
            function observe(point)
                kept_precision[] &= precision(only(point)) == bits
                return one(only(point))
            end
            f = batch === false ? observe : batch === true ? xs -> map(observe, xs) : xs -> map(observe, eachcol(xs))
            seed!(8274)
            fresh = vegas(f, [lo], [hi]; options..., batch)
            seed!(8274)
            warm = vegas(f, fresh; options..., batch)
            @test kept_precision[]
            @test fresh.adaptive_grid == warm.adaptive_grid
            @test fresh.grid_spacing == warm.grid_spacing
            @test length(unique(warm.adaptive_grid)) == options.nbins + 1
            @test abs(warm.integral_estimate - 2.0) <= 6warm.standard_deviation + eps(2.0)
            @test warm.integral_estimate == fresh.integral_estimate
            @test warm.standard_deviation == fresh.standard_deviation
        end
    end

    @testset "Native BigFloat callbacks resolve the coarse offset polynomial" begin
        setprecision(BigFloat, 128) do
            offset = BigFloat(2)^53
            a, b = offset + 1 // 8, offset + 63 // 8
            limit = 31 // 8
            expected = Float64(2 * (limit^7 / 7 - 8limit^5 / 5 + 16limit^3 / 3))
            options = (; stratified = true, ncalls = 4096, nbins = 4, maxiter = 1, alpha = 0.0)
            for batch in (false, true, :matrix)
                function polynomial(point::AbstractVector{BigFloat})
                    t = only(point) - offset - 4
                    return abs2(t * (t * t - 4))
                end
                f = batch === false ? polynomial : batch === true ? xs -> map(polynomial, xs) : xs -> map(polynomial, eachcol(xs))
                seed!(440)
                fresh = vegas(f, [a], [b]; options..., batch)
                seed!(440)
                warm = vegas(f, fresh; options..., batch)
                @test first(warm.adaptive_grid) == a
                @test last(warm.adaptive_grid) == b
                @test fresh.integral_estimate == warm.integral_estimate
                @test fresh.standard_deviation == warm.standard_deviation > 0
                @test abs(warm.integral_estimate - expected) < 6warm.standard_deviation
            end
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
        @test MonteCarloIntegration._rounded_mean(estimate) ≈ 6.0
        @test MonteCarloIntegration._rounded_error(error)^2 ≈ 5 / 24
        weights = MonteCarloIntegration._stratified_training_weights!(strata)
        @test weights ≈ [fill(4 / 3, 3); fill(4 / 5, 5)]

        MonteCarloIntegration._stratified_estimate!(strata, [0.0, 10, 20, 0, 0, 0, 0, 0])
        MonteCarloIntegration._redistribute_strata!(strata)
        @test strata.counts == [6, 2]
        estimate, error = MonteCarloIntegration._stratified_estimate!(strata, [fill(10.0, 6); 2.0; 2.0])
        @test (MonteCarloIntegration._rounded_mean(estimate), MonteCarloIntegration._rounded_error(error)) == (6.0, 0.0)
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

        big_term = 2.0^100
        small_term = 2.0^-100
        fifths = (big_term, 1.0, small_term, -big_term, -1.0)
        for values in (fifths, reverse(fifths), map(-, fifths), (big_term, 1.0, 2.0^-55, -big_term, -prevfloat(1.0)))
            exact = Float64(sum(Rational{BigInt}, values) / length(values))
            scalar = x -> values[floor(Int, 5 * x[1]) + 1]
            matrix = points -> map(scalar, eachcol(points))
            for calls in (20, 21), (integrand, batch) in ((scalar, false), (matrix, :matrix))
                seed!(1)
                result = vegas(integrand, [0.0], [1.0]; stratified = true, ncalls = calls, nbins = 1, maxiter = 1, batch)
                @test result.integral_estimate == exact
                @test result.standard_deviation == 0.0
            end
        end

        samples = [big_term, 1.0, small_term, -big_term, -1.0, 0.0, 0.0, 0.0]
        expected = Float64(sum(Rational{BigInt}, samples) / length(samples))
        for values in (samples, circshift(samples, 4))
            call = Ref(0)
            scalar = _ -> values[call[] += 1]
            matrix = points -> map(scalar, eachcol(points))
            for (integrand, batch) in ((scalar, false), (matrix, :matrix))
                call[] = 0
                seed!(1)
                result = vegas(integrand, [0.0], [1.0]; stratified = true, ncalls = 8, nbins = 1, maxiter = 1, batch)
                @test call[] == length(values)
                @test result.integral_estimate == expected
                @test result.standard_deviation ≈ big_term / sqrt(24)
            end
        end

        large = floatmax(Float64)
        average(values) = MonteCarloIntegration._rounded_mean(first(MonteCarloIntegration._stratum_average(values)))
        @test average([large, large]) == large
        @test average([large, large, -large, -large, 1.0e-200]) ≈ 2.0e-201
    end

    @testset "Scaled moments" begin
        strata = MonteCarloIntegration._VegasStrata(1, 8)
        for scale in (1.0e-250, 1.0e250)
            values = [fill(scale, 4); -scale; 0.0; 0.0; scale]
            estimate, error = MonteCarloIntegration._stratified_estimate!(strata, values)
            @test MonteCarloIntegration._rounded_mean(estimate) / scale ≈ 0.5
            @test MonteCarloIntegration._rounded_error(error) / scale ≈ sqrt(1 / 24)
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

    @testset "Unrounded stratum moments" begin
        q = nextfloat(0.0)
        edge = floatmin(Float64)
        tiny_cases = (
            ([0.0, q, q, q], [0.0, q, q, q], nothing, 2, [4, 4], q, 0.0, 50.0),
            ([0.0, 0.0, 2q, 2q], fill(q, 4), nothing, 2, [6, 2], q, 0.0, 98 / 5),
            ([0.0, q, q, q], fill(q, 4), nothing, 2, [6, 2], q, 0.0, 98 / 5),
            ([-q, q, q, q], [0.0, q, q, q], nothing, 1, [4, 4], q, 0.0, 0.0),
            ([1.0, -1.0, 0.0, 2q], [0.0, q, q, q], nothing, 1, [4, 4], q, sqrt(1 / 24), 0.0),
            ([nextfloat(edge, 2), edge, -edge, -edge], [0.0, q, q, q], nothing, 1, [4, 4], q, edge / sqrt(12), 0.0),
            ([0.0, 0.0, 0.0, 3q], [0.0, 0.0, 0.0, 3q], [0.0, 0.0, q, q], 2, [4, 4], q, 0.0, 6 / 31),
            (zeros(4), fill(q, 4), fill(q, 4), 2, [4, 4], q, 0.0, 0.0),
            ([-4q, -4q, -4q, -3q], [-4q, -4q, -4q, -3q], [-2q, q, q, 3q], 2, [4, 4], -4q, 0.0, 36.0),
            ([-4q, -4q, -q, 5q], [-4q, -3q, 2q, 5q], nothing, 1, [4, 4], -0.0, 2q, 0.0),
            ([2q, -2q, q, 4q], [2q, -2q, q, 4q], [-5q, -4q, -2q, -3q], 2, [4, 4], -2q, 0.0, 114 / 5),
            ([-3q, 2q, -2q, -3q], [-3q, 2q, -2q, -3q], [-q, 2q, -2q, 2q], 2, [4, 4], -0.0, q, 42 / 17),
        )
        translated_cases = map(((1.0, 1), (1.0e100, 1), (1.0, 1025))) do (a, steps)
            b = nextfloat(a, steps)
            delta = b - a
            return ([a, b, b, b], [a, a, a, b], [a, a, a, b], 2, [4, 4], muladd(3 / 8, delta, a), delta / 8, 1.0)
        end
        for (left, right, second, maxiter, final_counts, estimate, stderr, chi_squared) in (tiny_cases..., translated_cases...),
                batch in (false, true, :matrix)
            calls = Ref(0)
            counts = zeros(Int, 2, maxiter)
            function evaluate(point)
                iteration = calls[] ÷ 8 + 1
                cell = only(point) < 0.5 ? 1 : 2
                counts[cell, iteration] += 1
                calls[] += 1
                index = counts[cell, iteration]
                iteration == 1 && return (left, right)[cell][index]
                second === nothing && return index == 1 ? 1.0 : 3.0
                return second[index]
            end
            integrand = if batch === false
                evaluate
            elseif batch === true
                points -> map(evaluate, points)
            else
                points -> map(evaluate, eachcol(points))
            end
            seed!(131)
            result = vegas(
                integrand, [0.0], [1.0]; stratified = true, batch, maxiter, ncalls = 8,
                nbins = 1, alpha = 0.0, atol = 0.0, rtol = 0.0,
            )
            @test calls[] == 8maxiter
            @test counts[:, end] == final_counts
            @test result.integral_estimate === estimate
            @test result.standard_deviation ≈ stderr rtol = 8eps(Float64)
            @test result.chi_squared_average ≈ chi_squared rtol = 8eps(Float64)
        end
    end

    @testset "Cross-iteration cancellation with unequal allocation" begin
        magnitude = 2.0^100
        function moments(cells)
            means = map(cell -> sum(Rational{BigInt}, cell) / length(cell), cells)
            variances = map(cells, means) do cell, mean
                sum(value -> abs2(Rational{BigInt}(value) - mean), cell) / (length(cell) * (length(cell) - 1))
            end
            return sum(means) / 2, sum(variances) / 4
        end
        for tail in (1.0, 2.0^-100), direction in (-1, 1), batch in (false, true, :matrix)
            first = ([magnitude, tail, 0.0, 0.0], zeros(4))
            second = ([-1.5magnitude, 0.0, 0.0, 0.0, 0.0, 0.0], fill(tail, 2))
            mean1, variance1 = moments(first)
            mean2, variance2 = moments(second)
            expected_mean = Float64(direction * (mean1 * variance2 + mean2 * variance1) / (variance1 + variance2))
            expected_stderr = sqrt(Float64(variance1 * variance2 / (variance1 + variance2)))
            expected_chi = Float64(abs2(mean1 - mean2) / (variance1 + variance2))
            counts = zeros(Int, 2, 2)
            calls = Ref(0)
            function evaluate(point)
                iteration = calls[] ÷ 8 + 1
                cell = only(point) < 0.5 ? 1 : 2
                counts[cell, iteration] += 1
                calls[] += 1
                return direction * (first, second)[iteration][cell][counts[cell, iteration]]
            end
            integrand = if batch === false
                evaluate
            elseif batch === true
                points -> map(evaluate, points)
            else
                points -> map(evaluate, eachcol(points))
            end
            seed!(1)
            result = vegas(integrand, [0.0], [1.0]; stratified = true, batch, maxiter = 2, ncalls = 8, nbins = 1, atol = 0.0, rtol = 0.0)
            @test calls[] == 16
            @test counts == [4 6; 4 2]
            @test result.integral_estimate === expected_mean
            @test result.standard_deviation ≈ expected_stderr rtol = 8eps(Float64)
            @test result.chi_squared_average ≈ expected_chi rtol = 8eps(Float64)
        end
    end

    @testset "Exact history after deferred samples fill the budget" begin
        calls, positive_batches = 8192, 13
        strata = calls ÷ 4
        magnitude, tail = 2.0^100, 2.0^-100
        batches = Ref(0)
        seed!(1)
        result = vegas(
            [0.0], [1.0]; stratified = true, batch = :matrix, ncalls = calls, nbins = 1,
            maxiter = positive_batches + 1, atol = 0.0, rtol = 0.0,
        ) do points
            batches[] += 1
            last = batches[] > positive_batches
            return map(eachcol(points)) do point
                return if only(point) < 1 / strata
                    last ? tail : 0.0
                else
                    last ? -positive_batches * magnitude : magnitude
                end
            end
        end
        @test batches[] == positive_batches + 1
        @test result.integral_estimate === tail / (strata * (positive_batches + 1))
        @test result.standard_deviation ≈ magnitude * (1 - 1 / strata)
        @test iszero(result.chi_squared_average)
    end

    @testset "Exact mean with one tiny stratum" begin
        q = nextfloat(0.0)
        ncalls, nstrata = 2048, 512
        counts = zeros(Int, nstrata)
        observed = Matrix{Float64}(undef, 4, nstrata)
        batch_values = Vector{Float64}(undef, ncalls)
        function integrand(points)
            for i in axes(points, 2)
                x = points[1, i]
                cell = floor(Int, nstrata * x) + 1
                counts[cell] += 1
                value = cell == 1 ? (isodd(counts[cell]) ? 0.0 : q) : 1.0 + x
                observed[counts[cell], cell] = batch_values[i] = value
            end
            return batch_values
        end
        function integrate()
            seed!(177)
            fill!(counts, 0)
            return vegas(
                integrand, [0.0], [1.0]; stratified = true, batch = :matrix,
                ncalls, nbins = 1, maxiter = 1, alpha = 0.0,
            )
        end
        result = integrate()
        bytes = @allocated integrate()
        mean, error = setprecision(BigFloat, 256) do
            cells = eachcol(observed)
            centers = map(values -> sum(BigFloat, values) / 4, cells)
            variance = sum(zip(cells, centers)) do (values, center)
                sum(value -> (BigFloat(value) - center)^2, values) / 12
            end
            return Float64(sum(centers) / nstrata), Float64(sqrt(variance) / nstrata)
        end
        @test all(==(4), counts)
        @test result.integral_estimate === mean
        @test result.standard_deviation ≈ error rtol = 8eps(Float64)
        @test result.chi_squared_average == 0.0
        @test bytes < 1280 * 1024

        uneven_mean, variance = MonteCarloIntegration._exact_stratum_moments([2, 3], [q, 0, 0, 0, q], false)
        @test uneven_mean == Rational{BigInt}(q) * 5 // 12
        @test isnothing(variance)
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

@testset "Stratified BigFloat mapping preserves the callback rounding mode" begin
    lo = BigFloat(1; precision = 8)
    hi = nextfloat(lo)
    options = (; stratified = true, ncalls = 64, nbins = 1, maxiter = 1, alpha = 0.0)
    previous = vegas(Returns(1), [lo], [hi]; options...)
    original = nothing
    for mode in (RoundNearest, RoundUp, RoundDown)
        setrounding(BigFloat, mode) do
            for args in (([lo], [hi]), (previous,))
                points = BigFloat[]
                modes = RoundingMode[]
                function integrand(x)
                    push!(points, only(x))
                    push!(modes, rounding(BigFloat))
                    return big(10001) << 7
                end
                seed!(54170)
                result = vegas(integrand, args...; options...)
                snapshot = (points, rand(UInt64, 4))
                original = original === nothing ? snapshot : original
                @test isequal(snapshot, original)
                @test length(unique(points)) == 2
                @test all(x -> precision(x) == 8, points)
                @test modes == fill(mode, options.ncalls)
                @test result.integral_estimate == 10001.0
                @test iszero(result.standard_deviation)
                @test rounding(BigFloat) === mode
            end
        end
    end
end

@testset "Stratified VEGAS preserves native geometry and rounds weights once" begin
    options = (; stratified = true, ncalls = 4, maxiter = 1, alpha = 0.0)
    @testset "Weighted midpoint across callback layouts" begin
        setprecision(BigFloat, 128) do
            extent = 1 + ldexp(one(BigFloat), -60)
            value = (big(1) << 53) + 1
            expected = Float64(value + 1)
            for batch in (false, true, :matrix), orientation in (identity, reverse)
                lo, hi = orientation((BigFloat(0), extent))
                f = batch === false ? Returns(value) : Returns(fill(value, options.ncalls))
                result = vegas(f, [lo], [hi]; options..., nbins = 1, batch)
                @test result.integral_estimate === (lo < hi ? expected : -expected)
                @test iszero(result.standard_deviation)
            end
        end
    end
    @testset "Narrow interval with an irrational endpoint" begin
        setprecision(BigFloat, 512) do
            inward = BigFloat(π, RoundUp)
            hi = inward + ldexp(one(BigFloat), -302)
            result = vegas(Returns(1.0), [π], [hi]; options..., nbins = 1)
            @test vec(result.adaptive_grid) == [inward, hi]
            @test result.integral_estimate === Float64(hi - inward)
        end
    end
    @testset "Rounded mixed warm-start widths" begin
        setprecision(BigFloat, 128) do
            edges = reshape(BigFloat[0, BigFloat("0.1"), 1], :, 1)
            mixed = VEGASResult(0.0, 0.0, 0.0, edges, reshape([0.1, 0.9], :, 1))
            physical = VEGASResult(0.0, 0.0, 0.0, edges, diff(edges; dims = 1))
            seed!(6731)
            expected = vegas(only, physical; options...)
            seed!(6731)
            result = vegas(only, mixed; options...)
            @test result.grid_spacing == expected.grid_spacing
            @test (result.integral_estimate, result.standard_deviation) ==
                (expected.integral_estimate, expected.standard_deviation)
        end
    end
end
