using MonteCarloIntegration: MonteCarloIntegration, vegas
using Random: Random
using Test: @test, @testset, @test_throws

@testset "VEGAS batch inputs" begin
    lb = [0.0, 1.0]
    ub = [1.0, 3.0]
    ncalls = 128
    maxiter = 3
    options = (; nbins = 10, ncalls, maxiter, rtol = 0.0, atol = 0.0)

    calls = 0
    legacy_calls = 0
    matrix_buffers = Matrix{Float64}[]
    scalar(points) = (calls += 1; sum(points))
    function legacy(points)
        legacy_calls += 1
        @test points isa Vector{<:SubArray}
        @test length(points) == ncalls
        @test all(p -> length(p) == length(lb), points)
        return map(sum, points)
    end
    function matrix(points)
        @test points isa Matrix{Float64}
        @test size(points) == (length(lb), ncalls)
        push!(matrix_buffers, points)
        return vec(sum(points; dims = 1))
    end

    Random.seed!(2121)
    scalar_result = vegas(scalar, lb, ub; options...)
    Random.seed!(2121)
    legacy_result = vegas(legacy, lb, ub; options..., batch = true)
    Random.seed!(2121)
    matrix_result = vegas(matrix, lb, ub; options..., batch = :matrix)
    @test calls == ncalls * maxiter
    @test legacy_calls == length(matrix_buffers) == maxiter
    @test all(p -> p === first(matrix_buffers), matrix_buffers)
    @test scalar_result.integral_estimate ≈ 5.0 rtol = 0.05
    for result in (legacy_result, matrix_result)
        @test result.integral_estimate == scalar_result.integral_estimate
        @test result.standard_deviation == scalar_result.standard_deviation
        @test result.adaptive_grid == scalar_result.adaptive_grid
    end

    @test_throws ArgumentError vegas(error, lb, ub; options..., batch = :unknown)
    @test_throws DimensionMismatch vegas(x -> zeros(size(x, 2) - 1), lb, ub; options..., batch = :matrix)
    @test_throws ArgumentError vegas(x -> zeros(1, size(x, 2)), lb, ub; options..., batch = :matrix)
end

@testset "VEGAS adaptive-grid reuse" begin
    lb = [0.0, 1.0]
    ub = [1.0, 3.0]
    nbins = 10
    ncalls = 128
    peak(x) = exp(-25 * (x[1] - 0.1)^2) * x[2]
    Random.seed!(2116)
    trained = vegas(peak, lb, ub; nbins, ncalls, maxiter = 3, rtol = 0.0, atol = 0.0)
    @test vegas(sum, trained; ncalls, maxiter = 1).integral_estimate ≈ 5.0 rtol = 0.2
    edges = copy(trained.adaptive_grid)
    widths = copy(trained.grid_spacing)
    points = Matrix{Float64}(undef, length(lb), ncalls)
    function integrand(samples)
        points .= samples
        return samples[2, :]
    end

    Random.seed!(2117)
    fresh = vegas(integrand, lb, ub; nbins, ncalls, maxiter = 1, batch = :matrix)
    fresh_points = copy(points)
    Random.seed!(2117)
    reused = vegas(integrand, trained; ncalls, maxiter = 1, batch = :matrix)
    @test points != fresh_points
    @test reused.integral_estimate ≈ fresh.integral_estimate rtol = 0.15
    weighted_values = [
        points[2, j] *
            prod(nbins * widths[searchsortedlast(view(edges, :, dim), points[dim, j]), dim] for dim in eachindex(lb))
            for j in 1:ncalls
    ]
    expected_mean = sum(weighted_values) / ncalls
    expected_variance = sum(v -> abs2(v - expected_mean), weighted_values) / (ncalls * (ncalls - 1))
    @test reused.integral_estimate ≈ expected_mean
    @test reused.standard_deviation ≈ sqrt(expected_variance)
    @test trained.adaptive_grid == edges
    @test trained.grid_spacing == widths
    @test reused.adaptive_grid !== trained.adaptive_grid
    @test reused.grid_spacing !== trained.grid_spacing
    @test_throws ArgumentError vegas(sum, trained; nbins = nbins + 1, ncalls)

    random_points = MonteCarloIntegration.sample_from_adaptive_grid(trained, ncalls)
    @test random_points isa Vector{NTuple{2, Float64}}
    @test length(random_points) == ncalls
    @test all(p -> all(dim -> lb[dim] <= p[dim] <= ub[dim], eachindex(lb)), random_points)

    for (bad_edges, bad_widths) in (
            (edges[1:(end - 1), :], widths),
            (edges, zeros(size(widths))),
            (reverse(edges; dims = 1), widths),
            (fill(NaN, size(edges)), widths),
        )
        malformed = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, bad_edges, bad_widths)
        @test_throws ArgumentError vegas(sum, malformed; ncalls)
    end

    Random.seed!(2118)
    reversed = vegas(x -> 1, [ub[1], lb[2]], [lb[1], ub[2]]; nbins, ncalls, maxiter = 1, alpha = 0.0)
    reversed_reuse = vegas(x -> 2, reversed; ncalls, maxiter = 1)
    @test reversed_reuse.integral_estimate ≈ -4.0

    collapsed = vegas(sum, lb, [lb[1], ub[2]]; nbins, ncalls, maxiter = 1, alpha = 0.0)
    @test iszero(vegas(sum, collapsed; ncalls, maxiter = 1, alpha = 0.0).integral_estimate)
end

@testset "VEGAS zero-valued warm start" begin
    Random.seed!(2122)
    previous = vegas(x -> 1 + x[1], [0.0], [1.0]; nbins = 2, ncalls = 128, maxiter = 2)
    result = vegas(Returns(0.0), previous; ncalls = 128, maxiter = 2)
    @test iszero(result.integral_estimate)
    @test result.adaptive_grid == previous.adaptive_grid
    @test result.grid_spacing == previous.grid_spacing
end

@testset "VEGAS grid with rounded edges" begin
    lb = 1.0e16
    ub = nextfloat(lb, 2)
    nbins = 16
    edges = reshape(collect(range(lb, ub; length = nbins + 1)), :, 1)
    widths = fill((ub - lb) / nbins, nbins, 1)
    @test any(iszero, diff(edges; dims = 1))
    previous = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, edges, widths)
    @test_throws ArgumentError vegas(Returns(1), previous; ncalls = 32, maxiter = 1, alpha = 0.0)
    resolved = vegas(Returns(1), [BigFloat(lb)], [BigFloat(ub)]; nbins, ncalls = 32, maxiter = 1, alpha = 0.0)
    @test resolved.integral_estimate ≈ ub - lb

    inconsistent = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, edges, 2 .* widths)
    @test_throws ArgumentError vegas(Returns(1), inconsistent; ncalls = 32)

    short_edges = reshape([lb, nextfloat(lb), ub], :, 1)
    excessive_widths = fill(10.0, 2, 1)
    invalid = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, short_edges, excessive_widths)
    @test_throws ArgumentError vegas(Returns(1), invalid; ncalls = 32)

    shifted_widths = reshape([0.5, 3.5], :, 1)
    shifted = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, short_edges, shifted_widths)
    @test_throws ArgumentError vegas(Returns(1), shifted; ncalls = 32)

    snapped_edges = reshape([lb, lb, nextfloat(lb)], :, 1)
    snapped_widths = reshape([0.0, nextfloat(lb) - lb], :, 1)
    snapped = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, snapped_edges, snapped_widths)
    Random.seed!(2120)
    accepted = vegas(Returns(1), snapped; ncalls = 512, maxiter = 1, alpha = 0.0)
    @test accepted.integral_estimate ≈ nextfloat(lb) - lb rtol = 0.2
end

@testset "VEGAS adaptation of rounded edges" begin
    lower = 1.0e16
    upper = nextfloat(lower, 6)
    nbins = 9
    peak(x) = x[1] == lower + 4.0 ? 1.0e6 : 1.0
    for (lb, ub) in ((lower, upper), (upper, lower))
        lower_bound, upper_bound = BigFloat(lb), BigFloat(ub)
        previous = vegas(Returns(0), [lower_bound], [upper_bound]; nbins, ncalls = 32, maxiter = 1)
        for domain in (([lower_bound], [upper_bound]), (previous,))
            @testset "Reversed: $(lb > ub), reused: $(length(domain) == 1)" begin
                Random.seed!(1)
                trained = vegas(peak, domain...; nbins, ncalls = 128, maxiter = 1)
                @test issorted(vec(trained.adaptive_grid); rev = lb > ub)
                @test all(w -> sign(ub - lb) * w >= 0, trained.grid_spacing)
                Random.seed!(2)
                reused = vegas(Returns(1), trained; ncalls = 2048, maxiter = 2)
                @test reused.integral_estimate ≈ ub - lb rtol = 0.05
            end
        end
    end
end

@testset "VEGAS sampling stays within rounded bounds" begin
    lower, upper = prevfloat(-2.0^53), -2.0^53
    for (lb, ub) in ((lower, upper), (upper, lower))
        @testset "Reversed: $(lb > ub)" begin
            fresh = vegas(Returns(0.0), [BigFloat(lb)], [BigFloat(ub)]; nbins = 3, ncalls = 16, maxiter = 1)
            historical = deepcopy(fresh)
            function bounded(p)
                x = only(p)
                return sqrt(x - lower) + sqrt(upper - x)
            end
            for (name, previous) in (("fresh", fresh), ("historical", historical))
                @testset "$name grid" begin
                    Random.seed!(1)
                    points = MonteCarloIntegration.sample_from_adaptive_grid(previous, 1000)
                    @test all(p -> lower <= only(p) <= upper, points)
                    Random.seed!(1)
                    reused = vegas(bounded, previous; ncalls = 1000, maxiter = 1, alpha = 0.0)
                    @test isfinite(reused.integral_estimate)
                end
            end
        end
    end
end

@testset "VEGAS warm starts at extreme integrand scales" begin
    Random.seed!(1)
    trained = vegas(Returns(1.0), [0.0], [1.0]; nbins = 4, ncalls = 32, maxiter = 1, alpha = 0.0)
    for scale in (1.0e-315, 1.0e-200, 1.0e-50, 1.0, 1.0e50, 1.0e200, 1.0e305)
        @testset "Scale: $scale" begin
            Random.seed!(1)
            calls = Ref(0)
            result = vegas(trained; ncalls = 128, maxiter = 2, rtol = 1.0e-3, atol = Inf) do _
                calls[] += 1
                return scale
            end
            @test calls[] == 256
            @test result.integral_estimate == scale
            @test result.standard_deviation == 0.0
            @test result.chi_squared_average == 0.0
            @test result.adaptive_grid == trained.adaptive_grid
            @test result.grid_spacing == trained.grid_spacing
        end
    end
end

@testset "VEGAS uncertainty and scale covariance" begin
    peak(x) = exp(-200 * (x[1] - 0.5)^2)
    signed(x) = sin(2π * x[1]) + 0.25
    options = (; nbins = 50, ncalls = 500, maxiter = 10, rtol = 0.0, atol = 0.0)
    for (f, integral) in ((peak, sqrt(π / 200)), (signed, 0.25))
        Random.seed!(42)
        unit = vegas(f, [0.0], [1.0]; options...)
        @test abs(unit.integral_estimate - integral) <= 4unit.standard_deviation
        @test unit.standard_deviation > 0.0
        for scale in (1.0e-200, 1.0e-120, 1.0e-50, 1.0e50, 1.0e200)
            Random.seed!(42)
            result = vegas(x -> scale * f(x), [0.0], [1.0]; options...)
            @test result.integral_estimate / scale ≈ unit.integral_estimate
            @test result.standard_deviation / scale ≈ unit.standard_deviation
            @test result.chi_squared_average ≈ unit.chi_squared_average
        end
    end
end

@testset "VEGAS inverse-variance iteration estimates" begin
    for scale in (1.0e-200, 1.0, 1.0e200)
        batches = (scale .* [-1.0, 0.0, 1.0, 2.0], scale .* [-1.0, 2.0, 5.0, 8.0])
        calls = Ref(0)
        result = vegas(
            _ -> batches[calls[] += 1], [0.0], [1.0];
            nbins = 2, ncalls = 4, maxiter = 2, alpha = 0.0, batch = :matrix,
            rtol = 0.0, atol = 0.0,
        )
        @test result.integral_estimate / scale ≈ 0.8
        @test result.standard_deviation / scale ≈ sqrt(3 / 8)
        @test result.chi_squared_average ≈ 2.16
    end
end

@testset "VEGAS positive subnormal iteration errors" begin
    function scripted_estimate(batches, lower = 0.0, upper = 1.0)
        next_batch = Ref(0)
        return vegas(
            _ -> batches[next_batch[] += 1], [lower], [upper];
            nbins = 1, ncalls = length(first(batches)), maxiter = length(batches),
            alpha = 0.0, batch = :matrix, rtol = 0.0, atol = 0.0,
        )
    end

    q = nextfloat(0.0)
    tiny = [0.0, q, q, q]
    regular = [1.0, 3.0, 3.0, 3.0]
    for batches in ((tiny, regular), (regular, tiny)), (lower, upper) in ((0.0, 1.0), (1.0, 0.0))
        result = scripted_estimate(batches, lower, upper)
        # The two standard errors are q/4 and 1/2, both strictly positive.
        @test result.integral_estimate == (upper - lower) * q
        @test result.standard_deviation == 0.0
        @test result.chi_squared_average ≈ 25.0
    end

    # Both positive errors round to zero. Their squared ratio is 1/4, giving 1.4q.
    a = q .* [0, 1, 1, 1, 1, 1, 1, 2]
    b = q .* [2, 2, 2, 2, 4, 4, 4, 4]
    for batches in ((a, b), (b, a))
        result = scripted_estimate(batches)
        @test result.integral_estimate == q
        @test result.standard_deviation == 0.0
        @test result.chi_squared_average ≈ 22.4
    end

    # The exact variance ratio 4/9 differs from the rounded error ratio squared, 1/4.
    a = q .* [0, 0, 4, 4]
    b = q .* [6, 6, 12, 12]
    for bits in (8, 256), batches in ((a, b), (b, a))
        result = setprecision(BigFloat, bits) do
            scripted_estimate(batches)
        end
        @test result.integral_estimate == 4q
        @test result.standard_deviation == q
        @test result.chi_squared_average ≈ 147 / 13
    end

    largest = floatmax(Float64)
    result = scripted_estimate(([-largest, -largest, largest, largest], tiny))
    @test result.integral_estimate == q
    @test iszero(result.standard_deviation)
    @test iszero(result.chi_squared_average)

    for batches in ((fill(1.0, 4), tiny, regular), (regular, tiny, fill(1.0, 4)))
        result = scripted_estimate(batches)
        @test result.integral_estimate ≈ 7 / 6
        @test result.standard_deviation ≈ sqrt(19 / 36)
        @test result.chi_squared_average ≈ 25.0
    end

    calls = Ref(0)
    result = vegas(
        _ -> (calls[] += 1; tiny), [0.0], [1.0];
        nbins = 1, ncalls = 4, maxiter = 2, batch = :matrix, rtol = 0.4, atol = q,
    )
    @test calls[] == 1
    @test result.integral_estimate == q
    @test iszero(result.standard_deviation)

    Random.seed!(13)
    adapted = vegas(
        x -> only(x) > 0.5 ? q : 0.0, [0.0], [1.0];
        nbins = 2, ncalls = 64, maxiter = 1, rtol = 0.0, atol = 0.0,
    )
    @test adapted.adaptive_grid[2, 1] > 0.5
    @test iszero(adapted.standard_deviation)

    cases = (
        ((q .* [-3, -3, 1, 1], q .* [0, 2, 2, 2]), 75 / 19),
        ((tiny, -regular, q .* [-1, 1, 1, 1]), 12.6),
    )
    for (sample, chi_squared) in cases, batches in (sample, reverse(sample))
        result = scripted_estimate(batches)
        @test result.integral_estimate == q
        @test iszero(result.standard_deviation)
        @test result.chi_squared_average ≈ chi_squared
    end

    ties = (
        ((q .* [2, -2, 1, 4], q .* [-5, -4, -2, -3]), -2q, 57 / 5),
        ((q .* [-3, 2, -2, -3], q .* [-1, 2, -2, 2]), 0.0, 21 / 17),
    )
    for (sample, estimate, chi_squared) in ties, batches in (sample, reverse(sample))
        result = scripted_estimate(batches)
        @test result.integral_estimate == estimate
        @test result.standard_deviation == q
        @test result.chi_squared_average ≈ chi_squared
    end

    for a in (1.0, 1.0e100), steps in (1, 130, 258, 1025)
        b = nextfloat(a, steps)
        midpoint_floor = nextfloat(a, steps ÷ 2)
        rounded_midpoint = isodd(steps) && isodd(reinterpret(UInt64, midpoint_floor)) ?
            nextfloat(midpoint_floor) : midpoint_floor
        samples = ([a, b, b, b], [a, a, a, b])
        for batches in (samples, reverse(samples))
            result = scripted_estimate(batches)
            @test result.integral_estimate == rounded_midpoint
            @test result.standard_deviation ≈ (b - a) / sqrt(32)
            @test result.chi_squared_average ≈ 2.0
        end
    end

    for a in (q, floatmax(Float64))
        result = scripted_estimate((fill(a, 4), fill(a, 4)))
        @test result.integral_estimate == a
        @test iszero(result.standard_deviation)
        @test iszero(result.chi_squared_average)
    end
end

@testset "VEGAS zero-variance batches retain information" begin
    options = (; nbins = 2, ncalls = 2, alpha = 0.0, batch = :matrix)
    for a in (1.0e-200, 1.0, 1.0e200)
        b = nextfloat(a, 4)
        batches = ([a, a], [b, b])
        calls = Ref(0)
        result = vegas(_ -> batches[calls[] += 1], [0.0], [1.0]; options..., maxiter = 2)
        @test calls[] == 2
        @test result.integral_estimate == a + (b - a) / 2
        @test result.standard_deviation == (b - a) / 2
        @test result.chi_squared_average == 0.0
    end

    batches = ([1.0, 1.0], [0.0, 0.0], [0.0, 1.0])
    calls = Ref(0)
    result = vegas(
        _ -> batches[calls[] += 1], [0.0], [1.0];
        options..., maxiter = 3, atol = 1.0e-10, rtol = 1.0e-10,
    )
    @test calls[] == 3
    @test result.integral_estimate == 0.5
    @test result.standard_deviation ≈ sqrt(1 / 12)
    @test result.chi_squared_average == 0.0
end

@testset "VEGAS long histories with zero or subnormal errors" begin
    function integration(batches; tolerance = 0.0)
        step = Ref(0)
        result = vegas(
            _ -> batches[step[] += 1], [0.0], [1.0];
            nbins = 1, ncalls = 4, maxiter = length(batches),
            batch = :matrix, alpha = 0.0, rtol = tolerance, atol = tolerance,
        )
        return result, step[]
    end

    function reference(batches)
        return setprecision(BigFloat, 4096) do
            means = map(batches) do batch
                sum(BigFloat, batch) / length(batch)
            end
            variances = map(batches, means) do batch, mean
                sum(x -> abs2(BigFloat(x) - mean), batch) / (length(batch) * (length(batch) - 1))
            end
            weights = map(v -> iszero(v) ? zero(v) : inv(v), variances)
            n = length(batches)
            center = sum(weights .* means) / sum(weights)
            chi_squared = sum(weights .* abs2.(means .- center)) / (count(>(0), weights) - 1)
            if any(iszero, variances)
                mean = sum(means) / n
                scatter = sum(abs2, means .- mean) / (n * (n - 1))
                error = sqrt(max(scatter, sum(variances) / n^2))
                return Float64.((mean, error, chi_squared))
            end
            return Float64.((center, inv(sqrt(sum(weights))), chi_squared))
        end
    end

    q = nextfloat(0.0)
    for (first_batch, bytes_per_iteration) in ((zeros(4), 4096), ([-q, q, q, q], 65536))
        rng = Random.MersenneTwister(2026)
        batches = [first_batch, [1 .+ rand(rng, 4) for _ in 2:240]...]
        integration(batches[1:32]; tolerance = q)
        for (n, tolerance) in ((120, 0.0), (240, 0.0), (120, q))
            history = batches[1:n]
            expected_mean, expected_error, expected_chi_squared = reference(history)
            measured = @timed integration(history; tolerance)
            result, evaluated = measured.value
            @test evaluated == n
            if iszero(first_batch[2])
                @test result.integral_estimate ≈ expected_mean rtol = 16eps()
            else
                @test result.integral_estimate == expected_mean == q
            end
            @test result.standard_deviation ≈ expected_error rtol = 16eps()
            @test result.chi_squared_average ≈ expected_chi_squared rtol = 32eps()
            @test measured.bytes < bytes_per_iteration * n
        end
    end

    history = [
        [-1.0, 1.0, 0.0, q],
        [(isodd(i) ? 0.5 : 1.0) .+ [0.0, 1.0, 3.0, 5.0] ./ (i + 1) for i in 2:240]...,
    ]
    expected_mean, expected_error, expected_chi_squared = reference(history)
    integration(history[1:32]; tolerance = q)
    measured = @timed integration(history; tolerance = q)
    result, evaluated = measured.value
    @test evaluated == length(history)
    @test result.integral_estimate ≈ expected_mean rtol = 16eps()
    @test result.standard_deviation ≈ expected_error rtol = 16eps()
    @test result.chi_squared_average ≈ expected_chi_squared rtol = 32eps()
    @test measured.bytes < 65536 * length(history)

    rng = Random.MersenneTwister(7)
    ordinary = map(1:32) do _
        a = 1 + rand(rng)
        [-a, a, 9q, 9q]
    end
    for perturbed in (false, true)
        history = [[0.0, 6q, 6q, 6q], copy.(ordinary)...]
        perturbed && (history[end][end] = 10q)
        for sign in (-1, 1), batches in (history, reverse(history))
            result, evaluated = integration(map(batch -> sign .* batch, batches))
            @test evaluated == length(batches)
            @test result.integral_estimate == sign * (perturbed ? 5q : 4q)
            @test result.standard_deviation == q
            @test iszero(result.chi_squared_average)
        end
    end

    ties = repeat([q .* [2, -2, 1, 4], q .* [-5, -4, -2, -3]], 16)
    for sign in (-1, 1), batches in (ties, reverse(ties))
        result, _ = integration(map(batch -> sign .* batch, batches))
        @test result.integral_estimate == -sign * 2q
        @test iszero(result.standard_deviation)
        @test result.chi_squared_average ≈ (57 / 5) * 16 / 31
    end
end

@testset "VEGAS cancellation and stopping boundaries" begin
    function integrate_batches(batches; rtol = 0.0, atol = 0.0)
        calls = Ref(0)
        result = vegas(
            _ -> batches[calls[] += 1], [0.0], [1.0];
            nbins = 1, ncalls = length(first(batches)), maxiter = length(batches),
            batch = :matrix, alpha = 0.0, rtol, atol,
        )
        return result, calls[]
    end

    function reference_history(batches)
        moments = map(batches) do batch
            values = Rational{BigInt}.(batch)
            count = length(values)
            mean = sum(values) / count
            return mean, sum(v -> abs2(v - mean), values) / (count * (count - 1))
        end
        return map(eachindex(moments)) do n
            means = first.(moments[1:n])
            variances = last.(moments[1:n])
            if any(iszero, variances)
                mean = sum(means) / n
                scatter = n == 1 ? zero(mean) : sum(m -> abs2(m - mean), means) / n / (n - 1)
                return mean, max(scatter, sum(variances) / n / n)
            end
            weights = inv.(variances)
            return sum(weights .* means) / sum(weights), inv(sum(weights))
        end
    end

    terms = [1.0, 2.0^-54, 2.0^-110, -1.0, -1.0, -2.0^-54, 1.0, 2.0^-111]
    rng = Random.MersenneTwister(119)
    wide = filter(isfinite, reinterpret(Float64, rand(rng, UInt64, 64)))
    for samples in (wide, [floatmax(), -floatmax(), nextfloat(0.0)], [0.0, -1.25, 2.5, 3.75], zeros(4))
        exact = Rational{BigInt}.(samples)
        n = length(exact)
        mean = sum(exact) / n
        variance = sum(v -> abs2(v - mean), exact) / (n * (n - 1))
        for ordered in (samples, reverse(samples))
            @test MonteCarloIntegration._exact_binary64_moments(ordered, n) == (mean, variance)
        end
    end
    for values in (terms, reverse(terms)), scale in (-1.0, 1.0, 2.0^500)
        result, calls = integrate_batches((scale .* values,))
        @test calls == 1
        @test result.integral_estimate == scale * (3 * 2.0^-114)
    end

    a = 2.0^48
    pairs = ([a - 4, a + 4], [-1.0, 3.0], [-9a - 12, -9a + 12])
    triples = ([a - 2, a - 1, a + 3], [-1.0, 1.0, 3.0], [-9a - 6, -9a - 3, -9a + 9])
    for rows in (pairs, triples), order in (rows, reverse(rows)), scale in (1.0, -1.0, 2.0^-600, 2.0^600)
        batches = map(v -> scale .* v, order)
        expected, variance = last(reference_history(batches))
        result, calls = integrate_batches(batches)
        @test calls == length(batches)
        @test result.integral_estimate == Float64(expected)
        @test result.standard_deviation ≈ Float64(sqrt(BigFloat(variance))) rtol = 2eps()
    end

    cancellation = (
        [prevfloat(1.0), prevfloat(1.0), 1.0, 1.0],
        fill(-1.0, 4), fill(2.0^-54, 4), fill(2.0^-110, 4),
    )
    for batches in (cancellation, reverse(cancellation)), scale in (1.0, -1.0, 2.0^200)
        result, calls = integrate_batches(map(batch -> scale .* batch, batches))
        @test result.integral_estimate == scale * 2.0^-112
        @test calls == 4
    end

    q = nextfloat(0.0)
    for (seed, side) in ((2028, -1), (2027, 1))
        rng = Random.MersenneTwister(seed)
        draws = [q .* rand(rng, (10^11):(10^12), 4) for _ in 1:5]
        for batches in (draws, reverse(draws)), relative in (false, true)
            history = reference_history(batches)
            tolerance, expected = setprecision(BigFloat, 1024) do
                mean, variance = history[4]
                target = sqrt(BigFloat(variance))
                relative && (target /= abs(BigFloat(mean)))
                tolerance = target * (1 + side * ldexp(BigFloat(1), -200))
                exact = Rational{BigInt}(tolerance)
                index = findfirst(history) do (mean, variance)
                    variance < abs2(exact * (relative ? mean : 1))
                end
                return tolerance, something(index, length(batches))
            end
            for bits in (24, 1024)
                _, calls = setprecision(BigFloat, bits) do
                    relative ? integrate_batches(batches; rtol = tolerance, atol = Inf) :
                        integrate_batches(batches; rtol = Inf, atol = tolerance)
                end
                @test calls == expected
            end
        end
    end

    rng = Random.MersenneTwister(706)
    delta = 2.0^46 * q
    equality = [zeros(4)]
    for i in 2:8
        spread = rand(rng, (10^11):(10^12)) * q
        center = i == 8 ? 8delta : 0.0
        push!(equality, center .+ [-spread, -spread, spread, spread])
    end
    _, variance = reference_history(equality)[8]
    @test variance == Rational{BigInt}(delta)^2
    for relative in (false, true)
        batches = [equality; [fill(relative ? 8delta : 0.0, 4)]]
        exact = relative ? one(Rational{BigInt}) : Rational{BigInt}(delta)
        tolerances = (exact, BigFloat(exact; precision = 1024))
        for tolerance in tolerances
            _, calls = setprecision(BigFloat, 24) do
                relative ? integrate_batches(batches; rtol = tolerance, atol = Inf) :
                    integrate_batches(batches; rtol = Inf, atol = tolerance)
            end
            @test calls == 9
        end
    end

    draws = [q .* rand(rng, (10^11):(10^12), 4) for _ in 1:5]
    for (power, expected) in ((-100_000, 5), (100_000, 1)), relative in (false, true)
        tolerance = ldexp(BigFloat(1), power)
        _, calls = relative ? integrate_batches(draws; rtol = tolerance, atol = Inf) :
            integrate_batches(draws; rtol = Inf, atol = tolerance)
        @test calls == expected
    end

    # The inverse variances sum to 16, but their normalized ratio is not dyadic.
    unequal_variances = [big(4) // 9, big(4) // 55]
    for sign in (-1, 1), variances in (unequal_variances, reverse(unequal_variances))
        means = fill(big(sign) // 1, 2)
        mean, error, _ = MonteCarloIntegration._combine_bounded_estimates(means, variances, nothing)
        @test mean == sign
        @test MonteCarloIntegration._rounded_error(error) == 0.25
        for tolerance in (big(1) // 4, BigFloat(0.25; precision = 1024))
            @test !MonteCarloIntegration._meets_error_tolerances(error, mean, tolerance, Inf)
            @test !MonteCarloIntegration._meets_error_tolerances(error, mean, Inf, tolerance)
        end
    end
end

@testset "VEGAS warm starts with low-precision bounds" begin
    options = (; nbins = 3, ncalls = 32, maxiter = 1)
    previous = vegas(Returns(0), Float32[0], Float32[1]; options...)
    @test sum(previous.grid_spacing) ≈ 1.0
    result = vegas(Returns(1), previous; ncalls = options.ncalls, maxiter = options.maxiter, alpha = 0.0)
    @test result.integral_estimate ≈ 1.0 rtol = eps(Float32)

    width = 1.0f0 / 3
    edges = reshape(Float64[i * width for i in 0:3], :, 1)
    widths = fill(Float64(width), 3, 1)
    historical = MonteCarloIntegration.VEGASResult(0.0, 0.0, 0.0, edges, widths)
    reused = vegas(Returns(1), historical; ncalls = options.ncalls, maxiter = options.maxiter, alpha = 0.0)
    @test reused.integral_estimate ≈ 1.0
    @test edges[2] == Float64(width)
    @test reused.adaptive_grid == edges
    @test reused.grid_spacing == diff(edges; dims = 1)
end
