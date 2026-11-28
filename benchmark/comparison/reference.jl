import MCIntegration
using QuasiMonteCarlo: MatousekScramble, SobolSample, generate_design_matrices
using Random: Xoshiro
using Statistics: mean, std

include("workloads.jl")

function iid_monte_carlo(f, dim, seed)
    points = rand(Xoshiro(seed), dim, EVALUATION_BUDGET)
    values = map(f, eachcol(points))
    return mean(values), std(values) / sqrt(length(values)), length(values)
end

function randomized_sobol(f, dim, seed)
    sampler = SobolSample(R = MatousekScramble(base = 2, pad = 32, rng = Xoshiro(seed)))
    batches = generate_design_matrices(CALLS, dim, sampler, ITERATIONS)
    means = map(points -> mean(f, eachcol(points)), batches)
    evaluations = sum(points -> size(points, 2), batches)
    return mean(means), std(means) / sqrt(length(means)), evaluations
end

function adaptive_vegas(f, ndim, seed)
    variables = ntuple(_ -> MCIntegration.Continuous(0.0, 1.0), ndim)
    point = Vector{Float64}(undef, ndim)
    evaluations = Ref(0)

    function counted_integrand(vars, _)
        coordinates = vars isa Tuple ? vars : (vars,)
        for dim in eachindex(point)
            point[dim] = coordinates[dim][1]
        end
        evaluations[] += 1
        return f(point)
    end

    result = MCIntegration.integrate(
        counted_integrand; solver = :vegas, var = variables, dof = [ones(Int, ndim)],
        seed, niter = ITERATIONS, neval = CALLS, block = 16, ignore = 1,
        parallel = :nothread, print = -2, verbose = -2,
    )
    return only(result.mean), only(result.stdev), evaluations[]
end

function main(args)
    2 <= length(args) <= 3 || error("usage: reference.jl OUTPUT.csv iid|sobol|adaptive [REPETITIONS]")
    output, method = args[1:2]
    repetitions = length(args) == 3 ? parse(Int, args[3]) : DEFAULT_REPETITIONS
    if method == "iid"
        return run_benchmark(iid_monte_carlo, output, "julia_iid", repetitions)
    elseif method == "sobol"
        return run_benchmark(randomized_sobol, output, "julia_sobol", repetitions)
    elseif method == "adaptive"
        return run_benchmark(adaptive_vegas, output, "mcintegration_vegas", repetitions)
    end
    error("method must be iid, sobol, or adaptive")
end

main(ARGS)
