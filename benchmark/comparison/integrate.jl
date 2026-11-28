using MonteCarloIntegration: vegas
using Random: seed!

include("workloads.jl")

const BINS = 100

function integrate(f, dim, mode, seed)
    seed!(seed)
    calls = Ref(0)
    function scalar(x)
        calls[] += 1
        return f(x)
    end
    function matrix(points)
        calls[] += size(points, 2)
        return map(f, eachcol(points))
    end
    lb, ub = zeros(dim), ones(dim)
    options = (; maxiter = ITERATIONS, ncalls = CALLS, nbins = BINS, rtol = 0.0, atol = 0.0)
    result = if mode == "stratified"
        vegas(matrix, lb, ub; options..., batch = :matrix, stratified = true)
    elseif mode == "matrix"
        vegas(matrix, lb, ub; options..., batch = :matrix)
    else
        vegas(scalar, lb, ub; options...)
    end
    calls[] == EVALUATION_BUDGET || error("unexpected VEGAS evaluation count: $(calls[])")
    return result.integral_estimate, result.standard_deviation, calls[]
end

function main(args)
    3 <= length(args) <= 4 || error("usage: integrate.jl OUTPUT.csv LABEL scalar|matrix|stratified [REPETITIONS]")
    output, label, mode = args[1:3]
    mode in ("scalar", "matrix", "stratified") || error("mode must be scalar, matrix or stratified")
    repetitions = length(args) == 4 ? parse(Int, args[4]) : DEFAULT_REPETITIONS
    return run_benchmark(output, label, repetitions) do f, dim, seed
        return integrate(f, dim, mode, seed)
    end
end

main(ARGS)
