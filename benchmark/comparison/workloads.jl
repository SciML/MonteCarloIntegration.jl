const ITERATIONS = 8
const CALLS = 4096
const EVALUATION_BUDGET = ITERATIONS * CALLS
const DEFAULT_REPETITIONS = 32
const HEADER = "method,problem,seed,evaluations,seconds,allocated_bytes,estimate,reported_se,exact"

smooth_product(x) = (1 + x[1]) * (1 + x[2]) * (1 + x[3])
laplace_peak(x) = exp(-12 * sum(v -> abs(v - 1 / 3), x))
diagonal_ridge(x) = exp(-40 * abs(x[1] - x[2]))

const PROBLEMS = (
    ("smooth_product", 3, smooth_product, (3 / 2)^3),
    ("laplace_peak", 4, laplace_peak, ((2 - exp(-4) - exp(-8)) / 12)^4),
    ("diagonal_ridge", 2, diagonal_ridge, 2 * (39 + exp(-40)) / 40^2),
)

function run_benchmark(integrate, output, label, repetitions)
    occursin(r"^[a-zA-Z0-9_.-]+$", label) || error("label must contain only alphanumeric characters, _, . or -")
    repetitions > 0 || error("repetitions must be positive")
    for (_, dim, f, _) in PROBLEMS
        integrate(f, dim, 0)
    end

    return open(output, "w") do io
        println(io, HEADER)
        for (problem, dim, f, exact) in PROBLEMS
            GC.gc()
            for seed in 1:repetitions
                measured = @timed integrate(f, dim, seed)
                estimate, se, evaluations = measured.value
                println(io, join((label, problem, seed, evaluations, measured.time, measured.bytes, estimate, se, exact), ','))
            end
        end
    end
end
