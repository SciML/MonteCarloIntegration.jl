using Printf: @printf
using Statistics: mean, median

include("workloads.jl")

const REFERENCES = Dict(problem => exact for (problem, _, _, exact) in PROBLEMS)

struct Trial
    evaluations::Int
    seconds::Float64
    bytes::Int
    estimate::Float64
    se::Float64
    csv::String
end

function load_trials(paths, repetitions)
    groups = Dict{Tuple{String, String}, Dict{Int, Trial}}()
    for path in paths
        open(path) do io
            readline(io) == HEADER || error("unexpected CSV header in $path")
            eof(io) && error("no trials in $path")
            for (row, line) in enumerate(eachline(io))
                fields = split(line, ',')
                length(fields) == 9 || error("expected 9 CSV fields in $path at data row $row")
                method, problem = fields[1:2]
                haskey(REFERENCES, problem) || error("unknown problem $problem in $path")
                seed, count = parse.(Int, fields[3:4])
                time = parse(Float64, fields[5])
                bytes = parse(Int, fields[6])
                estimate, se, exact = parse.(Float64, fields[7:9])
                key = (String(problem), String(method))
                trials = get!(() -> Dict{Int, Trial}(), groups, key)
                haskey(trials, seed) && error("duplicate trial $key, seed $seed")
                isfinite(time) && time > 0 && bytes >= 0 && count > 0 && isfinite(estimate) && isfinite(se) && se >= 0 ||
                    error("invalid estimate, uncertainty, time, allocation, or count for $key, seed $seed")
                expected = REFERENCES[problem]
                isapprox(exact, expected; rtol = 8eps(Float64)) ||
                    error("exact value $exact for $key, seed $seed must match $expected")
                trials[seed] = Trial(count, time, bytes, estimate, se, line)
            end
        end
    end
    methods = Set(method for (_, method) in keys(groups))
    expected_seeds = Set(1:repetitions)
    for problem in keys(REFERENCES), method in methods
        key = (problem, method)
        trials = get(groups, key, nothing)
        trials !== nothing && Set(keys(trials)) == expected_seeds ||
            error("expected seeds 1 through $repetitions for $key")
    end
    return groups
end

function report(groups)
    println("| Problem | Method | Trials | Mean calls | RMSE | Coverage | Median ms | Median allocated bytes |")
    println("|---|---|---:|---:|---:|---:|---:|---:|")
    for ((problem, method), trials) in sort!(collect(groups); by = first)
        rows = [trials[seed] for seed in sort!(collect(keys(trials)))]
        exact = REFERENCES[problem]
        rmse = sqrt(mean(row -> abs2(row.estimate - exact), rows))
        coverage = count(row -> abs(row.estimate - exact) <= 1.96 * row.se, rows)
        calls = mean(row -> row.evaluations, rows)
        time_ms = 1000 * median(row.seconds for row in rows)
        bytes = median(row.bytes for row in rows)
        @printf "| %s | %s | %d | %.1f | %.4g | %d/%d | %.3f | %.0f |\n" problem method length(rows) calls rmse coverage length(rows) time_ms bytes
    end
    return nothing
end

function main(args)
    usage = "usage: summarize.jl [--repetitions N] [--combined OUTPUT.csv] INPUT.csv..."
    "--help" in args && return println(usage)
    combined = nothing
    repetitions = DEFAULT_REPETITIONS
    paths = String[]
    arguments = copy(args)
    while !isempty(arguments)
        arg = popfirst!(arguments)
        if arg == "--combined" || arg == "--repetitions"
            isempty(arguments) && error("$arg requires a value")
            value = popfirst!(arguments)
            if arg == "--combined"
                combined = value
            else
                repetitions = parse(Int, value)
            end
        elseif startswith(arg, "--")
            error("unknown option $arg\n$usage")
        else
            push!(paths, arg)
        end
    end
    isempty(paths) && error(usage)
    repetitions > 0 || error("repetitions must be positive")

    groups = load_trials(paths, repetitions)
    if combined !== nothing
        open(combined, "w") do io
            println(io, HEADER)
            for (_, trials) in sort!(collect(groups); by = first), seed in 1:repetitions
                println(io, trials[seed].csv)
            end
        end
    end
    return report(groups)
end

main(ARGS)
