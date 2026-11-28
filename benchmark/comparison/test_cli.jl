using Test: @test, @testset

include("workloads.jl")

const EXPECTED = (
    ("smooth_product", 3.375),
    ("laplace_peak", 0.0007432224968301891),
    ("diagonal_ridge", 0.04875),
)

function invoke(script, args...; project = @__DIR__)
    out, err = IOBuffer(), IOBuffer()
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$project $(joinpath(@__DIR__, script)) $args`
    proc = run(pipeline(ignorestatus(cmd); stdout = out, stderr = err))
    return (; ok = success(proc), out = String(take!(out)), err = String(take!(err)))
end

function require_success(result)
    result.ok || error(result.err)
    return result.out
end

function check_trials(path, label)
    lines = readlines(path)
    @test popfirst!(lines) == HEADER
    rows = split.(lines, ',')
    @test length(rows) == length(EXPECTED)
    @test Set(row[2] for row in rows) == Set(first.(EXPECTED))
    for row in rows
        @test row[1] == label && row[3] == "1"
        @test parse(Int, row[4]) == EVALUATION_BUDGET
        @test isfinite(parse(Float64, row[7]))
        @test isfinite(parse(Float64, row[8]))
    end
    return nothing
end

function write_fixture(path, problems; corrupt = false)
    return open(path, "w") do io
        println(io, HEADER)
        for (problem, exact) in problems
            estimate, reference = corrupt ? (0.0, 0.0) : (exact, nextfloat(exact))
            println(io, join(("fixture", problem, 1, EVALUATION_BUDGET, 0.001, 0, estimate, 0.0, reference), ','))
        end
    end
end

function main(args)
    1 <= length(args) <= 2 || error("usage: test_cli.jl LEGACY_PROJECT [COMPATIBLE_PROJECT]")
    legacy = abspath(first(args))
    return mktempdir() do directory
        @testset "Julia comparison CLI" begin
            @testset "CSV validation" begin
                valid = joinpath(directory, "valid.csv")
                bad = joinpath(directory, "bad.csv")
                write_fixture(valid, EXPECTED)
                report = invoke("summarize.jl", "--repetitions", "1", valid)
                @test report.ok
                @test count(line -> occursin("| 0 | 1/1 |", line), split(report.out, '\n')) == length(EXPECTED)

                write_fixture(bad, EXPECTED; corrupt = true)
                result = invoke("summarize.jl", "--repetitions", "1", bad)
                @test !result.ok && occursin("exact value", result.err)
                write_fixture(bad, EXPECTED[1:2])
                result = invoke("summarize.jl", "--repetitions", "1", bad)
                @test !result.ok && occursin("expected seeds", result.err)
                result = invoke("summarize.jl", "--repetitions", "2", valid)
                @test !result.ok && occursin("expected seeds", result.err)
                write_fixture(bad, ())
                result = invoke("summarize.jl", "--repetitions", "1", valid, bad)
                @test !result.ok && occursin("no trials", result.err)
            end

            @testset "Unsupported modes preserve the output" begin
                for (mode, diagnostic) in (("matrix", "TypeError"), ("stratified", "MethodError"))
                    path = joinpath(directory, "$mode.csv")
                    result = invoke("integrate.jl", path, mode, mode, "1"; project = legacy)
                    @test !result.ok && occursin(diagnostic, result.err)
                    @test !isfile(path)
                    write(path, "previous trials\n")
                    result = invoke("integrate.jl", path, mode, mode, "1"; project = legacy)
                    @test !result.ok && occursin(diagnostic, result.err)
                    @test read(path, String) == "previous trials\n"
                end
            end

            @testset "Comparison workflow" begin
                scalar = joinpath(directory, "scalar.csv")
                require_success(invoke("integrate.jl", scalar, "julia_scalar", "scalar", "1"; project = legacy))
                check_trials(scalar, "julia_scalar")
                sources = [scalar]
                for (method, label) in (("iid", "julia_iid"), ("sobol", "julia_sobol"), ("adaptive", "mcintegration_vegas"))
                    path = joinpath(directory, "$method.csv")
                    require_success(invoke("reference.jl", path, method, "1"))
                    check_trials(path, label)
                    push!(sources, path)
                end
                combined = joinpath(directory, "combined.csv")
                require_success(invoke("summarize.jl", "--repetitions", "1", "--combined", combined, sources...))
                @test length(readlines(combined)) == 1 + length(sources) * length(EXPECTED)
            end

            if length(args) == 2
                @testset "Compatible VEGAS modes" begin
                    for mode in ("scalar", "matrix", "stratified")
                        path = joinpath(directory, "updated-$mode.csv")
                        require_success(invoke("integrate.jl", path, mode, mode, "1"; project = abspath(args[2])))
                        check_trials(path, mode)
                    end
                end
            end
        end
    end
end

main(ARGS)
