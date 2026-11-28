# Monte Carlo integration comparison

Each method integrates three functions over the unit box with a nominal budget
of 32,768 evaluations per trial. Each run uses seeds 1 through 32 by default.

| Function | Dimension | Exact integral |
|---|---:|---|
| `prod(1 + xᵢ)` | 3 | `(3/2)^3` |
| `exp(-12 * sum(abs(xᵢ - 1/3)))` | 4 | `((2 - exp(-4) - exp(-8))/12)^4` |
| `exp(-40 * abs(x₁ - x₂))` | 2 | `2 * (39 + exp(-40))/40^2` |

The first two integrals factor into one-dimensional integrals. For the third,
the distance between two uniform coordinates has density `2 * (1 - d)` on
`0 ≤ d ≤ 1`.

## Run

Run the following commands from `benchmark/comparison/`. They use the Julia
package in this checkout. Run the commands one at a time on the same machine.

```sh
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
julia --startup-file=no --project=../.. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=../.. integrate.jl julia.csv julia_scalar scalar
```

The reference methods use an isolated Julia environment in this directory.
It pins QuasiMonteCarlo.jl and MCIntegration.jl, both MIT-licensed.
[QuasiMonteCarlo.jl license](https://github.com/SciML/QuasiMonteCarlo.jl/blob/v0.4.4/LICENSE) ·
[MCIntegration.jl license](https://github.com/numericalEFT/MCIntegration.jl/blob/v0.4.2/license.md)

```sh
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=. reference.jl iid.csv iid
julia --startup-file=no --project=. reference.jl sobol.csv sobol
julia --startup-file=no --project=. reference.jl adaptive.csv adaptive
julia --startup-file=no --project=. summarize.jl --combined trials.csv julia.csv iid.csv sobol.csv adaptive.csv > summary.md
```

Append a positive repetition count to the runners for a shorter run and set
`--repetitions` on the summary command to match. The summary checks each supplied
reference against the analytic integral and expects all three problems and every
seed from 1 through the repetition count for each method. It reports
root-mean-square error, the number of exact values covered by
`estimate ± 1.96 * reported_se`, mean function calls, and median trial time.

## Compare upstream with matrix batches and stratification

For the seven-method comparison, instantiate both pinned Julia checkouts
before timing and reuse the three reference CSVs from the main recipe. The
benchmark-only checkout supports `scalar`. A checkout with both feature modes
can run the updated commands together.

- Upstream Julia source. [`faab55ec4d43166a05d2fc0028e837e5048ce50e`](https://github.com/SciML/MonteCarloIntegration.jl/commit/faab55ec4d43166a05d2fc0028e837e5048ce50e)
- Updated Julia source with `batch = :matrix` and `stratified = true`. [`f051e52e33bddf54350d762ec9df5704adb1afb2`](https://github.com/SciML/MonteCarloIntegration.jl/commit/f051e52e33bddf54350d762ec9df5704adb1afb2)

```sh
julia --startup-file=no --project=/path/to/upstream -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=/path/to/updated -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=/path/to/upstream integrate.jl upstream.csv julia_upstream_scalar scalar
julia --startup-file=no --project=/path/to/updated integrate.jl updated.csv julia_updated_scalar scalar
julia --startup-file=no --project=/path/to/updated integrate.jl matrix.csv julia_updated_matrix matrix
julia --startup-file=no --project=/path/to/updated integrate.jl stratified.csv julia_stratified_matrix stratified
julia --startup-file=no --project=. summarize.jl --combined trials.csv upstream.csv updated.csv matrix.csv stratified.csv iid.csv sobol.csv adaptive.csv > summary.md
```

The CLI smoke checks run one seed per problem. They use the upstream checkout
to verify that requesting an unsupported mode leaves the output untouched.
Supply the updated checkout to exercise all three Julia modes there as well.

```sh
julia --startup-file=no --project=. test_cli.jl /path/to/upstream /path/to/updated
```

## Measurement details

- MonteCarloIntegration uses eight iterations of 4,096 calls and 100 adaptive bins per axis.
  Scalar before/after runs use identical callbacks and options. Matrix runs
  map the integrand across the native sample columns.
- QuasiMonteCarlo uses eight independent Matoušek scrambles of a 4,096-point
  Sobol sequence. Its reported standard error comes from the replicate means.
- IID Monte Carlo draws one batch of 32,768 points with `Random.Xoshiro` and
  estimates uncertainty from the sample variance.
- MCIntegration runs its `:vegas` solver with eight iterations of 4,096 points
  and 16 blocks. The benchmark counts calls at the integrand, including the
  first iteration that trains the grid. MCIntegration computes the estimate and
  standard error from the seven retained iterations.
- Each trial includes setup, sampling, and estimation. Imports and one
  warm-up run per method and problem are outside the measurements. Julia
  allocation counts come from `@timed`. Seed labels identify reproducible
  trials within each method.
