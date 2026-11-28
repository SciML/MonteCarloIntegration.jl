module MonteCarloIntegration

abstract type MonteCarloIntegrationResult end
include("vegas.jl")
include("vegas_batch_grid.jl")

export vegas, VEGASResult

"""
    VEGASResult

Result returned by [`vegas`](@ref), with the following fields.

- `integral_estimate` is the estimated integral.
- `standard_deviation` is the estimated standard error of the integral.
- `chi_squared_average` is the reduced chi-squared statistic across iterations.
- `adaptive_grid` stores the final bin edges, with one column per dimension.
- `grid_spacing` stores the corresponding bin widths.

Grid matrices use the promoted floating-point type of the integration bounds.
A BigFloat grid retains its integration precision for subsequent calls.
"""
VEGASResult

end # module
