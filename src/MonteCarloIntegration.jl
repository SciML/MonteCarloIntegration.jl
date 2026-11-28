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
- `adaptive_grid` stores the displayed bin edges in the integration coordinates,
  with one column per dimension.
- `grid_spacing` stores the displayed bin widths, with one column per dimension.

The default displays widened grids as rounded `Float64` matrices where
possible. Integration and sampling retain the native grid for unchanged axes.
Set `grid_precision = :native` to expose the native matrices directly.
"""
VEGASResult

end # module
