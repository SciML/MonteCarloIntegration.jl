module MonteCarloIntegration

abstract type MonteCarloIntegrationResult end
include("vegas.jl")
include("vegas_batch_grid.jl")

export vegas

end # module
